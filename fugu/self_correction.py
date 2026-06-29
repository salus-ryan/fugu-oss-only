"""
Self-Correction via Reinforcement Learning (SCoRe) for the coordinator.

From the catalog:
- SCoRe uses a TWO-STAGE training approach to prevent behavior collapse:
  Stage I: Train model to correct first-turn errors while keeping first-turn
           distribution close to base model via KL constraint.
  Stage II: Jointly train both attempts with progress-based reward bonus
            for transitions that correct first-turn errors.
  
  r_bonus = r(attempt_2) - r(attempt_1)  (rewards correction over initial try)

- STaR (Self-Taught Reasoner): Iteratively generates rationales, filters incorrect
  answers, distills correct reasoning paths back into parameters.

Applied to Fugu's coordinator:
- When a Verifier says REVISE, the coordinator gets a SECOND CHANCE to re-route.
- The correction signal (REVISE diagnosis) is used to train the coordinator to
  make better routing decisions on the NEXT attempt.
- Progress reward: bonus for improving from attempt 1 → attempt 2.

This creates a self-improving loop:
1. Coordinator routes → Worker produces → Verifier REVISEs with diagnosis
2. Coordinator re-routes using diagnosis context
3. If attempt 2 succeeds, the TRANSITION is reinforced
4. Coordinator learns to self-correct its routing mistakes
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

from .coordinator import TrinityCoordinator, Role
from .worker_pool import WorkerPool
from .orchestrator import FuguOrchestrator, OrchestrationResult


@dataclass
class CorrectionEpisode:
    """Record of a self-correction episode (attempt 1 → attempt 2)."""
    query: str
    # Attempt 1
    attempt1_result: OrchestrationResult
    attempt1_reward: float
    attempt1_decisions: List[dict]  # [{model_index, role_value, transcript}]
    # Correction context
    diagnosis: str                   # Verifier's REVISE diagnosis
    # Attempt 2
    attempt2_result: OrchestrationResult
    attempt2_reward: float
    attempt2_decisions: List[dict]
    # Progress
    progress_reward: float           # r(attempt2) - r(attempt1)
    corrected: bool                  # Did attempt 2 succeed where attempt 1 failed?


@dataclass
class SCoReConfig:
    """Configuration for Self-Correction training."""
    # Stage I: KL-constrained correction learning
    stage1_epochs: int = 5
    stage1_lr: float = 1e-4
    stage1_kl_coeff: float = 0.1    # Strong KL to prevent first-turn collapse
    
    # Stage II: Joint training with progress bonus
    stage2_epochs: int = 10
    stage2_lr: float = 5e-5
    stage2_progress_weight: float = 2.0  # Amplify progress reward
    stage2_kl_coeff: float = 0.01   # Lighter KL in stage 2
    
    # General
    max_correction_attempts: int = 3
    min_progress_for_bonus: float = 0.1  # Minimum improvement to count as correction
    max_grad_norm: float = 1.0
    output_dir: str = "./checkpoints/score"


class SelfCorrectionTrainer:
    """
    SCoRe training for the coordinator.
    
    The key insight: When a Verifier says REVISE, the coordinator has a chance
    to LEARN from that feedback. The diagnosis tells it WHAT went wrong, and
    the re-routing attempt tells it HOW to fix its decision.
    
    Two-stage training prevents the coordinator from collapsing to always
    making bad first attempts (just to get correction bonuses).
    """

    def __init__(
        self,
        coordinator: TrinityCoordinator,
        worker_pool: WorkerPool,
        config: Optional[SCoReConfig] = None,
    ):
        self.coordinator = coordinator
        self.pool = worker_pool
        self.config = config or SCoReConfig()
        
        self.optimizer = torch.optim.Adam(
            self.coordinator.head.parameters(),
            lr=self.config.stage1_lr,
        )
        
        # Reference policy (base model before any correction training)
        self._reference_params = self.coordinator.head.get_flat_params().copy()
        
        # Collected correction episodes
        self.episodes: List[CorrectionEpisode] = []
        self.history: List[dict] = []

    def collect_correction_episode(
        self,
        query: str,
        ground_truth: Optional[str] = None,
    ) -> Optional[CorrectionEpisode]:
        """
        Run orchestration, and if Verifier REVISEs, run a correction attempt.
        
        Returns a CorrectionEpisode if self-correction was triggered, else None.
        """
        orchestrator = FuguOrchestrator(
            coordinator=self.coordinator,
            worker_pool=self.pool,
            verbose=False,
        )
        
        # Attempt 1
        result1 = orchestrator.run(query)
        reward1 = self._compute_reward(result1, ground_truth)
        
        # Check if we got a REVISE diagnosis
        diagnosis = self._extract_diagnosis(result1)
        if diagnosis is None:
            # No REVISE → no self-correction opportunity
            return None
        
        # Attempt 2: Re-run with diagnosis context injected
        augmented_query = (
            f"{query}\n\n[Previous attempt feedback: {diagnosis}]\n"
            f"Please try a different approach based on this feedback."
        )
        
        result2 = orchestrator.run(augmented_query)
        reward2 = self._compute_reward(result2, ground_truth)
        
        # Compute progress
        progress = reward2 - reward1
        corrected = (reward2 > reward1 + self.config.min_progress_for_bonus)
        
        # Extract decision traces
        decisions1 = self._extract_decisions(query, result1)
        decisions2 = self._extract_decisions(augmented_query, result2)
        
        episode = CorrectionEpisode(
            query=query,
            attempt1_result=result1,
            attempt1_reward=reward1,
            attempt1_decisions=decisions1,
            diagnosis=diagnosis,
            attempt2_result=result2,
            attempt2_reward=reward2,
            attempt2_decisions=decisions2,
            progress_reward=progress,
            corrected=corrected,
        )
        
        self.episodes.append(episode)
        return episode

    def _extract_diagnosis(self, result: OrchestrationResult) -> Optional[str]:
        """Extract the last Verifier REVISE diagnosis from a result."""
        for turn in reversed(result.turns):
            if turn.role == Role.VERIFIER and turn.response.diagnosis:
                return turn.response.diagnosis
        return None

    def _extract_decisions(self, query: str, result: OrchestrationResult) -> List[dict]:
        """Extract decision trace with transcript states."""
        decisions = []
        transcript = f"User query: {query}\n"
        
        for turn in result.turns:
            decisions.append({
                "model_index": turn.decision.model_index,
                "role_value": turn.decision.role.value,
                "transcript_state": transcript,
            })
            role_label = turn.role.name.capitalize()
            transcript += f"\n[{role_label} — {turn.worker_name}]:\n{turn.response.content}\n"
        
        return decisions

    def _compute_reward(
        self,
        result: OrchestrationResult,
        ground_truth: Optional[str] = None,
    ) -> float:
        """Compute outcome reward for a result."""
        if not result.answer:
            return -1.0
        reward = 0.0
        if result.accepted:
            reward += 1.0
        if ground_truth and ground_truth.strip().lower() in result.answer.lower():
            reward += 0.5
        reward -= result.num_turns * 0.05
        return reward

    def _compute_trajectory_log_prob(self, decisions: List[dict]) -> torch.Tensor:
        """Compute total log prob of a decision sequence under current policy."""
        total_lp = torch.tensor(0.0)
        for d in decisions:
            h = self.coordinator.get_hidden_state(d["transcript_state"])
            model_logits, role_logits = self.coordinator.head(h)
            model_lp = F.log_softmax(model_logits, dim=-1)[0, d["model_index"]]
            role_lp = F.log_softmax(role_logits, dim=-1)[0, d["role_value"]]
            total_lp = total_lp + model_lp + role_lp
        return total_lp

    def _compute_reference_log_prob(self, decisions: List[dict]) -> torch.Tensor:
        """Log prob under reference (frozen) policy."""
        original = self.coordinator.head.get_flat_params()
        self.coordinator.head.set_flat_params(self._reference_params)
        with torch.no_grad():
            lp = self._compute_trajectory_log_prob(decisions)
        self.coordinator.head.set_flat_params(original)
        return lp

    def train_stage1(self, episodes: Optional[List[CorrectionEpisode]] = None):
        """
        Stage I: Train coordinator to correct errors while keeping
        first-turn distribution close to base model.
        
        Objective: Maximize log π(attempt2 | diagnosis) subject to
                   KL(π_attempt1 || π_ref) < ε
        
        This prevents the model from learning to deliberately make bad
        first attempts just to trigger corrections.
        """
        eps = episodes or self.episodes
        if not eps:
            print("  No correction episodes available for Stage I")
            return
        
        print("  === Stage I: KL-constrained correction learning ===")
        print(f"    Episodes: {len(eps)}")
        
        self.optimizer.param_groups[0]["lr"] = self.config.stage1_lr
        
        for epoch in range(self.config.stage1_epochs):
            epoch_loss = 0.0
            
            for episode in eps:
                self.optimizer.zero_grad()
                
                # Reward for correct routing on attempt 2
                attempt2_lp = self._compute_trajectory_log_prob(episode.attempt2_decisions)
                correction_reward = max(episode.progress_reward, 0.0)
                
                # Policy gradient: reinforce successful corrections
                policy_loss = -correction_reward * attempt2_lp
                
                # KL constraint on attempt 1 (don't collapse first attempts)
                if episode.attempt1_decisions:
                    attempt1_lp = self._compute_trajectory_log_prob(episode.attempt1_decisions)
                    attempt1_ref = self._compute_reference_log_prob(episode.attempt1_decisions)
                    kl_loss = attempt1_lp - attempt1_ref  # Approximate KL
                    total_loss = policy_loss + self.config.stage1_kl_coeff * kl_loss.abs()
                else:
                    total_loss = policy_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.coordinator.head.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / max(len(eps), 1)
            print(f"    [Epoch {epoch+1}] loss={avg_loss:.4f}")

    def train_stage2(self, episodes: Optional[List[CorrectionEpisode]] = None):
        """
        Stage II: Joint training with progress-based reward bonus.
        
        r_total = r(attempt2) + progress_weight * max(0, r(attempt2) - r(attempt1))
        
        This amplifies the reward for transitions that actually improve
        over the first attempt.
        """
        eps = episodes or self.episodes
        if not eps:
            print("  No correction episodes available for Stage II")
            return
        
        print("  === Stage II: Progress-bonus joint training ===")
        print(f"    Episodes: {len(eps)}")
        
        self.optimizer.param_groups[0]["lr"] = self.config.stage2_lr
        
        for epoch in range(self.config.stage2_epochs):
            epoch_loss = 0.0
            corrections_found = 0
            
            for episode in eps:
                self.optimizer.zero_grad()
                
                # Progress bonus: amplify reward for actual improvement
                progress_bonus = max(0.0, episode.progress_reward)
                total_reward = (
                    episode.attempt2_reward
                    + self.config.stage2_progress_weight * progress_bonus
                )
                
                if episode.corrected:
                    corrections_found += 1
                
                # Reinforce attempt 2 decisions with total reward
                attempt2_lp = self._compute_trajectory_log_prob(episode.attempt2_decisions)
                policy_loss = -total_reward * attempt2_lp
                
                # Light KL penalty (less restrictive than Stage I)
                if episode.attempt2_decisions:
                    ref_lp = self._compute_reference_log_prob(episode.attempt2_decisions)
                    kl = (attempt2_lp - ref_lp).abs()
                    policy_loss = policy_loss + self.config.stage2_kl_coeff * kl
                
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.coordinator.head.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                epoch_loss += policy_loss.item()
            
            avg_loss = epoch_loss / max(len(eps), 1)
            correction_rate = corrections_found / max(len(eps), 1)
            
            self.history.append({
                "stage": 2,
                "epoch": epoch + 1,
                "loss": avg_loss,
                "correction_rate": correction_rate,
            })
            
            if (epoch + 1) % 2 == 0:
                print(f"    [Epoch {epoch+1}] loss={avg_loss:.4f} "
                      f"corrections={correction_rate:.1%}")

    def train(self, queries: List[Tuple[str, Optional[str]]], collect_episodes: int = 20):
        """
        Full SCoRe training pipeline:
        1. Collect correction episodes
        2. Stage I: KL-constrained correction learning
        3. Stage II: Joint training with progress bonus
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=== SCoRe Self-Correction Training ===")
        print(f"  Queries: {len(queries)}")
        print(f"  Episodes to collect: {collect_episodes}")
        print()
        
        # 1. Collect correction episodes
        print("  Collecting correction episodes...")
        collected = 0
        attempts = 0
        while collected < collect_episodes and attempts < collect_episodes * 3:
            query, gt = queries[attempts % len(queries)]
            episode = self.collect_correction_episode(query, gt)
            if episode is not None:
                collected += 1
                status = "CORRECTED" if episode.corrected else "not improved"
                print(f"    [{collected}/{collect_episodes}] "
                      f"progress={episode.progress_reward:.2f} ({status})")
            attempts += 1
        
        print(f"  Collected {collected} episodes from {attempts} attempts")
        print()
        
        if not self.episodes:
            print("  No correction episodes collected. Skipping training.")
            return
        
        # 2. Stage I
        self.train_stage1()
        print()
        
        # 3. Stage II
        self.train_stage2()
        
        # Save
        self.coordinator.save(str(output_dir / "score_final.npy"))
        print(f"\n  Saved to {output_dir / 'score_final.npy'}")
        
        # Stats
        corrected_count = sum(1 for e in self.episodes if e.corrected)
        print("\n  Summary:")
        print(f"    Total episodes: {len(self.episodes)}")
        print(f"    Successfully corrected: {corrected_count} ({corrected_count/len(self.episodes):.1%})")
        print(f"    Avg progress: {np.mean([e.progress_reward for e in self.episodes]):.3f}")
