"""
DPO / SimPO / KTO Training for the coordinator head.

From the catalog:
- DPO: Replaces explicit reward modeling by expressing the latent reward in
  closed form using the policy model. Optimizes over preference pairs (x, y_w, y_l).
  L_DPO = -log σ(β * [log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)])

- SimPO: Reference-free, uses average token log probability as implicit reward.
  No reference model needed.

- KTO: Unpaired binary utility optimization. Only needs (x, y) + desirable/undesirable label.

Applied to Fugu's coordinator:
- Preference pairs = routing trajectories ranked by reward
  - "chosen" = routing sequence that led to correct answer
  - "rejected" = routing sequence that failed
- The coordinator head's log probs over (model, role) decisions form the policy

This is the OFFLINE counterpart to online GRPO:
- GRPO: online, samples fresh trajectories
- DPO: offline, learns from stored preference pairs
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from .coordinator import TrinityCoordinator
from .orchestrator import OrchestrationResult


@dataclass
class RoutingPreferencePair:
    """A preference pair for DPO training on routing decisions."""
    query: str
    # Chosen trajectory (higher reward)
    chosen_decisions: List[dict]   # [{model_index, role_value, transcript_state}]
    chosen_reward: float
    # Rejected trajectory (lower reward)
    rejected_decisions: List[dict]
    rejected_reward: float


@dataclass
class DPOConfig:
    """Configuration for DPO/SimPO/KTO training."""
    beta: float = 0.1              # Temperature parameter
    learning_rate: float = 5e-5
    method: str = "dpo"            # "dpo", "simpo", or "kto"
    max_grad_norm: float = 1.0
    batch_size: int = 4
    num_epochs: int = 10
    min_reward_gap: float = 0.3    # Minimum reward difference for valid pair
    label_smoothing: float = 0.0   # Label smoothing (prevents overconfident updates)
    simpo_gamma: float = 0.5       # SimPO margin parameter
    kto_beta_desirable: float = 0.1
    kto_beta_undesirable: float = 0.2  # Loss aversion (Kahneman-Tversky)
    output_dir: str = "./checkpoints/dpo"


class TrajectoryStore:
    """
    Stores orchestration trajectories for offline DPO training.
    
    Accumulates routing traces over time, then generates preference pairs
    by comparing high-reward vs low-reward trajectories for the same query.
    """

    def __init__(self):
        self.trajectories: List[dict] = []

    def add(
        self,
        query: str,
        result: OrchestrationResult,
        reward: float,
    ):
        """Store a completed orchestration trajectory."""
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
        
        self.trajectories.append({
            "query": query,
            "decisions": decisions,
            "reward": reward,
            "accepted": result.accepted,
            "num_turns": result.num_turns,
        })

    def generate_pairs(self, min_reward_gap: float = 0.3) -> List[RoutingPreferencePair]:
        """
        Generate DPO preference pairs from stored trajectories.
        
        Groups trajectories by query, sorts by reward,
        pairs highest with lowest (must have min_reward_gap).
        """
        # Group by query
        groups: dict = {}
        for traj in self.trajectories:
            q = traj["query"]
            if q not in groups:
                groups[q] = []
            groups[q].append(traj)
        
        pairs = []
        for query, group in groups.items():
            if len(group) < 2:
                continue
            
            # Sort by reward (descending)
            group.sort(key=lambda x: x["reward"], reverse=True)
            
            best = group[0]
            worst = group[-1]
            
            if best["reward"] - worst["reward"] >= min_reward_gap:
                pairs.append(RoutingPreferencePair(
                    query=query,
                    chosen_decisions=best["decisions"],
                    chosen_reward=best["reward"],
                    rejected_decisions=worst["decisions"],
                    rejected_reward=worst["reward"],
                ))
        
        return pairs

    def save(self, path: str):
        """Save trajectory store to JSON."""
        with open(path, "w") as f:
            json.dump(self.trajectories, f, indent=2)

    def load(self, path: str):
        """Load trajectory store from JSON."""
        with open(path) as f:
            self.trajectories = json.load(f)


class DPOTrainer:
    """
    Offline preference optimization for the coordinator head.
    
    DPO loss:
      L = -log σ(β * [log π(chosen) - log π_ref(chosen) - log π(rejected) + log π_ref(rejected)])
    
    SimPO (reference-free):
      L = -log σ(β * [avg_log_prob(chosen) - avg_log_prob(rejected) - γ])
    
    KTO (unpaired):
      L = -KTO_utility(y | desirable/undesirable label)
    """

    def __init__(
        self,
        coordinator: TrinityCoordinator,
        config: Optional[DPOConfig] = None,
    ):
        self.coordinator = coordinator
        self.config = config or DPOConfig()
        
        self.optimizer = torch.optim.Adam(
            self.coordinator.head.parameters(),
            lr=self.config.learning_rate,
        )
        
        # Reference policy (frozen head params at training start)
        self._reference_params = self.coordinator.head.get_flat_params().copy()
        
        self.history: List[dict] = []

    def _compute_trajectory_log_prob(
        self,
        decisions: List[dict],
    ) -> torch.Tensor:
        """
        Compute total log probability of a routing trajectory under current policy.
        Sum of log π(model_i, role_i | transcript_i) for each step.
        """
        total_log_prob = torch.tensor(0.0)
        
        for decision in decisions:
            transcript = decision["transcript_state"]
            model_idx = decision["model_index"]
            role_idx = decision["role_value"]
            
            h = self.coordinator.get_hidden_state(transcript)
            model_logits, role_logits = self.coordinator.head(h)
            
            model_lp = F.log_softmax(model_logits, dim=-1)[0, model_idx]
            role_lp = F.log_softmax(role_logits, dim=-1)[0, role_idx]
            
            total_log_prob = total_log_prob + model_lp + role_lp
        
        return total_log_prob

    def _compute_trajectory_log_prob_reference(
        self,
        decisions: List[dict],
    ) -> torch.Tensor:
        """Compute log prob under the reference (frozen) policy."""
        original_params = self.coordinator.head.get_flat_params()
        self.coordinator.head.set_flat_params(self._reference_params)
        
        with torch.no_grad():
            log_prob = self._compute_trajectory_log_prob(decisions)
        
        self.coordinator.head.set_flat_params(original_params)
        return log_prob

    def _dpo_loss(self, pair: RoutingPreferencePair) -> torch.Tensor:
        """
        Standard DPO loss:
        L = -log σ(β * [log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)])
        """
        # Current policy log probs
        chosen_lp = self._compute_trajectory_log_prob(pair.chosen_decisions)
        rejected_lp = self._compute_trajectory_log_prob(pair.rejected_decisions)
        
        # Reference policy log probs
        chosen_ref_lp = self._compute_trajectory_log_prob_reference(pair.chosen_decisions)
        rejected_ref_lp = self._compute_trajectory_log_prob_reference(pair.rejected_decisions)
        
        # DPO reward difference
        chosen_ratio = chosen_lp - chosen_ref_lp
        rejected_ratio = rejected_lp - rejected_ref_lp
        
        logit = self.config.beta * (chosen_ratio - rejected_ratio)
        
        # Label smoothing
        if self.config.label_smoothing > 0:
            loss = (
                -(1 - self.config.label_smoothing) * F.logsigmoid(logit)
                - self.config.label_smoothing * F.logsigmoid(-logit)
            )
        else:
            loss = -F.logsigmoid(logit)
        
        return loss

    def _simpo_loss(self, pair: RoutingPreferencePair) -> torch.Tensor:
        """
        SimPO loss (reference-free):
        L = -log σ(β * [avg_log_prob(chosen) - avg_log_prob(rejected) - γ])
        
        Uses average log probability as implicit reward (no reference model needed).
        """
        chosen_lp = self._compute_trajectory_log_prob(pair.chosen_decisions)
        rejected_lp = self._compute_trajectory_log_prob(pair.rejected_decisions)
        
        # Length-normalize (average per-step log prob)
        n_chosen = max(len(pair.chosen_decisions), 1)
        n_rejected = max(len(pair.rejected_decisions), 1)
        
        avg_chosen = chosen_lp / n_chosen
        avg_rejected = rejected_lp / n_rejected
        
        # SimPO margin
        logit = self.config.beta * (avg_chosen - avg_rejected - self.config.simpo_gamma)
        
        return -F.logsigmoid(logit)

    def _kto_loss(
        self,
        decisions: List[dict],
        is_desirable: bool,
    ) -> torch.Tensor:
        """
        KTO loss (unpaired, uses Kahneman-Tversky prospect theory):
        - Desirable outputs: maximize (r - baseline)
        - Undesirable outputs: penalize more heavily (loss aversion)
        """
        log_prob = self._compute_trajectory_log_prob(decisions)
        ref_log_prob = self._compute_trajectory_log_prob_reference(decisions)
        
        ratio = log_prob - ref_log_prob
        
        if is_desirable:
            # Utility for desirable outputs
            loss = -F.logsigmoid(self.config.kto_beta_desirable * ratio)
        else:
            # Loss aversion for undesirable outputs (heavier penalty)
            loss = -F.logsigmoid(-self.config.kto_beta_undesirable * ratio)
        
        return loss

    def train(self, pairs: List[RoutingPreferencePair]):
        """
        Train the coordinator head on preference pairs.
        
        Supports DPO, SimPO, and KTO methods.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"=== {self.config.method.upper()} Training ===")
        print(f"  Pairs: {len(pairs)}")
        print(f"  Beta: {self.config.beta}")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  LR: {self.config.learning_rate}")
        print()
        
        for epoch in range(self.config.num_epochs):
            # Shuffle pairs
            indices = np.random.permutation(len(pairs))
            epoch_losses = []
            
            for i in range(0, len(pairs), self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                batch_loss = torch.tensor(0.0, requires_grad=True)
                
                for idx in batch_indices:
                    pair = pairs[idx]
                    
                    if self.config.method == "dpo":
                        loss = self._dpo_loss(pair)
                    elif self.config.method == "simpo":
                        loss = self._simpo_loss(pair)
                    elif self.config.method == "kto":
                        # KTO: treat chosen as desirable, rejected as undesirable
                        loss_chosen = self._kto_loss(pair.chosen_decisions, is_desirable=True)
                        loss_rejected = self._kto_loss(pair.rejected_decisions, is_desirable=False)
                        loss = loss_chosen + loss_rejected
                    else:
                        raise ValueError(f"Unknown method: {self.config.method}")
                    
                    batch_loss = batch_loss + loss
                
                batch_loss = batch_loss / len(batch_indices)
                
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.coordinator.head.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                
                epoch_losses.append(batch_loss.item())
            
            mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            self.history.append({"epoch": epoch + 1, "mean_loss": mean_loss})
            
            if (epoch + 1) % 2 == 0:
                print(f"  [Epoch {epoch+1:3d}] loss={mean_loss:.4f}")
        
        # Save
        self.coordinator.save(str(output_dir / f"{self.config.method}_final.npy"))
        print(f"\n  Saved to {output_dir / f'{self.config.method}_final.npy'}")
