"""
GRPO / RLOO Training for the TRINITY coordinator.

From the catalog:
- GRPO estimates advantages from a GROUP of G sampled outputs per prompt.
  Rewards are normalized to construct baseline advantage metrics:
    A_i = (r_i - mean(r)) / std(r)
  
- RLOO constructs an unbiased baseline for sample i by computing the mean
  of the REMAINING G-1 samples (leave-one-out).

Applied to Fugu's coordinator:
- For each query, we sample G different routing decisions (model+role sequences)
- Each routing trajectory gets a reward from task outcome
- Advantages are computed via GRPO or RLOO normalization
- The coordinator head is updated via policy gradient weighted by advantages

This is COMPLEMENTARY to sep-CMA-ES:
- CMA-ES: population-based, derivative-free, good for initial training
- GRPO: gradient-based on the head logits, better for fine-grained refinement

The coordinator head outputs logits → we CAN compute gradients through softmax.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

from .coordinator import TrinityCoordinator, CoordinatorDecision
from .worker_pool import WorkerPool
from .orchestrator import FuguOrchestrator, OrchestrationResult


@dataclass
class GRPOConfig:
    """Configuration for GRPO/RLOO training."""
    group_size: int = 8          # G: number of routing samples per query
    learning_rate: float = 1e-4
    kl_coeff: float = 0.01       # KL divergence penalty (anchor to reference)
    clip_epsilon: float = 0.2    # PPO-style clipping
    max_grad_norm: float = 1.0
    method: str = "grpo"         # "grpo" or "rloo"
    epochs_per_batch: int = 3    # How many gradient steps per collected batch
    temperature: float = 1.0     # Sampling temperature for routing decisions
    entropy_bonus: float = 0.01  # Encourage exploration
    output_dir: str = "./checkpoints/grpo"


@dataclass
class RoutingTrajectory:
    """A single sampled routing trajectory for a query."""
    query: str
    decisions: List[CoordinatorDecision]
    result: OrchestrationResult
    reward: float
    log_probs: List[float]       # Log probabilities of each routing decision
    step_rewards: List[float]    # Per-step rewards (from PRM, if available)


class GRPOTrainer:
    """
    Trains the coordinator head via Group Relative Policy Optimization.
    
    Key insight: The coordinator head produces logits that go through softmax.
    We CAN backpropagate through the softmax to update the head weights using
    policy gradient with group-normalized advantages.
    
    This is more sample-efficient than CMA-ES for refining an already-decent policy.
    """

    def __init__(
        self,
        coordinator: TrinityCoordinator,
        worker_pool: WorkerPool,
        config: Optional[GRPOConfig] = None,
    ):
        self.coordinator = coordinator
        self.pool = worker_pool
        self.config = config or GRPOConfig()
        
        # Optimizer only for the head (SVF deltas handled separately)
        self.optimizer = torch.optim.Adam(
            self.coordinator.head.parameters(),
            lr=self.config.learning_rate,
        )
        
        # Reference policy (frozen copy of head at start of training)
        self._reference_head_params = self.coordinator.head.get_flat_params().copy()
        
        # Training history
        self.history: List[dict] = []

    def _sample_routing_decision(
        self,
        hidden_state: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[int, int, float]:
        """
        Sample a routing decision (model + role) from the coordinator head.
        
        Returns:
            (model_idx, role_idx, log_prob)
        """
        model_logits, role_logits = self.coordinator.head(hidden_state)
        
        # Apply temperature
        model_probs = F.softmax(model_logits / temperature, dim=-1)
        role_probs = F.softmax(role_logits / temperature, dim=-1)
        
        # Sample
        model_dist = torch.distributions.Categorical(model_probs)
        role_dist = torch.distributions.Categorical(role_probs)
        
        model_idx = model_dist.sample()
        role_idx = role_dist.sample()
        
        # Joint log probability
        log_prob = model_dist.log_prob(model_idx) + role_dist.log_prob(role_idx)
        
        return model_idx.item(), role_idx.item(), log_prob.item()

    def _compute_log_prob(
        self,
        hidden_state: torch.Tensor,
        model_idx: int,
        role_idx: int,
    ) -> torch.Tensor:
        """Compute log probability of a specific decision (differentiable)."""
        model_logits, role_logits = self.coordinator.head(hidden_state)
        
        model_log_probs = F.log_softmax(model_logits, dim=-1)
        role_log_probs = F.log_softmax(role_logits, dim=-1)
        
        log_prob = model_log_probs[0, model_idx] + role_log_probs[0, role_idx]
        return log_prob

    def _compute_kl_divergence(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """KL divergence between current policy and reference policy."""
        # Current policy
        model_logits, role_logits = self.coordinator.head(hidden_state)
        current_model_probs = F.softmax(model_logits, dim=-1)
        current_role_probs = F.softmax(role_logits, dim=-1)
        
        # Reference policy (reconstruct from saved params)
        with torch.no_grad():
            original_params = self.coordinator.head.get_flat_params()
            self.coordinator.head.set_flat_params(self._reference_head_params)
            ref_model_logits, ref_role_logits = self.coordinator.head(hidden_state)
            ref_model_probs = F.softmax(ref_model_logits, dim=-1)
            ref_role_probs = F.softmax(ref_role_logits, dim=-1)
            self.coordinator.head.set_flat_params(original_params)
        
        # KL(current || reference)
        kl_model = F.kl_div(
            current_model_probs.log(), ref_model_probs, reduction="batchmean"
        )
        kl_role = F.kl_div(
            current_role_probs.log(), ref_role_probs, reduction="batchmean"
        )
        
        return kl_model + kl_role

    def collect_group(
        self,
        query: str,
        ground_truth: Optional[str] = None,
    ) -> List[RoutingTrajectory]:
        """
        Collect G routing trajectories for a single query.
        Each trajectory uses stochastic sampling from the coordinator head.
        """
        trajectories = []
        
        for _ in range(self.config.group_size):
            orchestrator = FuguOrchestrator(
                coordinator=self.coordinator,
                worker_pool=self.pool,
                verbose=False,
            )
            result = orchestrator.run(query)
            
            # Compute outcome reward
            reward = 0.0
            if not result.answer:
                reward = -1.0
            else:
                if result.accepted:
                    reward += 1.0
                if ground_truth and ground_truth.strip().lower() in result.answer.lower():
                    reward += 0.5
                reward -= result.num_turns * 0.05
            
            # Collect log probs from each decision
            log_probs = []
            for turn in result.turns:
                d = turn.decision
                # Approximate log prob from logits
                model_lp = F.log_softmax(
                    torch.tensor(d.model_logits).unsqueeze(0), dim=-1
                )[0, d.model_index].item()
                role_lp = F.log_softmax(
                    torch.tensor(d.role_logits).unsqueeze(0), dim=-1
                )[0, d.role.value].item()
                log_probs.append(model_lp + role_lp)
            
            trajectories.append(RoutingTrajectory(
                query=query,
                decisions=[t.decision for t in result.turns],
                result=result,
                reward=reward,
                log_probs=log_probs,
                step_rewards=[],  # Filled by PRM if available
            ))
        
        return trajectories

    def compute_advantages_grpo(self, trajectories: List[RoutingTrajectory]) -> List[float]:
        """
        GRPO: Normalize rewards within the group.
        A_i = (r_i - mean(r)) / std(r)
        """
        rewards = [t.reward for t in trajectories]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        return [(r - mean_r) / std_r for r in rewards]

    def compute_advantages_rloo(self, trajectories: List[RoutingTrajectory]) -> List[float]:
        """
        RLOO: Leave-one-out baseline.
        A_i = r_i - mean(r_{j != i})
        """
        rewards = [t.reward for t in trajectories]
        n = len(rewards)
        total = sum(rewards)
        advantages = []
        for i, r in enumerate(rewards):
            baseline = (total - r) / (n - 1) if n > 1 else 0.0
            advantages.append(r - baseline)
        return advantages

    def update_step(
        self,
        trajectories: List[RoutingTrajectory],
        advantages: List[float],
    ) -> float:
        """
        Perform one gradient update on the coordinator head.
        
        Uses the GRPO/RLOO objective:
        L = -sum_i [ A_i * sum_t log π(a_t | s_t) ] + β * KL(π || π_ref)
        
        With PPO-style clipping on importance ratios.
        """
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        for traj, advantage in zip(trajectories, advantages):
            if not traj.decisions:
                continue
            
            traj_loss = torch.tensor(0.0, requires_grad=True)
            
            # Reconstruct transcript to get hidden states
            transcript = f"User query: {traj.query}\n"
            
            for i, decision in enumerate(traj.decisions):
                # Get hidden state for this transcript state
                h = self.coordinator.get_hidden_state(transcript)
                
                # Compute current log prob (differentiable)
                log_prob = self._compute_log_prob(h, decision.model_index, decision.role.value)
                
                # Policy gradient with advantage
                traj_loss = traj_loss - advantage * log_prob
                
                # Entropy bonus (encourage exploration)
                model_logits, role_logits = self.coordinator.head(h)
                model_entropy = torch.distributions.Categorical(
                    logits=model_logits
                ).entropy()
                role_entropy = torch.distributions.Categorical(
                    logits=role_logits
                ).entropy()
                traj_loss = traj_loss - self.config.entropy_bonus * (model_entropy + role_entropy)
                
                # Update transcript for next step
                if i < len(traj.result.turns):
                    turn = traj.result.turns[i]
                    role_label = turn.role.name.capitalize()
                    transcript += f"\n[{role_label} — {turn.worker_name}]:\n{turn.response.content}\n"
            
            total_loss = total_loss + traj_loss
        
        # KL penalty (use first transcript as representative)
        if trajectories and trajectories[0].decisions:
            h = self.coordinator.get_hidden_state(f"User query: {trajectories[0].query}\n")
            kl = self._compute_kl_divergence(h)
            total_loss = total_loss + self.config.kl_coeff * kl
        
        # Average over group
        total_loss = total_loss / max(len(trajectories), 1)
        
        # Backward + clip + step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.coordinator.head.parameters(),
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        
        return total_loss.item()

    def train_on_queries(
        self,
        queries: List[Tuple[str, Optional[str]]],
        num_epochs: int = 10,
    ):
        """
        Full GRPO/RLOO training loop over a set of queries.
        
        Args:
            queries: List of (query, ground_truth) tuples
            num_epochs: Number of passes over all queries
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"=== GRPO Training ({self.config.method.upper()}) ===")
        print(f"  Group size: {self.config.group_size}")
        print(f"  Queries: {len(queries)}")
        print(f"  Epochs: {num_epochs}")
        print(f"  LR: {self.config.learning_rate}")
        print(f"  KL coeff: {self.config.kl_coeff}")
        print()
        
        compute_advantages = (
            self.compute_advantages_grpo if self.config.method == "grpo"
            else self.compute_advantages_rloo
        )
        
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_rewards = []
            
            for query, gt in queries:
                # 1. Collect G trajectories
                trajectories = self.collect_group(query, ground_truth=gt)
                
                # 2. Compute advantages
                advantages = compute_advantages(trajectories)
                
                # 3. Policy gradient update
                for _ in range(self.config.epochs_per_batch):
                    loss = self.update_step(trajectories, advantages)
                    epoch_losses.append(loss)
                
                epoch_rewards.extend([t.reward for t in trajectories])
            
            # Log
            mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            mean_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
            
            entry = {
                "epoch": epoch + 1,
                "mean_loss": mean_loss,
                "mean_reward": mean_reward,
            }
            self.history.append(entry)
            
            print(f"  [Epoch {epoch+1:3d}] loss={mean_loss:.4f} reward={mean_reward:.3f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.coordinator.save(str(output_dir / f"grpo_epoch_{epoch+1}.npy"))
        
        # Final save
        self.coordinator.save(str(output_dir / "grpo_final.npy"))
        print(f"\n  Saved to {output_dir / 'grpo_final.npy'}")
