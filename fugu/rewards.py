"""
Process Reward Model (PRM) and Outcome Reward Model (ORM) for Fugu.

From the catalog:
- PRM: Scores INTERMEDIATE reasoning steps (each turn in orchestration)
- ORM: Evaluates only the FINAL response correctness
- RBRM: Rule-Based Reward Model (programmatic checks)

Applied to Fugu's multi-turn orchestration:
- Each turn gets a step reward based on:
  1. Role appropriateness (was Thinker/Worker/Verifier used at the right time?)
  2. Model selection quality (was the right model picked for the task type?)
  3. Progress signal (did this turn move toward solution?)
  4. Efficiency (was this turn redundant?)

The PRM enables:
- Credit assignment to individual routing decisions (not just final outcome)
- Denser reward signal for GRPO training
- Better advantage estimation
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict

from .coordinator import Role
from .orchestrator import OrchestrationResult, Turn


@dataclass
class StepReward:
    """Reward breakdown for a single orchestration turn."""
    turn_number: int
    role_appropriateness: float   # Was the role choice good for this stage?
    model_quality: float          # Was the model a good fit?
    progress_signal: float        # Did this turn make progress?
    efficiency_signal: float      # Was this turn non-redundant?
    total: float                  # Weighted sum
    
    @property
    def components(self) -> Dict[str, float]:
        return {
            "role_appropriateness": self.role_appropriateness,
            "model_quality": self.model_quality,
            "progress_signal": self.progress_signal,
            "efficiency_signal": self.efficiency_signal,
        }


@dataclass 
class RewardConfig:
    """Weights for the PRM components."""
    # Role appropriateness weights
    w_role: float = 0.3
    # Model quality weights
    w_model: float = 0.2
    # Progress signal weight
    w_progress: float = 0.3
    # Efficiency weight
    w_efficiency: float = 0.2
    
    # Discount factor for temporal credit assignment
    gamma: float = 0.95
    
    # Outcome reward scaling
    outcome_weight: float = 1.0
    process_weight: float = 0.5


class ProcessRewardModel:
    """
    Step-level reward model for orchestration turns.
    
    Assigns dense rewards to each routing decision, enabling:
    - Better credit assignment in GRPO/RLOO
    - Identification of which turns contributed to success/failure
    - Training signal even when final outcome is ambiguous
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def score_trajectory(
        self,
        result: OrchestrationResult,
        ground_truth: Optional[str] = None,
    ) -> List[StepReward]:
        """
        Score each turn in an orchestration trajectory.
        
        Returns step-level rewards that can be used for advantage estimation.
        """
        turns = result.turns
        n_turns = len(turns)
        step_rewards = []
        
        for i, turn in enumerate(turns):
            # 1. Role appropriateness
            role_score = self._score_role(turn, i, n_turns, result.accepted)
            
            # 2. Model quality
            model_score = self._score_model(turn)
            
            # 3. Progress signal
            progress_score = self._score_progress(turn, i, turns)
            
            # 4. Efficiency
            efficiency_score = self._score_efficiency(turn, i, turns)
            
            # Weighted total
            total = (
                self.config.w_role * role_score
                + self.config.w_model * model_score
                + self.config.w_progress * progress_score
                + self.config.w_efficiency * efficiency_score
            )
            
            step_rewards.append(StepReward(
                turn_number=i + 1,
                role_appropriateness=role_score,
                model_quality=model_score,
                progress_signal=progress_score,
                efficiency_signal=efficiency_score,
                total=total,
            ))
        
        # Apply temporal discount from outcome
        outcome_reward = self._compute_outcome_reward(result, ground_truth)
        step_rewards = self._apply_temporal_credit(step_rewards, outcome_reward)
        
        return step_rewards

    def _score_role(
        self,
        turn: Turn,
        turn_idx: int,
        total_turns: int,
        accepted: bool,
    ) -> float:
        """
        Score role appropriateness based on position in trajectory.
        
        Heuristic priors:
        - Thinker should appear early (planning phase)
        - Worker should appear in the middle (execution phase)
        - Verifier should appear late (validation phase)
        - Bonus if Verifier accepted at the end
        """
        role = turn.role
        position_ratio = turn_idx / max(total_turns - 1, 1)
        
        if role == Role.THINKER:
            # Best early: reward peaks at position 0, decays toward 1
            score = 1.0 - position_ratio
        elif role == Role.WORKER:
            # Best in middle: Gaussian-like peak at 0.5
            score = 1.0 - 2.0 * abs(position_ratio - 0.5)
        elif role == Role.VERIFIER:
            # Best late: reward grows with position
            score = position_ratio
            # Bonus for accepting at the end
            if turn.response.accepted:
                score += 0.5
        else:
            score = 0.0
        
        return max(0.0, min(1.0, score))

    def _score_model(self, turn: Turn) -> float:
        """
        Score model selection quality.
        
        Simple heuristic based on response quality indicators:
        - Non-empty response: base score
        - Length appropriate for role
        - No error markers
        """
        response = turn.response
        content = response.content
        
        if not content or content.startswith("[Worker"):
            return 0.0
        
        # Base score for producing output
        score = 0.5
        
        # Length appropriateness by role
        word_count = len(content.split())
        if turn.role == Role.THINKER:
            # Thinkers should produce moderate-length plans
            if 20 <= word_count <= 200:
                score += 0.3
            elif word_count > 200:
                score += 0.1  # Too verbose
        elif turn.role == Role.WORKER:
            # Workers should produce substantial output
            if word_count >= 10:
                score += 0.3
        elif turn.role == Role.VERIFIER:
            # Verifiers should be concise
            if word_count <= 100:
                score += 0.3
        
        # Bonus for tokens used (model actually engaged)
        if response.tokens_used > 0:
            score += 0.2
        
        return min(1.0, score)

    def _score_progress(
        self,
        turn: Turn,
        turn_idx: int,
        all_turns: List[Turn],
    ) -> float:
        """
        Score whether this turn made progress toward the solution.
        
        Heuristics:
        - Verifier ACCEPT = maximum progress
        - Worker producing unique content = good progress
        - Thinker providing new structure = moderate progress
        - Repeating previous content = no progress
        """
        content = turn.response.content
        
        if turn.role == Role.VERIFIER and turn.response.accepted:
            return 1.0
        
        if not content:
            return 0.0
        
        # Check for repetition with previous turns
        for prev_turn in all_turns[:turn_idx]:
            if prev_turn.response.content and content == prev_turn.response.content:
                return -0.5  # Exact repetition is bad
            
            # Rough similarity check
            if prev_turn.response.content:
                overlap = self._content_overlap(content, prev_turn.response.content)
                if overlap > 0.8:
                    return 0.0  # High overlap = no progress
        
        # Non-repetitive content is progress
        return 0.5

    def _score_efficiency(
        self,
        turn: Turn,
        turn_idx: int,
        all_turns: List[Turn],
    ) -> float:
        """
        Score turn efficiency (was this turn necessary?).
        
        - Redundant roles in sequence = inefficient
        - Same model called twice in a row for same role = inefficient
        """
        if turn_idx == 0:
            return 1.0  # First turn is always necessary
        
        prev_turn = all_turns[turn_idx - 1]
        
        # Same role twice in a row (except Worker)
        if turn.role == prev_turn.role and turn.role != Role.WORKER:
            return 0.2
        
        # Same model + same role = definitely redundant
        if (turn.decision.model_index == prev_turn.decision.model_index
                and turn.role == prev_turn.role):
            return 0.0
        
        return 0.8

    def _content_overlap(self, a: str, b: str) -> float:
        """Rough word-level overlap ratio."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        return len(intersection) / min(len(words_a), len(words_b))

    def _compute_outcome_reward(
        self,
        result: OrchestrationResult,
        ground_truth: Optional[str] = None,
    ) -> float:
        """Compute the final outcome reward (ORM)."""
        if not result.answer:
            return -1.0
        
        reward = 0.0
        if result.accepted:
            reward += 1.0
        if ground_truth and ground_truth.strip().lower() in result.answer.lower():
            reward += 1.0
        
        return reward

    def _apply_temporal_credit(
        self,
        step_rewards: List[StepReward],
        outcome_reward: float,
    ) -> List[StepReward]:
        """
        Apply discounted outcome reward to step rewards (GAE-like).
        
        Each step gets:
          final_reward = process_weight * step_reward + outcome_weight * gamma^(T-t) * outcome
        
        This propagates the final outcome back to earlier decisions with temporal discount.
        """
        n = len(step_rewards)
        for i, sr in enumerate(step_rewards):
            steps_from_end = n - 1 - i
            discounted_outcome = (self.config.gamma ** steps_from_end) * outcome_reward
            sr.total = (
                self.config.process_weight * sr.total
                + self.config.outcome_weight * discounted_outcome
            )
        
        return step_rewards


class RuleBasedRewardModel:
    """
    RBRM: Programmatic reward checks for structural compliance.
    
    Checks hard constraints that should always be satisfied:
    - Verifier must appear before acceptance
    - At least one Worker must appear before Verifier
    - No more than 2 consecutive same-role turns
    """

    def score_structural_compliance(self, result: OrchestrationResult) -> float:
        """Score whether the orchestration followed structural rules."""
        turns = result.turns
        if not turns:
            return 0.0
        
        score = 1.0
        violations = 0
        
        # Rule 1: Worker must appear before Verifier accepts
        worker_appeared = False
        for turn in turns:
            if turn.role == Role.WORKER:
                worker_appeared = True
            if turn.role == Role.VERIFIER and turn.response.accepted:
                if not worker_appeared:
                    violations += 1
                    score -= 0.3
        
        # Rule 2: No more than 3 consecutive same-role turns
        consecutive = 1
        for i in range(1, len(turns)):
            if turns[i].role == turns[i-1].role:
                consecutive += 1
                if consecutive > 3:
                    violations += 1
                    score -= 0.2
            else:
                consecutive = 1
        
        # Rule 3: Thinker should not be the last role if not accepted
        if not result.accepted and turns[-1].role == Role.THINKER:
            score -= 0.1
        
        return max(0.0, score)


class CombinedRewardModel:
    """
    Combines PRM + ORM + RBRM into a unified reward signal.
    
    This is used by the GRPO trainer to get dense, multi-component rewards
    for each routing trajectory.
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.prm = ProcessRewardModel(config)
        self.rbrm = RuleBasedRewardModel()
        self.config = config or RewardConfig()

    def score(
        self,
        result: OrchestrationResult,
        ground_truth: Optional[str] = None,
    ) -> tuple:
        """
        Compute full reward breakdown.
        
        Returns:
            (total_reward, step_rewards, structural_score)
        """
        # Step-level PRM
        step_rewards = self.prm.score_trajectory(result, ground_truth)
        
        # Structural compliance (RBRM)
        structural_score = self.rbrm.score_structural_compliance(result)
        
        # Outcome (ORM)
        outcome = self.prm._compute_outcome_reward(result, ground_truth)
        
        # Combined total
        step_total = np.mean([sr.total for sr in step_rewards]) if step_rewards else 0.0
        total_reward = (
            0.4 * outcome
            + 0.4 * step_total
            + 0.2 * structural_score
        )
        
        return total_reward, step_rewards, structural_score
