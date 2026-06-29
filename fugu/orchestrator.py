"""
Orchestrator — runs the multi-turn Thinker/Worker/Verifier loop.

Given a user query Q:
1. Coordinator reads transcript, picks model + role
2. Worker is dispatched with role-specific prompt
3. Response is appended to transcript
4. Repeat until Verifier ACCEPTs or max_turns reached
5. Return the final accepted answer to the user

This is the "Fugu Mini" mode — single coordinator making per-turn decisions.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .coordinator import TrinityCoordinator, CoordinatorConfig, CoordinatorDecision, Role
from .worker_pool import WorkerPool, WorkerConfig, WorkerResponse


@dataclass
class Turn:
    """Record of a single orchestration turn."""
    turn_number: int
    decision: CoordinatorDecision
    worker_name: str
    role: Role
    response: WorkerResponse


@dataclass
class OrchestrationResult:
    """Complete result of orchestrating a query."""
    query: str
    answer: str
    turns: List[Turn] = field(default_factory=list)
    accepted: bool = False
    total_tokens: int = 0
    num_turns: int = 0


class FuguOrchestrator:
    """
    The main Fugu Mini orchestration loop.
    
    Coordinator selects model + role at each turn.
    Loop terminates when Verifier ACCEPTs or max_turns is hit.
    """

    def __init__(
        self,
        coordinator: Optional[TrinityCoordinator] = None,
        worker_pool: Optional[WorkerPool] = None,
        max_turns: int = 8,
        verbose: bool = True,
    ):
        self.coordinator = coordinator or TrinityCoordinator()
        self.pool = worker_pool or WorkerPool()
        self.max_turns = max_turns
        self.verbose = verbose

        # Ensure coordinator knows how many workers exist
        if self.coordinator.config.num_workers != self.pool.num_workers:
            self.coordinator.config.num_workers = self.pool.num_workers

    def run(self, query: str) -> OrchestrationResult:
        """
        Orchestrate a complete response to a user query.
        
        The coordinator iterates through turns, selecting workers and roles,
        until either:
        - A Verifier role returns ACCEPT
        - max_turns is reached (last Worker output is used)
        """
        turns: List[Turn] = []
        transcript = f"User query: {query}\n"
        total_tokens = 0
        last_worker_content = ""

        for turn_num in range(self.max_turns):
            # 1. Coordinator decides
            decision = self.coordinator.decide(transcript)
            
            # Clamp model index to valid range
            model_idx = min(decision.model_index, self.pool.num_workers - 1)
            model_idx = max(0, model_idx)
            worker_name = self.pool.workers[model_idx].name

            if self.verbose:
                print(f"  [Turn {turn_num + 1}] {worker_name} as {decision.role.name}")

            # 2. Dispatch to worker
            response = self.pool.dispatch(
                worker_index=model_idx,
                role=decision.role,
                query=query,
                transcript=transcript,
            )

            total_tokens += response.tokens_used

            # 3. Record turn
            turn = Turn(
                turn_number=turn_num + 1,
                decision=decision,
                worker_name=worker_name,
                role=decision.role,
                response=response,
            )
            turns.append(turn)

            # 4. Append to transcript
            role_label = decision.role.name.capitalize()
            transcript += f"\n[{role_label} — {worker_name}]:\n{response.content}\n"

            # 5. Check termination
            if decision.role == Role.VERIFIER:
                if response.accepted:
                    if self.verbose:
                        print(f"  ✓ Verifier ACCEPTED at turn {turn_num + 1}")
                    return OrchestrationResult(
                        query=query,
                        answer=last_worker_content or response.content,
                        turns=turns,
                        accepted=True,
                        total_tokens=total_tokens,
                        num_turns=turn_num + 1,
                    )
                else:
                    if self.verbose:
                        diag = response.diagnosis or "no details"
                        print(f"  ✗ Verifier REVISED: {diag[:80]}")
            elif decision.role == Role.WORKER:
                last_worker_content = response.content

        # Max turns reached without acceptance
        if self.verbose:
            print(f"  ⚠ Max turns ({self.max_turns}) reached without acceptance")

        return OrchestrationResult(
            query=query,
            answer=last_worker_content or turns[-1].response.content if turns else "",
            turns=turns,
            accepted=False,
            total_tokens=total_tokens,
            num_turns=len(turns),
        )

    def run_with_reward(self, query: str, ground_truth: Optional[str] = None) -> tuple:
        """
        Run orchestration and compute a reward signal for training.
        
        Returns:
            (result, reward): OrchestrationResult and float reward
        """
        result = self.run(query)
        
        # Reward heuristic:
        # +1.0 if verifier accepted
        # +0.5 if answer matches ground truth (when provided)
        # -0.2 per turn used (efficiency penalty)
        # -1.0 if no answer produced
        
        reward = 0.0
        
        if not result.answer:
            reward = -1.0
        else:
            if result.accepted:
                reward += 1.0
            
            if ground_truth and ground_truth.strip().lower() in result.answer.lower():
                reward += 0.5
            
            # Efficiency: fewer turns = better
            efficiency_penalty = result.num_turns * 0.05
            reward -= efficiency_penalty
        
        return result, reward
