"""
sep-CMA-ES Training for the TRINITY coordinator.

Key insight from the paper:
- The coordination head has ~19.5K parameters with BLOCK-SEPARABLE structure
- This means sep-CMA-ES (diagonal covariance) is near-optimal
- Improvement grows LINEARLY with iterations (vs log for random search)
- No gradients needed — just binary reward (task solved or not)

Training loop:
1. Sample population of perturbed parameter vectors
2. For each candidate: set params -> run orchestration -> get reward
3. Fitness-weighted recombination -> new parent
4. Repeat

Budget: ~1.5K-40K evaluations for ~10K-20K parameters.
"""

import numpy as np
import cma
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, Tuple

from .coordinator import TrinityCoordinator, CoordinatorConfig
from .worker_pool import WorkerPool
from .orchestrator import FuguOrchestrator


@dataclass
class TrainingConfig:
    """Configuration for sep-CMA-ES training."""
    sigma0: float = 0.5            # Initial step size
    population_size: int = 32      # lambda (CMA-ES population)
    max_iterations: int = 200      # Total CMA-ES generations
    num_evals_per_candidate: int = 16  # Replications per candidate (reduce noise)
    save_every: int = 10           # Save checkpoint every N iterations
    output_dir: str = "./checkpoints"
    seed: int = 42
    
    # Early stopping
    min_improvement: float = 0.001
    patience: int = 20


@dataclass
class EvalTask:
    """A single evaluation task for computing fitness."""
    query: str
    ground_truth: Optional[str] = None  # Expected answer (for reward computation)
    difficulty: str = "easy"


# Default training tasks (GSM8K-style + general knowledge)
DEFAULT_TASKS = [
    EvalTask("What is 15 * 7?", "105"),
    EvalTask("Write a Python function that reverses a string.", "def reverse"),
    EvalTask("What is the capital of France?", "Paris"),
    EvalTask("Solve: If x + 3 = 10, what is x?", "7"),
    EvalTask("Explain the difference between a list and a tuple in Python.", "immutable"),
    EvalTask("What is 2^10?", "1024"),
    EvalTask("Write a bash command to find all .py files in the current directory.", "find"),
    EvalTask("What is the time complexity of binary search?", "log"),
]


class CMAESTrainer:
    """
    Trains the TRINITY coordinator via sep-CMA-ES.
    
    The fitness function runs full orchestration episodes and scores
    the coordinator's routing decisions by task success rate.
    """

    def __init__(
        self,
        coordinator: TrinityCoordinator,
        worker_pool: WorkerPool,
        tasks: Optional[List[EvalTask]] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.coordinator = coordinator
        self.pool = worker_pool
        self.tasks = tasks or DEFAULT_TASKS
        self.config = config or TrainingConfig()
        self.orchestrator = FuguOrchestrator(
            coordinator=coordinator,
            worker_pool=worker_pool,
            verbose=False,
        )
        
        # Training state
        self.history: List[dict] = []
        self.best_reward: float = -float("inf")
        self.best_params: Optional[np.ndarray] = None

    def fitness(self, params: np.ndarray) -> float:
        """
        Evaluate a candidate parameter vector.
        
        Sets coordinator params, runs tasks, returns NEGATIVE reward
        (CMA-ES minimizes, so we negate the reward).
        """
        self.coordinator.set_flat_params(params)
        
        total_reward = 0.0
        num_tasks = min(self.config.num_evals_per_candidate, len(self.tasks))
        
        # Sample tasks for this evaluation
        task_indices = np.random.choice(len(self.tasks), num_tasks, replace=True)
        
        for idx in task_indices:
            task = self.tasks[idx]
            try:
                _, reward = self.orchestrator.run_with_reward(
                    query=task.query,
                    ground_truth=task.ground_truth,
                )
                total_reward += reward
            except Exception as e:
                total_reward -= 1.0  # Crash penalty

        avg_reward = total_reward / num_tasks
        return -avg_reward  # Negate for minimization

    def train(self, custom_fitness: Optional[Callable] = None):
        """
        Run sep-CMA-ES training loop.
        
        Args:
            custom_fitness: Optional custom fitness function(params) -> float (lower is better)
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get initial params
        x0 = self.coordinator.get_flat_params()
        n_params = len(x0)
        
        print(f"=== sep-CMA-ES Training ===")
        print(f"  Parameters: {n_params}")
        print(f"  Population: {self.config.population_size}")
        print(f"  Max iterations: {self.config.max_iterations}")
        print(f"  Tasks per eval: {self.config.num_evals_per_candidate}")
        print(f"  Output: {output_dir}")
        print()

        # Configure CMA-ES with separable (diagonal) covariance
        opts = {
            "seed": self.config.seed,
            "popsize": self.config.population_size,
            "maxiter": self.config.max_iterations,
            "CMA_diagonal": True,  # KEY: use sep-CMA-ES (diagonal covariance)
            "verb_disp": 0,        # We handle our own logging
            "tolfun": self.config.min_improvement,
        }

        es = cma.CMAEvolutionStrategy(x0, self.config.sigma0, opts)
        
        eval_fn = custom_fitness or self.fitness
        no_improvement_count = 0
        
        iteration = 0
        while not es.stop():
            iteration += 1
            t0 = time.time()
            
            # Sample population
            solutions = es.ask()
            
            # Evaluate each candidate
            fitnesses = [eval_fn(s) for s in solutions]
            
            # Update CMA-ES
            es.tell(solutions, fitnesses)
            
            # Track best
            best_fitness = min(fitnesses)
            best_reward = -best_fitness  # Convert back to reward
            mean_reward = -np.mean(fitnesses)
            
            elapsed = time.time() - t0
            
            # Record history
            entry = {
                "iteration": iteration,
                "best_reward": best_reward,
                "mean_reward": mean_reward,
                "sigma": es.sigma,
                "elapsed_sec": elapsed,
            }
            self.history.append(entry)
            
            # Update global best
            if best_reward > self.best_reward:
                self.best_reward = best_reward
                self.best_params = solutions[np.argmin(fitnesses)].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Logging
            if iteration % 5 == 0 or iteration == 1:
                print(f"  [Iter {iteration:4d}] best={best_reward:.3f} "
                      f"mean={mean_reward:.3f} sigma={es.sigma:.4f} "
                      f"({elapsed:.1f}s)")
            
            # Checkpointing
            if iteration % self.config.save_every == 0:
                self._save_checkpoint(output_dir, iteration)

            # Early stopping
            if no_improvement_count >= self.config.patience:
                print(f"\n  Early stopping at iter {iteration} (no improvement for {self.config.patience} iters)")
                break

        # Final save
        self._save_checkpoint(output_dir, iteration, final=True)
        
        # Restore best params
        if self.best_params is not None:
            self.coordinator.set_flat_params(self.best_params)
        
        print(f"\n=== Training Complete ===")
        print(f"  Best reward: {self.best_reward:.3f}")
        print(f"  Total iterations: {iteration}")
        print(f"  Final checkpoint: {output_dir / 'coordinator_final.npy'}")

    def _save_checkpoint(self, output_dir: Path, iteration: int, final: bool = False):
        """Save current best parameters and training history."""
        if self.best_params is not None:
            suffix = "final" if final else f"iter_{iteration}"
            np.save(output_dir / f"coordinator_{suffix}.npy", self.best_params)
        
        # Save history
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)


def train_mock(num_workers: int = 4, iterations: int = 50):
    """
    Mock training loop for testing (no real LLM calls).
    Uses a synthetic fitness function that rewards correct routing patterns.
    """
    print("=== Mock Training (no LLM calls) ===\n")
    
    config = CoordinatorConfig(
        backbone_model="Qwen/Qwen2.5-0.5B",
        num_workers=num_workers,
        hidden_dim=896,
    )
    
    coordinator = TrinityCoordinator(config)
    
    # Mock: just train the head with random hidden states
    n_params = coordinator.head.num_params
    print(f"  Head params: {n_params}")
    
    # Synthetic fitness: reward coordinating patterns
    # (e.g., prefer Thinker first, then Worker, then Verifier)
    def mock_fitness(params: np.ndarray) -> float:
        coordinator.head.set_flat_params(params)
        
        import torch
        total_score = 0.0
        
        for trial in range(10):
            # Simulate hidden states for different transcript stages
            h_early = torch.randn(1, config.hidden_dim) * 0.5   # Early in conversation
            h_mid = torch.randn(1, config.hidden_dim) * 1.0     # Middle
            h_late = torch.randn(1, config.hidden_dim) * 1.5    # Late
            
            m1, r1 = coordinator.head(h_early)
            m2, r2 = coordinator.head(h_mid)
            m3, r3 = coordinator.head(h_late)
            
            # Reward: Thinker early, Worker mid, Verifier late
            score = 0.0
            if r1.argmax().item() == 0:  # Thinker
                score += 1.0
            if r2.argmax().item() == 1:  # Worker
                score += 1.0
            if r3.argmax().item() == 2:  # Verifier
                score += 1.0
            
            # Reward diversity in model selection
            models_used = {m1.argmax().item(), m2.argmax().item(), m3.argmax().item()}
            score += len(models_used) * 0.5
            
            total_score += score
        
        return -total_score / 10.0  # Negate for minimization

    # Run CMA-ES
    x0 = coordinator.head.get_flat_params()
    opts = {
        "seed": 42,
        "popsize": 32,
        "maxiter": iterations,
        "CMA_diagonal": True,
        "verb_disp": 0,
    }
    
    es = cma.CMAEvolutionStrategy(x0, 0.5, opts)
    
    best_score = -float("inf")
    for i in range(iterations):
        solutions = es.ask()
        fitnesses = [mock_fitness(s) for s in solutions]
        es.tell(solutions, fitnesses)
        
        current_best = -min(fitnesses)
        if current_best > best_score:
            best_score = current_best
        
        if (i + 1) % 10 == 0:
            print(f"  [Iter {i+1:3d}] best_score={best_score:.3f} sigma={es.sigma:.4f}")
    
    print(f"\n  Final best score: {best_score:.3f}")
    print(f"  (Random baseline ≈ 1.5, optimal ≈ 4.5)")
    
    # Save
    best_params = solutions[np.argmin(fitnesses)]
    np.save("./coordinator_mock.npy", best_params)
    print(f"  Saved to ./coordinator_mock.npy")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run mock training (no LLMs)")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    if args.mock:
        train_mock(num_workers=args.workers, iterations=args.iterations)
    else:
        print("Full training requires Ollama workers. Use --mock for testing.")
        print("For full training, use the CMAESTrainer class programmatically.")
