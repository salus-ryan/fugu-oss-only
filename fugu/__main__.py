"""Entry point: python -m fugu"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Fugu OSS-Only — Multi-model orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  serve       Start the OpenAI-compatible API server
  train       Train the TRINITY coordinator via sep-CMA-ES
  train-mock  Run mock training (no LLM calls needed)
  train-grpo  Train coordinator head via GRPO/RLOO policy gradient
  train-dpo   Train coordinator head via DPO preference optimization
  train-score Train coordinator via SCoRe self-correction loop
  demo        Run a single orchestration demo
        """,
    )
    parser.add_argument(
        "command",
        choices=["serve", "train", "train-mock", "train-grpo", "train-dpo", "train-score", "demo"],
    )
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--coordinator", type=str, default=None)
    parser.add_argument("--workers", type=str, default=None)
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--method", type=str, default="grpo", help="grpo|rloo|dpo|simpo|kto")
    parser.add_argument("--group-size", type=int, default=8, help="GRPO group size G")
    parser.add_argument("--query", type=str, default="What is the capital of France?")
    args = parser.parse_args()

    if args.command == "serve":
        from .serve import main as serve_main
        sys.argv = [sys.argv[0], "--port", str(args.port)]
        if args.coordinator:
            sys.argv.extend(["--coordinator", args.coordinator])
        if args.workers:
            sys.argv.extend(["--workers", args.workers])
        sys.argv.extend(["--backbone", args.backbone, "--device", args.device])
        serve_main()

    elif args.command == "train":
        print("Full CMA-ES training requires:")
        print("  1. Ollama running with worker models pulled")
        print("  2. Coordinator backbone downloaded")
        print("")
        print("Use 'train-mock' to test the training loop without LLMs.")
        print("For full training, use the CMAESTrainer class directly.")

    elif args.command == "train-mock":
        from .train_cmaes import train_mock
        train_mock(iterations=args.iterations)

    elif args.command == "train-grpo":
        from .coordinator import TrinityCoordinator, CoordinatorConfig
        from .worker_pool import WorkerPool
        from .train_grpo import GRPOTrainer, GRPOConfig

        config = CoordinatorConfig(backbone_model=args.backbone)
        coordinator = TrinityCoordinator(config)
        coordinator.load_backbone(device=args.device)
        if args.coordinator:
            coordinator.load(args.coordinator)

        pool = WorkerPool()
        grpo_config = GRPOConfig(
            group_size=args.group_size,
            method=args.method if args.method in ("grpo", "rloo") else "grpo",
        )
        trainer = GRPOTrainer(coordinator, pool, grpo_config)

        # Example training queries (user should supply their own dataset)
        queries = [
            ("What is the capital of France?", "Paris"),
            ("Write a Python function to reverse a string.", "def reverse"),
            ("Explain quantum entanglement in simple terms.", None),
            ("What is 2 + 2?", "4"),
        ]
        trainer.train_on_queries(queries, num_epochs=args.epochs)

    elif args.command == "train-dpo":
        from .coordinator import TrinityCoordinator, CoordinatorConfig
        from .worker_pool import WorkerPool
        from .orchestrator import FuguOrchestrator
        from .train_dpo import DPOTrainer, DPOConfig, TrajectoryStore

        config = CoordinatorConfig(backbone_model=args.backbone)
        coordinator = TrinityCoordinator(config)
        coordinator.load_backbone(device=args.device)
        if args.coordinator:
            coordinator.load(args.coordinator)

        pool = WorkerPool()
        store = TrajectoryStore()

        # Collect trajectories
        print("Collecting routing trajectories for DPO...")
        orchestrator = FuguOrchestrator(coordinator=coordinator, worker_pool=pool, verbose=False)
        queries = [
            ("What is the capital of France?", "Paris"),
            ("Write a Python function to reverse a string.", "def reverse"),
            ("Explain quantum entanglement in simple terms.", None),
        ]
        for query, gt in queries:
            for _ in range(args.group_size):
                result, reward = orchestrator.run_with_reward(query, ground_truth=gt)
                store.add(query, result, reward)

        pairs = store.generate_pairs()
        print(f"  Generated {len(pairs)} preference pairs")

        dpo_config = DPOConfig(
            method=args.method if args.method in ("dpo", "simpo", "kto") else "dpo",
            num_epochs=args.epochs,
        )
        trainer = DPOTrainer(coordinator, dpo_config)
        trainer.train(pairs)

    elif args.command == "train-score":
        from .coordinator import TrinityCoordinator, CoordinatorConfig
        from .worker_pool import WorkerPool
        from .self_correction import SelfCorrectionTrainer, SCoReConfig

        config = CoordinatorConfig(backbone_model=args.backbone)
        coordinator = TrinityCoordinator(config)
        coordinator.load_backbone(device=args.device)
        if args.coordinator:
            coordinator.load(args.coordinator)

        pool = WorkerPool()
        score_config = SCoReConfig(
            stage1_epochs=args.epochs // 2,
            stage2_epochs=args.epochs,
        )
        trainer = SelfCorrectionTrainer(coordinator, pool, score_config)

        queries = [
            ("What is the capital of France?", "Paris"),
            ("Write a Python function to reverse a string.", "def reverse"),
            ("Explain quantum entanglement in simple terms.", None),
            ("What is 2 + 2?", "4"),
        ]
        trainer.train(queries, collect_episodes=args.iterations)

    elif args.command == "demo":
        from .orchestrator import FuguOrchestrator
        from .coordinator import TrinityCoordinator, CoordinatorConfig
        from .worker_pool import WorkerPool

        config = CoordinatorConfig(backbone_model=args.backbone)
        coordinator = TrinityCoordinator(config)
        coordinator.load_backbone(device=args.device)
        
        if args.coordinator:
            coordinator.load(args.coordinator)

        pool = WorkerPool()
        orchestrator = FuguOrchestrator(coordinator=coordinator, worker_pool=pool)
        
        print(f"\nQuery: {args.query}")
        print("=" * 60)
        result = orchestrator.run(args.query)
        print("=" * 60)
        print(f"\nFinal answer ({result.num_turns} turns, accepted={result.accepted}):")
        print(result.answer[:500])


if __name__ == "__main__":
    main()
