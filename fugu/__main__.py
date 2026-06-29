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
  demo        Run a single orchestration demo
        """,
    )
    parser.add_argument("command", choices=["serve", "train", "train-mock", "demo"])
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--coordinator", type=str, default=None)
    parser.add_argument("--workers", type=str, default=None)
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=50)
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
        print("Full training requires:")
        print("  1. Ollama running with worker models pulled")
        print("  2. Coordinator backbone downloaded")
        print("")
        print("Use 'train-mock' to test the training loop without LLMs.")
        print("For full training, use the CMAESTrainer class directly.")

    elif args.command == "train-mock":
        from .train_cmaes import train_mock
        train_mock(iterations=args.iterations)

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
