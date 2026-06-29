"""
OpenAI-compatible serving endpoint for Fugu OSS-Only.

Exposes the orchestrator behind a standard /v1/chat/completions endpoint.
From the outside, it looks like one model. Inside, TRINITY coordinates
multiple OSS workers.

Usage:
    python -m fugu.serve --port 8088 --coordinator ./checkpoints/coordinator_final.npy
"""

import time
import uuid
import argparse
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .coordinator import TrinityCoordinator, CoordinatorConfig
from .worker_pool import WorkerPool, WorkerConfig
from .orchestrator import FuguOrchestrator


app = FastAPI(title="Fugu OSS-Only", version="0.1.0")

# Global state (initialized in main)
_orchestrator: Optional[FuguOrchestrator] = None


# --- Request/Response Models (OpenAI-compatible) ---

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "fugu-oss"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    orchestration_input_tokens: int = 0
    orchestration_output_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "fugu-oss"
    choices: List[Choice]
    usage: Usage


# --- Endpoints ---

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions endpoint."""
    global _orchestrator
    
    if _orchestrator is None:
        return ChatCompletionResponse(
            choices=[Choice(message=Message(role="assistant", content="Server not initialized"))],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    # Extract the user's query from messages
    user_messages = [m for m in request.messages if m.role == "user"]
    query = user_messages[-1].content if user_messages else ""

    # Run orchestration
    result = _orchestrator.run(query)

    # Compute token usage
    prompt_tokens = len(query.split())
    completion_tokens = len(result.answer.split())
    orchestration_tokens = result.total_tokens

    return ChatCompletionResponse(
        model="fugu-oss",
        choices=[
            Choice(
                message=Message(role="assistant", content=result.answer),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens + orchestration_tokens,
            orchestration_input_tokens=orchestration_tokens,
            orchestration_output_tokens=completion_tokens,
        ),
    )


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "fugu-oss",
                "object": "model",
                "owned_by": "fugu-oss-only",
                "created": int(time.time()),
            }
        ],
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "orchestrator_loaded": _orchestrator is not None}


def create_orchestrator(
    coordinator_path: Optional[str] = None,
    workers_csv: Optional[str] = None,
    backbone: str = "Qwen/Qwen2.5-0.5B",
    device: str = "cpu",
) -> FuguOrchestrator:
    """Initialize the full orchestration stack."""
    
    # Setup worker pool
    if workers_csv:
        model_ids = [m.strip() for m in workers_csv.split(",")]
        workers = [
            WorkerConfig(name=m.split(":")[0], model_id=m)
            for m in model_ids
        ]
        pool = WorkerPool(workers)
    else:
        pool = WorkerPool()  # Default pool

    # Setup coordinator
    config = CoordinatorConfig(
        backbone_model=backbone,
        num_workers=pool.num_workers,
    )
    coordinator = TrinityCoordinator(config)
    coordinator.load_backbone(device=device)

    # Load trained weights if available
    if coordinator_path:
        coordinator.load(coordinator_path)
        print(f"Loaded coordinator weights from {coordinator_path}")

    # Create orchestrator
    return FuguOrchestrator(
        coordinator=coordinator,
        worker_pool=pool,
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Fugu OSS-Only Server")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--coordinator", type=str, default=None,
                        help="Path to trained coordinator .npy file")
    parser.add_argument("--workers", type=str, default=None,
                        help="Comma-separated Ollama model IDs")
    parser.add_argument("--backbone", default="Qwen/Qwen2.5-0.5B",
                        help="Coordinator backbone model")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    global _orchestrator
    _orchestrator = create_orchestrator(
        coordinator_path=args.coordinator,
        workers_csv=args.workers,
        backbone=args.backbone,
        device=args.device,
    )

    print(f"\nFugu OSS-Only server starting on {args.host}:{args.port}")
    print(f"  Endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  Workers: {_orchestrator.pool.model_names}")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
