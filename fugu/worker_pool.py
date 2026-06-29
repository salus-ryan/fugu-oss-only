"""
Worker Pool — OSS-only model backends via Ollama.

Each worker is an open-source model running locally. The coordinator
selects which worker to dispatch at each turn and assigns it a role.

Supported backends:
- Ollama (primary — local inference)
- litellm (fallback — can route to any OpenAI-compatible endpoint)
"""

from dataclasses import dataclass
from typing import List, Optional

import ollama

from .coordinator import Role


# Role-specific system prompts (injected into each worker call)
ROLE_PROMPTS = {
    Role.THINKER: (
        "You are acting as a Thinker. Your job is to analyze the problem and provide "
        "strategic guidance: high-level plans, decompositions, or critiques. "
        "Do NOT produce a final answer. Instead, outline the approach and key steps."
    ),
    Role.WORKER: (
        "You are acting as a Worker. Your job is to execute the task and produce "
        "concrete, actionable output: code, derivations, calculations, or specific answers. "
        "Be precise and complete."
    ),
    Role.VERIFIER: (
        "You are acting as a Verifier. Your job is to evaluate whether the current "
        "solution is correct, complete, and responsive to the original question. "
        "Respond with ACCEPT if the solution is satisfactory, or REVISE followed by "
        "a specific diagnosis of what needs fixing."
    ),
}


@dataclass
class WorkerConfig:
    """Configuration for a single worker model."""
    name: str                    # Human-readable name
    model_id: str               # Ollama model ID (e.g., "llama3.1:8b")
    backend: str = "ollama"     # "ollama" or "litellm"
    strengths: List[str] = None # e.g., ["code", "math", "reasoning"]
    context_window: int = 4096
    
    def __post_init__(self):
        if self.strengths is None:
            self.strengths = []


@dataclass
class WorkerResponse:
    """Response from a worker model."""
    content: str
    model_id: str
    role: Role
    tokens_used: int
    accepted: Optional[bool] = None  # Only for Verifier role
    diagnosis: Optional[str] = None  # Only for Verifier role with REVISE


# Default OSS worker pool — all freely available via Ollama
DEFAULT_WORKERS = [
    WorkerConfig(
        name="Llama-3.1-8B",
        model_id="llama3.1:8b",
        strengths=["general", "reasoning", "instruction-following"],
    ),
    WorkerConfig(
        name="Qwen2.5-7B",
        model_id="qwen2.5:7b",
        strengths=["code", "math", "multilingual"],
    ),
    WorkerConfig(
        name="Mistral-7B",
        model_id="mistral:7b",
        strengths=["general", "creative", "fast"],
    ),
    WorkerConfig(
        name="DeepSeek-Coder-V2",
        model_id="deepseek-coder-v2:16b",
        strengths=["code", "debugging", "technical"],
    ),
]


class WorkerPool:
    """Manages the pool of OSS worker models."""

    def __init__(self, workers: Optional[List[WorkerConfig]] = None):
        self.workers = workers or DEFAULT_WORKERS
        self._available: Optional[List[bool]] = None

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    @property
    def model_names(self) -> List[str]:
        return [w.name for w in self.workers]

    def check_availability(self) -> List[bool]:
        """Check which models are actually pulled in Ollama."""
        try:
            models_response = ollama.list()
            available_models = {m.model for m in models_response.models}
        except Exception:
            available_models = set()

        self._available = []
        for worker in self.workers:
            # Ollama model names may include :latest suffix
            is_available = (
                worker.model_id in available_models
                or f"{worker.model_id}:latest" in available_models
            )
            self._available.append(is_available)

        return self._available

    def dispatch(
        self,
        worker_index: int,
        role: Role,
        query: str,
        transcript: str,
        temperature: float = 0.7,
    ) -> WorkerResponse:
        """
        Dispatch a query to a specific worker with a role assignment.
        
        Args:
            worker_index: Index into self.workers
            role: Thinker/Worker/Verifier
            query: Original user query
            transcript: Full conversation so far
            temperature: Sampling temperature
        
        Returns:
            WorkerResponse with the model's output
        """
        worker = self.workers[worker_index]
        role_prompt = ROLE_PROMPTS[role]

        # Construct messages
        messages = [
            {"role": "system", "content": role_prompt},
        ]

        # Include transcript context
        if transcript.strip():
            messages.append({
                "role": "user",
                "content": f"Original question: {query}\n\nConversation so far:\n{transcript}\n\nNow fulfill your role."
            })
        else:
            messages.append({
                "role": "user",
                "content": query,
            })

        # Call Ollama
        try:
            response = ollama.chat(
                model=worker.model_id,
                messages=messages,
                options={"temperature": temperature},
            )
            content = response["message"]["content"]
            tokens = response.get("eval_count", len(content.split()))
        except Exception as e:
            content = f"[Worker {worker.name} error: {str(e)}]"
            tokens = 0

        # Parse verifier response
        accepted = None
        diagnosis = None
        if role == Role.VERIFIER:
            content_upper = content.upper()
            if "ACCEPT" in content_upper:
                accepted = True
            elif "REVISE" in content_upper:
                accepted = False
                # Extract diagnosis (everything after REVISE)
                revise_idx = content_upper.find("REVISE")
                diagnosis = content[revise_idx + 6:].strip(": \n")

        return WorkerResponse(
            content=content,
            model_id=worker.model_id,
            role=role,
            tokens_used=tokens,
            accepted=accepted,
            diagnosis=diagnosis,
        )

    def get_worker_description(self) -> str:
        """Get a formatted description of the worker pool (for coordinator context)."""
        lines = ["Available workers:"]
        for i, w in enumerate(self.workers):
            strengths = ", ".join(w.strengths) if w.strengths else "general"
            lines.append(f"  [{i}] {w.name} ({w.model_id}) — strengths: {strengths}")
        return "\n".join(lines)
