# Fugu OSS-Only

**An open-source reimplementation of Sakana AI's Fugu multi-model orchestrator, constrained to use ONLY open-source/open-weight models.**

Fugu is not a model — it's a *policy over models*. A tiny coordinator (~0.6B backbone + ~19.5K trainable params) decides which worker LLM to dispatch and what role to assign at each turn. The workers do the actual thinking. The coordinator just routes.

## Architecture

Based on Sakana AI's ICLR 2026 papers:
- **[TRINITY](https://arxiv.org/abs/2512.04695)** — Evolved LLM Coordinator (sep-CMA-ES)
- **[Conductor](https://arxiv.org/abs/2512.04388)** — RL-trained orchestrator for DAG workflows
- **[Fugu Technical Report](https://arxiv.org/abs/2606.21228)** — Production system combining both

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              TRINITY COORDINATOR (~0.6B backbone)                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Qwen2.5-0.5B (frozen) + SVF (~9 matrices, ~144 params)  │   │
│  │                    ▼                                      │   │
│  │ Linear Head (bias-free, ~6K params)                       │   │
│  │   → Model logits [L]  +  Role logits [3]                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│  Total trainable: <20K params | Trained via sep-CMA-ES          │
└─────────────┬─────────────────────────────────┬─────────────────┘
              │ selects model + role             │
              ▼                                 ▼
┌─────────────────────────┐    ┌────────────────────────────────┐
│   WORKER POOL (Ollama)  │    │        TRI-ROLE SYSTEM          │
│   ┌───────────────────┐ │    │                                │
│   │ Llama-3.1-8B      │ │    │  🧠 Thinker — strategize       │
│   │ Qwen2.5-7B        │ │    │  ⚒️  Worker  — execute          │
│   │ Mistral-7B        │ │    │  ✓  Verifier — accept/revise   │
│   │ DeepSeek-Coder    │ │    │                                │
│   └───────────────────┘ │    │  Loop until ACCEPT or max_turns│
└─────────────────────────┘    └────────────────────────────────┘
```

## Key Differences from Sakana's Fugu

| Aspect | Sakana Fugu | This Repo |
|--------|-------------|-----------|
| Worker models | Gemini, Claude, GPT-4 (closed APIs) | Ollama local models only |
| Coordinator backbone | Undisclosed | Qwen2.5-0.5B (Apache 2.0) |
| Training | Proprietary | sep-CMA-ES (reproduced from paper) |
| Inference | Cloud API ($) | 100% local, free |
| Conductor weights | Closed | Train your own (or use OpenFugu's) |
| License | Proprietary | Apache 2.0 |

## Quickstart

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed
- At least 2 models pulled (e.g., `ollama pull llama3.1:8b && ollama pull qwen2.5:7b`)

### Install

```bash
pip install -r requirements.txt
```

### Mock Training (no LLMs needed)

Test the sep-CMA-ES training loop with a synthetic fitness function:

```bash
python -m fugu train-mock --iterations 100
```

### Full Training

```bash
# Pull worker models
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b

# Train coordinator (requires Ollama running)
python -m fugu.train_cmaes
```

### Serve

```bash
# Start OpenAI-compatible server
python -m fugu serve --port 8088 --coordinator ./checkpoints/coordinator_final.npy

# Query it like any OpenAI API
curl localhost:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a Python function to merge two sorted lists"}]}'
```

### Demo

```bash
python -m fugu demo --query "What is 15 * 7?" --backbone Qwen/Qwen2.5-0.5B
```

## How It Works

### 1. Coordinator Decision (per turn)

The backbone processes the conversation transcript and produces a hidden state `h ∈ R^896`. A bias-free linear head projects this to `(L+3)` logits:
- **L logits** → softmax → select worker model
- **3 logits** → softmax → select role (Thinker/Worker/Verifier)

### 2. Tri-Role Orchestration

Each turn, the selected worker receives a role-specific system prompt:
- **Thinker**: "Analyze and plan, don't produce final answer"
- **Worker**: "Execute and produce concrete output"  
- **Verifier**: "Is this correct? ACCEPT or REVISE"

The loop terminates when a Verifier says ACCEPT, or max_turns is reached.

### 3. sep-CMA-ES Training

Why evolutionary, not gradient-based?
- The head has **block-separable** structure (~19.5K params)
- Each evaluation requires a **full multi-turn LLM rollout** (expensive, non-differentiable)
- Reward is **binary** (task solved or not)
- sep-CMA-ES improvement grows **linearly** with iterations (vs log for random search)
- Only ~1.5K-40K evaluations needed

### 4. Singular Value Fine-tuning (SVF)

On 9 selected backbone matrices:
```
W = U @ diag(S) @ V^T
W_new = U @ diag(S * (1 + delta)) @ V^T
```
Only `delta` (a few hundred params per matrix) is trainable. This adapts the backbone's representation space without full fine-tuning.

## Project Structure

```
fugu/
├── __init__.py         # Package metadata
├── __main__.py         # CLI entry point
├── coordinator.py      # TRINITY coordinator (backbone + head + SVF)
├── worker_pool.py      # Ollama-backed worker models
├── orchestrator.py     # Multi-turn T/W/V loop
├── train_cmaes.py      # sep-CMA-ES training
├── serve.py            # OpenAI-compatible FastAPI server
requirements.txt
README.md
```

## References

- Xu et al. (2026). *TRINITY: An Evolved LLM Coordinator.* ICLR 2026. [arXiv:2512.04695](https://arxiv.org/abs/2512.04695)
- Nielsen et al. (2026). *Learning to Orchestrate Agents in Natural Language with the Conductor.* ICLR 2026. [arXiv:2512.04388](https://arxiv.org/abs/2512.04388)
- Fugu Team, Sakana AI (2026). *Sakana Fugu Technical Report.* [arXiv:2606.21228](https://arxiv.org/abs/2606.21228)
- Akiba et al. (2024). *Evolutionary Optimization of Model Merging Recipes.* [arXiv:2403.13187](https://arxiv.org/abs/2403.13187)
- [OpenFugu](https://github.com/trotsky1997/OpenFugu) — prior open reimplementation (for comparison)

## License

Apache 2.0. No Sakana AI code or weights are included or redistributed.
