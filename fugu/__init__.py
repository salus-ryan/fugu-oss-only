"""
Fugu OSS-Only — Open-source multi-model orchestrator.

Reimplements Sakana AI's Fugu architecture using ONLY open-source/open-weight models:
- TRINITY coordinator: tiny SLM (~0.6B) + linear head (~19.5K params)
- sep-CMA-ES evolutionary training (no gradients needed)
- Tri-role orchestration: Thinker / Worker / Verifier
- All workers are local OSS models via Ollama
"""

__version__ = "0.1.0"
