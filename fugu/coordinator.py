"""
TRINITY Coordinator — the core of Fugu.

Architecture (from Sakana's ICLR 2026 paper):
- A pre-trained SLM backbone (Qwen3-0.6B) produces hidden states
- A tiny linear head (~19.5K params) maps hidden state -> (model selection, role assignment)
- Singular Value Fine-tuning (SVF) on ~9 backbone matrices for representation adaptation
- Total trainable params: <20K

The coordinator NEVER answers the user directly. It only:
1. Reads the transcript
2. Selects a worker model from the pool
3. Assigns a role (Thinker/Worker/Verifier)

Trained via sep-CMA-ES (evolutionary, gradient-free).
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class Role(Enum):
    THINKER = 0   # Strategize, plan, decompose
    WORKER = 1    # Execute, produce concrete output
    VERIFIER = 2  # Evaluate, accept/revise


@dataclass
class CoordinatorDecision:
    """Output of the coordinator at each turn."""
    model_index: int        # Which worker model to dispatch
    role: Role              # What role to assign
    model_logits: np.ndarray  # Raw logits over models (for debugging)
    role_logits: np.ndarray   # Raw logits over roles


@dataclass
class CoordinatorConfig:
    """Configuration for the TRINITY coordinator."""
    backbone_model: str = "Qwen/Qwen2.5-0.5B"  # OSS backbone (~0.6B params)
    hidden_dim: int = 896       # Hidden dim of Qwen2.5-0.5B
    num_workers: int = 4        # Number of models in the worker pool
    num_roles: int = 3          # Thinker, Worker, Verifier
    svf_matrices: int = 9       # Number of backbone matrices to SVF-adapt
    svf_rank: int = 16          # SVF rank for each adapted matrix
    max_turns: int = 8          # Max coordination turns per query
    use_penultimate_token: bool = True  # Use hidden state at penultimate token


class CoordinationHead(nn.Module):
    """
    Lightweight bias-free linear head.
    Maps hidden state h ∈ R^d -> (L + 3) logits:
      - L logits for selecting a worker model
      - 3 logits for assigning a role (T/W/V)
    
    ~19.5K params when d=896, L=4 → 896 * 7 = 6272 params from head alone.
    The rest comes from SVF singular value scales.
    """

    def __init__(self, hidden_dim: int, num_workers: int, num_roles: int = 3):
        super().__init__()
        self.num_workers = num_workers
        self.num_roles = num_roles
        output_dim = num_workers + num_roles
        
        # Bias-free linear projection (as per paper)
        self.proj = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: [batch, hidden_dim] — last hidden state from backbone
        Returns:
            model_logits: [batch, num_workers]
            role_logits: [batch, num_roles]
        """
        logits = self.proj(hidden_state)
        model_logits = logits[:, :self.num_workers]
        role_logits = logits[:, self.num_workers:]
        return model_logits, role_logits

    def get_flat_params(self) -> np.ndarray:
        """Extract all parameters as a flat numpy array (for CMA-ES)."""
        return self.proj.weight.detach().cpu().numpy().flatten()

    def set_flat_params(self, params: np.ndarray):
        """Set parameters from a flat numpy array (from CMA-ES)."""
        weight_shape = self.proj.weight.shape
        weight = torch.tensor(params.reshape(weight_shape), dtype=self.proj.weight.dtype)
        self.proj.weight.data = weight.to(self.proj.weight.device)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SVFAdapter:
    """
    Singular Value Fine-tuning adapter.
    
    For selected backbone weight matrices:
    1. Compute SVD: W = U @ diag(S) @ V^T
    2. Keep U, V^T frozen
    3. Only learn scale factors on S: S_new = S * (1 + delta)
    
    This gives a few hundred extra trainable params per matrix.
    """

    def __init__(self, model: nn.Module, config: CoordinatorConfig):
        self.config = config
        self.adaptations: List[dict] = []
        self._setup_svf(model)

    def _setup_svf(self, model: nn.Module):
        """Find target matrices and decompose them."""
        target_names = self._select_target_matrices(model)
        
        for name in target_names[:self.config.svf_matrices]:
            _modules = dict(model.named_modules())
            # Navigate to the parameter
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, param_name = parts
                parent = dict(model.named_modules()).get(parent_name)
                if parent is None:
                    continue
                param = getattr(parent, param_name, None)
                if param is None or not isinstance(param, nn.Parameter):
                    continue
            else:
                continue

            # SVD decomposition
            W = param.data.float()
            if W.dim() != 2:
                continue
                
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            rank = min(self.config.svf_rank, S.shape[0])
            
            self.adaptations.append({
                "name": name,
                "U": U[:, :rank],
                "S_original": S[:rank],
                "Vh": Vh[:rank, :],
                "delta": np.zeros(rank),  # Trainable: scale perturbations
                "parent_name": parts[0] if len(parts) == 2 else "",
                "param_name": parts[1] if len(parts) == 2 else name,
            })

    def _select_target_matrices(self, model: nn.Module) -> List[str]:
        """Select which matrices to adapt (q_proj, k_proj, v_proj, etc.)."""
        targets = []
        for name, param in model.named_parameters():
            if any(key in name for key in ["q_proj", "k_proj", "v_proj"]):
                if param.dim() == 2:
                    targets.append(name)
        return targets

    def get_flat_params(self) -> np.ndarray:
        """Get all SVF delta scales as flat array."""
        if not self.adaptations:
            return np.array([])
        return np.concatenate([a["delta"] for a in self.adaptations])

    def set_flat_params(self, params: np.ndarray):
        """Set SVF deltas from flat array."""
        offset = 0
        for adaptation in self.adaptations:
            rank = len(adaptation["delta"])
            adaptation["delta"] = params[offset:offset + rank]
            offset += rank

    def apply_to_model(self, model: nn.Module):
        """Apply current SVF adaptations to the model weights."""
        for adaptation in self.adaptations:
            U = adaptation["U"]
            S = adaptation["S_original"]
            Vh = adaptation["Vh"]
            delta = torch.tensor(adaptation["delta"], dtype=S.dtype)
            
            # W_new = U @ diag(S * (1 + delta)) @ Vh
            S_new = S * (1.0 + delta)
            W_new = U @ torch.diag(S_new) @ Vh
            
            # Write back to model
            parent = dict(model.named_modules()).get(adaptation["parent_name"])
            if parent is not None:
                param = getattr(parent, adaptation["param_name"], None)
                if param is not None:
                    param.data = W_new.to(param.dtype).to(param.device)

    @property
    def num_params(self) -> int:
        return sum(len(a["delta"]) for a in self.adaptations)


class TrinityCoordinator:
    """
    Full TRINITY coordinator: backbone + SVF + head.
    
    Usage:
        coordinator = TrinityCoordinator(config)
        coordinator.load_backbone()
        decision = coordinator.decide(transcript_tokens)
    """

    def __init__(self, config: Optional[CoordinatorConfig] = None):
        self.config = config or CoordinatorConfig()
        self.backbone = None
        self.tokenizer = None
        self.head = CoordinationHead(
            self.config.hidden_dim,
            self.config.num_workers,
            self.config.num_roles,
        )
        self.svf = None
        self._device = "cpu"

    def load_backbone(self, device: str = "cpu"):
        """Load the SLM backbone (frozen, except SVF-adapted matrices)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self._device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_model)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            self.config.backbone_model,
            torch_dtype=torch.float32,
            device_map=device,
        )
        
        # Freeze all backbone params
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Setup SVF on selected matrices
        self.svf = SVFAdapter(self.backbone, self.config)
        
        # Move head to device
        self.head = self.head.to(device)
        
        print(f"Coordinator loaded: backbone={self.config.backbone_model}")
        print(f"  Head params: {self.head.num_params}")
        print(f"  SVF params: {self.svf.num_params}")
        print(f"  Total trainable: {self.head.num_params + self.svf.num_params}")

    def get_flat_params(self) -> np.ndarray:
        """Get all trainable params (head + SVF) as a flat vector for CMA-ES."""
        head_params = self.head.get_flat_params()
        svf_params = self.svf.get_flat_params() if self.svf else np.array([])
        return np.concatenate([head_params, svf_params])

    def set_flat_params(self, params: np.ndarray):
        """Set all trainable params from a flat vector (from CMA-ES)."""
        head_size = self.head.num_params
        self.head.set_flat_params(params[:head_size])
        if self.svf and len(params) > head_size:
            self.svf.set_flat_params(params[head_size:])
            self.svf.apply_to_model(self.backbone)

    @property
    def total_trainable_params(self) -> int:
        svf_count = self.svf.num_params if self.svf else 0
        return self.head.num_params + svf_count

    @torch.no_grad()
    def get_hidden_state(self, text: str) -> torch.Tensor:
        """Run text through backbone and extract the penultimate token's hidden state."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        outputs = self.backbone(**inputs, output_hidden_states=True)
        
        # Get last hidden state
        last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
        
        if self.config.use_penultimate_token and last_hidden.shape[1] > 1:
            # Use penultimate token (as per paper — allows faster inference)
            h = last_hidden[:, -2, :]
        else:
            h = last_hidden[:, -1, :]
        
        return h  # [1, hidden_dim]

    @torch.no_grad()
    def decide(self, transcript: str) -> CoordinatorDecision:
        """
        Given the current conversation transcript, decide which model + role to dispatch.
        
        Args:
            transcript: Full conversation so far (Q, O1, O2, ... Ok-1)
        
        Returns:
            CoordinatorDecision with model_index and role
        """
        h = self.get_hidden_state(transcript)
        model_logits, role_logits = self.head(h)
        
        # Greedy selection
        model_idx = model_logits.argmax(dim=-1).item()
        role_idx = role_logits.argmax(dim=-1).item()
        
        return CoordinatorDecision(
            model_index=model_idx,
            role=Role(role_idx),
            model_logits=model_logits.cpu().numpy().flatten(),
            role_logits=role_logits.cpu().numpy().flatten(),
        )

    def save(self, path: str):
        """Save trainable parameters (head weights + SVF deltas)."""
        params = self.get_flat_params()
        np.save(path, params)
        print(f"Saved {len(params)} parameters to {path}")

    def load(self, path: str):
        """Load trainable parameters."""
        params = np.load(path)
        self.set_flat_params(params)
        print(f"Loaded {len(params)} parameters from {path}")
