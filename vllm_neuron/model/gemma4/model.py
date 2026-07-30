"""Native Gemma 4 model implementation scaffold.

This module is the path-2 porting seam. It must own vLLM's paged KV-cache
writes and sampling-position contract; it must not route through NxDI model
registries or architecture-rewrite shims.
"""

import math

import torch
import torch.nn as nn


class Gemma4RMSNorm(nn.Module):
    """Gemma RMSNorm with the checkpoint's +1 weight convention."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * (1.0 + self.weight.float())).to(input_dtype)


class Gemma4ValueNorm(nn.Module):
    """RMS normalization used on attention values in Gemma 4."""

    def __init__(self, head_dim: int, eps: float = 1e-6, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim, dtype=dtype))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        x = values.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(values.dtype)


class Gemma4RotaryEmbedding(nn.Module):
    """Generate local/global rotary factors without owning KV-cache state."""

    def __init__(self, head_dim: int, theta: float, rotary_dim: int | None = None):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim or head_dim
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype=torch.bfloat16):
        positions = position_ids.float().reshape(-1)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class Gemma4MoeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        raise NotImplementedError(
            "Gemma4 native vLLM-Neuron layers are not implemented yet; "
            "use the committed serving baseline while this port is developed."
        )
