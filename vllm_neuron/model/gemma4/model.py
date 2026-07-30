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


class Gemma4PagedKVCache:
    """Explicit paged KV storage shared by local/global attention layers.

    The cache is deliberately a small tensor contract: callers provide
    `slot_mapping` and the layer's native head width; no layer may reshape a
    global cache into the local layout. The Neuron implementation can replace
    the backing tensors with device handles without changing this interface.
    """

    def __init__(self, num_slots: int, num_kv_heads: int, head_dim: int, dtype=torch.bfloat16):
        self.key = torch.zeros(num_slots, num_kv_heads, head_dim, dtype=dtype)
        self.value = torch.zeros_like(self.key)

    @property
    def shape(self):
        return self.key.shape

    def write(self, slot_mapping: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        slots = slot_mapping.to(device="cpu", dtype=torch.long).flatten()
        if key.shape != value.shape:
            raise ValueError(f"key/value shape mismatch: {key.shape} vs {value.shape}")
        if key.ndim != 3 or key.shape[0] != slots.numel():
            raise ValueError(
                f"cache write expects [num_slots, heads, head_dim], got {key.shape} "
                f"for {slots.numel()} mapped slots"
            )
        if slots.numel() and (slots.min() < 0 or slots.max() >= self.key.shape[0]):
            raise IndexError(f"slot_mapping exceeds cache size {self.key.shape[0]}")
        self.key.index_copy_(0, slots, key.to(self.key.dtype))
        self.value.index_copy_(0, slots, value.to(self.value.dtype))

    def read(self, slot_mapping: torch.Tensor):
        slots = slot_mapping.to(device="cpu", dtype=torch.long).flatten()
        if slots.numel() and (slots.min() < 0 or slots.max() >= self.key.shape[0]):
            raise IndexError(f"slot_mapping exceeds cache size {self.key.shape[0]}")
        return self.key.index_select(0, slots), self.value.index_select(0, slots)


class Gemma4ReferenceAttention(nn.Module):
    """Small CPU oracle for validating native attention/cache seams.

    This is not the serving kernel. It intentionally uses ordinary PyTorch
    operations so discrepancies can be localized before replacing it with a
    Neuron paged-attention implementation.
    """

    def __init__(self, head_dim: int, num_query_heads: int, num_kv_heads: int):
        super().__init__()
        if num_query_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        self.head_dim = head_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.scale = head_dim ** -0.5

    def forward(self, query, key, value, cache=None, slot_mapping=None):
        # Inputs are [tokens, heads, head_dim]. Cache writes preserve the
        # native per-layer head width and are performed before reading history.
        if cache is not None:
            if slot_mapping is None:
                raise ValueError("slot_mapping is required when using a KV cache")
            cache.write(slot_mapping, key, value)
            key, value = cache.read(slot_mapping)
        repeat = self.num_query_heads // self.num_kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        scores = torch.einsum("thd,shd->hts", query, key) * self.scale
        causal = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        return torch.einsum("hts,shd->thd", probs, value)


class Gemma4MoeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        raise NotImplementedError(
            "Gemma4 native vLLM-Neuron layers are not implemented yet; "
            "use the committed serving baseline while this port is developed."
        )
