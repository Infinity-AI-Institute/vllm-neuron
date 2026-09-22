# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 full-attention math that differs from Qwen3
===================================================

Pure torch, no vLLM imports, tested on CPU against transformers 5.17.0.

Three places where reusing the Qwen3 layer would give plausible but wrong
numbers:

1. **Zero-centred RMSNorm.** ``Qwen3_5RMSNorm`` stores ``w`` initialised to
   zero and applies ``x * (1 + w)`` in F32 before casting back. Qwen3's norm
   applies ``w`` directly. This form is used for the decoder norms, the final
   norm, and the per-head ``q_norm`` / ``k_norm``. (The Gated DeltaNet's own
   ``RMSNormGated`` is *not* zero-centred; see ``gated_deltanet.py``.)
2. **Partial rotary.** Only the first ``rotary_dim`` (64 of 256) channels of
   each head rotate; the rest pass through. For text, Qwen3.5's M-RoPE sees
   three identical position axes, so it reduces to standard RoPE over those
   64 channels.
3. **Gated output.** ``q_proj`` emits ``2 * head_dim`` per head, laid out as
   ``[query | gate]``, and the attention output is multiplied by
   ``sigmoid(gate)`` *before* ``o_proj``.
"""

import torch


def zero_centred_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """``Qwen3_5RMSNorm``: ``(x * rsqrt(mean(x^2) + eps)) * (1 + w)`` in F32."""
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return (h * (1.0 + weight.float())).type_as(x)


def rotary_cos_sin(
    positions: torch.Tensor, rotary_dim: int, theta: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[T, rotary_dim]`` cos/sin for standard RoPE over ``rotary_dim`` channels."""
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=positions.device)
            / rotary_dim
        )
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_partial_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate the first ``cos.shape[-1]`` channels (rotate_half); pass the rest.

    ``x`` is ``[..., T, head_dim]`` and ``cos`` / ``sin`` are ``[T, rotary_dim]``.
    """
    rotary_dim = cos.shape[-1]
    rot, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    rotated = torch.cat((-rot[..., half:], rot[..., :half]), dim=-1)
    return torch.cat((rot * cos + rotated * sin, passthrough), dim=-1)


def split_query_gate(
    q_proj_out: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[T, num_heads * 2 * head_dim]`` to ``query [T, H, Dh]`` and flat ``gate [T, H * Dh]``.

    Each head's slice is ``[query | gate]``, so sharding ``q_proj`` rows by
    whole heads keeps every query with its own gate.
    """
    tokens = q_proj_out.shape[0]
    pair = q_proj_out.view(tokens, num_heads, 2 * head_dim)
    query, gate = torch.chunk(pair, 2, dim=-1)
    return query, gate.reshape(tokens, num_heads * head_dim)


def gate_output(attention: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Multiply the flat ``[T, H * Dh]`` attention output by ``sigmoid(gate)``."""
    return attention * torch.sigmoid(gate)
