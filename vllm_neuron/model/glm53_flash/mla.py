# SPDX-License-Identifier: Apache-2.0
"""All-NoPE MLA projections for GLM-5.3-Flash."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import Glm53FlashInferenceConfig, validate_fp8_scale


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    value = x.to(torch.float32)
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    return (value * weight.to(torch.float32)).to(x.dtype)


@dataclass(frozen=True)
class Glm53MlaProjection:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


class Glm53NopeMla(nn.Module):
    """Expanded MLA with a deliberately absent rotary/position interface."""

    def __init__(self, config: Glm53FlashInferenceConfig) -> None:
        super().__init__()
        if config.qk_rope_head_dim != 0:
            raise ValueError("Glm53NopeMla cannot be constructed with RoPE dims")
        self.config = config
        self.q_a_proj = nn.Linear(
            config.hidden_size,
            config.q_lora_rank,
            bias=False,
            dtype=config.torch_dtype,
        )
        self.q_a_norm = nn.Parameter(
            torch.ones(config.q_lora_rank, dtype=config.torch_dtype)
        )
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            config.num_attention_heads * config.qk_head_dim,
            bias=False,
            dtype=config.torch_dtype,
        )
        self.kv_a_proj = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank,
            bias=False,
            dtype=config.torch_dtype,
        )
        self.kv_a_norm = nn.Parameter(
            torch.ones(config.kv_lora_rank, dtype=config.torch_dtype)
        )
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
            dtype=config.torch_dtype,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.v_head_dim,
            config.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
        )
        self.register_buffer(
            "weight_scale",
            torch.tensor(config.fp8_weight_scale_default, dtype=torch.float32),
        )
        self.register_buffer(
            "activation_scale",
            torch.tensor(config.fp8_activation_scale_default, dtype=torch.float32),
        )
        self.register_buffer(
            "key_cache_quant_multiplier",
            torch.tensor(config.key_cache_quant_multiplier, dtype=torch.float32),
        )
        self.register_buffer(
            "value_cache_quant_multiplier",
            torch.tensor(config.value_cache_quant_multiplier, dtype=torch.float32),
        )

    def project(self, hidden_states: torch.Tensor) -> Glm53MlaProjection:
        if hidden_states.ndim != 3:
            raise ValueError("MLA expects [batch, sequence, hidden]")
        batch, length, _ = hidden_states.shape
        q_latent = rms_norm(
            self.q_a_proj(hidden_states), self.q_a_norm, self.config.rms_norm_eps
        )
        query = self.q_b_proj(q_latent).view(
            batch, length, self.config.num_attention_heads, self.config.qk_head_dim
        )
        kv_latent = rms_norm(
            self.kv_a_proj(hidden_states), self.kv_a_norm, self.config.rms_norm_eps
        )
        expanded = self.kv_b_proj(kv_latent).view(
            batch,
            length,
            self.config.num_attention_heads,
            self.config.qk_nope_head_dim + self.config.v_head_dim,
        )
        key, value = torch.split(
            expanded,
            [self.config.qk_nope_head_dim, self.config.v_head_dim],
            dim=-1,
        )
        return Glm53MlaProjection(query=query, key=key, value=value)

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        for name in (
            "weight_scale",
            "activation_scale",
            "key_cache_quant_multiplier",
            "value_cache_quant_multiplier",
        ):
            validate_fp8_scale(getattr(self, name), name)


__all__ = ["Glm53MlaProjection", "Glm53NopeMla", "rms_norm"]
