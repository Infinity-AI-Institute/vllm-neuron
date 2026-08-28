# SPDX-License-Identifier: Apache-2.0
"""Hard layer-index dispatch between KDA v2 and sparse All-NoPE MLA."""

from __future__ import annotations

import torch
from torch import nn

from ._reference_kernels import load_reference_kernel
from .config import Glm53FlashInferenceConfig
from .indexer import Glm53FlashIndexer
from .kda import Glm53KdaAttention
from .mla import Glm53NopeMla
from .telemetry import Glm53FlashTelemetry


class Glm53DsaAttention(nn.Module):
    def __init__(self, config: Glm53FlashInferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.mla = Glm53NopeMla(config)
        self.indexer = Glm53FlashIndexer(config)
        self._key_cache: torch.Tensor | None = None
        self._value_cache: torch.Tensor | None = None
        self._index_hidden_cache: torch.Tensor | None = None

    def reset_state(self) -> int:
        had_state = self._key_cache is not None
        self._key_cache = None
        self._value_cache = None
        self._index_hidden_cache = None
        return int(had_state)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        projection = self.mla.project(hidden_states)
        batch, _, _, _ = projection.query.shape
        if self._key_cache is not None and self._key_cache.shape[0] != batch:
            self.reset_state()
        if self._key_cache is None:
            self._key_cache = projection.key.detach().clone()
            self._value_cache = projection.value.detach().clone()
            self._index_hidden_cache = hidden_states.detach().clone()
        else:
            self._key_cache = torch.cat(
                (self._key_cache, projection.key.detach()), dim=1
            )
            self._value_cache = torch.cat(
                (self._value_cache, projection.value.detach()), dim=1
            )
            self._index_hidden_cache = torch.cat(
                (self._index_hidden_cache, hidden_states.detach()), dim=1
            )
        total_length = self._key_cache.shape[1]
        key_lengths = torch.full(
            (batch,), total_length, dtype=torch.int64, device=hidden_states.device
        )
        topk = self.indexer(
            projection.query,
            self._index_hidden_cache,
            position_ids,
            key_lengths,
        )
        golden = load_reference_kernel("dsa")
        attended = golden.dsa_sparse_attention_forward(
            projection.query,
            self._key_cache,
            self._value_cache,
            topk,
            position_ids,
            key_lengths,
            topk=topk.shape[-1],
            scaling=self.config.qk_head_dim**-0.5,
            causal=True,
        )
        return self.mla.o_proj(attended.flatten(-2))


class Glm53FlashAttention(nn.Module):
    """Dispatch is frozen at construction and has no fallback branch."""

    def __init__(
        self,
        config: Glm53FlashInferenceConfig,
        *,
        layer_idx: int,
        telemetry: Glm53FlashTelemetry,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.telemetry = telemetry
        layer_type = config.layer_types[layer_idx]
        if layer_type == "deepseek_sparse_attention":
            self.path = "dsa"
            self.impl = Glm53DsaAttention(config)
        elif layer_type == "linear_attention":
            self.path = "kda"
            self.impl = Glm53KdaAttention(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"unsupported layer type {layer_type!r} at {layer_idx}")

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        if self.path == "dsa":
            self.telemetry.increment("dsa_path_active", self.layer_idx)
            self.telemetry.increment("mla_active", self.layer_idx)
            return self.impl(hidden_states, position_ids)
        self.telemetry.increment("kda_path_active", self.layer_idx)
        self.telemetry.increment("linear_path_active", self.layer_idx)
        return self.impl(hidden_states)

    def reset_state(self, reset_mask: torch.Tensor | None = None) -> int:
        if self.path == "dsa":
            return self.impl.reset_state()
        count = self.impl.reset_state(reset_mask)
        self.telemetry.state_buffer_reset_count[self.layer_idx] += count
        return count


__all__ = ["Glm53DsaAttention", "Glm53FlashAttention"]
