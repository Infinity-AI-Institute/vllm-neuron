# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 DSA wrapper around the canonical CPU golden."""

from __future__ import annotations

import torch
from torch import nn

from ._reference_kernels import load_reference_kernel
from .config import Glm53FlashInferenceConfig, validate_fp8_scale

DSA_KERNEL_SLUG = "nki_v0_reference_lightning_indexer"


class Glm53FlashIndexer(nn.Module):
    def __init__(self, config: Glm53FlashInferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Parameter(
            torch.empty(
                config.index_n_heads,
                config.num_attention_heads * config.qk_head_dim,
                config.index_head_dim,
                dtype=config.torch_dtype,
            )
        )
        nn.init.normal_(self.q_proj, std=config.hidden_size**-0.5)
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.index_n_heads * config.index_head_dim,
            bias=False,
            dtype=config.torch_dtype,
        )
        self.pool_weights = nn.Parameter(
            torch.full((config.index_kpool,), 1.0 / config.index_kpool)
        )
        self.register_buffer(
            "cache_quant_multiplier",
            torch.tensor(config.indexer_cache_quant_multiplier, dtype=torch.float32),
        )

    def forward(
        self,
        query: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        key_lengths: torch.Tensor,
    ) -> torch.Tensor:
        golden = load_reference_kernel("dsa")
        batch, length, _ = hidden_states.shape
        raw_keys = self.k_proj(hidden_states).view(
            batch,
            length,
            self.config.index_n_heads,
            self.config.index_head_dim,
        )
        pool = self.config.index_kpool
        pool_length = (length + pool - 1) // pool
        padding = pool_length * pool - length
        if padding:
            raw_keys = torch.cat(
                (
                    raw_keys,
                    torch.zeros(
                        batch,
                        padding,
                        self.config.index_n_heads,
                        self.config.index_head_dim,
                        dtype=raw_keys.dtype,
                        device=raw_keys.device,
                    ),
                ),
                dim=1,
            )
        packed_keys = (
            raw_keys.view(
                batch,
                pool_length,
                pool,
                self.config.index_n_heads,
                self.config.index_head_dim,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(
                batch,
                pool_length,
                self.config.index_n_heads * pool,
                self.config.index_head_dim,
            )
        )
        pooled_keys = golden.dsa_index_pool_projection(
            packed_keys, self.pool_weights, index_pool=pool
        )
        topk = min(self.config.index_topk, pool_length)
        pool_indices = golden.lightning_indexer_topk(
            query,
            self.q_proj,
            pooled_keys,
            torch.div(position_ids, pool, rounding_mode="floor"),
            torch.div(key_lengths + pool - 1, pool, rounding_mode="floor"),
            topk=topk,
            causal=True,
            index_pool=1,
        )
        pool_cells = torch.arange(pool, device=query.device, dtype=pool_indices.dtype)
        indices = (pool_indices.unsqueeze(-1) * pool + pool_cells).flatten(-2)
        if self.config.index_kpool_always_select_tail:
            tail = position_ids.clamp(max=length - 1).to(indices.dtype)
            missing = ~(indices == tail.unsqueeze(-1)).any(dim=-1)
            indices[..., -1] = torch.where(missing, tail, indices[..., -1])
        return indices

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        validate_fp8_scale(self.cache_quant_multiplier, "cache_quant_multiplier")
        # This imports and uses the authoritative bound checker as an additional
        # guard; the local validator also enforces dtype and non-None defaults.
        fp8 = load_reference_kernel("fp8")
        fp8.assert_indexer_multiplier_bounded(
            float(torch.max(self.cache_quant_multiplier).item()),
            layer_idx=-1,
        )


__all__ = ["DSA_KERNEL_SLUG", "Glm53FlashIndexer"]
