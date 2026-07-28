# SPDX-License-Identifier: Apache-2.0
"""Paged-cache layout for GLM-5.2 expanded MLA and shared DSA indexers."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec

from .config import Glm52MoeDsaConfig


@dataclass(frozen=True)
class IndexerCacheBinding:
    """Map one full-indexer layer onto K or V of a paired cache allocation."""

    layer_idx: int
    cache_name: str
    cache_slot: int


@dataclass(frozen=True)
class Glm52CacheLayout:
    """Cache specification and binding map for one TP rank.

    The correctness baseline stores fully expanded K/V for one attention head
    per TP64 rank. Full DSA indexers need one additional key tensor apiece;
    two indexer layers share the K/V allocations of one synthetic cache spec
    so the standard two-tensor vLLM cache ABI does not waste half the memory.
    Shared indexer layers own no index-key cache and reuse the preceding full
    layer's top-k indices.
    """

    kv_spec: KVSpec
    indexer_bindings: tuple[IndexerCacheBinding, ...]

    @classmethod
    def build(
        cls,
        config: Glm52MoeDsaConfig,
        *,
        world_size: int,
        cache_dtype: torch.dtype,
    ) -> "Glm52CacheLayout":
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if config.num_attention_heads % world_size:
            raise ValueError("attention heads must divide evenly over TP")
        heads_per_rank = config.num_attention_heads // world_size
        if heads_per_rank != 1:
            raise ValueError(
                "the initial GLM-5.2 expanded-cache path requires one "
                "attention head per rank (TP64)"
            )
        if cache_dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            raise ValueError(f"unsupported GLM cache dtype: {cache_dtype}")

        layers = [
            LayerSpec(
                name=f"layers.{layer_idx}.self_attn",
                num_kv_heads=heads_per_rank,
                head_size=config.qk_head_dim,
                dtype=cache_dtype,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]

        bindings = []
        for full_position, layer_idx in enumerate(config.full_indexer_layer_ids):
            cache_pair = full_position // 2
            cache_name = f"glm52.indexer_cache.{cache_pair}"
            bindings.append(
                IndexerCacheBinding(
                    layer_idx=layer_idx,
                    cache_name=cache_name,
                    cache_slot=full_position % 2,
                )
            )

        for cache_pair in range(ceil(len(config.full_indexer_layer_ids) / 2)):
            layers.append(
                LayerSpec(
                    name=f"glm52.indexer_cache.{cache_pair}",
                    num_kv_heads=1,
                    head_size=config.index_head_dim,
                    dtype=cache_dtype,
                )
            )

        return cls(
            kv_spec=KVSpec(layers=layers),
            indexer_bindings=tuple(bindings),
        )

    def indexer_binding(self, layer_idx: int) -> IndexerCacheBinding:
        for binding in self.indexer_bindings:
            if binding.layer_idx == layer_idx:
                return binding
        raise KeyError(f"layer {layer_idx} does not own a full DSA indexer")

    def bytes_per_token_per_rank(self) -> int:
        return sum(
            2
            * layer.num_kv_heads
            * layer.head_size
            * torch.empty((), dtype=layer.dtype).element_size()
            for layer in self.kv_spec.layers
        )
