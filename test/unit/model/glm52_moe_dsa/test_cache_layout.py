# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_neuron.model.glm52_moe_dsa.cache_layout import Glm52CacheLayout
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig


def test_frozen_tp64_cache_layout_pairs_only_full_indexers() -> None:
    config = Glm52MoeDsaConfig()
    layout = Glm52CacheLayout.build(
        config,
        world_size=64,
        cache_dtype=torch.float8_e4m3fn,
    )

    assert len(config.full_indexer_layer_ids) == 21
    assert len(layout.indexer_bindings) == 21
    assert len(layout.kv_spec.layers) == 78 + 11
    assert layout.indexer_binding(0).cache_name == "glm52.indexer_cache.0"
    assert layout.indexer_binding(0).cache_slot == 0
    assert layout.indexer_binding(1).cache_slot == 1
    assert layout.indexer_binding(2).cache_name == "glm52.indexer_cache.1"
    assert layout.indexer_binding(2).cache_slot == 0
    assert layout.indexer_binding(6).cache_slot == 1
    assert layout.indexer_binding(74).cache_slot == 0

    bound_layers = {binding.layer_idx for binding in layout.indexer_bindings}
    assert bound_layers == set(config.full_indexer_layer_ids)
    assert bound_layers.isdisjoint(config.shared_indexer_layer_ids)


def test_frozen_fp8_cache_capacity_is_exact() -> None:
    layout = Glm52CacheLayout.build(
        Glm52MoeDsaConfig(),
        world_size=64,
        cache_dtype=torch.float8_e4m3fn,
    )

    # Main expanded K/V: 78 * 2 * 256 bytes/token.
    # Paired index caches: ceil(21/2) * 2 * 128 bytes/token.
    assert layout.bytes_per_token_per_rank() == 42_752
    assert layout.bytes_per_token_per_rank() * 32_768 == 1_400_897_536


def test_bf16_cache_cost_is_exactly_twice_fp8() -> None:
    config = Glm52MoeDsaConfig()
    fp8 = Glm52CacheLayout.build(
        config,
        world_size=64,
        cache_dtype=torch.float8_e4m3fn,
    )
    bf16 = Glm52CacheLayout.build(
        config,
        world_size=64,
        cache_dtype=torch.bfloat16,
    )

    assert bf16.bytes_per_token_per_rank() == 2 * fp8.bytes_per_token_per_rank()
