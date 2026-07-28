# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_neuron.model.glm52_moe_dsa.cache_layout import IndexerCacheBinding
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.indexer import (
    Glm52FullIndexer,
    advance_index_share_state,
)


def _tiny_config() -> Glm52MoeDsaConfig:
    return Glm52MoeDsaConfig(
        hidden_size=4,
        num_hidden_layers=7,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=2,
        kv_lora_rank=2,
        qk_head_dim=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        first_k_dense_replace=1,
    )


def _indexer(config: Glm52MoeDsaConfig) -> Glm52FullIndexer:
    torch.manual_seed(7)
    return Glm52FullIndexer(
        config,
        layer_idx=0,
        cache_binding=IndexerCacheBinding(
            layer_idx=0,
            cache_name="glm52.indexer_cache.0",
            cache_slot=0,
        ),
        dtype=torch.bfloat16,
    )


def test_full_indexer_paged_path_matches_dense_reference() -> None:
    config = _tiny_config()
    indexer = _indexer(config)
    cache = torch.zeros(2, 1, 2, config.index_head_dim, dtype=torch.bfloat16)
    indexer.bind_key_cache(cache)

    hidden = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.bfloat16,
    )
    q_resid = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
        dtype=torch.bfloat16,
    )
    cos = torch.ones(4, 2, dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    positions = torch.tensor([0, 1, 0, 1])
    metadata = {
        "glm52.indexer_cache.0": {
            "slot_mapping": torch.tensor([2, 3, 0, 1]),
            "block_size": 2,
            "block_table_tensor": torch.tensor([[1], [0]]),
        }
    }

    paged = indexer.forward_paged(
        hidden,
        q_resid,
        cos,
        sin,
        position_ids=positions,
        attn_metadata=metadata,
    )

    projection = indexer.project(hidden, q_resid, cos, sin)
    dense_keys = torch.stack(
        (projection.key[:2], projection.key[2:]),
        dim=0,
    )
    dense = indexer.forward_dense(
        hidden.reshape(2, 2, 4),
        q_resid.reshape(2, 2, 2),
        cos.reshape(2, 2, 2),
        sin.reshape(2, 2, 2),
        key_cache=dense_keys,
        position_ids=positions.reshape(2, 2),
    )

    torch.testing.assert_close(paged, dense.reshape(4, 1))


def test_index_share_state_preserves_and_replaces_exact_tensor() -> None:
    config = _tiny_config()
    first = torch.tensor([[1, 2]], dtype=torch.int32)
    replacement = torch.tensor([[3, 4]], dtype=torch.int32)

    state = advance_index_share_state(
        config,
        layer_idx=0,
        previous=None,
        computed_topk=first,
    )
    state = advance_index_share_state(
        config,
        layer_idx=3,
        previous=state,
        computed_topk=None,
    )
    assert state.source_layer_idx == 0
    assert state.topk_indices is first

    state = advance_index_share_state(
        config,
        layer_idx=6,
        previous=state,
        computed_topk=replacement,
    )
    assert state.source_layer_idx == 6
    assert state.topk_indices is replacement


def test_shared_indexer_rejects_missing_previous_topk() -> None:
    config = _tiny_config()
    with pytest.raises(ValueError, match="requires a previous full indexer"):
        advance_index_share_state(
            config,
            layer_idx=3,
            previous=None,
            computed_topk=None,
        )
