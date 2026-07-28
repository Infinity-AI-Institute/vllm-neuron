# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_neuron.model.glm52_moe_dsa.cache_layout import Glm52CacheLayout
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.indexer import Glm52IndexShareState
from vllm_neuron.model.glm52_moe_dsa.mla import Glm52MlaAttention


def _reduced_config() -> Glm52MoeDsaConfig:
    return Glm52MoeDsaConfig(
        hidden_size=8,
        num_hidden_layers=4,
        intermediate_size=16,
        num_attention_heads=1,
        num_key_value_heads=1,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_head_dim=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        first_k_dense_replace=3,
    )


def test_mla_projection_matches_frozen_low_rank_equations() -> None:
    config = _reduced_config()
    layout = Glm52CacheLayout.build(
        config,
        world_size=1,
        cache_dtype=torch.bfloat16,
    )
    module = Glm52MlaAttention(
        config,
        layer_idx=3,
        cache_layout=layout,
        world_size=1,
        static_fp8=False,
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(52)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    generator=generator,
                )
                * 0.1
            )
        module.q_a_layernorm.fill_(1)
        module.kv_a_layernorm.fill_(1)

    hidden = torch.randn(2, config.hidden_size, generator=generator)
    cos = torch.ones(2, config.qk_rope_head_dim)
    sin = torch.zeros_like(cos)
    projected = module.project(hidden, cos, sin)

    q_a = hidden @ module.q_a_proj.weight
    q_a = q_a * torch.rsqrt(
        q_a.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps
    )
    expected_query = (q_a @ module.q_b_proj.weight).reshape(
        2,
        1,
        config.qk_head_dim,
    )
    q_pass, q_rot = torch.split(
        expected_query,
        [config.qk_nope_head_dim, config.qk_rope_head_dim],
        dim=-1,
    )
    q_rot = torch.cat((q_rot[..., 0::2], q_rot[..., 1::2]), dim=-1)
    expected_query = torch.cat((q_pass, q_rot), dim=-1)
    compressed = hidden @ module.kv_a_proj_with_mqa.weight
    kv_pass, k_rot = torch.split(
        compressed,
        [config.kv_lora_rank, config.qk_rope_head_dim],
        dim=-1,
    )
    kv_pass = kv_pass * torch.rsqrt(
        kv_pass.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps
    )
    expanded = (kv_pass @ module.kv_b_proj.weight).reshape(
        2,
        1,
        config.qk_nope_head_dim + config.v_head_dim,
    )
    k_nope, expected_value = torch.split(
        expanded,
        [config.qk_nope_head_dim, config.v_head_dim],
        dim=-1,
    )
    k_rot = torch.cat((k_rot[..., 0::2], k_rot[..., 1::2]), dim=-1)
    expected_key = torch.cat((k_nope, k_rot.unsqueeze(1)), dim=-1)

    torch.testing.assert_close(projected.q_resid, q_a)
    torch.testing.assert_close(projected.query, expected_query)
    torch.testing.assert_close(projected.key, expected_key)
    torch.testing.assert_close(projected.value, expected_value)


def test_shared_index_layer_reuses_state_and_updates_main_cache() -> None:
    config = _reduced_config()
    layout = Glm52CacheLayout.build(
        config,
        world_size=1,
        cache_dtype=torch.bfloat16,
    )
    module = Glm52MlaAttention(
        config,
        layer_idx=3,
        cache_layout=layout,
        world_size=1,
        static_fp8=False,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        module.q_a_layernorm.fill_(1)
        module.kv_a_layernorm.fill_(1)

    key_cache = torch.ones(2, 1, 2, config.qk_head_dim)
    value_cache = torch.ones(2, 1, 2, config.v_head_dim)
    previous = Glm52IndexShareState(
        topk_indices=torch.tensor([[2, 3]], dtype=torch.int32),
        source_layer_idx=2,
    )
    metadata = {
        "layers.3.self_attn": {
            "slot_mapping": torch.tensor([3]),
            "block_size": 2,
            "block_table_tensor": torch.tensor([[0, 1]], dtype=torch.int32),
        }
    }

    output, state = module.forward_paged_decode(
        torch.zeros(1, config.hidden_size, dtype=torch.bfloat16),
        (
            torch.ones(1, config.qk_rope_head_dim, dtype=torch.bfloat16),
            torch.zeros(1, config.qk_rope_head_dim, dtype=torch.bfloat16),
        ),
        torch.tensor([3]),
        metadata,
        key_cache=key_cache,
        value_cache=value_cache,
        previous_index_state=previous,
    )

    assert state is previous
    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(key_cache[1, 0, 1], torch.zeros(config.qk_head_dim))
    torch.testing.assert_close(
        value_cache[1, 0, 1],
        torch.zeros(config.v_head_dim),
    )


def test_full_index_layer_updates_fp8_index_and_main_caches() -> None:
    config = _reduced_config()
    layout = Glm52CacheLayout.build(
        config,
        world_size=1,
        cache_dtype=torch.float8_e4m3fn,
    )
    module = Glm52MlaAttention(
        config,
        layer_idx=0,
        cache_layout=layout,
        world_size=1,
        static_fp8=False,
        dtype=torch.bfloat16,
    )
    assert module.indexer is not None
    module.indexer.topk_backend = "torch"
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        module.q_a_layernorm.fill_(1)
        module.kv_a_layernorm.fill_(1)
        module.indexer.k_norm.bias[0] = 1
    module.set_cache_quant_multipliers(key=1, value=1)
    module.indexer.set_cache_quant_multiplier(16)

    key_cache = torch.zeros(
        2,
        1,
        2,
        config.qk_head_dim,
        dtype=torch.float8_e4m3fn,
    )
    value_cache = torch.zeros(
        2,
        1,
        2,
        config.v_head_dim,
        dtype=torch.float8_e4m3fn,
    )
    indexer_cache = torch.zeros(
        2,
        1,
        2,
        config.index_head_dim,
        dtype=torch.float8_e4m3fn,
    )
    metadata = {
        "layers.0.self_attn": {
            "slot_mapping": torch.tensor([3]),
            "block_size": 2,
            "block_table_tensor": torch.tensor([[0, 1]], dtype=torch.int32),
        },
        "glm52.indexer_cache.0": {
            "slot_mapping": torch.tensor([3]),
            "block_size": 2,
            "block_table_tensor": torch.tensor([[0, 1]], dtype=torch.int32),
        },
    }

    output, state = module.forward_paged_decode(
        torch.zeros(1, config.hidden_size, dtype=torch.bfloat16),
        (
            torch.ones(1, config.qk_rope_head_dim, dtype=torch.bfloat16),
            torch.zeros(1, config.qk_rope_head_dim, dtype=torch.bfloat16),
        ),
        torch.tensor([3]),
        metadata,
        key_cache=key_cache,
        value_cache=value_cache,
        indexer_cache=indexer_cache,
        previous_index_state=None,
    )

    assert state.source_layer_idx == 0
    assert state.topk_indices.shape == (1, config.index_topk)
    torch.testing.assert_close(output, torch.zeros_like(output))
    assert float(indexer_cache[1, 0, 1, 0].detach()) == 16
    torch.testing.assert_close(
        key_cache[1, 0, 1].float(),
        torch.zeros(config.qk_head_dim),
    )
    torch.testing.assert_close(
        value_cache[1, 0, 1].float(),
        torch.zeros(config.v_head_dim),
    )
