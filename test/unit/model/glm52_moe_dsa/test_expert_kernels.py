# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.expert_kernels import (
    Glm52RoutedExperts,
    dense_glm52_affinities,
)
from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan
from vllm_neuron.utils.weight_loader import get_weight_loader


def _config() -> Glm52MoeDsaConfig:
    return Glm52MoeDsaConfig(
        vocab_size=16,
        hidden_size=4,
        num_hidden_layers=4,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=4,
        kv_lora_rank=2,
        qk_head_dim=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        index_n_heads=2,
        index_head_dim=2,
        index_topk=2,
        index_skip_topk_offset=1,
        index_topk_freq=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        first_k_dense_replace=1,
    )


def test_dense_affinities_preserve_only_selected_weights() -> None:
    indices = torch.tensor([[3, 1], [0, 2]], dtype=torch.int64)
    weights = torch.tensor([[1.5, 1.0], [0.25, 2.25]], dtype=torch.float32)

    dense = dense_glm52_affinities(indices, weights, num_experts=4)

    torch.testing.assert_close(
        dense,
        torch.tensor(
            [
                [0.0, 1.0, 0.0, 1.5],
                [0.25, 0.0, 2.25, 0.0],
            ]
        ),
    )
    torch.testing.assert_close(dense.sum(dim=-1), weights.sum(dim=-1))


def test_routed_experts_own_exact_local_layout_and_loaders() -> None:
    plan = RoutedExpertPlan(4, 2, 4, 8)
    module = Glm52RoutedExperts(_config(), plan, global_rank=3)

    assert module.ep_rank == 1
    assert module.block_size == 256
    assert module.gate_up_proj.shape == (2, 4, 2, 4)
    assert module.down_proj.shape == (2, 4, 4)
    assert module.gate_up_proj_scale.shape == (2, 2, 4)
    assert module.down_proj_scale.shape == (2, 4)
    assert module.gate_up_proj.dtype == torch.float8_e4m3fn
    assert module.down_proj.dtype == torch.float8_e4m3fn
    assert get_weight_loader(module.gate_up_proj).transform is not None
    assert get_weight_loader(module.down_proj).transform is not None
    assert get_weight_loader(module.gate_up_proj_scale).transform is not None
    assert get_weight_loader(module.down_proj_scale).transform is not None


def test_dense_affinities_reject_out_of_range_expert() -> None:
    try:
        dense_glm52_affinities(
            torch.tensor([[4]], dtype=torch.int64),
            torch.ones(1, 1),
            num_experts=4,
        )
    except ValueError as error:
        assert "outside the configured expert range" in str(error)
    else:
        raise AssertionError("out-of-range expert index was accepted")
