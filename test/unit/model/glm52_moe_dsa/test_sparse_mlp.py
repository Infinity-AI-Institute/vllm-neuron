# SPDX-License-Identifier: Apache-2.0

import types

import torch

from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan
from vllm_neuron.model.glm52_moe_dsa.sparse_mlp import (
    Glm52SparseMlp,
    glm52_rms_norm,
)


class FakeGroup:
    def __init__(self, world_size: int, reduce_multiplier: float) -> None:
        self.world_size = world_size
        self.reduce_multiplier = reduce_multiplier

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.reduce_multiplier


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


def test_rms_norm_uses_fp32_variance_and_restores_dtype() -> None:
    hidden = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)
    weight = torch.tensor([2.0, 0.5], dtype=torch.bfloat16)

    actual = glm52_rms_norm(hidden, weight, eps=0.0)
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    expected = (hidden.float() * torch.rsqrt(variance) * weight.float()).to(
        torch.bfloat16
    )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_decode_combines_routed_world_once_and_shared_subgroup_once() -> None:
    config = _config()
    plan = RoutedExpertPlan(4, 2, 4, 8)
    full_group = FakeGroup(4, 4.0)
    subgroup = FakeGroup(2, 2.0)
    module = Glm52SparseMlp(
        config,
        plan,
        global_rank=0,
        tp_group=full_group,
        expert_tp_group=subgroup,
    )

    module.gate.forward = types.MethodType(
        lambda self, hidden: (
            torch.zeros(hidden.shape[0], 2, dtype=torch.int64),
            torch.ones(hidden.shape[0], 2),
        ),
        module.gate,
    )
    module.experts.forward_decode = types.MethodType(
        lambda self, hidden, indices, weights: torch.ones_like(hidden),
        module.experts,
    )
    module.shared_experts.forward_decode = types.MethodType(
        lambda self, hidden: torch.full_like(hidden, 2.0),
        module.shared_experts,
    )

    output = module.forward_decode(
        torch.ones(2, 4, dtype=torch.bfloat16),
        norm_weight=torch.ones(4, dtype=torch.bfloat16),
    )

    # Routed: 1 * full world 4. Shared: 2 * subgroup 2.
    torch.testing.assert_close(output, torch.full_like(output, 8.0))


def test_static_fp8_selects_neuron_router_topk() -> None:
    config = _config()
    plan = RoutedExpertPlan(4, 2, 4, 8)
    module = Glm52SparseMlp(
        config,
        plan,
        global_rank=0,
        tp_group=FakeGroup(4, 4.0),
        expert_tp_group=FakeGroup(2, 2.0),
        static_fp8=True,
    )

    assert module.gate.topk_backend == "neuron"
