# SPDX-License-Identifier: Apache-2.0

import types

import torch

from vllm_neuron.functional.moe.moe_blockwise import build_blockwise_mapping
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan
from vllm_neuron.model.glm52_moe_dsa.sparse_mlp import (
    Glm52SparseMlp,
    _mask_padded_affinities,
    _prefill_padding_mask,
    glm52_rms_norm,
)


class FakeGroup:
    rank_in_group = 0

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


def test_prefill_padding_mask_uses_slot_mapping_sentinels() -> None:
    active = _prefill_padding_mask(
        torch.arange(288, 320),
        num_tokens=32,
    )
    fully_padded = _prefill_padding_mask(
        torch.full((32,), -1),
        num_tokens=32,
    )

    torch.testing.assert_close(active, torch.ones(32, dtype=torch.bool))
    torch.testing.assert_close(fully_padded, torch.zeros(32, dtype=torch.bool))


def test_prefill_rejects_slot_mapping_with_wrong_local_token_count() -> None:
    try:
        _prefill_padding_mask(
            torch.tensor([0]),
            num_tokens=2,
        )
    except ValueError as error:
        assert "rank-local prefill token" in str(error)
    else:
        raise AssertionError("mismatched slot_mapping length was accepted")


def test_padded_tokens_do_not_enter_blockwise_mapping_or_routed_output() -> None:
    total_tokens = 32
    real_tokens = 10
    num_experts = 4
    top_k = 2
    block_size = 8

    affinities = torch.zeros(total_tokens, num_experts)
    for token_id in range(total_tokens):
        affinities[token_id, token_id % num_experts] = 0.6
        affinities[token_id, (token_id + 1) % num_experts] = 0.4
    padding_mask = torch.arange(total_tokens) < real_tokens
    masked_affinities = _mask_padded_affinities(affinities, padding_mask)

    (
        flattened_affinities,
        token_position_to_id,
        block_to_expert,
        conditions,
    ) = build_blockwise_mapping(
        expert_affinities=masked_affinities,
        num_local_experts=num_experts,
        num_experts_per_token=top_k,
        block_size=block_size,
        moe_group=FakeGroup(1, 1.0),
        tp_degree=1,
        padding_mask=padding_mask,
    )

    valid_token_ids = token_position_to_id[token_position_to_id >= 0]
    assert torch.all(valid_token_ids < real_tokens)
    torch.testing.assert_close(
        flattened_affinities.reshape(total_tokens, num_experts),
        masked_affinities,
    )

    # CPU reference for the blockwise routed scatter. Its result must match a
    # direct per-token expert sum for real rows and remain zero for padding.
    hidden_states = torch.arange(1, total_tokens + 1, dtype=torch.float32).reshape(
        -1, 1
    )
    routed_from_mapping = torch.zeros_like(hidden_states)
    token_blocks = token_position_to_id.reshape(-1, block_size)
    for block_idx, expert_id in enumerate(block_to_expert.tolist()):
        if not conditions[block_idx]:
            continue
        token_ids = token_blocks[block_idx]
        token_ids = token_ids[token_ids >= 0].to(torch.long)
        routed_from_mapping[token_ids] += (
            hidden_states[token_ids]
            * masked_affinities[token_ids, expert_id].unsqueeze(1)
            * (expert_id + 1)
        )

    expected = torch.zeros_like(hidden_states)
    expert_multipliers = torch.arange(1, num_experts + 1, dtype=torch.float32)
    expected[:real_tokens] = hidden_states[:real_tokens] * (
        masked_affinities[:real_tokens] @ expert_multipliers
    ).unsqueeze(1)
    torch.testing.assert_close(routed_from_mapping, expected)
