# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_neuron.model.glm52_moe_dsa.moe import select_glm52_experts


def test_correction_bias_changes_selection_not_weights() -> None:
    hidden_states = torch.ones(1, 1)
    gate_weight = torch.tensor([[4.0], [2.0], [0.0], [-2.0]])
    correction_bias = torch.tensor([0.0, 0.0, 9.0, 10.0])

    expert_indices, routing_weights = select_glm52_experts(
        hidden_states,
        gate_weight,
        correction_bias,
        top_k=2,
        routed_scaling_factor=2.5,
    )

    selected = {
        int(expert_id): float(weight)
        for expert_id, weight in zip(expert_indices[0], routing_weights[0])
    }
    unbiased_scores = torch.sigmoid(torch.tensor([0.0, -2.0]))
    expected = unbiased_scores / unbiased_scores.sum() * 2.5

    assert set(selected) == {2, 3}
    assert torch.isclose(torch.tensor(selected[2]), expected[0])
    assert torch.isclose(torch.tensor(selected[3]), expected[1])


def test_routing_weights_sum_to_scaling_factor() -> None:
    generator = torch.Generator().manual_seed(52)
    hidden_states = torch.randn(7, 16, generator=generator)
    gate_weight = torch.randn(32, 16, generator=generator)
    correction_bias = torch.randn(32, generator=generator)

    _, routing_weights = select_glm52_experts(
        hidden_states,
        gate_weight,
        correction_bias,
        top_k=8,
        routed_scaling_factor=2.5,
    )

    torch.testing.assert_close(
        routing_weights.sum(dim=-1),
        torch.full((7,), 2.5),
    )
    assert routing_weights.dtype is torch.float32
