# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest
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


def test_neuron_backend_uses_rotational_topk_without_changing_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, int | None]] = []

    def fake_neuron_topk(
        tensor: torch.Tensor,
        k: int,
        dim: int,
        gather_dim: int,
        process_group=None,
        rank=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(
            {
                "k": k,
                "dim": dim,
                "gather_dim": gather_dim,
                "process_group": process_group,
                "rank": rank,
            }
        )
        return torch.topk(tensor, k=k, dim=dim)

    monkeypatch.setitem(
        sys.modules,
        "vllm_neuron.functional.topk",
        types.SimpleNamespace(topk=fake_neuron_topk),
    )
    hidden_states = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    gate_weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5], [0.25, -0.75]])
    correction_bias = torch.tensor([0.0, 0.25, -0.5, 0.75])

    actual_indices, actual_weights = select_glm52_experts(
        hidden_states,
        gate_weight,
        correction_bias,
        top_k=2,
        routed_scaling_factor=2.5,
        topk_backend="neuron",
    )
    router_scores = torch.sigmoid(hidden_states.float() @ gate_weight.float().T)
    _, expected_indices = torch.topk(
        router_scores + correction_bias.float(),
        k=2,
        dim=-1,
    )
    expected_weights = torch.gather(router_scores, -1, expected_indices)
    expected_weights = expected_weights / expected_weights.sum(
        dim=-1,
        keepdim=True,
    )
    expected_weights *= 2.5

    assert calls == [
        {
            "k": 2,
            "dim": -1,
            "gather_dim": -1,
            "process_group": None,
            "rank": None,
        }
    ]
    torch.testing.assert_close(actual_indices, expected_indices)
    torch.testing.assert_close(actual_weights, expected_weights)


def test_rejects_unknown_topk_backend() -> None:
    with pytest.raises(ValueError, match="topk_backend"):
        select_glm52_experts(
            torch.ones(1, 2),
            torch.ones(4, 2),
            torch.zeros(4),
            top_k=2,
            routed_scaling_factor=2.5,
            topk_backend="sort",
        )
