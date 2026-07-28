# SPDX-License-Identifier: Apache-2.0
"""Compile/numeric probe for GLM-5.2's exact Trn2 expert router."""

import argparse
import json
import statistics
import time

import torch

import vllm_neuron  # noqa: F401
from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.model.glm52_moe_dsa.moe import select_glm52_experts

_HIDDEN_SIZE = 6_144
_NUM_EXPERTS = 256
_TOP_K = 8
_ROUTED_SCALING_FACTOR = 2.5


class RouterTopkProbe(torch.nn.Module):
    def __init__(
        self,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("gate_weight", gate_weight)
        self.register_buffer("correction_bias", correction_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return select_glm52_experts(
            hidden_states,
            self.gate_weight,
            self.correction_bias,
            top_k=_TOP_K,
            routed_scaling_factor=_ROUTED_SCALING_FACTOR,
            topk_backend="neuron",
        )


def _sort_routes(
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(indices, dim=-1)
    return (
        torch.gather(indices, -1, order),
        torch.gather(weights, -1, order),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(1, 32), default=32)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")

    hidden_states = torch.zeros(
        args.tokens,
        _HIDDEN_SIZE,
        dtype=torch.bfloat16,
    )
    hidden_states[:, 0] = torch.linspace(0.5, 1.5, args.tokens)
    gate_weight = torch.zeros(
        _NUM_EXPERTS,
        _HIDDEN_SIZE,
        dtype=torch.float32,
    )
    gate_weight[:, 0] = torch.linspace(-1.5, 1.5, _NUM_EXPERTS)
    correction_bias = torch.zeros(_NUM_EXPERTS, dtype=torch.float32)
    correction_bias[:_TOP_K] = torch.linspace(4.0, 3.3, _TOP_K)

    expected_indices, expected_weights = select_glm52_experts(
        hidden_states,
        gate_weight,
        correction_bias,
        top_k=_TOP_K,
        routed_scaling_factor=_ROUTED_SCALING_FACTOR,
        topk_backend="torch",
    )
    expected_indices, expected_weights = _sort_routes(
        expected_indices,
        expected_weights,
    )

    device = torch.device("neuron:0")
    model = RouterTopkProbe(gate_weight, correction_bias).to(device)
    device_hidden = hidden_states.to(device)

    started = time.perf_counter()
    compiled = torch.compile(model, backend=get_compile_backend_name())
    actual_indices, actual_weights = compiled(device_hidden)
    actual_indices = actual_indices.to("cpu")
    actual_weights = actual_weights.to("cpu")
    compile_and_first_run_seconds = time.perf_counter() - started
    actual_indices, actual_weights = _sort_routes(
        actual_indices,
        actual_weights,
    )

    torch.testing.assert_close(actual_indices, expected_indices)
    torch.testing.assert_close(
        actual_weights,
        expected_weights,
        atol=1e-3,
        rtol=3e-3,
    )
    torch.testing.assert_close(
        actual_weights.sum(dim=-1),
        torch.full((args.tokens,), _ROUTED_SCALING_FACTOR),
        atol=2e-3,
        rtol=1e-3,
    )

    hot_seconds = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        compiled(device_hidden)
        hot_seconds.append(time.perf_counter() - started)

    print(
        json.dumps(
            {
                "status": "passed",
                "backend": get_compile_backend_name(),
                "tokens": args.tokens,
                "hidden_size": _HIDDEN_SIZE,
                "experts": _NUM_EXPERTS,
                "top_k": _TOP_K,
                "routed_scaling_factor": _ROUTED_SCALING_FACTOR,
                "selected_experts": actual_indices[0].tolist(),
                "routing_weight_sum_min": float(actual_weights.sum(dim=-1).min()),
                "routing_weight_sum_max": float(actual_weights.sum(dim=-1).max()),
                "compile_and_first_run_seconds": compile_and_first_run_seconds,
                "hot_run_seconds_min": min(hot_seconds),
                "hot_run_seconds_median": statistics.median(hot_seconds),
                "repetitions": args.repetitions,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
