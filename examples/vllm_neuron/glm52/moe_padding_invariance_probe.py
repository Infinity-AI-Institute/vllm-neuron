# SPDX-License-Identifier: Apache-2.0
"""Qualify padded GLM-5.2 routed-MoE prefill on one Trainium2 rank.

The production graph pads a short prompt to 2,048 tokens, builds a blockwise
mapping from only the real rows, and passes many dummy ``-1`` token slots to
``moe_cte(shard_on_i)`` with ``skip_token=True``.  This probe evaluates the
same real rows twice: once inside an all-valid mapping and once inside a
padding-heavy mapping.  Routed expert output for the real rows must be
invariant, and padded rows must remain zero.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation

import vllm_neuron  # noqa: F401
from moe_static_fp8_probe import _make_cte_inputs, _shape_for_ep
from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.functional.moe.moe_blockwise import build_blockwise_mapping


class _SingleRankGroup:
    rank_in_group = 0
    world_size = 1


class PaddingInvariantProbe(torch.nn.Module):
    def __init__(
        self,
        kernel: torch.nn.Module,
        *,
        local_experts: int,
        top_k: int,
        real_tokens: int,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.local_experts = local_experts
        self.top_k = top_k
        self.real_tokens = real_tokens
        self.group = _SingleRankGroup()

    def _run(
        self,
        hidden: torch.Tensor,
        affinities: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_affinities = affinities * padding_mask.to(affinities.dtype).unsqueeze(1)
        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            _,
        ) = build_blockwise_mapping(
            expert_affinities=masked_affinities,
            num_local_experts=self.local_experts,
            num_experts_per_token=self.top_k,
            block_size=self.kernel.block_size,
            moe_group=self.group,
            tp_degree=1,
            padding_mask=padding_mask,
        )
        return self.kernel(
            hidden,
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        affinities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = torch.arange(hidden.shape[0], device=hidden.device)
        all_valid = torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        padding_mask = token_ids < self.real_tokens
        all_valid_output = self._run(hidden, affinities, all_valid)
        padded_output = self._run(hidden, affinities, padding_mask)
        return all_valid_output, padded_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-degree", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--real-tokens", type=int, default=192)
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="fp8")
    args = parser.parse_args()

    if args.tokens != 2048:
        raise ValueError("the production first-serving prefill shape is 2,048 tokens")
    if not 0 < args.real_tokens < args.tokens:
        raise ValueError("real-tokens must be between 1 and tokens-1")

    use_fp8 = args.dtype == "fp8"
    local_experts, _ = _shape_for_ep(args.ep_degree)
    kernel, inputs, expected = _make_cte_inputs(
        args.ep_degree,
        args.tokens,
        use_fp8,
        MoECTEImplementation.shard_on_i,
    )
    hidden, flattened_affinities, _, _ = inputs
    affinities = flattened_affinities.view(args.tokens, local_experts)
    model = PaddingInvariantProbe(
        kernel,
        local_experts=local_experts,
        top_k=8,
        real_tokens=args.real_tokens,
    )

    device = torch.device("neuron:0")
    model = model.to(device)
    hidden_device = hidden.to(device)
    affinities_device = affinities.to(device)

    compile_start = time.perf_counter()
    compiled = torch.compile(model, backend=get_compile_backend_name())
    all_valid_output, padded_output = compiled(hidden_device, affinities_device)
    all_valid_output = all_valid_output.to("cpu").float()
    padded_output = padded_output.to("cpu").float()
    elapsed = time.perf_counter() - compile_start

    real = slice(0, args.real_tokens)
    padded = slice(args.real_tokens, args.tokens)
    expected_float = expected.float()
    atol = 0.75 if use_fp8 else 0.15
    rtol = 0.08 if use_fp8 else 0.02
    torch.testing.assert_close(
        all_valid_output[real],
        expected_float[real],
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        padded_output[real],
        expected_float[real],
        atol=atol,
        rtol=rtol,
    )

    real_cosine = float(
        F.cosine_similarity(
            all_valid_output[real].reshape(1, -1),
            padded_output[real].reshape(1, -1),
        ).item()
    )
    real_max_abs_delta = float(
        (all_valid_output[real] - padded_output[real]).abs().max().item()
    )
    padded_max_abs = float(padded_output[padded].abs().max().item())
    if real_cosine < 0.99:
        raise AssertionError(
            f"real-token padded/all-valid cosine {real_cosine:.8f} < 0.99"
        )
    if padded_max_abs != 0.0:
        raise AssertionError(
            f"padded routed-MoE output must be zero, observed {padded_max_abs}"
        )

    print(
        json.dumps(
            {
                "status": "passed",
                "kernel": "moe_cte",
                "implementation": "shard_on_i",
                "skip_token": True,
                "ep_degree": args.ep_degree,
                "tokens": args.tokens,
                "real_tokens": args.real_tokens,
                "dtype": args.dtype,
                "local_experts": local_experts,
                "compile_and_first_run_seconds": elapsed,
                "real_padded_vs_all_valid_cosine": real_cosine,
                "real_padded_vs_all_valid_max_abs_delta": real_max_abs_delta,
                "padded_output_max_abs": padded_max_abs,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
