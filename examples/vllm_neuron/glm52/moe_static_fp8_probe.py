# SPDX-License-Identifier: Apache-2.0
"""Compile/numeric probe for GLM-5.2's local Trn2 MoE shapes.

This is deliberately a local-kernel probe, not a model implementation. It
tests the four TP64 hybrid-EP shapes without requiring checkpoint weights or a
64-rank launch.
"""

import argparse
import json
import time

import nki.language as nl
import torch
import torch.nn.functional as F
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
)

import vllm_neuron  # noqa: F401
from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.functional.moe.moe_cte import moe_cte
from vllm_neuron.functional.moe.moe_tkg import moe_tkg

_WORLD_SIZE = 64
_NUM_EXPERTS = 256
_HIDDEN_SIZE = 6_144
_EXPERT_INTERMEDIATE = 2_048
_TOP_K = 8
_FP8_MAX = 240.0


def _shape_for_ep(ep_degree: int) -> tuple[int, int]:
    if ep_degree not in (8, 16, 32, 64):
        raise ValueError("ep_degree must be one of 8, 16, 32, 64")
    expert_tp = _WORLD_SIZE // ep_degree
    return _NUM_EXPERTS // ep_degree, _EXPERT_INTERMEDIATE // expert_tp


def _quantize_per_expert_projection(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[E, H, P, I]`` with one scale per expert/projection.

    The Trn2 TKG kernel currently validates the row-scale ABI
    ``[E, P, I]`` even when a static scale is broadcast across every output
    row.  Keep the quantization static, but materialize that required ABI.
    """
    max_abs = weight.abs().amax(dim=(1, 3), keepdim=True)
    scale = (max_abs / _FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (weight / scale).clamp(-_FP8_MAX, _FP8_MAX).to(
        torch.float8_e4m3fn
    )
    row_scale = scale.squeeze(1).expand(-1, -1, weight.shape[-1]).contiguous()
    return quantized, row_scale


def _quantize_per_expert(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[E, I, H]`` and broadcast its static scale to ``[E, H]``."""
    max_abs = weight.abs().amax(dim=(1, 2), keepdim=True)
    scale = (max_abs / _FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (weight / scale).clamp(-_FP8_MAX, _FP8_MAX).to(
        torch.float8_e4m3fn
    )
    row_scale = scale.squeeze(1).expand(-1, weight.shape[-1]).contiguous()
    return quantized, row_scale


class LocalMoeProbe(torch.nn.Module):
    def __init__(
        self,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        gate_up_scale: torch.Tensor | None,
        down_scale: torch.Tensor | None,
        use_fp8: bool,
    ) -> None:
        super().__init__()
        self.use_fp8 = use_fp8
        self.register_buffer("gate_up_weight", gate_up_weight)
        self.register_buffer("down_weight", down_weight)
        if use_fp8:
            assert gate_up_scale is not None and down_scale is not None
            self.register_buffer("gate_up_scale", gate_up_scale)
            self.register_buffer("down_scale", down_scale)
        else:
            self.gate_up_scale = None
            self.down_scale = None

    def forward(
        self,
        hidden: torch.Tensor,
        affinities: torch.Tensor,
        expert_indices: torch.Tensor,
        rank_id: torch.Tensor,
    ) -> torch.Tensor:
        kwargs = dict(
            hidden_input=hidden,
            expert_gate_up_weights=self.gate_up_weight,
            expert_down_weights=self.down_weight,
            expert_affinities=affinities,
            expert_index=expert_indices,
            is_all_expert=True,
            rank_id=rank_id,
            expert_gate_up_weights_scale=self.gate_up_scale,
            expert_down_weights_scale=self.down_scale,
            mask_unselected_experts=True,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            activation_fn=ActFnType.SiLU,
            output_dtype=torch.bfloat16,
        )
        return moe_tkg(**kwargs)


class LocalMoeCteProbe(torch.nn.Module):
    def __init__(
        self,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        gate_up_scale: torch.Tensor | None,
        down_scale: torch.Tensor | None,
        implementation: MoECTEImplementation,
    ) -> None:
        super().__init__()
        self.implementation = implementation
        self.register_buffer("gate_up_weight", gate_up_weight)
        self.register_buffer("down_weight", down_weight)
        if gate_up_scale is not None:
            local_experts, _, intermediate = gate_up_scale.shape
            self.register_buffer(
                "gate_up_scale",
                gate_up_scale.reshape(local_experts, 1, 2 * intermediate),
            )
            assert down_scale is not None
            self.register_buffer("down_scale", down_scale.unsqueeze(1))
        else:
            self.gate_up_scale = None
            self.down_scale = None

    def forward(
        self,
        hidden: torch.Tensor,
        affinities: torch.Tensor,
        token_position_to_id: torch.Tensor,
        block_to_expert: torch.Tensor,
    ) -> torch.Tensor:
        return moe_cte(
            hidden_states=hidden,
            expert_affinities_masked=affinities,
            gate_up_proj_weight=self.gate_up_weight,
            down_proj_weight=self.down_weight,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            block_size=128,
            implementation=self.implementation,
            gate_up_proj_scale=self.gate_up_scale,
            down_proj_scale=self.down_scale,
            activation_function=ActFnType.SiLU,
            compute_dtype=nl.bfloat16,
            is_tensor_update_accumulating=True,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        )


def _make_tkg_inputs(
    ep_degree: int,
    tokens: int,
    use_fp8: bool,
) -> tuple[LocalMoeProbe, tuple[torch.Tensor, ...], torch.Tensor]:
    local_experts, intermediate = _shape_for_ep(ep_degree)
    generator = torch.Generator().manual_seed(5200 + ep_degree + tokens)

    gate_up_reference = torch.zeros(
        local_experts,
        _HIDDEN_SIZE,
        2,
        intermediate,
        dtype=torch.bfloat16,
    )
    down_reference = torch.zeros(
        local_experts,
        intermediate,
        _HIDDEN_SIZE,
        dtype=torch.bfloat16,
    )
    gate_up_reference[0].normal_(mean=0.0, std=0.02, generator=generator)
    down_reference[0].normal_(mean=0.0, std=0.02, generator=generator)

    if use_fp8:
        gate_up_weight, gate_up_scale = _quantize_per_expert_projection(
            gate_up_reference.float()
        )
        down_weight, down_scale = _quantize_per_expert(down_reference.float())
    else:
        gate_up_weight, down_weight = gate_up_reference, down_reference
        gate_up_scale = down_scale = None

    hidden = torch.randn(
        tokens,
        _HIDDEN_SIZE,
        dtype=torch.bfloat16,
        generator=generator,
    )
    affinities = torch.zeros(tokens, _NUM_EXPERTS, dtype=torch.bfloat16)
    affinities[:, 0] = 2.5
    expert_indices = torch.arange(_TOP_K, dtype=torch.int32).expand(tokens, -1)
    rank_id = torch.zeros((1, 1), dtype=torch.int32)

    gate_up = torch.matmul(
        hidden.float(),
        gate_up_reference[0].float().reshape(_HIDDEN_SIZE, 2 * intermediate),
    ).view(tokens, 2, intermediate)
    gate, up = gate_up.unbind(dim=1)
    expected = torch.matmul(
        F.silu(gate) * up,
        down_reference[0].float(),
    )
    expected = (expected * 2.5).to(torch.bfloat16)

    model = LocalMoeProbe(
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        use_fp8,
    )
    return model, (hidden, affinities, expert_indices, rank_id), expected


def _make_cte_inputs(
    ep_degree: int,
    tokens: int,
    use_fp8: bool,
    implementation: MoECTEImplementation,
) -> tuple[LocalMoeCteProbe, tuple[torch.Tensor, ...], torch.Tensor]:
    if tokens % 128 != 0:
        raise ValueError("CTE tokens must be divisible by block size 128")
    local_experts, intermediate = _shape_for_ep(ep_degree)
    generator = torch.Generator().manual_seed(5252 + ep_degree + tokens)

    gate_up_reference = torch.zeros(
        local_experts,
        _HIDDEN_SIZE,
        2,
        intermediate,
        dtype=torch.bfloat16,
    )
    down_reference = torch.zeros(
        local_experts,
        intermediate,
        _HIDDEN_SIZE,
        dtype=torch.bfloat16,
    )
    gate_up_reference[0].normal_(mean=0.0, std=0.02, generator=generator)
    down_reference[0].normal_(mean=0.0, std=0.02, generator=generator)

    if use_fp8:
        gate_up_weight, gate_up_scale = _quantize_per_expert_projection(
            gate_up_reference.float()
        )
        down_weight, down_scale = _quantize_per_expert(down_reference.float())
    else:
        gate_up_weight, down_weight = gate_up_reference, down_reference
        gate_up_scale = down_scale = None

    hidden = torch.randn(
        tokens,
        _HIDDEN_SIZE,
        dtype=torch.bfloat16,
        generator=generator,
    )
    active_experts = min(_TOP_K, local_experts)
    per_expert_affinity = 2.5 / active_experts
    affinities = torch.zeros(tokens, local_experts, dtype=torch.bfloat16)
    affinities[:, :active_experts] = per_expert_affinity

    blocks_per_expert = tokens // 128
    token_blocks = torch.arange(tokens, dtype=torch.int32).view(
        blocks_per_expert, 128
    )
    token_position_to_id = (
        token_blocks.unsqueeze(0)
        .expand(local_experts, -1, -1)
        .reshape(-1)
        .contiguous()
    )
    block_to_expert = torch.arange(
        local_experts, dtype=torch.int32
    ).repeat_interleave(blocks_per_expert)

    gate_up = torch.matmul(
        hidden.float(),
        gate_up_reference[0].float().reshape(_HIDDEN_SIZE, 2 * intermediate),
    ).view(tokens, 2, intermediate)
    gate, up = gate_up.unbind(dim=1)
    expected = torch.matmul(
        F.silu(gate) * up,
        down_reference[0].float(),
    )
    expected = (expected * per_expert_affinity).to(torch.bfloat16)

    model = LocalMoeCteProbe(
        gate_up_weight,
        down_weight,
        gate_up_scale,
        down_scale,
        implementation,
    )
    inputs = (
        hidden,
        affinities.reshape(tokens * local_experts, 1),
        token_position_to_id,
        block_to_expert,
    )
    return model, inputs, expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=("tkg", "cte"), default="tkg")
    parser.add_argument(
        "--cte-implementation",
        choices=("shard_on_block", "shard_on_i"),
        default="shard_on_block",
    )
    parser.add_argument("--ep-degree", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="fp8")
    args = parser.parse_args()
    if args.kernel == "tkg" and args.tokens not in (1, 2, 4, 8, 16):
        raise ValueError("TKG tokens must be one of 1, 2, 4, 8, 16")
    if args.kernel == "cte" and args.tokens not in (128, 256, 512):
        raise ValueError("CTE tokens must be one of 128, 256, 512")

    use_fp8 = args.dtype == "fp8"
    if args.kernel == "tkg":
        model, inputs, expected = _make_tkg_inputs(
            args.ep_degree, args.tokens, use_fp8
        )
    else:
        implementation = getattr(MoECTEImplementation, args.cte_implementation)
        model, inputs, expected = _make_cte_inputs(
            args.ep_degree,
            args.tokens,
            use_fp8,
            implementation,
        )
    device = torch.device("neuron:0")
    model = model.to(device)
    device_inputs = tuple(value.to(device) for value in inputs)

    compile_start = time.perf_counter()
    compiled = torch.compile(model, backend=get_compile_backend_name())
    actual = compiled(*device_inputs).to("cpu")
    elapsed = time.perf_counter() - compile_start

    atol = 0.75 if use_fp8 else 0.15
    rtol = 0.08 if use_fp8 else 0.02
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    local_experts, intermediate = _shape_for_ep(args.ep_degree)
    print(
        json.dumps(
            {
                "status": "passed",
                "kernel": args.kernel,
                "cte_implementation": (
                    args.cte_implementation if args.kernel == "cte" else None
                ),
                "ep_degree": args.ep_degree,
                "tokens": args.tokens,
                "dtype": args.dtype,
                "fp8_scale_layout": "row-broadcast-static" if use_fp8 else None,
                "local_experts": local_experts,
                "intermediate_per_rank": intermediate,
                "compile_and_first_run_seconds": elapsed,
                "max_abs_error": float(
                    (actual.float() - expected.float()).abs().max()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
