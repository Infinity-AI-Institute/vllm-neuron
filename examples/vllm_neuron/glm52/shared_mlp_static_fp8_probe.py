# SPDX-License-Identifier: Apache-2.0
"""Compile/numeric probe for GLM-5.2's always-active shared expert."""

import argparse
import json
import time

import torch
import torch.nn.functional as F
from nkilib.core.utils.common_types import ActFnType, QuantizationType

import vllm_neuron  # noqa: F401
from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.functional.mlp import mlp

_HIDDEN_SIZE = 6_144
_SHARED_INTERMEDIATE = 2_048
_FP8_MAX = 240.0
_SCALE_ROWS = 128


def _quantize_static(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (
        tensor.abs().amax().to(torch.float32) / _FP8_MAX
    ).clamp_min(torch.finfo(torch.float32).tiny)
    quantized = (tensor.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(
        torch.float8_e4m3fn
    )
    return quantized, scale


def _dequantize_activation(
    tensor: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return (
        (tensor.float() / scale)
        .clamp(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
        .float()
        * scale
    )


class SharedMlpProbe(torch.nn.Module):
    def __init__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
        scales: tuple[torch.Tensor, ...] | None,
    ) -> None:
        super().__init__()
        self.use_fp8 = scales is not None
        self.register_buffer("gate", gate)
        self.register_buffer("up", up)
        self.register_buffer("down", down)
        if scales is None:
            self.gate_scale = None
            self.up_scale = None
            self.down_scale = None
            self.gate_up_input_scale = None
            self.down_input_scale = None
        else:
            (
                gate_scale,
                up_scale,
                down_scale,
                gate_up_input_scale,
                down_input_scale,
            ) = scales
            self.register_buffer("gate_scale", gate_scale)
            self.register_buffer("up_scale", up_scale)
            self.register_buffer("down_scale", down_scale)
            self.register_buffer(
                "gate_up_input_scale",
                gate_up_input_scale,
            )
            self.register_buffer("down_input_scale", down_input_scale)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return mlp(
            hidden,
            self.gate,
            self.up,
            self.down,
            act_fn=ActFnType.SiLU,
            quantization_type=(
                QuantizationType.STATIC
                if self.use_fp8
                else QuantizationType.NONE
            ),
            gate_w_scale=self.gate_scale,
            up_w_scale=self.up_scale,
            down_w_scale=self.down_scale,
            gate_up_in_scale=self.gate_up_input_scale,
            down_in_scale=self.down_input_scale,
            output_dtype=torch.bfloat16,
        )


def _make_probe(
    tp_degree: int,
    tokens: int,
    use_fp8: bool,
) -> tuple[SharedMlpProbe, torch.Tensor, torch.Tensor]:
    if _SHARED_INTERMEDIATE % tp_degree:
        raise ValueError("shared intermediate size must divide by TP degree")
    local_intermediate = _SHARED_INTERMEDIATE // tp_degree
    generator = torch.Generator().manual_seed(5_252 + tp_degree + tokens)
    hidden = torch.randn(
        tokens,
        _HIDDEN_SIZE,
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate_ref = torch.randn(
        _HIDDEN_SIZE,
        local_intermediate,
        dtype=torch.bfloat16,
        generator=generator,
    ) * 0.02
    up_ref = torch.randn(
        _HIDDEN_SIZE,
        local_intermediate,
        dtype=torch.bfloat16,
        generator=generator,
    ) * 0.02
    down_ref = torch.randn(
        local_intermediate,
        _HIDDEN_SIZE,
        dtype=torch.bfloat16,
        generator=generator,
    ) * 0.02

    if not use_fp8:
        expected = (
            F.silu(hidden.float() @ gate_ref.float())
            * (hidden.float() @ up_ref.float())
        ) @ down_ref.float()
        return (
            SharedMlpProbe(gate_ref, up_ref, down_ref, None),
            hidden,
            expected.to(torch.bfloat16),
        )

    gate, gate_scale_scalar = _quantize_static(gate_ref)
    up, up_scale_scalar = _quantize_static(up_ref)
    down, down_scale_scalar = _quantize_static(down_ref)
    _, hidden_scale_scalar = _quantize_static(hidden)
    hidden_dequant = _dequantize_activation(hidden, hidden_scale_scalar)
    # Static-FP8 CTE requires the normalized activation to be quantized before
    # NF.mlp. Decode/TKG accepts BF16 and performs this quantization internally.
    model_hidden = (
        (hidden.float() / hidden_scale_scalar)
        .clamp(-_FP8_MAX, _FP8_MAX)
        .to(torch.float8_e4m3fn)
        if tokens > 128
        else hidden
    )
    gate_dequant = gate.float() * gate_scale_scalar
    up_dequant = up.float() * up_scale_scalar
    intermediate = (
        F.silu(hidden_dequant @ gate_dequant)
        * (hidden_dequant @ up_dequant)
    )
    _, down_input_scale_scalar = _quantize_static(intermediate)
    intermediate_dequant = _dequantize_activation(
        intermediate,
        down_input_scale_scalar,
    )
    expected = intermediate_dequant @ (down.float() * down_scale_scalar)

    def rows(scalar: torch.Tensor) -> torch.Tensor:
        return scalar.reshape(1, 1).expand(_SCALE_ROWS, 1).contiguous()

    scales = (
        rows(gate_scale_scalar),
        rows(up_scale_scalar),
        rows(down_scale_scalar),
        rows(hidden_scale_scalar),
        rows(down_input_scale_scalar),
    )
    return (
        SharedMlpProbe(gate, up, down, scales),
        model_hidden,
        expected.to(torch.bfloat16),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tp-degree",
        type=int,
        choices=(1, 2, 4, 8, 16, 32, 64),
        required=True,
    )
    parser.add_argument("--tokens", type=int, choices=(1, 256), default=1)
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="fp8")
    args = parser.parse_args()

    model, hidden, expected = _make_probe(
        args.tp_degree,
        args.tokens,
        args.dtype == "fp8",
    )
    device = torch.device("neuron:0")
    model = model.to(device)
    hidden = hidden.to(device)

    started = time.perf_counter()
    compiled = torch.compile(model, backend=get_compile_backend_name())
    output_cpu = compiled(hidden).to("cpu")
    elapsed = time.perf_counter() - started
    max_abs_error = (output_cpu.float() - expected.float()).abs().max().item()
    torch.testing.assert_close(
        output_cpu.float(),
        expected.float(),
        atol=0.75 if args.dtype == "fp8" else 0.15,
        rtol=0.08 if args.dtype == "fp8" else 0.02,
    )

    print(
        json.dumps(
            {
                "status": "passed",
                "backend": get_compile_backend_name(),
                "dtype": args.dtype,
                "tp_degree": args.tp_degree,
                "tokens": args.tokens,
                "hidden_size": _HIDDEN_SIZE,
                "local_intermediate": _SHARED_INTERMEDIATE // args.tp_degree,
                "compile_and_first_run_seconds": elapsed,
                "max_abs_error": max_abs_error,
                "output_finite": bool(torch.isfinite(output_cpu).all()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
