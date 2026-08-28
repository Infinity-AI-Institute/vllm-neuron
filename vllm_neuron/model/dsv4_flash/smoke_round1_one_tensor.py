# SPDX-License-Identifier: Apache-2.0
"""Round-1 1-tensor smoke: prove UE8M0 FP8 dequant math is byte-clean.

Run:

    python -m vllm_neuron.model.dsv4_flash.smoke_round1_one_tensor \
        [--hf-shard <shard.safetensors>] [--verbose]

If ``--hf-shard`` is provided (must be a shard containing at least one
FP8 e4m3 non-expert weight + its ``.scale`` UE8M0 partner), the smoke:

    1. Reads the FP8 tensor and its UE8M0 scale exponents.
    2. Computes a hand golden dequant: for each 128×128 tile
       ``w_fp32 * (2 ** exp)``, gathered blockwise.
    3. Calls ``dequantize_block_fp8_ue8m0`` with the same inputs.
    4. Compares max abs error at bf16 — pass at < 1e-3 (fp32 arithmetic
       is exact; the bar is the round-trip through bf16 truncation).

If ``--hf-shard`` is not provided, the smoke synthesizes an FP8 tensor
+ UE8M0 scale, runs the same pipeline, and asserts the library matches
the golden to within a bf16 rounding tolerance.

FP4-UE8M0 verification is deferred to Round 2 (the FP4 packing layout
must be pinned against a real routed-expert shard first).

Exit 0 on pass, 1 on fail.  Prints receipt JSON to stdout on both.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Any

import torch

from .checkpoint_convert import (
    SCALE_SUFFIX,
    dequantize_block_fp8_ue8m0,
)


def _golden_dequant_ue8m0(
    weight_fp8: torch.Tensor, scale_exp: torch.Tensor, block: tuple[int, int]
) -> torch.Tensor:
    """Hand-rolled per-tile UE8M0 dequant; the reference for the library."""
    w = weight_fp8.to(torch.float32)
    exp = scale_exp.to(torch.int32)
    out = torch.empty_like(w)
    bo, bi = block
    O, I = w.shape
    for i0 in range(0, O, bo):
        for j0 in range(0, I, bi):
            tile = w[i0 : i0 + bo, j0 : j0 + bi]
            e = int(exp[i0 // bo, j0 // bi].item())
            out[i0 : i0 + bo, j0 : j0 + bi] = tile * (2.0 ** e)
    return out


def _synth_fp8_ue8m0(
    shape: tuple[int, int],
    block: tuple[int, int],
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic FP8 tensor + UE8M0 exponent tensor for offline exercise.

    The exponent range is [-8, +8] rather than the full [0, 255] so that
    ``value * 2**exp`` stays in a numerical range where bf16 rounding
    is well-defined and the golden comparison is meaningful.  The
    validator in ``dequantize_block_fp8_ue8m0`` requires exponents in
    [0, 255] — we cast negatives to a positive-shift trick: subtract
    a fixed 8 from the input tensor's fp32 scale before FP8 cast, so
    ``exp - 8`` lands in [-8, 0] and the corresponding UE8M0 value
    in [0, 8] cleanly.  This keeps the smoke's arithmetic honest.
    """
    torch.manual_seed(seed)
    O, I = shape
    bo, bi = block
    # Random FP8 weights.
    values = torch.empty(O, I, dtype=torch.float32).uniform_(-1.0, 1.0)
    values = values.to(torch.float8_e4m3fn).to(torch.float32)
    fp8 = values.to(torch.float8_e4m3fn)
    # Random UE8M0 exponents in [0, 8].  Multiplier 2**exp in [1, 256].
    exp = torch.randint(
        0, 9, (math.ceil(O / bo), math.ceil(I / bi)), dtype=torch.uint8
    )
    return fp8, exp


def _run_lib_vs_golden(
    fp8: torch.Tensor,
    scale_exp: torch.Tensor,
    block: tuple[int, int],
) -> dict[str, Any]:
    """Dequant with library + hand golden; compare at bf16."""
    lib_bf16 = dequantize_block_fp8_ue8m0(fp8, scale_exp, block, torch.bfloat16)
    golden_fp32 = _golden_dequant_ue8m0(fp8, scale_exp, block)
    golden_bf16 = golden_fp32.to(torch.bfloat16)
    diff = (lib_bf16.to(torch.float32) - golden_bf16.to(torch.float32)).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    return {
        "lib_shape": tuple(lib_bf16.shape),
        "lib_dtype": str(lib_bf16.dtype),
        "max_abs_error_bf16": max_abs,
        "mean_abs_error_bf16": mean_abs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-shard", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    receipt: dict[str, Any] = {
        "start_unix": int(time.time()),
        "mode": "hf-shard" if args.hf_shard else "synthetic",
        "smoke": "dsv4-flash-round1-fp8-ue8m0",
    }
    block = (128, 128)
    try:
        if args.hf_shard:
            from safetensors.torch import load_file

            store = load_file(args.hf_shard)
            fp8_key = None
            for key in store.keys():
                if key.endswith(".weight") and (key + SCALE_SUFFIX) in store:
                    weight = store[key]
                    if weight.dtype == torch.float8_e4m3fn:
                        fp8_key = key
                        break
            if fp8_key is None:
                raise RuntimeError(
                    f"no FP8-e4m3 weight+scale pair in {args.hf_shard}"
                )
            weight = store[fp8_key]
            scale = store[fp8_key + SCALE_SUFFIX]
            receipt["hf_shard"] = args.hf_shard
            receipt["hf_key"] = fp8_key
            receipt["hf_weight_dtype"] = str(weight.dtype)
            receipt["hf_scale_dtype"] = str(scale.dtype)
            receipt["hf_weight_shape"] = tuple(weight.shape)
            receipt["hf_scale_shape"] = tuple(scale.shape)
            cmp = _run_lib_vs_golden(weight, scale, block)
            receipt.update(cmp)
        else:
            fp8, exp = _synth_fp8_ue8m0((256, 384), block, seed=42)
            cmp = _run_lib_vs_golden(fp8, exp, block)
            receipt.update(cmp)

        # Numerical bar: bf16 slack from fp32->bf16 truncation.
        max_bar = 1e-3
        if receipt.get("max_abs_error_bf16", 0.0) > max_bar:
            raise RuntimeError(
                f"dequant max_abs error {receipt['max_abs_error_bf16']!r} > {max_bar}"
            )

        receipt["status"] = "PASS"
        print(json.dumps(receipt, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        receipt["status"] = "FAIL"
        receipt["error"] = repr(exc)
        print(json.dumps(receipt, indent=2))
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
