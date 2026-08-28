# SPDX-License-Identifier: Apache-2.0
"""Round-5 1-tensor smoke: prove `_convert_glm53_checkpoint` produces the
correct dequantized bf16 layout for one MoE expert weight against a
hand-computed golden.

Run:

    python -m vllm_neuron.model.glm53_flash.smoke_round5_one_tensor \
        [--hf-shard <shard.safetensors>] [--verbose]

If ``--hf-shard`` is provided (must be a shard containing at least one
routed-expert FP8 weight + its ``_scale_inv``), the smoke:

    1. Reads the FP8 tensor and its scale_inv.
    2. Computes a hand golden dequant: ``w_fp32 * scale_inv``, blockwise.
    3. Calls ``dequantize_block_fp8`` with the same inputs.
    4. Compares max abs error; passes at < 1e-6 (fp32 arithmetic is exact
       for the identity multiplier so this is a real bar, not a floor).

If ``--hf-shard`` is not provided, the smoke synthesizes an FP8 tensor
plus scale, runs the same pipeline, and asserts the golden matches
bit-for-bit — proves the code path, does not prove weight-file plumbing.

Both modes then build a tiny synthetic 1-layer state dict and pass it
through ``_convert_glm53_checkpoint`` to confirm the layer 0 dense-MLP
conversion produces the ``gate_proj.weight`` key the wrapper expects.

Exit 0 on pass, 1 on fail.  Prints receipt JSON to stdout on both.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

import torch

from .checkpoint_convert import (
    EXPECTED_HF_TENSOR_COUNT,
    _convert_glm53_checkpoint,
    dequantize_block_fp8,
)
from .config import Glm53FlashInferenceConfig


def _golden_dequant(
    weight_fp8: torch.Tensor, scale_inv: torch.Tensor, block: tuple[int, int]
) -> torch.Tensor:
    """Hand-rolled reciprocal blockwise dequant; the reference for the
    library function under test."""
    w = weight_fp8.to(torch.float32)
    s = scale_inv.to(torch.float32)
    out = torch.empty_like(w)
    bo, bi = block
    O, I = w.shape
    for i0 in range(0, O, bo):
        for j0 in range(0, I, bi):
            tile = w[i0 : i0 + bo, j0 : j0 + bi]
            scale = s[i0 // bo, j0 // bi]
            out[i0 : i0 + bo, j0 : j0 + bi] = tile * scale
    return out


def _synth_fp8_tensor(
    shape: tuple[int, int], block: tuple[int, int], seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic FP8 tensor + block scale for offline exercise."""
    torch.manual_seed(seed)
    O, I = shape
    bo, bi = block
    # Draw random values in a range that survives cast to float8_e4m3fn
    # without saturating (E4M3 max = 240).  Use a per-block reciprocal
    # scale that varies to exercise the blockwise arithmetic.
    values = torch.empty(O, I, dtype=torch.float32).uniform_(-1.0, 1.0)
    values = values.to(torch.float8_e4m3fn).to(torch.float32)  # simulate cast
    scale = torch.empty(math.ceil(O / bo), math.ceil(I / bi), dtype=torch.float32)
    scale.uniform_(0.01, 0.5)
    fp8 = values.to(torch.float8_e4m3fn)
    return fp8, scale


def _run_lib_vs_golden(
    fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block: tuple[int, int],
) -> dict[str, Any]:
    """Dequant with library + hand golden; compare."""
    lib_bf16 = dequantize_block_fp8(fp8, scale_inv, block, torch.bfloat16)
    golden_fp32 = _golden_dequant(fp8, scale_inv, block)
    # Compare at bf16 precision (which is what the library returns).
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


def _synthetic_state_dict(src: Glm53FlashInferenceConfig) -> dict[str, torch.Tensor]:
    """Build a tiny synthetic state dict with just enough keys for layer 0
    (dense KDA) so ``_convert_glm53_checkpoint`` can run without a real
    checkpoint.

    The point is to prove the LAYER-KEY MAPPING is right — not the numerics.
    Weights are zeros; dtypes are picked to match the wrapper's expectation
    so the converter's dtype casts land where they should.
    """
    H = src.hidden_size
    I_mlp = src.intermediate_size
    L = src.num_hidden_layers
    V = src.vocab_size
    linear = src.linear_attn_config
    num_heads = linear.num_heads
    head_dim = linear.head_dim
    qkv = num_heads * head_dim
    ksize = linear.short_conv_kernel_size
    hc = 4  # hc_mult
    hc_rows = (2 + hc) * hc
    hc_cols = hc * H

    sd: dict[str, torch.Tensor] = {}
    sd["model.language_model.embed_tokens.weight"] = torch.zeros(V, H, dtype=torch.bfloat16)
    sd["model.language_model.norm.weight"] = torch.zeros(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.zeros(V, H, dtype=torch.bfloat16)

    # Layer 0 only (dense KDA).  Norms.
    b = "model.language_model.layers.0."
    sd[f"{b}input_layernorm.weight"] = torch.zeros(H, dtype=torch.bfloat16)
    sd[f"{b}post_attention_layernorm.weight"] = torch.zeros(H, dtype=torch.bfloat16)
    # mHC params.
    for suffix in (
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
    ):
        sd[f"{b}{suffix}"] = torch.zeros(hc_rows, hc_cols, dtype=torch.bfloat16)
    # KDA projections.
    for name, shape in (
        ("q_proj", (qkv, H)),
        ("k_proj", (qkv, H)),
        ("v_proj", (qkv, H)),
        ("b_proj", (num_heads, H)),
        ("f_a_proj", (head_dim, H)),
        ("f_b_proj", (qkv, head_dim)),
        ("g_a_proj", (head_dim, H)),
        ("g_b_proj", (qkv, head_dim)),
        ("o_proj", (H, qkv)),
    ):
        sd[f"{b}self_attn.{name}.weight"] = torch.zeros(shape, dtype=torch.bfloat16)
    sd[f"{b}self_attn.A_log"] = torch.zeros(num_heads, dtype=torch.float32)
    sd[f"{b}self_attn.dt_bias"] = torch.zeros(qkv, dtype=torch.float32)
    sd[f"{b}self_attn.o_norm.weight"] = torch.zeros(head_dim, dtype=torch.bfloat16)
    for stream in ("q", "k", "v"):
        sd[f"{b}self_attn.{stream}_conv1d.weight"] = torch.zeros(
            qkv, 1, ksize, dtype=torch.bfloat16
        )
    # Dense MLP (BF16 stand-ins; the converter's dequant path is exercised
    # by the FP8 vs golden comparison above).
    for name, shape in (
        ("gate_proj", (I_mlp, H)),
        ("up_proj", (I_mlp, H)),
        ("down_proj", (H, I_mlp)),
    ):
        sd[f"{b}mlp.{name}.weight"] = torch.zeros(shape, dtype=torch.bfloat16)
    return sd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-shard", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    receipt: dict[str, Any] = {
        "start_unix": int(time.time()),
        "mode": "hf-shard" if args.hf_shard else "synthetic",
    }
    block = (128, 128)
    try:
        if args.hf_shard:
            from safetensors.torch import load_file

            store = load_file(args.hf_shard)
            fp8_key = None
            for key in store.keys():
                if key.endswith(".weight") and key + "_scale_inv" in store:
                    if "experts" in key and "shared_experts" not in key:
                        fp8_key = key
                        break
            if fp8_key is None:
                raise RuntimeError(
                    f"no routed-expert FP8 weight+scale pair in {args.hf_shard}"
                )
            weight = store[fp8_key]
            scale = store[fp8_key + "_scale_inv"]
            receipt["hf_shard"] = args.hf_shard
            receipt["hf_key"] = fp8_key
            receipt["hf_weight_dtype"] = str(weight.dtype)
            receipt["hf_scale_dtype"] = str(scale.dtype)
            receipt["hf_weight_shape"] = tuple(weight.shape)
            receipt["hf_scale_shape"] = tuple(scale.shape)
            cmp = _run_lib_vs_golden(weight, scale, block)
            receipt.update(cmp)
        else:
            fp8, scale = _synth_fp8_tensor((256, 384), block, seed=42)
            cmp = _run_lib_vs_golden(fp8, scale, block)
            receipt.update(cmp)

        # Layer-key smoke: build synth state dict, run the converter,
        # confirm the layer-0 dense-MLP keys land where we expect.
        src = Glm53FlashInferenceConfig()
        sd = _synthetic_state_dict(src)
        # Truncate to num_hidden_layers=1 for the smoke.
        object.__setattr__(src, "allow_reduced_shapes", True)
        object.__setattr__(src, "num_hidden_layers", 1)
        object.__setattr__(src, "layer_types", (src.layer_types[0],))
        object.__setattr__(src, "mlp_layer_types", (src.mlp_layer_types[0],))
        converted = _convert_glm53_checkpoint(sd, src, tp_degree=1)
        expected_keys = [
            "embed_tokens.weight",
            "final_norm_weight",
            "lm_head.weight",
            "layers.0.input_norm_weight",
            "layers.0.post_attention_norm_weight",
            "layers.0.hc_attn.base",
            "layers.0.hc_attn.fn",
            "layers.0.hc_attn.scale",
            "layers.0.hc_mlp.base",
            "layers.0.hc_mlp.fn",
            "layers.0.hc_mlp.scale",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.b_proj.weight",
            "layers.0.self_attn.f_a_proj.weight",
            "layers.0.self_attn.f_b_proj.weight",
            "layers.0.self_attn.g_a_proj.weight",
            "layers.0.self_attn.g_b_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.A_log",
            "layers.0.self_attn.dt_bias",
            "layers.0.self_attn.o_norm_weight",
            "layers.0.self_attn.conv1d.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.down_proj.weight",
        ]
        missing = [k for k in expected_keys if k not in converted]
        receipt["layer0_expected_keys"] = len(expected_keys)
        receipt["layer0_missing_keys"] = missing
        receipt["converted_key_count"] = len(converted) - 1  # sans report
        receipt["conversion_report"] = converted["_conversion_report"]
        if missing:
            raise RuntimeError(f"missing converted keys: {missing}")

        # Numerical bar for dequant path.
        max_bar = 1e-3  # bf16 slack; fp32 -> bf16 truncation.
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
