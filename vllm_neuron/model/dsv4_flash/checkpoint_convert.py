# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash HF -> Neuron checkpoint conversion (Round 1 skeleton).

The FP4/UE8M0 arithmetic below is the tightest correctness gate in the
whole port: without a byte-clean dequant, every downstream test looks
plausible and is silently wrong.  This module implements two dequant
primitives with fail-loud validators, and enumerates the HF-key ->
wrapper-key map for every module class the checkpoint carries.  The
actual per-layer conversion routines are stubbed with
``NotImplementedError`` referencing the module docstring; each stub is
paired with the specific structural decision it depends on (grouped
output projection axis order, hash-MoE lookup layout, compressor
overlap-state shape, etc.).

FP4-UE8M0 vs FP8-UE8M0 vs FP8-E4M3-reciprocal
---------------------------------------------
GLM-5.3-Flash reads FP8 e4m3 blockwise weights with a **reciprocal**
scale: ``w_bf16 = (w_fp32 * scale_inv).to(bf16)``.  DeepSeek-V4-Flash
uses a completely different pair of formats:

  * Routed experts: **FP4** (4 bits per weight, MXFP4-style packed) with
    **UE8M0** block scale.  UE8M0 is an unsigned 8-bit exponent — the
    multiplier is ``2 ** exp``, NEVER a raw float.  A 128x128 tile shares
    a single 8-bit exponent.  Dequant is
    ``w_bf16 = (w_fp32 * (2 ** exp)).to(bf16)``.
  * Non-experts (MLA-ish projections, dense MLP, shared expert, indexer
    where applicable): FP8 e4m3 with the same UE8M0 block scale.

The UE8M0 scale space and the E4M3-reciprocal scale space differ by
several orders of magnitude AND direction; treating one as the other
loads and runs and is quietly wrong (identical failure mode to the
OCP-448/legacy-240 divergence GLM-5.2 chased).  The validators below
refuse a suspicious scale rather than let it through.
"""

from __future__ import annotations

import math
import re
from typing import Any

import torch

from .config import (
    DeepseekV4FlashInferenceConfig,
    HF_SNAPSHOT_SHA,
    validate_ue8m0_scale,
)

# HF prefix contract (after NxDI's ``_STATE_DICT_MODEL_PREFIX = "model."`` strip).
# Verified against the transformers 5.15.1 ``deepseek_v4`` module — the text
# tree lives under ``layers.<i>....`` after the strip; ``embed_tokens.weight``
# and ``lm_head.weight`` remain top-level.
TEXT_LAYER_PREFIX = "layers."
LM_HEAD_KEY = "lm_head.weight"
EMBED_KEY = "embed_tokens.weight"
FINAL_NORM_KEY = "norm.weight"

# Reference tensor count from ``model.safetensors.index.json`` for HF snapshot
# 7872f01b1d1fe23eabc4c98b48bffcef5a386062.
EXPECTED_HF_TENSOR_COUNT = 2561
EXPECTED_HF_TOTAL_SIZE_BYTES = 166_878_536_440

SCALE_SUFFIX = ".scale"
_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.*)$")

# mHC parameters whose names contain "scale" but are NOT block scales.
HC_PARAM_SUFFIXES = (
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
)


def is_mtp_key(key: str, *, mtp_start_layer: int = 43) -> bool:
    """MTP module keys carry a layer index >= num_hidden_layers.

    HF snapshot 7872f01b's checkpoint keeps MTP under ``layers.43....``
    (num_nextn_predict_layers=1).  The DSpark speculative-decode block
    is a separate module identified by ``dspark_*`` prefixes; both are
    dropped in the converter (no spec-decode).
    """
    match = _LAYER_RE.match(key)
    return bool(match) and int(match.group(1)) >= mtp_start_layer


def is_dspark_key(key: str) -> bool:
    """DSpark speculative-decode submodule prefixes; unconditionally dropped."""
    return "dspark" in key


def is_block_scale_key(key: str) -> bool:
    """True only for real UE8M0 block scales; guards against mHC ``*_scale``."""
    if key.endswith(HC_PARAM_SUFFIXES):
        return False
    return key.endswith(SCALE_SUFFIX)


def dequantize_block_fp8_ue8m0(
    weight_fp8: torch.Tensor,
    scale_exp: torch.Tensor,
    block_size: tuple[int, int],
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Blockwise FP8-e4m3 -> ``out_dtype`` using UE8M0 (unsigned E8M0) block scales.

    ``weight_fp8`` is ``[..., out, in]`` in ``float8_e4m3fn``.  ``scale_exp``
    is ``[..., ceil(out/bo), ceil(in/bi)]`` in an integer dtype holding
    UE8M0 exponents (values 0..255).  Per-tile multiplier is ``2 **
    scale_exp[o0, i0]``.  Shape agreement is asserted rather than
    broadcast-guessed.

    Correctness contract (paper §2.4, "Dequantization"):
      value = fp8_to_fp32(w) * (2 ** exp)

    Direction of the scale is fixed — this is NOT a reciprocal.  A caller
    that inverts it will see (2**-exp) for a positive exponent and
    silently produce an underflowed all-zero tile.
    """
    if scale_exp is None:
        raise ValueError(
            "dequantize_block_fp8_ue8m0 requires an explicit non-None block-scale"
        )
    validate_ue8m0_scale(scale_exp, "block_scale")
    out_features, in_features = weight_fp8.shape[-2], weight_fp8.shape[-1]
    block_out, block_in = block_size
    expected = (
        math.ceil(out_features / block_out),
        math.ceil(in_features / block_in),
    )
    if tuple(scale_exp.shape[-2:]) != expected:
        raise ValueError(
            f"block-scale shape {tuple(scale_exp.shape)} disagrees with "
            f"weight {tuple(weight_fp8.shape)} under block_size={block_size}; "
            f"expected {expected}. Refusing to broadcast-guess."
        )
    value = weight_fp8.to(torch.float32)
    exp = scale_exp.to(torch.int32)
    # UE8M0 -> multiplier: 2**exp.  Use ldexp for numerical exactness (a
    # single integer bit-shift of the exponent field, no rounding).
    ones = torch.ones_like(exp, dtype=torch.float32)
    scale = torch.ldexp(ones, exp)
    # Expand each block scalar over its tile, then trim the ragged edge.
    scale = scale.repeat_interleave(block_out, dim=-2).repeat_interleave(
        block_in, dim=-1
    )[..., :out_features, :in_features]
    return (value * scale).to(out_dtype)


def dequantize_block_fp4_ue8m0(
    weight_fp4_packed: torch.Tensor,
    scale_exp: torch.Tensor,
    block_size: tuple[int, int],
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Blockwise FP4 (MXFP4-style, 2 nibbles per byte) -> ``out_dtype``.

    NOT YET IMPLEMENTED — pending decision on packing layout.  The HF
    checkpoint's exact FP4 packing (nibble-lo-first vs lo-hi-interleaved
    across the tile) is verified only by the 1-tensor smoke against a
    real routed-expert shard.  Until that lands, the FP4 pathway is a
    fail-loud stub.  See ``smoke_round1_one_tensor.py`` for the
    verification harness.

    Reference for the packing convention: ``vllm_neuron/model/gpt_oss/
    weight_loaders_mxfp4.py:_pack_fp4_x4_uint16`` and
    ``mxfp4_gate_up_blocks_loader`` — that packs at load time, so the
    inverse is what we need here at converter time.
    """
    raise NotImplementedError(
        "dequantize_block_fp4_ue8m0 is deferred to Round 2; see the module "
        "docstring for the paper-cited dequant formula and the packing-"
        "layout verification plan in smoke_round1_one_tensor.py"
    )


def _dequant_or_cast(
    state_dict: dict[str, Any],
    key: str,
    block_size: tuple[int, int],
    out_dtype: torch.dtype,
    *,
    dtype_hint: str = "fp8",
) -> torch.Tensor | None:
    """Fetch a possibly-quantized tensor and materialise it as ``out_dtype``.

    ``dtype_hint`` selects the dequant primitive:
      * "fp8"  -> FP8 e4m3 blockwise with UE8M0 scale.
      * "fp4"  -> FP4 packed blockwise with UE8M0 scale (routed-expert weights).
      * "bf16" / "fp32" -> straight cast, no scale required.

    A tensor whose HF dtype is FP8/FP4 but which has no paired ``.scale``
    is rejected loudly.
    """
    weight = state_dict.get(key)
    if weight is None:
        return None
    if dtype_hint in ("bf16", "fp32"):
        if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError(
                f"{key} is FP8 but declared bf16/fp32 in the layout map"
            )
        return weight.to(out_dtype)
    scale = state_dict.get(f"{key}{SCALE_SUFFIX}")
    if scale is None:
        raise ValueError(
            f"{key} declared as {dtype_hint} but has no paired {key}{SCALE_SUFFIX}"
        )
    if dtype_hint == "fp8":
        return dequantize_block_fp8_ue8m0(weight, scale, block_size, out_dtype)
    if dtype_hint == "fp4":
        return dequantize_block_fp4_ue8m0(weight, scale, block_size, out_dtype)
    raise ValueError(f"unknown dtype_hint {dtype_hint!r}")


def _convert_dsv4_checkpoint(
    state_dict: dict[str, Any],
    src: DeepseekV4FlashInferenceConfig,
    *,
    tp_degree: int,
) -> dict[str, Any]:
    """Map HF names onto the wrapper's module tree.

    NOT YET IMPLEMENTED for the per-layer path — this scaffold routes
    the top-level embed/lm_head/final_norm tensors so a 1-layer synthetic
    smoke can walk the code without hitting the per-layer stubs; per-
    layer conversion is gated behind Round 2.  Every HF -> wrapper key
    decision lands in a dedicated ``_convert_*_layer`` helper matching
    the GLM-5.3-Flash structure at
    ``vllm_neuron/model/glm53_flash/checkpoint_convert.py:415-593``.

    Structural decisions (each with a top-of-file citation):
      * Layer 43 (MTP) and all ``dspark_*`` tensors dropped explicitly.
      * Non-expert FP8 weights dequantized to bf16 here (UE8M0 scale).
      * Routed-expert FP4 weights dequantized to bf16 here (UE8M0 scale)
        AFTER the packing layout is verified byte-clean against a real
        routed-expert shard (Round-2 pre-req; see 1-tensor smoke).
      * Grouped output projection: ``wo_a.weight`` is a 3-D tensor
        ``[o_groups, num_heads * head_dim, o_lora_rank]`` before flattening
        into NxDI's ColumnParallel primitive; the group axis is treated
        as an extra head-count axis with ``o_groups * o_lora_rank`` as
        the output dim of ``o_b``.  Confirmed against transformers
        5.15.1 ``DeepseekV4GroupedLinear`` (paper §2.3.1).
      * Hash-MoE bootstrap: the frozen ``tid2eid[input_ids]`` lookup
        table sits alongside the routed-expert weights in the checkpoint
        as ``layers.<0..2>.mlp.hash_table`` int32 buffer.  Copied through
        as an ``nn.Parameter(requires_grad=False)`` in the wrapper.
    """
    if tp_degree <= 0:
        raise ValueError(f"tp_degree must be positive; got {tp_degree}")

    dtype = src.torch_dtype
    block_size = tuple(src.quantization_config.weight_block_size)
    num_layers = src.num_hidden_layers

    converted: dict[str, Any] = {}
    dropped_mtp: list[str] = []
    dropped_dspark: list[str] = []

    # ---- top-level tensors (embed, norm, lm_head) ----
    if EMBED_KEY not in state_dict:
        raise ValueError(f"missing {EMBED_KEY!r} in state_dict")
    converted["embed_tokens.weight"] = state_dict[EMBED_KEY].to(dtype)
    if FINAL_NORM_KEY not in state_dict:
        raise ValueError(f"missing {FINAL_NORM_KEY!r} in state_dict")
    converted["final_norm_weight"] = state_dict[FINAL_NORM_KEY].to(dtype)
    if LM_HEAD_KEY in state_dict:
        converted["lm_head.weight"] = state_dict[LM_HEAD_KEY].to(dtype)
    elif getattr(src, "tie_word_embeddings", False):
        converted["lm_head.weight"] = converted["embed_tokens.weight"]
    else:
        raise ValueError(
            f"missing {LM_HEAD_KEY!r} in state_dict and tie_word_embeddings=False"
        )

    # ---- per-layer conversion (Round 2 lands the block helpers) ----
    for layer_idx in range(num_layers):
        raise NotImplementedError(
            f"per-layer conversion is Round 2 — layer {layer_idx} would dispatch on "
            f"layer_types[{layer_idx}]={src.layer_types[layer_idx]!r} + "
            f"mlp_layer_types[{layer_idx}]={src.mlp_layer_types[layer_idx]!r}. "
            "See checkpoint_convert.py top-of-file for the structural decision list."
        )

    # ---- dropped-tensor bookkeeping (unreachable until per-layer lands) ----
    for key in state_dict.keys():  # pragma: no cover - unreachable in Round 1
        if is_mtp_key(key):
            dropped_mtp.append(key)
        elif is_dspark_key(key):
            dropped_dspark.append(key)

    converted["_conversion_report"] = {
        "input_tensor_count": len(state_dict),
        "expected_input_tensor_count": EXPECTED_HF_TENSOR_COUNT,
        "converted_tensor_count": len(converted) - 1,
        "dropped_mtp_tensors": len(dropped_mtp),
        "dropped_dspark_tensors": len(dropped_dspark),
        "block_size": block_size,
        "hf_snapshot_sha": HF_SNAPSHOT_SHA,
        "tp_degree": tp_degree,
        "mtp_layer_excluded_from": src.num_hidden_layers,
    }
    return converted


__all__ = [
    "EMBED_KEY",
    "EXPECTED_HF_TENSOR_COUNT",
    "EXPECTED_HF_TOTAL_SIZE_BYTES",
    "FINAL_NORM_KEY",
    "HC_PARAM_SUFFIXES",
    "LM_HEAD_KEY",
    "SCALE_SUFFIX",
    "TEXT_LAYER_PREFIX",
    "_convert_dsv4_checkpoint",
    "_dequant_or_cast",
    "dequantize_block_fp4_ue8m0",
    "dequantize_block_fp8_ue8m0",
    "is_block_scale_key",
    "is_dspark_key",
    "is_mtp_key",
]
