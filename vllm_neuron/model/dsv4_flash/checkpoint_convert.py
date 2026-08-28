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
    ue8m0_scale_to_fp32_multiplier,
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

# ---------------------------------------------------------------------------
# FP4-E2M1 codebook (source-cited).
#
# Copied byte-for-byte from the DeepSeek-V4-Flash HF inference reference
# ``inference/convert.py::FP4_TABLE`` (repo
# ``deepseek-ai/DeepSeek-V4-Flash-0731`` @ HF SHA
# ``7872f01b1d1fe23eabc4c98b48bffcef5a386062``, lines 11-14).  The table is
# the standard OCP MXFP4 / IEEE-style FP4-E2M1 codebook (1 sign bit, 2
# exponent bits, 1 mantissa bit; bias=1) — magnitudes ``{0, ±0.5, ±1.0,
# ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}``.  Nibble bit layout inside each stored
# byte, mirroring ``inference/convert.py:30-33``::
#
#     x_uint8 = stored_byte
#     low_nibble  = x_uint8 & 0x0F          # first FP4 value (lowest along K)
#     high_nibble = (x_uint8 >> 4) & 0x0F   # second FP4 value (next along K)
#     out = stack([TABLE[low], TABLE[high]], dim=-1).flatten(-2)  # doubles last dim
#
# Positive codes 0..7 are the first half, negative codes 8..15 are the
# second half (sign bit = MSB of the nibble).
_FP4_E2M1_TABLE = (
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

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


def _broadcast_block_scale(
    scale_fp32: torch.Tensor,
    block_size: tuple[int, int],
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Expand a ``[..., ceil(O/bo), ceil(I/bi)]`` fp32 block-scale
    tensor to ``[..., O, I]`` by repeat-interleave over each tile axis
    and then trimming the ragged edge.
    """
    block_out, block_in = block_size
    return scale_fp32.repeat_interleave(block_out, dim=-2).repeat_interleave(
        block_in, dim=-1
    )[..., :out_features, :in_features]


def dequantize_block_fp8_ue8m0(
    weight_fp8: torch.Tensor,
    scale_exp: torch.Tensor,
    block_size: tuple[int, int],
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Blockwise FP8-e4m3 -> ``out_dtype`` using UE8M0 (unsigned E8M0) block scales.

    ``weight_fp8`` is ``[..., out, in]`` in ``float8_e4m3fn``.  ``scale_exp``
    is ``[..., ceil(out/bo), ceil(in/bi)]`` in either ``float8_e8m0fnu``
    (raw UE8M0 storage) or an integer dtype carrying the same raw code.
    Per-tile multiplier is ``2 ** (raw_code - 127)`` — the standard E8M0
    bias.  Shape agreement is asserted rather than broadcast-guessed.

    Correctness contract (DeepSeek-V4-Flash HF inference reference
    ``inference/kernel.py::fp8_gemm_kernel`` @ HF SHA
    ``7872f01b1d1fe23eabc4c98b48bffcef5a386062`` lines 244, 249: the
    accumulator is ``C_local[i,j] * scale_a * scale_b`` where each scale
    is the ``.to(FP32)`` of the E8M0 tensor)::

        value = fp8_to_fp32(w) * ue8m0_to_fp32(scale)

    Direction of the scale is fixed — this is NOT a reciprocal.  A caller
    that inverts it will see ``2**-(exp-127)`` for a positive exponent
    and silently produce an underflowed all-zero tile.
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
    scale_fp32 = ue8m0_scale_to_fp32_multiplier(scale_exp)
    scale = _broadcast_block_scale(
        scale_fp32, block_size, out_features, in_features
    )
    return (value * scale).to(out_dtype)


def _fp4_codebook(device: torch.device) -> torch.Tensor:
    """Return the FP4-E2M1 codebook (16 values) as fp32 on ``device``.

    Kept as a module-level function so the tensor is allocated once per
    device.  Ordering matches ``_FP4_E2M1_TABLE`` — do NOT re-order.
    """
    return torch.tensor(_FP4_E2M1_TABLE, dtype=torch.float32, device=device)


def dequantize_block_fp4_ue8m0(
    weight_fp4_packed: torch.Tensor,
    scale_exp: torch.Tensor,
    block_size: tuple[int, int],
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Blockwise FP4-E2M1 (packed 2 nibbles/byte) -> ``out_dtype`` using
    UE8M0 (unsigned E8M0) block scales.

    ``weight_fp4_packed`` is ``[..., out, in_bytes]`` in one of
    ``{torch.int8, torch.uint8, torch.float4_e2m1fn_x2}``.  The logical
    FP4 shape is ``[..., out, in_bytes * 2]`` — packing is along the LAST
    dim (K/in), low nibble first (bits 0..3 = index into
    :data:`_FP4_E2M1_TABLE`), high nibble second (bits 4..7).  This is
    the DeepSeek-V4-Flash checkpoint convention, verified byte-for-byte
    against the HF inference reference (``inference/convert.py::
    cast_e2m1fn_to_e4m3fn`` @ HF SHA ``7872f01b...``, lines 30-33)::

        x       = weight.view(torch.uint8)          # [..., O, in_bytes]
        low     = x & 0x0F                          # nibble at bit 0..3
        high    = (x >> 4) & 0x0F                   # nibble at bit 4..7
        fp4vals = stack([TABLE[low], TABLE[high]], -1).flatten(-2)  # [..., O, 2*in_bytes]

    ``scale_exp`` is ``[..., ceil(out/bo), ceil(in_logical/bi)]`` in
    either ``float8_e8m0fnu`` (the HF storage format — verified against
    ``layers.3.ffn.experts.0.w2.scale`` in ``model-00005-of-00048.safe-
    tensors`` — every entry decodes to a finite positive multiplier via
    ``.to(torch.float32)``) or an integer dtype carrying the same raw
    E8M0 code.  Per-tile multiplier is ``2 ** (raw_code - 127)`` — same
    bias as the FP8 path.

    Correctness contract (DeepSeek-V4-Flash HF inference reference
    ``inference/model.py:18``: ``fp4_block_size = 32``; and
    ``inference/kernel.py::fp4_gemm_kernel`` lines 500-509: the FP4
    weight scale is cast to fp32 and multiplied into the accumulator)::

        w_fp32   = FP4_E2M1_TABLE[nibble]
        scale_m  = ue8m0_to_fp32(scale)
        bf16_val = (w_fp32 * scale_m).to(bfloat16)

    For DSv4-Flash routed experts the block shape is ``(1, 32)`` — per
    output row × per 32 logical input elements.  The function accepts
    arbitrary ``block_size`` for offline verification; the caller
    supplies ``(1, 32)`` for real weights (the top-level
    ``quantization_config.weight_block_size`` field is ``(128, 128)``
    which describes the NON-EXPERT FP8 pathway, not the routed-expert
    FP4 pathway — the two are distinct even though the code lives in the
    same converter).

    Fail-loud checks (each is a silent-corruption defence):
      * scale must be a UE8M0 tensor with no NaN (raw byte 255) — see
        :func:`validate_ue8m0_scale`;
      * weight dtype must be one of the three listed above — a bf16
        weight would silently reinterpret its low byte as a nibble
        packet and produce plausible garbage;
      * ``in_bytes * 2`` must be an exact multiple of ``block_in`` —
        anything else means the caller pasted an unexpected packing;
      * scale shape must match ``ceil(out/bo), ceil(in_logical/bi)`` —
        we refuse to broadcast-guess a mismatched scale.
    """
    if scale_exp is None:
        raise ValueError(
            "dequantize_block_fp4_ue8m0 requires an explicit non-None block-scale"
        )
    if weight_fp4_packed.dtype not in (
        torch.int8,
        torch.uint8,
        torch.float4_e2m1fn_x2,
    ):
        raise TypeError(
            "dequantize_block_fp4_ue8m0 refuses weight dtype "
            f"{weight_fp4_packed.dtype}; expected int8/uint8/float4_e2m1fn_x2."
        )
    if weight_fp4_packed.ndim < 2:
        raise ValueError(
            f"weight must have at least 2 dims (got shape "
            f"{tuple(weight_fp4_packed.shape)})"
        )
    validate_ue8m0_scale(scale_exp, "block_scale")

    out_features = int(weight_fp4_packed.shape[-2])
    in_bytes = int(weight_fp4_packed.shape[-1])
    in_features = in_bytes * 2
    block_out, block_in = block_size
    if block_out <= 0 or block_in <= 0:
        raise ValueError(f"block_size must be positive; got {block_size}")
    if in_features % block_in != 0:
        raise ValueError(
            f"in_features={in_features} (2 * {in_bytes}) is not a multiple "
            f"of block_in={block_in}; refusing to over-broadcast a partial tile."
        )
    expected_scale_tail = (
        math.ceil(out_features / block_out),
        math.ceil(in_features / block_in),
    )
    if tuple(scale_exp.shape[-2:]) != expected_scale_tail:
        raise ValueError(
            f"block-scale shape {tuple(scale_exp.shape)} disagrees with "
            f"packed weight {tuple(weight_fp4_packed.shape)} "
            f"(logical in={in_features}) under block_size={block_size}; "
            f"expected trailing shape {expected_scale_tail}. Refusing "
            "to broadcast-guess."
        )

    # 1. Unpack nibbles.  ``.view(torch.uint8)`` reinterprets the storage
    #    without a copy for int8/uint8/float4_e2m1fn_x2 — element_size==1
    #    holds for all three.
    x_uint8 = weight_fp4_packed.view(torch.uint8)
    low = x_uint8 & 0x0F
    high = (x_uint8 >> 4) & 0x0F
    # Sanity: with 0x0F mask the nibbles are already in [0, 15]; assert
    # the invariant so a future storage change is caught immediately.
    if low.max().item() > 15 or high.max().item() > 15:
        raise ValueError(
            "internal invariant broken: unpacked nibble > 15 after 0x0F "
            "mask; check the storage dtype path."
        )
    codebook = _fp4_codebook(x_uint8.device)
    low_vals = codebook[low.long()]
    high_vals = codebook[high.long()]
    # Interleave along the last dim: [..., O, in_bytes, 2] -> [..., O, in_features]
    fp4_fp32 = torch.stack([low_vals, high_vals], dim=-1).flatten(-2)

    # 2. Decode the UE8M0 block scale to fp32 multiplier.
    scale_fp32 = ue8m0_scale_to_fp32_multiplier(scale_exp)

    # 3. Broadcast the block scale over its tile and multiply.
    scale = _broadcast_block_scale(
        scale_fp32, block_size, out_features, in_features
    )
    result = fp4_fp32 * scale

    # 4. Cast to the requested output dtype.  For bf16 the cast is lossless
    #    across the entire FP4 codebook × any single-bit E8M0 multiplier
    #    (the product is either 0 or of the form ``m * 2**k`` with m in
    #    {1, 1.5, 3} for magnitudes 6.0 → the bf16 mantissa carries it
    #    exactly).  For fp16 / fp32 the caller opts in explicitly.
    return result.to(out_dtype)


# ---------------------------------------------------------------------------
# HF DSv4-Flash routed-MoE checkpoint layout (source-cited).
#
# Verified against ``model.safetensors.index.json`` of HF snapshot
# ``deepseek-ai/DeepSeek-V4-Flash-0731 @ 7872f01b1d1fe23eabc4c98b48bffcef5a386062``
# (72,317 total keys; 11,776 keys each for w1/w2/w3 = 3 * 256 experts *
# ~46 layers).
#
# Router-only keys (per routed-MoE layer i >= num_hash_layers):
#     layers.<i>.ffn.gate.weight    : [n_routed_experts=256, hidden=4096]
#     layers.<i>.ffn.gate.bias      : [n_routed_experts=256] fp32
#         `.bias` is the `e_score_correction_bias` from
#         `DeepseekV4TopKRouter` (`modeling_deepseek_v4.py:1042`);
#         DeepSeek's on-disk `.pt` -> safetensors converter renames it
#         `e_score_correction_bias -> bias`.  There is no additive bias in
#         the `nn.Linear(bias=False)` router — the field is the
#         selection-only correction.
#
# Shared-expert keys (identical layout on every MoE layer, hash or routed):
#     layers.<i>.ffn.shared_experts.w1.{weight, scale}  -> gate_proj
#     layers.<i>.ffn.shared_experts.w3.{weight, scale}  -> up_proj
#     layers.<i>.ffn.shared_experts.w2.{weight, scale}  -> down_proj
#
# Routed-expert keys (per expert e in [0, n_routed_experts)):
#     layers.<i>.ffn.experts.<e>.w1.{weight, scale}  -> gate  [I, H]
#     layers.<i>.ffn.experts.<e>.w3.{weight, scale}  -> up    [I, H]
#     layers.<i>.ffn.experts.<e>.w2.{weight, scale}  -> down  [H, I]
#     (w1/w3 stored FP4-E2M1 packed 2 nibbles / byte along K axis;
#      w2 same.  Scale is UE8M0 `float8_e8m0fnu`, block (1, 32) on K.)
#
# The ``w1/w2/w3`` naming comes from DeepSeek's original inference
# reference (repo ``deepseek-ai/DeepSeek-V4-Flash-0731`` at
# ``inference/model.py``); the transformers HF wrapper renames it at load
# time to ``gate_proj/down_proj/up_proj``, and further packs w1+w3 into a
# single ``gate_up_proj`` per-expert 3-D parameter (see
# ``modeling_deepseek_v4.py:1001``, ``DeepseekV4Experts.gate_up_proj``:
# ``nn.Parameter(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim))``).
# We do the same fuse in the converter, and additionally transpose to
# NxDI's ``[E, hidden, 2*I]`` shape convention (identical structural
# layout to GLM-5.3-Flash Round 5, see
# ``glm53_flash/checkpoint_convert.py::_convert_moe_layer``).
# ---------------------------------------------------------------------------

FFN_PREFIX = "ffn."
ROUTED_EXPERTS_SUBTREE = "experts"
SHARED_EXPERTS_SUBTREE = "shared_experts"
ROUTER_KEY = "ffn.gate.weight"
ROUTER_CORRECTION_BIAS_KEY = "ffn.gate.bias"
# HF DSv4-Flash routed-expert w1/w2/w3 FP4 block shape (K axis, per-row).
DSV4_FP4_BLOCK_SIZE: tuple[int, int] = (1, 32)


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


def _dsv4_scale_key_for(weight_key: str) -> str:
    """Return the HF scale key paired with a DSv4-Flash weight key.

    HF DSv4-Flash convention (verified against the safetensors index of
    snapshot ``deepseek-ai/DeepSeek-V4-Flash-0731 @ 7872f01b1d1fe...``):
    every quantized weight ``<base>.<name>.weight`` has its block scale
    stored as a sibling ``<base>.<name>.scale`` — NOT
    ``<base>.<name>.weight.scale`` (which is what
    :func:`_dequant_or_cast` expects for GLM-5.3-Flash's
    ``_scale_inv`` convention).  Getting this wrong is a silent
    correctness failure: the weight would land at ``2**0=1`` scale
    across every tile instead of the real UE8M0 exponent.
    """
    if not weight_key.endswith(".weight"):
        raise ValueError(
            f"DSv4-Flash scale-key computation expects a '.weight' suffix; "
            f"got {weight_key!r}"
        )
    return weight_key[: -len(".weight")] + SCALE_SUFFIX


def _dequant_expert_fp4_weight(
    state_dict: dict[str, Any],
    key: str,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize one FP4-UE8M0 routed-expert tensor to ``out_dtype``.

    Wraps :func:`dequantize_block_fp4_ue8m0` at the DSv4-Flash-frozen
    ``block_size=(1, 32)``.  Every silent-corruption guard the primitive
    ships (weight-dtype, block-scale shape, NaN scale) is inherited.  A
    missing weight OR scale raises loudly — a partial routed expert is a
    correctness bug, not a "soft skip" case.

    Uses the DSv4 scale-naming convention via :func:`_dsv4_scale_key_for`.
    """
    weight = state_dict.get(key)
    if weight is None:
        raise KeyError(f"missing routed-expert weight {key!r}")
    scale_key = _dsv4_scale_key_for(key)
    scale = state_dict.get(scale_key)
    if scale is None:
        raise KeyError(
            f"missing paired UE8M0 block-scale {scale_key!r} for {key!r}"
        )
    return dequantize_block_fp4_ue8m0(
        weight, scale, DSV4_FP4_BLOCK_SIZE, out_dtype
    )


def _dequant_shared_fp8_weight(
    state_dict: dict[str, Any],
    key: str,
    block_size: tuple[int, int],
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize one FP8-UE8M0 shared-expert (or other non-routed) tensor.

    Same DSv4 scale-naming convention as
    :func:`_dequant_expert_fp4_weight`, dispatched to
    :func:`dequantize_block_fp8_ue8m0` (native-E4M3 path with UE8M0
    scale).  Missing weight OR scale raises loudly.
    """
    weight = state_dict.get(key)
    if weight is None:
        raise KeyError(f"missing shared/non-routed weight {key!r}")
    scale_key = _dsv4_scale_key_for(key)
    scale = state_dict.get(scale_key)
    if scale is None:
        raise KeyError(
            f"missing paired UE8M0 block-scale {scale_key!r} for {key!r}"
        )
    return dequantize_block_fp8_ue8m0(weight, scale, block_size, out_dtype)


def _convert_shared_expert(
    state_dict: dict[str, Any],
    converted: dict[str, Any],
    hf_base: str,
    wrapper_target: str,
    dtype: torch.dtype,
    block_size_fp8: tuple[int, int],
) -> None:
    """Shared-expert path: dequant + rename w1/w3/w2 -> gate/up/down.

    Same layout convention as ``DeepseekV4MLP`` (line 974 in the HF
    modeling file).  Weight naming matches DeepSeek's inference reference:
    ``w1 -> gate_proj``, ``w3 -> up_proj``, ``w2 -> down_proj``.  Shared
    experts are FP8-e4m3 with UE8M0 block scale at block ``(128, 128)``
    (the ``quantization_config.weight_block_size`` on the top-level
    config), NOT the routed FP4 block ``(1, 32)`` — this is why the
    converter carries two block-size arguments.
    """
    hf_map = (
        ("w1", "gate_proj"),
        ("w3", "up_proj"),
        ("w2", "down_proj"),
    )
    for hf_name, wrapper_name in hf_map:
        key = f"{hf_base}{FFN_PREFIX}{SHARED_EXPERTS_SUBTREE}.{hf_name}.weight"
        tensor = _dequant_shared_fp8_weight(
            state_dict, key, block_size_fp8, dtype
        )
        converted[f"{wrapper_target}shared_expert.{wrapper_name}.weight"] = (
            tensor
        )


def _convert_routed_moe_layer(
    state_dict: dict[str, Any],
    converted: dict[str, Any],
    layer_idx: int,
    src: DeepseekV4FlashInferenceConfig,
    *,
    hf_prefix: str = "",
    wrapper_prefix: str = "",
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Convert one DSv4-Flash routed-MoE layer HF -> wrapper module tree.

    This is the routed-MoE per-layer helper.  It is safe to invoke on
    layers ``[num_hash_layers, num_hidden_layers)``; the caller decides
    whether the schedule permits it (the frozen
    ``src.mlp_layer_types[layer_idx]`` MUST equal ``"moe"``).

    Structure it produces (for one layer i):

      * ``{wrapper_prefix}layers.<i>.mlp.router.weight``            fp32
      * ``{wrapper_prefix}layers.<i>.mlp.e_score_correction_bias``  fp32
      * ``{wrapper_prefix}layers.<i>.mlp.shared_expert.gate_proj.weight``  dtype
      * ``{wrapper_prefix}layers.<i>.mlp.shared_expert.up_proj.weight``    dtype
      * ``{wrapper_prefix}layers.<i>.mlp.shared_expert.down_proj.weight``  dtype
      * ``{wrapper_prefix}layers.<i>.mlp.expert_mlps.mlp_op.gate_up_proj.weight``
            shape ``[n_routed_experts, hidden, 2 * moe_intermediate_size]``
            stride=2 fused ``[gate_full | up_full]`` on last axis
      * ``{wrapper_prefix}layers.<i>.mlp.expert_mlps.mlp_op.down_proj.weight``
            shape ``[n_routed_experts, moe_intermediate_size, hidden]``

    Also returns a per-layer conversion report keyed under
    ``_routed_moe_report`` on the ``converted`` dict for debugging and
    the per-layer smoke — the report is a plain dict, not a tensor, so it
    never lands in NxDI's state_dict traversal.

    The per-expert dequant is FP4-UE8M0 at block ``(1, 32)`` on the K
    axis (per :func:`dequantize_block_fp4_ue8m0`) — bit-exact against the
    HF inference reference at
    ``inference/kernel.py::fp4_gemm_kernel`` @ SHA ``7872f01b1d1fe...``.

    NxDI's ExpertMLPs weight layout is ``[E, hidden, 2*I]`` for
    ``gate_up_proj`` and ``[E, I, hidden]`` for ``down_proj`` — same as
    GLM-5.3-Flash Round 5.
    """
    dtype = dtype if dtype is not None else src.torch_dtype
    if src.mlp_layer_types[layer_idx] != "moe":
        raise ValueError(
            f"_convert_routed_moe_layer called for layer {layer_idx} but "
            f"mlp_layer_types[{layer_idx}]="
            f"{src.mlp_layer_types[layer_idx]!r} — refusing to route a "
            "hash-MoE layer through the routed-MoE converter."
        )
    hidden = src.hidden_size
    inter = src.moe_intermediate_size
    n_experts = src.n_routed_experts
    block_fp8 = tuple(src.quantization_config.weight_block_size)

    hf_base = f"{hf_prefix}layers.{layer_idx}."
    target = f"{wrapper_prefix}layers.{layer_idx}.mlp."

    # ---- router + correction bias ----
    router_key = f"{hf_base}{ROUTER_KEY}"
    router = state_dict.get(router_key)
    if router is None:
        raise KeyError(f"missing router weight {router_key!r}")
    if tuple(router.shape) != (n_experts, hidden):
        raise ValueError(
            f"router weight {router_key!r} shape {tuple(router.shape)} "
            f"disagrees with expected ({n_experts}, {hidden})"
        )
    # Router lives in fp32 in the wrapper — cast at conversion so no cast
    # is inserted at forward time.
    converted[f"{target}router.weight"] = router.to(torch.float32)

    corr_key = f"{hf_base}{ROUTER_CORRECTION_BIAS_KEY}"
    corr = state_dict.get(corr_key)
    if corr is None:
        # A missing correction bias degrades gracefully to top-k on the raw
        # scores; the wrapper declares the parameter with a zero default so
        # this is a documented soft fallback.  Record it in the report.
        e_bias_present = False
    else:
        if tuple(corr.shape) != (n_experts,):
            raise ValueError(
                f"correction bias {corr_key!r} shape {tuple(corr.shape)} "
                f"disagrees with expected ({n_experts},)"
            )
        converted[f"{target}e_score_correction_bias"] = corr.to(torch.float32)
        e_bias_present = True

    # ---- shared expert (FP8-UE8M0, block (128, 128)) ----
    _convert_shared_expert(state_dict, converted, hf_base, target, dtype, block_fp8)

    # ---- routed experts (FP4-UE8M0, block (1, 32)) ----
    #
    # HF stores per expert:
    #   w1 (gate): [inter, hidden]      packed FP4 -> [inter, hidden/2] int8
    #   w3 (up)  : [inter, hidden]      packed FP4 -> [inter, hidden/2] int8
    #   w2 (down): [hidden, inter]      packed FP4 -> [hidden, inter/2] int8
    #
    # We dequant each to bf16 (or `dtype`) at native shape, then stack across
    # experts and fuse gate|up on the intermediate axis, then transpose to
    # [E, hidden, 2*I] / [E, I, hidden].
    #
    # Materialising all 256 experts into one stacked tensor per layer is
    # ~4.0 GiB (gate_up) + ~2.0 GiB (down) at bf16 — smaller than GLM's
    # 288-expert case because DSv4's moe_intermediate_size (2048) is the
    # same and n_experts is one fewer.  Streaming would still be Round 3
    # material; for the per-layer smoke we do it in-memory.
    gate_stack: list[torch.Tensor] = []
    up_stack: list[torch.Tensor] = []
    down_stack: list[torch.Tensor] = []
    for e in range(n_experts):
        base_e = f"{hf_base}{FFN_PREFIX}{ROUTED_EXPERTS_SUBTREE}.{e}."
        gate = _dequant_expert_fp4_weight(
            state_dict, f"{base_e}w1.weight", dtype
        )
        up = _dequant_expert_fp4_weight(
            state_dict, f"{base_e}w3.weight", dtype
        )
        down = _dequant_expert_fp4_weight(
            state_dict, f"{base_e}w2.weight", dtype
        )
        if tuple(gate.shape) != (inter, hidden):
            raise ValueError(
                f"expert {e} w1 (gate) shape {tuple(gate.shape)} != "
                f"({inter}, {hidden}); layer {layer_idx}"
            )
        if tuple(up.shape) != (inter, hidden):
            raise ValueError(
                f"expert {e} w3 (up) shape {tuple(up.shape)} != "
                f"({inter}, {hidden}); layer {layer_idx}"
            )
        if tuple(down.shape) != (hidden, inter):
            raise ValueError(
                f"expert {e} w2 (down) shape {tuple(down.shape)} != "
                f"({hidden}, {inter}); layer {layer_idx}"
            )
        gate_stack.append(gate)
        up_stack.append(up)
        down_stack.append(down)

    # Stack along the new leading expert axis.
    gate_stacked = torch.stack(gate_stack, dim=0)      # [E, I, H]
    up_stacked = torch.stack(up_stack, dim=0)          # [E, I, H]
    down_stacked = torch.stack(down_stack, dim=0)      # [E, H, I]
    # Fuse gate|up on the intermediate axis, then transpose to [E, H, 2I].
    gate_up_stacked = torch.cat([gate_stacked, up_stacked], dim=1)  # [E, 2I, H]
    gate_up_stacked = gate_up_stacked.transpose(1, 2).contiguous()  # [E, H, 2I]
    down_stacked = down_stacked.transpose(1, 2).contiguous()        # [E, I, H]

    converted[f"{target}expert_mlps.mlp_op.gate_up_proj.weight"] = (
        gate_up_stacked
    )
    converted[f"{target}expert_mlps.mlp_op.down_proj.weight"] = down_stacked

    report = {
        "layer_idx": layer_idx,
        "n_routed_experts": n_experts,
        "hidden": hidden,
        "moe_intermediate": inter,
        "gate_up_shape": tuple(gate_up_stacked.shape),
        "down_shape": tuple(down_stacked.shape),
        "e_score_correction_bias_present": e_bias_present,
        "dtype": str(dtype),
        "block_size_fp4": DSV4_FP4_BLOCK_SIZE,
        "block_size_fp8_shared": block_fp8,
    }
    converted.setdefault("_routed_moe_reports", {})[layer_idx] = report
    return report


ATTN_PREFIX = "attn."
ATTN_NORM_KEY = "attn_norm.weight"
# MQA-block-owned parameter names (mirror of `_MQABlock.PARAM_KEYS` in
# neuron_wrapper.py — kept in sync at review time).  Every quantized
# entry is FP8 e4m3 with a UE8M0 block scale under the checkpoint's
# `quantization_config.weight_block_size = (128, 128)`.  The `.scale`
# tail is added by :func:`_dsv4_scale_key_for` and is present on every
# `.weight` in this list except the two RMSNorm gains and the sink,
# which are stored dense.
_MQA_FP8_WEIGHT_NAMES: tuple[str, ...] = (
    "wq_a.weight",
    "wq_b.weight",
    "wkv.weight",
    "wo_a.weight",
    "wo_b.weight",
)
_MQA_DENSE_NAMES: tuple[str, ...] = (
    "q_norm.weight",
    "kv_norm.weight",
    "attn_sink",
)


def _convert_mqa_block(
    state_dict: dict[str, Any],
    layer_idx: int,
    src: DeepseekV4FlashInferenceConfig,
    *,
    hf_prefix: str = "",
    wrapper_prefix: str = "",
    dtype: torch.dtype | None = None,
    require_attn_sink: bool = True,
) -> dict[str, Any]:
    """Convert one DSv4-Flash MQA attention block HF -> wrapper module tree.

    Reads the 8 `layers.<i>.attn.*` HF tensors (verified against
    `model.safetensors.index.json` of snapshot 7872f01b), dequants the
    five FP8-UE8M0 weights to `dtype` (bf16 by default), and carries the
    three dense tensors (`q_norm.weight`, `kv_norm.weight`, `attn_sink`)
    through unchanged.

    Also returns the sibling `layers.<i>.attn_norm.weight` (the
    pre-attention RMSNorm at the decoder-layer level) — it does not live
    inside the MQA block but is emitted alongside so a caller that owns
    the whole layer tree can drop the returned dict into `converted` in
    one call.

    Shape assertions match the wrapper's declared parameter shapes:

      * wq_a.weight       [q_lora_rank, hidden_size]
      * wq_b.weight       [num_heads * head_dim, q_lora_rank]
      * wkv.weight        [head_dim, hidden_size]           (single KV head)
      * wo_a.weight       [o_groups * o_lora_rank,
                           (num_heads * head_dim) // o_groups]
      * wo_b.weight       [hidden_size, o_groups * o_lora_rank]
      * q_norm.weight     [q_lora_rank]
      * kv_norm.weight    [head_dim]
      * attn_sink         [num_heads]
      * attn_norm.weight  [hidden_size]                     (sibling, not MQA)

    Fail-loud on any missing weight or scale — silently degrading to a
    zero-fill or an identity RMSNorm would produce plausible-looking
    logits that are quietly wrong.  `attn_sink` is a per-head learnable
    bias with a documented default of zeros; the checkpoint carries a
    trained value.  `require_attn_sink=True` refuses to fall back to
    zeros because the training run relies on it — set False only for
    smoke tests that deliberately zero the sink.
    """
    dtype = dtype if dtype is not None else src.torch_dtype
    hidden = int(src.hidden_size)
    num_heads = int(src.num_attention_heads)
    head_dim = int(src.head_dim)
    q_lora_rank = int(src.q_lora_rank)
    o_groups = int(src.o_groups)
    o_lora_rank = int(src.o_lora_rank)
    in_per_group = (num_heads * head_dim) // o_groups
    block_fp8 = tuple(src.quantization_config.weight_block_size)

    hf_base = f"{hf_prefix}layers.{layer_idx}.{ATTN_PREFIX}"
    layer_root = f"{hf_prefix}layers.{layer_idx}."
    target = f"{wrapper_prefix}layers.{layer_idx}.attn."

    expected_shapes: dict[str, tuple[int, ...]] = {
        "wq_a.weight": (q_lora_rank, hidden),
        "wq_b.weight": (num_heads * head_dim, q_lora_rank),
        "wkv.weight": (head_dim, hidden),
        "wo_a.weight": (o_groups * o_lora_rank, in_per_group),
        "wo_b.weight": (hidden, o_groups * o_lora_rank),
        "q_norm.weight": (q_lora_rank,),
        "kv_norm.weight": (head_dim,),
        "attn_sink": (num_heads,),
    }

    converted: dict[str, Any] = {}

    # ---- FP8-UE8M0 weights (dequant to dtype) ----
    for name in _MQA_FP8_WEIGHT_NAMES:
        hf_key = f"{hf_base}{name}"
        tensor = _dequant_shared_fp8_weight(state_dict, hf_key, block_fp8, dtype)
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(
                f"{hf_key!r} dequant shape {tuple(tensor.shape)} disagrees "
                f"with wrapper-expected {expected_shapes[name]}"
            )
        converted[f"{target}{name}"] = tensor

    # ---- dense tensors (norm gains + sink) ----
    for name in _MQA_DENSE_NAMES:
        hf_key = f"{hf_base}{name}"
        raw = state_dict.get(hf_key)
        if raw is None:
            if name == "attn_sink" and not require_attn_sink:
                # Documented soft fallback for smoke: a zero sink degrades
                # to plain softmax over the KV axis.
                raw = torch.zeros(num_heads, dtype=dtype)
            else:
                raise KeyError(f"missing {hf_key!r}")
        if tuple(raw.shape) != expected_shapes[name]:
            raise ValueError(
                f"{hf_key!r} shape {tuple(raw.shape)} disagrees with "
                f"wrapper-expected {expected_shapes[name]}"
            )
        converted[f"{target}{name}"] = raw.to(dtype)

    # ---- sibling: pre-attention RMSNorm at the decoder-layer level ----
    attn_norm_hf = f"{layer_root}{ATTN_NORM_KEY}"
    attn_norm_raw = state_dict.get(attn_norm_hf)
    if attn_norm_raw is None:
        raise KeyError(f"missing {attn_norm_hf!r}")
    if tuple(attn_norm_raw.shape) != (hidden,):
        raise ValueError(
            f"{attn_norm_hf!r} shape {tuple(attn_norm_raw.shape)} != ({hidden},)"
        )
    converted[f"{wrapper_prefix}layers.{layer_idx}.attn_norm.weight"] = (
        attn_norm_raw.to(dtype)
    )

    return converted


# ---------------------------------------------------------------------------
# HCA compressor subtree layout (source-cited).
#
# Verified against ``model.safetensors.index.json`` @ HF SHA
# ``7872f01b1d1fe23eabc4c98b48bffcef5a386062`` and shard header of
# ``model-00005-of-00048.safetensors`` (layer 3, the first HCA layer per the
# frozen ``compress_ratios`` schedule):
#
#   layers.3.attn.compressor.ape          F32  [128, 512]  (compress_rate, head_dim)
#   layers.3.attn.compressor.norm.weight  BF16 [512]       (head_dim)
#   layers.3.attn.compressor.wgate.weight BF16 [512, 4096] (head_dim, hidden_size)
#   layers.3.attn.compressor.wkv.weight   BF16 [512, 4096] (head_dim, hidden_size)
#
# All four are stored *dense* on disk (no ``.scale`` companion), so the
# converter carries them through with a straight dtype cast — the HCA
# compressor is not FP8-quantised in this snapshot.  This matches
# transformers' ``_keep_in_fp32_modules`` for ``self_attn.compressor.kv_proj``
# and ``.gate_proj`` (fp32-kept precision does NOT imply fp32 storage on
# disk — it's a compute-precision hint the wrapper can honour at compile
# time; our converter emits at the wrapper's module dtype, matching the
# _MQABlock convention).
# ---------------------------------------------------------------------------

COMPRESSOR_PREFIX = "compressor."
_HCA_COMPRESSOR_DENSE_NAMES: tuple[str, ...] = (
    "wkv.weight",
    "wgate.weight",
    "ape",
    "norm.weight",
)


def _convert_hca_block(
    state_dict: dict[str, Any],
    layer_idx: int,
    src: DeepseekV4FlashInferenceConfig,
    *,
    hf_prefix: str = "",
    wrapper_prefix: str = "",
    dtype: torch.dtype | None = None,
    require_attn_sink: bool = True,
) -> dict[str, Any]:
    """Convert one DSv4-Flash HCA attention block HF -> wrapper module tree.

    Fuses :func:`_convert_mqa_block` (the shared MQA attention subtree) with
    the HCA-specific 4-tensor compressor subtree that lives under
    ``layers.<i>.attn.compressor.*``.

    Contract on the caller:
      * ``src.layer_types[layer_idx]`` MUST equal
        ``"heavily_compressed_attention"`` — refusing to load a mismatched
        layer avoids silently mapping CSA weights (which live under the same
        ``compressor.`` subtree but with different shapes and an added
        ``indexer.`` sub-subtree) onto the HCA wrapper.
      * ``src.compress_ratios[layer_idx]`` MUST equal 128.

    Structure it produces (for one layer i under wrapper_prefix ""):

      * 8 entries from :func:`_convert_mqa_block` (attention subtree) —
        ``layers.<i>.attn.{wq_a.weight, wq_b.weight, q_norm.weight,
        wkv.weight, kv_norm.weight, wo_a.weight, wo_b.weight, attn_sink}``.
      * 1 sibling from :func:`_convert_mqa_block` (pre-attn RMSNorm) —
        ``layers.<i>.attn_norm.weight``.
      * 4 entries for the HCA compressor subtree —
        ``layers.<i>.attn.compressor.{wkv.weight, wgate.weight, ape,
        norm.weight}``.
      * Total: 13 tensors per layer.

    Shape assertions (fail-loud):

      * wkv.weight   [head_dim, hidden_size]
      * wgate.weight [head_dim, hidden_size]
      * ape          [compress_rate=128, head_dim]
      * norm.weight  [head_dim]

    Fail-loud on any missing tensor — an HCA layer without its compressor
    weights would silently degrade to plain MQA at that layer (all
    compressed KV entries would be uninitialised parameter noise multiplied
    by a random gate), producing plausible-looking logits that are quietly
    wrong.  Identical failure mode to the ``attn_sink`` guard on the MQA
    converter, escalated here to the 4 compressor tensors.
    """
    if src.layer_types[layer_idx] != "heavily_compressed_attention":
        raise ValueError(
            f"_convert_hca_block called for layer {layer_idx} but "
            f"layer_types[{layer_idx}]={src.layer_types[layer_idx]!r} — "
            "refusing to route a non-HCA layer through the HCA converter."
        )
    ratio = int(src.compress_ratios[layer_idx])
    if ratio != 128:
        raise ValueError(
            f"_convert_hca_block requires compress_ratios[{layer_idx}]=128; "
            f"got {ratio}."
        )
    dtype = dtype if dtype is not None else src.torch_dtype
    hidden = int(src.hidden_size)
    head_dim = int(src.head_dim)

    # 1. Delegate the MQA subtree + sibling attn_norm to the MQA converter.
    converted = _convert_mqa_block(
        state_dict,
        layer_idx,
        src,
        hf_prefix=hf_prefix,
        wrapper_prefix=wrapper_prefix,
        dtype=dtype,
        require_attn_sink=require_attn_sink,
    )

    # 2. HCA-specific 4-tensor compressor subtree.  Every entry is dense
    # on disk in the pinned snapshot (verified against shard 00005-of-00048
    # header — see source-cited layout comment above); we cast to the
    # wrapper's module dtype without dequant.
    hf_base = (
        f"{hf_prefix}layers.{layer_idx}.{ATTN_PREFIX}{COMPRESSOR_PREFIX}"
    )
    target = (
        f"{wrapper_prefix}layers.{layer_idx}.{ATTN_PREFIX}{COMPRESSOR_PREFIX}"
    )
    expected_compressor_shapes: dict[str, tuple[int, ...]] = {
        "wkv.weight": (head_dim, hidden),
        "wgate.weight": (head_dim, hidden),
        "ape": (ratio, head_dim),
        "norm.weight": (head_dim,),
    }
    for name in _HCA_COMPRESSOR_DENSE_NAMES:
        hf_key = f"{hf_base}{name}"
        raw = state_dict.get(hf_key)
        if raw is None:
            raise KeyError(
                f"missing HCA compressor tensor {hf_key!r} for layer "
                f"{layer_idx} — refusing to silently substitute zeros / "
                "random init and degrade to plain MQA at this layer."
            )
        expected = expected_compressor_shapes[name]
        if tuple(raw.shape) != expected:
            raise ValueError(
                f"{hf_key!r} shape {tuple(raw.shape)} disagrees with "
                f"wrapper-expected {expected}"
            )
        converted[f"{target}{name}"] = raw.to(dtype)

    return converted


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
    "ATTN_NORM_KEY",
    "ATTN_PREFIX",
    "COMPRESSOR_PREFIX",
    "DSV4_FP4_BLOCK_SIZE",
    "EMBED_KEY",
    "EXPECTED_HF_TENSOR_COUNT",
    "EXPECTED_HF_TOTAL_SIZE_BYTES",
    "FFN_PREFIX",
    "FINAL_NORM_KEY",
    "HC_PARAM_SUFFIXES",
    "LM_HEAD_KEY",
    "ROUTED_EXPERTS_SUBTREE",
    "ROUTER_CORRECTION_BIAS_KEY",
    "ROUTER_KEY",
    "SCALE_SUFFIX",
    "SHARED_EXPERTS_SUBTREE",
    "TEXT_LAYER_PREFIX",
    "_FP4_E2M1_TABLE",
    "_HCA_COMPRESSOR_DENSE_NAMES",
    "_MQA_DENSE_NAMES",
    "_MQA_FP8_WEIGHT_NAMES",
    "_convert_dsv4_checkpoint",
    "_convert_hca_block",
    "_convert_mqa_block",
    "_convert_routed_moe_layer",
    "_convert_shared_expert",
    "_dequant_expert_fp4_weight",
    "_dequant_or_cast",
    "_dequant_shared_fp8_weight",
    "_dsv4_scale_key_for",
    "dequantize_block_fp4_ue8m0",
    "dequantize_block_fp8_ue8m0",
    "is_block_scale_key",
    "is_dspark_key",
    "is_mtp_key",
]
