# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash HF -> Neuron checkpoint conversion (Round 3).

Every name in this module was read off the real
``model.safetensors.index.json`` for snapshot
``04c4e9e95c5da8862dced7e5056455116f83a7e0`` — nothing is guessed.  The
counts that pin the schema:

  * 76,108 tensors total; 347 under ``model.visual.``; 75,761 text.
  * Text prefix ``model.language_model.``; ``lm_head.weight`` is the ONLY
    key with no ``model.`` prefix.
  * 46 layer indices (0-45).  ``num_hidden_layers`` is 45 — **layer 45 is the
    MTP/nextn module** (the only layer with ``eh_proj``/``enorm``/``hnorm``/
    ``shared_head`` and the only one WITHOUT ``hc_*``).  It is dropped here:
    MTP is a speculative-decode surface and the campaign forbids it.
  * The only FP8 scale suffix is ``weight_scale_inv`` (37,338 of them).
    ``weight_scale`` / ``input_scale`` are absent.

Converter traps this module handles explicitly
----------------------------------------------
1. ``hc_attn_scale`` / ``hc_ffn_scale`` (45 each) match a naive ``*scale*``
   filter but are **hyper-connection parameters, not FP8 scales**.  A filter
   that treats them as scales corrupts mHC.
2. ``kv_b_proj.weight`` has **no** scale while ``q_a_proj`` / ``q_b_proj`` /
   ``kv_a_proj_with_mqa`` / ``o_proj`` all do — the MLA block is mixed
   precision, so "all MLA tensors are FP8" is wrong.
3. The indexer tensors (``wk``, ``wq_b``, ``k_norm.{weight,bias}``,
   ``weights_proj``, ``index_kpool_compress_{ape,gate}``) carry **no** scales
   at all.
4. ``weight_scale_inv`` is a **reciprocal** block scale.  Dequantization is
   ``w_fp32 = w_fp8 * scale_inv`` blockwise, NOT a divide.
"""

from __future__ import annotations

import math
import re
from typing import Any

import torch

from .config import Glm53FlashInferenceConfig, validate_fp8_scale

TEXT_PREFIX = "model.language_model."
LM_HEAD_KEY = "lm_head.weight"
EMBED_KEY = "model.language_model.embed_tokens.weight"
FINAL_NORM_KEY = "model.language_model.norm.weight"
VISION_PREFIX = "model.visual."

SCALE_SUFFIX = "weight_scale_inv"
# Native-E4M3 (neuron_legacy_e4m3fn) max magnitude. NOT the OCP 448.0 value —
# inheriting OCP-448 when a scale is missing is the exact bug this port
# refuses to reproduce.
NEURON_E4M3_QMAX = 240.0

_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.*)$")

# mHC parameters whose names contain "scale" but are NOT FP8 scales.
HC_PARAM_SUFFIXES = (
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
)

# Tensors held in BF16 regardless of the FP8 surface (the campaign's
# hold-out list, cross-checked against the checkpoint's actual scale set).
BF16_HOLDOUT_SUBSTRINGS = (
    "input_layernorm",
    "post_attention_layernorm",
    "o_norm",
    "q_a_layernorm",
    "kv_a_layernorm",
    "k_norm",
    "embed_tokens",
    "norm.weight",
    "mlp.gate.weight",
    "e_score_correction_bias",
    "A_log",
    "dt_bias",
    "f_a_proj",
    "f_b_proj",
    "g_a_proj",
    "g_b_proj",
    "conv1d",
    "indexer",
    "kv_b_proj",
) + HC_PARAM_SUFFIXES


def is_mtp_key(key: str) -> bool:
    """Layer 45 is the MTP module — excluded (no speculative decode)."""
    match = _LAYER_RE.match(key)
    return bool(match) and int(match.group(1)) == 45


def is_vision_key(key: str) -> bool:
    return key.startswith(VISION_PREFIX)


def is_fp8_scale_key(key: str) -> bool:
    """True only for real FP8 block scales, never for the mHC `*_scale`."""
    if key.endswith(HC_PARAM_SUFFIXES):
        return False
    return key.endswith(SCALE_SUFFIX)


def dequantize_block_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    """Blockwise FP8 -> fp32 using the checkpoint's *reciprocal* scales.

    ``weight`` is ``[out, in]``; ``scale_inv`` is
    ``[ceil(out/bo), ceil(in/bi)]``.  Each 128x128 tile is multiplied by its
    scalar — ``w * scale_inv``, not ``w / scale``.  Getting this backwards
    produces a model that loads and runs and is quietly wrong, so the shape
    agreement is asserted rather than broadcast-guessed.
    """
    out_features, in_features = weight.shape[-2], weight.shape[-1]
    block_out, block_in = block_size
    expected = (
        math.ceil(out_features / block_out),
        math.ceil(in_features / block_in),
    )
    if tuple(scale_inv.shape[-2:]) != expected:
        raise ValueError(
            f"block-scale shape {tuple(scale_inv.shape)} disagrees with "
            f"weight {tuple(weight.shape)} under block_size={block_size}; "
            f"expected {expected}. Refusing to broadcast-guess."
        )
    value = weight.to(torch.float32)
    scale = scale_inv.to(torch.float32)
    # Expand each block scalar over its tile, then trim the ragged edge.
    scale = scale.repeat_interleave(block_out, dim=-2).repeat_interleave(
        block_in, dim=-1
    )[..., :out_features, :in_features]
    return value * scale


def requantize_per_tensor(
    weight_fp32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor E4M3 requantization at qmax=240 (neuron_legacy_e4m3fn).

    Returns ``(quantized_fp32_values, scale)`` where ``scale`` is a strictly
    positive scalar with an explicit non-``None`` default of 1.0 for an
    all-zero tensor.  There is no path here that yields ``None`` — that
    ``None`` is what the inherited OCP-448 bug keyed off.
    """
    amax = weight_fp32.abs().max()
    if not torch.isfinite(amax) or amax == 0:
        scale = torch.tensor(1.0, dtype=torch.float32)
    else:
        scale = (amax / NEURON_E4M3_QMAX).to(torch.float32)
    scale = torch.clamp(scale, min=torch.finfo(torch.float32).tiny)
    validate_fp8_scale(scale, "per_tensor_weight_scale")
    quantized = torch.clamp(
        weight_fp32 / scale, -NEURON_E4M3_QMAX, NEURON_E4M3_QMAX
    )
    return quantized, scale


def _shard(tensor: torch.Tensor, dim: int, tp_degree: int) -> list[torch.Tensor]:
    if tensor.shape[dim] % tp_degree:
        raise ValueError(
            f"cannot shard dim {dim} of size {tensor.shape[dim]} across "
            f"tp_degree={tp_degree}"
        )
    return list(torch.chunk(tensor, tp_degree, dim=dim))


def _convert_glm53_checkpoint(
    state_dict: dict[str, Any],
    src: Glm53FlashInferenceConfig,
    *,
    tp_degree: int,
) -> dict[str, Any]:
    """Map HF names onto the Round-3 wrapper's module tree.

    Returns a dict keyed by the wrapper's own parameter names.  Structural
    decisions:

      * Layer 45 (MTP) and the whole vision tower are dropped.
      * MLA + indexer FP8 tensors are dequantized per-block to fp32, then
        REQUANTIZED per-tensor at qmax=240, because the wrapper's MLA path
        uses per-tensor scales.
      * MoE expert tensors stay block-quantized: they are stacked along a new
        expert axis and their block scales are carried through untouched.
      * The three KDA short convs (``q_conv1d`` / ``k_conv1d`` / ``v_conv1d``)
        are concatenated on the channel axis into the wrapper's single
        depthwise ``conv1d`` — exact, because a depthwise conv is per-channel
        and concatenation along channels composes the three independent convs.
    """
    block_size = tuple(src.quantization_config.weight_block_size)
    converted: dict[str, Any] = {}
    dropped_mtp = 0
    dropped_vision = 0

    def dequant_to_per_tensor(key: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize a block-quant weight and requantize per-tensor."""
        weight = state_dict[key]
        scale_inv = state_dict.get(f"{key}_scale_inv")
        if scale_inv is None:
            # BF16 tensor (e.g. kv_b_proj) — carry through, unit scale.
            value = weight.to(torch.float32)
            return value, torch.tensor(1.0, dtype=torch.float32)
        full = dequantize_block_fp8(weight, scale_inv, block_size)
        return requantize_per_tensor(full)

    for key in list(state_dict.keys()):
        if is_vision_key(key):
            dropped_vision += 1
            continue
        if is_mtp_key(key):
            dropped_mtp += 1
            continue

    # ---- non-layer tensors ----
    if EMBED_KEY in state_dict:
        converted["embed_tokens.weight"] = state_dict[EMBED_KEY]
    if FINAL_NORM_KEY in state_dict:
        converted["final_norm_weight"] = state_dict[FINAL_NORM_KEY]
    if LM_HEAD_KEY in state_dict:
        converted["lm_head.weight"] = state_dict[LM_HEAD_KEY]

    num_layers = src.num_hidden_layers
    for layer_idx in range(num_layers):
        base = f"{TEXT_PREFIX}layers.{layer_idx}."
        out = f"layers.{layer_idx}."
        attn_kind = src.layer_types[layer_idx]
        mlp_kind = src.mlp_layer_types[layer_idx]

        # --- norms + mHC (all BF16, carried verbatim) ---
        if f"{base}input_layernorm.weight" in state_dict:
            converted[f"{out}input_norm_weight"] = state_dict[
                f"{base}input_layernorm.weight"
            ]
        if f"{base}post_attention_layernorm.weight" in state_dict:
            converted[f"{out}post_attention_norm_weight"] = state_dict[
                f"{base}post_attention_layernorm.weight"
            ]
        for suffix in HC_PARAM_SUFFIXES:
            hf_key = f"{base}{suffix}"
            if hf_key in state_dict:
                target = "hc_attn" if "attn" in suffix else "hc_mlp"
                leaf = suffix.split("_", 2)[-1]  # base / fn / scale
                converted[f"{out}{target}.{leaf}"] = state_dict[hf_key]

        # --- attention ---
        if attn_kind == "linear_attention":
            _convert_kda_layer(state_dict, converted, base, out)
        elif attn_kind == "deepseek_sparse_attention":
            _convert_dsa_layer(
                state_dict, converted, base, out, dequant_to_per_tensor
            )
        else:
            raise ValueError(f"unknown layer_type {attn_kind!r}")

        # --- MLP ---
        if mlp_kind == "dense":
            for name in ("gate_proj", "up_proj", "down_proj"):
                hf = f"{base}mlp.{name}.weight"
                if hf in state_dict:
                    converted[f"{out}mlp.{name}.weight"] = state_dict[hf]
                    scale = state_dict.get(f"{hf}_scale_inv")
                    if scale is not None:
                        converted[f"{out}mlp.{name}.weight_scale_inv"] = scale
        elif mlp_kind == "sparse":
            _convert_moe_layer(
                state_dict, converted, base, out, src, tp_degree=tp_degree
            )
        else:
            raise ValueError(f"unknown mlp_layer_type {mlp_kind!r}")

    converted["_conversion_report"] = {
        "dropped_mtp_tensors": dropped_mtp,
        "dropped_vision_tensors": dropped_vision,
        "converted_tensors": len(converted) - 1,
        "block_size": block_size,
        "qmax": NEURON_E4M3_QMAX,
        "mtp_layer_excluded": 45,
    }
    return converted


def _convert_kda_layer(
    state_dict: dict[str, Any],
    converted: dict[str, Any],
    base: str,
    out: str,
) -> None:
    """KDA block: entirely BF16 in this checkpoint — no scales anywhere."""
    attn = f"{base}self_attn."
    target = f"{out}self_attn."
    simple = {
        "q_proj.weight": "q_proj.weight",
        "k_proj.weight": "k_proj.weight",
        "v_proj.weight": "v_proj.weight",
        "b_proj.weight": "b_proj.weight",
        "f_a_proj.weight": "f_a_proj.weight",
        "f_b_proj.weight": "f_b_proj.weight",
        "g_a_proj.weight": "g_a_proj.weight",
        "g_b_proj.weight": "g_b_proj.weight",
        "o_proj.weight": "o_proj.weight",
        "A_log": "A_log",
        "dt_bias": "dt_bias",
        "o_norm.weight": "o_norm_weight",
    }
    for hf_suffix, dest in simple.items():
        hf_key = f"{attn}{hf_suffix}"
        if hf_key in state_dict:
            converted[f"{target}{dest}"] = state_dict[hf_key]

    # The wrapper holds ONE depthwise conv over [q|k|v] channels; the
    # checkpoint stores three separate depthwise convs.  Concatenating on the
    # channel (output) axis is exact for a depthwise conv, because each
    # channel's kernel is independent — no cross-channel mixing to preserve.
    parts = []
    for stream in ("q", "k", "v"):
        hf_key = f"{attn}{stream}_conv1d.weight"
        if hf_key not in state_dict:
            parts = []
            break
        parts.append(state_dict[hf_key])
    if parts:
        converted[f"{target}conv1d.weight"] = torch.cat(parts, dim=0)


def _convert_dsa_layer(
    state_dict: dict[str, Any],
    converted: dict[str, Any],
    base: str,
    out: str,
    dequant_to_per_tensor,
) -> None:
    """DSA block: mixed precision MLA + fully-BF16 indexer."""
    attn = f"{base}self_attn."
    target = f"{out}self_attn."

    # MLA projections that DO carry block scales -> per-tensor requantized.
    for hf_suffix, dest in (
        ("q_a_proj.weight", "mla.q_a_proj.weight"),
        ("q_b_proj.weight", "mla.q_b_proj.weight"),
        ("kv_a_proj_with_mqa.weight", "mla.kv_a_proj.weight"),
        ("o_proj.weight", "mla.o_proj.weight"),
    ):
        hf_key = f"{attn}{hf_suffix}"
        if hf_key not in state_dict:
            continue
        value, scale = dequant_to_per_tensor(hf_key)
        converted[f"{target}{dest}"] = value
        converted[f"{target}{dest}_scale"] = scale

    # kv_b_proj is BF16 in this checkpoint despite its siblings being FP8.
    if f"{attn}kv_b_proj.weight" in state_dict:
        converted[f"{target}mla.kv_b_proj.weight"] = state_dict[
            f"{attn}kv_b_proj.weight"
        ]

    # MLA latent norms.
    for hf_suffix, dest in (
        ("q_a_layernorm.weight", "mla.q_a_norm"),
        ("kv_a_layernorm.weight", "mla.kv_a_norm"),
    ):
        hf_key = f"{attn}{hf_suffix}"
        if hf_key in state_dict:
            converted[f"{target}{dest}"] = state_dict[hf_key]

    # Indexer — no scales exist for any of these.
    for hf_suffix, dest in (
        ("indexer.wk.weight", "indexer.k_proj.weight"),
        ("indexer.wq_b.weight", "indexer.q_proj"),
        ("indexer.weights_proj.weight", "indexer.pool_weights_proj"),
        ("indexer.k_norm.weight", "indexer.k_norm_weight"),
        ("indexer.k_norm.bias", "indexer.k_norm_bias"),
        ("indexer.index_kpool_compress_ape", "indexer.kpool_compress_ape"),
        ("indexer.index_kpool_compress_gate", "indexer.kpool_compress_gate"),
    ):
        hf_key = f"{attn}{hf_suffix}"
        if hf_key in state_dict:
            converted[f"{target}{dest}"] = state_dict[hf_key]


def _convert_moe_layer(
    state_dict: dict[str, Any],
    converted: dict[str, Any],
    base: str,
    out: str,
    src: Glm53FlashInferenceConfig,
    *,
    tp_degree: int,
) -> None:
    """Sparse MoE: router + shared expert + 288 stacked routed experts.

    Routed experts stay **block-quantized** — they are the bulk of the model
    (12,384 scale tensors per projection family) and dequantizing them to
    fp32 would defeat the point of an FP8 checkpoint.  They are stacked along
    a new leading expert axis and sharded on the intermediate axis to match
    the wrapper's `moe_intermediate_per_tp` declaration.
    """
    mlp = f"{base}mlp."
    target = f"{out}mlp."

    if f"{mlp}gate.weight" in state_dict:
        converted[f"{target}router.weight"] = state_dict[f"{mlp}gate.weight"]
    if f"{mlp}gate.e_score_correction_bias" in state_dict:
        converted[f"{target}e_score_correction_bias"] = state_dict[
            f"{mlp}gate.e_score_correction_bias"
        ]

    for name in ("gate_proj", "up_proj", "down_proj"):
        hf = f"{mlp}shared_experts.{name}.weight"
        if hf in state_dict:
            converted[f"{target}shared_expert.{name}.weight"] = state_dict[hf]
            scale = state_dict.get(f"{hf}_scale_inv")
            if scale is not None:
                converted[
                    f"{target}shared_expert.{name}.weight_scale_inv"
                ] = scale

    n_experts = src.n_routed_experts
    for hf_name, dest in (
        ("gate_proj", "gate"),
        ("up_proj", "up"),
        ("down_proj", "down"),
    ):
        weights = []
        scales = []
        for expert in range(n_experts):
            hf = f"{mlp}experts.{expert}.{hf_name}.weight"
            if hf not in state_dict:
                weights = []
                break
            weights.append(state_dict[hf])
            scale = state_dict.get(f"{hf}_scale_inv")
            if scale is not None:
                scales.append(scale)
        if not weights:
            continue
        stacked = torch.stack(weights, dim=0)
        converted[f"{target}{dest}"] = stacked
        if scales:
            converted[f"{target}{dest}_scale_inv"] = torch.stack(scales, dim=0)


__all__ = [
    "NEURON_E4M3_QMAX",
    "SCALE_SUFFIX",
    "TEXT_PREFIX",
    "_convert_glm53_checkpoint",
    "dequantize_block_fp8",
    "is_fp8_scale_key",
    "is_mtp_key",
    "is_vision_key",
    "requantize_per_tensor",
]
