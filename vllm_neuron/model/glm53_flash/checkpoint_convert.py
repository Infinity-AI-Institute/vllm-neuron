# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash HF -> Neuron checkpoint conversion (Round 5).

Round 5 rewrite reasons this converter must match `_MoEBlock` post-Round 4:

  * Round 4 replaced the hand-rolled ``mlp.{gate,up,down}`` per-expert slabs
    with NxDI's ``ExpertMLPs`` module, whose weights live at
    ``mlp.expert_mlps.mlp_op.gate_up_proj.weight`` (stride=2 fused
    ``[gate|up]`` on the intermediate axis) and
    ``mlp.expert_mlps.mlp_op.down_proj.weight``.
  * The wrapper declares those slabs as BF16 (``ExpertFused{Column,Row}
    ParallelLinear`` inherits ``dtype=neuron_config.torch_dtype``, no
    quantization declared) so the converter must **dequantize the FP8 checkpoint
    to bf16 here**.  The blockwise NKI kernel dequants block-scale-carrying
    tensors before the matmul internally, but the checkpoint pathway does not
    receive them — the state_dict tensor shape and dtype must match the
    declared module parameter or NxDI's loader aborts with a shape mismatch.
  * The upstream state_dict prefix strip removes ``model.`` from every key
    that starts with it (``NeuronApplicationBase._STATE_DICT_MODEL_PREFIX``,
    default ``"model."``) BEFORE ``convert_hf_to_neuron_state_dict`` is
    called.  So the keys we read here are ``language_model.layers.<i>....``,
    ``lm_head.weight`` (unchanged), ``visual.encoder....``.  The Round-3
    module wrote its constants with the pre-strip prefix and would have
    matched nothing.

Every name in this module was read off the real
``model.safetensors.index.json`` for snapshot
``04c4e9e95c5da8862dced7e5056455116f83a7e0`` — nothing is guessed.  The
counts that pin the schema:

  * 76,108 tensors total; 347 under ``model.visual.``; 75,761 text.
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
   at all.  Round-7 wrapper (``_DSAIndexerBlock`` after the HF-parity
   rewrite) maps every one of them 1:1 to their HF names — ``wq_b`` and
   ``wk`` stay as ``indexer.wq_b.weight`` / ``indexer.wk.weight`` (no
   reshape; both are the HF 2D layouts), ``weights_proj`` stays as
   ``indexer.weights_proj.weight``, ``k_norm.{weight,bias}`` are carried
   through unchanged, and both compress params are copied through as
   raw parameters.  The Round-6 wrapper's rank-3 ``q_proj`` /
   scalar ``pool_weights`` scaffolds are gone — that mapping was
   provably non-numerical (``wq_b`` is a low-rank projection off q_lora,
   not a per-head reformulation of ``q_proj @ Q_B``; see
   ``DSA-INDEXER-MAPPING-FIX-2026-08-28.md``).
4. ``weight_scale_inv`` is a **reciprocal** block scale.  Dequantization is
   ``w_bf16 = (w_fp8_fp32 * scale_inv).to(bf16)`` blockwise, NOT a divide.
5. HF's routed-expert ``gate_proj`` / ``up_proj`` weights are stored as
   ``[intermediate, hidden]`` (nn.Linear convention).  NxDI's
   ``ExpertFusedColumnParallelLinear`` expects
   ``[num_experts, hidden, 2*intermediate]`` in the ``[gate | up]`` order —
   the shape and axis order Dbrx's converter also produces
   (``models/dbrx/modeling_dbrx.py:82-83``).  Getting the transpose or the
   ordering wrong compiles fine and produces plausible-but-wrong logits.
6. HF routed ``down_proj`` weight is ``[hidden, intermediate]``; NxDI's
   ``ExpertFusedRowParallelLinear`` expects
   ``[num_experts, intermediate, hidden]``.  Same transpose discipline.
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Any

import torch

from .config import Glm53FlashInferenceConfig

# After NxDI's default ``_STATE_DICT_MODEL_PREFIX = "model."`` strip, the
# GLM-5.3-Flash text tree lives under ``language_model.`` and the visual
# tower under ``visual.``.  ``lm_head.weight`` never carried ``model.`` in
# the HF index so it survives the strip untouched.
TEXT_PREFIX = "language_model."
LM_HEAD_KEY = "lm_head.weight"
EMBED_KEY = "language_model.embed_tokens.weight"
FINAL_NORM_KEY = "language_model.norm.weight"
VISION_PREFIX = "visual."

# Compat: some callers may pass through un-stripped keys (custom
# _STATE_DICT_MODEL_PREFIX overrides on subclasses, direct calls in tests).
# ``_normalize_state_dict`` below auto-strips any leading ``model.`` before
# the conversion runs, so callers on either side of the strip work.
_LEGACY_MODEL_PREFIX = "model."

SCALE_SUFFIX = "weight_scale_inv"
# Native-E4M3 (neuron_legacy_e4m3fn) max magnitude.  NOT the OCP 448.0 value
# — inheriting OCP-448 when a scale is missing is the exact bug this port
# refuses to reproduce.
NEURON_E4M3_QMAX = 240.0

# Total number of tensors in the reference HF snapshot (from
# ``model.safetensors.index.json`` for
# ``04c4e9e95c5da8862dced7e5056455116f83a7e0``).  Used to assert we saw the
# whole checkpoint at load time.
EXPECTED_HF_TENSOR_COUNT = 76108

_LAYER_RE = re.compile(r"^language_model\.layers\.(\d+)\.(.*)$")

# mHC parameters whose names contain "scale" but are NOT FP8 scales.
HC_PARAM_SUFFIXES = (
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
)


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
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Blockwise FP8 -> ``out_dtype`` using the checkpoint's *reciprocal* scales.

    ``weight`` is ``[..., out, in]``; ``scale_inv`` is
    ``[..., ceil(out/bo), ceil(in/bi)]``.  Each 128x128 tile is multiplied
    by its scalar — ``w * scale_inv``, not ``w / scale``.  Getting this
    backwards produces a model that loads and runs and is quietly wrong, so
    the shape agreement is asserted rather than broadcast-guessed.

    The scale is validated on the way in — a scale value above 240.0 would
    trip the OCP-vs-neuron-legacy divergence (``qmax=448`` inheriting the
    scale space that ``qmax=240`` produced) which is one of the exact bugs
    this port refuses to reproduce.
    """
    if scale_inv is None:
        raise ValueError(
            "dequantize_block_fp8 requires an explicit non-None weight_scale_inv"
        )
    scale_max = torch.max(scale_inv.to(torch.float32)).item()
    if not math.isfinite(scale_max) or scale_max <= 0.0:
        raise ValueError(
            f"weight_scale_inv max is {scale_max!r}; expected a positive finite value"
        )
    # The reciprocal-scale convention: block-scale values themselves live in
    # the same numerical band the fp8 weight does, so >240 flags a caller
    # that read the file as a divisor.  A real reciprocal E4M3 scale is far
    # below 240 in practice; keeping the cap at 240 preserves the campaign's
    # "no inherited OCP-448" invariant.
    if scale_max > NEURON_E4M3_QMAX:
        raise ValueError(
            f"weight_scale_inv max {scale_max!r} exceeds native-E4M3 qmax "
            f"{NEURON_E4M3_QMAX}; this looks like an OCP-448 scale space "
            "and would silently produce wrong dequantized values"
        )
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
    return (value * scale).to(out_dtype)


def _normalize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a view of ``state_dict`` with a possibly-missing ``model.`` prefix stripped.

    NxDI's ``NeuronApplicationBase.get_state_dict`` runs a
    ``startswith(_STATE_DICT_MODEL_PREFIX).replace(...)`` pass before calling
    ``convert_hf_to_neuron_state_dict``, so on the compile path the keys arrive
    stripped.  A direct caller (tests, the 1-tensor smoke) may still supply
    the raw HF names.  This function accepts either.
    """
    sample = next(iter(state_dict), None)
    if sample is None:
        return state_dict
    # Detect whether the strip has already happened.  Post-strip keys never
    # start with ``model.`` (that prefix has been removed); pre-strip keys
    # start with ``model.language_model.`` or ``model.visual.``.
    if not sample.startswith(_LEGACY_MODEL_PREFIX):
        return state_dict
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith(_LEGACY_MODEL_PREFIX):
            normalized[key[len(_LEGACY_MODEL_PREFIX):]] = value
        else:
            normalized[key] = value
    return normalized


def _dequant_or_cast(
    state_dict: dict[str, Any],
    key: str,
    block_size: tuple[int, int],
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    """Fetch a possibly-FP8 tensor and materialise it as ``out_dtype``.

    Returns ``None`` if the key is missing entirely (caller decides whether
    that is a hard error or a documented drop).  Handles the two branches:

    * Tensor has a paired ``<key>_scale_inv`` -> blockwise FP8 -> bf16
      dequantization.
    * Tensor has no paired scale -> assumed already floating (bf16 or fp32)
      and cast to ``out_dtype``.

    Never returns a raw FP8 tensor to callers.
    """
    weight = state_dict.get(key)
    if weight is None:
        return None
    scale_inv = state_dict.get(f"{key}_scale_inv")
    if scale_inv is not None:
        return dequantize_block_fp8(weight, scale_inv, block_size, out_dtype)
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(
            f"{key} is FP8 (dtype={weight.dtype}) but has no paired "
            f"{key}_scale_inv — cannot dequantize without a scale."
        )
    return weight.to(out_dtype)


def _convert_glm53_checkpoint(
    state_dict: dict[str, Any],
    src: Glm53FlashInferenceConfig,
    *,
    tp_degree: int,
) -> dict[str, Any]:
    """Map HF names onto the Round-4 wrapper's module tree, dequantizing to bf16.

    Structural decisions (each with a specific reason on top-of-file):

      * Layer 45 (MTP) and the whole vision tower are dropped explicitly.
      * All FP8 block-quant weights (MLA projections, dense MLP, shared
        expert, routed experts) are **dequantized to bf16** here.
        ``ExpertFused{Column,Row}ParallelLinear`` is built at bf16 with no
        quantisation declared, and the blockwise kernel dequants block-scale
        carriers before the matmul internally (see driver header) — either
        way the state-dict handoff to NxDI must be bf16.
      * MoE routed experts are **stacked and axis-flipped** to match
        ``ExpertMLPs.mlp_op.gate_up_proj`` shape
        ``[E, hidden, 2*intermediate]`` (fused gate|up on the last axis,
        ``stride=2`` so ColumnParallelLinear yields
        ``[gate_slice_r | up_slice_r]`` per rank) and
        ``ExpertMLPs.mlp_op.down_proj`` shape ``[E, intermediate, hidden]``
        (RowParallelLinear).
      * The three KDA short convs (``q_conv1d`` / ``k_conv1d`` /
        ``v_conv1d``) are concatenated on the channel axis into the
        wrapper's single depthwise ``conv1d`` — exact, because a depthwise
        conv is per-channel and concatenation along channels composes the
        three independent convs.
    """
    state_dict = _normalize_state_dict(state_dict)

    if tp_degree <= 0:
        raise ValueError(f"tp_degree must be positive; got {tp_degree}")

    dtype = src.torch_dtype
    block_size = tuple(src.quantization_config.weight_block_size)
    n_experts = src.n_routed_experts
    num_layers = src.num_hidden_layers  # 45 in production
    intermediate = src.intermediate_size
    moe_intermediate = src.moe_intermediate_size
    hidden = src.hidden_size

    converted: dict[str, Any] = {}
    dropped_mtp: list[str] = []
    dropped_vision: list[str] = []
    unmapped_indexer: list[str] = []

    # ---- top-level tensors ----
    if EMBED_KEY not in state_dict:
        raise ValueError(
            f"missing {EMBED_KEY!r} in state_dict — the HF prefix strip may "
            "not have run; call this function via NxDI's get_state_dict or "
            "supply keys with model. stripped."
        )
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

    # ---- per-layer conversion ----
    for layer_idx in range(num_layers):
        base = f"{TEXT_PREFIX}layers.{layer_idx}."
        out = f"layers.{layer_idx}."
        attn_kind = src.layer_types[layer_idx]
        mlp_kind = src.mlp_layer_types[layer_idx]

        # --- norms (BF16, carried verbatim) ---
        norm_map = {
            f"{base}input_layernorm.weight": f"{out}input_norm_weight",
            f"{base}post_attention_layernorm.weight": f"{out}post_attention_norm_weight",
        }
        for hf_key, tgt in norm_map.items():
            if hf_key in state_dict:
                converted[tgt] = state_dict[hf_key].to(dtype)

        # --- mHC hyper-connection params (BF16, per-mixer split into attn/mlp) ---
        for suffix in HC_PARAM_SUFFIXES:
            hf_key = f"{base}{suffix}"
            if hf_key not in state_dict:
                continue
            target = "hc_attn" if "attn" in suffix else "hc_mlp"
            leaf = suffix.split("_", 2)[-1]  # base / fn / scale
            converted[f"{out}{target}.{leaf}"] = state_dict[hf_key].to(dtype)

        # --- attention ---
        if attn_kind == "linear_attention":
            _convert_kda_layer(
                state_dict, converted, base, out, dtype, block_size
            )
        elif attn_kind == "deepseek_sparse_attention":
            _convert_dsa_layer(
                state_dict,
                converted,
                base,
                out,
                dtype,
                block_size,
                unmapped_indexer=unmapped_indexer,
            )
        else:
            raise ValueError(f"unknown layer_type {attn_kind!r} at {layer_idx}")

        # --- MLP ---
        if mlp_kind == "dense":
            _convert_dense_mlp_layer(
                state_dict, converted, base, out, dtype, block_size
            )
        elif mlp_kind == "sparse":
            _convert_moe_layer(
                state_dict,
                converted,
                base,
                out,
                dtype,
                block_size,
                n_experts=n_experts,
                hidden=hidden,
                moe_intermediate=moe_intermediate,
            )
        else:
            raise ValueError(f"unknown mlp_layer_type {mlp_kind!r} at {layer_idx}")

    # ---- dropped-tensor bookkeeping ----
    for key in state_dict.keys():
        if is_vision_key(key):
            dropped_vision.append(key)
        elif is_mtp_key(key):
            dropped_mtp.append(key)

    converted["_conversion_report"] = {
        "input_tensor_count": len(state_dict),
        "expected_input_tensor_count": EXPECTED_HF_TENSOR_COUNT,
        "converted_tensor_count": len(converted) - 1,  # minus this report
        "dropped_mtp_tensors": len(dropped_mtp),
        "dropped_vision_tensors": len(dropped_vision),
        "unmapped_indexer_tensors": len(unmapped_indexer),
        "unmapped_indexer_sample": unmapped_indexer[:6],
        "block_size": block_size,
        "qmax": NEURON_E4M3_QMAX,
        "mtp_layer_excluded": 45,
        "tp_degree": tp_degree,
    }
    return converted


def _convert_kda_layer(
    state_dict: dict[str, Any],
    converted: dict[str, Any],
    base: str,
    out: str,
    dtype: torch.dtype,
    block_size: tuple[int, int],
) -> None:
    """KDA block: entirely BF16 in this checkpoint — no scales anywhere.

    KDA's Q/K/V/B and gate projections are declared as
    ``_NxdColumnParallelLinear`` weights (leaf key ``.weight``); the delta
    projections f_a/f_b/g_a/g_b are the same.  The three per-stream short
    convs are concatenated into the wrapper's single depthwise conv on the
    channel axis (exact for depthwise: no cross-channel mixing to
    preserve).  Everything else (``A_log``, ``dt_bias``, ``o_norm``) is a
    small replicated parameter.
    """
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
        tensor = _dequant_or_cast(
            state_dict, f"{attn}{hf_suffix}", block_size, dtype
        )
        if tensor is not None:
            converted[f"{target}{dest}"] = tensor

    # Concatenate the three depthwise short-conv weights on the channel axis
    # to match the wrapper's single depthwise Conv1d.  This is exact for a
    # depthwise conv: each output channel's kernel is independent.
    parts: list[torch.Tensor] = []
    for stream in ("q", "k", "v"):
        hf_key = f"{attn}{stream}_conv1d.weight"
        if hf_key not in state_dict:
            parts = []
            break
        parts.append(state_dict[hf_key].to(dtype))
    if parts:
        converted[f"{target}conv1d.weight"] = torch.cat(parts, dim=0)


def _convert_dsa_layer(
    state_dict: dict[str, Any],
    converted: dict[str, Any],
    base: str,
    out: str,
    dtype: torch.dtype,
    block_size: tuple[int, int],
    *,
    unmapped_indexer: list[str],
) -> None:
    """DSA block: mixed-precision MLA (mostly FP8 dequantized to bf16, kv_b BF16) + BF16 indexer.

    The Round-7 wrapper's ``_NoPeMLABlock`` uses these parameter names
    (rename from HF's ``kv_a_proj_with_mqa`` to ``kv_a_proj``).  MLA is
    mixed precision: q_a, q_b, kv_a, o carry block scales; kv_b does not.
    Dequantize the ones that carry scales; carry kv_b through as bf16.

    Round-7 indexer contract (Option A: wrapper adapts to HF layout — see
    ``DSA-INDEXER-MAPPING-FIX-2026-08-28.md`` for why Option B was
    provably lossy):

      * ``indexer.wq_b.weight`` -> ``indexer.wq_b.weight`` (unchanged 2D
        ``[index_n_heads * index_head_dim, q_lora_rank]``)
      * ``indexer.wk.weight``   -> ``indexer.wk.weight`` (unchanged 2D
        ``[index_head_dim, hidden_size]`` — SINGLE-head)
      * ``indexer.k_norm.weight`` / ``indexer.k_norm.bias`` (LayerNorm on
        ``head_dim``) -> carried through unchanged.
      * ``indexer.weights_proj.weight`` -> ``indexer.weights_proj.weight``
        (unchanged 2D ``[index_n_heads, hidden_size]``).
      * ``indexer.index_kpool_compress_ape`` -> unchanged
        (``[index_kpool, index_head_dim]``).
      * ``indexer.index_kpool_compress_gate`` -> unchanged
        (``[index_head_dim, hidden_size]``).

    All indexer tensors are BF16 with no block-scale carriage; the
    ``keep_in_fp32_modules`` list carries ``weights_proj`` in the
    upstream HF forward but the wrapper's forward up-casts to fp32
    internally so bf16 storage is safe.
    """
    attn = f"{base}self_attn."
    target = f"{out}self_attn."

    # MLA projections that DO carry block scales -> bf16 dequant.
    for hf_suffix, dest in (
        ("q_a_proj.weight", "mla.q_a_proj.weight"),
        ("q_b_proj.weight", "mla.q_b_proj.weight"),
        ("kv_a_proj_with_mqa.weight", "mla.kv_a_proj.weight"),
        ("o_proj.weight", "mla.o_proj.weight"),
    ):
        tensor = _dequant_or_cast(
            state_dict, f"{attn}{hf_suffix}", block_size, dtype
        )
        if tensor is not None:
            converted[f"{target}{dest}"] = tensor

    # kv_b_proj is BF16 in this checkpoint despite its siblings being FP8.
    kv_b = _dequant_or_cast(
        state_dict, f"{attn}kv_b_proj.weight", block_size, dtype
    )
    if kv_b is not None:
        converted[f"{target}mla.kv_b_proj.weight"] = kv_b

    # MLA latent norms (bf16, no scales).
    for hf_suffix, dest in (
        ("q_a_layernorm.weight", "mla.q_a_norm"),
        ("kv_a_layernorm.weight", "mla.kv_a_norm"),
    ):
        hf_key = f"{attn}{hf_suffix}"
        if hf_key in state_dict:
            converted[f"{target}{dest}"] = state_dict[hf_key].to(dtype)

    # DSA indexer: straight-through carry into the HF-parity module tree.
    # Every HF key lands at ``indexer.<same-name>`` on the wrapper; the
    # loader's default slicing takes care of any TP sharding
    # (ColumnParallelLinear on ``wq_b`` / ``wk`` / ``weights_proj``).  No
    # reshape here: the Round-4 rank-3 ``q_proj`` scaffold, and the scalar
    # ``pool_weights`` mapping onto HF's per-token ``weights_proj``, were
    # both provably non-numerical — see ``DSA-INDEXER-MAPPING-FIX``.
    for hf_suffix in (
        "indexer.wq_b.weight",
        "indexer.wk.weight",
        "indexer.k_norm.weight",
        "indexer.k_norm.bias",
        "indexer.weights_proj.weight",
        "indexer.index_kpool_compress_ape",
        "indexer.index_kpool_compress_gate",
    ):
        hf_key = f"{attn}{hf_suffix}"
        if hf_key in state_dict:
            converted[f"{target}{hf_suffix}"] = state_dict[hf_key].to(dtype)


def _convert_dense_mlp_layer(
    state_dict: dict[str, Any],
    converted: dict[str, Any],
    base: str,
    out: str,
    dtype: torch.dtype,
    block_size: tuple[int, int],
) -> None:
    """First-3-layers dense MLP: gate/up/down (each Column/Row Parallel).

    These CARRY block scales in the HF checkpoint per its
    ``modules_to_not_convert`` list (the first 3 dense layers ARE FP8 —
    ``modules_to_not_convert`` names ``model.language_model.mlp.layers.0-2``
    only for the SHARED experts, not the dense MLP itself).  Dequant to
    bf16 to match the wrapper's ``_DenseMLPBlock``.
    """
    mlp = f"{base}mlp."
    target = f"{out}mlp."
    for name in ("gate_proj", "up_proj", "down_proj"):
        tensor = _dequant_or_cast(
            state_dict, f"{mlp}{name}.weight", block_size, dtype
        )
        if tensor is not None:
            converted[f"{target}{name}.weight"] = tensor


def _convert_moe_layer(
    state_dict: dict[str, Any],
    converted: dict[str, Any],
    base: str,
    out: str,
    dtype: torch.dtype,
    block_size: tuple[int, int],
    *,
    n_experts: int,
    hidden: int,
    moe_intermediate: int,
) -> None:
    """Sparse MoE layer: router + 288 stacked routed experts (fused) + shared expert.

    Routed experts:
      * Per-expert HF weights are dequantized FP8 -> BF16 first, then stacked
        along a new leading expert axis and axis-flipped to match NxDI's
        ``ExpertMLPs.mlp_op`` shapes:

          gate_up_proj.weight : [E, hidden, 2*intermediate]  (stride=2, [gate|up])
          down_proj.weight    : [E, intermediate, hidden]

      This is exactly the layout Dbrx's converter produces (see
      ``models/dbrx/modeling_dbrx.py:82-90``); getting the transpose /
      ordering wrong compiles fine and silently produces wrong logits.

    Shared expert: same three projections as the dense MLP, mapped 1:1 to
    the wrapper's ``_MoESharedExpert.{gate_proj, up_proj, down_proj}``.

    Router: HF ``mlp.gate.weight`` -> wrapper ``mlp.router.weight``.  The
    Round-4 wrapper carries an ``e_score_correction_bias`` on the router;
    the HF ``mlp.gate.e_score_correction_bias`` maps to it verbatim (bf16
    is fine — NxDI casts to fp32 at forward-time inside the wrapper).
    """
    mlp = f"{base}mlp."
    target = f"{out}mlp."

    # Router
    router_key = f"{mlp}gate.weight"
    if router_key in state_dict:
        converted[f"{target}router.weight"] = state_dict[router_key].to(dtype)

    # Correction bias
    corr_key = f"{mlp}gate.e_score_correction_bias"
    if corr_key in state_dict:
        # The wrapper declares this as fp32 explicitly (nn.Parameter default);
        # match that so no cast is inserted at forward time.
        converted[f"{target}e_score_correction_bias"] = state_dict[corr_key].to(torch.float32)

    # Shared expert (single MLP)
    for name in ("gate_proj", "up_proj", "down_proj"):
        tensor = _dequant_or_cast(
            state_dict, f"{mlp}shared_experts.{name}.weight", block_size, dtype
        )
        if tensor is not None:
            converted[f"{target}shared_expert.{name}.weight"] = tensor

    # Routed experts: gate/up fused with stride=2 semantics.
    #
    # For each expert e, HF stores:
    #   gate: [moe_intermediate, hidden]  (nn.Linear convention)
    #   up:   [moe_intermediate, hidden]
    #   down: [hidden, moe_intermediate]
    #
    # NxDI's ExpertFusedColumnParallelLinear.weight is
    #   [num_local_experts, hidden, output_size_per_partition]
    # so pre-sharding the full weight has shape [E, hidden, 2*moe_intermediate].
    #
    # Concatenate [gate; up] on the output-features axis (so gate takes the
    # first `moe_intermediate` cols; up takes the next `moe_intermediate`)
    # and transpose to put hidden first.  This is exactly what Dbrx's
    # converter does (``torch.cat([gate, up], dim=1).transpose(1, 2)``).
    #
    # Materialising all 288 experts into a single stacked tensor per layer
    # is ~9.66 GiB (gate_up) + 4.83 GiB (down) at bf16.  That is what makes
    # the whole state dict ~610 GiB — see the driver header's HBM preflight.
    # Streaming this to disk one expert at a time would need a full
    # checkpoint_loader_fn override; for now we do it in-memory and let the
    # RAM ceiling decide.
    if all(
        f"{mlp}experts.{e}.gate_proj.weight" in state_dict for e in range(n_experts)
    ):
        gate_stack = []
        up_stack = []
        down_stack = []
        for e in range(n_experts):
            gate = _dequant_or_cast(
                state_dict, f"{mlp}experts.{e}.gate_proj.weight", block_size, dtype
            )
            up = _dequant_or_cast(
                state_dict, f"{mlp}experts.{e}.up_proj.weight", block_size, dtype
            )
            down = _dequant_or_cast(
                state_dict, f"{mlp}experts.{e}.down_proj.weight", block_size, dtype
            )
            if gate is None or up is None or down is None:
                raise ValueError(
                    f"partial expert {e} in {mlp}: gate={gate is not None} "
                    f"up={up is not None} down={down is not None}"
                )
            # Sanity: HF shapes.
            if gate.shape != (moe_intermediate, hidden):
                raise ValueError(
                    f"expert {e} gate_proj shape {tuple(gate.shape)} != "
                    f"({moe_intermediate}, {hidden}); layer {base}"
                )
            if down.shape != (hidden, moe_intermediate):
                raise ValueError(
                    f"expert {e} down_proj shape {tuple(down.shape)} != "
                    f"({hidden}, {moe_intermediate}); layer {base}"
                )
            gate_stack.append(gate)
            up_stack.append(up)
            down_stack.append(down)
        # Stack along leading expert axis.
        gate_stacked = torch.stack(gate_stack, dim=0)   # [E, I, H]
        up_stacked = torch.stack(up_stack, dim=0)       # [E, I, H]
        down_stacked = torch.stack(down_stack, dim=0)   # [E, H, I]
        # Fuse gate|up on the intermediate axis, then move hidden to axis 1.
        gate_up_stacked = torch.cat([gate_stacked, up_stacked], dim=1)  # [E, 2I, H]
        gate_up_stacked = gate_up_stacked.transpose(1, 2).contiguous()  # [E, H, 2I]
        # Down: move intermediate to axis 1.
        down_stacked = down_stacked.transpose(1, 2).contiguous()        # [E, I, H]
        converted[f"{target}expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_stacked
        converted[f"{target}expert_mlps.mlp_op.down_proj.weight"] = down_stacked


__all__ = [
    "EXPECTED_HF_TENSOR_COUNT",
    "NEURON_E4M3_QMAX",
    "SCALE_SUFFIX",
    "TEXT_PREFIX",
    "_convert_glm53_checkpoint",
    "_normalize_state_dict",
    "dequantize_block_fp8",
    "is_fp8_scale_key",
    "is_mtp_key",
    "is_vision_key",
]
