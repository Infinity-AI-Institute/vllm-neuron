# SPDX-License-Identifier: Apache-2.0
"""NxDI compile-integration wrapper skeleton for DeepSeek-V4-Flash (Round 1).

Ports the GLM-5.3-Flash Round-2 wrapper structure verbatim for every
piece that transfers architecturally, and stubs the four attention
families (Sliding, CSA, HCA), the Grouped Output Projection, the
per-attention-type state cache, and the Hash-MoE bootstrap as
``NotImplementedError`` with pointers to the enablement-draft blockers.

What this skeleton establishes today:

1. NxDI-toolchain guarded imports and ``_require_nxdi()`` pattern (verbatim
   from ``glm53_flash/neuron_wrapper.py:106-183``).
2. ``build_neuron_config`` including the same MoE blockwise-mm workaround
   from user memory ``nxdi-container-moe-blockwise-mm-workaround-20260827``
   — the flag ``blockwise_matmul_config.use_shard_on_intermediate_dynamic_while
   = True`` is REQUIRED for every DeepSeek-V4-Flash MoE compile against
   container ``sha256:011d49c7495457fc2932dedd3fbecf67d28833a3f12c147377cee4d72889ebc1``
   or its successor.
3. ``FORBIDDEN_FP8_KV_KEYS`` guard mirroring GLM-5.2/5.3 (structural
   reason identical: this wrapper does not use NxDI's KVCacheManager).
4. Thin ``_NxdiInferenceConfig`` subclass forwarding the frozen fields.
5. The wrapper class ``NeuronDeepseekV4FlashForCausalLM`` raises
   ``NotImplementedError`` from ``init_model`` — the block-composition
   scaffold lands in Round 2.

What is deliberately NOT here yet:
- ``_MQABlock`` (replaces MLA): needs partial-RoPE + per-head attention
  sink + shared K=V single-head layout.
- ``_GroupedOutputProjectionBlock``: needs the ``[o_groups, num_heads *
  head_dim, o_lora_rank]`` 3-D column-parallel layout decision.
- ``_CSABlock`` / ``_HCABlock`` / ``_SlidingOnlyAttentionBlock``: three
  attention forwards; each needs its own compressor + cache alias set.
- ``_HashMoEBlock``: needs the ``tid2eid[input_ids]`` graph-time gather
  and the input-id side channel through the decoder forward.
- ``_MoEBlock`` with ``sqrt(softplus(x))`` scoring (fork of GLM-5.3
  ``_MoEBlock``; single-line scoring swap plus expert-count constants).

See the enablement draft at
``harness-v2/staging/reference-sweep-20260826T2150Z/lanes/deepseek-v4-flash/
ENABLEMENT-DRAFT-2026-08-28.md`` for the full block-by-block delta.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DeepseekV4FlashInferenceConfig

logger = logging.getLogger(__name__)

# NxDI-container guarded imports — same pattern as glm53_flash/neuron_wrapper.py.
try:
    from neuronx_distributed_inference.models.model_base import (
        NeuronBaseForCausalLM,
        NeuronBaseModel,
    )
    from neuronx_distributed_inference.models.config import (
        InferenceConfig as _NxdiInferenceConfig,
        MoENeuronConfig as _NxdiMoENeuronConfig,
    )
    from neuronx_distributed.parallel_layers.layers import (
        ColumnParallelLinear as _NxdColumnParallelLinear,
        ParallelEmbedding as _NxdParallelEmbedding,
        RowParallelLinear as _NxdRowParallelLinear,
    )
    # Round 2: routed-MoE lands on NxDI's own blockwise ExpertMLPs (identical
    # rationale to GLM-5.3-Flash Round 4 — the Python token-major gather
    # materialises `[T*top_k, hidden, inter]` per token which OOMs the tracer
    # for the 256-expert × 43-layer DSv4-Flash).
    from neuronx_distributed.modules.moe.expert_mlps import (
        ExpertMLPs as _NxdExpertMLPs,
    )
    from neuronx_distributed.modules.moe.model_utils import GLUType as _NxdGLUType
    _NXDI_AVAILABLE = True
    _NXDI_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - CPU-only guard
    class _NxdiUnavailable:
        """Placeholder used only when NxDI is missing on this host."""

    NeuronBaseForCausalLM = _NxdiUnavailable  # type: ignore[assignment,misc]
    NeuronBaseModel = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdiInferenceConfig = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdiMoENeuronConfig = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdColumnParallelLinear = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdRowParallelLinear = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdParallelEmbedding = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdExpertMLPs = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdGLUType = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NXDI_AVAILABLE = False
    _NXDI_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Router — pure PyTorch, ships in both NxDI and CPU-only paths.
#
# Source-cited spec (`deepseek-ai/DeepSeek-V4-Flash-0731` @ HF SHA
# `7872f01b1d1fe23eabc4c98b48bffcef5a386062`):
#
#   * Score function.  `transformers/activations.py:217-221` registers
#     `SqrtSoftplusActivation` under the key `sqrtsoftplus`:
#         nn.functional.softplus(input).sqrt()
#     `configuration_deepseek_v4.py:151` pins
#     `scoring_func: str = "sqrtsoftplus"` by default and the frozen
#     `DeepseekV4FlashInferenceConfig` in this package refuses any other
#     value.
#
#   * Top-k selection.  `modeling_deepseek_v4.py:1044-1051`
#     (`DeepseekV4TopKRouter.forward`):
#         flat = hidden_states.reshape(-1, self.hidden_dim)
#         logits = F.linear(flat, self.weight)                # [T, E] fp32
#         scores = self.score_fn(logits)                      # sqrtsoftplus
#         indices = torch.topk(
#             scores + self.e_score_correction_bias,
#             self.top_k, dim=-1, sorted=False
#         ).indices                                            # [T, top_k]
#         weights = scores.gather(1, indices)                  # [T, top_k]
#         weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
#         return logits, weights * self.routed_scaling_factor, indices
#
#   * `noaux_tc` topk-method (config default).  The `e_score_correction_bias`
#     shifts WHICH experts are picked (selection score) but MUST NOT leak into
#     the routing weights (the weights are the raw `scores.gather(..., indices)`).
#
#   * `routed_scaling_factor = 1.5` (config default).  Applied to the OUTPUT
#     (weights * factor).  Because the expert combination is linear in the
#     weights and `normalize_top_k_affinities=True` re-normalises them, we
#     apply this factor to the routed-branch OUTPUT after the ExpertMLPs
#     dispatch — bit-equivalent, cheaper, and avoids being cancelled by the
#     normalize.  Same trade GLM-5.3-Flash makes at
#     `nki_bindings.py::glm53_route_affinities`.
#
# TP / kernel note. `torch.topk(..., sorted=False)` lowers to `sort` on
# torch_xla; neuronx-cc rejects it:
#   [NCC_EVRF029] Operation sort is not supported on trn2. Use supported
#   equivalent operation like TopK.
# Every DSv4-Flash router call therefore MUST pass `sorted=True` (the torch
# default).  Order is irrelevant — ExpertMLPs builds a top-k-hot mask over
# the expert axis and the combination is a sum.
# ---------------------------------------------------------------------------

DSV4_ROUTED_SCORING_FUNC: str = "sqrtsoftplus"


def dsv4_route_affinities(
    hidden_states: torch.Tensor,      # [B, L, hidden] or [T, hidden]
    router_weight: torch.Tensor,      # [n_routed_experts, hidden]
    *,
    top_k: int,
    scoring_func: str,
    correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DSv4-Flash routing in the shape contract NxDI's ExpertMLPs expects.

    Returns ``(affinities[T, n_routed_experts] fp32, expert_index[T, top_k]
    int64)`` — the *full-width* per-expert score vector, NOT the gathered
    top-k weights.

    NxDI's ``ExpertMLPs.forward`` builds the top-k-hot mask itself
    (``get_expert_mask``), masks the affinities to the selected experts, and —
    when ``normalize_top_k_affinities=True`` — L1-normalises across them
    (``get_expert_affinities_masked``).  That reproduces DSv4's
    ``weights / (weights.sum(...) + 1e-20)`` normalise exactly, so this
    function must NOT pre-normalise.

    Two DSv4 specifics that survive the handoff (identical structure to
    GLM's ``noaux_tc`` handling):

    * The learned ``correction_bias`` moves *which* experts win top-k but
      must never leak into the weights, so it is added to the SELECTION
      score only and the returned affinities are the raw sqrt(softplus)
      scores.
    * ``routed_scaling_factor`` is deliberately NOT applied here.  Applying
      it before ``ExpertMLPs`` would be cancelled by the L1 normalise.
      Because the expert combination ``sum_e a_e * MLP_e(x)`` is linear in
      ``a``, scaling the *output* by the same constant is exactly
      equivalent — the caller does that after ExpertMLPs returns.

    ``sorted=True`` on ``torch.topk`` is REQUIRED, not a preference: the
    ``sort`` op lowers unsupported on trn2 (NCC_EVRF029).  Order is
    irrelevant — see module-level comment.
    """
    if scoring_func != DSV4_ROUTED_SCORING_FUNC:
        raise NotImplementedError(
            f"DSv4-Flash router scoring_func={scoring_func!r}; only "
            f"{DSV4_ROUTED_SCORING_FUNC!r} is qualified.  Refusing to guess a "
            "sigmoid/softmax equivalent — score-function drift would move "
            "which experts win top-k on every token and is a silent-quality "
            "failure mode."
        )
    hidden = hidden_states.shape[-1]
    flat = hidden_states.reshape(-1, hidden)
    logits = F.linear(flat.to(torch.float32), router_weight.to(torch.float32))
    # sqrt(softplus(logits)) — never negative, monotone; the post-gather
    # L1 normalise is well-defined and can never divide by zero as long as
    # any selected score is strictly positive (which softplus guarantees
    # for finite logits).
    scores = F.softplus(logits).sqrt()              # [T, E] fp32
    selection = scores
    if correction_bias is not None:
        selection = scores + correction_bias.to(torch.float32)
    # Unpack positionally — `torch.topk` under torch_neuronx's profiler
    # `__torch_dispatch__` wrapper can return a plain list rather than the
    # `return_types.topk` namedtuple, and `.indices` fails on that path.
    _values, indices = torch.topk(selection, k=top_k, dim=-1)
    return scores, indices.to(torch.int64)


def dsv4_reference_router_forward(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    top_k: int,
    correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.5,
    weight_eps: float = 1e-20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference DSv4 router that matches HF's ``DeepseekV4TopKRouter.forward``
    verbatim (with ``sorted=True`` for trn2 compatibility).

    Returns ``(logits[T, E], weights[T, top_k], indices[T, top_k])`` —
    scaled + normalised per HF, as a golden the wrapper's ExpertMLPs path
    must reproduce structurally.  Kept as a numerical reference for the
    per-layer smoke; NOT invoked at inference time.
    """
    hidden = hidden_states.shape[-1]
    flat = hidden_states.reshape(-1, hidden)
    logits = F.linear(flat.to(torch.float32), router_weight.to(torch.float32))
    scores = F.softplus(logits).sqrt()
    selection = scores
    if correction_bias is not None:
        selection = scores + correction_bias.to(torch.float32)
    _values, indices = torch.topk(selection, k=top_k, dim=-1)
    indices = indices.to(torch.int64)
    weights = scores.gather(-1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + weight_eps)
    return logits, weights * routed_scaling_factor, indices


# Container `sha256:011d49c7...` MoE workaround — identical to GLM-5.3-Flash.
DSV4_BLOCKWISE_MATMUL_WORKAROUND: dict[str, bool] = {
    "use_shard_on_intermediate_dynamic_while": True,
    "skip_dma_token": True,
}

# Fields that would flag an FP8-packed KV cache — DELIBERATELY FORBIDDEN.
# Same structural reason as GLM-5.2 and GLM-5.3-Flash: this wrapper replaces
# NxDI's KVCacheManager with its own per-attention-type state cache
# (sliding-window KV + compressor pool + optional indexer pool).  Aliased
# state tensors are declared bf16 explicitly.
FORBIDDEN_FP8_KV_KEYS: tuple[str, ...] = (
    "fp8_packed_kv",
    "kv_cache_quant",
    "kv_quant_config",
)


# ---------------------------------------------------------------------------
# Partial-RoPE helpers — CPU-portable pure-torch, ship in both NxDI and
# CPU-only paths.  Source-cited against transformers 5.15.1
# `deepseek_v4/modeling_deepseek_v4.py` @ HF SHA
# `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.
# ---------------------------------------------------------------------------


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotate the interleaved even/odd pairs by 90 degrees.

    Source-cited byte-for-byte against ``modeling_deepseek_v4.py:335-339``
    (``def rotate_half``): given ``x`` with trailing dim ``d``, take
    ``x1 = x[..., 0::2]``, ``x2 = x[..., 1::2]``, then interleave
    ``stack((-x2, x1), dim=-1).flatten(-2)`` — i.e. every even index i gets
    ``-x[..., i+1]`` and every odd index i+1 gets ``x[..., i]``.

    This is DIFFERENT from Llama-style RoPE's ``rotate_half`` which splits
    the head into two contiguous halves.  DeepSeek-V4 uses INTERLEAVED
    pairs, so the ``inv_freq`` table only has ``rope_dim/2`` unique
    entries — the ``repeat_interleave(2)`` inside :func:`apply_partial_rope`
    is the mirror-image expansion that keeps the pair-wise rotation math
    aligned.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """DeepSeek-V4 partial-RoPE applied to the trailing rope slice of ``x``.

    Source-cited byte-for-byte against ``modeling_deepseek_v4.py:342-359``
    (``def apply_rotary_pos_emb``).  ``cos`` / ``sin`` come in HALF-SIZED
    (one entry per interleaved pair, from
    ``DeepseekV4RotaryEmbedding.forward``).  This helper expands them to
    the full rope dim with ``repeat_interleave(2, dim=-1)``, unsqueezes a
    head-broadcast axis at ``unsqueeze_dim``, then rotates the last
    ``2 * cos.shape[-1]`` channels of ``x`` with the standard
    ``x*cos + rotate_half_interleaved(x)*sin`` formula in fp32 (up-cast to
    keep the rotation numerically stable for bf16 inputs) and leaves the
    leading NoPE channels untouched.

    V4-Flash lays each head out as ``[nope | rope]`` with
    ``rope_dim = 2 * cos.shape[-1] = qk_rope_head_dim = 64`` of
    ``head_dim = 512``, matching the reference's ``x[..., -rd:]`` indexing.

    Note the DIRECTION of ``sin``: the reference calls this same helper
    on the ATTENTION OUTPUT with ``sin`` negated (``-sin``) so that the
    contribution of each KV entry stays a function of the RELATIVE
    distance between the query and the KV entry (paper eq. 26).  Callers
    that need the conjugate rotation pass ``-sin`` explicitly here.
    """
    cos_expanded = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin_expanded = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos_expanded.shape[-1]
    if x.shape[-1] < rope_dim:
        raise ValueError(
            f"apply_partial_rope: input trailing dim {x.shape[-1]} is smaller "
            f"than rope_dim {rope_dim} — refusing to rotate a slice that "
            "does not exist."
        )
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = (
        (rope.float() * cos_expanded)
        + (_rotate_half_interleaved(rope).float() * sin_expanded)
    ).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


def build_main_rope_cos_sin(
    positions: torch.Tensor,
    *,
    rope_dim: int,
    rope_theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute half-sized (cos, sin) for DeepSeek-V4 main RoPE.

    Source-cited against ``modeling_deepseek_v4.py:114-149``
    (``compute_default_rope_parameters`` — the "main" branch, rope_type
    defaults to "default" with no yarn scaling) and lines 151-168
    (``DeepseekV4RotaryEmbedding.forward``).

    ``positions`` has shape ``[B, S]``; the returned ``cos`` / ``sin`` have
    shape ``[B, S, rope_dim // 2]`` — HALF-SIZED because DSv4 uses
    interleaved pairs (one θ per pair).  :func:`apply_partial_rope` does
    the ``repeat_interleave(2)`` expansion inside the rotation math.

    ``rope_dim`` for V4-Flash main rope is
    ``head_dim * partial_rotary_factor = 512 * 64/512 = 64``.
    ``rope_theta`` for main-rope layers (``sliding_attention``) is
    ``10000.0`` — CSA/HCA layers use the yarn-scaled "compress" rope with
    ``rope_theta = 160000.0`` (see
    ``config.DeepseekV4RopeScalingConfig``), for which a caller should
    reach for a distinct helper (deferred to the CSA/HCA blocks).
    """
    if rope_dim <= 0 or rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be positive and even, got {rope_dim}")
    if positions.ndim != 2:
        raise ValueError(f"positions must be [B, S]; got shape {tuple(positions.shape)}")
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, rope_dim, 2, dtype=torch.int64, device=positions.device)
            .to(torch.float32)
            / rope_dim
        )
    )  # [rope_dim/2]
    inv_freq_expanded = (
        inv_freq[None, :, None].expand(positions.shape[0], -1, 1)
    )  # [B, rope_dim/2, 1]
    positions_expanded = positions[:, None, :].to(torch.float32)  # [B, 1, S]
    freqs = (inv_freq_expanded @ positions_expanded).transpose(1, 2)  # [B, S, rope_dim/2]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _weighted_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """DeepseekV4RMSNorm (weighted).

    Cited against ``modeling_deepseek_v4.py:55-60`` — cast to fp32, divide
    by rsqrt(mean(square) + eps), scale by weight, cast back to input
    dtype.
    """
    input_dtype = x.dtype
    value = x.to(torch.float32)
    variance = value.pow(2).mean(-1, keepdim=True)
    value = value * torch.rsqrt(variance + eps)
    return weight * value.to(input_dtype)


def _unweighted_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """DeepseekV4UnweightedRMSNorm.

    Cited against ``modeling_deepseek_v4.py:66-72`` — pointwise
    ``x * rsqrt(mean(x**2) + eps)`` with fp32 stats, output in input
    dtype.  Applied to Q AFTER Q_B (per-head, over head_dim) and BEFORE
    partial RoPE.
    """
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


class _MQABlock(nn.Module):
    """Shared K=V Multi-Query Attention block for DeepSeek-V4-Flash.

    Owns everything the ``layers.<i>.attn.*`` subtree holds (verified
    against ``model.safetensors.index.json`` for HF snapshot
    ``deepseek-ai/DeepSeek-V4-Flash-0731 @
    7872f01b1d1fe23eabc4c98b48bffcef5a386062``):

      * ``wq_a.weight`` [q_lora_rank, hidden_size]     (FP8-e4m3 on disk;
                                                        dequant to bf16)
      * ``wq_b.weight`` [num_heads * head_dim, q_lora_rank] (FP8→bf16)
      * ``q_norm.weight`` [q_lora_rank]                (bf16 — the
                                                        transformers name
                                                        ``q_a_norm``)
      * ``wkv.weight`` [head_dim, hidden_size]         (FP8→bf16;
                                                        SHARED K=V single
                                                        head)
      * ``kv_norm.weight`` [head_dim]                  (bf16)
      * ``wo_a.weight`` [o_groups * o_lora_rank,
                         (num_heads * head_dim) // o_groups]  (FP8→bf16)
      * ``wo_b.weight`` [hidden_size, o_groups * o_lora_rank]  (FP8→bf16)
      * ``attn_sink`` [num_heads]                      (bf16 — per-head
                                                        learnable logit)

    Forward (source-cited byte-for-byte against
    ``modeling_deepseek_v4.py:801-873``, the ``DeepseekV4Attention.forward``):

      1. Q = ``rms_norm(wq_a @ x, q_norm) → wq_b →
              view as [B, H, S, D] → unweighted_rms_norm (per-head, over D)
              → partial_rope(cos, sin)``.
      2. KV = ``rms_norm(wkv @ x, kv_norm) →
              view as [B, 1, S, D] → partial_rope(cos, sin)``.
              Single KV head; broadcast to all Q heads at attention time.
      3. Attention with per-head sink (``eager_attention_forward``, lines
         717-745): scale = ``1/sqrt(head_dim)``; scores = ``Q @ K^T *
         scale + mask``; concat per-head sinks ``[B, H, S, 1]``; subtract
         per-row max for BF16 stability; softmax over the extended
         ``S_kv + 1`` axis; drop the sink column; multiply by V.
      4. Undo K-side RoPE at the query position by re-applying partial
         RoPE with sin negated on the attention output's rope slice
         (line 868 — the paper's eq. 26 conjugate rotation that makes
         K=V's contribution a relative-distance function).
      5. Grouped output projection (``DeepseekV4GroupedLinear``, lines
         303-332): reshape ``[B, S, H, D]`` as ``[B, S, o_groups,
         num_heads * head_dim / o_groups]``, apply the block-diagonal
         group-wise linear ``wo_a`` viewed as
         ``[o_groups, o_lora_rank, (num_heads * head_dim / o_groups)]``,
         flatten the last two dims to ``[B, S, o_groups * o_lora_rank]``,
         then the plain linear ``wo_b`` to hidden_size.

    Layer-role and cache: sliding_attention layers (0, 1, 40, 41, 42) use
    just this block (composed by :class:`_SlidingOnlyAttentionBlock`);
    compressed_sparse_attention layers (Round-2 :class:`_CSABlock`) and
    heavily_compressed_attention layers (Round-2 :class:`_HCABlock`) call
    :meth:`project_q_kv` and :meth:`attend_and_project` while inserting
    their compressor's ``[B, 1, T_compressed, head_dim]`` KV entries
    between the two.  The MQA block itself never sees a cache.

    TP: this class stores parameters as plain ``nn.Parameter`` so it is
    CPU-testable (the byte-clean 1-tensor smoke lives at
    ``tests/test_mqa_1tensor.py``).  When NxDI is available a compile-time
    integration hook (deferred to the CSA/HCA blocks' NEFF wiring) may
    swap the ``nn.Parameter`` bearings for ColumnParallel/RowParallel
    primitives with the same on-disk key names; the plain-tensor forward
    stays as the CPU reference against which the compile-time forward is
    gated.
    """

    # State-dict spelling under this module — matches HF layer subtree
    # verbatim.  Kept as a class attribute so the converter and the test
    # can share the list of names.
    PARAM_KEYS: tuple[str, ...] = (
        "wq_a.weight",
        "wq_b.weight",
        "q_norm.weight",
        "wkv.weight",
        "kv_norm.weight",
        "wo_a.weight",
        "wo_b.weight",
        "attn_sink",
    )

    def __init__(
        self,
        config: Any,
        *,
        layer_idx: int,
    ) -> None:
        super().__init__()
        src = getattr(config, "source_config", None)
        if src is None:
            # Allow raw DeepseekV4FlashInferenceConfig too — the CPU smoke
            # test does not need the NxDI InferenceConfig wrapping.
            src = config
        self.layer_idx = layer_idx
        self.hidden_size = int(src.hidden_size)
        self.num_heads = int(src.num_attention_heads)
        self.num_kv_heads = int(src.num_key_value_heads)
        self.head_dim = int(src.head_dim)
        self.q_lora_rank = int(src.q_lora_rank)
        self.qk_rope_head_dim = int(src.qk_rope_head_dim)
        self.o_groups = int(src.o_groups)
        self.o_lora_rank = int(src.o_lora_rank)
        self.rms_eps = float(src.rms_norm_eps)
        self.rope_theta = float(src.rope_theta)
        self.scaling = self.head_dim ** -0.5
        # DSv4-Flash freeze: single shared KV head; verify to fail loudly
        # if a caller stripped that invariant off the frozen config.
        if self.num_kv_heads != 1:
            raise ValueError(
                "_MQABlock is only valid for shared-KV MQA "
                "(num_key_value_heads=1); got "
                f"num_key_value_heads={self.num_kv_heads}"
            )
        # Grouped-output projection divisibility (fail loud, never guess).
        if (self.num_heads * self.head_dim) % self.o_groups != 0:
            raise ValueError(
                f"grouped output projection requires num_heads*head_dim "
                f"({self.num_heads * self.head_dim}) divisible by o_groups "
                f"({self.o_groups})"
            )
        if self.o_lora_rank <= 0 or self.o_groups <= 0:
            raise ValueError(
                f"o_groups={self.o_groups}, o_lora_rank={self.o_lora_rank} "
                "must be positive"
            )
        if self.qk_rope_head_dim <= 0 or self.qk_rope_head_dim > self.head_dim:
            raise ValueError(
                f"qk_rope_head_dim={self.qk_rope_head_dim} must be in "
                f"(0, head_dim={self.head_dim}]"
            )
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError(
                f"qk_rope_head_dim={self.qk_rope_head_dim} must be even "
                "(interleaved RoPE has one θ per pair)"
            )
        self._in_features_per_group = (self.num_heads * self.head_dim) // self.o_groups
        dtype = getattr(config, "torch_dtype", None) or getattr(
            getattr(config, "neuron_config", None), "torch_dtype", None
        ) or src.torch_dtype
        self._dtype = dtype

        # Weights: stored as plain nn.Parameter so this class is CPU-portable
        # (the smoke at tests/test_mqa_1tensor.py runs on a dev laptop
        # without NxDI).  NxDI TP integration re-declares the same keys as
        # ColumnParallel/RowParallel primitives at compile time; state-dict
        # spelling is invariant across the two backings because both use
        # trailing ``.weight``.
        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False, dtype=dtype)
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.num_heads * self.head_dim, bias=False, dtype=dtype
        )
        self.wkv = nn.Linear(self.hidden_size, self.head_dim, bias=False, dtype=dtype)
        self.wo_a = nn.Linear(
            self._in_features_per_group,
            self.o_groups * self.o_lora_rank,
            bias=False,
            dtype=dtype,
        )
        self.wo_b = nn.Linear(
            self.o_groups * self.o_lora_rank, self.hidden_size, bias=False, dtype=dtype
        )
        # RMSNorm gains — small, replicated across ranks under any TP.
        self.q_norm = _MQANormParam(self.q_lora_rank, dtype)
        self.kv_norm = _MQANormParam(self.head_dim, dtype)
        # Per-head learnable sink (like GPT-OSS).  HF stores as
        # ``layers.<i>.attn.attn_sink`` with shape ``[num_heads]``.
        self.attn_sink = nn.Parameter(
            torch.zeros(self.num_heads, dtype=dtype), requires_grad=False
        )

    # ------------------------------------------------------------------
    # Reshape helpers — keep the block composable so _CSABlock / _HCABlock
    # can splice compressor KV in without re-running Q or KV projections.
    # ------------------------------------------------------------------

    def project_q(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(q [B, H, S, D], q_residual [B, S, q_lora_rank])``.

        ``q_residual`` is the post-Q_A + q_norm intermediate — the same
        tensor the compressor's indexer contracts against for CSA layers
        (paper §2.3.1).  Returning it lets ``_CSABlock`` share the Q_A
        cost.  Matches ``modeling_deepseek_v4.py:816-819``.
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                f"_MQABlock expects [B, S, hidden]; got shape "
                f"{tuple(hidden_states.shape)}"
            )
        batch, seq, _ = hidden_states.shape
        q_residual = _weighted_rms_norm(
            self.wq_a(hidden_states), self.q_norm.weight, self.rms_eps
        )
        q_flat = self.wq_b(q_residual)                                       # [B, S, H*D]
        q = q_flat.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        q = _unweighted_rms_norm(q, self.rms_eps)                            # per-head
        q = apply_partial_rope(q, cos, sin)                                  # trailing rope slice
        return q, q_residual

    def project_kv(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Return the single-head KV tensor ``[B, 1, S, D]`` post-RoPE.

        Shared K=V — the same tensor is read as both key and value in
        :meth:`attend_and_project` (and :meth:`forward`).  Matches
        ``modeling_deepseek_v4.py:821-822``.
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                f"_MQABlock expects [B, S, hidden]; got shape "
                f"{tuple(hidden_states.shape)}"
            )
        batch, seq, _ = hidden_states.shape
        kv_flat = _weighted_rms_norm(
            self.wkv(hidden_states), self.kv_norm.weight, self.rms_eps
        )                                                                     # [B, S, D]
        kv = kv_flat.view(batch, seq, 1, self.head_dim).transpose(1, 2)      # [B, 1, S, D]
        kv = apply_partial_rope(kv, cos, sin)                                # SAME rope as Q
        return kv

    def attend_and_project(
        self,
        q: torch.Tensor,               # [B, H, S, D]
        kv: torch.Tensor,              # [B, 1, T_kv, D] — SHARED K=V
        cos: torch.Tensor,             # [B, S, rope_dim/2] — for the -sin conjugate on output
        sin: torch.Tensor,             # [B, S, rope_dim/2]
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the attention math (with sink) + grouped output projection.

        ``kv`` is [B, 1, T_kv, D] — a single KV head shared across all Q
        heads (V4's MQA + shared K=V).  For CSA/HCA layers the caller
        catenates compressor KV entries onto the ``T_kv`` axis BEFORE
        calling this method (matching ``modeling_deepseek_v4.py:832``);
        the compressor's ``block_bias`` should be catted onto
        ``attention_mask`` in the same order.

        Returns ``[B, S, hidden_size]`` — the ready-to-add-into-residual
        block output.
        """
        batch, num_heads, seq, head_dim = q.shape
        if head_dim != self.head_dim or num_heads != self.num_heads:
            raise ValueError(
                f"q shape {(batch, num_heads, seq, head_dim)} mismatches "
                f"num_heads={self.num_heads}, head_dim={self.head_dim}"
            )
        if kv.shape[0] != batch or kv.shape[1] != 1 or kv.shape[-1] != head_dim:
            raise ValueError(
                f"kv shape {tuple(kv.shape)} must be [B={batch}, 1, T, D={head_dim}]"
            )
        t_kv = kv.shape[2]

        # Broadcast the single KV head across the query heads.  We keep an
        # explicit ``expand`` (no memory materialisation) rather than
        # reshaping so the backward pass — should this class ever be
        # trained — sees the correct grad-graph.  Same shape contract
        # ``eager_attention_forward`` uses via ``repeat_kv``.
        k = kv.expand(batch, num_heads, t_kv, head_dim)
        v = k

        # Attention scores + sink.  Scale is fp32-safe.
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # [B, H, S, T_kv]
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Per-head sink participates in the softmax denominator, then is
        # dropped from the weighted-value sum.  Matches
        # ``eager_attention_forward`` (modeling_deepseek_v4.py:733-741).
        sinks = self.attn_sink.reshape(1, num_heads, 1, 1).expand(batch, num_heads, seq, 1)
        combined_logits = torch.cat([attn_scores, sinks.to(attn_scores.dtype)], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        attn_weights = probs[..., :-1]                                       # drop sink
        attn_output = torch.matmul(attn_weights.to(v.dtype), v)              # [B, H, S, D]
        attn_output = attn_output.transpose(1, 2).contiguous()               # [B, S, H, D]

        # Undo K-side RoPE at the query position (paper eq. 26).  RoPE
        # helper expects ``[B, S, H, D]`` (its ``unsqueeze_dim=1`` adds a
        # head-broadcast axis to cos/sin).
        attn_output = apply_partial_rope(attn_output, cos, -sin, unsqueeze_dim=2)

        # Grouped output projection.  ``wo_a`` is stored as a plain
        # ``nn.Linear`` with weight shape
        # ``[o_groups * o_lora_rank, (num_heads * head_dim) / o_groups]``;
        # DeepseekV4GroupedLinear (modeling_deepseek_v4.py:303-332) views
        # it as ``[o_groups, o_lora_rank, in_per_group]`` and does a
        # batched-matmul against grouped input.
        input_shape = attn_output.shape[:-2]                                 # [B, S]
        hidden_per_group = attn_output.shape[-1]                             # head_dim=512
        # Reshape [B, S, H, D] → [B, S, o_groups, H*D/o_groups].
        grouped_in = attn_output.reshape(*input_shape, self.o_groups, -1)    # [B, S, G, H*D/G]
        # DeepseekV4GroupedLinear.forward viewed step-by-step:
        w = self.wo_a.weight.view(self.o_groups, self.o_lora_rank, self._in_features_per_group).transpose(1, 2)
        # w now [G, in_per_group, o_lora_rank]
        x = grouped_in.reshape(-1, self.o_groups, self._in_features_per_group).transpose(0, 1)
        # x now [G, B*S, in_per_group]
        y = torch.bmm(x, w).transpose(0, 1)                                  # [B*S, G, o_lora_rank]
        grouped_out = y.reshape(*input_shape, self.o_groups, self.o_lora_rank)
        # flatten the last two dims: [B, S, G*o_lora_rank]
        grouped_out = grouped_out.flatten(2)
        output = self.wo_b(grouped_out)                                      # [B, S, hidden]
        del hidden_per_group  # only used for the shape-doc comment above
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full MQA block forward for a sliding_attention layer.

        Composes :meth:`project_q`, :meth:`project_kv`,
        :meth:`attend_and_project`.  CSA/HCA layers wire the compressor
        between project_kv and attend_and_project.
        """
        q, _q_residual = self.project_q(hidden_states, cos, sin)
        kv = self.project_kv(hidden_states, cos, sin)
        return self.attend_and_project(q, kv, cos, sin, attention_mask=attention_mask)


class _MQANormParam(nn.Module):
    """Trivial container so ``self.q_norm.weight`` matches the HF
    state-dict spelling (``layers.<i>.attn.q_norm.weight``).

    Kept intentionally minimal — no ``forward`` — because the RMSNorm math
    is handled inline via :func:`_weighted_rms_norm` (identical to the way
    ``glm53_flash/neuron_wrapper.py`` uses ``_rms_norm_weight`` +
    ``_rms_norm``).
    """

    def __init__(self, hidden: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(hidden, dtype=dtype), requires_grad=False
        )


def _require_nxdi() -> None:
    if _NXDI_AVAILABLE:
        return
    detail = f": {_NXDI_IMPORT_ERROR!r}" if _NXDI_IMPORT_ERROR is not None else ""
    raise RuntimeError(
        "DeepSeek-V4-Flash NxDI wrapper requires the Neuron toolchain "
        "(`neuronx_distributed_inference` + `neuronx_distributed.parallel_layers`) "
        "inside container "
        "sha256:011d49c7495457fc2932dedd3fbecf67d28833a3f12c147377cee4d72889ebc1 "
        "or equivalent" + detail
    )


def build_neuron_config(
    *,
    tp_degree: int,
    ctx_batch_size: int,
    tkg_batch_size: int,
    seq_len: int,
    torch_dtype: torch.dtype = torch.bfloat16,
    is_continuous_batching: bool = True,
    disable_argmax_kernel: bool = False,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Construct an NxDI ``MoENeuronConfig`` with the DeepSeek-V4-Flash MoE
    workaround pinned.

    Structural notes are identical to GLM-5.3-Flash's ``build_neuron_config``
    (see ``glm53_flash/neuron_wrapper.py:185-284``): the flag MUST land on
    ``MoENeuronConfig`` (not the base ``NeuronConfig``) or NxDI silently
    drops it and the compile fails inside the raising stub.
    """
    _require_nxdi()
    if extra:
        offenders = sorted(k for k in FORBIDDEN_FP8_KV_KEYS if k in extra)
        if offenders:
            raise ValueError(
                "DeepSeek-V4-Flash refuses FP8-packed KV configuration: "
                f"{offenders!r}. This wrapper replaces NxDI's KVCacheManager "
                "with its own per-attention-type state cache."
            )
    kwargs = {
        "tp_degree": tp_degree,
        "batch_size": tkg_batch_size,
        "ctx_batch_size": ctx_batch_size,
        "tkg_batch_size": tkg_batch_size,
        "max_batch_size": tkg_batch_size,
        "kv_cache_batch_size": tkg_batch_size,
        "seq_len": seq_len,
        "n_active_tokens": seq_len,
        "torch_dtype": torch_dtype,
        "is_continuous_batching": is_continuous_batching,
        "token_generation_batches": [tkg_batch_size],
        "disable_argmax_kernel": disable_argmax_kernel,
        "blockwise_matmul_config": dict(DSV4_BLOCKWISE_MATMUL_WORKAROUND),
    }
    if extra:
        extra_bmc = extra.pop("blockwise_matmul_config", None)
        if "max_context_length" in extra and "n_active_tokens" not in extra:
            kwargs["n_active_tokens"] = int(extra["max_context_length"])
        kwargs.update(extra)
        if extra_bmc is not None:
            merged = dict(DSV4_BLOCKWISE_MATMUL_WORKAROUND)
            merged.update(extra_bmc)
            kwargs["blockwise_matmul_config"] = merged
    config = _NxdiMoENeuronConfig(**kwargs)
    bmc = getattr(config, "blockwise_matmul_config", None)
    if bmc is None or not getattr(
        bmc, "use_shard_on_intermediate_dynamic_while", False
    ):
        raise RuntimeError(
            "DeepSeek-V4-Flash MoE requires "
            "blockwise_matmul_config.use_shard_on_intermediate_dynamic_while=True "
            "to survive NeuronConfig construction; it did not.  Without it the "
            "LNC=2 blockwise dispatch falls into _call_shard_hidden_kernel, "
            "which raises NotImplementedError on this container.  Got: "
            f"{bmc!r}"
        )
    return config


class DeepseekV4FlashNeuronInferenceConfig(_NxdiInferenceConfig):
    """NxDI ``InferenceConfig`` wrapper carrying the frozen source config.

    Same structural role as ``Glm53FlashNeuronInferenceConfig`` — a thin
    forward of the fields NxDI's compile flow reads.  The frozen source
    is held on ``self.source_config`` and remains the single truth for
    every architectural constant.
    """

    if _NXDI_AVAILABLE:

        def __init__(
            self,
            neuron_config: Any,
            source_config: DeepseekV4FlashInferenceConfig | None = None,
            **kwargs: Any,
        ) -> None:
            self.source_config = source_config
            if source_config is not None:
                for name in (
                    "vocab_size",
                    "hidden_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "rms_norm_eps",
                    "max_position_embeddings",
                    "hidden_act",
                    "pad_token_id",
                    "torch_dtype",
                    "tie_word_embeddings",
                ):
                    kwargs.setdefault(name, getattr(source_config, name))
                kwargs.setdefault("head_dim", source_config.head_dim)
                kwargs.setdefault("rope_theta", source_config.rope_theta)
                # DeepSeek-V4 has no separate intermediate_size for the dense
                # path (there is no dense MLP; layers 0-2 are hash_moe).  Fill
                # NxDI's field with the shared-expert width so downstream code
                # sees a real value rather than the 0 default.
                kwargs.setdefault(
                    "intermediate_size", source_config.moe_intermediate_size
                )
            super().__init__(neuron_config=neuron_config, **kwargs)

        def get_required_attributes(self):
            return [
                "hidden_size",
                "num_attention_heads",
                "num_hidden_layers",
                "num_key_value_heads",
                "vocab_size",
                "max_position_embeddings",
                "rms_norm_eps",
                "hidden_act",
            ]

        def add_derived_config(self):
            self.num_cores_per_group = getattr(
                self.neuron_config, "num_cores_per_group", 1
            )


# ---------------------------------------------------------------------------
# Round-1 scaffold: block classes are stubs.  Round 2 lands the real ones.
# ---------------------------------------------------------------------------
if _NXDI_AVAILABLE:

    def _reduce_from_tp_region(x: torch.Tensor) -> torch.Tensor:
        """All-reduce a partial contraction across the TP group.

        Used by the routed-MoE block: ``ExpertMLPs`` constructs its
        ``down_proj`` with ``reduce_output=False`` (Experts owns the
        intermediate-axis shard), so each rank ends up holding a partial
        sum over its intermediate slice.  Without this reduce every rank
        emits 1/TP of the routed activation — a silent correctness bug
        that shows up as mild logit drift rather than a crash.

        Same helper GLM-5.3-Flash uses at
        ``glm53_flash/neuron_wrapper.py:_reduce_from_tp_region``; the
        rationale is identical.
        """
        try:
            from neuronx_distributed.parallel_layers.mappings import (
                reduce_from_tensor_model_parallel_region,
            )

            return reduce_from_tensor_model_parallel_region(x)
        except Exception as exc:
            from neuronx_distributed.parallel_layers.parallel_state import (
                get_tensor_model_parallel_size,
            )

            try:
                world = int(get_tensor_model_parallel_size())
            except Exception:
                world = 1
            if world > 1:
                raise RuntimeError(
                    "DeepSeek-V4-Flash routed MoE needs a TP all-reduce on the "
                    f"ExpertMLPs.down_proj partial sum at tp_degree={world}, "
                    "but NxD's reduce_from_tensor_model_parallel_region is "
                    f"unavailable: {exc!r}. Refusing to run a per-rank "
                    "1/TP-scaled routed activation, which would silently "
                    "produce wrong logits on every token."
                ) from exc
            return x

    class _SlidingOnlyAttentionBlock(nn.Module):
        """Sliding-only attention (bootstrap layers 0, 1): window=128, no compressor."""

        def __init__(self, config: Any, *, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self._config = config
            raise NotImplementedError(
                "_SlidingOnlyAttentionBlock is Round 2.  Composes _MQABlock + "
                "sliding-window causal mask over `sliding_window=128` KV positions."
            )

    class _CSABlock(nn.Module):
        """Compressed Sparse Attention (paper §2.3.1).

        Composes:
          * ``_MQABlock`` (main attention math shared).
          * Sliding-window K=V branch (window=128, always).
          * **Compressor with overlap state** (m=4, learned pool weights):
            per-window aliased overlap buffer of shape [B, m-1, D] carries
            across forward calls.  This is the largest new mechanism in the
            port and has no GLM-5.3-Flash analogue.
          * Lightning Indexer (paper eq. 13-17): scores queries against
            pooled entries, gathers top ``index_topk=512`` blocks per query.
            Mostly reusable from GLM-5.3-Flash `_DSAIndexerBlock` after
            index-topk / index-head-dim constants are re-pinned.
        """

        def __init__(self, config: Any, *, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self._config = config
            raise NotImplementedError(
                "_CSABlock is Round 2.  Blocker for first NEFF fire: compressor "
                "overlap-state aliasing (per user memory / enablement-draft §3-5)."
            )

    class _HCABlock(nn.Module):
        """Heavily Compressed Attention (paper §2.3.2).

        Composes:
          * ``_MQABlock`` (main attention math shared).
          * Sliding-window K=V branch (window=128, always).
          * **Compressor without overlap** (m'=128): non-overlapping
            pool; no indexer — every pooled entry is potentially visible
            once its window has closed.
        """

        def __init__(self, config: Any, *, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self._config = config
            raise NotImplementedError(
                "_HCABlock is Round 2.  Simpler than _CSABlock (no indexer, no "
                "overlap), still needs compressor pool aliasing."
            )

    class _HashMoEBlock(nn.Module):
        """Hash-MoE bootstrap for layers 0..num_hash_layers-1 (paper §2.1).

        Frozen ``tid2eid[input_ids]`` lookup selects experts; learned gate
        weights weight the selected experts.  Requires the input-id side
        channel through the decoder forward — NxDI's stock
        ``DecoderModelInstance.forward`` hands each layer only the hidden
        state, so ``input_ids`` must be threaded via a second graph input
        that survives lowering.
        """

        def __init__(self, config: Any, *, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self._config = config
            raise NotImplementedError(
                "_HashMoEBlock is Round 2.  Blocker for first NEFF fire: input-id "
                "side channel through decoder forward (see enablement-draft §3-6)."
            )

    class _MoESharedExpert(nn.Module):
        """Shared-expert branch of the DSv4-Flash sparse MoE.

        Matches ``DeepseekV4MLP`` in ``modeling_deepseek_v4.py:974-989`` bit
        for bit: three projections (gate=w1, up=w3, down=w2), silu(gate) *
        up with the ``swiglu_limit`` clamp on both gate (upper) and up
        (both).  Sharded on the intermediate axis via ColumnParallel gate/up
        + RowParallel down — the RowParallel does the reduce inside its own
        forward, so the shared branch never sees the ``ExpertMLPs`` partial
        sum.
        """

        def __init__(
            self, config: DeepseekV4FlashNeuronInferenceConfig
        ) -> None:
            super().__init__()
            src = getattr(config, "source_config", None)
            if src is None:
                raise RuntimeError(
                    "_MoESharedExpert requires config.source_config to carry "
                    "the frozen DeepseekV4FlashInferenceConfig."
                )
            self.hidden_size = src.hidden_size
            self.moe_intermediate_size = src.moe_intermediate_size
            self.limit = src.swiglu_limit
            dtype = config.neuron_config.torch_dtype
            tp_degree = config.neuron_config.tp_degree
            if self.moe_intermediate_size % tp_degree:
                raise NotImplementedError(
                    f"Shared-expert MLP requires moe_intermediate_size "
                    f"({self.moe_intermediate_size}) divisible by TP degree "
                    f"({tp_degree})."
                )
            self.gate_proj = _NxdColumnParallelLinear(
                self.hidden_size,
                self.moe_intermediate_size,
                bias=False,
                gather_output=False,
                dtype=dtype,
            )
            self.up_proj = _NxdColumnParallelLinear(
                self.hidden_size,
                self.moe_intermediate_size,
                bias=False,
                gather_output=False,
                dtype=dtype,
            )
            self.down_proj = _NxdRowParallelLinear(
                self.moe_intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=dtype,
            )

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            gate = self.gate_proj(hidden_states).clamp(max=self.limit)
            up = self.up_proj(hidden_states).clamp(-self.limit, self.limit)
            return self.down_proj(F.silu(gate) * up)

    class _RoutedMoEBlock(nn.Module):
        """DSv4-Flash 256-expert routed MoE (top-6) + one shared expert.

        Structure mirrors ``glm53_flash/neuron_wrapper.py::_MoEBlock`` Round
        4 verbatim (NxDI blockwise ``ExpertMLPs`` + separate shared branch
        + partial-sum all-reduce), with three DSv4-specific swaps:

          * Router scoring: ``sqrt(softplus(x))`` (see
            :func:`dsv4_route_affinities` for the source-cited spec).
          * Constants: ``n_routed_experts=256``, ``top_k=6``,
            ``routed_scaling_factor=1.5``, ``swiglu_limit=10.0`` — every
            value is read from the frozen ``DeepseekV4FlashInferenceConfig``
            rather than hard-coded here.
          * Correction bias: HF stores it as ``ffn.gate.bias`` (a scalar
            per expert), whereas GLM-5.3 stores it as
            ``mlp.gate.e_score_correction_bias``.  This wrapper renames it
            to ``e_score_correction_bias`` at the parameter level; the
            converter handles the on-disk name.

        NxDI compile-time constraints inherited from GLM-5.3 (they are
        properties of the ``blockwise_matmul`` dispatch, not of the
        specific model):

          * ``blockwise_matmul_config.use_shard_on_intermediate_dynamic_while=True``
            (the container sha256:011d49c7 workaround from user memory
            ``nxdi-container-moe-blockwise-mm-workaround-20260827``).
          * ``logical_nc_config=2`` (LNC=1 raises
            ``"LNC_1 kernels not available in nkilib"``).
        """

        def __init__(
            self,
            config: DeepseekV4FlashNeuronInferenceConfig,
            *,
            layer_idx: int,
        ) -> None:
            super().__init__()
            src = getattr(config, "source_config", None)
            if src is None:
                raise RuntimeError(
                    "_RoutedMoEBlock requires config.source_config to carry "
                    "the frozen DeepseekV4FlashInferenceConfig."
                )
            self.layer_idx = layer_idx
            self.hidden_size = src.hidden_size
            self.n_routed_experts = src.n_routed_experts
            self.num_experts_per_tok = src.num_experts_per_tok
            self.moe_intermediate_size = src.moe_intermediate_size
            self.routed_scaling_factor = src.routed_scaling_factor
            self.norm_topk_prob = src.norm_topk_prob
            self.scoring_func = src.scoring_func
            self.swiglu_limit = src.swiglu_limit
            dtype = config.neuron_config.torch_dtype
            tp_degree = config.neuron_config.tp_degree
            self.tp_degree = tp_degree
            if self.moe_intermediate_size % tp_degree:
                raise NotImplementedError(
                    f"Routed MoE requires moe_intermediate_size "
                    f"({self.moe_intermediate_size}) divisible by TP degree "
                    f"({tp_degree})."
                )
            self.moe_intermediate_per_tp = (
                self.moe_intermediate_size // tp_degree
            )
            # Router lives in fp32.  The scoring function is
            # numerically-sensitive (softplus of large-magnitude logits
            # would saturate at fp16 far short of where fp32 stays
            # informative), and the router is O(hidden * E) per token
            # which is inexpensive relative to the routed-expert path — so
            # there is no reason to accept the precision hit.
            self.router = nn.Linear(
                self.hidden_size,
                self.n_routed_experts,
                bias=False,
                dtype=torch.float32,
            )
            # NxDI blockwise ExpertMLPs — same call structure as GLM-5.3
            # Round 4 (glm53_flash/neuron_wrapper.py:1810-1832).  DSv4's
            # SwiGLU is `silu(clamp(gate, max=L)) * clamp(up, -L, L)` with
            # L=10.0, exactly what GLU + gate/up clamp limits express.
            blockwise = getattr(
                config.neuron_config, "blockwise_matmul_config", None
            )
            if blockwise is None or not getattr(
                blockwise, "use_shard_on_intermediate_dynamic_while", False
            ):
                raise RuntimeError(
                    "DSv4-Flash routed MoE requires "
                    "neuron_config.blockwise_matmul_config."
                    "use_shard_on_intermediate_dynamic_while=True.  Without "
                    "it the LNC=2 blockwise dispatch falls into "
                    "_call_shard_hidden_kernel, which raises "
                    "NotImplementedError on container sha256:011d49c7. "
                    f"Got: {blockwise!r}"
                )
            lnc = int(getattr(config.neuron_config, "logical_nc_config", 2))
            if lnc != 2:
                raise NotImplementedError(
                    "DSv4-Flash routed MoE requires LNC=2 on this container: "
                    "the LNC=1 branch raises "
                    '"LNC_1 kernels not available in nkilib". '
                    f"Got logical_nc_config={lnc}."
                )
            self.expert_mlps = _NxdExpertMLPs(
                num_experts=self.n_routed_experts,
                top_k=self.num_experts_per_tok,
                hidden_size=self.hidden_size,
                intermediate_size=self.moe_intermediate_size,
                hidden_act=src.hidden_act,
                glu_mlp=True,
                glu_type=_NxdGLUType.GLU,
                capacity_factor=None,          # dropless / full capacity
                normalize_top_k_affinities=self.norm_topk_prob,
                gate_clamp_upper_limit=self.swiglu_limit,
                gate_clamp_lower_limit=None,
                up_clamp_upper_limit=self.swiglu_limit,
                up_clamp_lower_limit=-self.swiglu_limit,
                early_expert_affinity_modulation=False,
                dtype=dtype,
                logical_nc_config=lnc,
                use_shard_on_intermediate_dynamic_while=True,
                skip_dma_token=bool(
                    getattr(blockwise, "skip_dma_token", True)
                ),
                block_size=int(getattr(blockwise, "block_size", 512)),
            )
            self.shared_expert = _MoESharedExpert(config)
            # DSv4's selection-only correction bias.  Declared unconditionally
            # with an explicit zero default so a checkpoint that omits it
            # degrades to plain top-k rather than silently to None.
            self.e_score_correction_bias = nn.Parameter(
                torch.zeros(self.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """DSv4 MoE forward: sqrt(softplus) router + NxDI blockwise
            ExpertMLPs + separate shared branch, with the same
            partial-sum all-reduce discipline GLM-5.3 Round 4 documents.
            """
            shape = hidden_states.shape
            length = shape[1] if hidden_states.ndim == 3 else 1
            flat = hidden_states.reshape(-1, self.hidden_size)

            shared = self.shared_expert(hidden_states)

            affinities, expert_index = dsv4_route_affinities(
                hidden_states,
                self.router.weight,
                top_k=self.num_experts_per_tok,
                scoring_func=self.scoring_func,
                correction_bias=self.e_score_correction_bias,
            )
            routed = self.expert_mlps(
                hidden_states=flat,
                expert_affinities=affinities.to(flat.dtype),
                expert_index=expert_index,
                seq_len=length,
            )
            routed = _reduce_from_tp_region(routed)
            routed = routed * self.routed_scaling_factor
            return shared + routed.view(shape).to(shared.dtype)

    # Backwards-compat alias so existing references keep resolving.
    _MoEBlock = _RoutedMoEBlock

    class _NeuronDeepseekV4FlashModel(NeuronBaseModel):
        """NxDI base model — Round-1 skeleton, forward stubbed."""

        def init_model(self, config: DeepseekV4FlashNeuronInferenceConfig) -> None:
            raise NotImplementedError(
                "DeepSeek-V4-Flash init_model is Round 2.  Blockers listed in "
                "the enablement draft ENABLEMENT-DRAFT-2026-08-28.md §3."
            )

        def init_inference_optimization(
            self, config: DeepseekV4FlashNeuronInferenceConfig
        ) -> None:
            # Deliberately empty — this wrapper does NOT use NxDI's
            # KVCacheManager (see FORBIDDEN_FP8_KV_KEYS docstring).  Round 2
            # lands the per-attention-type state cache here.
            self.kv_mgr = None

    class NeuronDeepseekV4FlashForCausalLM(NeuronBaseForCausalLM):
        """Public NxDI wrapper class.

        Round 1: init raises NotImplementedError.  Round 2 wires the
        block scaffold above through ``_NeuronDeepseekV4FlashModel``.
        """

        _model_cls = _NeuronDeepseekV4FlashModel

        @classmethod
        def from_configs(
            cls,
            hf_config: Any,
            neuron_config: Any,
        ) -> "NeuronDeepseekV4FlashForCausalLM":
            _require_nxdi()
            source_config = DeepseekV4FlashInferenceConfig.from_configs(hf_config)
            inference_config = DeepseekV4FlashNeuronInferenceConfig(
                neuron_config=neuron_config, source_config=source_config
            )
            return cls(inference_config)

else:  # pragma: no cover - CPU-only guard

    class NeuronDeepseekV4FlashForCausalLM:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_nxdi()


__all__ = [
    "DSV4_BLOCKWISE_MATMUL_WORKAROUND",
    "DSV4_ROUTED_SCORING_FUNC",
    "DeepseekV4FlashNeuronInferenceConfig",
    "FORBIDDEN_FP8_KV_KEYS",
    "NeuronDeepseekV4FlashForCausalLM",
    "_MQABlock",
    "apply_partial_rope",
    "build_main_rope_cos_sin",
    "build_neuron_config",
    "dsv4_reference_router_forward",
    "dsv4_route_affinities",
]
