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


class _HCACompressor(nn.Module):
    """DSv4-Flash Heavily Compressed Attention compressor (paper §2.3.2).

    Source-cited byte-for-byte against
    ``transformers/models/deepseek_v4/modeling_deepseek_v4.py::
    DeepseekV4HCACompressor`` lines 362-443 (HF SHA
    ``7872f01b1d1fe23eabc4c98b48bffcef5a386062``).

    Owns the four HF layer-subtree tensors that live under
    ``layers.<i>.attn.compressor.*`` (verified against
    ``model.safetensors.index.json`` for the pinned snapshot; shard
    ``model-00005-of-00048.safetensors`` for layer 3, all four keys
    stored dense — no ``.scale`` companion):

      * ``wkv.weight``   [head_dim, hidden_size]     (BF16 on disk;
                                                       ``H·W^{KV}`` in eq. 20)
      * ``wgate.weight`` [head_dim, hidden_size]     (BF16 on disk;
                                                       ``H·W^Z`` in eq. 21 — the
                                                       gate that will feed softmax)
      * ``ape``          [compress_rate, head_dim]   (F32 on disk — the ``B``
                                                       positional bias added to
                                                       each window's gate before
                                                       softmax; wrapper stores
                                                       as bf16 to match module
                                                       dtype, matches HF module
                                                       cast from F32 → bf16 at
                                                       ``__init__`` time)
      * ``norm.weight``  [head_dim]                  (BF16; RMSNorm gain
                                                       applied to the summed
                                                       compressed vector)

    Compression math per closed window (paper §2.3.2 eq. 22-23; HF
    lines 412-422):

      1. ``C = wkv(hidden_states)``,   shape ``[B, S, head_dim]``
      2. ``Z = wgate(hidden_states)``, shape ``[B, S, head_dim]``
      3. Trim ``S`` down to ``usable = (S // compress_rate) * compress_rate``
         source tokens (stateless: HCA has non-overlapping windows and
         the leftover would need cache-buffered persistence which is a
         later-round concern).
      4. Reshape ``C, Z`` to ``[B, n_windows, compress_rate, head_dim]``.
      5. ``Z_bias = Z + ape``  — ape broadcasts across the window
         axis (``[compress_rate, head_dim]`` broadcasts against the last
         two dims of the reshaped tensor); this is HF line 415.
      6. ``w = softmax(Z_bias, dim=window_axis, dtype=torch.float32)``
         — softmax over the ``compress_rate=128`` intra-window axis, in
         fp32 for numerical stability (HF line 417).
      7. ``compressed = kv_norm( (C * w).sum(window_axis) )`` — the
         convex combination becomes one entry per window, then RMSNorm
         with the ``norm`` gain.
      8. Apply the "compress" RoPE at deterministic window position
         ``w_idx * compress_rate + first_window_position`` — HF lines
         419-422.  The caller pre-computes ``cos_win, sin_win`` and
         passes them in (same pattern as :class:`_MQABlock`); no state
         is owned inside the compressor, so ``first_window_position`` is
         a caller argument that stays 0 in the stateless smoke path.

    Returns ``[B, 1, n_windows, head_dim]`` — ready to cat onto
    :meth:`_MQABlock.project_kv`'s output on the KV axis.

    HCA has **no overlap state** (contrast :class:`_CSABlock`'s
    ``overlap_kv``/``overlap_gate``) and **no indexer** (contrast
    ``_CSAIndexer``): every closed window emits one entry, and every
    query whose position has reached that window can attend to it.  The
    caller builds the causal ``block_bias`` via
    :meth:`build_block_bias`.
    """

    # State-dict spelling under this module — matches HF layer subtree
    # verbatim.  Kept as a class attribute so the converter and the test
    # can share the list of names.
    PARAM_KEYS: tuple[str, ...] = (
        "wkv.weight",
        "wgate.weight",
        "ape",
        "norm.weight",
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
            src = config
        self.layer_idx = layer_idx
        self.hidden_size = int(src.hidden_size)
        self.head_dim = int(src.head_dim)
        self.qk_rope_head_dim = int(src.qk_rope_head_dim)
        # HCA is the "128" bucket of the compress_ratios schedule (paper §2.3.2).
        # We refuse to instantiate the compressor at a layer whose ratio is not
        # 128 — pinning this here catches a caller that accidentally reused this
        # class for a CSA layer (where compress_rate=4) or a sliding layer
        # (where the block has no compressor at all).
        ratio = int(src.compress_ratios[layer_idx])
        if ratio != 128:
            raise ValueError(
                f"_HCACompressor requires compress_ratios[{layer_idx}]=128 "
                f"(HCA/heavily_compressed_attention); got {ratio}. HCA is the "
                "non-overlapping m'=128 pool per paper §2.3.2 — refusing to "
                "silently reinterpret a CSA (4) or sliding (0) schedule as HCA."
            )
        self.compress_rate = ratio
        self.rms_eps = float(src.rms_norm_eps)
        self.compress_rope_theta = float(src.compress_rope_theta)
        dtype = getattr(config, "torch_dtype", None) or getattr(
            getattr(config, "neuron_config", None), "torch_dtype", None
        ) or src.torch_dtype
        self._dtype = dtype

        # Weight tensors — plain nn.Parameter so this class is CPU-portable
        # (same rationale documented on _MQABlock).  Names are chosen to
        # match the HF layer subtree byte-for-byte: `wkv.weight`,
        # `wgate.weight`, `ape`, `norm.weight`.
        self.wkv = nn.Linear(self.hidden_size, self.head_dim, bias=False, dtype=dtype)
        self.wgate = nn.Linear(self.hidden_size, self.head_dim, bias=False, dtype=dtype)
        # Absolute Position Encoding — the paper's ``B_j`` positional-bias
        # per intra-window position.  Broadcasts against the last two dims
        # of ``chunk_gate.view(B, n_windows, compress_rate, head_dim)``.
        self.ape = nn.Parameter(
            torch.zeros(self.compress_rate, self.head_dim, dtype=dtype),
            requires_grad=False,
        )
        # RMSNorm gain for the compressed vector.
        self.norm = _MQANormParam(self.head_dim, dtype)

    def compress(
        self,
        hidden_states: torch.Tensor,             # [B, S, hidden]
        cos_win: torch.Tensor,                   # [B, n_windows, rope_dim/2]
        sin_win: torch.Tensor,                   # [B, n_windows, rope_dim/2]
    ) -> torch.Tensor:
        """Emit compressed KV entries — one per closed non-overlapping window.

        Returns ``[B, 1, n_windows, head_dim]`` — the KV axis is single-headed
        (matching :meth:`_MQABlock.project_kv`) so it cats cleanly onto the
        main KV along ``dim=2``.

        Stateless: leftover source tokens shorter than one window are
        dropped.  The caller must feed sequences of length divisible by
        ``compress_rate`` (or accept the tail truncation — the smoke test
        uses ``S=256`` = 2 × 128 so no tail is dropped).
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                f"_HCACompressor.compress expects [B, S, hidden]; got shape "
                f"{tuple(hidden_states.shape)}"
            )
        batch, seq, _ = hidden_states.shape
        usable = (seq // self.compress_rate) * self.compress_rate
        if usable == 0:
            return hidden_states.new_zeros((batch, 1, 0, self.head_dim))

        chunk = hidden_states[:, :usable]
        kv = self.wkv(chunk)                                          # [B, U, D]
        gate = self.wgate(chunk)                                      # [B, U, D]
        n_windows = usable // self.compress_rate

        # Reshape into per-window tiles.  ``self.ape`` broadcasts across the
        # (batch, n_windows) leading dims to match the [compress_rate, D]
        # trailing shape — same broadcast HF relies on at line 415.
        kv_r = kv.view(batch, n_windows, self.compress_rate, self.head_dim)
        gate_r = gate.view(batch, n_windows, self.compress_rate, self.head_dim) + self.ape

        # Softmax over the intra-window axis in fp32 for stability (HF line
        # 417).  Cast back to kv's dtype for the weighted sum so the
        # accumulator dtype tracks the source-tensor dtype.
        softmax_w = gate_r.softmax(dim=2, dtype=torch.float32).to(kv_r.dtype)
        compressed = (kv_r * softmax_w).sum(dim=2)                    # [B, n_windows, D]

        # RMSNorm with the learned gain.
        compressed = _weighted_rms_norm(compressed, self.norm.weight, self.rms_eps)

        # Apply the compressor's own RoPE at window positions.  The KV axis
        # is single-headed here — unsqueeze a head axis so
        # ``apply_partial_rope`` (which expects [B, H, S, D] with
        # ``unsqueeze_dim=1``) broadcasts cleanly.  Result is
        # [B, 1, n_windows, D] which is exactly the shape the outer
        # attention wants on its KV axis.
        return apply_partial_rope(
            compressed.unsqueeze(1), cos_win, sin_win, unsqueeze_dim=1
        )

    def build_block_bias(
        self,
        position_ids: torch.Tensor,              # [B, S]
        compressed_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Additive-log-space bias forbidding queries from seeing compressed
        entries whose source window has not yet closed at that query's
        position.

        Source-cited: ``modeling_deepseek_v4.py:435-443`` — for query at
        position ``t``, compressed entry index ``w`` is visible iff
        ``w < (t + 1) // compress_rate`` (equivalently: entry ``w`` covers
        source tokens ``[w*compress_rate, (w+1)*compress_rate)``, so a
        query cannot legally see it until ``t`` has reached the end of
        that window — the source information the entry aggregates is
        strictly the *past* from that query's viewpoint).

        Returns ``[B, 1, S, compressed_len]`` with ``0.0`` on visible
        slots and ``-inf`` on forbidden slots, or ``None`` when there is
        nothing to attend to (or single-token decode where HF short-
        circuits to ``None`` at line 432).
        """
        if compressed_len == 0:
            return None
        batch, seq = position_ids.shape
        if seq == 1:
            # HF line 432 short-circuits decode-of-one to a None block_bias
            # because the single-query case cannot violate the constraint —
            # there is only one row and either the entry is visible or it
            # is not, resolved by the top-k logic HF uses for CSA.  For HCA
            # there is no indexer, so we mirror the None return; the caller
            # then simply cats the compressed entries with no additive bias
            # and the causal step lands in the outer attention mask.
            #
            # For prefill (seq > 1) we build the full [S, T] mask.
            return None
        entry_indices = torch.arange(compressed_len, device=device)
        causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]
        block_bias = torch.zeros(
            (batch, 1, seq, compressed_len), dtype=dtype, device=device
        )
        block_bias = block_bias.masked_fill(
            entry_indices.view(1, 1, 1, -1)
            >= causal_threshold.unsqueeze(1).unsqueeze(-1),
            float("-inf"),
        )
        return block_bias


class _HCABlock(nn.Module):
    """DSv4-Flash Heavily Compressed Attention block — the composability
    check for what :class:`_MQABlock` just landed.

    Source-cited byte-for-byte against
    ``transformers/models/deepseek_v4/modeling_deepseek_v4.py::
    DeepseekV4Attention.forward`` lines 801-873 restricted to the
    ``layer_type == "heavily_compressed_attention"`` branch
    (``self.compressor = DeepseekV4HCACompressor(...)``,
    ``self.rope_layer_type = "compress"``).

    Composes:
      * :class:`_MQABlock` — reuses its ``project_q``, ``project_kv``,
        ``attend_and_project`` boundaries verbatim.  No fork of the MQA
        math.
      * :class:`_HCACompressor` — emits ``[B, 1, n_windows, head_dim]``
        compressed KV entries and the ``[B, 1, S, T_compressed]``
        block_bias that keeps every query causal with respect to the
        window it just closed.

    HCA-specific arrangement (HF lines 828-844):

      1. ``q = mqa.project_q(hidden_states, cos_compress, sin_compress)``
      2. ``kv_main = mqa.project_kv(hidden_states, cos_compress, sin_compress)``
         — both use the "compress" rope (θ = 160 000) because the
         compressor emits KV entries pre-rotated with that same rope, so
         the inner-product ``q · k`` needs to be in the same rotation
         frame.  This is the ONLY reason the outer Q/KV use compress rope
         on HCA layers even though they see per-source-token positions,
         not window positions — matching frames matters more than
         matching theta scale.
      3. ``compressed_kv = compressor.compress(hidden_states, cos_win, sin_win)``
         — shape ``[B, 1, T_c, head_dim]`` with the compressor's own
         "compress" rope applied at window positions.
      4. ``kv_extended = cat([kv_main, compressed_kv], dim=2)`` — HF line
         832.  The extension is on the KV/time axis.
      5. Attention-mask extension: cat ``block_bias`` onto whatever the
         caller passed (or synthesise a full-visibility mask over
         ``kv_main`` if the caller passed None) — HF lines 840-844.
      6. ``mqa.attend_and_project(q, kv_extended, cos_compress,
         sin_compress, attention_mask=extended)`` — inherits the
         per-head attention sink softmax + the ``-sin`` conjugate
         rotation on the output rope slice + the grouped output
         projection unchanged from :class:`_MQABlock`.

    **No overlap state, no indexer, no input-id side channel** — HCA is
    the simplest attention family beyond sliding-only, and by
    construction the smallest exercise of the ``_MQABlock`` boundary
    that adds a new mechanism (the compressor's per-window pool + causal
    block_bias).
    """

    def __init__(
        self,
        config: Any,
        *,
        layer_idx: int,
    ) -> None:
        super().__init__()
        src = getattr(config, "source_config", None)
        if src is None:
            src = config
        # Verify the layer_type schedule pins HCA at this index.  Fail-loud;
        # a mismatched call is a caller bug that would compose the wrong
        # attention math without complaint (an HCA compressor with the CSA
        # `_MQABlock` main-rope choice would silently break the inner-product
        # rotation frame).
        got = src.layer_types[layer_idx]
        if got != "heavily_compressed_attention":
            raise ValueError(
                f"_HCABlock requires heavily_compressed_attention at "
                f"layer_idx={layer_idx}; got layer_types[{layer_idx}]={got!r}. "
                "The frozen HF schedule places HCA only at the '128' entries "
                "of compress_ratios (paper §2.3.2)."
            )
        self.layer_idx = layer_idx
        # The wrapper module-tree spells the attention subtree as `attn` so
        # the state-dict lands under `layers.<i>.attn.*` (matching the HF
        # snapshot's on-disk names verbatim).  Storing _MQABlock and
        # _HCACompressor as siblings here means the compressor lives under
        # `layers.<i>.attn.compressor.*` — again matching HF byte-for-byte.
        self.mqa = _MQABlock(config, layer_idx=layer_idx)
        self.compressor = _HCACompressor(config, layer_idx=layer_idx)
        self.compress_rate = self.compressor.compress_rate

    def forward(
        self,
        hidden_states: torch.Tensor,             # [B, S, hidden]
        cos: torch.Tensor,                       # [B, S, rope_dim/2] "compress"
        sin: torch.Tensor,                       # [B, S, rope_dim/2] "compress"
        cos_win: torch.Tensor,                   # [B, n_windows, rope_dim/2]
        sin_win: torch.Tensor,                   # [B, n_windows, rope_dim/2]
        position_ids: torch.Tensor,              # [B, S]
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full HCA block forward — see class docstring for the source-cited
        step-by-step decomposition."""
        if hidden_states.ndim != 3:
            raise ValueError(
                f"_HCABlock expects hidden_states [B, S, hidden]; got shape "
                f"{tuple(hidden_states.shape)}"
            )
        batch, seq, _ = hidden_states.shape
        if position_ids.shape != (batch, seq):
            raise ValueError(
                f"_HCABlock expects position_ids shape ({batch}, {seq}); got "
                f"{tuple(position_ids.shape)}"
            )

        # 1. Q + main KV via _MQABlock hooks.
        q, _q_residual = self.mqa.project_q(hidden_states, cos, sin)
        kv_main = self.mqa.project_kv(hidden_states, cos, sin)       # [B, 1, S, D]

        # 2. HCA compressor emits [B, 1, T_c, D] compressed KV entries.
        compressed_kv = self.compressor.compress(hidden_states, cos_win, sin_win)
        t_compressed = compressed_kv.shape[2]

        # 3. Build the block-bias over the compressed slots (per-query
        # causality — early queries cannot see later-window entries).
        block_bias = self.compressor.build_block_bias(
            position_ids,
            t_compressed,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # 4. Cat compressed KV entries onto the main KV axis (HF line 832).
        kv_extended = torch.cat([kv_main, compressed_kv], dim=2)      # [B, 1, S+T_c, D]

        # 5. Extend the attention mask.  Contract mirrors HF lines 840-844:
        #    * If the caller passed an attention_mask AND block_bias exists
        #      → cat block_bias onto the trailing KV axis (in the caller's
        #        dtype).
        #    * If the caller passed an attention_mask AND block_bias is None
        #      → pad zeros for the new slots (they are all visible).
        #    * If the caller passed NO attention_mask AND block_bias exists
        #      → synthesise a zero prefix over kv_main and cat block_bias so
        #        the compressed slots still receive the causality bias.
        #        This matches HF's behaviour when the caller passes
        #        `attention_mask=None`: the full mask is zero over kv_main
        #        (no masking) but block_bias must still apply.
        if attention_mask is not None:
            if block_bias is not None:
                extended_mask = torch.cat(
                    [attention_mask, block_bias.to(attention_mask.dtype)], dim=-1
                )
            else:
                extended_mask = F.pad(
                    attention_mask, (0, t_compressed), value=0.0
                )
        else:
            if block_bias is not None:
                zeros_prefix = hidden_states.new_zeros(
                    (batch, 1, seq, kv_main.shape[2])
                )
                extended_mask = torch.cat(
                    [zeros_prefix, block_bias.to(zeros_prefix.dtype)], dim=-1
                )
            else:
                extended_mask = None

        # 6. Delegate to _MQABlock.attend_and_project — inherits the
        # attention-sink softmax, the -sin conjugate rotation on the output
        # rope slice (using the same "compress" rope), and the grouped
        # output projection.  Note that the -sin conjugate rotation is only
        # applied to `attn_output` in the *outer* attention math; the
        # compressor's internal rope on `compressed_kv` is a K-side rotation
        # that participates in the inner-product with q's rotation, and the
        # conjugate on the output re-aligns the head-space direction of
        # every KV entry — main and compressed alike — to be a relative-
        # distance function of the query position, per paper eq. 26.
        return self.mqa.attend_and_project(
            q, kv_extended, cos, sin, attention_mask=extended_mask
        )


class _CSAOverlapCompressor(nn.Module):
    """Two-series overlap-aware compressor shared by :class:`_CSABlock` and
    :class:`_LightningIndexerHead`.

    Source-cited byte-for-byte against
    ``transformers/models/deepseek_v4/modeling_deepseek_v4.py``:

      * ``DeepseekV4CSACompressor.__init__`` (lines 612-621) — same 4-tensor
        wrapper-tree layout with 2 * head_dim wide ``wkv`` / ``wgate`` / ``ape``
        projections.
      * ``DeepseekV4CSACompressor.forward`` (lines 623-702) — the Ca/Cb window
        layout (paper §2.3.1 eq. 9-12), softmax over ``2 * compress_rate``
        slots per window, RoPE at window positions, overlap-state read/write.
      * ``DeepseekV4CSACache.update_overlap_state`` (lines 286-300) — persists
        ``chunk[:, -1, :, :head_dim]`` (the *Ca* slice of the last full window)
        so the next forward call's window-0 can consume it as its "prior
        window's Ca" slot; Cb of the last window is folded into that window's
        emitted compressed entry and is never read again.

    Owns 4 tensors under the on-disk subtree ``<name>.<one of>``
    (matches the HF layer subtree verbatim on disk):

      * ``wkv.weight``   [2 * head_dim, hidden_size]   (BF16 on disk; the
                                                         Ca (first half) +
                                                         Cb (second half)
                                                         projections of
                                                         ``H·W^{KV}`` in
                                                         paper eq. 20)
      * ``wgate.weight`` [2 * head_dim, hidden_size]   (BF16 on disk; the
                                                         Ca/Cb gate that
                                                         feeds softmax)
      * ``ape``          [compress_rate, 2 * head_dim] (F32 on disk — the
                                                         absolute-position
                                                         bias broadcast
                                                         across window axis)
      * ``norm.weight``  [head_dim]                    (BF16 on disk;
                                                         RMSNorm gain over
                                                         the *pooled* (single-
                                                         width) compressed
                                                         vector)

    Two callers:

      * The outer CSA attention compressor uses ``head_dim = 512``,
        ``compress_rate = 4`` and its output goes into the KV-catenation of
        the main attention.
      * The Lightning Indexer's internal compressor uses
        ``head_dim = index_head_dim = 128`` at the same
        ``compress_rate = 4`` and its output feeds the scorer's inner
        product with the indexer's Q_B projection.

    **Stateful contract.**  Call it with:

      * ``overlap_kv_prev`` / ``overlap_gate_prev`` = the *previous* forward
        call's returned ``new_overlap_kv`` / ``new_overlap_gate`` (or None
        on the very first call).  Shape ``[B, compress_rate, head_dim]``.

    Return value ``(compressed, new_overlap_kv, new_overlap_gate)``:

      * ``compressed [B, 1, n_windows, head_dim]`` — ready to cat onto the
        main KV axis (outer CSA) or feed the indexer scorer.
      * ``new_overlap_kv`` / ``new_overlap_gate [B, compress_rate, head_dim]``
        — the Ca slice of *this* call's last window, for the next call.

    **Statelessness of window 0 on the FIRST call.**  When
    ``overlap_kv_prev is None``, window 0's first-half (Ca of the phantom
    prior window) stays zero-kv with ``-inf`` gate — softmax weight 0.  This
    matches HF line 656-657 exactly (``new_kv`` initialised zeros,
    ``new_gate`` initialised ``-inf``); the "no prior window on the first
    call" case is a first-class initialisation, not a special-case branch.
    """

    PARAM_KEYS: tuple[str, ...] = (
        "wkv.weight",
        "wgate.weight",
        "ape",
        "norm.weight",
    )

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        compress_rate: int,
        rms_eps: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if compress_rate <= 0:
            raise ValueError(f"compress_rate must be positive; got {compress_rate}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive; got {head_dim}")
        self.hidden_size = int(hidden_size)
        self.head_dim = int(head_dim)
        self.compress_rate = int(compress_rate)
        self.rms_eps = float(rms_eps)
        self._dtype = dtype
        # Note: 2 * head_dim projection width — the Ca (first half) + Cb
        # (second half) split (paper §2.3.1 eq. 9-12).
        self.wkv = nn.Linear(
            self.hidden_size, 2 * self.head_dim, bias=False, dtype=dtype
        )
        self.wgate = nn.Linear(
            self.hidden_size, 2 * self.head_dim, bias=False, dtype=dtype
        )
        # ape shape mirrors HF: [compress_rate, 2 * head_dim] — broadcasts
        # against the last two dims of chunk_gate reshaped to
        # [B, n_windows, compress_rate, 2 * head_dim].
        self.ape = nn.Parameter(
            torch.zeros(self.compress_rate, 2 * self.head_dim, dtype=dtype),
            requires_grad=False,
        )
        # RMSNorm gain applied to the *pooled* per-window vector, which has
        # width `head_dim` (not 2*head_dim) — the softmax convex combination
        # collapses the 2*compress_rate axis and the last dim is the single
        # `head_dim` of the compressed representation.
        self.norm = _MQANormParam(self.head_dim, dtype)

    def compress(
        self,
        hidden_states: torch.Tensor,               # [B, S, hidden]
        cos_win: torch.Tensor,                     # [B, n_windows, rope_dim/2]
        sin_win: torch.Tensor,                     # [B, n_windows, rope_dim/2]
        overlap_kv_prev: torch.Tensor | None = None,   # [B, compress_rate, head_dim]
        overlap_gate_prev: torch.Tensor | None = None, # [B, compress_rate, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Emit compressed entries and the state hand-off for the next call.

        Returns
        -------
        compressed : [B, 1, n_windows, head_dim] — pre-rotated with the
            "compress" RoPE at window positions ``[0, m, 2m, ..., (n_win-1)*m]``.
            (The rope is applied at *within-this-call* window positions;
            callers that continue a prior call must offset those positions
            with ``first_window_position`` — the state-aliasing NEFF wiring
            handles that via a separate ``entry_count`` counter, mirror of
            HF's ``DeepseekV4CSACache.entry_count``.  The stateless single-
            shot path used in the CPU-portable smoke test starts at 0.)
        new_overlap_kv : [B, compress_rate, head_dim] — the Ca slice of
            *this* call's last window, ready to become the next call's
            ``overlap_kv_prev``.
        new_overlap_gate : [B, compress_rate, head_dim] — the Ca gate slice
            paired with the KV state.

        When there are no complete windows (``S < compress_rate``), returns
        an empty ``compressed`` tensor and leaves the overlap state
        unchanged (equal to the prev values, or None-safe zeros).  HF's
        cache path handles the same case by draining `store_compression_weights`
        into the buffer without emitting an entry; the stateless smoke does
        not exercise the buffered-partial-window branch (Round 6 wires the
        NxDI-aliased ``buffer_kv`` / ``buffer_gate`` tensors alongside the
        overlap state — same alias pair contract as GLM-5.3-Flash's KDA
        conv-state entry declared under ``state_cache_specs``).
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                f"_CSAOverlapCompressor.compress expects [B, S, hidden]; "
                f"got shape {tuple(hidden_states.shape)}"
            )
        batch, seq, _ = hidden_states.shape
        usable = (seq // self.compress_rate) * self.compress_rate
        if usable == 0:
            empty_c = hidden_states.new_zeros((batch, 1, 0, self.head_dim))
            empty_state = hidden_states.new_zeros(
                (batch, self.compress_rate, self.head_dim)
            )
            return empty_c, empty_state, empty_state
        chunk = hidden_states[:, :usable]
        kv = self.wkv(chunk)                                    # [B, U, 2*D]
        gate = self.wgate(chunk)                                # [B, U, 2*D]
        n_windows = usable // self.compress_rate
        # Reshape to per-window tiles [B, n_windows, compress_rate, 2*D].
        chunk_kv = kv.view(batch, n_windows, self.compress_rate, -1)
        chunk_gate = gate.view(batch, n_windows, self.compress_rate, -1) + self.ape

        # --------- Ca/Cb window scheme (paper §2.3.1) — HF lines 656-669. ---------
        # new_kv / new_gate carry `2 * compress_rate` slots per window (width doubled).
        # Slot layout:
        #   [0 : compress_rate)                 = Ca slice of the *previous* window
        #                                          (this call's window-0 gets
        #                                          `overlap_kv_prev`; window-j (j>0)
        #                                          gets `chunk_kv[:, j-1, :, :head_dim]`).
        #   [compress_rate : 2*compress_rate)   = Cb slice of the *current* window.
        # Cells left unset stay zero-kv / -inf-gate → softmax weight 0.
        ratio = self.compress_rate
        new_kv = chunk_kv.new_zeros(
            (batch, n_windows, 2 * ratio, self.head_dim)
        )
        new_gate = chunk_gate.new_full(
            (batch, n_windows, 2 * ratio, self.head_dim), float("-inf")
        )
        # Cb of the current window → second half.
        new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
        new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim:]
        # Ca of the prior window → first half (windows 1..n_windows-1).
        if n_windows > 1:
            new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
            new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
        # Ca of the *last window of the previous forward call* → window-0
        # first half.  On the very first call `overlap_*_prev` is None and
        # window-0's first half stays zero-kv / -inf-gate — softmax weight 0.
        if overlap_kv_prev is not None:
            if overlap_gate_prev is None:
                raise ValueError(
                    "_CSAOverlapCompressor: overlap_gate_prev must be provided "
                    "whenever overlap_kv_prev is (they are paired; a gate slot "
                    "with no KV feeds softmax(-inf) → weight 0 which discards "
                    "the KV silently — refusing to accept the pair-broken call)."
                )
            if tuple(overlap_kv_prev.shape) != (batch, ratio, self.head_dim):
                raise ValueError(
                    "_CSAOverlapCompressor: overlap_kv_prev shape "
                    f"{tuple(overlap_kv_prev.shape)} must be "
                    f"({batch}, {ratio}, {self.head_dim})"
                )
            if tuple(overlap_gate_prev.shape) != (batch, ratio, self.head_dim):
                raise ValueError(
                    "_CSAOverlapCompressor: overlap_gate_prev shape "
                    f"{tuple(overlap_gate_prev.shape)} must be "
                    f"({batch}, {ratio}, {self.head_dim})"
                )
            new_kv[:, 0, :ratio] = overlap_kv_prev.to(new_kv.dtype)
            new_gate[:, 0, :ratio] = overlap_gate_prev.to(new_gate.dtype)

        # Softmax over the 2*compress_rate intra-window slots in fp32 for
        # stability (HF line 671-675) — matches HCA's fp32 softmax reason.
        softmax_w = new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)
        compressed = (new_kv * softmax_w).sum(dim=2)               # [B, n_windows, D]
        compressed = _weighted_rms_norm(compressed, self.norm.weight, self.rms_eps)
        # Compress-rope at window positions; caller pre-computes cos_win, sin_win.
        compressed = apply_partial_rope(
            compressed.unsqueeze(1), cos_win, sin_win, unsqueeze_dim=1
        )                                                          # [B, 1, n_win, D]

        # Persist the Ca slice of *this* call's last window for the next call.
        new_overlap_kv = chunk_kv[:, -1, :, : self.head_dim].clone()
        new_overlap_gate = chunk_gate[:, -1, :, : self.head_dim].clone()

        return compressed, new_overlap_kv, new_overlap_gate


class _LightningIndexerHead(nn.Module):
    """DSv4-Flash CSA Lightning Indexer (paper §2.3.1 eq. 13-17).

    Source-cited byte-for-byte against
    ``transformers/models/deepseek_v4/modeling_deepseek_v4.py``:

      * ``DeepseekV4Indexer.__init__`` (lines 493-505) — same 6-tensor
        wrapper-tree layout: an inner overlap-aware compressor at
        ``index_head_dim``, a per-head Q_B projection from ``q_lora_rank``
        to ``index_n_heads * index_head_dim``, and the scorer's
        per-head weight projection ``weights_proj`` from ``hidden_size``
        to ``index_n_heads``.
      * ``DeepseekV4Indexer.forward`` (lines 507-586) — the overlap
        compressor at index_head_dim, RoPE on both compressed keys (window
        positions) and queries (per-token positions), scorer inner product
        with ReLU + fp32 softmax scale, top-``index_topk`` selection with
        the ``-1`` sentinel for indices that would violate per-query causality.
      * ``DeepseekV4IndexerScorer.forward`` (lines 455-459) — the
        ``∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)`` reduction:
        ``scores * (index_head_dim**-0.5)`` scaled by ``(index_n_heads**-0.5)``
        weight normalisation.

    Owns 7 wrapper-tree tensors (matches the HF layer subtree
    verbatim on disk under ``layers.<i>.attn.indexer.``):

      * ``compressor.wkv.weight``   [2 * index_head_dim, hidden_size]  BF16
      * ``compressor.wgate.weight`` [2 * index_head_dim, hidden_size]  BF16
      * ``compressor.ape``          [compress_rate, 2 * index_head_dim]  BF16
      * ``compressor.norm.weight``  [index_head_dim]                  BF16
      * ``wq_b.weight`` [index_n_heads * index_head_dim, q_lora_rank] FP8
        (on-disk paired ``.scale`` companion follows the same UE8M0
        convention as the outer MQA weights.)
      * ``weights_proj.weight``     [index_n_heads, hidden_size]      BF16

    Forward returns ``(top_k_indices, new_overlap_kv, new_overlap_gate)``:

      * ``top_k_indices [B, S, K]`` int64 — indices into the compressed KV
        axis to gather per query.  ``K = min(index_topk, compressed_len)``.
        Entries whose selected index would violate the query's causality
        threshold ``(position_ids + 1) // compress_rate`` are replaced with
        the ``-1`` sentinel (HF line 583-584); the CSA block's `block_bias`
        scatter drops them.
      * ``new_overlap_kv / new_overlap_gate [B, compress_rate, index_head_dim]``
        — the indexer's own overlap state, structurally identical to the
        outer CSA compressor's overlap state but at ``index_head_dim`` (128)
        instead of ``head_dim`` (512).  This is the second aliased state
        pair the CSA block declares under ``state_cache_specs``.
    """

    def __init__(
        self,
        config: Any,
        *,
        layer_idx: int,
    ) -> None:
        super().__init__()
        src = getattr(config, "source_config", None)
        if src is None:
            src = config
        self.layer_idx = layer_idx
        self.hidden_size = int(src.hidden_size)
        self.q_lora_rank = int(src.q_lora_rank)
        self.head_dim = int(src.index_head_dim)
        self.num_heads = int(src.index_n_heads)
        self.index_topk = int(src.index_topk)
        self.qk_rope_head_dim = int(src.qk_rope_head_dim)
        self.rms_eps = float(src.rms_norm_eps)
        ratio = int(src.compress_ratios[layer_idx])
        if ratio != 4:
            raise ValueError(
                f"_LightningIndexerHead requires compress_ratios[{layer_idx}]=4 "
                f"(CSA); got {ratio}. Only CSA layers own an indexer per HF "
                "COMPRESSOR_CLASSES table (modeling_deepseek_v4.py:748-752)."
            )
        self.compress_rate = ratio
        dtype = getattr(config, "torch_dtype", None) or getattr(
            getattr(config, "neuron_config", None), "torch_dtype", None
        ) or src.torch_dtype
        self._dtype = dtype

        # Softmax scale for the inner product (HF line 451): head_dim**-0.5.
        self.softmax_scale = self.head_dim ** -0.5
        # Weight normalisation for the sum over heads (HF line 452):
        # index_n_heads**-0.5.  Applied to weights_proj output, not to inputs.
        self.weights_scaling = self.num_heads ** -0.5

        # Inner compressor at index_head_dim — same overlap-aware Ca/Cb
        # scheme as the outer CSA compressor, minus the outer's larger
        # head_dim.
        self.compressor = _CSAOverlapCompressor(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            compress_rate=self.compress_rate,
            rms_eps=self.rms_eps,
            dtype=dtype,
        )
        # Q_B projection: q_residual [B, S, q_lora_rank] →
        # [B, S, index_n_heads * index_head_dim].  This is separate from the
        # outer MQA's Q_B (which projects to num_attention_heads * head_dim);
        # the shared piece is q_residual (post Q_A + q_norm), computed once
        # by the outer MQA and threaded in here.
        self.wq_b = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
            dtype=dtype,
        )
        # Scorer's per-head weight projection.
        self.weights_proj = nn.Linear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,             # [B, S, hidden]
        q_residual: torch.Tensor,                # [B, S, q_lora_rank]
        cos: torch.Tensor,                       # [B, S, rope_dim/2] "compress"
        sin: torch.Tensor,                       # [B, S, rope_dim/2] "compress"
        cos_win: torch.Tensor,                   # [B, n_windows, rope_dim/2]
        sin_win: torch.Tensor,                   # [B, n_windows, rope_dim/2]
        position_ids: torch.Tensor,              # [B, S]
        overlap_kv_prev: torch.Tensor | None = None,
        overlap_gate_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-query top-``K = min(index_topk, compressed_len)`` indices.

        Composability contract: the outer CSA block passes the *same*
        `cos, sin, cos_win, sin_win` tables to both this method and its own
        main-attention path.  HF collapses the two rope tables into a single
        `{"main", "compress"}` dict and every CSA-family caller (outer
        compressor, indexer, main attention) reads the same `"compress"`
        entry from it; that is why the same rope frame keeps queries and
        compressed keys inner-product-comparable.

        Returns
        -------
        top_k_indices : [B, S, K] int64 — the per-query selected compressed
            entries, with ``-1`` sentinel for causality-violating picks.
        new_overlap_kv, new_overlap_gate : [B, compress_rate, index_head_dim]
            — indexer's own Ca overlap state for the next forward call.
        """
        batch, seq, _ = hidden_states.shape
        # Indexer's inner compressor emits [B, 1, n_win, index_head_dim].
        compressed_kv_bh, new_overlap_kv, new_overlap_gate = self.compressor.compress(
            hidden_states,
            cos_win,
            sin_win,
            overlap_kv_prev=overlap_kv_prev,
            overlap_gate_prev=overlap_gate_prev,
        )
        # HF's scorer sees compressed_kv as [B, T, D] (line 455-459: it does
        # `compressed_kv.transpose(-1, -2).float().unsqueeze(1)` giving
        # [B, 1, D, T]).  We match by squeezing the head axis.
        compressed_kv = compressed_kv_bh.squeeze(1)                # [B, T, D]
        compressed_len = compressed_kv.shape[1]

        # Q_B projection + partial RoPE at per-source-token positions (HF
        # lines 563-565).  q has shape [B, S, H_idx, D_idx]; rope is applied
        # via `apply_partial_rope` with unsqueeze_dim=1 (adds head-broadcast
        # axis to cos/sin) on the [B, H_idx, S, D_idx] transpose, then
        # transposed back to [B, S, H_idx, D_idx] per HF.
        q_flat = self.wq_b(q_residual)                             # [B, S, H*D]
        q = q_flat.view(batch, seq, self.num_heads, self.head_dim)
        q = apply_partial_rope(q.transpose(1, 2), cos, sin).transpose(1, 2)
        # After transpose-back q is [B, S, H_idx, D_idx].  HF's scorer expects
        # exactly this layout — `matmul(q.float(), compressed_kv...)`.

        # Scorer: `∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)`
        # Inner product q · K per head: q [B, S, H, D] × K [B, T, D]^T → [B, S, H, T].
        # HF line 456: `matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))`
        # ``compressed_kv.transpose(-1, -2).unsqueeze(1)`` → [B, 1, D, T]; broadcasts
        # over the H axis of `q`.  fp32 for numerical stability of ReLU + softmax scale.
        if compressed_len == 0:
            top_k = min(self.index_topk, compressed_len)
            # Return empty top_k of shape [B, S, 0].
            return (
                torch.zeros(
                    (batch, seq, top_k), dtype=torch.int64, device=q.device
                ),
                new_overlap_kv,
                new_overlap_gate,
            )
        q_fp32 = q.float()
        k_fp32 = compressed_kv.transpose(-1, -2).float().unsqueeze(1)   # [B, 1, D, T]
        scores = torch.matmul(q_fp32, k_fp32)                           # [B, S, H, T]
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling
        # weights: [B, S, H] → [B, S, H, 1] for broadcast.
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)      # [B, S, T]

        top_k = min(self.index_topk, compressed_len)
        # Per-query causality — the same threshold the outer CSA compressor's
        # block_bias uses (HF lines 577-584).  Entries at cache position `w`
        # cover source positions `[w*ratio, (w+1)*ratio)`; a query at absolute
        # position `t` may only see them once `t >= w*ratio + ratio - 1` i.e.
        # `w < (t + 1) // ratio`.
        causal_threshold = (position_ids + 1) // self.compress_rate     # [B, S]
        entry_indices = torch.arange(
            compressed_len, device=index_scores.device
        )
        future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
        index_scores = index_scores.masked_fill(future_mask, float("-inf"))
        # `topk` — HF line 582; picks that still land ≥ causal_threshold
        # (only possible when there are fewer legal entries than K) are
        # tagged with the ``-1`` sentinel HF line 583-584 defines.
        top_k_indices = index_scores.topk(top_k, dim=-1).indices        # [B, S, k]
        invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
        top_k_indices = torch.where(
            invalid, torch.full_like(top_k_indices, -1), top_k_indices
        )
        return top_k_indices.to(torch.int64), new_overlap_kv, new_overlap_gate


class _CSABlock(nn.Module):
    """DSv4-Flash Compressed Sparse Attention block (paper §2.3.1).

    Source-cited byte-for-byte against
    ``transformers/models/deepseek_v4/modeling_deepseek_v4.py``:

      * ``DeepseekV4Attention.forward`` (lines 801-873) restricted to the
        ``layer_type == "compressed_sparse_attention"`` branch.
        Note the CSA branch uses the *same* ``"compress"`` RoPE frame as
        HCA (line 776-777: ``self.rope_layer_type = "compress"`` for any
        non-``sliding_attention`` layer).
      * ``DeepseekV4CSACompressor.forward`` (lines 623-702) — the outer
        compressor, driven by :class:`_CSAOverlapCompressor` at
        ``head_dim=512``.  Its returned ``block_bias`` (line 700-702) is
        the *indexer-gated* top-K mask (built from the Lightning Indexer's
        top_k_indices), NOT the HCA-style pure causality mask.
      * ``DeepseekV4Indexer.forward`` (lines 507-586) — the Lightning
        Indexer, driven by :class:`_LightningIndexerHead`.

    Composability with :class:`_MQABlock`.  CSA is the most complex family
    but the shared boundary is identical to HCA's — the block reuses the
    MQA hooks (``project_q``, ``project_kv``, ``attend_and_project``)
    verbatim.  Two new mechanisms sit on top:

      1. **Overlap-state aliasing.**  The outer CSA compressor and the
         indexer's inner compressor each own a pair of ``(overlap_kv,
         overlap_gate)`` tensors of shape ``[B, compress_rate, head_dim]``.
         Both are declared under :meth:`state_cache_specs` for NxDI's
         ``input_output_aliases`` at NEFF wiring time — same alias-pair
         mechanism GLM-5.3-Flash's KDA uses for its recurrent state (see
         ``glm53_flash/neuron_wrapper.py::_KDABlock.state_cache_specs``).
         A misaligned aliasing wire would silently corrupt every decode
         step's compressed entries — this is the load-bearing new
         verification discipline for CSA.
      2. **Lightning Indexer top-K gating.**  Only the top-``K =
         min(index_topk, compressed_len)`` compressed entries per query
         are visible; the rest are pushed to ``-inf`` in the extended
         attention mask.  The reduction over the compressed axis becomes
         O(K * S) instead of O(T_c * S).

    Wrapper-tree keys (verified against the HF layer subtree; disk names
    for the ``indexer.*`` subtree match HF's checkpoint spelling
    verbatim, not the HF *class* attribute names).  For one CSA layer i:

      * 8 MQA params under ``mqa.<one of PARAM_KEYS>``
      * 4 CSA compressor params under ``compressor.<one of PARAM_KEYS>``
      * 4 indexer inner-compressor params under
        ``indexer.compressor.<one of PARAM_KEYS>``
      * 2 indexer projection params: ``indexer.wq_b.weight``,
        ``indexer.weights_proj.weight``
      * Total: 8 + 4 + 4 + 2 = 18 params per CSA layer (plus the sibling
        ``layers.<i>.attn_norm.weight`` owned at the decoder-layer level).
    """

    def __init__(
        self,
        config: Any,
        *,
        layer_idx: int,
    ) -> None:
        super().__init__()
        src = getattr(config, "source_config", None)
        if src is None:
            src = config
        got = src.layer_types[layer_idx]
        if got != "compressed_sparse_attention":
            raise ValueError(
                f"_CSABlock requires compressed_sparse_attention at "
                f"layer_idx={layer_idx}; got layer_types[{layer_idx}]={got!r}. "
                "The frozen HF schedule places CSA only at the '4' entries of "
                "compress_ratios (paper §2.3.1)."
            )
        ratio = int(src.compress_ratios[layer_idx])
        if ratio != 4:
            raise ValueError(
                f"_CSABlock requires compress_ratios[{layer_idx}]=4 (CSA); "
                f"got {ratio}."
            )
        self.layer_idx = layer_idx
        self.head_dim = int(src.head_dim)
        self.compress_rate = ratio
        self.compress_rope_theta = float(src.compress_rope_theta)
        self.qk_rope_head_dim = int(src.qk_rope_head_dim)
        dtype = getattr(config, "torch_dtype", None) or getattr(
            getattr(config, "neuron_config", None), "torch_dtype", None
        ) or src.torch_dtype
        self._dtype = dtype
        # Same wrapper-tree convention as _HCABlock: `mqa` + `compressor` +
        # (new for CSA) `indexer`.  The state-dict lands under
        # `layers.<i>.attn.{mqa|compressor|indexer}.*` when the wrapper is
        # walked by NxDI's state-dict traversal (verified against
        # `test_hca_1layer.py::test_hca_wrapper_tree_key_set` — the same
        # walk logic applies here).
        self.mqa = _MQABlock(config, layer_idx=layer_idx)
        self.compressor = _CSAOverlapCompressor(
            hidden_size=int(src.hidden_size),
            head_dim=self.head_dim,
            compress_rate=self.compress_rate,
            rms_eps=float(src.rms_norm_eps),
            dtype=dtype,
        )
        self.indexer = _LightningIndexerHead(config, layer_idx=layer_idx)

    def state_cache_specs(
        self, batch: int, seq_len: int
    ) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
        """Aliased state this CSA layer owns, in graph input/output order.

        Same alias-pair contract GLM-5.3-Flash uses for its KDA state (see
        ``glm53_flash/neuron_wrapper.py::_KDABlock.state_cache_specs``):
        each entry becomes an ``nn.Parameter`` on the caller side that NxDI
        aliases via ``input_output_aliases`` — the entry is read as an
        argument to the graph, updated inside forward, and written back to
        the same slot on the output.  The mechanism is identical whether
        the payload is a linear-attention recurrent state (KDA) or an
        overlap Ca-slice (CSA); what matters is that the state-dict spelling
        stays stable across graph rewiring and that the aliased dtype
        matches this class's compute dtype.

        CSA overlap state is *sequence-length independent* — it is a fixed
        ``[compress_rate, head_dim]`` window slice, not a growing per-position
        buffer.  Dropping the ``seq_len`` argument mirrors KDA's contract.
        """
        del seq_len  # CSA overlap size is fixed by compress_rate, not seq_len.
        return [
            (
                "compressor_overlap_kv",
                (batch, self.compress_rate, self.head_dim),
                self._dtype,
            ),
            (
                "compressor_overlap_gate",
                (batch, self.compress_rate, self.head_dim),
                self._dtype,
            ),
            (
                "indexer_overlap_kv",
                (batch, self.compress_rate, self.indexer.head_dim),
                self._dtype,
            ),
            (
                "indexer_overlap_gate",
                (batch, self.compress_rate, self.indexer.head_dim),
                self._dtype,
            ),
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,             # [B, S, hidden]
        cos: torch.Tensor,                       # [B, S, rope_dim/2] "compress"
        sin: torch.Tensor,                       # [B, S, rope_dim/2] "compress"
        cos_win: torch.Tensor,                   # [B, n_windows, rope_dim/2]
        sin_win: torch.Tensor,                   # [B, n_windows, rope_dim/2]
        position_ids: torch.Tensor,              # [B, S]
        attention_mask: torch.Tensor | None = None,
        overlap_state: dict[str, torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """CSA block forward with two overlap-state aliased pairs.

        ``overlap_state`` is an optional dict with keys:
          * ``compressor_overlap_kv``, ``compressor_overlap_gate`` — the outer
            CSA compressor's Ca slice from the *previous* forward call.
          * ``indexer_overlap_kv``, ``indexer_overlap_gate`` — the indexer's
            own Ca slice from the previous forward call.

        Any missing / ``None`` key means "no prior window" — that branch's
        window-0 first half stays zero-kv / -inf-gate (softmax weight 0),
        matching the very-first-call semantics HF's cache initialiser
        provides (``self.overlap_kv[name] = None``).

        Returns ``(output, new_overlap_state)`` where ``new_overlap_state``
        carries the four next-call Ca slices under the same keys.  The
        NxDI-wired NEFF caller pushes them back to the aliased parameters
        after the forward; the CPU-portable smoke test in
        ``tests/test_csa_1layer.py`` uses the dict directly.
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                f"_CSABlock expects hidden_states [B, S, hidden]; got shape "
                f"{tuple(hidden_states.shape)}"
            )
        batch, seq, _ = hidden_states.shape
        if position_ids.shape != (batch, seq):
            raise ValueError(
                f"_CSABlock expects position_ids shape ({batch}, {seq}); got "
                f"{tuple(position_ids.shape)}"
            )
        overlap_state = overlap_state or {}
        comp_kv_prev = overlap_state.get("compressor_overlap_kv")
        comp_gate_prev = overlap_state.get("compressor_overlap_gate")
        idx_kv_prev = overlap_state.get("indexer_overlap_kv")
        idx_gate_prev = overlap_state.get("indexer_overlap_gate")

        # 1. Q + main KV via _MQABlock hooks.  Q_A + q_norm gives q_residual,
        # which the indexer *reuses* (avoids recomputing Q_A twice per layer).
        q, q_residual = self.mqa.project_q(hidden_states, cos, sin)
        kv_main = self.mqa.project_kv(hidden_states, cos, sin)       # [B, 1, S, D]

        # 2. CSA compressor emits [B, 1, T_c, head_dim] compressed KV +
        # new overlap-state tensors for the next forward call.
        compressed_kv, new_comp_kv, new_comp_gate = self.compressor.compress(
            hidden_states,
            cos_win,
            sin_win,
            overlap_kv_prev=comp_kv_prev,
            overlap_gate_prev=comp_gate_prev,
        )
        t_compressed = compressed_kv.shape[2]

        # 3. Lightning Indexer top-K gating over compressed entries.
        top_k_indices, new_idx_kv, new_idx_gate = self.indexer(
            hidden_states,
            q_residual,
            cos,
            sin,
            cos_win,
            sin_win,
            position_ids,
            overlap_kv_prev=idx_kv_prev,
            overlap_gate_prev=idx_gate_prev,
        )                                                             # [B, S, K]

        # 4. Build indexer-gated per-query block_bias (HF lines 693-702).
        # `valid` marks non-sentinel picks; `safe_indices` clamps sentinels
        # into a padding column that the trailing slice drops.  Scatter 0
        # onto the K valid columns and leave every other slot at -inf.
        if t_compressed > 0:
            valid = top_k_indices >= 0
            safe_indices = torch.where(
                valid,
                top_k_indices,
                torch.full_like(top_k_indices, t_compressed),
            )
            block_bias = compressed_kv.new_full(
                (batch, 1, seq, t_compressed + 1), float("-inf")
            )
            block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
            block_bias = block_bias[..., :t_compressed]
        else:
            block_bias = None

        # 5. Cat compressed KV onto main KV axis (HF line 832) + extend mask.
        kv_extended = torch.cat([kv_main, compressed_kv], dim=2)      # [B, 1, S+T_c, D]
        if attention_mask is not None:
            if block_bias is not None:
                extended_mask = torch.cat(
                    [attention_mask, block_bias.to(attention_mask.dtype)], dim=-1
                )
            else:
                extended_mask = F.pad(
                    attention_mask, (0, t_compressed), value=0.0
                )
        else:
            if block_bias is not None:
                zeros_prefix = hidden_states.new_zeros(
                    (batch, 1, seq, kv_main.shape[2])
                )
                extended_mask = torch.cat(
                    [zeros_prefix, block_bias.to(zeros_prefix.dtype)], dim=-1
                )
            else:
                extended_mask = None

        # 6. Delegate to _MQABlock.attend_and_project — inherits the
        # per-head attention-sink softmax, the -sin conjugate rotation on
        # the output rope slice (using the same "compress" rope), and the
        # grouped output projection.  Compressed slots outside the top-K
        # gate get logit -inf via block_bias → softmax weight 0.
        output = self.mqa.attend_and_project(
            q, kv_extended, cos, sin, attention_mask=extended_mask
        )
        new_state = {
            "compressor_overlap_kv": new_comp_kv,
            "compressor_overlap_gate": new_comp_gate,
            "indexer_overlap_kv": new_idx_kv,
            "indexer_overlap_gate": new_idx_gate,
        }
        return output, new_state


def build_sliding_window_causal_mask(
    position_ids: torch.Tensor,
    *,
    sliding_window: int,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Additive-log-space sliding-window causal mask (per-layer).

    Source-cited byte-for-byte against ``transformers/masking_utils.py``:

      * ``causal_mask_function`` (line 80):  ``kv_idx <= q_idx``
      * ``sliding_window_overlay`` (lines 92-101):
            ``kv_idx > q_idx - sliding_window``
      * ``sliding_window_causal_mask_function`` (line 138):
            ``and_masks(sliding_window_overlay(sliding_window),
                        causal_mask_function)``
      * ``LAYER_TYPE_TO_MASK_CREATION_FUNCTION`` (line 1478) routes
        ``"sliding_attention"`` layers through
        ``create_sliding_window_causal_mask``.

    A query at absolute position ``t`` is legal for KV positions ``k``
    iff ``t - sliding_window < k <= t`` — i.e. the sliding window is
    the inclusive interval ``[t - sliding_window + 1, t]``.  Early
    queries (``t < sliding_window``) see all past KV; late queries see
    exactly ``sliding_window`` past KV.

    Returns shape ``[B, 1, S, S]`` with ``0.0`` on visible slots and
    ``-inf`` on forbidden slots.  The head axis is 1 for broadcast
    across the MQA block's ``num_heads`` in ``attend_and_project``.
    """
    if position_ids.ndim != 2:
        raise ValueError(
            f"position_ids must be [B, S]; got shape {tuple(position_ids.shape)}"
        )
    if sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive; got {sliding_window}")
    if device is None:
        device = position_ids.device
    q_broadcast = position_ids.unsqueeze(-1)
    kv_broadcast = position_ids.unsqueeze(-2)
    visible = (kv_broadcast <= q_broadcast) & (
        kv_broadcast > (q_broadcast - int(sliding_window))
    )
    neg_inf = torch.full((), float("-inf"), dtype=dtype, device=device)
    zero = torch.zeros((), dtype=dtype, device=device)
    mask = torch.where(visible, zero, neg_inf)
    return mask.unsqueeze(1)


class _SlidingOnlyAttentionBlock(nn.Module):
    """DSv4-Flash sliding-only attention block (paper layers 0-1 bootstrap).

    Source-cited byte-for-byte against
    ``transformers/models/deepseek_v4/modeling_deepseek_v4.py``:

      * ``DeepseekV4Attention.__init__`` (lines 770-799),
        ``layer_type == "sliding_attention"`` branch:
          - ``self.rope_layer_type = "main"`` (line 777) — plain
            theta=10000 rope, NOT the yarn-scaled "compress" rope
            (theta=160000) that CSA/HCA layers share.
          - ``self.sliding_window = config.sliding_window`` (line 781).
          - ``self.compressor = None`` (line 797-799).
      * ``DeepseekV4Attention.forward`` (lines 801-873), sliding branch:
        Q + KV via _MQABlock hooks, no compressor cat, attend with the
        sliding-window mask, output projection.
      * ``masking_utils.py`` sliding+causal predicate — see
        :func:`build_sliding_window_causal_mask`.

    Composes :class:`_MQABlock` verbatim.  No compressor, no indexer,
    no overlap state, no input-id side channel — the simplest attention
    family, and the base against which CSA/HCA add their compressor
    branches.

    Wrapper-tree keys for one sliding layer i:

      * 8 MQA params under ``mqa.<one of _MQABlock.PARAM_KEYS>``
      * (Sibling ``layers.<i>.attn_norm.weight`` at the decoder-layer
        level — NOT owned by this block, same convention as _HCABlock
        and _CSABlock.)
    """

    def __init__(self, config: Any, *, layer_idx: int) -> None:
        super().__init__()
        src = getattr(config, "source_config", None)
        if src is None:
            src = config
        # Refuse a non-sliding layer index — a caller that routed a CSA/HCA
        # layer through this block would silently drop the compressor
        # branch AND flip the rope table from "compress" (theta=160000) to
        # "main" (theta=10000), producing plausible-looking logits that
        # are quietly wrong.
        got = src.layer_types[layer_idx]
        if got != "sliding_attention":
            raise ValueError(
                f"_SlidingOnlyAttentionBlock requires sliding_attention at "
                f"layer_idx={layer_idx}; got layer_types[{layer_idx}]={got!r}. "
                "The frozen HF schedule places sliding_attention only at "
                "the '0' entries of compress_ratios (layers 0, 1 in the "
                "43-layer bootstrap; trailing 0-entries at 43-45 are the "
                "MTP region and are dropped)."
            )
        ratio = int(src.compress_ratios[layer_idx])
        if ratio != 0:
            raise ValueError(
                f"_SlidingOnlyAttentionBlock requires compress_ratios"
                f"[{layer_idx}]=0 (sliding-only); got {ratio}."
            )
        self.layer_idx = layer_idx
        self.sliding_window = int(src.sliding_window)
        if self.sliding_window <= 0:
            raise ValueError(
                f"sliding_window must be positive; got {self.sliding_window}"
            )
        # Same wrapper-tree convention as _HCABlock / _CSABlock: a single
        # `mqa` sub-module.  State-dict lands under `layers.<i>.attn.mqa.*`.
        self.mqa = _MQABlock(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,             # [B, S, hidden]
        cos: torch.Tensor,                       # [B, S, rope_dim/2] "main"
        sin: torch.Tensor,                       # [B, S, rope_dim/2] "main"
        position_ids: torch.Tensor,              # [B, S]
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sliding-only forward — Q + KV via _MQABlock hooks, sliding-window
        causal mask, delegate to :meth:`_MQABlock.attend_and_project`.

        The ``cos`` / ``sin`` MUST be built from the "main" rope table
        (``rope_theta = src.rope_theta = 10000.0``).  A caller that fed
        the "compress" rope (theta=160000) here would silently produce
        wrong logits.
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                f"_SlidingOnlyAttentionBlock expects hidden_states "
                f"[B, S, hidden]; got shape {tuple(hidden_states.shape)}"
            )
        batch, seq, _ = hidden_states.shape
        if position_ids.shape != (batch, seq):
            raise ValueError(
                f"_SlidingOnlyAttentionBlock expects position_ids shape "
                f"({batch}, {seq}); got {tuple(position_ids.shape)}"
            )

        # 1. Q + KV via _MQABlock hooks (same boundary CSA/HCA compose).
        q, _q_residual = self.mqa.project_q(hidden_states, cos, sin)
        kv = self.mqa.project_kv(hidden_states, cos, sin)                    # [B, 1, S, D]

        # 2. Build (or extend) the additive sliding-window causal mask.
        sliding_mask = build_sliding_window_causal_mask(
            position_ids,
            sliding_window=self.sliding_window,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if attention_mask is not None:
            extended_mask = attention_mask + sliding_mask.to(attention_mask.dtype)
        else:
            extended_mask = sliding_mask

        # 3. Delegate to _MQABlock.attend_and_project.
        return self.mqa.attend_and_project(
            q, kv, cos, sin, attention_mask=extended_mask
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
    "_CSABlock",
    "_CSAOverlapCompressor",
    "_HCABlock",
    "_HCACompressor",
    "_LightningIndexerHead",
    "_MQABlock",
    "_SlidingOnlyAttentionBlock",
    "apply_partial_rope",
    "build_main_rope_cos_sin",
    "build_neuron_config",
    "build_sliding_window_causal_mask",
    "dsv4_reference_router_forward",
    "dsv4_route_affinities",
]
