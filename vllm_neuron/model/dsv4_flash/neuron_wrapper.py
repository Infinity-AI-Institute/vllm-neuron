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

    class _MQABlock(nn.Module):
        """Shared K=V MQA block — replaces GLM-5.3-Flash's `_NoPeMLABlock`.

        DeepSeek-V4-Flash layout (paper §2.3 + transformers 5.15.1
        DeepseekV4Attention):

        * ``num_key_value_heads=1`` — one KV head shared across all 64
          query heads.  ``kv_proj`` writes a single ``[hidden -> head_dim]``
          projection that serves as BOTH K and V.
        * ``q_lora_rank=1024`` — ``wq_a: [hidden, q_lora_rank]`` + norm +
          ``wq_b: [q_lora_rank, num_heads * head_dim]``.
        * Partial RoPE on ``qk_rope_head_dim=64`` of ``head_dim=512``
          channels per head (12.5%), interleaved-pair, base
          ``rope_theta=10000``.  Applied to query rope slice and key rope
          slice.  Output rope slice rotated at position ``-i`` (paper eq. 26)
          so the KV contribution stays a relative-distance function.
        * Per-head learnable attention sink (``attn_sink``, paper eq. 27):
          a constant logit that participates in the softmax denominator.
        * Grouped Output Projection (o_groups=8, o_lora_rank=1024):
          ``o_a: [num_heads * head_dim, o_groups * o_lora_rank]``
          decomposed into 8 group-local ``[num_heads * head_dim,
          o_lora_rank]`` blocks, followed by
          ``o_b: [o_groups * o_lora_rank, hidden]``.

        Sharded via ColumnParallel on the head axis for Q_A/Q_B and on
        the group axis for o_a; RowParallel on o_b.
        """

        def __init__(self, config: Any, *, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self._config = config
            raise NotImplementedError(
                "_MQABlock is Round 2.  Fields required at init time: partial-RoPE "
                "geometry (qk_rope_head_dim=64/head_dim=512), per-head attention-sink "
                "parameter shape [num_heads], grouped output projection axis order "
                "(o_groups=8, o_lora_rank=1024).  See ENABLEMENT-DRAFT §2 for the "
                "block-by-block delta vs _NoPeMLABlock."
            )

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
    "build_neuron_config",
    "dsv4_reference_router_forward",
    "dsv4_route_affinities",
]
