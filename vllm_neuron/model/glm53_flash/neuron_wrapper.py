# SPDX-License-Identifier: Apache-2.0
"""NxDI compile-integration wrapper for GLM-5.3-Flash (Round 2).

Codex Alpha shipped ``NeuronGlm53FlashForCausalLMImpl`` (in ``model.py``) as
the CPU source-qualified reference: pure Python autoregressive with reference
kernels, telemetry, and per-expert loops.  That class inherits from
``torch.nn.Module`` and cannot be handed to NxDI's compile pipeline directly.

This module supplies the compile-facing wrapper class,
``NeuronGlm53FlashForCausalLM``, that subclasses
``neuronx_distributed_inference.models.model_base.NeuronBaseForCausalLM`` so
the standard ``Neuron{X}ForCausalLM(model_path, config).compile(out_path)``
pattern binds.

Round 2 replaces the Round-1 single-Linear residual shell with real per-layer
NxDI parallel primitives:

- ``_NeuronGlm53FlashModel.init_model`` builds a
  ``ParallelEmbedding + [Glm53FlashLayer]*num_hidden_layers + norm +
  ColumnParallelLinear lm_head`` stack.
- ``Glm53FlashLayer`` picks its attention path per layer-index
  (``KDA`` for KDA layers, ``DSA+NoPeMLA`` for DSA layers) and its MLP path
  (``DenseMLP`` for the first ``first_k_dense_replace`` layers, sparse-MoE for
  the rest), matching ``config.layer_types`` / ``config.mlp_layer_types``.
- ``_NoPeMLABlock`` lowers Q_A, Q_B, KV_A, KV_B, O projections to
  ``ColumnParallelLinear`` (Q_B/KV_B along the head-count axis) and
  ``RowParallelLinear`` (O along the head-count axis), with RMSNorm weights
  held as ``nn.Parameter`` (small, replicated across ranks).
- ``_KDABlock`` builds Q/K/V and F/G/B parallel projections along the head
  axis; the KDA-state kernel body is deferred to Round 3 (a device-attached
  NKI binding) — the ``forward`` raises ``NotImplementedError`` with a
  pointer to ``kda_state_v2.py``'s reference until then.  This is the
  fallback-rule discipline: no silent fall-through to ``nn.Linear`` or CPU.
- ``_DSABlock`` composes ``_NoPeMLABlock`` for the QKV+O math and adds the
  DSA indexer projections (K_proj → ``ColumnParallelLinear``, Q_proj as a
  per-index-head ``nn.Parameter`` of shape ``[H_i, hidden_dim, index_head]``,
  IndexPool tail-select pooling parameter as a replicated ``nn.Parameter``).
  The sparse-attn kernel call is deferred to Round 3.
- ``_MoEBlock`` builds the routed-expert weights ``gate`` / ``up`` / ``down``
  as fused-expert ``nn.Parameter`` tensors of shape
  ``[num_routed_experts, hidden, moe_intermediate_size]`` and lowers the
  shared-expert MLP (gate/up/down) to ``ColumnParallelLinear + RowParallelLinear``.
  Blockwise-MoE dispatch (288 experts × top-8) is deferred to Round 3.
- ``_DenseMLPBlock`` lowers gate/up to ``ColumnParallelLinear`` and down to
  ``RowParallelLinear``.
- ``_MHCBlock`` mirrors ``mhc.py`` for the mHC 4-stream pre/post mixer;
  parameters are small and replicated across TP ranks.

The MoE blockwise-mm workaround from
``[[nxdi-container-moe-blockwise-mm-workaround-20260827]]`` — every MoE
compile on container ``sha256:011d49c7`` must set
``blockwise_matmul_config.use_shard_on_intermediate_dynamic_while=True``
before ``InferenceConfig`` init — is applied in ``build_neuron_config`` so
that any GLM-5.3-Flash compile driver honours it without re-quoting the
memory.

``load_weights`` implements the sharded-FP8 contract described in Round 1:
per-rank presharded loads via NxDI's base ``load_weights``, plus the
GLM-5.3-Flash indexer FP8 scale bounded-check from Fleet A's
``glm52_indexer_fp8_scale_fix.assert_indexer_multiplier_bounded`` (imported
opportunistically so the wrapper stays importable when the harness scratchpad
isn't on ``PYTHONPATH``).

Guarded imports keep the module importable on CPU-only hosts that lack the
Neuron toolchain — the CPU reference tests never touch the wrapper class, so
that path stays green.  Instantiating the wrapper on a host without NxDI
raises ``RuntimeError`` with a clear message.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ._reference_kernels import load_reference_kernel
from .checkpoint_convert import _convert_glm53_checkpoint
from .config import Glm53FlashInferenceConfig
from .kernel_dispatch import (
    DSA_CPU_GOLDEN_SLUG,
    DSA_NKI_V2_SLUG,
    KDA_CPU_GOLDEN_SLUG,
    KDA_NKI_V3P2_SLUG,
    get_emitted_kernel_slugs,
    resolve_dsa_impl_slug,
    resolve_kda_impl_slug,
)
from .model import NeuronGlm53FlashForCausalLMImpl
from .nki_bindings import (
    build_glm53_moe_dispatch_config,
    glm53_route_affinities,
    kda_state_forward_torch,
)
from .registry import _GLM53_GRAPH_ID, GLM53_SOURCE_CACHE_ABI

logger = logging.getLogger(__name__)

# NxDI-container guarded imports.  On CPU-only hosts (developer laptops
# running the source-qualification tests) the Neuron toolchain is absent; the
# wrapper class must still be importable so the package `__init__.py` can
# expose it in a single place.  Instantiation raises when the toolchain is
# actually missing.
try:
    # Round 4: the routed-expert branch now runs on NxDI's own blockwise-MoE
    # module instead of a hand-rolled token-major gather.  See `_MoEBlock`.
    from neuronx_distributed.modules.moe.expert_mlps import (
        ExpertMLPs as _NxdExpertMLPs,
    )
    from neuronx_distributed.modules.moe.model_utils import GLUType as _NxdGLUType
    from neuronx_distributed.parallel_layers.layers import (
        ColumnParallelLinear as _NxdColumnParallelLinear,
    )
    from neuronx_distributed.parallel_layers.layers import (
        ParallelEmbedding as _NxdParallelEmbedding,
    )
    from neuronx_distributed.parallel_layers.layers import (
        RowParallelLinear as _NxdRowParallelLinear,
    )
    from neuronx_distributed_inference.models.config import (
        InferenceConfig as _NxdiInferenceConfig,
    )
    from neuronx_distributed_inference.models.config import (
        MoENeuronConfig as _NxdiMoENeuronConfig,
    )
    from neuronx_distributed_inference.models.config import (
        NeuronConfig as _NxdiNeuronConfig,
    )
    from neuronx_distributed_inference.models.model_base import (
        NeuronBaseForCausalLM,
        NeuronBaseModel,
    )

    _NXDI_AVAILABLE = True
    _NXDI_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - CPU-only guard

    class _NxdiUnavailable:
        """Placeholder used only when NxDI is missing on this host."""

    NeuronBaseForCausalLM = _NxdiUnavailable  # type: ignore[assignment,misc]
    NeuronBaseModel = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdiInferenceConfig = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdiNeuronConfig = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdiMoENeuronConfig = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdColumnParallelLinear = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdRowParallelLinear = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdParallelEmbedding = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdExpertMLPs = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdGLUType = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NXDI_AVAILABLE = False
    _NXDI_IMPORT_ERROR = exc


# The MoE workaround from [[nxdi-container-moe-blockwise-mm-workaround-20260827]].
# Container `sha256:011d49c7...` is missing `_call_shard_hidden_kernel`; every
# GLM-5.3-Flash MoE compile must set this before `InferenceConfig` is created.
GLM53_BLOCKWISE_MATMUL_WORKAROUND: dict[str, bool] = {
    "use_shard_on_intermediate_dynamic_while": True,
    "skip_dma_token": True,
}

# Fields that would flag an FP8-packed KV cache — DELIBERATELY FORBIDDEN on
# GLM-5.3-Flash.  Mirrors the GLM-5.2 fail-loud guard at
# ``vllm-neuron:apuroop/glm5-2-enablement:vllm_neuron/model/glm52_moe_dsa/
# factory.py:260-261``.  Structural reason: this wrapper replaces NxDI's
# ``KVCacheManager`` with its own hybrid state cache (see
# ``_NeuronGlm53FlashModel.init_inference_optimization``) — the aliased
# tensors are declared bf16 explicitly, and any FP8-KV request would emit a
# neuron_config.json advertising ``float8_e4m3fn`` KV while the actual
# tensors stayed bf16.  That "requested-vs-emitted" split is exactly the
# silent-drop failure class this port refuses to inherit.
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
        "GLM-5.3-Flash NxDI wrapper requires the Neuron toolchain "
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
    """Construct an NxDI ``MoENeuronConfig`` with the GLM-5.3 MoE workaround pinned.

    **This must be a ``MoENeuronConfig``, not a ``NeuronConfig``.**
    ``blockwise_matmul_config`` is popped and frozen at
    ``models/config.py:837-839``, which is inside ``MoENeuronConfig``
    (declared at :798, next class at :849) — **not** the base ``NeuronConfig``
    (:84).  Passing it to the base class makes NxDI log

        NeuronConfig init: Unexpected keyword arguments: {'blockwise_matmul_config': ...}

    and silently drop it.  That is not cosmetic: with the flag dropped, the
    LNC=2 blockwise dispatch (``modules/moe/blockwise.py:1005-1017``) falls
    through to ``_call_shard_hidden_kernel``, which in this container is a stub
    that unconditionally raises ``NotImplementedError`` (``blockwise.py:267``).
    So the flag is not a performance knob — it is the only way to reach a real
    kernel on this build.  (``use_shard_on_block_dynamic_while`` is the one
    alternative; the two are mutually exclusive per the assert at :927.)

    LNC=1 is not an option at all for MoE here — the ``else`` branch raises
    ``"LNC_1 kernels not available in nkilib"``.
    """
    _require_nxdi()
    # Fail-loud FP8-KV guard.  Any legacy contract that carries
    # ``fp8_packed_kv`` / ``kv_cache_quant`` / ``kv_quant_config`` blows up
    # here rather than silently landing on bf16 KV.  See
    # ``FORBIDDEN_FP8_KV_KEYS`` for the structural reason.  Mirrored at the
    # compile-driver layer in ``/mnt/compile/shared-images/glm53-flash-command.sh``
    # so both entry points refuse the same set.
    if extra:
        offenders = sorted(k for k in FORBIDDEN_FP8_KV_KEYS if k in extra)
        if offenders:
            raise ValueError(
                "GLM-5.3-Flash refuses FP8-packed KV configuration: "
                f"{offenders!r}. This wrapper replaces NxDI's KVCacheManager "
                "with its own hybrid state cache (KDA + DSA + indexer); the "
                "aliased state tensors are declared bf16 explicitly and any "
                "FP8-KV request would silently mismatch the emitted "
                "neuron_config.json against the actual tensor dtypes."
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
        "blockwise_matmul_config": dict(GLM53_BLOCKWISE_MATMUL_WORKAROUND),
    }
    if extra:
        # Extra kwargs win over defaults; the workaround dict is merged shallow
        # so the caller can override individual sub-fields without losing the
        # container fix.
        extra_bmc = extra.pop("blockwise_matmul_config", None)
        # A caller that narrows `max_context_length` is asking for a *smaller
        # prefill bucket* than the KV window, which is the lever that makes the
        # CTE graph tractable: the KDA scan unrolls `num_kda_layers x
        # n_active_tokens` steps and the DSA sparse gather is O(Q x topk), so
        # both prefill costs key off this number, not off `seq_len`.  Keeping
        # `n_active_tokens` pinned to `seq_len` here would silently ignore it.
        if "max_context_length" in extra and "n_active_tokens" not in extra:
            kwargs["n_active_tokens"] = int(extra["max_context_length"])
        kwargs.update(extra)
        if extra_bmc is not None:
            merged = dict(GLM53_BLOCKWISE_MATMUL_WORKAROUND)
            merged.update(extra_bmc)
            kwargs["blockwise_matmul_config"] = merged
    config = _NxdiMoENeuronConfig(**kwargs)
    # Fail loudly if the flag did not survive construction — a silently
    # dropped flag compiles and then dies inside the raising stub, far from
    # the cause.
    bmc = getattr(config, "blockwise_matmul_config", None)
    if bmc is None or not getattr(
        bmc, "use_shard_on_intermediate_dynamic_while", False
    ):
        raise RuntimeError(
            "GLM-5.3-Flash MoE requires "
            "blockwise_matmul_config.use_shard_on_intermediate_dynamic_while=True "
            "to survive NeuronConfig construction; it did not. Without it the "
            "LNC=2 blockwise dispatch falls into _call_shard_hidden_kernel, "
            "which raises NotImplementedError on this container. Got: "
            f"{bmc!r}"
        )
    return config


def _dsa_pool_topk(
    index_scores: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select DSA pools with the pinned Trn2 TopK/NKI frontend.

    ``torch.topk`` is kept only as the CPU fallback.  On a Neuron/XLA tensor,
    an out-of-envelope shape is an error rather than an implicit fallback:
    the pinned compiler lowers that fallback to HLO ``sort``, which Trn2
    rejects with ``NCC_EVRF029``.  Returning both tensors keeps the values and
    indices contract explicit for the adversarial semantic gate even though
    the DSA expansion currently consumes only the indices.
    """
    # At S128 the GLM config has 32 pools and asks for all 32 (2048 // 4).
    # Rotational TopK intentionally rejects k == vocab_size.  No reduction is
    # needed in this case: preserve the complete candidate set in pool order.
    # DSA consumes the indices as a set; keeping the original order also makes
    # ties deterministic without introducing a sort.
    if k == index_scores.shape[-1]:
        indices = torch.arange(
            index_scores.shape[-1], device=index_scores.device, dtype=torch.int64
        )
        indices = indices.expand(*index_scores.shape[:-1], -1)
        return index_scores, indices

    from vllm_neuron.functional.topk import _can_use_nki_topk
    from vllm_neuron.functional.topk import topk as neuron_topk

    if str(index_scores.device) != "cpu" and not _can_use_nki_topk(
        index_scores, k, dim=-1
    ):
        raise RuntimeError(
            "GLM-5.3-Flash DSA pool top-k shape is outside the pinned "
            "Trn2 rotational-NKI envelope; refusing torch.topk fallback "
            "because it lowers to unsupported HLO sort."
        )
    return neuron_topk(index_scores, k, dim=-1, gather_dim=-1)


class Glm53FlashNeuronInferenceConfig(_NxdiInferenceConfig):
    """NxDI ``InferenceConfig`` wrapper that carries the Codex Alpha config.

    Holds the frozen ``Glm53FlashInferenceConfig`` (source-of-truth for
    architecture constants) as ``self.source_config`` and forwards the
    fields NxDI's compile pipeline probes (``num_hidden_layers``,
    ``hidden_size``, ``vocab_size`` …).  The class stays deliberately thin —
    the wrapper subclass, not this config, owns compile behaviour.
    """

    if _NXDI_AVAILABLE:

        def __init__(
            self,
            neuron_config: Any,
            source_config: Glm53FlashInferenceConfig | None = None,
            **kwargs: Any,
        ) -> None:
            self.source_config = source_config
            if source_config is not None:
                # Mirror the frozen fields NxDI's compile flow reads.
                for name in (
                    "vocab_size",
                    "hidden_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "intermediate_size",
                    "rms_norm_eps",
                    "max_position_embeddings",
                    "hidden_act",
                    "pad_token_id",
                    "torch_dtype",
                    "tie_word_embeddings",
                ):
                    kwargs.setdefault(name, getattr(source_config, name))
                kwargs.setdefault("head_dim", source_config.qk_head_dim)
                kwargs.setdefault("rope_theta", 10000.0)
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
            # Match GLM-5.2 flash-decoding default; overridden by
            # `neuron_config.num_cores_per_group` when the caller sets it.
            self.num_cores_per_group = getattr(
                self.neuron_config, "num_cores_per_group", 1
            )


# ---------------------------------------------------------------------------
# Per-layer NxDI-primitive blocks (Round 2).
# ---------------------------------------------------------------------------
if _NXDI_AVAILABLE:

    def _rms_norm_weight(hidden: int, dtype: torch.dtype) -> nn.Parameter:
        """Replicated per-rank RMS-norm gain (small, kept off TP shard)."""
        return nn.Parameter(torch.ones(hidden, dtype=dtype), requires_grad=False)

    def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Torch-primitive RMSNorm used by every GLM-5.3 sublayer."""
        value = x.to(torch.float32)
        value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
        return (value * weight.to(torch.float32)).to(x.dtype)

    def _tp_rank() -> int:
        """Trace-time TP rank.

        NxDI traces one graph per rank, so this is a Python constant at trace
        time and folds into the graph.  Small replicated parameters that the
        checkpoint stores full-width (``A_log``, ``dt_bias``, the DSA
        ``q_proj`` index-head cube) are sliced with it so each rank's
        recurrence uses the heads its Q/K/V column-parallel shard produced.

        Returns 0 when the parallel state is not initialised (single-rank CPU
        import, unit tests) — the slice then degenerates to the full tensor.
        """
        try:
            from neuronx_distributed.parallel_layers.parallel_state import (
                get_tensor_model_parallel_rank,
            )

            return int(get_tensor_model_parallel_rank())
        except Exception:  # pragma: no cover - single-rank / uninitialised
            return 0

    def _aliased_kv_parameters(model: nn.Module) -> list[torch.Tensor]:
        """The state parameters NxDI will alias, in alias order.

        Mirrors ``DecoderModelInstance.get()`` (model_wrapper.py:1614-1619):
        prefer ``kv_mgr.past_key_values``, else the model's own
        ``past_key_values``.  GLM-5.3-Flash sets ``kv_mgr = None`` and owns the
        second branch — see ``_NeuronGlm53FlashModel.init_inference_optimization``
        for why a plain ``KVCacheManager`` cannot describe a hybrid KDA/DSA
        stack.  Returns an empty list when neither exists.

        Note that *unused example inputs* need no such handling — torch_neuronx
        filters those to an ``exclude`` list with a warning
        (hlo_conversion.py:465-485).  Only the alias list is unfiltered, which
        is why an aliased-but-unread parameter aborts lowering.
        """
        kv_mgr = getattr(model, "kv_mgr", None)
        if kv_mgr is not None:
            values = getattr(kv_mgr, "past_key_values", None)
            if values is not None:
                return list(values)
        values = getattr(model, "past_key_values", None)
        if values is not None:
            return list(values)
        return []

    def _reduce_from_tp_region(x: torch.Tensor) -> torch.Tensor:
        """All-reduce a partial contraction across the TP group.

        Used by the DSA indexer, whose query-side projection contracts over
        the sharded main-attention head axis.  Without the reduce each rank
        would score positions from only its own head shard and select a
        different sparse set — a silent correctness bug that would look like
        mild quality loss rather than a crash, which is exactly the failure
        mode this campaign refuses to ship.

        Raises when the TP group is real but the primitive is unavailable;
        a no-op reduce at TP>1 would be that silent bug.
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
                    "GLM-5.3-Flash DSA indexer needs a TP all-reduce on the "
                    f"query projection at tp_degree={world}, but NxD's "
                    "`reduce_from_tensor_model_parallel_region` is "
                    f"unavailable: {exc!r}. Refusing to run a per-rank "
                    "partial top-k, which would silently select different "
                    "sparse positions on every rank."
                ) from exc
            return x

    def _sequence_carry(
        position_ids: torch.Tensor, dtype: torch.dtype, ndim: int
    ) -> torch.Tensor:
        """``0`` when this window starts a fresh sequence, ``1`` otherwise.

        Every aliased state buffer is multiplied by this before use.  It does
        two jobs at once, and the second is what makes the first safe:

        1. **Correct reset.**  A context-encoding window that begins at
           position 0 must not inherit the previous sequence's KDA state or KV
           cache.  Deriving the reset from ``position_ids`` rather than from
           the static bucket length also handles the general case (a window
           that continues an existing sequence keeps its state) instead of
           hard-coding "CTE means fresh".

        2. **A read the compiler cannot fold away.**  The alias list is
           appended to ``input_parameter_numbers`` *without* the ``-1`` filter
           (hlo_conversion.py:490-496), so every aliased parameter must appear
           in the lowering context or the trace aborts with "parameter not
           found in lowering context".  Writing the reset as ``state * 0`` or
           ``torch.zeros_like(state)`` does not satisfy that: a literal-zero
           multiply is algebraically foldable and ``zeros_like`` reads only
           metadata, so in both cases the parameter can vanish from the graph.
           This factor is a *runtime* value derived from a graph input, so the
           multiply survives to the lowered HLO by construction.

        The first Round-4 KDA+dense trace died exactly this way, with the same
        error message Round 3 had already chased once.
        """
        carry = (position_ids.to(torch.int64).amin(dim=-1) > 0).to(dtype)
        return carry.view(-1, *([1] * (ndim - 1)))

    def _write_positions(
        cache: torch.Tensor,  # [B, S, ...]
        new: torch.Tensor,  # [B, L, ...]
        position_ids: torch.Tensor,  # [B, L] int64
    ) -> torch.Tensor:
        """Return ``cache`` with ``new`` written at ``position_ids``.

        Functional, not in-place: mutating an aliased ``nn.Parameter`` inside
        ``forward`` is exactly the pattern torch_neuronx refuses to lower
        ("attempted in-place operation, or ... nn.Parameter.data").  The
        updated tensor is returned and handed back as a graph output, which is
        what makes the alias a real read-modify-write on device.

        Two trace-time shapes, selected by the *static* bucket length — NxDI
        traces CTE and TKG as separate graphs, so the branch is on a
        compile-time constant.  Both branches blend the old cache in through a
        *runtime-derived* mask, so the aliased parameter is genuinely read in
        either graph (see ``_sequence_carry``).

        * ``L > 1`` — prefill.  The window occupies positions ``0..span-1``;
          the write is a pad on the sequence axis and the untouched tail keeps
          its previous contents.  ``span`` comes from ``position_ids``, so the
          "keep" mask is a runtime tensor rather than a foldable constant.
        * ``L == 1`` — decode.  A one-hot write over the ``S`` axis, which
          costs ``B*S*D`` multiply-adds (~8 MFLOP for the largest GLM cache at
          S=2048) and keeps the tracer free of dynamic indexing.
        """
        length = int(new.shape[1])
        seq = int(cache.shape[1])
        value = new.to(cache.dtype)
        base = cache * _sequence_carry(position_ids, cache.dtype, cache.ndim)
        arange = torch.arange(seq, device=cache.device, dtype=torch.int64)

        if length > 1:
            span = position_ids.to(torch.int64).amax(dim=-1) + 1  # [B]
            covered = (arange.view(1, seq) < span.view(-1, 1)).to(cache.dtype)
            keep = (1.0 - covered).view(-1, seq, *([1] * (cache.ndim - 2)))
            pad = (0, 0) * (value.ndim - 2) + (0, seq - length)
            return F.pad(value, pad) + base * keep

        batch = cache.shape[0]
        trailing = tuple(cache.shape[2:])
        onehot = (
            position_ids[:, :1].to(torch.int64).unsqueeze(-1) == arange.view(1, 1, seq)
        ).to(cache.dtype)  # [B, 1, S]
        flat_base = base.reshape(batch, seq, -1)
        flat_new = value.reshape(batch, 1, -1)
        update = torch.einsum("bls,bld->bsd", onehot, flat_new)
        keep = (1.0 - onehot.sum(dim=1)).unsqueeze(-1)  # [B, S, 1]
        merged = flat_base * keep + update
        return merged.reshape(batch, seq, *trailing)

    def _is_prefill_length(length: int) -> bool:
        """True when this graph is a context-encoding (prefill) trace."""
        return int(length) > 1

    class _NoPeMLABlock(nn.Module):
        """All-NoPE MLA projections lowered to NxDI parallel primitives.

        Head-sharding contract (mirrors GLM-5.2 factory):
          - Q_A_proj:  hidden -> q_lora_rank (replicated, small; RowParallel with
                       input_is_parallel=False is unavailable here so we use a
                       ColumnParallelLinear with gather_output=True so every rank
                       observes the full latent).
          - Q_A_norm:  RMSNorm gain on q_lora_rank (replicated).
          - Q_B_proj:  q_lora_rank -> num_heads*qk_head_dim (ColumnParallel along
                       head-count axis, per-rank slice: heads_per_rank * qk_head_dim).
          - KV_A_proj: hidden -> kv_lora_rank (replicated).
          - KV_A_norm: RMSNorm gain on kv_lora_rank (replicated).
          - KV_B_proj: kv_lora_rank -> num_heads*(qk_nope_head_dim + v_head_dim)
                       (ColumnParallel along head-count axis).
          - O_proj:    num_heads*v_head_dim -> hidden (RowParallel along the
                       head-count axis so per-rank in-shard matches Q_B/KV_B out-shard).

        ``project`` returns split (query, key, value) with per-rank head count.
        The dot-product-attention + KV-cache write kernel call is exercised by
        ``_DSABlock`` (dense All-NoPE via ``dsa_sparse_attention_forward``); the
        KDA path uses only Q/K/V for its state kernel.
        """

        def __init__(
            self,
            config: Glm53FlashNeuronInferenceConfig,
            *,
            layer_idx: int,
        ) -> None:
            super().__init__()
            src = _require_source_config(config)
            if src.qk_rope_head_dim != 0:
                raise ValueError(
                    "_NoPeMLABlock requires qk_rope_head_dim=0; got "
                    f"{src.qk_rope_head_dim}"
                )
            self.layer_idx = layer_idx
            self.hidden_size = src.hidden_size
            self.num_heads = src.num_attention_heads
            self.qk_head_dim = src.qk_head_dim
            self.qk_nope_head_dim = src.qk_nope_head_dim
            self.v_head_dim = src.v_head_dim
            self.q_lora_rank = src.q_lora_rank
            self.kv_lora_rank = src.kv_lora_rank
            self.rms_eps = src.rms_norm_eps
            dtype = config.neuron_config.torch_dtype
            tp_degree = config.neuron_config.tp_degree
            if self.num_heads % tp_degree:
                raise NotImplementedError(
                    f"GLM-5.3-Flash MLA requires num_attention_heads "
                    f"({self.num_heads}) divisible by TP degree ({tp_degree}); "
                    "Round-3 will add head-padded fallback."
                )
            self.tp_degree = tp_degree
            self.heads_per_rank = self.num_heads // tp_degree
            self.state_dtype = dtype
            # Q_A / KV_A are latent-rank projections: ColumnParallelLinear with
            # gather_output=True so every rank observes the full latent and
            # feeds Q_B / KV_B (which shard along heads) with the correct input.
            self.q_a_proj = _NxdColumnParallelLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                gather_output=True,
                dtype=dtype,
            )
            self.q_a_norm = _rms_norm_weight(self.q_lora_rank, dtype)
            self.q_b_proj = _NxdColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                gather_output=False,
                dtype=dtype,
            )
            self.kv_a_proj = _NxdColumnParallelLinear(
                self.hidden_size,
                self.kv_lora_rank,
                bias=False,
                gather_output=True,
                dtype=dtype,
            )
            self.kv_a_norm = _rms_norm_weight(self.kv_lora_rank, dtype)
            self.kv_b_proj = _NxdColumnParallelLinear(
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
                gather_output=False,
                dtype=dtype,
            )
            self.o_proj = _NxdRowParallelLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=dtype,
            )

        def project(
            self, hidden_states: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Return ``(query, key, value, q_latent)``.

            ``q_latent`` is the post-Q_A + q_a_norm residual (HF's ``q_resid``)
            — the input the DSA indexer's ``wq_b`` contracts against.  Round 6
            passed the post-Q_B ``query`` to the indexer, which would have
            required a hypothetical [pooled_index_heads, heads_per_rank *
            qk_head_dim, index_head_dim] rank-3 ``q_proj``; HF actually stores
            ``wq_b`` as a low-rank ``[n_heads * head_dim, q_lora_rank]``
            projection off ``q_lora`` (not off the expanded ``Q_B`` output),
            so the mathematically-correct forward is ``q_idx = wq_b @
            q_latent`` — see Glm5NextTextIndexer in transformers/models/
            glm5_next/modeling_glm5_next.py at revision 5.14.1+.  Returning
            ``q_latent`` alongside the main-attention triple lets ``_DSABlock``
            share the Q_A + q_a_norm compute across the main-attention path
            and the indexer path — the same wiring HF uses.
            """
            if hidden_states.ndim != 3:
                raise ValueError("MLA expects [batch, sequence, hidden]")
            batch, length, _ = hidden_states.shape
            heads_per_rank = None  # inferred from B output shape below
            q_latent = _rms_norm(
                self.q_a_proj(hidden_states), self.q_a_norm, self.rms_eps
            )
            q_flat = self.q_b_proj(q_latent)
            heads_per_rank = q_flat.shape[-1] // self.qk_head_dim
            query = q_flat.view(batch, length, heads_per_rank, self.qk_head_dim)
            kv_latent = _rms_norm(
                self.kv_a_proj(hidden_states), self.kv_a_norm, self.rms_eps
            )
            kv_flat = self.kv_b_proj(kv_latent).view(
                batch,
                length,
                heads_per_rank,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            key, value = torch.split(
                kv_flat, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            return query, key, value, q_latent

    class _DSAIndexerBlock(nn.Module):
        """DSA lightning-indexer matching HF ``Glm5NextTextIndexer`` bit-exactly.

        Round 7 rewrite: Round-4 declared a rank-3 ``q_proj`` of shape
        ``[pooled_index_heads, heads_per_rank * qk_head_dim, index_head_dim]``
        and a scalar ``pool_weights[index_kpool]`` — layouts that DO NOT
        correspond to how HF actually stores the indexer weights.  HF's
        ``Glm5NextTextIndexer`` (transformers 5.14.x ``glm5_next`` /
        ``glm_moe_dsa`` families, ``modeling_glm5_next.py:749-877`` for the
        canonical GLM-5.3-Flash variant) stores:

          * ``wq_b.weight``  as ``[n_heads * head_dim, q_lora_rank]``
            (``[4096, 1536]`` at GLM-5.3-Flash), a low-rank projection off
            **q_lora** (the post-Q_A + q_a_norm residual), NOT off the
            expanded post-Q_B query.  ``q_proj @ Q_B @ q_lora`` cannot be
            losslessly reformulated as ``wq_b @ q_lora`` because ``Q_B`` has
            shape ``[n_heads_main * qk_head_dim, q_lora_rank] = [16384, 1536]``
            and rank 1536 < 16384 forecloses a right-inverse.  This forces
            Option A (wrapper adapts to HF layout); Option B (converter-side
            reformulation) is provably lossy.
          * ``wk.weight`` as ``[head_dim, hidden_size]`` — SINGLE-head, not
            per-index-head.  The indexer's K side is broadcast across the
            ``n_heads`` slots at score time.
          * ``k_norm.{weight,bias}`` — a LayerNorm on the ``head_dim``-wide K
            projection.
          * ``weights_proj.weight`` as ``[n_heads, hidden_size]`` — a
            per-token, per-head learned weight over the indexer heads, NOT a
            per-pool ``[index_kpool]`` scalar.
          * ``index_kpool_compress_ape`` as ``[index_kpool, head_dim]`` — a
            per-pool-slot learned position embedding used inside the pool
            softmax.
          * ``index_kpool_compress_gate`` as ``[head_dim, hidden_size]`` — a
            projection from hidden states into per-position "gate scores"
            that are cached alongside K and drive the pool-collapse softmax.

        The Round-6 status doc left ``q_proj`` / ``pool_weights`` unpopulated
        (NxDI ``strict=False`` accepted the ``torch.empty(...)`` scaffold);
        Round 7 aligns the wrapper's declared param shapes and forward math
        so that the load path is a straight-through carry from HF into the
        wrapper, and the ``index_scores`` this block emits agree with
        ``Glm5NextTextIndexer.forward`` at BF16 tolerance.  The mini-golden
        at ``tests/test_dsa_indexer_mapping.py`` proves the equivalence.

        TP contract: every indexer weight is **replicated** across ranks.
        The score composition is per-token and the top-k selection has to be
        identical on every rank so the sparse KV gather agrees; using
        ``ColumnParallelLinear(..., gather_output=True)`` on ``wq_b``,
        ``wk``, ``weights_proj`` and ``index_kpool_compress_gate`` gets us
        that with the checkpoint's ``.weight`` state-dict spelling intact.
        Total per-rank weight memory ~15 MB, which is well inside the
        indexer's HBM budget on Trainium2.
        """

        def __init__(
            self, config: Glm53FlashNeuronInferenceConfig, *, layer_idx: int
        ) -> None:
            super().__init__()
            src = _require_source_config(config)
            self.layer_idx = layer_idx
            # Advertised DSA kernel slug for this layer's NEFF cache key.
            # Resolved from `DSA_KERNEL_IMPL` at every `forward()` call
            # (see the assignment inside `forward`). Initialised here so
            # a construct-then-inspect audit (test / compile driver)
            # returns a valid slug even before the first forward.
            self._emitted_dsa_slug = resolve_dsa_impl_slug()
            self.hidden_size = src.hidden_size
            self.index_n_heads = src.index_n_heads
            self.index_head_dim = src.index_head_dim
            self.index_topk = src.index_topk
            self.index_kpool = src.index_kpool
            self.always_select_tail = src.index_kpool_always_select_tail
            self.compress = src.index_kpool_compress
            self.num_attention_heads = src.num_attention_heads
            self.qk_head_dim = src.qk_head_dim
            self.q_lora_rank = src.q_lora_rank
            self.k_norm_eps = 1e-6
            self.softmax_scale = self.index_head_dim**-0.5
            dtype = config.neuron_config.torch_dtype
            tp_degree = config.neuron_config.tp_degree
            self.tp_degree = tp_degree
            self.heads_per_rank = self.num_attention_heads // tp_degree

            # ``wq_b``: q_lora_rank -> n_heads * head_dim, replicated via
            # ``gather_output=True``.  State-dict key ``wq_b.weight`` matches
            # HF; the shape is ``[n_heads * head_dim, q_lora_rank]`` full
            # width.  NxDI's ``ColumnParallelLinear`` shards axis 0 by TP;
            # ``gather_output=True`` all-gathers back to full width so every
            # rank produces the same ``q_idx``.
            self.wq_b = _NxdColumnParallelLinear(
                self.q_lora_rank,
                self.index_n_heads * self.index_head_dim,
                bias=False,
                gather_output=True,
                dtype=dtype,
            )

            # ``wk``: hidden_size -> head_dim (SINGLE-head).  Replicated via
            # ``gather_output=True`` for the same reason as ``wq_b``.
            self.wk = _NxdColumnParallelLinear(
                self.hidden_size,
                self.index_head_dim,
                bias=False,
                gather_output=True,
                dtype=dtype,
            )

            # LayerNorm on the head_dim-wide K.  Kept as a real
            # ``nn.LayerNorm`` so state-dict keys ``k_norm.weight`` and
            # ``k_norm.bias`` line up with HF.
            self.k_norm = nn.LayerNorm(self.index_head_dim, eps=self.k_norm_eps)
            self.k_norm.weight = nn.Parameter(
                torch.ones(self.index_head_dim, dtype=dtype), requires_grad=False
            )
            self.k_norm.bias = nn.Parameter(
                torch.zeros(self.index_head_dim, dtype=dtype),
                requires_grad=False,
            )

            # ``weights_proj``: hidden_size -> n_heads per-token weight
            # tensor.  HF keeps this in fp32 (``_keep_in_fp32_modules``);
            # this wrapper loads it in the config dtype (bf16 by default)
            # and up-casts inside the forward — the difference is measured
            # in the mini-golden and stays inside BF16 tolerance.
            if tp_degree == 32:
                self.weights_proj = _NxdColumnParallelLinear(
                    self.hidden_size,
                    self.index_n_heads,
                    bias=False,
                    gather_output=True,
                    dtype=dtype,
                )
                self.weights_proj_ownership = "tp32_sharded_gathered"
            elif tp_degree == 64:
                # GLM has 32 index heads. Sharding this 32-wide output across
                # 64 TP ranks would create zero-width output partitions; keep
                # the small projection replicated without changing its
                # [B, Q, 32] scorer/state contract.
                self.weights_proj = nn.Linear(
                    self.hidden_size,
                    self.index_n_heads,
                    bias=False,
                    dtype=dtype,
                )
                self.weights_proj_ownership = "tp64_replicated"
            else:
                raise NotImplementedError(
                    f"GLM-5.3 indexer supports TP32 or TP64, got TP{tp_degree}"
                )

            # ``index_kpool_compress_ape``: per-pool-slot APE, added inside
            # the softmax over the pool axis.  Small parameter, replicated.
            self.index_kpool_compress_ape = nn.Parameter(
                torch.zeros(self.index_kpool, self.index_head_dim, dtype=dtype),
                requires_grad=False,
            )

            # ``index_kpool_compress_gate``: hidden -> head_dim "gate scores"
            # that are cached alongside K.  Held as a raw ``nn.Parameter``
            # (HF stores it under key ``index_kpool_compress_gate`` — no
            # ``.weight`` suffix — because HF declares it as ``nn.Parameter``,
            # not ``nn.Linear``).  Shape ``[head_dim, hidden_size]`` matches
            # the ``F.linear(hidden, gate)`` call convention.
            self.index_kpool_compress_gate = nn.Parameter(
                torch.zeros(self.index_head_dim, self.hidden_size, dtype=dtype),
                requires_grad=False,
            )

            self.register_buffer(
                "cache_quant_multiplier",
                torch.tensor(src.indexer_cache_quant_multiplier, dtype=torch.float32),
            )

        # ------------------------------------------------------------------
        # Cache-plumbing helpers used by ``_DSABlock``.  Both are shape
        # transforms only — no learned params, no TP awareness needed.
        # ------------------------------------------------------------------

        def project_index_k(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Post-``k_norm`` per-position K for the current window.

            Shape ``[B, L, head_dim]`` — single-head, matching HF's
            ``k = k_norm(wk(hidden_states)).squeeze(2)``.  Written into the
            aliased ``index_k_cache`` by ``_DSABlock`` before the indexer
            scores against the full cache.
            """
            k_raw = self.wk(hidden_states)
            return self.k_norm(k_raw)

        def project_index_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Per-position gate scores for the current window.

            Shape ``[B, L, head_dim]``.  Cached alongside K so the
            pool-collapse softmax can consume every past position on decode.
            HF stores these packed with K into a single cache tensor; we
            split them because the two aliased buffers are cheaper to
            reason about than a single wider one, and both are the same
            width.
            """
            return F.linear(hidden_states, self.index_kpool_compress_gate)

        def _pool_and_score(
            self,
            q_idx: torch.Tensor,
            k_cache: torch.Tensor,
            gate_cache: torch.Tensor,
            valid_keys: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Pool-collapsed indexer scores + expanded raw-token indices.

            Mirrors HF ``Glm5NextTextIndexer.get_pooled_states`` +
            ``Glm5NextTextIndexer.forward`` between the pool build and the
            top-k call.  ``first_key`` is fixed to 0 because this wrapper's
            aliased caches are written starting at slot 0 (see
            ``_write_positions``); left-padded prompts are handled by
            zeroing hidden_states before the projections, not by tracking a
            first-key offset.

            Returns ``(index_scores[B, Q, P], pool_indices[B, P,
            index_kpool], pool_valid[B, P])``.  The pool axis ``P`` is
            derived from the cache length and is compile-time constant.
            """
            batch, seq_len, _ = k_cache.shape
            device = k_cache.device

            number_of_pools = (seq_len + self.index_kpool - 1) // self.index_kpool
            padded = number_of_pools * self.index_kpool

            pool_offsets = torch.arange(padded, device=device, dtype=torch.int64)
            # first_key := 0 (see docstring).
            pool_indices = pool_offsets.view(1, number_of_pools, self.index_kpool)
            pool_indices = pool_indices.expand(batch, -1, -1)

            safe_indices = pool_indices.clamp(0, seq_len - 1)
            batch_idx = torch.arange(batch, device=device)[:, None, None]
            grouped_keys = k_cache[batch_idx, safe_indices]
            grouped_gate = gate_cache[batch_idx, safe_indices]
            grouped_valid = valid_keys[batch_idx, safe_indices]
            # Positions padded past ``seq_len`` are always invalid.
            grouped_valid = grouped_valid & (pool_indices < seq_len)
            pool_valid = grouped_valid.all(dim=-1)
            pool_indices = pool_indices.masked_fill(~grouped_valid, -1)

            # Per-pool weighted average of K via a softmax over the pool
            # axis.  ``compress_ape`` is the learned APE; masked pool cells
            # get ``-inf`` so the softmax ignores them.  ``nan_to_num``
            # keeps a fully-invalid pool row from turning the whole
            # column into NaNs.
            logits = (
                grouped_gate.to(torch.float32)
                + (self.index_kpool_compress_ape.to(torch.float32)[None, None])
            )
            logits = logits.masked_fill(~grouped_valid[..., None], float("-inf"))
            probabilities = torch.nan_to_num(logits.softmax(dim=2)).to(
                grouped_keys.dtype
            )
            pool_keys = (probabilities * grouped_keys).sum(dim=2)

            # Score q_idx against every pool centroid.  ``q_idx`` is
            # ``[B, Q, H_idx, D_idx]``; ``pool_keys`` is
            # ``[B, P, D_idx]`` and is broadcast across the ``H_idx``
            # axis (HF's single-head K).  ReLU + fp32 scale mirrors HF.
            scores = torch.matmul(
                q_idx.float(),
                pool_keys.transpose(-1, -2).float().unsqueeze(1),
            )
            scores = F.relu(scores * self.softmax_scale)

            return scores, pool_indices, pool_valid

        def compute_index_scores(
            self,
            hidden_states: torch.Tensor,
            q_latent: torch.Tensor,
            k_cache: torch.Tensor,
            gate_cache: torch.Tensor,
            position_ids: torch.Tensor,
            key_lengths: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Pre-top-k pool scores + auxiliary tensors.

            Split out of ``forward`` so the mini-golden can compare against
            HF ``Glm5NextTextIndexer`` at the ``index_scores`` boundary and
            the sparse-attention path can consume the ``pool_indices`` for
            the raw-token top-k downstream.

            ``k_cache`` / ``gate_cache`` are the aliased caches, already
            updated with the current window's positions before this call.
            ``key_lengths[b]`` counts valid slots in cache row ``b``.
            """
            batch, length, _ = hidden_states.shape
            seq_len = int(k_cache.shape[1])
            device = hidden_states.device

            # q_idx = wq_b(q_latent).view(B, L, H_idx, D_idx).  Layout is
            # identical to HF (``q.view(B, S, n_heads, head_dim)``).
            q_flat = self.wq_b(q_latent)
            q_idx = q_flat.view(batch, length, self.index_n_heads, self.index_head_dim)

            # Per-position validity: slot i is valid iff i < key_lengths[b].
            valid_keys = (
                torch.arange(seq_len, device=device, dtype=torch.int64)[None, :]
                < key_lengths.to(torch.int64)[:, None]
            )

            scores, pool_indices, pool_valid = self._pool_and_score(
                q_idx, k_cache, gate_cache, valid_keys
            )

            # Per-token, per-head learned weight; ``* n_heads ** -0.5``
            # matches HF's ``self.n_heads ** -0.5`` scale.
            weights = self.weights_proj(hidden_states).float() * (
                self.index_n_heads**-0.5
            )
            # [B, Q, 1, H] @ [B, Q, H, P] -> [B, Q, P].
            index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

            # Pool visibility: a pool is selectable iff its final raw token
            # is visible to the query (causality) AND all its raw tokens
            # are valid (padding).
            causal_last_ok = self._pool_last_causal_ok(
                pool_indices, position_ids, key_lengths, seq_len
            )
            valid_candidates = causal_last_ok & pool_valid[:, None, :]
            index_scores = index_scores.masked_fill(
                ~valid_candidates,
                torch.finfo(index_scores.dtype).min,
            )
            return index_scores, pool_indices, pool_valid, valid_candidates

        def _pool_last_causal_ok(
            self,
            pool_indices: torch.Tensor,  # [B, P, index_kpool] int64 with -1 sentinels
            position_ids: torch.Tensor,  # [B, Q] int64
            key_lengths: torch.Tensor,  # [B] int64
            seq_len: int,
        ) -> torch.Tensor:
            """Broadcast HF's causal + key-length gate onto every pool's tail.

            HF checks visibility of ``pool_indices[..., -1]`` — the last
            raw token in each pool — against the causal mask.  We
            reconstruct that mask from ``position_ids`` and ``key_lengths``
            so we do not need to allocate a full ``visible_tokens`` matrix.
            """
            batch = pool_indices.shape[0]
            device = pool_indices.device
            pool_last = pool_indices[..., -1].clamp(0, seq_len - 1)
            # Causality: pool_last <= q_position.
            q_pos = position_ids.to(torch.int64).view(batch, -1, 1)
            key_ok_causal = pool_last[:, None, :].to(torch.int64) <= q_pos
            # Key-length: pool_last < key_lengths.
            key_ok_len = pool_last[:, None, :].to(torch.int64) < key_lengths.to(
                torch.int64
            ).view(batch, 1, 1)
            return key_ok_causal & key_ok_len

        def select_topk(
            self,
            index_scores: torch.Tensor,  # [B, Q, P]
            pool_indices: torch.Tensor,  # [B, P, index_kpool]
            valid_candidates: torch.Tensor,  # [B, Q, P]
        ) -> torch.Tensor:
            """Convert per-pool scores into per-token top-k raw indices.

            Mirrors HF's post-``index_scores`` tail: pick ``index_topk //
            index_kpool`` best pools per query, expand each to its
            ``index_kpool`` raw tokens, mask invalid tail cells to ``-1``.
            Optional tail-append (``index_kpool_always_select_tail=True``)
            adds up to ``index_kpool - 1`` fresh raw tokens; the output
            width is then ``index_topk + index_kpool - 1``, matching HF.
            """
            batch, q_len, pools = index_scores.shape
            device = index_scores.device
            select_k = min(self.index_topk // self.index_kpool, pools)

            # ``aten::topk`` lowers to an HLO ``sort`` in the pinned
            # torch-neuronx stack; use the Trn2 TopK/NKI seam instead.
            _selected_values, selected = _dsa_pool_topk(index_scores, select_k)
            batch_idx = torch.arange(batch, device=device)[:, None, None]
            selected_valid = valid_candidates.gather(-1, selected)
            selected_indices = pool_indices[batch_idx, selected]

            topk_indices = selected_indices.flatten(-2)
            mask = ~selected_valid[..., None].expand_as(selected_indices).flatten(-2)
            topk_indices = topk_indices.masked_fill(mask, -1)

            output_width = self.index_topk
            if self.always_select_tail:
                output_width += self.index_kpool - 1
            pad_amount = output_width - topk_indices.shape[-1]
            if pad_amount > 0:
                topk_indices = F.pad(topk_indices, (0, pad_amount), value=-1)
            return topk_indices[..., :output_width].to(torch.int32)

        def forward(
            self,
            hidden_states: torch.Tensor,
            q_latent: torch.Tensor,
            position_ids: torch.Tensor,
            query: torch.Tensor,
            kv_cache_k: torch.Tensor,
            kv_cache_v: torch.Tensor,
            key_lengths: torch.Tensor,
            index_k_cache: torch.Tensor,
            index_gate_cache: torch.Tensor,
            *,
            return_lse: bool = False,
        ):
            """Round-7 HF-parity DSA lightning-indexer + sparse attention.

            Contract:
              * ``q_latent`` — post-Q_A + q_a_norm, ``[B, L, q_lora_rank]``.
                Contracts against ``wq_b`` (HF's low-rank projection),
                bypassing Q_B.  Round 6 passed the post-Q_B ``query`` here,
                which the previous rank-3 ``q_proj`` scaffold would have
                consumed; HF's low-rank ``wq_b`` cannot be losslessly
                reformulated back through Q_B so the wrapper had to change.
              * ``index_k_cache`` — cached ``[B, S, head_dim]`` post-k_norm
                keys, single-head per HF.
              * ``index_gate_cache`` — cached ``[B, S, head_dim]`` gate
                scores driving the pool-collapse softmax.
              * ``key_lengths`` — count of valid cache slots per batch row.

            Emits either the sparse-attention output alone, or (when
            ``return_lse=True``) a ``(out, lse)`` pair — same shape contract
            as Round 4.  Top-k selection happens over pools first
            (``index_topk // index_kpool`` best), then expands back to raw
            token indices which are handed to ``dsa_sparse_attention_forward``
            (the golden's sparse gather + softmax) as-is; no downstream
            re-scoring is done.
            """
            (
                index_scores,
                pool_indices,
                _pool_valid,
                valid_candidates,
            ) = self.compute_index_scores(
                hidden_states,
                q_latent,
                index_k_cache,
                index_gate_cache,
                position_ids,
                key_lengths,
            )
            topk_indices = self.select_topk(
                index_scores, pool_indices, valid_candidates
            )
            golden = load_reference_kernel("dsa")
            effective_topk = int(topk_indices.shape[-1])
            # Emitted-slug dispatch: env `DSA_KERNEL_IMPL` picks the slug
            # this graph will advertise in the NEFF cache key. Runtime
            # execution stays on `golden.dsa_sparse_attention_forward`
            # (v0 CPU golden, traceable through NxDI). The v2 NKI device
            # kernel replaces this at NEFF-load time when the compile
            # driver has warmed it for this shape; the wrapper's job here
            # is to advertise the correct cache identity.
            self._emitted_dsa_slug = resolve_dsa_impl_slug()
            return golden.dsa_sparse_attention_forward(
                query,
                kv_cache_k,
                kv_cache_v,
                topk_indices,
                position_ids,
                key_lengths,
                topk=effective_topk,
                causal=True,
                return_lse=return_lse,
            )

    class _DSABlock(nn.Module):
        """DSA layer = All-NoPE MLA + DSA lightning-indexer + sparse-attn.

        The Round-2 wrapper lowers the projections but defers the sparse-attn
        kernel call to Round 3 (per the fallback rule; no silent nn.Linear /
        CPU fall-through).
        """

        def __init__(
            self, config: Glm53FlashNeuronInferenceConfig, *, layer_idx: int
        ) -> None:
            super().__init__()
            self.mla = _NoPeMLABlock(config, layer_idx=layer_idx)
            self.indexer = _DSAIndexerBlock(config, layer_idx=layer_idx)

        def state_cache_specs(
            self, batch: int, seq_len: int
        ) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
            """Aliased caches this layer owns, in graph input/output order.

            Four, not three (Round 7): HF ``Glm5NextTextIndexer`` runs a
            pool-collapse softmax over ``gate_scores = F.linear(hidden,
            index_kpool_compress_gate)`` co-cached with the K side.  Round 4
            didn't cache these because it used the golden CPU reference's
            simpler score composition (constant pool weights); Round 7 caches
            them so the score composition is bit-parity with HF's forward.

            Two shape corrections vs Round 4:

              * ``index_k_cache`` becomes ``[B, S, head_dim]`` (SINGLE-head)
                — HF's ``wk`` outputs ``head_dim``, not ``n_heads * head_dim``.
                The head-axis broadcast at score time is exactly the same
                ``[B, Q, H_idx, D_idx] @ [B, P, D_idx].T`` broadcast HF uses.
              * A new ``index_gate_cache`` of the same shape holds the
                per-position gate scores.  Cached as its own tensor rather
                than packed with K because both are the same width and two
                aliased buffers are easier to reason about at debug time
                than one wider one.

            All indexer caches are replicated (single-head, no TP shard);
            the top-k selection is therefore bit-identical on every rank.
            """
            mla, idx = self.mla, self.indexer
            dtype = mla.state_dtype
            return [
                (
                    "k_cache",
                    (batch, seq_len, mla.heads_per_rank, mla.qk_nope_head_dim),
                    dtype,
                ),
                (
                    "v_cache",
                    (batch, seq_len, mla.heads_per_rank, mla.v_head_dim),
                    dtype,
                ),
                (
                    "index_k_cache",
                    (batch, seq_len, idx.index_head_dim),
                    dtype,
                ),
                (
                    "index_gate_cache",
                    (batch, seq_len, idx.index_head_dim),
                    dtype,
                ),
            ]

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            caches: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            key_lengths: torch.Tensor | None = None,
        ) -> tuple[
            torch.Tensor,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ]:
            """Round-7 HF-parity DSA layer: MLA + cached indexer + sparse attn.

            Takes and returns the four aliased caches this layer owns —
            ``(k_cache, v_cache, index_k_cache, index_gate_cache)`` — as
            graph input/output pairs (see ``state_cache_specs`` for the
            Round-4 -> Round-7 delta).

            ``key_lengths`` is derived from ``position_ids``, not from
            ``attention_mask``: the mask describes the *current window*, while
            the sparse gather needs the number of valid *cache* positions.
            At decode the mask would say 1 and the cache holds ``p+1``.
            """
            k_cache, v_cache, index_k_cache, index_gate_cache = caches
            query, key, value, q_latent = self.mla.project(hidden_states)
            index_k = self.indexer.project_index_k(hidden_states)
            index_gate = self.indexer.project_index_gate(hidden_states)

            new_k = _write_positions(k_cache, key, position_ids)
            new_v = _write_positions(v_cache, value, position_ids)
            new_index_k = _write_positions(index_k_cache, index_k, position_ids)
            new_index_gate = _write_positions(
                index_gate_cache, index_gate, position_ids
            )

            context_len = int(new_k.shape[1])
            valid = position_ids.to(torch.int64).amax(dim=-1) + 1
            if key_lengths is not None and _is_prefill_length(
                int(hidden_states.shape[1])
            ):
                # Prefill only: a right-padded window spans more positions than
                # it has real tokens, and the mask is the tighter bound.  At
                # decode the mask describes a 1-token window while the cache
                # holds p+1 valid positions, so the mask must NOT be consulted
                # — doing so would collapse every decode step to a 1-position
                # context, which is precisely the silent-wrongness this round
                # exists to remove.
                valid = torch.minimum(valid, key_lengths.to(torch.int64))
            key_lengths = valid.clamp(min=1, max=context_len)

            attn = self.indexer(
                hidden_states,
                q_latent,
                position_ids,
                query,
                new_k,
                new_v,
                key_lengths,
                new_index_k,
                new_index_gate,
            )
            if isinstance(attn, tuple):
                attn = attn[0]
            batch, length = attn.shape[0], attn.shape[1]
            out = self.mla.o_proj(
                attn.reshape(batch, length, -1).to(hidden_states.dtype)
            )
            return out, (new_k, new_v, new_index_k, new_index_gate)

    class _KDABlock(nn.Module):
        """Kimi Delta Attention (KDA) projections + state-kernel wrapper.

        Weight lowering (mirrors ``kda.py``):
          - q_proj, k_proj, v_proj: hidden -> num_heads*head_dim
            (ColumnParallel along head axis).
          - conv1d: depthwise conv over 3*num_heads*head_dim channels; kept as a
            replicated ``nn.Conv1d`` because per-channel-group sharding across
            TP has no NxDI primitive.  Round-3 will replace with a NKI
            depthwise-conv kernel.
          - f_a_proj / f_b_proj: hidden -> head_dim (ColumnParallel, gather),
            head_dim -> num_heads*head_dim (ColumnParallel, no-gather).
          - g_a_proj / g_b_proj: same pattern.
          - b_proj: hidden -> num_heads (ColumnParallel along head axis).
          - dt_bias, A_log: small per-head parameters, replicated.
          - o_norm.weight: RMSNorm gain over head_dim (replicated).
          - o_proj: num_heads*head_dim -> hidden (RowParallel).

        The KDA state kernel body is a Round-3 NKI binding; forward raises
        ``NotImplementedError`` with a pointer to the reference until then.
        """

        def __init__(
            self, config: Glm53FlashNeuronInferenceConfig, *, layer_idx: int
        ) -> None:
            super().__init__()
            src = _require_source_config(config)
            linear = src.linear_attn_config
            self.layer_idx = layer_idx
            # Advertised KDA kernel slug for this layer's NEFF cache key.
            # Resolved from `KDA_KERNEL_IMPL` at every `forward()` call
            # (see the assignment inside `forward`). Initialised here so
            # a construct-then-inspect audit (test / compile driver)
            # returns a valid slug even before the first forward.
            self._emitted_kda_slug = resolve_kda_impl_slug()
            self.hidden_size = src.hidden_size
            self.num_heads = linear.num_heads
            self.head_dim = linear.head_dim
            self.qkv_dim = self.num_heads * self.head_dim
            self.short_conv_kernel_size = linear.short_conv_kernel_size
            self.gate_lower_bound = linear.gate_lower_bound
            self.l2norm_eps = linear.l2norm_eps
            self.rms_eps = src.rms_norm_eps
            dtype = config.neuron_config.torch_dtype
            tp_degree = config.neuron_config.tp_degree
            if self.num_heads % tp_degree:
                raise NotImplementedError(
                    f"KDA requires num_heads ({self.num_heads}) divisible by "
                    f"TP degree ({tp_degree}); Round-3 will add head-padded "
                    "fallback."
                )
            self.tp_degree = tp_degree
            self.heads_per_rank = self.num_heads // tp_degree
            self.qkv_dim_local = self.heads_per_rank * self.head_dim
            self.max_batch_size = config.neuron_config.max_batch_size
            self.q_proj = _NxdColumnParallelLinear(
                self.hidden_size,
                self.qkv_dim,
                bias=False,
                gather_output=False,
                dtype=dtype,
            )
            self.k_proj = _NxdColumnParallelLinear(
                self.hidden_size,
                self.qkv_dim,
                bias=False,
                gather_output=False,
                dtype=dtype,
            )
            self.v_proj = _NxdColumnParallelLinear(
                self.hidden_size,
                self.qkv_dim,
                bias=False,
                gather_output=False,
                dtype=dtype,
            )
            # Depthwise-groups conv1d over the *rank-local* channel set.
            # Round 2 declared this at full width, which contradicted the
            # column-parallel Q/K/V it consumes (each rank only ever holds
            # `qkv_dim_local` channels).  A depthwise conv is per-channel, so
            # the channel axis shards exactly like Q/K/V's output axis and the
            # rank-local declaration is the correct one — no cross-rank
            # communication is needed for the short conv.
            channels = 3 * self.qkv_dim_local
            self.conv_channels = channels
            self.conv1d = nn.Conv1d(
                channels,
                channels,
                kernel_size=self.short_conv_kernel_size,
                groups=channels,
                bias=False,
                dtype=dtype,
            )
            self.f_a_proj = _NxdColumnParallelLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                gather_output=True,
                dtype=dtype,
            )
            self.f_b_proj = _NxdColumnParallelLinear(
                self.head_dim,
                self.qkv_dim,
                bias=False,
                gather_output=False,
                dtype=dtype,
            )
            self.g_a_proj = _NxdColumnParallelLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                gather_output=True,
                dtype=dtype,
            )
            self.g_b_proj = _NxdColumnParallelLinear(
                self.head_dim,
                self.qkv_dim,
                bias=False,
                gather_output=False,
                dtype=dtype,
            )
            self.b_proj = _NxdColumnParallelLinear(
                self.hidden_size,
                self.num_heads,
                bias=False,
                gather_output=False,
                dtype=dtype,
            )
            self.dt_bias = nn.Parameter(
                torch.zeros(self.qkv_dim, dtype=torch.float32),
                requires_grad=False,
            )
            self.A_log = nn.Parameter(
                torch.zeros(self.num_heads, dtype=torch.float32),
                requires_grad=False,
            )
            self.o_norm_weight = nn.Parameter(
                torch.ones(self.head_dim, dtype=dtype), requires_grad=False
            )
            self.o_proj = _NxdRowParallelLinear(
                self.qkv_dim,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=dtype,
            )
            # Recurrent-state SHAPE contract, laid out as the vLLM KDA cache
            # does: [num_slots, HV, V, K] == [max_batch, heads_per_rank,
            # head_dim, head_dim].  Deliberately NOT a registered buffer.
            #
            # A `persistent=False` buffer is excluded from `state_dict()`, and
            # NxDI's model builder derives the graph's parameter set from the
            # state dict — so reading such a buffer inside forward produces
            # "Unable to lower HLO: parameter not found in lowering context".
            # A `persistent=True` buffer would instead demand a checkpoint
            # tensor that does not exist.  The state is therefore materialised
            # inside forward (a graph constant for prefill-from-zero) or
            # supplied by the caller as a real tensor argument.
            self.kda_state_shape = (
                self.heads_per_rank,
                self.head_dim,
                self.head_dim,
            )
            self.conv_state_shape = (
                self.conv_channels,
                self.short_conv_kernel_size - 1,
            )
            self.state_dtype = dtype

        def state_cache_specs(
            self, batch: int, seq_len: int
        ) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
            """Aliased state this layer owns, in graph input/output order.

            KDA is a linear-attention recurrence, so its "cache" is a fixed
            ``[HV, V, K]`` state matrix per slot rather than a growing
            per-position buffer — it does not scale with ``seq_len``, which is
            the whole point of the mechanism.  The short-conv history
            (``kernel_size - 1`` columns) is the second piece of state and is
            just as load-bearing: dropping it re-runs the depthwise conv with
            zero history on every decode step.
            """
            del seq_len  # KDA state is sequence-length independent
            return [
                ("kda_state", (batch,) + self.kda_state_shape, self.state_dtype),
                ("conv_state", (batch,) + self.conv_state_shape, self.state_dtype),
            ]

        def _local_heads(self) -> slice:
            """Rank-local head slice for the checkpoint-width small params."""
            rank = _tp_rank()
            return slice(rank * self.heads_per_rank, (rank + 1) * self.heads_per_rank)

        def _short_conv(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            conv_state: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Causal depthwise short conv + SiLU, mirroring ``kda.py``.

            Returns the three post-conv streams plus the updated conv state.
            """
            batch, length = query.shape[0], query.shape[1]
            joined = torch.cat((query, key, value), dim=-1)
            joined = joined.flatten(-2).transpose(1, 2)  # [B, C, L]
            conv_input = torch.cat((conv_state.to(joined.dtype), joined), dim=-1)
            history = self.short_conv_kernel_size - 1
            new_conv_state = conv_input[..., -history:] if history else conv_state
            convolved = F.silu(self.conv1d(conv_input))
            convolved = convolved.transpose(1, 2).view(
                batch, length, self.heads_per_rank, 3 * self.head_dim
            )
            q_c, k_c, v_c = torch.split(convolved, self.head_dim, dim=-1)
            return q_c, k_c, v_c, new_conv_state

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            kda_state: torch.Tensor | None = None,
            conv_state: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Round-3 bound KDA forward.

            Kernel identity: ``KDA_KERNEL_SLUG_V2``
            (``kda_state.decode.kda_gate.rank1_delta.bf16_state.v1``).

            The state recurrence is the torch transcription of the numpy
            golden ``kda_state_v2._kda_delta_rule_step`` living in
            ``nki_bindings.kda_state_forward_torch``; it preserves the four
            FLA v0.5.2 parity pieces (per-channel gate with LOWER_BOUND=-5.0,
            in-kernel L2-norm at eps=1e-6, query scale ``D_qk ** -0.5``, bf16
            state).  Bit-exactness against the numpy golden is asserted by
            ``nki_bindings.kda_reference_parity_check``.

            No fallback: there is no branch here that reaches softmax or dense
            attention.  KDA is a linear-attention recurrence; a dense
            substitute would be a different model, not a slower one.
            """
            if hidden_states.ndim != 3:
                raise ValueError("KDA expects [batch, sequence, hidden]")
            batch, length, _ = hidden_states.shape
            heads = self.heads_per_rank

            query = self.q_proj(hidden_states).view(batch, length, heads, self.head_dim)
            key = self.k_proj(hidden_states).view(batch, length, heads, self.head_dim)
            value = self.v_proj(hidden_states).view(batch, length, heads, self.head_dim)

            # A window that starts at position 0 begins a fresh sequence and
            # must not inherit the previous sequence's state; a window that
            # starts later continues one.  `_sequence_carry` encodes exactly
            # that, and — being derived from `position_ids` at runtime — also
            # guarantees the aliased parameter survives into the lowered HLO.
            # See its docstring: `state * 0` and `torch.zeros_like(state)` both
            # let the parameter vanish, which aborts the trace with "parameter
            # not found in lowering context".
            prefill = _is_prefill_length(length)
            if conv_state is None:
                conv_state = torch.zeros(
                    (batch,) + self.conv_state_shape,
                    dtype=self.state_dtype,
                    device=hidden_states.device,
                )
            else:
                conv_state = conv_state * _sequence_carry(
                    position_ids, conv_state.dtype, conv_state.ndim
                )
            query, key, value, new_conv_state = self._short_conv(
                query, key, value, conv_state
            )

            # Per-channel gate logits and per-head beta logit.
            g_raw = self.f_b_proj(self.f_a_proj(hidden_states)).view(
                batch, length, heads, self.head_dim
            )
            beta_raw = self.b_proj(hidden_states).view(batch, length, heads)

            local = self._local_heads()
            a_log = self.A_log[local]
            g_bias = self.dt_bias.view(self.num_heads, self.head_dim)[local]

            if kda_state is None:
                kda_state = torch.zeros(
                    (batch,) + self.kda_state_shape,
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
            else:
                # The aliased buffer is stored bf16 (the KDA_KERNEL_SLUG_V2
                # `bf16_state` identity); the recurrence runs in fp32 and
                # re-quantizes at the store boundary each step, so promoting
                # here loses nothing.  Same carry factor as the conv state.
                kda_state = kda_state.to(torch.float32) * _sequence_carry(
                    position_ids, torch.float32, kda_state.ndim
                )

            # Emitted-slug dispatch: read `KDA_KERNEL_IMPL` and record the
            # slug this graph will advertise in its NEFF cache key. The
            # actual math STAYS on `kda_state_forward_torch` (the torch
            # transcription of the numpy CPU golden that traces through
            # NxDI cleanly). The v3.2 NKI device kernel replaces this at
            # NEFF-load time when the compile driver has warmed it for
            # this shape; the wrapper's job here is to advertise the
            # correct cache identity, which the compile driver reads via
            # `get_emitted_kernel_slugs()` at fire time.
            self._emitted_kda_slug = resolve_kda_impl_slug()
            output, new_state = kda_state_forward_torch(
                kda_state,
                query,
                key,
                value,
                g_raw,
                beta_raw,
                a_log,
                g_bias,
                lower_bound=self.gate_lower_bound,
                l2norm_eps=self.l2norm_eps,
                impl="torch",
            )

            # State hand-off is FUNCTIONAL, not a buffer mutation.
            #
            # Reassigning `self.kda_state` inside forward is exactly the
            # pattern torch_neuronx refuses to lower:
            #   "Unable to lower HLO: parameter not found in lowering context.
            #    This is likely caused by an attempted in-place operation, or
            #    an attempted access of nn.Parameter.data or nn.Buffer.data."
            #
            # Round 4 makes the state survive across decode steps by returning
            # it: the caller holds the state as an `nn.Parameter` that NxDI
            # aliases via `input_output_aliases`, reads it in as an argument,
            # and returns the updated tensor after the logits in alias order —
            # the same input/output-pair contract the KV cache uses
            # (kv_cache_manager.py:152-162, model_wrapper.py:1614-1619).  Round
            # 3 left the update on a Python attribute, which the tracer
            # ignores, so a multi-step TKG graph restarted from zero state
            # every step: it compiled, ran, and benchmarked cleanly while being
            # wrong.
            self._new_kda_state = new_state
            self._new_conv_state = new_conv_state

            # Gated output RMSNorm over head_dim, then row-parallel out-proj.
            output = output.to(hidden_states.dtype)
            gate = self.g_b_proj(self.g_a_proj(hidden_states)).view_as(output)
            normed = _rms_norm(output, self.o_norm_weight, self.rms_eps)
            normed = normed * torch.sigmoid(gate.to(torch.float32)).to(normed.dtype)
            return (
                self.o_proj(normed.flatten(-2)),
                new_state.to(self.state_dtype),
                new_conv_state.to(self.state_dtype),
            )

    class _DenseMLPBlock(nn.Module):
        """GLM-5.3-Flash dense MLP (first ``first_k_dense_replace`` layers).

        gate/up as ColumnParallel (along intermediate axis), down as RowParallel.
        SwiGLU with clamp limit matches ``dense_mlp.py``.
        """

        def __init__(self, config: Glm53FlashNeuronInferenceConfig) -> None:
            super().__init__()
            src = _require_source_config(config)
            self.hidden_size = src.hidden_size
            self.intermediate_size = src.intermediate_size
            self.limit = src.swiglu_limit
            dtype = config.neuron_config.torch_dtype
            tp_degree = config.neuron_config.tp_degree
            if self.intermediate_size % tp_degree:
                raise NotImplementedError(
                    f"Dense MLP requires intermediate_size ({self.intermediate_size}) "
                    f"divisible by TP degree ({tp_degree})."
                )
            self.gate_proj = _NxdColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=dtype,
            )
            self.up_proj = _NxdColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=dtype,
            )
            self.down_proj = _NxdRowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=dtype,
            )

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            gate = self.gate_proj(hidden_states).clamp(max=self.limit)
            up = self.up_proj(hidden_states).clamp(-self.limit, self.limit)
            return self.down_proj(F.silu(gate) * up)

    class _MoESharedExpert(nn.Module):
        """Shared expert branch of the GLM-5.3 sparse MoE."""

        def __init__(self, config: Glm53FlashNeuronInferenceConfig) -> None:
            super().__init__()
            src = _require_source_config(config)
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

    class _MoEBlock(nn.Module):
        """288-expert routed MoE (top-8) + one shared expert.

        **Round 4 replaces the routed dispatch with NxDI's own
        ``ExpertMLPs``.**  Round 3 hand-rolled a token-major top-k gather
        (``moe_gather_dispatch_torch``).  Its FLOPs were right — O(top_k), not
        O(288) — but its *memory* was not: it materialised one full expert
        weight slab per token, ``[B*L*top_k, hidden, inter]``.  Measured, that
        is ~2.1 GB per slab at 128 prefill tokens (~6.4 GB for gate/up/down)
        and ~34 GB per slab at 2048, and it is what aborted the Round-3
        4-layer coverage smoke with ``double free or corruption`` inside the
        XLA tracer's allocator during CTE HLO generation.

        ``ExpertMLPs`` owns the capacity dispatch instead: it holds the expert
        weights as ``ExpertFusedColumnParallelLinear`` (gate+up fused,
        ``stride=2``) and ``ExpertFusedRowParallelLinear``, both sharded on
        the intermediate axis, and picks its own inference path from the token
        count (``expert_mlps_v2.py:1407-1500``):

          * ``seq_len == 1`` (TKG): ``T*top_k/E = 8/288 < 1.0`` so it takes
            ``forward_selective_loading`` — only the 8 chosen expert slabs are
            loaded, which is the decode behaviour we want.
          * ``seq_len > 1`` (CTE): at ``T*top_k >= block_size`` (512) it takes
            ``forward_blockwise``, i.e. the fused NKI blockwise kernel that
            ``use_shard_on_intermediate_dynamic_while`` selects.

        The routed output is **not** reduced inside ``ExpertMLPs``
        (``Experts`` constructs ``down_proj`` with ``reduce_output=False``);
        NxDI's own ``MoE.forward`` does the delayed all-reduce afterwards
        (``modules/moe/model.py:238-245``).  This block reproduces that
        contract explicitly — see ``forward``.

        The shared expert stays a plain Column/Row-parallel MLP: its
        ``down_proj`` is a ``RowParallelLinear`` with ``input_is_parallel``,
        so it reduces itself and must be added *after* the routed reduce, not
        before.
        """

        def __init__(self, config: Glm53FlashNeuronInferenceConfig) -> None:
            super().__init__()
            src = _require_source_config(config)
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
            # Round-3: the routed-expert slabs are sharded on the intermediate
            # axis, exactly like the shared expert's ColumnParallel gate/up +
            # RowParallel down.  Round 2 declared them at full width and
            # replicated, which is 288 x 4096 x 2048 x 2 B = 4.8 GiB per slab
            # per layer per rank — unschedulable.  Sharding gives
            # moe_intermediate_per_tp = 2048/tp.
            self.moe_intermediate_per_tp = self.moe_intermediate_size // tp_degree
            router_dtype = torch.float32 if src.moe_router_dtype == "float32" else dtype
            self.router = nn.Linear(
                self.hidden_size,
                self.n_routed_experts,
                bias=False,
                dtype=router_dtype,
            )
            # Routed experts: NxDI's blockwise-MoE module.  The GLM SwiGLU is
            # `silu(clamp(gate, max=L)) * clamp(up, -L, L)`, which maps onto
            # `GLUType.GLU` with `hidden_act="silu"`, `hidden_act_scaling_factor=1`
            # and the four clamp limits — an exact spelling of `moe.py`'s
            # reference expression, not an approximation of it.
            #
            # `normalize_top_k_affinities` carries GLM's `norm_topk_prob`.
            # `routed_scaling_factor` is applied to the OUTPUT in `forward`,
            # because applying it to the affinities would be cancelled by that
            # same normalize (see `glm53_route_affinities`).
            blockwise = getattr(config.neuron_config, "blockwise_matmul_config", None)
            if blockwise is None or not getattr(
                blockwise, "use_shard_on_intermediate_dynamic_while", False
            ):
                raise RuntimeError(
                    "GLM-5.3-Flash routed MoE requires "
                    "neuron_config.blockwise_matmul_config."
                    "use_shard_on_intermediate_dynamic_while=True. Without it "
                    "the LNC=2 dispatch (modules/moe/blockwise.py:1005-1017) "
                    "falls into `_call_shard_hidden_kernel`, which is a stub "
                    "that unconditionally raises on container sha256:011d49c7. "
                    f"Got: {blockwise!r}"
                )
            lnc = int(getattr(config.neuron_config, "logical_nc_config", 2))
            if lnc != 2:
                raise NotImplementedError(
                    "GLM-5.3-Flash MoE requires LNC=2 on this container: the "
                    "LNC=1 branch of the blockwise dispatch raises "
                    '"LNC_1 kernels not available in nkilib" '
                    "(modules/moe/blockwise.py:1018). "
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
                capacity_factor=None,  # dropless / full capacity
                normalize_top_k_affinities=self.norm_topk_prob,
                gate_clamp_upper_limit=self.swiglu_limit,
                gate_clamp_lower_limit=None,
                up_clamp_upper_limit=self.swiglu_limit,
                up_clamp_lower_limit=-self.swiglu_limit,
                early_expert_affinity_modulation=False,
                dtype=dtype,
                logical_nc_config=lnc,
                use_shard_on_intermediate_dynamic_while=True,
                skip_dma_token=bool(getattr(blockwise, "skip_dma_token", True)),
                block_size=int(getattr(blockwise, "block_size", 512)),
            )
            self.shared_expert = _MoESharedExpert(config)
            # GLM's router uses a selection-only correction bias (see
            # `glm52_moe_dsa.moe.select_glm52_experts`): it moves WHICH experts
            # win top-k but must never leak into the routing weights.  Declared
            # unconditionally with an explicit zero default so a checkpoint
            # that omits it degrades to plain top-k rather than to `None`.
            self.e_score_correction_bias = nn.Parameter(
                torch.zeros(self.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
            # Tier-1 CPU battery on the device-kernel identity: partition cap,
            # I_TP % 16, top-k in the tested set.  Runs at construction so a
            # bad shape family can never reach a compile submit.
            self.dispatch_config = build_glm53_moe_dispatch_config(
                hidden=self.hidden_size,
                num_experts=self.n_routed_experts,
                top_k=self.num_experts_per_tok,
                intermediate_global=self.moe_intermediate_size,
                tp_degree=tp_degree,
                renormalize_topk=self.norm_topk_prob,
            )

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            """Round-4 MoE: GLM routing + NxDI blockwise ExpertMLPs + shared.

            Kernel identity: ``MOE_KERNEL_SLUG_V1``.  ``self.dispatch_config``
            still carries the Tier-1-validated shape identity (partition cap
            288 < 16384, ``I_TP = 2048/16 = 128`` clears the ``%16`` wall,
            ``top_k=8`` in the tested set); it is now a *gate* on the config
            rather than the thing that does the dispatch.

            No fallback: there is no branch that silently drops to
            ``torch_blockwise_matmul_inference`` (``use_torch_block_wise``
            stays False) or skips the routed half.
            """
            shape = hidden_states.shape
            length = shape[1] if hidden_states.ndim == 3 else 1
            flat = hidden_states.reshape(-1, self.hidden_size)

            shared = self.shared_expert(hidden_states)

            affinities, expert_index = glm53_route_affinities(
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
            # `ExpertMLPs` builds `down_proj` with `reduce_output=False`, so
            # each rank holds a partial sum over its intermediate slice.  NxDI's
            # own `MoE.forward` does this reduce afterwards
            # (modules/moe/model.py:238-245); this block is not that class, so
            # it must do it here.  Without the reduce every rank would emit
            # 1/tp of the routed activation — a silent correctness bug, not a
            # crash.
            routed = _reduce_from_tp_region(routed)
            # GLM's `routed_scaling_factor`, applied to the output rather than
            # to the affinities: the expert combination is linear in the
            # affinities, and pre-scaling them would be cancelled by
            # `normalize_top_k_affinities`.
            routed = routed * self.routed_scaling_factor
            return shared + routed.view(shape).to(shared.dtype)

    class _MHCBlock(nn.Module):
        """One mHC 4-stream pre/post mixer (Sinkhorn manifold projection).

        Weights are small (rows = (2+hc_mult)*hc_mult, cols = hc_mult*hidden)
        and replicated across ranks.  Torch-primitive forward matches
        ``mhc.py`` bit-for-bit.
        """

        def __init__(self, config: Glm53FlashNeuronInferenceConfig) -> None:
            super().__init__()
            src = _require_source_config(config)
            self.hc_mult = src.hc_mult
            self.hidden_size = src.hidden_size
            self.rms_eps = src.rms_norm_eps
            self.hc_eps = src.hc_eps
            self.sinkhorn_iters = src.hc_sinkhorn_iters
            self.post_alpha = src.hc_post_alpha
            mix_rows = (2 + self.hc_mult) * self.hc_mult
            dtype = config.neuron_config.torch_dtype
            self.fn = nn.Parameter(
                torch.empty(mix_rows, self.hc_mult * self.hidden_size, dtype=dtype),
                requires_grad=False,
            )
            self.base = nn.Parameter(
                torch.zeros(mix_rows, dtype=torch.float32), requires_grad=False
            )
            self.scale = nn.Parameter(
                torch.ones(3, dtype=torch.float32), requires_grad=False
            )

        def pre(
            self, residual: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if residual.shape[-2:] != (self.hc_mult, self.hidden_size):
                raise ValueError("mHC residual has an incorrect stream/hidden shape")
            outer_shape = residual.shape[:-2]
            flat = residual.reshape(-1, self.hc_mult, self.hidden_size)
            tokens = flat.shape[0]
            x = flat.flatten(1).to(torch.float32)
            mixes = x @ self.fn.to(torch.float32).t()
            variance = x.square().mean(dim=-1, keepdim=True)
            mixes = mixes * torch.rsqrt(variance + self.rms_eps)

            pre_logits = (
                mixes[:, : self.hc_mult] * self.scale[0] + self.base[: self.hc_mult]
            )
            pre_mix = torch.sigmoid(pre_logits) + self.hc_eps
            post_logits = (
                mixes[:, self.hc_mult : 2 * self.hc_mult] * self.scale[1]
                + self.base[self.hc_mult : 2 * self.hc_mult]
            )
            post_mix = torch.sigmoid(post_logits) * self.post_alpha
            comb_logits = mixes[:, 2 * self.hc_mult :].view(
                tokens, self.hc_mult, self.hc_mult
            )
            comb_logits = comb_logits * self.scale[2] + self.base[
                2 * self.hc_mult :
            ].view(1, self.hc_mult, self.hc_mult)
            comb_mix = torch.softmax(comb_logits, dim=-1) + self.hc_eps
            comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + self.hc_eps)
            for _ in range(self.sinkhorn_iters - 1):
                comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + self.hc_eps)
                comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + self.hc_eps)
            layer_input = torch.sum(
                pre_mix.unsqueeze(-1) * flat.to(torch.float32), dim=1
            ).to(residual.dtype)
            return (
                post_mix.view(*outer_shape, self.hc_mult, 1),
                comb_mix.view(*outer_shape, self.hc_mult, self.hc_mult),
                layer_input.view(*outer_shape, self.hidden_size),
            )

        @staticmethod
        def post(
            layer_output: torch.Tensor,
            residual: torch.Tensor,
            post_mix: torch.Tensor,
            comb_mix: torch.Tensor,
        ) -> torch.Tensor:
            mixed_residual = torch.einsum(
                "...ij,...ih->...jh",
                comb_mix.to(torch.float32),
                residual.to(torch.float32),
            )
            post_term = post_mix.to(torch.float32) * layer_output.unsqueeze(-2).to(
                torch.float32
            )
            return (mixed_residual + post_term).to(residual.dtype)

    class Glm53FlashLayer(nn.Module):
        """One GLM-5.3-Flash decoder layer, dispatched by layer_idx."""

        def __init__(
            self, config: Glm53FlashNeuronInferenceConfig, *, layer_idx: int
        ) -> None:
            super().__init__()
            src = _require_source_config(config)
            self.layer_idx = layer_idx
            self.rms_eps = src.rms_norm_eps
            dtype = config.neuron_config.torch_dtype
            self.input_norm_weight = _rms_norm_weight(src.hidden_size, dtype)
            self.post_attention_norm_weight = _rms_norm_weight(src.hidden_size, dtype)

            layer_type = src.layer_types[layer_idx]
            if layer_type == "deepseek_sparse_attention":
                self.attn_kind = "dsa"
                self.self_attn = _DSABlock(config, layer_idx=layer_idx)
            elif layer_type == "linear_attention":
                self.attn_kind = "kda"
                self.self_attn = _KDABlock(config, layer_idx=layer_idx)
            else:
                raise ValueError(
                    f"unsupported layer_type {layer_type!r} at index {layer_idx}"
                )

            mlp_type = src.mlp_layer_types[layer_idx]
            if mlp_type == "dense":
                self.mlp_kind = "dense"
                self.mlp = _DenseMLPBlock(config)
            elif mlp_type == "sparse":
                self.mlp_kind = "sparse"
                self.mlp = _MoEBlock(config)
            else:
                raise ValueError(
                    f"unsupported mlp_type {mlp_type!r} at index {layer_idx}"
                )

            self.hc_attn = _MHCBlock(config)
            self.hc_mlp = _MHCBlock(config)

        def state_cache_specs(
            self, batch: int, seq_len: int
        ) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
            return self.self_attn.state_cache_specs(batch, seq_len)

        def forward(
            self,
            residual_streams: torch.Tensor,
            position_ids: torch.Tensor,
            caches: tuple[torch.Tensor, ...],
            key_lengths: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
            """One layer + its aliased state, threaded in and back out.

            ``caches`` is this layer's slice of the model's aliased parameter
            list — 2 tensors for a KDA layer, 3 for a DSA layer — and the
            second return value is the updated slice, in the same order.  The
            model returns them all after the logits so NxDI's
            ``input_output_aliases`` map turns each into an on-device
            read-modify-write.
            """
            post_mix, comb_mix, hidden_states = self.hc_attn.pre(residual_streams)
            normalized = _rms_norm(hidden_states, self.input_norm_weight, self.rms_eps)
            if self.attn_kind == "dsa":
                attn_out, new_caches = self.self_attn(
                    normalized, position_ids, caches, key_lengths=key_lengths
                )
            else:
                attn_out, new_state, new_conv = self.self_attn(
                    normalized, position_ids, caches[0], caches[1]
                )
                new_caches = (new_state, new_conv)
            residual_streams = self.hc_attn.post(
                attn_out, residual_streams, post_mix, comb_mix
            )
            post_mix, comb_mix, hidden_states = self.hc_mlp.pre(residual_streams)
            normalized = _rms_norm(
                hidden_states, self.post_attention_norm_weight, self.rms_eps
            )
            mlp_out = self.mlp(normalized)
            return (
                self.hc_mlp.post(mlp_out, residual_streams, post_mix, comb_mix),
                new_caches,
            )

    class _NeuronGlm53FlashModel(NeuronBaseModel):
        """Real per-layer NxDI-primitive GLM-5.3-Flash graph.

        Round 2: replaces the Round-1 single-Linear shell with
        ``ParallelEmbedding + 45 x Glm53FlashLayer + norm + ColumnParallelLinear
        LM head``.  Layer dispatch honours ``config.layer_types`` and
        ``config.mlp_layer_types``.

        Round 3 (device): drops in NKI kernels for KDA-state, DSA sparse-attn,
        and blockwise-MoE dispatch.  The Round-2 forward for those paths
        raises ``NotImplementedError`` per the fallback-rule discipline.
        """

        def setup_attr_for_model(self, config: Any) -> None:
            self.on_device_sampling = (
                config.neuron_config.on_device_sampling_config is not None
            )
            self.tp_degree = config.neuron_config.tp_degree
            self.hidden_size = config.hidden_size
            self.num_attention_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.max_batch_size = config.neuron_config.max_batch_size
            self.buckets = config.neuron_config.buckets

        def init_model(self, config: Any) -> None:
            src = _require_source_config(config)
            self.padding_idx = getattr(config, "pad_token_id", 0)
            self.vocab_size = config.vocab_size
            self.hc_mult = src.hc_mult
            self.embed_tokens = _NxdParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=True,
                pad=True,
                sequence_parallel_enabled=(
                    config.neuron_config.sequence_parallel_enabled
                ),
            )
            self.layers = nn.ModuleList(
                [
                    Glm53FlashLayer(config, layer_idx=layer_idx)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )
            self.final_norm_weight = _rms_norm_weight(
                config.hidden_size, config.neuron_config.torch_dtype
            )
            self.lm_head = _NxdColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                pad=True,
                gather_output=not self.on_device_sampling,
                dtype=config.neuron_config.torch_dtype,
            )
            self.rms_eps = src.rms_norm_eps

        def init_inference_optimization(self, config: Any) -> None:
            """Replace NxDI's KV-cache manager with GLM-5.3's hybrid state cache.

            ``KVCacheManager`` allocates a uniform ``2 x num_layers`` K/V set
            shaped from ``num_key_value_heads`` and ``head_dim``.  GLM-5.3-Flash
            is not that model: 34 of its 45 layers are KDA, whose state is a
            fixed ``[HV, V, K]`` matrix plus a short-conv history and does not
            grow with sequence length at all, and the 11 DSA layers need a
            *third* buffer (the lightning indexer's index-K) that no
            attention-shaped manager knows about.  A uniform manager would
            therefore allocate the wrong tensors for 76% of the stack and still
            leave the indexer uncached.

            So ``kv_mgr`` is set to ``None`` and this model owns
            ``self.past_key_values`` directly — the documented second branch of
            ``DecoderModelInstance.get()`` (model_wrapper.py:1614-1619).  Every
            entry becomes an ``input_output_aliases`` pair, is read as a graph
            input in ``forward``, and is returned after the logits in the same
            order.

            The super() call still runs first: it builds the on-device sampler.
            Its ``KVCacheManager`` is constructed and then dropped, which costs
            a transient CPU allocation at build time and nothing on device.
            """
            super().init_inference_optimization(config)
            neuron_config = config.neuron_config
            if getattr(neuron_config, "is_prefix_caching", False):
                raise NotImplementedError(
                    "GLM-5.3-Flash state aliasing assumes a context-encoding "
                    "graph starts a fresh sequence at position 0 (see "
                    "`_write_positions`). Prefix caching breaks that "
                    "assumption; refusing rather than writing the cache at the "
                    "wrong offsets."
                )
            if getattr(neuron_config, "is_block_kv_layout", False):
                raise NotImplementedError(
                    "GLM-5.3-Flash does not implement a paged/block KV layout; "
                    "its KDA state is per-slot and not per-block."
                )
            self.kv_mgr = None

            batch = int(
                getattr(neuron_config, "kv_cache_batch_size", None)
                or neuron_config.max_batch_size
            )
            seq_len = int(neuron_config.seq_len)
            self.state_cache_batch = batch
            self.state_cache_seq_len = seq_len

            specs: list[tuple[str, tuple[int, ...], torch.dtype]] = []
            self.layer_cache_slices: list[tuple[int, int]] = []
            for layer_idx, layer in enumerate(self.layers):
                layer_specs = layer.state_cache_specs(batch, seq_len)
                start = len(specs)
                specs.extend(
                    (f"layer{layer_idx}.{name}", shape, dtype)
                    for name, shape, dtype in layer_specs
                )
                self.layer_cache_slices.append((start, len(specs)))
            self.state_cache_names = [name for name, _, _ in specs]
            self.past_key_values = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
                    for _, shape, dtype in specs
                ]
            )

        def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            seq_ids: torch.Tensor | None = None,
            sampling_params: torch.Tensor | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            """Match ``NeuronBaseModel.forward``'s positional contract.

            NxDI's tracer calls the model with its full positional input list
            (input_ids, attention_mask, position_ids, seq_ids, sampling_params,
            then a long optional tail).  Round 2 declared ``(input_ids,
            positions, **kwargs)``, so the tracer's 8 positional arguments
            never bound — the graph could not be generated at all.  The
            trailing ``*args`` absorbs the optional tail (slot_mapping,
            block tables, tile indices, …) that this graph does not consume.
            """
            positions = position_ids
            hidden_states = self.embed_tokens(input_ids)
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.unsqueeze(0)
            batch, length, _ = hidden_states.shape
            if positions is None:
                positions = (
                    torch.arange(length, dtype=torch.int64, device=hidden_states.device)
                    .unsqueeze(0)
                    .expand(batch, -1)
                )
            if positions.ndim == 1:
                positions = positions.expand(batch, -1)
            positions = positions.to(torch.int64)
            # `attention_mask` is consumed for real: it zeroes padded tokens so
            # they cannot contribute to the KDA recurrence (which, being a
            # running state, would otherwise carry padding forward into every
            # subsequent step) and it supplies DSA's key lengths.
            key_lengths = None
            if attention_mask is not None:
                mask = attention_mask.to(torch.int64)
                if mask.ndim == 2 and mask.shape[-1] == length:
                    hidden_states = hidden_states * mask.unsqueeze(-1).to(
                        hidden_states.dtype
                    )
                key_lengths = mask.reshape(mask.shape[0], -1).sum(dim=-1)

            # mHC 4-stream residual widening (matches Impl.forward at model.py:140).
            residual_streams = hidden_states.unsqueeze(-2).repeat(1, 1, self.hc_mult, 1)
            # Aliased state in, updated state out — one slice per layer.
            caches_in = list(self.past_key_values)
            caches_out: list[torch.Tensor] = []
            for layer_idx, layer in enumerate(self.layers):
                start, stop = self.layer_cache_slices[layer_idx]
                residual_streams, updated = layer(
                    residual_streams,
                    positions,
                    tuple(caches_in[start:stop]),
                    key_lengths=key_lengths,
                )
                if len(updated) != stop - start:
                    raise RuntimeError(
                        f"layer {layer_idx} returned {len(updated)} state "
                        f"tensors but declared {stop - start}; the alias list "
                        "and the output list must agree exactly or lowering "
                        "aborts with 'parameter not found in lowering context'"
                    )
                caches_out.extend(updated)
            hidden_states = residual_streams.mean(dim=-2)
            hidden_states = _rms_norm(
                hidden_states, self.final_norm_weight, self.rms_eps
            )
            logits = self.lm_head(hidden_states)

            # State alias contract.
            #
            # `DecoderModelInstance.get()` (model_wrapper.py:1614-1619) builds
            # `input_output_aliases` from `kv_mgr.past_key_values`, or — when
            # `kv_mgr is None`, which is GLM-5.3-Flash's case — from the
            # model's own `past_key_values`.  Each entry maps to output index
            # `num_output_from_trace + i`.  Unlike the example inputs, that
            # alias list is NOT filtered for -1 before `linearize_indices`
            # (hlo_conversion.py:490-496), so a state parameter that the graph
            # aliases but never reads aborts lowering with
            # "parameter not found in lowering context".
            #
            # Every NxDI model in-tree avoids this by not overriding `forward`
            # at all — the base `NeuronBaseModel.forward` reads the cache via
            # `kv_mgr.get_cache` and returns `outputs += updated_kv_cache`.
            # This graph keeps its own forward (GLM-5.3 is a hybrid KDA/DSA
            # stack that the base decode loop does not model), so it honours
            # the same contract explicitly.
            #
            # Round 4 returns the *updated* tensors here.  Round 3 returned the
            # parameters themselves — which lowered cleanly and made every
            # alias a no-op write, so a multi-step TKG graph restarted from
            # zero KDA state and a one-position DSA context on every step.  The
            # difference between those two is invisible in a benchmark and is
            # the whole point of this round.
            expected = len(self.past_key_values)
            if len(caches_out) != expected:
                raise RuntimeError(
                    f"forward produced {len(caches_out)} state outputs but "
                    f"{expected} parameters are aliased; NxDI maps alias i to "
                    "output index num_output_from_trace + i, so the lists must "
                    "be the same length and in the same order"
                )
            if expected:
                return [logits] + caches_out
            return logits

    def _require_source_config(
        config: Any,
    ) -> Glm53FlashInferenceConfig:
        """Extract the frozen Glm53FlashInferenceConfig from an NxDI config."""
        source = getattr(config, "source_config", None)
        if source is None:
            raise RuntimeError(
                "the NxDI config passed to the Round-2 GLM-5.3-Flash wrapper "
                "must carry a `source_config` of type Glm53FlashInferenceConfig; "
                "call `Glm53FlashNeuronInferenceConfig(..., source_config=...)` "
                "or use `NeuronGlm53FlashForCausalLM.build_inference_config`"
            )
        if not isinstance(source, Glm53FlashInferenceConfig):
            raise TypeError(
                "source_config must be a Glm53FlashInferenceConfig; got "
                f"{type(source).__name__}"
            )
        return source

    class NeuronGlm53FlashForCausalLM(NeuronBaseForCausalLM):
        """Public NxDI-compatible GLM-5.3-Flash class.

        Compile invocation (see ``command.sh``):

            wrapper = NeuronGlm53FlashForCausalLM(model_path, inference_config)
            wrapper.compile(out_path)              # Round-2 primitive compile
            wrapper.compile(out_path, dry_run=True)  # driver-binding smoke

        The class-level ``_model_cls`` selects the real per-layer graph.  The
        `GLM53_SOURCE_CACHE_ABI` / `_GLM53_GRAPH_ID` fields (mirrored on the
        class) pin the compile cache per the same convention GLM-5.2 uses so
        the modular-compile flywheel indexer treats independent GLM-5.3-Flash
        artifacts as cache-distinct from GLM-5.2 and from Round-1 shells.
        """

        _model_cls = _NeuronGlm53FlashModel

        GLM53_SOURCE_CACHE_ABI = GLM53_SOURCE_CACHE_ABI
        _GLM53_GRAPH_ID = _GLM53_GRAPH_ID

        # HF checkpoint prefix strip.  We accept NxDI's default
        # ``_STATE_DICT_MODEL_PREFIX = "model."`` (strip to ``""``), so the
        # converter (``checkpoint_convert._convert_glm53_checkpoint``) sees
        # keys under ``language_model.`` (text) and ``visual.`` (vision).
        # ``lm_head.weight`` never carries a ``model.`` prefix in the HF
        # index and survives the strip untouched.  Keeping the default
        # matters because the strip only replaces one occurrence: switching
        # to ``"model.language_model."`` would leave ``model.visual.``
        # untouched, which would then miss ``is_vision_key`` and try to
        # convert vision tensors into the text module tree.

        # NxDI's default `emit_phases` (via `enable_context_encoding`
        # + `enable_token_generation`) covers CTE + TKG together.  A caller
        # that wants TKG-only or CTE-only artifacts sets the corresponding
        # env var before instantiation (mirrors the qwen35-2b driver):
        #   NXDI_EMIT_PHASES=TKG   -> only enable_token_generation
        #   NXDI_EMIT_PHASES=CTE   -> only enable_context_encoding
        #   NXDI_EMIT_PHASES=BOTH  -> default (CTE + TKG)
        _EMIT_PHASE_VALUES = frozenset({"BOTH", "CTE", "TKG"})

        @classmethod
        def get_config_cls(cls):
            return Glm53FlashNeuronInferenceConfig

        @staticmethod
        def load_hf_model(model_path: str, **kwargs: Any):
            """HF-side loader shim.

            NxDI's ``get_state_dict`` only calls this when ``model_path`` is
            neither an existing directory nor an existing file (i.e. a
            Hub model id).  On the compile path ``model_path`` is the
            snapshot directory ``.../04c4e9e9...`` so NxDI takes the
            ``load_state_dict(directory)`` branch instead and never touches
            this method.

            For the direct-HuggingFace path we intentionally do NOT
            instantiate a HF ``AutoModelForCausalLM`` — GLM-5.3-Flash has no
            in-tree HF modelling module and the transformers loader would
            need the full FP8 kernels the campaign explicitly does not
            ship.  Rather than raise, we return a minimal ``nn.Module``
            whose ``state_dict()`` reads the safetensors shards from disk
            via NxDI's own ``load_state_dict`` helper, which is exactly what
            the directory branch would have used.  This keeps the two
            code paths semantically identical so a caller that forgets the
            directory contract does not silently drop into a stub.

            Any other ``kwargs`` (``trust_remote_code``, ``revision``, ...)
            are accepted for API parity and ignored — the campaign never
            fetches from the network.
            """
            del kwargs  # network-fetch kwargs are inert here.
            import os

            from neuronx_distributed_inference.modules.checkpoint import (
                load_state_dict as _load_state_dict,
            )

            if not os.path.isdir(model_path):
                raise NotImplementedError(
                    "GLM-5.3-Flash load_hf_model requires a local snapshot "
                    f"directory; got {model_path!r}. Hub-name fetch is not "
                    "supported by this campaign."
                )
            sd = _load_state_dict(model_path)

            class _Glm53HfShell(torch.nn.Module):
                def __init__(self, state: dict) -> None:
                    super().__init__()
                    self._state = state

                def state_dict(self, *_args: Any, **_kwargs: Any) -> dict:
                    return self._state

            return _Glm53HfShell(sd)

        @staticmethod
        def convert_hf_to_neuron_state_dict(state_dict: dict, config: Any) -> dict:
            """Round-3 HF -> Neuron checkpoint conversion.

            Verified against the real index (`model.safetensors.index.json`,
            snapshot 04c4e9e9): 76,108 tensors, text prefix
            ``model.language_model.``, `lm_head.weight` unprefixed, 62 shards.

            Layer signatures actually present (4 distinct):
              SIG0  layers 0-2    KDA + dense MLP            (29 tensors)
              SIG1  layers 3,7..43 DSA + sparse MoE          (34 + 288x6)
              SIG2  the other 31  KDA + sparse MoE           (31 + 288x6)
              SIG3  layer 45      MTP (no hc_*)              -> DROPPED

            Quantization facts that drive this function (from the checkpoint,
            not assumed):
              * The ONLY scale suffix present is ``weight_scale_inv`` — a
                per-block reciprocal scale under
                ``weight_block_size = [128, 128]``.  ``weight_scale`` and
                ``input_scale`` are ABSENT (activation_scheme is "dynamic",
                so no static activation scales are stored).
              * 37,338 scale tensors, all on MoE experts / shared experts /
                dense MLP / MLA q_a,q_b,kv_a_proj_with_mqa,o_proj.
              * The whole KDA block is BF16 — no KDA projection carries a
                scale.  ``kv_b_proj`` is BF16 too, unlike its siblings.
              * ``fused_qkvbfg_a_proj`` / ``qkv_proj`` appear in the config's
                ``modules_to_not_convert`` but do NOT exist as tensors; KDA
                Q/K/V are separate, as are ``q_conv1d`` / ``k_conv1d`` /
                ``v_conv1d``.

            Anti-inheritance (the OCP-448-when-None bug): every FP8 scale
            field gets an explicit non-``None`` default and a load-time
            ``max(scale) <= 240.0`` assertion via ``validate_fp8_scale``.
            ``normalize_static_fp8_weight_format()`` is deliberately NOT
            called — its OCP-448 fallback when a scale is ``None`` is the
            defect this port refuses to inherit.
            """
            src = _require_source_config(config)
            tp_degree = config.neuron_config.tp_degree
            return _convert_glm53_checkpoint(state_dict, src, tp_degree=tp_degree)

        def __init__(
            self,
            model_path: str,
            config: Any | None = None,
            neuron_config: Any | None = None,
        ) -> None:
            _require_nxdi()
            emit_phases = os.environ.get("NXDI_EMIT_PHASES", "BOTH").upper()
            if emit_phases not in self._EMIT_PHASE_VALUES:
                raise ValueError(
                    f"NXDI_EMIT_PHASES={emit_phases!r} must be one of "
                    f"{sorted(self._EMIT_PHASE_VALUES)}"
                )
            self._emit_phases = emit_phases
            # The CPU-reference oracle stays wired but lazy: only the round-3
            # correctness-gate needs to actually instantiate it, and building
            # 45 layers × 288 experts eagerly would blow CPU memory on the
            # compile host.
            self._cpu_oracle: NeuronGlm53FlashForCausalLMImpl | None = None
            self._source_config: Glm53FlashInferenceConfig | None = None
            if isinstance(config, Glm53FlashInferenceConfig):
                self._source_config = config
                if neuron_config is None:
                    raise ValueError(
                        "GLM-5.3-Flash NxDI wrapper requires an explicit "
                        "NxDI NeuronConfig (build via `build_neuron_config`)."
                    )
                config = Glm53FlashNeuronInferenceConfig(
                    neuron_config=neuron_config,
                    source_config=config,
                )
            elif isinstance(config, Glm53FlashNeuronInferenceConfig):
                self._source_config = config.source_config
            super().__init__(model_path, config=config, neuron_config=neuron_config)
            self._apply_emit_phases()

        def _apply_emit_phases(self) -> None:
            """Honour ``NXDI_EMIT_PHASES`` by pruning ``self.models``.

            Round 3 parsed this env var and stored it on ``self._emit_phases``
            but never acted on it, so ``NXDI_EMIT_PHASES=TKG`` silently
            compiled both graphs anyway — found while trying to probe the
            45-layer decode contract without paying for the prefill trace.

            NxDI has no config switch for this: ``NeuronBaseForCausalLM``
            calls ``enable_context_encoding()`` unconditionally
            (model_base.py:3062) and appends each wrapper to ``self.models``,
            which is what ``compile()`` iterates.  Pruning that list is
            therefore the intervention point.  The attributes themselves stay
            in place so nothing downstream sees a half-built object; only the
            compile set changes.
            """
            if self._emit_phases == "BOTH":
                return
            cte = getattr(self, "context_encoding_model", None)
            tkg = getattr(self, "token_generation_model", None)
            keep = cte if self._emit_phases == "CTE" else tkg
            if keep is None:
                raise RuntimeError(
                    f"NXDI_EMIT_PHASES={self._emit_phases} but the "
                    "corresponding model wrapper was never built"
                )
            self.models = [m for m in self.models if m is keep]
            logger.info(
                "NXDI_EMIT_PHASES=%s -> compiling only %s",
                self._emit_phases,
                getattr(keep, "tag", keep),
            )

        def _copy_past_key_values(self, outputs: Any) -> None:  # type: ignore[override]
            """Thread state between graphs on the CPU-simulation path.

            On device this never runs: ``input_output_aliases`` makes the
            output buffer *be* the input buffer, and NxDI only calls this when
            ``not generation_model.is_neuron()`` (model_base.py:3442). The base
            implementation writes into
            ``<model>.model.kv_mgr.past_key_values`` (model_base.py:3789-3795),
            which is ``None`` here — GLM-5.3-Flash owns ``past_key_values``
            directly. Overriding keeps CPU simulation usable instead of leaving
            an ``AttributeError`` for whoever first tries to debug the model
            without a device.
            """
            offset = self._get_captured_tensors_offset()
            skip = (
                2
                if (
                    self.neuron_config.output_logits
                    and self.neuron_config.on_device_sampling_config
                )
                else 1
            )
            new_state = outputs[skip + offset :]
            for model in (self.token_generation_model, self.context_encoding_model):
                target = getattr(getattr(model, "model", None), "past_key_values", None)
                if target is None:
                    continue
                for i, value in enumerate(new_state):
                    target[i].data = value

        def get_cpu_oracle(self) -> NeuronGlm53FlashForCausalLMImpl:
            """Materialize the CPU-reference impl for correctness gating.

            Cached so repeated calls in a test session are cheap.  Kept off the
            compile path — nothing in ``compile()`` or ``load_weights()``
            touches this method.
            """
            if self._cpu_oracle is None:
                if self._source_config is None:
                    raise RuntimeError(
                        "no Glm53FlashInferenceConfig on this wrapper; pass "
                        "the source config to `__init__` (or set "
                        "`self._source_config` explicitly)"
                    )
                self._cpu_oracle = NeuronGlm53FlashForCausalLMImpl(self._source_config)
            return self._cpu_oracle

        def load_weights(  # type: ignore[override]
            self, compiled_model_path: str, **kwargs: Any
        ) -> None:
            """Rank-sharded FP8 load — delegates to NxDI's base with a preflight.

            The Round-2 loader follows the contract laid out in the Round-1
            docstring:

            1. Preflight: bounded-check every DSA-layer's
               ``indexer.cache_quant_multiplier`` scalar against the Trainium2
               native-e4m3fn cap of 240.0.  Uses Fleet A's audit helper
               (``glm52_indexer_fp8_scale_fix.assert_indexer_multiplier_bounded``)
               when it is importable; otherwise falls back to an in-wrapper
               scalar cap check.  Fail-closed: an out-of-range multiplier
               refuses the load.
            2. Delegate to ``NeuronApplicationBase.load_weights`` so the
               per-rank ``weights/tp{rank}_sharded_checkpoint.safetensors``
               contract is honoured and each rank's ``traced_model.nxd_model``
               initialises with the sharded weights.

            The per-layer cache-multiplier scalars themselves are materialised
            by NxDI's base loader via the ``cache_quant_multiplier`` buffer
            declared on ``_DSAIndexerBlock``; step 1 only *validates* the
            values before compile-time constants are frozen.
            """
            self._assert_indexer_multipliers_bounded(compiled_model_path)
            super().load_weights(compiled_model_path, **kwargs)

        def _assert_indexer_multipliers_bounded(self, compiled_model_path: str) -> None:
            """Bounded-check every DSA layer's indexer_cache_quant_multiplier."""
            # Best-effort import of Fleet A's audit helper; the wrapper stays
            # importable when the harness scratchpad isn't on PYTHONPATH.
            checker = _load_indexer_bound_checker()
            src = self._source_config
            if src is None:
                return
            for layer_idx, layer_type in enumerate(src.layer_types):
                if layer_type != "deepseek_sparse_attention":
                    continue
                value = float(src.indexer_cache_quant_multiplier)
                if checker is not None:
                    checker(value, layer_idx=layer_idx)
                else:
                    # Fallback: bound at the Trainium2 e4m3fn cap.
                    if not (0.0 < value <= 240.0):
                        raise ValueError(
                            f"indexer cache_quant_multiplier for layer "
                            f"{layer_idx} = {value!r} outside (0, 240] "
                            "Trainium2 e4m3fn range"
                        )

        # ------------------------------------------------------------------
        # Helpers exposed to the compile driver in `command.sh`.
        # ------------------------------------------------------------------

        @classmethod
        def build_inference_config(
            cls,
            source_config: Glm53FlashInferenceConfig,
            *,
            tp_degree: int,
            ctx_batch_size: int,
            tkg_batch_size: int,
            seq_len: int,
            **extra_neuron_kwargs: Any,
        ) -> Glm53FlashNeuronInferenceConfig:
            neuron_config = build_neuron_config(
                tp_degree=tp_degree,
                ctx_batch_size=ctx_batch_size,
                tkg_batch_size=tkg_batch_size,
                seq_len=seq_len,
                torch_dtype=source_config.torch_dtype,
                extra=extra_neuron_kwargs or None,
            )
            return Glm53FlashNeuronInferenceConfig(
                neuron_config=neuron_config,
                source_config=source_config,
            )

        @classmethod
        def build_one_layer_smoke_config(
            cls,
            source_config: Glm53FlashInferenceConfig,
            *,
            tp_degree: int = 8,
            ctx_batch_size: int = 1,
            tkg_batch_size: int = 1,
            seq_len: int = 128,
        ) -> Glm53FlashNeuronInferenceConfig:
            """Config override for the 1-layer compile-driver smoke.

            Frozen-field validators in ``Glm53FlashInferenceConfig`` reject
            arbitrary architecture changes; we clone the source config with
            ``allow_reduced_shapes=True`` so ``num_hidden_layers=1`` passes.
            The frozen fields (vocab, hidden_size, expert count, etc.) stay
            untouched — the layer stack walks a proportionally shorter list,
            which is enough evidence the compile driver binds.

            Round-2 subtlety: the shortened layer is forced to KDA + dense so
            the Round-2 smoke does NOT trip the routed-MoE / DSA-sparse-attn
            NotImplementedError guards.  The single-KDA-layer forward still
            raises inside ``_KDABlock.forward`` (KDA state kernel is Round-3),
            so this config is intended for compile-driver *binding* smoke
            only, not for a compile-through-to-artifact test.
            """
            reduced = copy.deepcopy(source_config)
            fields_dict = {
                name: getattr(reduced, name) for name in reduced.__dataclass_fields__
            }
            fields_dict["allow_reduced_shapes"] = True
            fields_dict["num_hidden_layers"] = 1
            fields_dict["layer_types"] = ("linear_attention",)
            fields_dict["mlp_layer_types"] = ("dense",)
            fields_dict["linear_attn_config"] = copy.deepcopy(
                reduced.linear_attn_config
            )
            fields_dict["linear_attn_config"].kda_layers = (0,)
            fields_dict["linear_attn_config"].full_attn_layers = ()
            fields_dict["indexer_types"] = ("full",)
            slim = Glm53FlashInferenceConfig(**fields_dict)
            return cls.build_inference_config(
                slim,
                tp_degree=tp_degree,
                ctx_batch_size=ctx_batch_size,
                tkg_batch_size=tkg_batch_size,
                seq_len=seq_len,
            )

        # Named reduced layer stacks for bisecting which bound kernel breaks a
        # trace.  Each entry is an ordered list of (layer_type, mlp_type).
        SMOKE_RECIPES: dict[str, tuple[tuple[str, str], ...]] = {
            # KDA + dense only — the Round-3 passing baseline.
            "kda-dense": (("linear_attention", "dense"),),
            # Adds the 288-expert routed MoE, nothing else.
            "kda-moe": (
                ("linear_attention", "dense"),
                ("linear_attention", "sparse"),
            ),
            # Adds the DSA indexer + sparse attention, nothing else.
            "dsa-dense": (
                ("linear_attention", "dense"),
                ("deepseek_sparse_attention", "dense"),
            ),
            # The real layer-0..3 prefix: all three kernels in one graph.
            "full": (
                ("linear_attention", "dense"),
                ("linear_attention", "dense"),
                ("linear_attention", "dense"),
                ("deepseek_sparse_attention", "sparse"),
            ),
        }

        @classmethod
        def build_recipe_smoke_config(
            cls,
            source_config: Glm53FlashInferenceConfig,
            *,
            recipe: str,
            tp_degree: int = 16,
            ctx_batch_size: int = 1,
            tkg_batch_size: int = 1,
            seq_len: int = 128,
            **extra_neuron_kwargs: Any,
        ) -> Glm53FlashNeuronInferenceConfig:
            """Reduced config for one named layer recipe (see ``SMOKE_RECIPES``).

            Exists so a trace failure can be attributed to a *specific* bound
            kernel instead of to "the 4-layer smoke".  Every recipe keeps the
            frozen architecture constants (vocab, hidden, expert count, head
            dims); only the layer stack shortens.
            """
            if recipe not in cls.SMOKE_RECIPES:
                raise ValueError(
                    f"unknown smoke recipe {recipe!r}; have {sorted(cls.SMOKE_RECIPES)}"
                )
            stack = cls.SMOKE_RECIPES[recipe]
            reduced = copy.deepcopy(source_config)
            fields_dict = {
                name: getattr(reduced, name) for name in reduced.__dataclass_fields__
            }
            fields_dict["allow_reduced_shapes"] = True
            fields_dict["num_hidden_layers"] = len(stack)
            fields_dict["layer_types"] = tuple(a for a, _ in stack)
            fields_dict["mlp_layer_types"] = tuple(m for _, m in stack)
            fields_dict["indexer_types"] = ("full",) * len(stack)
            linear = copy.deepcopy(reduced.linear_attn_config)
            linear.kda_layers = tuple(
                i for i, (a, _) in enumerate(stack) if a == "linear_attention"
            )
            linear.full_attn_layers = tuple(
                i for i, (a, _) in enumerate(stack) if a == "deepseek_sparse_attention"
            )
            fields_dict["linear_attn_config"] = linear
            slim = Glm53FlashInferenceConfig(**fields_dict)
            return cls.build_inference_config(
                slim,
                tp_degree=tp_degree,
                ctx_batch_size=ctx_batch_size,
                tkg_batch_size=tkg_batch_size,
                seq_len=seq_len,
                **extra_neuron_kwargs,
            )

        @classmethod
        def build_kernel_coverage_smoke_config(
            cls,
            source_config: Glm53FlashInferenceConfig,
            *,
            tp_degree: int = 8,
            ctx_batch_size: int = 1,
            tkg_batch_size: int = 1,
            seq_len: int = 128,
        ) -> Glm53FlashNeuronInferenceConfig:
            """4-layer reduced config that exercises ALL THREE bound kernels.

            ``build_one_layer_smoke_config`` deliberately forces KDA + dense so
            the stack is minimal — which means a pass there says nothing about
            the DSA sparse-attention path or the routed-MoE path.  This config
            reproduces the real layer-0..3 prefix of GLM-5.3-Flash:

                layer 0  linear_attention + dense    (KDA kernel, dense MLP)
                layer 1  linear_attention + dense
                layer 2  linear_attention + dense
                layer 3  deepseek_sparse_attention + sparse
                         (DSA indexer kernel + 288-expert routed MoE kernel)

            so a pass covers KDA, DSA and MoE in one trace.  Layer 3 being the
            first DSA/MoE layer matches the checkpoint exactly
            (``first_k_dense_replace=3``, DSA at indices 3,7,...,43).
            """
            reduced = copy.deepcopy(source_config)
            fields_dict = {
                name: getattr(reduced, name) for name in reduced.__dataclass_fields__
            }
            fields_dict["allow_reduced_shapes"] = True
            fields_dict["num_hidden_layers"] = 4
            fields_dict["layer_types"] = (
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "deepseek_sparse_attention",
            )
            fields_dict["mlp_layer_types"] = (
                "dense",
                "dense",
                "dense",
                "sparse",
            )
            fields_dict["linear_attn_config"] = copy.deepcopy(
                reduced.linear_attn_config
            )
            fields_dict["linear_attn_config"].kda_layers = (0, 1, 2)
            fields_dict["linear_attn_config"].full_attn_layers = (3,)
            fields_dict["indexer_types"] = ("full",) * 4
            slim = Glm53FlashInferenceConfig(**fields_dict)
            return cls.build_inference_config(
                slim,
                tp_degree=tp_degree,
                ctx_batch_size=ctx_batch_size,
                tkg_batch_size=tkg_batch_size,
                seq_len=seq_len,
            )

    def _load_indexer_bound_checker():
        """Best-effort import of Fleet A's indexer multiplier bound checker.

        The staging scratchpad path is preferred; a package-local fallback
        makes the checker a no-op when the harness is not on PYTHONPATH (the
        in-wrapper 240.0 cap check still runs — see
        ``_assert_indexer_multipliers_bounded``).
        """
        try:
            from glm52_indexer_fp8_scale_fix import (
                assert_indexer_multiplier_bounded,
            )

            return assert_indexer_multiplier_bounded
        except Exception:  # pragma: no cover - optional import
            pass
        try:
            import importlib.util
            import sys

            candidate = (
                "C:\\Users\\apumu\\research\\InfinityAI\\gemma4-trn2-handoff\\"
                "harness-v2\\staging\\reference-sweep-20260826T2150Z\\kernels\\"
                "glm52_indexer_fp8_scale_fix.py"
            )
            if not os.path.isfile(candidate):
                return None
            spec = importlib.util.spec_from_file_location(
                "_glm53_flash_indexer_bound_checker", candidate
            )
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return getattr(module, "assert_indexer_multiplier_bounded", None)
        except Exception:  # pragma: no cover - optional import
            return None

else:  # pragma: no cover - CPU-only fallback

    class NeuronGlm53FlashForCausalLM:  # type: ignore[no-redef]
        """CPU-fallback stub: raises on instantiation when NxDI is missing."""

        _model_cls = None
        GLM53_SOURCE_CACHE_ABI = GLM53_SOURCE_CACHE_ABI
        _GLM53_GRAPH_ID = _GLM53_GRAPH_ID

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _require_nxdi()


__all__ = [
    "DSA_CPU_GOLDEN_SLUG",
    "DSA_NKI_V2_SLUG",
    "GLM53_BLOCKWISE_MATMUL_WORKAROUND",
    "KDA_CPU_GOLDEN_SLUG",
    "KDA_NKI_V3P2_SLUG",
    "Glm53FlashNeuronInferenceConfig",
    "NeuronGlm53FlashForCausalLM",
    "build_neuron_config",
    "get_emitted_kernel_slugs",
]
