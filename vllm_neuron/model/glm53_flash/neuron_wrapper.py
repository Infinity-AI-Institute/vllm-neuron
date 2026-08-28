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
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint_convert import _convert_glm53_checkpoint
from .config import Glm53FlashInferenceConfig
from .model import NeuronGlm53FlashForCausalLMImpl
from .nki_bindings import (
    DSA_KERNEL_SLUG_V0,
    KDA_KERNEL_SLUG_V2,
    MOE_KERNEL_SLUG_V1,
    build_glm53_moe_dispatch_config,
    dsa_attend_from_scores,
    dsa_scores_from_qidx,
    glm53_route,
    kda_state_forward_torch,
    moe_gather_dispatch_torch,
)
from .registry import GLM53_SOURCE_CACHE_ABI, _GLM53_GRAPH_ID

logger = logging.getLogger(__name__)

# NxDI-container guarded imports.  On CPU-only hosts (developer laptops
# running the source-qualification tests) the Neuron toolchain is absent; the
# wrapper class must still be importable so the package `__init__.py` can
# expose it in a single place.  Instantiation raises when the toolchain is
# actually missing.
try:
    from neuronx_distributed_inference.models.model_base import (
        NeuronBaseForCausalLM,
        NeuronBaseModel,
    )
    from neuronx_distributed_inference.models.config import (
        InferenceConfig as _NxdiInferenceConfig,
        MoENeuronConfig as _NxdiMoENeuronConfig,
        NeuronConfig as _NxdiNeuronConfig,
    )
    from neuronx_distributed.parallel_layers.layers import (
        ColumnParallelLinear as _NxdColumnParallelLinear,
        ParallelEmbedding as _NxdParallelEmbedding,
        RowParallelLinear as _NxdRowParallelLinear,
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
    _NXDI_AVAILABLE = False
    _NXDI_IMPORT_ERROR = exc


# The MoE workaround from [[nxdi-container-moe-blockwise-mm-workaround-20260827]].
# Container `sha256:011d49c7...` is missing `_call_shard_hidden_kernel`; every
# GLM-5.3-Flash MoE compile must set this before `InferenceConfig` is created.
GLM53_BLOCKWISE_MATMUL_WORKAROUND: dict[str, bool] = {
    "use_shard_on_intermediate_dynamic_while": True,
    "skip_dma_token": True,
}


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
        """The KV-cache parameters NxDI will alias, in alias order.

        Mirrors ``DecoderModelInstance.get()`` (model_wrapper.py:1614-1619):
        prefer ``kv_mgr.past_key_values``, else the model's own
        ``past_key_values``.  Returns an empty list when neither exists, in
        which case there are no aliases to honour.

        Note that *unused example inputs* need no such handling — torch_neuronx
        filters those to an ``exclude`` list with a warning
        (hlo_conversion.py:465-485).  Only the alias list is unfiltered.
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
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            return query, key, value

    class _DSAIndexerBlock(nn.Module):
        """DSA lightning-indexer projections + IndexPool tail-select params.

        - K_proj is column-parallel along the indexer head axis so the per-rank
          slice materialises ``[index_n_heads_per_rank * index_head_dim]``.
        - Q_proj is a per-index-head parameter of shape
          ``[index_n_heads, num_attention_heads * qk_head_dim, index_head_dim]``.
          The frozen index-heads-per-rank slice is materialised at load time via
          the ``local_indexer_head_slice(rank)`` helper below.
        - pool_weights is a small ``[index_kpool]`` parameter replicated across ranks.

        The sparse-attn kernel + tail-select selection call is deferred to
        Round 3 (NKI v0 lightning-indexer wrapper); ``forward`` raises
        ``NotImplementedError`` until then.
        """

        def __init__(
            self, config: Glm53FlashNeuronInferenceConfig, *, layer_idx: int
        ) -> None:
            super().__init__()
            src = _require_source_config(config)
            self.layer_idx = layer_idx
            self.hidden_size = src.hidden_size
            self.index_n_heads = src.index_n_heads
            self.index_head_dim = src.index_head_dim
            self.index_topk = src.index_topk
            self.index_kpool = src.index_kpool
            self.always_select_tail = src.index_kpool_always_select_tail
            self.compress = src.index_kpool_compress
            self.num_attention_heads = src.num_attention_heads
            self.qk_head_dim = src.qk_head_dim
            dtype = config.neuron_config.torch_dtype
            tp_degree = config.neuron_config.tp_degree
            # Round-3 TP contract.  The indexer's K side is *replicated*, not
            # sharded: the top-k it produces has to be identical on every rank
            # (ranks hold different head shards of the same KV, and they must
            # gather the same sparse positions).  `gather_output=True` gives
            # every rank the full index-K, so the selection is bit-identical
            # across ranks by construction rather than by an extra all-reduce.
            # The projection is small (hidden -> 32*128) so replicating its
            # compute is cheap relative to getting the selection wrong.
            #
            # Round 2's `index_n_heads % tp_degree` guard was the wrong
            # invariant — at TP=16 it passes (32 % 16 == 0) but leaves 2 index
            # heads per rank, which then fails the IndexPool=4 collapse. The
            # pool divisibility is the real constraint.
            self.pooled_index_heads = self.index_n_heads // self.index_kpool
            if self.index_n_heads % self.index_kpool:
                raise ValueError(
                    f"DSA indexer requires index_n_heads ({self.index_n_heads}) "
                    f"divisible by IndexPool ({self.index_kpool})"
                )
            self.tp_degree = tp_degree
            self.heads_per_rank = self.num_attention_heads // tp_degree
            self.k_proj = _NxdColumnParallelLinear(
                self.hidden_size,
                self.index_n_heads * self.index_head_dim,
                bias=False,
                gather_output=True,
                dtype=dtype,
            )
            # Q_proj is a rank-3 parameter; store the full tensor and slice at
            # load time by rank.  Kept as an nn.Parameter (not a ColumnParallelLinear)
            # because the slicing axis (index-head) is the leading axis and
            # NxDI's ColumnParallelLinear shards only the output axis of a 2D
            # weight.  Round-3 loader materialises the rank-local slice.
            # Q_proj contracts the *main-attention* query (which IS sharded by
            # head) into the indexer space, so its middle axis is rank-local
            # and the resulting q_idx must be summed across ranks before it can
            # be scored.  Leading axis is the POOLED index-head count: the
            # golden scores `q_idx[B,Q,H_idx,D_idx]` against the pool-collapsed
            # `k_pooled[B,L,H_idx,D_idx]`, and the collapse divides the 32
            # stored index heads by IndexPool=4 -> 8.  `_assert_indexer_shapes`
            # re-checks this at trace time so a wrong reading of the HF layout
            # is a hard error, never silent corruption.
            self.q_proj = nn.Parameter(
                torch.empty(
                    self.pooled_index_heads,
                    self.heads_per_rank * self.qk_head_dim,
                    self.index_head_dim,
                    dtype=dtype,
                ),
                requires_grad=False,
            )
            self.pool_weights = nn.Parameter(
                torch.full((self.index_kpool,), 1.0 / self.index_kpool),
                requires_grad=False,
            )
            self.register_buffer(
                "cache_quant_multiplier",
                torch.tensor(
                    src.indexer_cache_quant_multiplier, dtype=torch.float32
                ),
            )

        def local_query_slice(self, rank: int) -> slice:
            """Rank-local slice of the main-attention query feature axis."""
            width = self.heads_per_rank * self.qk_head_dim
            return slice(rank * width, (rank + 1) * width)

        def _assert_indexer_shapes(
            self, q_idx: torch.Tensor, k_pooled_heads: int
        ) -> None:
            if q_idx.shape[-2] != k_pooled_heads:
                raise ValueError(
                    "GLM-5.3-Flash indexer head-count disagreement: q_proj "
                    f"produced {q_idx.shape[-2]} index heads but the "
                    f"IndexPool={self.index_kpool} collapse of a "
                    f"{self.index_n_heads}-head index-K cache produced "
                    f"{k_pooled_heads}. The HF checkpoint's indexer layout "
                    "does not match the pooled-head reading assumed here; fix "
                    "the loader rather than reshaping to make it fit."
                )

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            query: torch.Tensor,
            kv_cache_k: torch.Tensor,
            kv_cache_v: torch.Tensor,
            key_lengths: torch.Tensor,
            *,
            return_lse: bool = False,
        ):
            """Round-3 bound DSA lightning-indexer + sparse attention.

            Kernel identity: ``DSA_KERNEL_SLUG_V0``.  Scoring, masking, top-k,
            sparse gather and the sparse softmax all come from the torch
            golden ``dsa_lightning_indexer.py`` (IndexPool=4, natural-log LSE
            convention, ``-inf`` sentinel on fully-masked rows).  The only
            piece not delegated is the query-side projection, which has to be
            split around a TP reduce — see ``dsa_scores_from_qidx``.

            No fallback: the sparse path never degrades to dense attention.
            The one shape-driven adaptation is clamping ``index_topk`` to the
            context length, which is a degenerate top-k, not a substitution.
            """
            batch, length, _ = hidden_states.shape

            # Index-K side: replicated (gather_output=True), so this is the
            # full [B, L, index_n_heads, index_head_dim] cache on every rank.
            index_k = self.k_proj(hidden_states).view(
                batch, length, self.index_n_heads, self.index_head_dim
            )

            # Query side: rank-local contraction, then sum across ranks.
            q_flat = query.reshape(batch, query.shape[1], -1)
            q_idx = torch.einsum(
                "bqf,hfd->bqhd",
                q_flat.to(torch.float32),
                self.q_proj.to(torch.float32),
            )
            q_idx = _reduce_from_tp_region(q_idx)
            self._assert_indexer_shapes(q_idx, self.pooled_index_heads)

            scores = dsa_scores_from_qidx(
                q_idx,
                index_k,
                index_pool=self.index_kpool,
                pool_weights=self.pool_weights,
            )
            return dsa_attend_from_scores(
                scores,
                query,
                kv_cache_k,
                kv_cache_v,
                position_ids,
                key_lengths,
                topk=self.index_topk,
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

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
            key_lengths: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Round-3 bound DSA layer: All-NoPE MLA + indexer-selected sparse attn.

            When ``kv_cache`` is supplied the sparse gather runs against the
            full cached context; otherwise K/V come from the current window,
            which is the correct prefill/self-attention behaviour and the
            shape the 1-layer smoke exercises.
            """
            query, key, value = self.mla.project(hidden_states)
            if kv_cache is not None:
                key, value = kv_cache
            batch = hidden_states.shape[0]
            context_len = key.shape[1]
            if key_lengths is None:
                # No attention_mask supplied — every position is valid.
                key_lengths = torch.full(
                    (batch,),
                    context_len,
                    dtype=torch.int64,
                    device=hidden_states.device,
                )
            else:
                key_lengths = key_lengths.to(torch.int64).clamp(max=context_len)
            attn, _topk = self.indexer(
                hidden_states,
                position_ids,
                query,
                key,
                value,
                key_lengths,
            )
            batch, length = attn.shape[0], attn.shape[1]
            return self.mla.o_proj(
                attn.reshape(batch, length, -1).to(hidden_states.dtype)
            )

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

        def _local_heads(self) -> slice:
            """Rank-local head slice for the checkpoint-width small params."""
            rank = _tp_rank()
            return slice(
                rank * self.heads_per_rank, (rank + 1) * self.heads_per_rank
            )

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
            kda_state: torch.Tensor | None = None,
            conv_state: torch.Tensor | None = None,
        ) -> torch.Tensor:
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

            query = self.q_proj(hidden_states).view(
                batch, length, heads, self.head_dim
            )
            key = self.k_proj(hidden_states).view(
                batch, length, heads, self.head_dim
            )
            value = self.v_proj(hidden_states).view(
                batch, length, heads, self.head_dim
            )

            if conv_state is None:
                conv_state = torch.zeros(
                    (batch,) + self.conv_state_shape,
                    dtype=self.state_dtype,
                    device=hidden_states.device,
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
            # Reassigning `self.kda_state` / `self.conv_state` inside forward
            # is exactly the pattern torch_neuronx refuses to lower:
            #   "Unable to lower HLO: parameter not found in lowering context.
            #    This is likely caused by an attempted in-place operation, or
            #    an attempted access of nn.Parameter.data or nn.Buffer.data."
            # The recurrence therefore leaves its updated state on plain
            # Python attributes, which the tracer ignores, and the caller
            # reads them to thread state between graph invocations.
            #
            # NOTE (next blocker, tracked in the Round-3 status doc): making
            # the state survive across *decode steps* on device needs NxDI's
            # `input_output_aliases` state wireup, so the state tensor becomes
            # a real graph input/output pair rather than a Python attribute.
            # Until that lands, a CTE (prefill-from-zero) graph is exact and a
            # multi-step TKG graph would silently restart from zero state —
            # which is why the TKG contract must not be declared correct until
            # the aliasing is wired.
            self._new_kda_state = new_state
            self._new_conv_state = new_conv_state

            # Gated output RMSNorm over head_dim, then row-parallel out-proj.
            output = output.to(hidden_states.dtype)
            gate = self.g_b_proj(self.g_a_proj(hidden_states)).view_as(output)
            normed = _rms_norm(output, self.o_norm_weight, self.rms_eps)
            normed = normed * torch.sigmoid(gate.to(torch.float32)).to(
                normed.dtype
            )
            return self.o_proj(normed.flatten(-2))

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

        Routed-expert weights are fused across the expert axis:
          gate: [n_routed_experts, hidden, moe_intermediate_size]
          up:   [n_routed_experts, hidden, moe_intermediate_size]
          down: [n_routed_experts, moe_intermediate_size, hidden]
        Round-2 keeps these as replicated ``nn.Parameter`` tensors so the
        blockwise-MoE loader can address them by expert index.  Round-3 will
        introduce an expert-axis sharding via ExpertParallelism inside NxDI
        once the container ships ``_call_shard_hidden_kernel`` unconditionally.

        Router gate is a small ``nn.Linear`` (replicated); routed dispatch is
        deferred to Round 3 (blockwise MoE NKI kernel).  Shared expert MLP is
        lowered here so the shared-branch compile-graph binds now.
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
            self.gate = nn.Parameter(
                torch.empty(
                    self.n_routed_experts,
                    self.hidden_size,
                    self.moe_intermediate_per_tp,
                    dtype=dtype,
                ),
                requires_grad=False,
            )
            self.up = nn.Parameter(
                torch.empty(
                    self.n_routed_experts,
                    self.hidden_size,
                    self.moe_intermediate_per_tp,
                    dtype=dtype,
                ),
                requires_grad=False,
            )
            self.down = nn.Parameter(
                torch.empty(
                    self.n_routed_experts,
                    self.moe_intermediate_per_tp,
                    self.hidden_size,
                    dtype=dtype,
                ),
                requires_grad=False,
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
            """Round-3 bound MoE: GLM routing + top-k gather dispatch + shared.

            Kernel identity: ``MOE_KERNEL_SLUG_V1``.  The routed branch is
            O(top_k) FLOPs, not O(num_experts) — it gathers only the 8 selected
            expert slabs per token, matching the fused NKI kernel's asymptotics
            without its capacity-dispatch machinery.  ``self.dispatch_config``
            carries the validated shape identity the fused kernel compiles to.

            No fallback: there is no branch that silently drops to
            ``torch_blockwise_matmul_inference`` or skips the routed half.
            """
            shared = self.shared_expert(hidden_states)
            expert_indices, routing_weights = glm53_route(
                hidden_states,
                self.router.weight,
                top_k=self.num_experts_per_tok,
                scoring_func=self.scoring_func,
                norm_topk_prob=self.norm_topk_prob,
                routed_scaling_factor=self.routed_scaling_factor,
                correction_bias=self.e_score_correction_bias,
            )
            routed = moe_gather_dispatch_torch(
                hidden_states,
                expert_indices,
                routing_weights,
                self.gate,
                self.up,
                self.down,
                swiglu_limit=self.swiglu_limit,
            )
            # `down` is sharded on the intermediate axis, so each rank produced
            # a partial sum over its slice — the same RowParallelLinear
            # contract the shared expert's `down_proj` gets for free.  Without
            # this reduce every rank would emit 1/tp of the routed activation.
            routed = _reduce_from_tp_region(routed)
            return shared + routed.to(shared.dtype)

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
                torch.empty(
                    mix_rows, self.hc_mult * self.hidden_size, dtype=dtype
                ),
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
                mixes[:, : self.hc_mult] * self.scale[0]
                + self.base[: self.hc_mult]
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
                comb_mix = comb_mix / (
                    comb_mix.sum(dim=-1, keepdim=True) + self.hc_eps
                )
                comb_mix = comb_mix / (
                    comb_mix.sum(dim=-2, keepdim=True) + self.hc_eps
                )
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

        def forward(
            self,
            residual_streams: torch.Tensor,
            position_ids: torch.Tensor,
            key_lengths: torch.Tensor | None = None,
        ) -> torch.Tensor:
            post_mix, comb_mix, hidden_states = self.hc_attn.pre(residual_streams)
            normalized = _rms_norm(
                hidden_states, self.input_norm_weight, self.rms_eps
            )
            if self.attn_kind == "dsa":
                attn_out = self.self_attn(
                    normalized, position_ids, key_lengths=key_lengths
                )
            else:
                attn_out = self.self_attn(normalized)
            residual_streams = self.hc_attn.post(
                attn_out, residual_streams, post_mix, comb_mix
            )
            post_mix, comb_mix, hidden_states = self.hc_mlp.pre(residual_streams)
            normalized = _rms_norm(
                hidden_states, self.post_attention_norm_weight, self.rms_eps
            )
            mlp_out = self.mlp(normalized)
            return self.hc_mlp.post(mlp_out, residual_streams, post_mix, comb_mix)

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
                positions = torch.arange(
                    length, dtype=torch.int64, device=hidden_states.device
                ).unsqueeze(0).expand(batch, -1)
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
            residual_streams = hidden_states.unsqueeze(-2).repeat(
                1, 1, self.hc_mult, 1
            )
            for layer in self.layers:
                residual_streams = layer(
                    residual_streams, positions, key_lengths=key_lengths
                )
            hidden_states = residual_streams.mean(dim=-2)
            hidden_states = _rms_norm(
                hidden_states, self.final_norm_weight, self.rms_eps
            )
            logits = self.lm_head(hidden_states)

            # KV-cache alias contract.
            #
            # `DecoderModelInstance.get()` (model_wrapper.py:1614-1619) builds
            # `input_output_aliases` from `kv_mgr.past_key_values` — real
            # nn.Parameters — mapping each to output index
            # `num_output_from_trace + i`.  Unlike the example inputs, that
            # alias list is NOT filtered for -1 before `linearize_indices`
            # (hlo_conversion.py:490-496), so a cache parameter that the graph
            # aliases but never reads aborts lowering with
            # "parameter not found in lowering context".
            #
            # Every NxDI model in-tree avoids this by not overriding `forward`
            # at all — the base `NeuronBaseModel.forward` reads the cache via
            # `kv_mgr.get_cache` and returns `outputs += updated_kv_cache`.
            # This graph keeps its own forward (GLM-5.3 is a hybrid KDA/DSA
            # stack that the base decode loop does not model), so it must
            # honour the same contract explicitly: read each cache parameter
            # and return it directly after the logits, in alias order.
            caches = _aliased_kv_parameters(self)
            if caches:
                return [logits] + list(caches)
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
            raise NotImplementedError(
                "GLM-5.3-Flash HF direct load is deferred to Round 3 "
                "alongside device-attached NKI kernels; the compile-driver "
                "smoke uses `initialize_model_weights=False`."
            )

        @staticmethod
        def convert_hf_to_neuron_state_dict(
            state_dict: dict, config: Any
        ) -> dict:
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
            return _convert_glm53_checkpoint(
                state_dict, src, tp_degree=tp_degree
            )

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
                self._cpu_oracle = NeuronGlm53FlashForCausalLMImpl(
                    self._source_config
                )
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

        def _assert_indexer_multipliers_bounded(
            self, compiled_model_path: str
        ) -> None:
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
        ) -> "Glm53FlashNeuronInferenceConfig":
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
        ) -> "Glm53FlashNeuronInferenceConfig":
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
                name: getattr(reduced, name)
                for name in reduced.__dataclass_fields__
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

        @classmethod
        def build_kernel_coverage_smoke_config(
            cls,
            source_config: Glm53FlashInferenceConfig,
            *,
            tp_degree: int = 8,
            ctx_batch_size: int = 1,
            tkg_batch_size: int = 1,
            seq_len: int = 128,
        ) -> "Glm53FlashNeuronInferenceConfig":
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
                name: getattr(reduced, name)
                for name in reduced.__dataclass_fields__
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
    "GLM53_BLOCKWISE_MATMUL_WORKAROUND",
    "Glm53FlashNeuronInferenceConfig",
    "NeuronGlm53FlashForCausalLM",
    "build_neuron_config",
]
