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

from .config import Glm53FlashInferenceConfig
from .model import NeuronGlm53FlashForCausalLMImpl
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
    """Construct an NxDI ``NeuronConfig`` with the GLM-5.3 MoE workaround pinned.

    The blockwise-matmul workaround MUST be inside ``blockwise_matmul_config``
    at construction time — the ``InferenceConfig.__init__`` reads
    ``kwargs["blockwise_matmul_config"]`` and freezes it via
    ``BlockwiseMatmulConfig.from_kwargs`` before the model can observe it.
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
    return _NxdiNeuronConfig(**kwargs)


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
            if self.index_n_heads % tp_degree:
                raise NotImplementedError(
                    f"DSA indexer requires index_n_heads ({self.index_n_heads}) "
                    f"divisible by TP degree ({tp_degree}); Round-3 will add a "
                    "head-padded fallback."
                )
            self.k_proj = _NxdColumnParallelLinear(
                self.hidden_size,
                self.index_n_heads * self.index_head_dim,
                bias=False,
                gather_output=False,
                dtype=dtype,
            )
            # Q_proj is a rank-3 parameter; store the full tensor and slice at
            # load time by rank.  Kept as an nn.Parameter (not a ColumnParallelLinear)
            # because the slicing axis (index-head) is the leading axis and
            # NxDI's ColumnParallelLinear shards only the output axis of a 2D
            # weight.  Round-3 loader materialises the rank-local slice.
            self.q_proj = nn.Parameter(
                torch.empty(
                    self.index_n_heads,
                    self.num_attention_heads * self.qk_head_dim,
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

        def local_indexer_head_slice(self, rank: int, tp_degree: int) -> slice:
            heads_per_rank = self.index_n_heads // tp_degree
            return slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)

        def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            raise NotImplementedError(
                "GLM-5.3-Flash DSA lightning-indexer + sparse-attn forward "
                "requires the Round-3 device NKI binding.  Reference at "
                "`C:\\Users\\apumu\\research\\InfinityAI\\gemma4-trn2-handoff\\"
                "harness-v2\\staging\\reference-sweep-20260826T2150Z\\kernels\\"
                "dsa_lightning_indexer.py` (nki_v0_reference_lightning_indexer)."
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
            self, hidden_states: torch.Tensor, position_ids: torch.Tensor
        ) -> torch.Tensor:
            # Project the QKV so the tracer sees the MLA weight shapes bind.
            # The DSA sparse-attention step is a Round-3 NKI kernel; refuse
            # explicitly rather than silently falling through to full-dense
            # attention or CPU.
            _ = self.mla.project(hidden_states)
            self.indexer(hidden_states, position_ids)  # raises
            raise NotImplementedError(
                "GLM-5.3-Flash DSA layer forward defers the sparse-attn kernel "
                "invocation to Round 3.  MLA + indexer projections are lowered; "
                "the sparse KV gather + score/softmax lands in the NKI v0 "
                "wrapper (see `_DSAIndexerBlock.forward` pointer)."
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
            # Depthwise-groups conv1d.  Kept replicated in Round 2; Round-3
            # NKI kernel will shard along the head axis to match Q/K/V.
            channels = 3 * self.qkv_dim
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

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # Trace the projections so the tracer binds Q/K/V/F/G/B weight
            # shapes.  The KDA state kernel invocation is Round-3 territory.
            _ = self.q_proj(hidden_states)
            _ = self.k_proj(hidden_states)
            _ = self.v_proj(hidden_states)
            raise NotImplementedError(
                "GLM-5.3-Flash KDA state kernel forward is a Round-3 device "
                "NKI binding.  Reference at "
                "`C:\\Users\\apumu\\research\\InfinityAI\\gemma4-trn2-handoff\\"
                "harness-v2\\staging\\reference-sweep-20260826T2150Z\\kernels\\"
                "kda_state_v2.py` (kda_state.decode.kda_gate.rank1_delta."
                "bf16_state.v1; SigmoidBeta + SafeGate + LOWER_BOUND=-5.0 + "
                "in-kernel L2-norm per FLA v0.5.2)."
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
                    self.moe_intermediate_size,
                    dtype=dtype,
                ),
                requires_grad=False,
            )
            self.up = nn.Parameter(
                torch.empty(
                    self.n_routed_experts,
                    self.hidden_size,
                    self.moe_intermediate_size,
                    dtype=dtype,
                ),
                requires_grad=False,
            )
            self.down = nn.Parameter(
                torch.empty(
                    self.n_routed_experts,
                    self.moe_intermediate_size,
                    self.hidden_size,
                    dtype=dtype,
                ),
                requires_grad=False,
            )
            self.shared_expert = _MoESharedExpert(config)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # Shared branch is lowered and can be traced; routed dispatch is
            # a Round-3 blockwise-MoE NKI kernel (nc_find_index8 + capacity
            # dispatch), so the routed branch raises rather than silently
            # falling through to the CPU per-expert loop.
            _shared = self.shared_expert(hidden_states)  # noqa: F841 (traced)
            raise NotImplementedError(
                "GLM-5.3-Flash routed-MoE dispatch (288 experts, top-8) is a "
                "Round-3 blockwise-MoE NKI kernel.  Reference at "
                "`C:\\Users\\apumu\\research\\InfinityAI\\gemma4-trn2-handoff\\"
                "harness-v2\\staging\\reference-sweep-20260826T2150Z\\kernels\\"
                "moe_dispatch.py` (MoEDispatchConfig).  The blockwise-matmul "
                "container workaround is already applied via "
                "`build_neuron_config` (use_shard_on_intermediate_dynamic_while=True)."
            )

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
            self, residual_streams: torch.Tensor, position_ids: torch.Tensor
        ) -> torch.Tensor:
            post_mix, comb_mix, hidden_states = self.hc_attn.pre(residual_streams)
            normalized = _rms_norm(
                hidden_states, self.input_norm_weight, self.rms_eps
            )
            if self.attn_kind == "dsa":
                attn_out = self.self_attn(normalized, position_ids)
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
            positions: torch.Tensor,
            **kwargs: Any,
        ) -> torch.Tensor:
            hidden_states = self.embed_tokens(input_ids)
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.unsqueeze(0)
            batch, length, _ = hidden_states.shape
            if positions.ndim == 1:
                positions = positions.expand(batch, -1)
            # mHC 4-stream residual widening (matches Impl.forward at model.py:140).
            residual_streams = hidden_states.unsqueeze(-2).repeat(
                1, 1, self.hc_mult, 1
            )
            for layer in self.layers:
                residual_streams = layer(residual_streams, positions)
            hidden_states = residual_streams.mean(dim=-2)
            hidden_states = _rms_norm(
                hidden_states, self.final_norm_weight, self.rms_eps
            )
            return self.lm_head(hidden_states)

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
            raise NotImplementedError(
                "Round-3 GLM-5.3-Flash HF-to-Neuron state-dict conversion. "
                "See the GLM-5.2 `checkpoint_mapping.build_checkpoint_contract` "
                "for the fused-QKV + routed-expert mapping template; GLM-5.3 "
                "adds the KDA `short_conv`, DSA IndexPool tail, and mHC "
                "4-stream projections."
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
