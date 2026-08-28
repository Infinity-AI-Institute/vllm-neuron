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
    _NXDI_AVAILABLE = False
    _NXDI_IMPORT_ERROR = exc


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

    class _MoEBlock(nn.Module):
        """Top-k routed MoE with sqrt(softplus(x)) scoring.

        Fork of ``glm53_flash/neuron_wrapper.py::_MoEBlock`` (which uses
        sigmoid).  Constants swap: 288 -> 256 experts, top-8 -> top-6,
        drop the group-limited routing args (V4 dropped ``n_group`` /
        ``topk_group``), keep ``noaux_tc`` + ``e_score_correction_bias``.
        """

        def __init__(self, config: Any, *, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self._config = config
            raise NotImplementedError(
                "_MoEBlock is Round 2.  Single-line scoring swap vs GLM-5.3-Flash "
                "_MoEBlock: `sigmoid(x)` -> `torch.sqrt(F.softplus(x))`; expert "
                "count 288->256, top-k 8->6."
            )

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
    "DeepseekV4FlashNeuronInferenceConfig",
    "FORBIDDEN_FP8_KV_KEYS",
    "NeuronDeepseekV4FlashForCausalLM",
    "build_neuron_config",
]
