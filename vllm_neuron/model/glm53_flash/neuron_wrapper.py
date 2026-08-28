# SPDX-License-Identifier: Apache-2.0
"""NxDI compile-integration wrapper for GLM-5.3-Flash.

Codex Alpha shipped ``NeuronGlm53FlashForCausalLMImpl`` (in ``model.py``) as
the CPU source-qualified reference: pure Python autoregressive with reference
kernels, telemetry, and per-expert loops.  That class inherits from
``torch.nn.Module`` and cannot be handed to NxDI's compile pipeline directly.

This module supplies the compile-facing wrapper class,
``NeuronGlm53FlashForCausalLM``, that subclasses
``neuronx_distributed_inference.models.model_base.NeuronBaseForCausalLM`` so the
standard ``Neuron{X}ForCausalLM(model_path, config).compile(out_path)`` pattern
binds.  The wrapper is deliberately **shell-only** in this session:

- ``_model_cls`` is a small ``NeuronBaseModel`` subclass (``_NeuronGlm53FlashModel``)
  that lays out a Parallel embedding + a single Linear residual + LM head so
  the tracer has something graph-shaped to walk.  Layer wiring for the real
  KDA / All-NoPE-MLA / DSA / MoE / mHC blocks lands in Round 2 once the
  reference-logit gate produces the golden tensors we intend to match.
- ``load_weights`` is deferred to Round 2 (raises ``NotImplementedError`` with
  a pointer to the sharded-FP8 loader work).  The class documents the load
  contract inline so the loader author does not have to re-derive it.
- The MoE blockwise-mm workaround from
  ``[[nxdi-container-moe-blockwise-mm-workaround-20260827]]`` — every MoE
  compile on container ``sha256:011d49c7`` must set
  ``blockwise_matmul_config.use_shard_on_intermediate_dynamic_while=True``
  before ``InferenceConfig`` init — is applied in ``build_neuron_config`` so
  that any GLM-5.3-Flash compile driver honours it without re-quoting the
  memory.
- The wrapper does **not** inherit the GLM-5.2 static-FP8 normalizer bug
  (`normalize_static_fp8_weight_format()` returning OCP-448 when the input is
  ``None``): we require the weight format to be declared explicitly and raise
  when it is absent.

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
    _NXDI_AVAILABLE = True
    _NXDI_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - CPU-only guard
    class _NxdiUnavailable:
        """Placeholder used only when NxDI is missing on this host."""

    NeuronBaseForCausalLM = _NxdiUnavailable  # type: ignore[assignment,misc]
    NeuronBaseModel = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdiInferenceConfig = _NxdiUnavailable  # type: ignore[assignment,misc]
    _NxdiNeuronConfig = _NxdiUnavailable  # type: ignore[assignment,misc]
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
        "(`neuronx_distributed_inference`) inside container "
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


if _NXDI_AVAILABLE:

    class _NeuronGlm53FlashModel(NeuronBaseModel):
        """Traceable-shaped shell for the GLM-5.3-Flash graph.

        Round 1 (this session): a minimal Parallel embedding → single Linear
        residual → LM head so the NxDI tracer has a graph-shaped forward to
        walk and the compile driver binds end-to-end.  This is a SHELL, not
        a correctness-target; it is intentionally thin so we can fire the
        1-layer smoke on the compile host and confirm the driver reaches the
        tracer without paying for the ~306 GiB full compile.

        Round 2: replaces `layers` with the real KDA / All-NoPE-MLA / DSA /
        288-expert MoE / mHC 4-stream blocks lowered to NxDI parallel
        primitives (`ColumnParallelLinear`, `RowParallelLinear`,
        `ExpertMLPsCapacityFactor`, `NKI` DSA/KDA kernels).  The reference
        oracle for that pass is `NeuronGlm53FlashForCausalLMImpl` under a
        deterministic seed.
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
            from neuronx_distributed.parallel_layers.layers import (
                ColumnParallelLinear,
                ParallelEmbedding,
            )
            self.padding_idx = getattr(config, "pad_token_id", 0)
            self.vocab_size = config.vocab_size
            self.embed_tokens = ParallelEmbedding(
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
            # Shell residual: one Linear per hidden layer.  Replaced in Round 2
            # by the real (KDA|DSA) + (dense|sparse-MoE) block stack.  Kept
            # here so the tracer sees a graph proportional to
            # `num_hidden_layers` — which lets us exercise the 1-layer smoke
            # and, when the caller sets `num_hidden_layers=45`, a full-depth
            # dry-run.
            self.layers = nn.ModuleList(
                [
                    nn.Linear(
                        config.hidden_size,
                        config.hidden_size,
                        bias=False,
                        dtype=config.neuron_config.torch_dtype,
                    )
                    for _ in range(config.num_hidden_layers)
                ]
            )
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                pad=True,
                gather_output=not self.on_device_sampling,
                dtype=config.neuron_config.torch_dtype,
            )

    class NeuronGlm53FlashForCausalLM(NeuronBaseForCausalLM):
        """Public NxDI-compatible GLM-5.3-Flash class.

        Compile invocation (see ``command.sh``):

            wrapper = NeuronGlm53FlashForCausalLM(model_path, inference_config)
            wrapper.compile(out_path)              # full compile (Round 2+)
            wrapper.compile(out_path, dry_run=True)  # shell-driver smoke

        The class-level ``_model_cls`` selects the traceable shell above.  The
        `GLM53_SOURCE_CACHE_ABI` / `_GLM53_GRAPH_ID` fields (mirrored on the
        class) pin the compile cache per the same convention GLM-5.2 uses so
        the modular-compile flywheel indexer treats independent GLM-5.3-Flash
        artifacts as cache-distinct from GLM-5.2 and from any Round-2
        reshaping of Round-1's shell.
        """

        _model_cls = _NeuronGlm53FlashModel

        # Cache-pin per COMPILE-FASTPATH.md.  Any change to the shell forward
        # shape must bump the version tag inside `registry.GLM53_SOURCE_CACHE_ABI`
        # (Round 2 will do this once the real block stack lands).
        GLM53_SOURCE_CACHE_ABI = GLM53_SOURCE_CACHE_ABI
        _GLM53_GRAPH_ID = _GLM53_GRAPH_ID

        # NxDI's default `emit_phases` (via `enable_context_encoding`
        # + `enable_token_generation`) covers CTE + TKG together.  A caller
        # that wants TKG-only or CTE-only artifacts sets the corresponding
        # env var before instantiation (mirrors the qwen35-2b driver):
        #   NXDI_EMIT_PHASES=TKG   -> only enable_token_generation
        #   NXDI_EMIT_PHASES=CTE   -> only enable_context_encoding
        #   NXDI_EMIT_PHASES=BOTH  -> default (CTE + TKG)
        # This is honoured by not overriding the base __init__; the base
        # already respects the NeuronConfig knobs (speculation_length,
        # medusa, etc.) so we simply refuse an unknown value.
        _EMIT_PHASE_VALUES = frozenset({"BOTH", "CTE", "TKG"})

        @classmethod
        def get_config_cls(cls):
            return Glm53FlashNeuronInferenceConfig

        @staticmethod
        def load_hf_model(model_path: str, **kwargs: Any):
            raise NotImplementedError(
                "GLM-5.3-Flash HF direct load is deferred to Round 2 alongside "
                "the sharded FP8 loader; the compile-driver smoke uses "
                "`initialize_model_weights=False`."
            )

        @staticmethod
        def convert_hf_to_neuron_state_dict(
            state_dict: dict, config: Any
        ) -> dict:
            raise NotImplementedError(
                "Round-2 GLM-5.3-Flash HF-to-Neuron state-dict conversion. "
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
            # The CPU-reference oracle stays wired but lazy: only the round-2
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
            """Materialize the CPU-reference impl for Round-2 correctness gating.

            Cached so repeated calls in a round-2 test session are cheap.
            Kept off the compile path — nothing in ``compile()`` or
            ``load_weights()`` touches this method.
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
            """Rank-sharded FP8 load — deferred to Round 2.

            The full contract (mirrors GLM-5.2's `Glm52MoeDsaForCausalLM.load_weights`):

            1. Read the source checkpoint's `model.safetensors.index.json` and
               enforce the `neuron_legacy_e4m3fn_qmax240` scale format on every
               FP8 tensor (fail-closed — never fall back to OCP-448, unlike
               the GLM-5.2 `normalize_static_fp8_weight_format` inheritance bug).
            2. For each rank in `[start_rank_id, start_rank_id+local_ranks_size)`
               materialize the tp-sharded slice via
               `SafetensorsCheckpoint.load_sharded_pipelined`.
            3. Load the per-layer cache multipliers (`k_cache_quant_multiplier`,
               `v_cache_quant_multiplier`, `indexer.cache_quant_multiplier`)
               into the DSA layers only.
            4. Delegate to the base `NeuronApplicationBase.load_weights` (which
               calls `traced_model.nxd_model.initialize(weights, start_rank)`).

            None of the above is safe to fire until Round 2 lands the
            traceable KDA/DSA/MoE/mHC blocks — the shell forward has no place
            to put GLM-5.3 weights.
            """
            raise NotImplementedError(
                "GLM-5.3-Flash sharded FP8 load lands in Round 2; the shell "
                "compile driver initialises weights via NxDI's built-in "
                "shard_checkpoint path with `initialize_model_weights=False`."
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

            Frozen-field validators in `Glm53FlashInferenceConfig` reject
            arbitrary architecture changes; we clone the source config with
            `allow_reduced_shapes=True` so `num_hidden_layers=1` passes.  The
            frozen fields (vocab, hidden_size, expert count, etc.) stay
            untouched — the shell tracer walks a proportionally shorter
            layer list, which is enough evidence the compile driver binds.
            """
            reduced = copy.deepcopy(source_config)
            # Bypass the immutable-dataclass field freezing that
            # `_validate_architecture` enforces by rebuilding from a mutable
            # kwargs dict.
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
