# SPDX-License-Identifier: Apache-2.0
"""Fail-closed bridge from the reviewed runtime profile to the NxDI wrapper."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .runtime_config import Glm53RuntimeConfig
from .runtime_factory import GLM53_RUNTIME_ADAPTER

GLM53_COMPILE_ADAPTER_SCHEMA = "glm53-nxdi-compile-adapter-v1"
_AUTO_CAST_NONE = "--auto-cast=none"
_JOINT_MODEL_TAGS = ("context_encoding_model", "token_generation_model")


class Glm53CompileAdapterError(ValueError):
    """The reviewed profile cannot be expressed by the GLM-5.3 wrapper."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Glm53CompileAdapterError(message)


def compile_kwargs(profile: Glm53RuntimeConfig) -> dict[str, Any]:
    """Translate every shape-affecting reviewed field into NxDI kwargs."""
    _require(profile.runtime_adapter == GLM53_RUNTIME_ADAPTER, "adapter identity drift")
    _require(
        profile.tensor_parallel_degree in (32, 64),
        "GLM-5.3 requires TP32 or TP64",
    )
    _require(profile.logical_neuron_cores == 2, "GLM-5.3 requires LNC2")
    _require(profile.weight_dtype == "bfloat16", "rank weights must be BF16")
    _require(profile.cache_dtype == "bfloat16", "cache must be BF16")
    _require(
        profile.runtime_quantization == "none",
        "runtime quantization must be none",
    )
    _require(profile.sampling_mode == "greedy", "formal gate requires greedy sampling")
    _require(
        profile.output_logits is True,
        "full-vocabulary correctness gate requires output_logits=true",
    )
    _require(not profile.speculative_decode, "speculative decoding is forbidden")
    _require(
        _AUTO_CAST_NONE in profile.compiler_flags,
        "compiler flags must contain --auto-cast=none",
    )
    _require(
        not any(
            flag.startswith("--auto-cast=") and flag != _AUTO_CAST_NONE
            for flag in profile.compiler_flags
        ),
        "conflicting compiler auto-cast flag",
    )
    return {
        "tp_degree": profile.tensor_parallel_degree,
        "ctx_batch_size": profile.batch_size,
        "tkg_batch_size": profile.batch_size,
        "seq_len": profile.max_sequence_length,
        "context_encoding_buckets": list(profile.context_encoding_buckets),
        "token_generation_buckets": list(profile.token_generation_buckets),
        "max_context_length": max(profile.context_encoding_buckets),
        "logical_nc_config": profile.logical_neuron_cores,
        "output_logits": profile.output_logits,
        "skip_sharding": True,
        "save_sharded_checkpoint": True,
    }


def assert_emitted_neuron_config(
    profile: Glm53RuntimeConfig, emitted: Mapping[str, Any]
) -> None:
    """Verify the serialized NxDI config before accepting a compiled artifact."""
    expected = compile_kwargs(profile)
    for key, value in expected.items():
        _require(key in emitted, f"emitted NeuronConfig missing {key}")
        _require(emitted[key] == value, f"emitted NeuronConfig drifted {key}")
    dtype = str(emitted.get("torch_dtype", "")).removeprefix("torch.")
    _require(dtype == "bfloat16", "emitted torch_dtype must be bfloat16")
    blockwise = emitted.get("blockwise_matmul_config")
    _require(isinstance(blockwise, Mapping), "emitted blockwise config missing")
    _require(
        blockwise.get("use_shard_on_intermediate_dynamic_while") is True,
        "required blockwise intermediate-shard path was dropped",
    )
    _require(
        blockwise.get("skip_dma_token") is True,
        "required DMA-token gate was dropped",
    )
    for forbidden in ("fp8_packed_kv", "kv_cache_quant", "kv_quant_config"):
        _require(forbidden not in emitted, f"forbidden emitted field: {forbidden}")


def _assert_joint_tp64_profile(profile: Glm53RuntimeConfig) -> None:
    """Require the one reviewed GLM TP64 BOTH compile shape."""
    _require(profile.tensor_parallel_degree == 64, "joint BOTH compile requires TP64")
    _require(profile.logical_neuron_cores == 2, "joint BOTH compile requires LNC2")
    _require(profile.batch_size == 1, "joint BOTH compile requires batch size one")
    _require(
        profile.max_sequence_length == 2560,
        "joint BOTH compile requires S2560 state capacity",
    )
    _require(
        profile.context_encoding_buckets == (2048,),
        "joint BOTH compile requires only CTE2048",
    )
    _require(
        profile.token_generation_buckets == (2560,),
        "joint BOTH compile requires only TKG position bucket 2560",
    )


def assert_joint_both_application(
    profile: Glm53RuntimeConfig, application: Any
) -> None:
    """Validate the pre-trace application shape without claiming artifact state.

    The pinned NxDI base class creates both phase wrappers on one application;
    its single ``get_builder`` call later adds ``application.models`` to one
    ``ModelBuilder``.  The compiled artifact inspector remains responsible for
    proving that the resulting ``model.pt`` owns one ``NxDModel.state``.
    """
    _assert_joint_tp64_profile(profile)
    _require(
        getattr(application, "_emit_phases", None) == "BOTH",
        "application did not retain BOTH phase selection",
    )
    models = getattr(application, "models", None)
    _require(isinstance(models, list), "application models must be an ordered list")
    tags = tuple(getattr(model, "tag", None) for model in models)
    _require(tags == _JOINT_MODEL_TAGS, f"joint model tags/order drift: {tags!r}")
    _require(
        getattr(application, "context_encoding_model", None) is models[0],
        "CTE attribute is not the first builder model",
    )
    _require(
        getattr(application, "token_generation_model", None) is models[1],
        "TKG attribute is not the second builder model",
    )
    _require(
        getattr(application, "_builder", None) is None,
        "builder was created before joint application admission",
    )

    cte_config = getattr(models[0], "neuron_config", None)
    tkg_config = getattr(models[1], "neuron_config", None)
    _require(cte_config is not None and tkg_config is not None, "phase config missing")
    _require(cte_config.is_prefill_stage is True, "CTE prefill marker drift")
    _require(cte_config.n_active_tokens == 2048, "CTE active-token count drift")
    _require(cte_config.buckets == [2048], "CTE bucket drift")
    _require(tkg_config.is_prefill_stage is False, "TKG prefill marker drift")
    _require(tkg_config.n_active_tokens == 1, "TKG active-token count drift")
    _require(tkg_config.buckets == [[1, 2560]], "TKG position bucket drift")
    for phase, config in (("CTE", cte_config), ("TKG", tkg_config)):
        _require(config.seq_len == 2560, f"{phase} state capacity drift")
        _require(config.tp_degree == 64, f"{phase} TP drift")
        _require(config.logical_nc_config == 2, f"{phase} LNC drift")


def construct_joint_both_application(
    profile: Glm53RuntimeConfig,
    *,
    model_path: str,
    source_config: Any,
    _wrapper_cls: Any | None = None,
) -> Any:
    """Construct exactly one untraced TP64 application containing CTE and TKG.

    This function does not call ``compile``, ``get_builder``, ``load``, or any
    runtime/device API.  ``NXDI_EMIT_PHASES`` must be explicitly pinned rather
    than relying on the wrapper's default.
    """
    _require(
        os.environ.get("NXDI_EMIT_PHASES") == "BOTH",
        "NXDI_EMIT_PHASES must be explicitly BOTH",
    )
    _assert_joint_tp64_profile(profile)
    kwargs = compile_kwargs(profile)
    if _wrapper_cls is None:
        from .neuron_wrapper import NeuronGlm53FlashForCausalLM

        _wrapper_cls = NeuronGlm53FlashForCausalLM
    inference_config = _wrapper_cls.build_inference_config(
        source_config,
        tp_degree=kwargs.pop("tp_degree"),
        ctx_batch_size=kwargs.pop("ctx_batch_size"),
        tkg_batch_size=kwargs.pop("tkg_batch_size"),
        seq_len=kwargs.pop("seq_len"),
        **kwargs,
    )
    application = _wrapper_cls(model_path, inference_config)
    assert_joint_both_application(profile, application)
    return application


__all__ = [
    "GLM53_COMPILE_ADAPTER_SCHEMA",
    "Glm53CompileAdapterError",
    "assert_emitted_neuron_config",
    "assert_joint_both_application",
    "compile_kwargs",
    "construct_joint_both_application",
]
