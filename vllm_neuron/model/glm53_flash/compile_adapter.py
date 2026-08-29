# SPDX-License-Identifier: Apache-2.0
"""Fail-closed bridge from the reviewed runtime profile to the NxDI wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .runtime_config import Glm53RuntimeConfig
from .runtime_factory import GLM53_RUNTIME_ADAPTER

GLM53_COMPILE_ADAPTER_SCHEMA = "glm53-nxdi-compile-adapter-v1"
_AUTO_CAST_NONE = "--auto-cast=none"


class Glm53CompileAdapterError(ValueError):
    """The reviewed profile cannot be expressed by the GLM-5.3 wrapper."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Glm53CompileAdapterError(message)


def compile_kwargs(profile: Glm53RuntimeConfig) -> dict[str, Any]:
    """Translate every shape-affecting reviewed field into NxDI kwargs."""
    _require(profile.runtime_adapter == GLM53_RUNTIME_ADAPTER, "adapter identity drift")
    _require(profile.tensor_parallel_degree == 32, "GLM-5.3 requires TP32")
    _require(profile.logical_neuron_cores == 2, "GLM-5.3 requires LNC2")
    _require(profile.weight_dtype == "bfloat16", "rank weights must be BF16")
    _require(profile.cache_dtype == "bfloat16", "cache must be BF16")
    _require(
        profile.runtime_quantization == "none",
        "runtime quantization must be none",
    )
    _require(profile.sampling_mode == "greedy", "formal gate requires greedy sampling")
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


__all__ = [
    "GLM53_COMPILE_ADAPTER_SCHEMA",
    "Glm53CompileAdapterError",
    "assert_emitted_neuron_config",
    "compile_kwargs",
]
