# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy

import pytest

from vllm_neuron.model.glm53_flash.compile_adapter import (
    Glm53CompileAdapterError,
    assert_emitted_neuron_config,
    compile_kwargs,
)
from vllm_neuron.model.glm53_flash.runtime_config import (
    GLM53_ARCHITECTURE,
    GLM53_CHECKPOINT_REVISION,
    GLM53_RUNTIME_CONFIG_SCHEMA,
    Glm53RuntimeConfig,
)
from vllm_neuron.model.glm53_flash.runtime_factory import GLM53_RUNTIME_ADAPTER


def _profile(**updates) -> Glm53RuntimeConfig:
    value = {
        "schema": GLM53_RUNTIME_CONFIG_SCHEMA,
        "architecture": GLM53_ARCHITECTURE,
        "checkpoint_revision": GLM53_CHECKPOINT_REVISION,
        "source_commit": "1" * 40,
        "source_tree": "2" * 40,
        "runtime_adapter": GLM53_RUNTIME_ADAPTER,
        "compiler_image_id": "sha256:" + "3" * 64,
        "compiler_image_digest": "example.invalid/neuron@sha256:" + "4" * 64,
        "compiler_version": "pinned",
        "runtime_packages": {"vllm-neuron": "1" * 40},
        "compiler_flags": ["--auto-cast=none"],
        "tensor_parallel_degree": 32,
        "logical_neuron_cores": 2,
        "batch_size": 1,
        "max_sequence_length": 8192,
        "context_encoding_buckets": [128, 512, 8192],
        "token_generation_buckets": [128, 512, 8192],
        "weight_dtype": "bfloat16",
        "cache_dtype": "bfloat16",
        "runtime_quantization": "none",
        "sampling_mode": "greedy",
        "speculative_decode": False,
    }
    value.update(updates)
    return Glm53RuntimeConfig.from_mapping(value)


def _emitted(profile: Glm53RuntimeConfig) -> dict:
    value = compile_kwargs(profile)
    value["torch_dtype"] = "torch.bfloat16"
    value["blockwise_matmul_config"] = {
        "use_shard_on_intermediate_dynamic_while": True,
        "skip_dma_token": True,
    }
    return value


def test_exact_profile_maps_every_shape_field_and_passes_emitted_gate():
    profile = _profile()
    expected = compile_kwargs(profile)
    assert expected == {
        "tp_degree": 32,
        "ctx_batch_size": 1,
        "tkg_batch_size": 1,
        "seq_len": 8192,
        "context_encoding_buckets": [128, 512, 8192],
        "token_generation_buckets": [128, 512, 8192],
        "max_context_length": 8192,
        "logical_nc_config": 2,
        "skip_sharding": True,
        "save_sharded_checkpoint": True,
    }
    assert_emitted_neuron_config(profile, _emitted(profile))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runtime_adapter", "wrong", "adapter identity"),
        ("logical_neuron_cores", 1, "LNC2"),
        ("weight_dtype", "float8_e4m3fn", "weights must be BF16"),
        ("cache_dtype", "float8_e4m3fn", "cache must be BF16"),
        ("runtime_quantization", "fp8", "quantization must be none"),
        ("compiler_flags", ["--auto-cast=matmult"], "--auto-cast=none"),
        (
            "compiler_flags",
            ["--auto-cast=none", "--auto-cast=matmult"],
            "conflicting compiler auto-cast",
        ),
    ],
)
def test_unexpressible_profile_fails_before_compile(field, value, message):
    with pytest.raises(Glm53CompileAdapterError, match=message):
        compile_kwargs(_profile(**{field: value}))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("logical_nc_config"),
        lambda value: value.__setitem__("logical_nc_config", 1),
        lambda value: value.__setitem__("torch_dtype", "float8_e4m3fn"),
        lambda value: value["blockwise_matmul_config"].__setitem__(
            "use_shard_on_intermediate_dynamic_while", False
        ),
        lambda value: value["blockwise_matmul_config"].pop("skip_dma_token"),
        lambda value: value.__setitem__("kv_cache_quant", "fp8"),
    ],
)
def test_emitted_config_drift_fails_closed(mutation):
    profile = _profile()
    emitted = copy.deepcopy(_emitted(profile))
    mutation(emitted)
    with pytest.raises(Glm53CompileAdapterError):
        assert_emitted_neuron_config(profile, emitted)
