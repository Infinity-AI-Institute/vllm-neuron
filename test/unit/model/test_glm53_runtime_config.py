# SPDX-License-Identifier: Apache-2.0
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[3]
    / "vllm_neuron"
    / "model"
    / "glm53_flash"
    / "runtime_config.py"
)
SPEC = importlib.util.spec_from_file_location("glm53_runtime_config", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from glm53_runtime_config import (
    GLM53_ARCHITECTURE,
    GLM53_CHECKPOINT_REVISION,
    GLM53_RUNTIME_CONFIG_SCHEMA,
    Glm53RuntimeConfig,
    Glm53RuntimeConfigError,
)


def valid_profile() -> dict:
    return {
        "schema": GLM53_RUNTIME_CONFIG_SCHEMA,
        "architecture": GLM53_ARCHITECTURE,
        "checkpoint_revision": GLM53_CHECKPOINT_REVISION,
        "source_commit": "2dc3d6a2a125cad006426d77a2998c5dd4b7bd13",
        "source_tree": "3c30c1774a5ce7aca8db4fe82dadd92b01f0d4ac",
        "runtime_adapter": "vllm_neuron.model.glm53_flash.FutureRuntimeAdapter",
        "compiler_image_id": "sha256:" + "1" * 64,
        "compiler_image_digest": "example.invalid/neuron@sha256:" + "2" * 64,
        "compiler_version": "explicit-test-version",
        "runtime_packages": {
            "neuronx-distributed-inference": "explicit-test-identity",
            "torch-neuronx": "explicit-test-identity",
            "vllm-neuron": "2dc3d6a2a125cad006426d77a2998c5dd4b7bd13",
        },
        "compiler_flags": ["--explicit-test-flag"],
        "tensor_parallel_degree": 32,
        "logical_neuron_cores": 2,
        "batch_size": 1,
        "max_sequence_length": 8192,
        "context_encoding_buckets": [128, 512, 8192],
        "token_generation_buckets": [128, 512, 8192],
        "weight_dtype": "bfloat16",
        "cache_dtype": "bfloat16",
        "runtime_quantization": "explicit-test-value",
        "sampling_mode": "greedy",
        "speculative_decode": False,
    }


def test_canonical_round_trip_and_digest_are_stable():
    profile = Glm53RuntimeConfig.from_mapping(valid_profile())
    payload = profile.canonical_bytes()

    assert Glm53RuntimeConfig.load_canonical(payload) == profile
    assert len(profile.sha256()) == 64
    assert payload.endswith(b"\n")


@pytest.mark.parametrize("field", sorted(valid_profile()))
def test_missing_field_fails_closed(field):
    requested = valid_profile()
    requested.pop(field)

    with pytest.raises(Glm53RuntimeConfigError, match="field-set mismatch"):
        Glm53RuntimeConfig.from_mapping(requested)


def test_extra_field_fails_closed():
    requested = valid_profile()
    requested["silent_runtime_kwarg"] = True

    with pytest.raises(Glm53RuntimeConfigError, match="extra=.*silent_runtime_kwarg"):
        Glm53RuntimeConfig.from_mapping(requested)


def test_duplicate_json_key_fails_closed():
    payload = Glm53RuntimeConfig.from_mapping(valid_profile()).canonical_bytes()
    duplicate = payload.replace(
        b'{"architecture":',
        b'{"architecture":"Glm5NextForConditionalGeneration","architecture":',
        1,
    )

    with pytest.raises(Glm53RuntimeConfigError, match="duplicate JSON key"):
        Glm53RuntimeConfig.load_canonical(duplicate)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("architecture", "Glm52MoeDsaForCausalLM", "wrong GLM-5.3 architecture"),
        ("checkpoint_revision", "0" * 40, "wrong GLM-5.3 checkpoint revision"),
        ("source_commit", "main", "lowercase 40-hex"),
        ("compiler_image_id", "latest", "exact sha256 image ID"),
        ("compiler_image_digest", "latest", "exact repository digest"),
        ("tensor_parallel_degree", 16, "requires TP32"),
        ("sampling_mode", "temperature", "requires greedy"),
        ("speculative_decode", True, "outside the GLM-5.3 formal gate"),
    ],
)
def test_identity_and_formal_gate_drift_fail_closed(field, value, message):
    requested = valid_profile()
    requested[field] = value

    with pytest.raises(Glm53RuntimeConfigError, match=message):
        Glm53RuntimeConfig.from_mapping(requested)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("logical_neuron_cores", True),
        ("context_encoding_buckets", [512, 128]),
        ("token_generation_buckets", [128, 128]),
        ("compiler_flags", ["--same", "--same"]),
        ("runtime_packages", {}),
    ],
)
def test_malformed_profile_fails_closed(field, value):
    requested = valid_profile()
    requested[field] = value

    with pytest.raises(Glm53RuntimeConfigError):
        Glm53RuntimeConfig.from_mapping(requested)


def test_bucket_beyond_max_sequence_fails_closed():
    requested = valid_profile()
    requested["token_generation_buckets"] = [128, 16384]

    with pytest.raises(Glm53RuntimeConfigError, match="cannot exceed"):
        Glm53RuntimeConfig.from_mapping(requested)


@pytest.mark.parametrize(
    "field",
    [
        "batch_size",
        "runtime_adapter",
        "compiler_version",
        "compiler_flags",
        "runtime_packages",
        "runtime_quantization",
        "cache_dtype",
    ],
)
def test_requested_vs_emitted_drift_fails_closed(field):
    requested = valid_profile()
    reviewed = Glm53RuntimeConfig.from_mapping(requested)
    emitted = copy.deepcopy(requested)
    if field == "batch_size":
        emitted[field] = 2
    elif field == "compiler_flags":
        emitted[field] = ["--different"]
    elif field == "runtime_packages":
        emitted[field]["torch-neuronx"] = "different"
    else:
        emitted[field] = "different"

    with pytest.raises(Glm53RuntimeConfigError, match="drifted"):
        reviewed.assert_emitted(emitted)


def test_noncanonical_json_bytes_fail_reload():
    profile = Glm53RuntimeConfig.from_mapping(valid_profile())
    pretty = (
        json.dumps(profile.to_mapping(), indent=2, sort_keys=True) + "\n"
    ).encode()

    with pytest.raises(Glm53RuntimeConfigError, match="not the canonical"):
        Glm53RuntimeConfig.load_canonical(pretty)


def test_matching_emitted_profile_passes():
    requested = valid_profile()
    Glm53RuntimeConfig.from_mapping(requested).assert_emitted(copy.deepcopy(requested))
