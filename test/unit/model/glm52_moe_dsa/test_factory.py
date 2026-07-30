# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from transformers import PretrainedConfig

from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.factory import GlmMoeDsaForCausalLM
from vllm_neuron.model.glm52_moe_dsa.static_fp8 import (
    NEURON_LEGACY_E4M3FN_QMAX240,
    OCP_E4M3FN_QMAX448,
)
from vllm_neuron.model.neuron_config import NeuronConfig


def _hf_config() -> PretrainedConfig:
    frozen = Glm52MoeDsaConfig()
    values = {
        field: getattr(frozen, field)
        for field in frozen.__dataclass_fields__
        if field not in (
            "neuron_config",
            "static_fp8_weight_format",
            "torch_dtype",
        )
    }
    values.update(
        architectures=["GlmMoeDsaForCausalLM"],
        torch_dtype="bfloat16",
        quantization_config={
            "quant_method": "modelopt",
            "quantization": {
                "quant_algo": "FP8",
                "exclude_modules": ["lm_head"],
            },
        },
        glm52_artifact={
            "artifact_version": "glm52-trn2-static-fp8-v1",
            "loader_ready": True,
            "mtp_enabled": False,
            "index_closure_status": "passed",
        },
    )
    return PretrainedConfig(**values)


def _neuron_config() -> NeuronConfig:
    return NeuronConfig(
        ep_degree=8,
        on_device_sampling_config=None,
    )


def test_factory_accepts_only_frozen_tp64_trn2_static_fp8(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setattr(
        "vllm_neuron.model.glm52_moe_dsa.factory._get_tp_world_size",
        lambda: 64,
    )

    GlmMoeDsaForCausalLM._validate_config(_hf_config(), _neuron_config())


def test_factory_accepts_explicit_bf16_shared_hybrid(monkeypatch) -> None:
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setattr(
        "vllm_neuron.model.glm52_moe_dsa.factory._get_tp_world_size",
        lambda: 64,
    )
    config = _hf_config()
    config.shared_expert_dtype = "bfloat16"
    config.glm52_artifact.update(
        artifact_version="glm52-trn2-static-fp8-bf16-shared-v1",
        shared_expert_dtype="bfloat16",
    )
    config.quantization_config["quantization"]["exclude_modules"].append(
        "model.layers.*.mlp.shared_experts.*"
    )

    GlmMoeDsaForCausalLM._validate_config(config, _neuron_config())


def test_factory_accepts_explicit_direct_legacy_hybrid(monkeypatch) -> None:
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setattr(
        "vllm_neuron.model.glm52_moe_dsa.factory._get_tp_world_size",
        lambda: 64,
    )
    config = _hf_config()
    config.shared_expert_dtype = "bfloat16"
    config.static_fp8_weight_format = NEURON_LEGACY_E4M3FN_QMAX240
    config.glm52_artifact.update(
        artifact_version=(
            "glm52-trn2-static-fp8-direct-legacy-bf16-shared-v1"
        ),
        shared_expert_dtype="bfloat16",
        static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
    )
    config.quantization_config["quantization"].update(
        weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
    )
    config.quantization_config["quantization"]["exclude_modules"].append(
        "model.layers.*.mlp.shared_experts.*"
    )

    GlmMoeDsaForCausalLM._validate_config(config, _neuron_config())


@pytest.mark.parametrize(
    ("location", "value"),
    (
        ("top", OCP_E4M3FN_QMAX448),
        ("artifact", OCP_E4M3FN_QMAX448),
        ("quantization", OCP_E4M3FN_QMAX448),
        ("missing_artifact", None),
    ),
)
def test_factory_rejects_mixed_or_partial_direct_weight_format(
    monkeypatch,
    location: str,
    value: str | None,
) -> None:
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    config = _hf_config()
    config.static_fp8_weight_format = NEURON_LEGACY_E4M3FN_QMAX240
    config.glm52_artifact.update(
        artifact_version="glm52-trn2-static-fp8-direct-legacy-v1",
        static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
    )
    config.quantization_config["quantization"]["weight_format"] = (
        NEURON_LEGACY_E4M3FN_QMAX240
    )
    if location == "top":
        config.static_fp8_weight_format = value
    elif location == "artifact":
        config.glm52_artifact["static_fp8_weight_format"] = value
    elif location == "quantization":
        config.quantization_config["quantization"]["weight_format"] = value
    else:
        config.glm52_artifact.pop("static_fp8_weight_format")

    with pytest.raises(ValueError, match="static-FP8|mixed"):
        GlmMoeDsaForCausalLM._validate_config(config, _neuron_config())


def test_factory_rejects_hybrid_marker_without_bf16_exclusion(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    config = _hf_config()
    config.shared_expert_dtype = "bfloat16"
    config.glm52_artifact.update(
        artifact_version="glm52-trn2-static-fp8-bf16-shared-v1",
        shared_expert_dtype="bfloat16",
    )

    with pytest.raises(ValueError, match="exclude shared experts"):
        GlmMoeDsaForCausalLM._validate_config(config, _neuron_config())


def test_factory_rejects_hybrid_config_with_static_artifact(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    config = _hf_config()
    config.shared_expert_dtype = "bfloat16"

    with pytest.raises(ValueError, match="artifact_version"):
        GlmMoeDsaForCausalLM._validate_config(config, _neuron_config())


def test_factory_rejects_native_block_fp8(monkeypatch) -> None:
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    config = _hf_config()
    config.quantization_config = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }

    try:
        GlmMoeDsaForCausalLM._validate_config(config, _neuron_config())
    except ValueError as error:
        assert "native block-FP8" in str(error)
    else:
        raise AssertionError("native block-FP8 artifact was accepted")


def test_factory_rejects_artifact_without_loader_closure(monkeypatch) -> None:
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    config = _hf_config()
    config.glm52_artifact["loader_ready"] = False

    try:
        GlmMoeDsaForCausalLM._validate_config(config, _neuron_config())
    except ValueError as error:
        assert "loader_ready" in str(error)
    else:
        raise AssertionError("incomplete converted artifact was accepted")


def test_factory_accepts_non_serving_compile_stub_only_in_cpu_compile(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setattr(
        "vllm_neuron.model.glm52_moe_dsa.factory._get_tp_world_size",
        lambda: 64,
    )
    config = _hf_config()
    config.glm52_artifact.update(
        loader_ready=False,
        compile_stub=True,
    )

    with pytest.raises(ValueError, match="CPU_COMPILE"):
        GlmMoeDsaForCausalLM._validate_config(config, _neuron_config())

    monkeypatch.setenv("VLLM_NEURON_CPU_COMPILE", "1")
    GlmMoeDsaForCausalLM._validate_config(config, _neuron_config())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("artifact_version", "forged-v2", "artifact_version"),
        ("mtp_enabled", True, "MTP disabled"),
        ("index_closure_status", "skipped", "index closure"),
    ),
)
def test_factory_rejects_incompatible_artifact_markers(
    monkeypatch,
    field: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    config = _hf_config()
    config.glm52_artifact[field] = value

    with pytest.raises(ValueError, match=message):
        GlmMoeDsaForCausalLM._validate_config(config, _neuron_config())


def test_factory_rejects_dp_and_on_device_sampling(monkeypatch) -> None:
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setattr(
        "vllm_neuron.model.glm52_moe_dsa.factory._get_tp_world_size",
        lambda: 64,
    )
    config = _neuron_config()
    config.attention_dp_size = 2

    try:
        GlmMoeDsaForCausalLM._validate_config(_hf_config(), config)
    except ValueError as error:
        assert "attention_dp_size" in str(error)
    else:
        raise AssertionError("unsupported attention DP was accepted")

    config.attention_dp_size = 1
    config.on_device_sampling_config = object()
    try:
        GlmMoeDsaForCausalLM._validate_config(_hf_config(), config)
    except ValueError as error:
        assert "on-device sampling" in str(error)
    else:
        raise AssertionError("unsupported on-device sampling was accepted")


def test_registry_uses_exact_hugging_face_architecture_key() -> None:
    repository = Path(__file__).parents[4]
    registry_source = (repository / "vllm_neuron" / "model" / "registry.py").read_text(
        encoding="utf-8"
    )
    assert '("GlmMoeDsaForCausalLM", GlmMoeDsaForCausalLM)' in registry_source
