# SPDX-License-Identifier: Apache-2.0
"""Dependency-isolated acceptance tests for the GLM joint BOTH constructor."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]
PACKAGE = ROOT / "vllm_neuron" / "model" / "glm53_flash"


def _load(name: str):
    qualified = f"vllm_neuron.model.glm53_flash.{name}"
    spec = importlib.util.spec_from_file_location(qualified, PACKAGE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


for package_name, path in (
    ("vllm_neuron", ROOT / "vllm_neuron"),
    ("vllm_neuron.model", ROOT / "vllm_neuron" / "model"),
    ("vllm_neuron.model.glm53_flash", PACKAGE),
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(path)]
    sys.modules.setdefault(package_name, package)

RUNTIME_CONFIG = _load("runtime_config")
runtime_factory = types.ModuleType("vllm_neuron.model.glm53_flash.runtime_factory")
runtime_factory.GLM53_RUNTIME_ADAPTER = (
    "vllm_neuron.model.glm53_flash.runtime_factory.Glm53RuntimeFactory"
)
sys.modules[runtime_factory.__name__] = runtime_factory
ADAPTER = _load("compile_adapter")


def _profile(**updates):
    value = {
        "schema": RUNTIME_CONFIG.GLM53_RUNTIME_CONFIG_SCHEMA,
        "architecture": RUNTIME_CONFIG.GLM53_ARCHITECTURE,
        "checkpoint_revision": RUNTIME_CONFIG.GLM53_CHECKPOINT_REVISION,
        "source_commit": "1" * 40,
        "source_tree": "2" * 40,
        "runtime_adapter": runtime_factory.GLM53_RUNTIME_ADAPTER,
        "compiler_image_id": "sha256:" + "3" * 64,
        "compiler_image_digest": "example.invalid/neuron@sha256:" + "4" * 64,
        "compiler_version": "pinned",
        "runtime_packages": {"vllm-neuron": "1" * 40},
        "compiler_flags": ["--auto-cast=none"],
        "tensor_parallel_degree": 64,
        "logical_neuron_cores": 2,
        "batch_size": 1,
        "max_sequence_length": 2560,
        "context_encoding_buckets": [2048],
        "token_generation_buckets": [2560],
        "weight_dtype": "bfloat16",
        "cache_dtype": "bfloat16",
        "runtime_quantization": "none",
        "sampling_mode": "greedy",
        "output_logits": True,
        "speculative_decode": False,
    }
    value.update(updates)
    return RUNTIME_CONFIG.Glm53RuntimeConfig.from_mapping(value)


def _phase(tag: str, prefill: bool, active: int, buckets: list):
    return SimpleNamespace(
        tag=tag,
        neuron_config=SimpleNamespace(
            is_prefill_stage=prefill,
            n_active_tokens=active,
            buckets=buckets,
            seq_len=2560,
            tp_degree=64,
            logical_nc_config=2,
        ),
    )


def _application():
    cte = _phase("context_encoding_model", True, 2048, [2048])
    tkg = _phase("token_generation_model", False, 1, [[1, 2560]])
    return SimpleNamespace(
        _emit_phases="BOTH",
        _builder=None,
        models=[cte, tkg],
        context_encoding_model=cte,
        token_generation_model=tkg,
    )


def test_exact_joint_application_is_admitted():
    ADAPTER.assert_joint_both_application(_profile(), _application())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tensor_parallel_degree", 32),
        ("logical_neuron_cores", 1),
        ("batch_size", 2),
        ("max_sequence_length", 3072),
        ("context_encoding_buckets", [1024, 2048]),
        ("token_generation_buckets", [2048]),
    ],
)
def test_profile_drift_is_rejected(field, value):
    with pytest.raises(ADAPTER.Glm53CompileAdapterError):
        ADAPTER.assert_joint_both_application(_profile(**{field: value}), _application())


@pytest.mark.parametrize("mutation", ["phase", "order", "owner", "cte", "tkg", "builder"])
def test_phase_or_builder_drift_is_rejected(mutation):
    application = _application()
    if mutation == "phase":
        application._emit_phases = "CTE"
    elif mutation == "order":
        application.models.reverse()
    elif mutation == "owner":
        application.token_generation_model = object()
    elif mutation == "cte":
        application.models[0].neuron_config.n_active_tokens = 1024
    elif mutation == "tkg":
        application.models[1].neuron_config.buckets = [[1, 2048]]
    else:
        application._builder = object()
    with pytest.raises(ADAPTER.Glm53CompileAdapterError):
        ADAPTER.assert_joint_both_application(_profile(), application)


def test_constructor_only_constructs_one_application(monkeypatch):
    calls = []

    class Wrapper:
        @classmethod
        def build_inference_config(cls, source_config, **kwargs):
            calls.append(("config", source_config, kwargs))
            return "inference-config"

        def __new__(cls, model_path, config):
            calls.append(("construct", model_path, config))
            return _application()

    monkeypatch.setenv("NXDI_EMIT_PHASES", "BOTH")
    ADAPTER.construct_joint_both_application(
        _profile(),
        model_path="/immutable/checkpoint",
        source_config="source-config",
        _wrapper_cls=Wrapper,
    )
    assert [row[0] for row in calls] == ["config", "construct"]
    kwargs = calls[0][2]
    assert kwargs["tp_degree"] == 64
    assert kwargs["ctx_batch_size"] == kwargs["tkg_batch_size"] == 1
    assert kwargs["seq_len"] == 2560
    assert kwargs["context_encoding_buckets"] == [2048]
    assert kwargs["token_generation_buckets"] == [2560]
    assert kwargs["max_context_length"] == 2048


@pytest.mark.parametrize("selection", [None, "CTE", "TKG", "both"])
def test_constructor_requires_explicit_uppercase_both(monkeypatch, selection):
    if selection is None:
        monkeypatch.delenv("NXDI_EMIT_PHASES", raising=False)
    else:
        monkeypatch.setenv("NXDI_EMIT_PHASES", selection)
    with pytest.raises(ADAPTER.Glm53CompileAdapterError, match="explicitly BOTH"):
        ADAPTER.construct_joint_both_application(
            _profile(),
            model_path="/immutable/checkpoint",
            source_config=object(),
            _wrapper_cls=object(),
        )
