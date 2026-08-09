# SPDX-License-Identifier: Apache-2.0
import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


KIMI_K3_FACTORY = (
    "neuronx_distributed_inference.models.kimi_k3.serving.factory"
)
BASE_MODELS = (
    "LlamaForCausalLM",
    "GptOssForCausalLM",
    "Eagle3LlamaForCausalLM",
    "Qwen3VLForConditionalGeneration",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "InklingForConditionalGeneration",
    "InklingForCausalLM",
)


@pytest.fixture
def registry(monkeypatch):
    package = "_isolated_vllm_neuron_model"
    package_module = ModuleType(package)
    package_module.__path__ = []
    monkeypatch.setitem(sys.modules, package, package_module)

    dependencies = {
        "llama3": ("LlamaForCausalLM", "Eagle3LlamaForCausalLM"),
        "gpt_oss": ("GptOssForCausalLM",),
        "qwen3_vl": ("Qwen3VLForConditionalGeneration",),
        "gemma4": ("Gemma4ForCausalLM",),
        "inkling": ("InklingForConditionalGeneration",),
    }
    for relative_name, class_names in dependencies.items():
        module_name = f"{package}.{relative_name}"
        module = ModuleType(module_name)
        for class_name in class_names:
            setattr(module, class_name, type(class_name, (), {}))
        monkeypatch.setitem(sys.modules, module_name, module)

    registry_path = (
        Path(__file__).parents[3] / "vllm_neuron" / "model" / "registry.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"{package}.registry", registry_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _replace_k3_import(monkeypatch, outcome):
    original_import = builtins.__import__

    def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == KIMI_K3_FACTORY:
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", controlled_import)


def test_registry_includes_kimi_k3_when_ndi_import_is_available(
    registry, monkeypatch
) -> None:
    factory = ModuleType(KIMI_K3_FACTORY)
    kimi_k3_class = type("KimiK3ForCausalLM", (), {})
    factory.KimiK3ForCausalLM = kimi_k3_class
    _replace_k3_import(monkeypatch, factory)

    models = dict(registry.get_models())

    assert models["KimiK3ForCausalLM"] is kimi_k3_class


def test_registry_preserves_other_models_when_ndi_import_is_unavailable(
    registry, monkeypatch
) -> None:
    _replace_k3_import(
        monkeypatch,
        ModuleNotFoundError("No module named 'neuronx_distributed_inference'"),
    )

    assert tuple(name for name, _ in registry.get_models()) == BASE_MODELS


def test_registry_does_not_swallow_non_import_errors(registry, monkeypatch) -> None:
    _replace_k3_import(monkeypatch, RuntimeError("NDI initialization failed"))

    with pytest.raises(RuntimeError, match="NDI initialization failed"):
        registry.get_models()
