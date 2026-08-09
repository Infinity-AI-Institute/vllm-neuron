import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_envs():
    path = Path(__file__).parents[2] / "vllm_neuron" / "envs.py"
    spec = importlib.util.spec_from_file_location("vllm_neuron_envs_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_runtime_uses_registered_libtorch_compile_backend(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_BACKEND", "neuron_native")
    capture_backend = ModuleType("vllm_neuron.compile.capture_backend")
    capture_backend.select_native_capture_backend = lambda: "native-callable"
    monkeypatch.setitem(
        sys.modules, "vllm_neuron.compile.capture_backend", capture_backend
    )
    envs = _load_envs()

    assert envs.get_compile_backend_name() == "neuron_libtorch"
    assert envs.get_graph_capture_backend() == "native-callable"
    assert envs.get_dist_backend() == "gloo"


def test_xla_runtime_keeps_vllm_compile_backend_and_gloo_metadata(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_BACKEND", "vllm_neuron")
    envs = _load_envs()

    assert envs.get_compile_backend_name() == "vllm_neuron"
    assert envs.get_graph_capture_backend() == "vllm_neuron_graph_capture"
    assert envs.get_dist_backend() == "gloo"
