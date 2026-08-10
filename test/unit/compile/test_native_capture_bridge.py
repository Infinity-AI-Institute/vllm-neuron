import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_bridge():
    path = (
        Path(__file__).parents[3]
        / "vllm_neuron"
        / "compile"
        / "native_capture_bridge.py"
    )
    spec = importlib.util.spec_from_file_location(
        "native_capture_bridge_under_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native_capture_module():
    module = ModuleType("libtorch_neuronx_lite.compile.capture_backend")
    exec(
        "def bundled_factory():\n"
        "    return 'bundled-pass-manager'\n\n"
        "get_default_pass_manager = bundled_factory\n\n"
        "def capture(*args, **kwargs):\n"
        "    return get_default_pass_manager()\n\n"
        "def run_fx_to_hlo_pipeline():\n"
        "    from libtorch_neuronx_lite.compile.hlo import convert_fx_to_hlo\n"
        "    return convert_fx_to_hlo()\n",
        module.__dict__,
    )
    return module


def _native_hlo_module():
    module = ModuleType("libtorch_neuronx_lite.compile.hlo")
    exec(
        "def convert_fx_to_hlo(*args, **kwargs):\n    return 'bundled-hlo'\n",
        module.__dict__,
    )
    return module


def _lookup_for(native_capture, compatibility_capture=None):
    backends = {
        "neuron_libtorch_graph_capture": native_capture.capture,
        "vllm_neuron_graph_capture": (
            native_capture.capture
            if compatibility_capture is None
            else compatibility_capture
        ),
    }
    return backends.__getitem__


def _install_native_hlo_import(monkeypatch, native_hlo):
    package = ModuleType("libtorch_neuronx_lite")
    package.__path__ = []
    compile_package = ModuleType("libtorch_neuronx_lite.compile")
    compile_package.__path__ = []
    compile_package.hlo = native_hlo
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, compile_package.__name__, compile_package)
    monkeypatch.setitem(sys.modules, native_hlo.__name__, native_hlo)


def test_native_capture_uses_source_graph_transformations(monkeypatch):
    bridge = _load_bridge()
    native_capture = _native_capture_module()
    native_hlo = _native_hlo_module()
    _install_native_hlo_import(monkeypatch, native_hlo)

    def source_factory():
        return "source-pass-manager"

    def source_hlo_converter(*args, **kwargs):
        return "source-hlo"

    bridge.bind_source_pass_manager_to_native_capture(
        native_capture_backend=native_capture,
        native_hlo_module=native_hlo,
        pass_manager_factory=source_factory,
        source_hlo_converter=source_hlo_converter,
        lookup_backend=_lookup_for(native_capture),
    )

    assert native_capture.get_default_pass_manager is source_factory
    assert native_capture.get_default_pass_manager() == "source-pass-manager"
    assert native_capture.capture() == "source-pass-manager"
    assert native_hlo.convert_fx_to_hlo is source_hlo_converter
    assert native_hlo.convert_fx_to_hlo() == "source-hlo"
    assert native_capture.run_fx_to_hlo_pipeline() == "source-hlo"

    bridge.bind_source_pass_manager_to_native_capture(
        native_capture_backend=native_capture,
        native_hlo_module=native_hlo,
        pass_manager_factory=source_factory,
        source_hlo_converter=source_hlo_converter,
        lookup_backend=_lookup_for(native_capture),
    )
    assert native_capture.run_fx_to_hlo_pipeline() == "source-hlo"


def test_native_capture_binding_rejects_decoy_compatibility_alias():
    bridge = _load_bridge()
    native_capture = _native_capture_module()
    native_hlo = _native_hlo_module()
    original_factory = native_capture.get_default_pass_manager

    def decoy_capture(*args, **kwargs):
        return None

    with pytest.raises(
        RuntimeError,
        match="vllm_neuron_graph_capture is not bound to the native capture backend",
    ):
        bridge.bind_source_pass_manager_to_native_capture(
            native_capture_backend=native_capture,
            native_hlo_module=native_hlo,
            pass_manager_factory=lambda: "source-pass-manager",
            source_hlo_converter=lambda: "source-hlo",
            lookup_backend=_lookup_for(native_capture, decoy_capture),
        )

    assert native_capture.get_default_pass_manager is original_factory


def test_native_capture_binding_fails_closed_when_seam_disappears():
    bridge = _load_bridge()
    native_capture = _native_capture_module()
    native_hlo = _native_hlo_module()
    del native_capture.__dict__["get_default_pass_manager"]

    with pytest.raises(RuntimeError, match="does not expose get_default_pass_manager"):
        bridge.bind_source_pass_manager_to_native_capture(
            native_capture_backend=native_capture,
            native_hlo_module=native_hlo,
            pass_manager_factory=lambda: None,
            source_hlo_converter=lambda: None,
            lookup_backend=_lookup_for(native_capture),
        )


def test_native_capture_binding_fails_closed_when_hlo_seam_disappears():
    bridge = _load_bridge()
    native_capture = _native_capture_module()
    native_hlo = _native_hlo_module()
    original_factory = native_capture.get_default_pass_manager
    del native_hlo.__dict__["convert_fx_to_hlo"]

    with pytest.raises(
        RuntimeError, match="native HLO module does not expose convert_fx_to_hlo"
    ):
        bridge.bind_source_pass_manager_to_native_capture(
            native_capture_backend=native_capture,
            native_hlo_module=native_hlo,
            pass_manager_factory=lambda: None,
            source_hlo_converter=lambda: None,
            lookup_backend=_lookup_for(native_capture),
        )

    assert native_capture.get_default_pass_manager is original_factory


def test_native_capture_binding_rejects_noncallable_source_hlo_converter():
    bridge = _load_bridge()
    native_capture = _native_capture_module()
    native_hlo = _native_hlo_module()
    original_factory = native_capture.get_default_pass_manager
    original_converter = native_hlo.convert_fx_to_hlo

    with pytest.raises(RuntimeError, match="source HLO converter is not callable"):
        bridge.bind_source_pass_manager_to_native_capture(
            native_capture_backend=native_capture,
            native_hlo_module=native_hlo,
            pass_manager_factory=lambda: None,
            source_hlo_converter=object(),
            lookup_backend=_lookup_for(native_capture),
        )

    assert native_capture.get_default_pass_manager is original_factory
    assert native_hlo.convert_fx_to_hlo is original_converter


def test_exact_native_package_resolves_source_hlo_converter(monkeypatch):
    """Exercise the image package's real function-local converter import."""
    pytest.importorskip("libtorch_neuronx_lite")
    monkeypatch.setenv("VLLM_NEURON_BACKEND", "neuron_native")

    import torch._dynamo.backends.registry as registry
    from libtorch_neuronx_lite.compile import capture_backend, hlo

    import vllm_neuron
    from vllm_neuron.compile.hlo import convert_fx_to_hlo

    vllm_neuron._init_backend()

    assert registry.lookup_backend("vllm_neuron_graph_capture") is (
        capture_backend.capture
    )
    assert registry.lookup_backend("neuron_libtorch_graph_capture") is (
        capture_backend.capture
    )
    assert hlo.convert_fx_to_hlo is convert_fx_to_hlo
    assert capture_backend.run_fx_to_hlo_pipeline.__globals__ is (
        capture_backend.__dict__
    )
    assert "convert_fx_to_hlo" not in (
        capture_backend.run_fx_to_hlo_pipeline.__globals__
    )
    assert convert_fx_to_hlo.__code__.co_filename.endswith("vllm_neuron/compile/hlo.py")


def test_exact_native_package_source_converter_serializes_tiny_hlo(
    monkeypatch, tmp_path
):
    pytest.importorskip("libtorch_neuronx_lite")
    monkeypatch.setenv("VLLM_NEURON_BACKEND", "neuron_native")
    monkeypatch.setenv("PJRT_DEVICE", "CPU")

    import torch
    from libtorch_neuronx_lite.compile import hlo

    import vllm_neuron
    from vllm_neuron.compile.hlo import convert_fx_to_hlo

    vllm_neuron._init_backend()
    graph_module = torch.fx.symbolic_trace(lambda value: value + 1)
    module, unused, has_rng = hlo.convert_fx_to_hlo(
        graph_module,
        (torch.zeros(2, dtype=torch.float32),),
        log_path=f"{tmp_path}/",
    )

    assert hlo.convert_fx_to_hlo is convert_fx_to_hlo
    assert module.computations
    assert module.SerializeToString()
    assert unused == []
    assert has_rng is False
    assert (tmp_path / "step1_torch_xla_trace.hlo").is_file()
