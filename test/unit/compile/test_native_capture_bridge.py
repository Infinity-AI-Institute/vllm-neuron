import importlib.util
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
        "    return get_default_pass_manager()\n",
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


def test_native_capture_uses_source_pass_manager_factory():
    bridge = _load_bridge()
    native_capture = _native_capture_module()

    def source_factory():
        return "source-pass-manager"

    bridge.bind_source_pass_manager_to_native_capture(
        native_capture_backend=native_capture,
        pass_manager_factory=source_factory,
        lookup_backend=_lookup_for(native_capture),
    )

    assert native_capture.get_default_pass_manager is source_factory
    assert native_capture.get_default_pass_manager() == "source-pass-manager"
    assert native_capture.capture() == "source-pass-manager"


def test_native_capture_binding_rejects_decoy_compatibility_alias():
    bridge = _load_bridge()
    native_capture = _native_capture_module()
    original_factory = native_capture.get_default_pass_manager

    def decoy_capture(*args, **kwargs):
        return None

    with pytest.raises(
        RuntimeError,
        match="vllm_neuron_graph_capture is not bound to the native capture backend",
    ):
        bridge.bind_source_pass_manager_to_native_capture(
            native_capture_backend=native_capture,
            pass_manager_factory=lambda: "source-pass-manager",
            lookup_backend=_lookup_for(native_capture, decoy_capture),
        )

    assert native_capture.get_default_pass_manager is original_factory


def test_native_capture_binding_fails_closed_when_seam_disappears():
    bridge = _load_bridge()
    native_capture = _native_capture_module()
    del native_capture.__dict__["get_default_pass_manager"]

    with pytest.raises(
        RuntimeError, match="does not expose get_default_pass_manager"
    ):
        bridge.bind_source_pass_manager_to_native_capture(
            native_capture_backend=native_capture,
            pass_manager_factory=lambda: None,
            lookup_backend=_lookup_for(native_capture),
        )
