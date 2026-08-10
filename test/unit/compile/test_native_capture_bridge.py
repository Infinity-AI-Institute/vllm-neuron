import importlib.util
from pathlib import Path
from types import SimpleNamespace


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


def test_native_capture_uses_source_pass_manager_factory():
    bridge = _load_bridge()
    bundled_factory = object()
    native_capture = SimpleNamespace(get_default_pass_manager=bundled_factory)

    def source_factory():
        return "source-pass-manager"

    bridge.bind_source_pass_manager_to_native_capture(
        native_capture_backend=native_capture,
        pass_manager_factory=source_factory,
    )

    assert native_capture.get_default_pass_manager is source_factory
    assert native_capture.get_default_pass_manager() == "source-pass-manager"


def test_native_capture_binding_fails_closed_when_seam_disappears():
    bridge = _load_bridge()
    native_capture = SimpleNamespace()

    try:
        bridge.bind_source_pass_manager_to_native_capture(
            native_capture_backend=native_capture,
            pass_manager_factory=lambda: None,
        )
    except RuntimeError as exc:
        assert "does not expose get_default_pass_manager" in str(exc)
    else:
        raise AssertionError("missing native capture seam was accepted")
