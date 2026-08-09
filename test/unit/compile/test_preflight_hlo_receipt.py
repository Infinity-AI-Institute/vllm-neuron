"""Same-process representative FX-to-HLO diagnostic receipt contracts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[3]


def _load_capture(monkeypatch, *, receipt_dir: str | None):
    package = ModuleType("vllm_neuron")
    package.__path__ = [str(ROOT / "vllm_neuron")]
    package.envs = SimpleNamespace(
        VLLM_NEURON_CPU_MODE=False,
        VLLM_NEURON_TRACE_PREFLIGHT_ONLY=True,
        VLLM_NEURON_TRACE_PREFLIGHT_LOWER_HLO=True,
        VLLM_NEURON_TRACE_PREFLIGHT_HLO_RECEIPT_DIR=receipt_dir,
        VLLM_NEURON_TRACE_MILESTONE_DIR=None,
    )
    monkeypatch.setitem(sys.modules, "vllm_neuron", package)

    compile_package = ModuleType("vllm_neuron.compile")
    compile_package.__path__ = [str(ROOT / "vllm_neuron/compile")]
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile", compile_package)

    backend = ModuleType("vllm_neuron.compile.backend")
    backend.preprocess_and_validate_inputs = lambda gm, inputs: (gm, inputs)
    backend._apply_platform_compiler_args = lambda options: {
        **options,
        "target_device": "xla",
    }
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile.backend", backend)

    fx_passes = ModuleType("vllm_neuron.fx_passes")
    fx_passes.get_default_pass_manager = lambda: pytest.fail("stubbed pipeline")
    monkeypatch.setitem(sys.modules, "vllm_neuron.fx_passes", fx_passes)
    pass_manager = ModuleType("vllm_neuron.fx_passes.pass_manager")
    pass_manager._format_replica_groups_header = lambda gm: "groups=[[0,1]]"
    monkeypatch.setitem(sys.modules, "vllm_neuron.fx_passes.pass_manager", pass_manager)
    timer = ModuleType("vllm_neuron.utils.timer")
    timer.timer = lambda: pytest.fail("stubbed pipeline")
    monkeypatch.setitem(sys.modules, "vllm_neuron.utils.timer", timer)

    milestones = ModuleType("vllm_neuron.compile.trace_milestones")
    events = []
    milestones.emit_trace_milestone = lambda event, **fields: events.append(
        (event, fields)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_neuron.compile.trace_milestones",
        milestones,
    )

    cache = ModuleType("vllm_neuron.compile.cache")
    cache.get_local = lambda *args, **kwargs: pytest.fail("no cache lookup")
    cache.save_artifact_metadata = lambda *args, **kwargs: pytest.fail(
        "no cache publication"
    )
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile.cache", cache)

    name = "vllm_neuron.compile.capture_backend"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "vllm_neuron/compile/capture_backend.py",
    )
    assert spec is not None and spec.loader is not None
    capture = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, capture)
    spec.loader.exec_module(capture)
    return capture, events


def test_hlo_preflight_environment_is_default_off(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_NEURON_TRACE_PREFLIGHT_LOWER_HLO", raising=False)
    monkeypatch.delenv(
        "VLLM_NEURON_TRACE_PREFLIGHT_HLO_RECEIPT_DIR",
        raising=False,
    )
    name = "_preflight_hlo_env_test_target"
    spec = importlib.util.spec_from_file_location(name, ROOT / "vllm_neuron/envs.py")
    assert spec is not None and spec.loader is not None
    envs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(envs)
    assert envs.VLLM_NEURON_TRACE_PREFLIGHT_LOWER_HLO is False
    assert envs.VLLM_NEURON_TRACE_PREFLIGHT_HLO_RECEIPT_DIR is None


def test_opt_in_lowers_live_fx_to_diagnostic_receipt(monkeypatch, tmp_path) -> None:
    capture, events = _load_capture(monkeypatch, receipt_dir=str(tmp_path))
    gm = torch.fx.symbolic_trace(torch.nn.Identity())
    pipeline_calls = []

    class FakeHlo:
        def SerializeToString(self):
            return b"representative-hlo"

    def fake_pipeline(live_gm, inputs, options, workdir):
        pipeline_calls.append((live_gm, inputs, options, workdir))
        return FakeHlo(), [2], False, {0: 1}, 1, 12.5

    monkeypatch.setattr(capture, "run_fx_to_hlo_pipeline", fake_pipeline)
    bail = capture.capture(gm, [torch.ones(2)], {})

    assert len(pipeline_calls) == 1
    assert pipeline_calls[0][0] is gm
    with pytest.raises(capture.CaptureComplete):
        bail()

    receipts = list(tmp_path.glob("rank-0-pid-*/receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["status"] == "complete"
    assert receipt["same_process_fx_to_hlo"] is True
    assert receipt["normal_all_rank_extraction_still_required"] is True
    assert receipt["cache_lookup_performed"] is False
    assert receipt["cache_published"] is False
    assert receipt["neff_written"] is False
    assert receipt["runtime_bypass_enabled"] is False
    assert receipt["inputs"]["tensor_payload_values_included"] is False
    assert receipt["lowering"]["unused_input_indices"] == [2]
    assert (receipts[0].parent / "graph.hlo").read_bytes() == b"representative-hlo"
    assert [event for event, _ in events] == [
        "capture_backend_reached",
        "preflight_hlo_persisted",
    ]


def test_opt_in_requires_separate_receipt_root(monkeypatch) -> None:
    capture, _ = _load_capture(monkeypatch, receipt_dir=None)
    gm = torch.fx.symbolic_trace(torch.nn.Identity())
    with pytest.raises(RuntimeError, match="HLO_RECEIPT_DIR is required"):
        capture.capture(gm, [torch.ones(1)], {})
