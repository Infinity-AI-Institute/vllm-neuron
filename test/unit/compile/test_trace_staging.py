"""Representative-rank trace staging and milestone contracts."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[3]


def _load(monkeypatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def staging(monkeypatch):
    package = ModuleType("vllm_neuron")
    package.__path__ = [str(ROOT / "vllm_neuron")]
    package.envs = SimpleNamespace(
        VLLM_NEURON_PARALLEL_TRACE_WORKERS=1,
        VLLM_NEURON_TRACE_RANK_CONCURRENCY=None,
        VLLM_NEURON_TRACE_PREFLIGHT_RANK=0,
        VLLM_NEURON_TRACE_PREFLIGHT_JOBS=None,
        VLLM_NEURON_TRACE_PREFLIGHT_TIMEOUT_SECONDS=14400,
        VLLM_NEURON_TRACE_PREFLIGHT_HEARTBEAT_SECONDS=300,
        VLLM_NEURON_TRACE_PREFLIGHT_ONLY=False,
        VLLM_NEURON_TRACE_MILESTONE_DIR=None,
    )
    monkeypatch.setitem(sys.modules, "vllm_neuron", package)

    compile_package = ModuleType("vllm_neuron.compile")
    compile_package.__path__ = [str(ROOT / "vllm_neuron/compile")]
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile", compile_package)

    return _load(
        monkeypatch,
        "vllm_neuron.compile.parallel_trace",
        ROOT / "vllm_neuron/compile/parallel_trace.py",
    )


def _distributed(monkeypatch, staging, *, rank: int, world_size: int = 2):
    monkeypatch.setattr(staging.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(staging.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(staging.torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(staging.torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(
        staging.torch.distributed, "new_group", lambda ranks, timeout: "preflight-group"
    )
    monkeypatch.setattr(
        staging.torch.distributed, "destroy_process_group", lambda group: None
    )


def test_representative_probe_is_discarded_then_normal_trace_runs(monkeypatch, staging):
    _distributed(monkeypatch, staging, rank=0)
    jobs = [(object(), {}), (object(), {})]
    calls = []

    def fake_trace(trace_jobs, parent_rank=0):
        calls.append(
            (
                trace_jobs,
                parent_rank,
                os.environ.get("VLLM_NEURON_TRACE_PREFLIGHT_ONLY"),
            )
        )

    monkeypatch.setattr(staging, "parallel_trace", fake_trace)
    monkeypatch.setattr(
        staging.torch.distributed,
        "broadcast_object_list",
        lambda payload, src, group, device: None,
    )

    staging.parallel_trace_with_preflight(jobs, parent_rank=0)

    assert calls == [(jobs, 0, "1"), (jobs, 0, None)]
    assert "VLLM_NEURON_TRACE_PREFLIGHT_ONLY" not in os.environ


def test_unset_preflight_preserves_single_normal_trace(monkeypatch, staging):
    staging.envs.VLLM_NEURON_TRACE_PREFLIGHT_RANK = None
    jobs = [(object(), {})]
    calls = []
    monkeypatch.setattr(
        staging,
        "parallel_trace",
        lambda trace_jobs, parent_rank=0: calls.append((trace_jobs, parent_rank)),
    )

    staging.parallel_trace_with_preflight(jobs, parent_rank=3)

    assert calls == [(jobs, 3)]


def test_waiting_rank_receives_success_before_starting_normal_trace(
    monkeypatch, staging
):
    _distributed(monkeypatch, staging, rank=1)
    jobs = [(object(), {})]
    calls = []

    def release(payload, src, group, device):
        assert group == "preflight-group"
        payload[0] = {"ok": True, "representative_rank": 0, "staged_jobs": 1}

    monkeypatch.setattr(staging.torch.distributed, "broadcast_object_list", release)
    monkeypatch.setattr(
        staging,
        "parallel_trace",
        lambda trace_jobs, parent_rank=0: calls.append((trace_jobs, parent_rank)),
    )

    staging.parallel_trace_with_preflight(jobs, parent_rank=1)

    assert calls == [(jobs, 1)]


def test_first_probe_failure_is_broadcast_and_blocks_normal_trace(monkeypatch, staging):
    _distributed(monkeypatch, staging, rank=0)
    jobs = [(object(), {})]
    broadcasts = []

    def fail_trace(trace_jobs, parent_rank=0):
        assert os.environ["VLLM_NEURON_TRACE_PREFLIGHT_ONLY"] == "1"
        raise RuntimeError("fake tensor device mismatch")

    def broadcast(payload, src, group, device):
        broadcasts.append(payload[0].copy())

    monkeypatch.setattr(staging, "parallel_trace", fail_trace)
    monkeypatch.setattr(staging.torch.distributed, "broadcast_object_list", broadcast)

    with pytest.raises(RuntimeError, match="fake tensor device mismatch"):
        staging.parallel_trace_with_preflight(jobs, parent_rank=0)

    assert broadcasts[0]["ok"] is False
    assert broadcasts[0]["error_type"] == "RuntimeError"
    assert "VLLM_NEURON_TRACE_PREFLIGHT_ONLY" not in os.environ


def test_waiting_rank_raises_representative_failure_without_tracing(
    monkeypatch, staging
):
    _distributed(monkeypatch, staging, rank=1)
    calls = []

    def reject(payload, src, group, device):
        payload[0] = {
            "ok": False,
            "representative_rank": 0,
            "staged_jobs": 1,
            "error_type": "RuntimeError",
            "error_message": "common failure",
            "traceback": "representative traceback",
        }

    monkeypatch.setattr(staging.torch.distributed, "broadcast_object_list", reject)
    monkeypatch.setattr(
        staging,
        "parallel_trace",
        lambda trace_jobs, parent_rank=0: calls.append((trace_jobs, parent_rank)),
    )

    with pytest.raises(RuntimeError, match="common failure"):
        staging.parallel_trace_with_preflight([(object(), {})], parent_rank=1)

    assert calls == []


def test_job_limit_only_changes_probe_not_normal_trace(monkeypatch, staging):
    _distributed(monkeypatch, staging, rank=0)
    staging.envs.VLLM_NEURON_TRACE_PREFLIGHT_JOBS = 1
    jobs = [(object(), {}), (object(), {})]
    calls = []
    monkeypatch.setattr(
        staging,
        "parallel_trace",
        lambda trace_jobs, parent_rank=0: calls.append(list(trace_jobs)),
    )
    monkeypatch.setattr(
        staging.torch.distributed,
        "broadcast_object_list",
        lambda payload, src, group, device: None,
    )

    staging.parallel_trace_with_preflight(jobs, parent_rank=0)

    assert calls == [jobs[:1], jobs]


def test_preflight_rank_must_exist(monkeypatch, staging):
    _distributed(monkeypatch, staging, rank=0, world_size=2)
    staging.envs.VLLM_NEURON_TRACE_PREFLIGHT_RANK = 2

    with pytest.raises(ValueError, match="outside the distributed world"):
        staging.parallel_trace_with_preflight([(object(), {})], parent_rank=0)


def test_preflight_group_is_created_before_representative_trace(monkeypatch, staging):
    _distributed(monkeypatch, staging, rank=0, world_size=4)
    events = []

    def new_group(*, ranks, timeout):
        events.append(("new_group", ranks, timeout.total_seconds()))
        return "isolated-control-group"

    def trace(jobs, parent_rank=0):
        events.append(("trace", parent_rank))

    def broadcast(payload, src, group, device):
        events.append(("broadcast", src, group, device.type))

    monkeypatch.setattr(staging.torch.distributed, "new_group", new_group)
    monkeypatch.setattr(staging.torch.distributed, "broadcast_object_list", broadcast)
    monkeypatch.setattr(staging, "parallel_trace", trace)

    staging.parallel_trace_with_preflight([(object(), {})], parent_rank=0)

    assert events == [
        ("new_group", [0, 1, 2, 3], 14400.0),
        ("trace", 0),
        ("broadcast", 0, "isolated-control-group", "cpu"),
        ("trace", 0),
    ]


def test_preflight_rendezvous_failure_is_bounded_and_group_is_destroyed(
    monkeypatch, staging
):
    _distributed(monkeypatch, staging, rank=1)
    destroyed = []
    monkeypatch.setattr(
        staging.torch.distributed,
        "destroy_process_group",
        lambda group: destroyed.append(group),
    )
    monkeypatch.setattr(
        staging.torch.distributed,
        "broadcast_object_list",
        lambda payload, src, group, device: (_ for _ in ()).throw(
            TimeoutError("control deadline")
        ),
    )

    with pytest.raises(RuntimeError, match="dedicated 14400s deadline"):
        staging.parallel_trace_with_preflight([(object(), {})], parent_rank=1)

    assert destroyed == ["preflight-group"]


def test_preflight_heartbeat_must_be_shorter_than_deadline(monkeypatch, staging):
    _distributed(monkeypatch, staging, rank=1)
    staging.envs.VLLM_NEURON_TRACE_PREFLIGHT_TIMEOUT_SECONDS = 300
    staging.envs.VLLM_NEURON_TRACE_PREFLIGHT_HEARTBEAT_SECONDS = 300

    with pytest.raises(ValueError, match="must be less than"):
        staging.parallel_trace_with_preflight([(object(), {})], parent_rank=1)


def test_wait_heartbeat_reports_elapsed_time_without_extending_deadline(
    monkeypatch, staging
):
    records = []

    class FakeEvent:
        def __init__(self):
            self.waits = 0

        def wait(self, timeout):
            assert timeout == 300
            self.waits += 1
            return self.waits > 1

        def set(self):
            pass

    class ImmediateThread:
        def __init__(self, *, target, name, daemon):
            assert name == "preflight-heartbeat-rank-62"
            assert daemon is True
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout):
            assert timeout == 1.0

    monkeypatch.setattr(staging.threading, "Event", FakeEvent)
    monkeypatch.setattr(staging.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(
        staging,
        "emit_trace_milestone",
        lambda event, **fields: records.append((event, fields)),
    )

    with staging._preflight_wait_heartbeat(
        parent_rank=62,
        representative_rank=0,
        timeout_seconds=14400,
        heartbeat_seconds=300,
    ):
        pass

    assert len(records) == 1
    event, fields = records[0]
    assert event == "preflight_wait_heartbeat"
    assert fields["parent_rank"] == 62
    assert fields["representative_rank"] == 0
    assert fields["timeout_seconds"] == 14400
    assert fields["elapsed_seconds"] >= 0


def test_capture_preflight_stops_before_cache_or_hlo(monkeypatch):
    package = ModuleType("vllm_neuron")
    package.__path__ = [str(ROOT / "vllm_neuron")]
    package.envs = SimpleNamespace(
        VLLM_NEURON_CPU_MODE=False,
        VLLM_NEURON_TRACE_PREFLIGHT_ONLY=True,
        VLLM_NEURON_TRACE_MILESTONE_DIR=None,
    )
    monkeypatch.setitem(sys.modules, "vllm_neuron", package)

    compile_package = ModuleType("vllm_neuron.compile")
    compile_package.__path__ = [str(ROOT / "vllm_neuron/compile")]
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile", compile_package)

    backend = ModuleType("vllm_neuron.compile.backend")
    backend.preprocess_and_validate_inputs = lambda gm, inputs: (gm, inputs)
    backend._apply_platform_compiler_args = lambda options: pytest.fail(
        "platform options are after the preflight boundary"
    )
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile.backend", backend)

    fx_passes = ModuleType("vllm_neuron.fx_passes")
    fx_passes.get_default_pass_manager = lambda: pytest.fail("no FX passes")
    monkeypatch.setitem(sys.modules, "vllm_neuron.fx_passes", fx_passes)
    pass_manager = ModuleType("vllm_neuron.fx_passes.pass_manager")
    pass_manager._format_replica_groups_header = lambda gm: ""
    monkeypatch.setitem(sys.modules, "vllm_neuron.fx_passes.pass_manager", pass_manager)
    timer = ModuleType("vllm_neuron.utils.timer")
    timer.timer = lambda: pytest.fail("no FX-to-HLO timer")
    monkeypatch.setitem(sys.modules, "vllm_neuron.utils.timer", timer)

    capture = _load(
        monkeypatch,
        "vllm_neuron.compile.capture_backend",
        ROOT / "vllm_neuron/compile/capture_backend.py",
    )
    gm = torch.fx.symbolic_trace(torch.nn.Identity())
    bail = capture.capture(gm, [torch.ones(1)], {})

    with pytest.raises(capture.CaptureComplete):
        bail()
    assert "vllm_neuron.compile.cache" not in sys.modules


def test_milestones_are_one_json_record_per_line(monkeypatch, tmp_path):
    package = ModuleType("vllm_neuron")
    package.__path__ = [str(ROOT / "vllm_neuron")]
    package.envs = SimpleNamespace(VLLM_NEURON_TRACE_MILESTONE_DIR=str(tmp_path))
    monkeypatch.setitem(sys.modules, "vllm_neuron", package)
    milestones = _load(
        monkeypatch,
        "vllm_neuron.compile.trace_milestones",
        ROOT / "vllm_neuron/compile/trace_milestones.py",
    )

    milestones.emit_trace_milestone(
        "job_started", parent_rank=7, stage="preflight", job_index=1
    )
    milestones.emit_trace_milestone(
        "job_completed", parent_rank=7, stage="preflight", job_index=1
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "rank-7.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "job_started",
        "job_completed",
    ]
    assert all(record["schema_version"] == 1 for record in records)
