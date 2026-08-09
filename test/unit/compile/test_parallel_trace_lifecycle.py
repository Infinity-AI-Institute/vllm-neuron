"""Trace-pool process-lifetime regression tests."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]


@pytest.fixture
def parallel_trace_module(monkeypatch):
    """Load the module without importing the vLLM platform plugin."""
    package = ModuleType("vllm_neuron")
    package.__path__ = [str(ROOT / "vllm_neuron")]
    package.envs = SimpleNamespace(VLLM_NEURON_PARALLEL_TRACE_WORKERS=1)
    monkeypatch.setitem(sys.modules, "vllm_neuron", package)

    compile_package = ModuleType("vllm_neuron.compile")
    compile_package.__path__ = [str(ROOT / "vllm_neuron/compile")]
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile", compile_package)

    name = "vllm_neuron.compile.parallel_trace"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "vllm_neuron/compile/parallel_trace.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _job() -> tuple[object, dict[str, object]]:
    return (object(), {})


def test_sequential_mode_uses_one_fully_waited_pool_per_job(
    monkeypatch,
    parallel_trace_module,
):
    jobs = [_job(), _job()]
    calls = []

    def fake_pool(
        pool_jobs,
        parent_rank,
        num_workers,
        trace_rank_concurrency=None,
    ):
        calls.append(
            (pool_jobs, parent_rank, num_workers, trace_rank_concurrency)
        )

    monkeypatch.setattr(parallel_trace_module, "_run_pool_fork", fake_pool)

    parallel_trace_module._run_fresh_children_sequentially(jobs, parent_rank=7)

    assert calls == [([jobs[0]], 7, 1, None), ([jobs[1]], 7, 1, None)]


def test_sequential_mode_stops_after_indexed_child_failure(
    monkeypatch,
    parallel_trace_module,
):
    jobs = [_job(), _job(), _job()]
    calls = []

    def fake_pool(
        pool_jobs,
        parent_rank,
        num_workers,
        trace_rank_concurrency=None,
    ):
        calls.append(
            (pool_jobs, parent_rank, num_workers, trace_rank_concurrency)
        )
        if pool_jobs == [jobs[1]]:
            raise RuntimeError("child failed")

    monkeypatch.setattr(parallel_trace_module, "_run_pool_fork", fake_pool)

    with pytest.raises(
        RuntimeError,
        match=r"job=2/3 parent_rank=11",
    ):
        parallel_trace_module._run_fresh_children_sequentially(
            jobs,
            parent_rank=11,
        )

    assert calls == [
        ([jobs[0]], 11, 1, None),
        ([jobs[1]], 11, 1, None),
    ]


def test_throttle_is_acquired_before_child_graph_setup(
    monkeypatch,
    parallel_trace_module,
):
    events = []

    @contextmanager
    def fake_slot(limit, *, parent_rank, lane_idx):
        events.append(("acquire", limit, parent_rank, lane_idx))
        yield 0
        events.append(("release",))

    def fake_child(lane_idx, parent_rank, jobs_slice, result_path):
        events.append(
            ("graph", lane_idx, parent_rank, jobs_slice, result_path)
        )

    monkeypatch.setattr(parallel_trace_module, "host_trace_slot", fake_slot)
    monkeypatch.setattr(parallel_trace_module, "_fork_child_main", fake_child)
    jobs = [_job()]

    parallel_trace_module._run_throttled_child(
        2,
        17,
        jobs,
        "/tmp/status",
        8,
    )

    assert events == [
        ("acquire", 8, 17, 2),
        ("graph", 2, 17, jobs, "/tmp/status"),
        ("release",),
    ]
