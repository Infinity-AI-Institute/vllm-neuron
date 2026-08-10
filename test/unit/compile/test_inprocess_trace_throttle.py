"""Runtime tests for throttled in-process graph extraction."""

import ast
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]
WORKER_PATH = ROOT / "vllm_neuron/vllm/worker/neuron_worker.py"


def _load_worker_harness():
    """Load only the worker methods under test without importing vLLM."""
    tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"))
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NeuronWorker"
    )
    selected = {
        "_run_parallel_trace_jobs",
        "_extract_graphs",
        "_extract_prefill_graphs_sequential",
        "_extract_decode_graphs_sequential",
        "_extract_vision_graphs",
    }
    harness = ast.ClassDef(
        name="WorkerHarness",
        bases=[],
        keywords=[],
        body=[
            node
            for node in worker.body
            if isinstance(node, ast.FunctionDef) and node.name in selected
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    namespace = {
        "envs": SimpleNamespace(
            VLLM_NEURON_DISABLE_PARALLEL_TRACE=True,
            VLLM_NEURON_TRACE_RANK_CONCURRENCY=2,
        ),
        "host_trace_slot": None,
        "logger": logging.getLogger(__name__),
        "math": __import__("math"),
        "torch": None,
    }
    exec(  # noqa: S102 - execute the reviewed method-only AST test harness.
        compile(module, str(WORKER_PATH), "exec"), namespace
    )
    return namespace["WorkerHarness"], namespace


def _install_no_parallel_trace(monkeypatch):
    module = ModuleType("vllm_neuron.compile.parallel_trace")

    def forbidden_parallel_trace(*_args, **_kwargs):
        raise AssertionError("parallel_trace must not run in in-process mode")

    module.parallel_trace = forbidden_parallel_trace
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        os,
        "fork",
        lambda: (_ for _ in ()).throw(
            AssertionError("os.fork must not run in in-process mode")
        ),
        raising=False,
    )


def test_inprocess_rank_limit_caps_target_extraction_without_fork(monkeypatch):
    harness_type, namespace = _load_worker_harness()
    _install_no_parallel_trace(monkeypatch)
    semaphore = threading.Semaphore(2)
    lock = threading.Lock()
    active = 0
    max_active = 0
    slot_calls = []

    @contextmanager
    def fake_slot(limit, *, parent_rank, lane_idx):
        nonlocal active, max_active
        assert limit == 2
        semaphore.acquire()
        with lock:
            active += 1
            max_active = max(max_active, active)
            slot_calls.append((limit, parent_rank, lane_idx))
        try:
            yield lane_idx
        finally:
            with lock:
                active -= 1
            semaphore.release()

    namespace["host_trace_slot"] = fake_slot

    def make_worker(rank):
        worker = harness_type()
        worker.rank = rank

        def extract_prefill(*_args):
            assert active > 0
            time.sleep(0.05)

        worker.model_runner = SimpleNamespace(
            drafter=None,
            extract_prefill_graphs=extract_prefill,
            extract_decode_graphs=lambda *_args, **_kwargs: None,
        )
        worker._prefill_compile_targets = lambda: [(128, 128)]
        worker._decode_compile_targets = list
        worker._build_vision_trace_jobs = list
        worker._build_prefill_trace_jobs = lambda _buckets: []
        worker._build_decode_trace_jobs = lambda _targets: []
        return worker

    workers = [make_worker(rank) for rank in range(4)]
    threads = [
        threading.Thread(
            target=worker._extract_graphs,
            kwargs={
                "skip_prefill": False,
                "skip_decode": True,
                "skip_vision": True,
            },
        )
        for worker in workers
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert max_active == 2
    assert active == 0
    assert sorted(rank for _, rank, _ in slot_calls) == [0, 1, 2, 3]


def test_inprocess_slot_is_released_when_extraction_raises(monkeypatch):
    harness_type, namespace = _load_worker_harness()
    _install_no_parallel_trace(monkeypatch)
    events = []

    @contextmanager
    def fake_slot(limit, *, parent_rank, lane_idx):
        events.append(("acquire", limit, parent_rank, lane_idx))
        try:
            yield lane_idx
        finally:
            events.append(("release", parent_rank, lane_idx))

    namespace["host_trace_slot"] = fake_slot
    worker = harness_type()
    worker.rank = 7
    worker.model_runner = SimpleNamespace(
        extract_decode_graphs=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("capture failed")
        )
    )

    with pytest.raises(RuntimeError, match="capture failed"):
        worker._extract_decode_graphs_sequential([(1, 512)])

    assert events == [
        ("acquire", 2, 7, 0),
        ("release", 7, 0),
    ]


def test_vision_inputs_are_built_before_the_slot_and_capture_runs_inside_it(
    monkeypatch,
):
    harness_type, namespace = _load_worker_harness()
    active = False
    events = []

    @contextmanager
    def fake_slot(limit, *, parent_rank, lane_idx):
        nonlocal active
        events.append(("acquire", limit, parent_rank, lane_idx))
        active = True
        try:
            yield lane_idx
        finally:
            active = False
            events.append(("release", parent_rank, lane_idx))

    class FakeTorch:
        int64 = "int64"

        @staticmethod
        def device(name):
            assert not active
            return name

        @staticmethod
        def empty(*_args, **_kwargs):
            assert not active
            return "buffer"

        @staticmethod
        def zeros(*_args, **_kwargs):
            assert not active
            return "ids"

    class CaptureComplete(Exception):
        pass

    capture_module = ModuleType("vllm_neuron.compile.capture_backend")
    capture_module.CaptureComplete = CaptureComplete
    monkeypatch.setitem(sys.modules, capture_module.__name__, capture_module)
    namespace["host_trace_slot"] = fake_slot
    namespace["torch"] = FakeTorch

    worker = harness_type()
    worker.rank = 9
    config = SimpleNamespace(
        num_vision_tokens_buckets=[16],
        vision_attention_block_size=8,
        dp_size=1,
    )
    cache = SimpleNamespace(num_blocks=2, block_size=4, fat_dim=8, dtype="bf16")

    def build_inputs(*_args):
        assert not active
        return {}

    def capture(**_kwargs):
        assert active
        events.append(("capture",))
        raise CaptureComplete

    worker.model_runner = SimpleNamespace(
        vision_neuron_config=config,
        vision_capture_backend=capture,
        encoder_cache=cache,
    )
    worker._unwrap_vision_model = lambda: SimpleNamespace(
        build_vision_synthetic_inputs=build_inputs
    )

    worker._extract_vision_graphs()

    assert events == [
        ("acquire", 2, 9, 0),
        ("capture",),
        ("release", 9, 0),
    ]
