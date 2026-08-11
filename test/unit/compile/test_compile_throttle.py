# SPDX-License-Identifier: Apache-2.0
"""Process-level tests for the host-wide neuronx-cc concurrency bound."""

import importlib.util
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]


def _load_compile_throttle():
    name = "compile_throttle_under_test"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "vllm_neuron/compile/compile_throttle.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_compiler_job(root: str, job_idx: int, events) -> None:
    """One dummy 'compiler': hold a global slot while a sleep stand-in runs."""
    throttle = _load_compile_throttle()
    with throttle.global_compile_slot(cap=2, sem_dir=root, timeout_s=120.0):
        events.put(("enter", job_idx, os.getpid(), time.monotonic()))
        try:
            # Stand-in for the neuronx-cc subprocess spawn.
            subprocess.run([sys.executable, "-c", "import time; time.sleep(0.4)"])
        finally:
            events.put(("exit", job_idx, os.getpid(), time.monotonic()))


def _rank_worker(root: str, job_indices, events) -> None:
    """A simulated rank: runs its assigned compiler jobs sequentially."""
    for job_idx in job_indices:
        _run_compiler_job(root, job_idx, events)


def test_cap_two_never_exceeded_across_three_ranks(tmp_path):
    """5 dummy compilers across 3 rank processes never exceed 2 concurrent."""
    ctx = multiprocessing.get_context("fork")
    events = ctx.Queue()
    root = str(tmp_path / "sem")
    rank_jobs = [[0, 1], [2, 3], [4]]
    ranks = [
        ctx.Process(target=_rank_worker, args=(root, jobs, events))
        for jobs in rank_jobs
    ]
    for proc in ranks:
        proc.start()
    for proc in ranks:
        proc.join(timeout=120)
        assert proc.exitcode == 0, f"rank process failed: exitcode={proc.exitcode}"

    intervals = {}
    pending = {}
    import queue

    while True:
        try:
            kind, job_idx, _pid, stamp = events.get(timeout=5)
        except queue.Empty:
            break
        if kind == "enter":
            pending[job_idx] = stamp
        else:
            intervals[job_idx] = (pending[job_idx], stamp)
    assert len(intervals) == 5, f"missing jobs: {sorted(pending)}"

    # Sweep-line: at no point in time may more than 2 holds overlap.
    points = sorted({t for iv in intervals.values() for t in iv})
    for i, start in enumerate(points):
        for end in points[i + 1 :] or [start]:
            mid = (start + end) / 2
            overlap = sum(1 for a, b in intervals.values() if a <= mid < b)
            assert overlap <= 2, f"{overlap} concurrent holds at t={mid}"


def _slot_holder(root: str, ready, hold: float) -> None:
    throttle = _load_compile_throttle()
    with throttle.global_compile_slot(cap=1, sem_dir=root, timeout_s=30.0):
        ready.put(os.getpid())
        time.sleep(hold)


def test_slot_released_on_sigkill(tmp_path):
    """SIGKILLing a slot holder frees the lease (flock drops on fd close)."""
    ctx = multiprocessing.get_context("fork")
    root = str(tmp_path / "sem")
    ready = ctx.Queue()
    holder = ctx.Process(target=_slot_holder, args=(root, ready, 60.0))
    holder.start()
    holder_pid = ready.get(timeout=30)
    os.kill(holder_pid, signal.SIGKILL)
    holder.join(timeout=30)

    throttle = _load_compile_throttle()
    start = time.monotonic()
    with throttle.global_compile_slot(cap=1, sem_dir=root, timeout_s=10.0):
        pass
    assert time.monotonic() - start < 10.0, "slot leaked after holder SIGKILL"


def test_disabled_mode_touches_no_filesystem(tmp_path):
    """cap=None/0 is a no-op: no slot dir is created, body runs unchanged."""
    throttle = _load_compile_throttle()
    missing = tmp_path / "must-not-be-created"
    for disabled_cap in (None, 0):
        with throttle.global_compile_slot(
            cap=disabled_cap, sem_dir=str(missing), timeout_s=1.0
        ) as slot:
            assert slot is None
        assert not missing.exists(), f"disabled cap={disabled_cap} created dir"


def test_acquire_timeout_and_wait_logging(tmp_path, caplog):
    """A fully-leased semaphore raises TimeoutError and logs the wait."""
    ctx = multiprocessing.get_context("fork")
    root = str(tmp_path / "sem")
    ready = ctx.Queue()
    holder = ctx.Process(target=_slot_holder, args=(root, ready, 30.0))
    holder.start()
    ready.get(timeout=30)
    try:
        throttle = _load_compile_throttle()
        with caplog.at_level("INFO"):
            with pytest.raises(TimeoutError, match="global compile slot"):
                with throttle.global_compile_slot(
                    cap=1, sem_dir=root, timeout_s=1.0, heartbeat_s=0.3
                ):
                    pass
        assert any(
            "Waiting" in record.message and "global compile slot" in record.message
            for record in caplog.records
        ), f"no wait heartbeat logged: {[r.message for r in caplog.records]}"
    finally:
        holder.terminate()
        holder.join(timeout=30)


def test_parse_max_global_values():
    throttle = _load_compile_throttle()
    assert throttle.parse_max_global(None) is None
    assert throttle.parse_max_global("") is None
    assert throttle.parse_max_global("0") is None
    assert throttle.parse_max_global("16") == 16
    assert throttle.parse_max_global("24") == 24
    for bad in ("1", "15", "25", "-1", "1.5", "lots"):
        with pytest.raises(ValueError, match="VLLM_NEURON_COMPILE_MAX_GLOBAL"):
            throttle.parse_max_global(bad)


def test_sem_dir_resolution(monkeypatch, tmp_path):
    throttle = _load_compile_throttle()
    monkeypatch.setenv("VLLM_NEURON_COMPILE_SEM_DIR", str(tmp_path / "explicit"))
    assert throttle.resolve_sem_dir() == tmp_path / "explicit"
    monkeypatch.delenv("VLLM_NEURON_COMPILE_SEM_DIR")
    monkeypatch.setenv("NEURON_COMPILED_ARTIFACTS", str(tmp_path / "artifacts"))
    assert throttle.resolve_sem_dir() == tmp_path / "artifacts" / "compile-sem"
    monkeypatch.delenv("NEURON_COMPILED_ARTIFACTS")
    # No config anywhere: falls back to temp dir (warning path).
    assert "vllm-neuron-compile-sem" in str(throttle.resolve_sem_dir())
    # Explicit argument always wins.
    assert throttle.resolve_sem_dir("/some/where") == Path("/some/where")
