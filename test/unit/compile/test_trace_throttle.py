"""Process-level tests for the container-wide trace throttle."""

import importlib.util
import multiprocessing
import os
import signal
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]


def _load_trace_throttle():
    name = "trace_throttle_under_test"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "vllm_neuron/compile/trace_throttle.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timed_holder(root: str, events, hold_seconds: float) -> None:
    throttle = _load_trace_throttle()
    with throttle.host_trace_slot(
        2,
        parent_rank=0,
        lane_idx=0,
        root=Path(root),
    ):
        events.put(("enter", os.getpid()))
        time.sleep(hold_seconds)
        events.put(("exit", os.getpid()))


def _signal_holder(root: str, events) -> None:
    throttle = _load_trace_throttle()
    with throttle.host_trace_slot(
        1,
        parent_rank=0,
        lane_idx=0,
        root=Path(root),
    ):
        events.put(("enter", os.getpid()))
        signal.pause()


def _one_shot_holder(root: str, events) -> None:
    throttle = _load_trace_throttle()
    with throttle.host_trace_slot(
        1,
        parent_rank=0,
        lane_idx=0,
        root=Path(root),
    ):
        events.put(("enter", os.getpid()))


def test_unset_throttle_is_a_noop(tmp_path):
    throttle = _load_trace_throttle()
    unused_root = tmp_path / "unused"
    with throttle.host_trace_slot(
        None,
        parent_rank=3,
        lane_idx=1,
        root=unused_root,
    ) as slot:
        assert slot is None
    assert not unused_root.exists()


@pytest.mark.skipif(os.name != "posix", reason="requires Linux flock")
def test_host_wide_limit_bounds_concurrent_children(tmp_path):
    ctx = multiprocessing.get_context("fork")
    events = ctx.Queue()
    children = [
        ctx.Process(target=_timed_holder, args=(str(tmp_path), events, 0.15))
        for _ in range(6)
    ]
    for child in children:
        child.start()

    active: set[int] = set()
    max_active = 0
    for _ in range(12):
        event, pid = events.get(timeout=10)
        if event == "enter":
            active.add(pid)
            max_active = max(max_active, len(active))
        else:
            active.remove(pid)

    for child in children:
        child.join(timeout=10)
        assert child.exitcode == 0
    assert not active
    assert max_active <= 2


@pytest.mark.skipif(os.name != "posix", reason="requires Linux flock")
def test_exception_releases_slot(tmp_path):
    throttle = _load_trace_throttle()
    with (
        pytest.raises(RuntimeError, match="expected failure"),
        throttle.host_trace_slot(
            1,
            parent_rank=0,
            lane_idx=0,
            root=tmp_path,
        ),
    ):
        raise RuntimeError("expected failure")

    with throttle.host_trace_slot(
        1,
        parent_rank=1,
        lane_idx=0,
        root=tmp_path,
    ) as slot:
        assert slot == 0


@pytest.mark.skipif(os.name != "posix", reason="requires Linux flock")
def test_sigkill_releases_slot(tmp_path):
    ctx = multiprocessing.get_context("fork")
    events = ctx.Queue()
    holder = ctx.Process(target=_signal_holder, args=(str(tmp_path), events))
    holder.start()
    event, pid = events.get(timeout=10)
    assert event == "enter"

    os.kill(pid, signal.SIGKILL)
    holder.join(timeout=10)
    assert holder.exitcode == -signal.SIGKILL

    successor = ctx.Process(target=_one_shot_holder, args=(str(tmp_path), events))
    successor.start()
    assert events.get(timeout=10)[0] == "enter"
    successor.join(timeout=10)
    assert successor.exitcode == 0


@pytest.mark.skipif(os.name != "posix", reason="requires Linux flock")
def test_live_limit_mismatch_fails_closed(tmp_path):
    ctx = multiprocessing.get_context("fork")
    events = ctx.Queue()
    holder = ctx.Process(target=_signal_holder, args=(str(tmp_path), events))
    holder.start()
    assert events.get(timeout=10)[0] == "enter"

    throttle = _load_trace_throttle()
    with (
        pytest.raises(RuntimeError, match="limit mismatch"),
        throttle.host_trace_slot(
            2,
            parent_rank=1,
            lane_idx=0,
            root=tmp_path,
        ),
    ):
        pass

    holder.terminate()
    holder.join(timeout=10)
    assert holder.exitcode is not None
