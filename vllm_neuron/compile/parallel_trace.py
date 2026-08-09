# SPDX-License-Identifier: Apache-2.0
"""Fork-based pool for parallelizing graph trace.

Public API
----------

::

    parallel_trace.parallel_trace(jobs, parent_rank=0) -> None

A *job* is a ``(callable, kwargs)`` pair: a forked child invokes
``callable(**kwargs)``. The "callable" is typically a torch.compile
wrapper (``capture_backend_model``) so the call drives an FX→HLO trace
into the cache.

Why fork
--------

Dynamo's frame-eval / guard / code cache and torch_xla's IR tape are
process-global, so trace can't be parallelized within a single process.
``os.fork()`` from the already-fully-initialized parent worker is the
cheapest way to get N independent Dynamo states: the child inherits the
parent's distributed state (rank lists, group registry, vllm_config,
loaded model), so capture-side code looks up rank metadata and module
weights for free. The child only does FX→HLO lowering on synthetic
meta-device inputs — no real gloo collectives are executed, so the
inherited gloo socket is never touched.

Meta swap
---------

After fork, the parent's NRT runtime enters ``NRT_STATE_CHILD`` and
refuses any allocation or deallocation, so the inherited model — whose
parameters live on the neuron device — is unusable. Each child swaps
the *unique set* of underlying nn.Modules across its assigned jobs to
the meta device before running anything. KV caches that are different
views of one underlying buffer (e.g. ``typed_tensor[0]`` /
``typed_tensor[1]`` from ``initialize_kv_cache``) are remapped to a
single shared meta storage, so the capture backend's input-dedup pass
still collapses them to one FX placeholder rather than emitting two
independent ones.

Disable
-------

Set ``VLLM_NEURON_DISABLE_PARALLEL_TRACE=1`` to skip the fork pool and
run jobs sequentially in the parent process. Setting
``VLLM_NEURON_PARALLEL_TRACE_WORKERS=1`` runs every job serially in a
distinct, fully reaped child. This bounds peak trace memory to one graph
shape without depending on process-global cache or allocator cleanup.

Container-wide throttle
-----------------------

Set ``VLLM_NEURON_TRACE_RANK_CONCURRENCY=N`` to let all rank parents enter
the trace pool while allowing at most N fork children in the container to
construct graphs concurrently. Waiting children acquire a kernel-owned slot
before capture imports or meta-swapping, minimizing additional copy-on-write
memory. Unset preserves the existing behavior.
"""

import dataclasses
import functools
import inspect
import logging
import os
import shutil
import signal
import tempfile
import threading
import time
import traceback
import types
import weakref
from collections.abc import Callable
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import torch

from vllm_neuron import envs
from vllm_neuron.compile.trace_milestones import emit_trace_milestone
from vllm_neuron.compile.trace_throttle import host_trace_slot

logger = logging.getLogger(__name__)


# A trace job: a callable plus the kwargs that drive its forward pass.
Job = tuple[Callable[..., Any], dict[str, Any]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parallel_trace_with_preflight(jobs: list[Job], parent_rank: int = 0) -> None:
    """Optionally stage Python/FakeTensor tracing on one global rank.

    The representative runs in the ordinary fork pool, but its capture backend
    exits immediately after Dynamo produces the FX graph. No FX passes, HLO, or
    cache write occurs. Its fork child is then reaped. A small CPU object
    broadcast releases peers on success or propagates the captured failure.
    Every rank, including the representative, subsequently executes the normal
    trace path; no graph is shared or reused by this protocol.
    """
    representative = envs.VLLM_NEURON_TRACE_PREFLIGHT_RANK
    if representative is None:
        parallel_trace(jobs, parent_rank=parent_rank)
        return

    dist = torch.distributed
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else parent_rank
    world_size = dist.get_world_size() if distributed else 1
    if representative >= world_size:
        raise ValueError(
            "VLLM_NEURON_TRACE_PREFLIGHT_RANK is outside the distributed "
            f"world: rank={representative}, world_size={world_size}"
        )

    limit = envs.VLLM_NEURON_TRACE_PREFLIGHT_JOBS
    staged_jobs = jobs if limit is None else jobs[:limit]
    payload: list[dict[str, Any] | None] = [None]
    emit_trace_milestone(
        "preflight_waiting" if rank != representative else "preflight_selected",
        parent_rank=parent_rank,
        stage="preflight",
        representative_rank=representative,
        staged_jobs=len(staged_jobs),
        total_jobs=len(jobs),
    )

    preflight_group = None
    timeout_seconds = envs.VLLM_NEURON_TRACE_PREFLIGHT_TIMEOUT_SECONDS
    heartbeat_seconds = envs.VLLM_NEURON_TRACE_PREFLIGHT_HEARTBEAT_SECONDS
    if heartbeat_seconds >= timeout_seconds:
        raise ValueError(
            "VLLM_NEURON_TRACE_PREFLIGHT_HEARTBEAT_SECONDS must be less than "
            "VLLM_NEURON_TRACE_PREFLIGHT_TIMEOUT_SECONDS: "
            f"heartbeat={heartbeat_seconds}, timeout={timeout_seconds}"
        )
    if distributed:
        # Every rank must create this group before the representative starts
        # tracing. Creating it after the trace would leave parked ranks blocked
        # in new_group() under the default group's shorter timeout.
        preflight_group = dist.new_group(
            ranks=list(range(world_size)),
            timeout=timedelta(seconds=timeout_seconds),
        )
        emit_trace_milestone(
            "preflight_control_group_ready",
            parent_rank=parent_rank,
            stage="preflight",
            representative_rank=representative,
            timeout_seconds=timeout_seconds,
            heartbeat_seconds=heartbeat_seconds,
        )

    if rank == representative:
        try:
            with _preflight_child_mode():
                parallel_trace(staged_jobs, parent_rank=parent_rank)
            payload[0] = {
                "ok": True,
                "representative_rank": representative,
                "staged_jobs": len(staged_jobs),
            }
        except Exception as exc:
            payload[0] = {
                "ok": False,
                "representative_rank": representative,
                "staged_jobs": len(staged_jobs),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc()[-8192:],
            }

    if distributed:
        try:
            with _preflight_wait_heartbeat(
                parent_rank=parent_rank,
                representative_rank=representative,
                timeout_seconds=timeout_seconds,
                heartbeat_seconds=heartbeat_seconds,
            ):
                dist.broadcast_object_list(
                    payload,
                    src=representative,
                    group=preflight_group,
                    device=torch.device("cpu"),
                )
        except Exception as exc:
            emit_trace_milestone(
                "preflight_rendezvous_failed",
                parent_rank=parent_rank,
                stage="preflight",
                representative_rank=representative,
                timeout_seconds=timeout_seconds,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise RuntimeError(
                "Representative trace preflight control rendezvous failed "
                f"within its dedicated {timeout_seconds}s deadline"
            ) from exc
        finally:
            if preflight_group is not None:
                try:
                    dist.destroy_process_group(preflight_group)
                except Exception:
                    logger.warning(
                        "Unable to destroy preflight-only process group",
                        exc_info=True,
                    )

    result = payload[0]
    if not isinstance(result, dict):
        raise RuntimeError("Representative trace preflight returned no result")
    if not result.get("ok"):
        emit_trace_milestone(
            "preflight_failed",
            parent_rank=parent_rank,
            stage="preflight",
            representative_rank=representative,
            error_type=result.get("error_type"),
            error_message=result.get("error_message"),
        )
        raise RuntimeError(
            "Representative trace preflight failed before all-rank extraction: "
            f"rank={representative} type={result.get('error_type')} "
            f"message={result.get('error_message')}\n{result.get('traceback', '')}"
        )

    emit_trace_milestone(
        "preflight_released",
        parent_rank=parent_rank,
        stage="preflight",
        representative_rank=representative,
        staged_jobs=result.get("staged_jobs"),
    )
    parallel_trace(jobs, parent_rank=parent_rank)


@contextmanager
def _preflight_wait_heartbeat(
    *,
    parent_rank: int,
    representative_rank: int,
    timeout_seconds: int,
    heartbeat_seconds: int,
):
    """Emit progress without extending the preflight rendezvous deadline."""
    stopped = threading.Event()
    started = time.monotonic()

    def emit_heartbeats() -> None:
        while not stopped.wait(heartbeat_seconds):
            emit_trace_milestone(
                "preflight_wait_heartbeat",
                parent_rank=parent_rank,
                stage="preflight",
                representative_rank=representative_rank,
                elapsed_seconds=round(time.monotonic() - started, 3),
                timeout_seconds=timeout_seconds,
            )

    thread = threading.Thread(
        target=emit_heartbeats,
        name=f"preflight-heartbeat-rank-{parent_rank}",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=1.0)


@contextmanager
def _preflight_child_mode():
    """Temporarily mark only fork descendants as backend-boundary probes."""
    name = "VLLM_NEURON_TRACE_PREFLIGHT_ONLY"
    previous = os.environ.get(name)
    os.environ[name] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def parallel_trace(jobs: list[Job], parent_rank: int = 0) -> None:
    """Run each ``(callable, kwargs)`` job in a forked child.

    Args:
        jobs: list of ``(callable, kwargs)``. Inside each child the call
            is ``callable(**kwargs)``. All tensors in ``kwargs`` must be
            on the meta device — they're traced, not executed.
        parent_rank: engine-local rank, used to namespace status files
            and tag log records.

    The pool size is ``VLLM_NEURON_PARALLEL_TRACE_WORKERS``
    (default ``8``), capped at ``len(jobs)``; setting it to 1 runs each
    job in a distinct, fully reaped child. Set
    ``VLLM_NEURON_DISABLE_PARALLEL_TRACE=1`` to bypass the pool entirely
    and run jobs in the parent process.

    Raises:
        ValueError: if any job's kwargs include a non-meta tensor.
        RuntimeError: if any forked child fails. Other children are
            still waited on first so we don't leak processes.
    """
    if not jobs:
        return

    _validate_jobs_on_meta(jobs)

    num_workers = min(envs.VLLM_NEURON_PARALLEL_TRACE_WORKERS, len(jobs))
    trace_rank_concurrency = envs.VLLM_NEURON_TRACE_RANK_CONCURRENCY
    stage = "preflight" if envs.VLLM_NEURON_TRACE_PREFLIGHT_ONLY else "normal"
    logger.info(
        "Parallel trace: jobs=%d, lanes=%d, parent_rank=%d, host_limit=%s",
        len(jobs),
        num_workers,
        parent_rank,
        trace_rank_concurrency,
    )
    emit_trace_milestone(
        "pool_started",
        parent_rank=parent_rank,
        stage=stage,
        jobs=len(jobs),
        lanes=num_workers,
        host_limit=trace_rank_concurrency,
    )
    t0 = time.perf_counter()
    try:
        if num_workers == 1:
            _run_fresh_children_sequentially(jobs, parent_rank, trace_rank_concurrency)
        else:
            _run_pool_fork(jobs, parent_rank, num_workers, trace_rank_concurrency)
    except Exception as exc:
        emit_trace_milestone(
            "pool_failed",
            parent_rank=parent_rank,
            stage=stage,
            elapsed_seconds=time.perf_counter() - t0,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise
    elapsed = time.perf_counter() - t0
    emit_trace_milestone(
        "pool_completed",
        parent_rank=parent_rank,
        stage=stage,
        jobs=len(jobs),
        lanes=num_workers,
        elapsed_seconds=elapsed,
    )
    logger.info(
        "Parallel trace finished: jobs=%d, lanes=%d, %.2fs",
        len(jobs),
        num_workers,
        elapsed,
    )


def _run_fresh_children_sequentially(
    jobs: list[Job],
    parent_rank: int,
    trace_rank_concurrency: int | None = None,
) -> None:
    """Run every trace job in a distinct, fully reaped child process.

    ``_run_pool_fork`` waits for and reaps its child before returning. Calling
    it once per job establishes a strict process-lifetime boundary between
    graph shapes, releasing Dynamo, FX, XLA, Python, and allocator state.
    """
    for job_idx, job in enumerate(jobs):
        logger.info(
            "Sequential fresh trace child: job=%d/%d parent_rank=%d",
            job_idx + 1,
            len(jobs),
            parent_rank,
        )
        try:
            _run_pool_fork(
                [job],
                parent_rank,
                num_workers=1,
                trace_rank_concurrency=trace_rank_concurrency,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Sequential fresh trace child failed: "
                f"job={job_idx + 1}/{len(jobs)} parent_rank={parent_rank}"
            ) from exc


# ---------------------------------------------------------------------------
# Pool driver
# ---------------------------------------------------------------------------


def _run_pool_fork(
    jobs: list[Job],
    parent_rank: int,
    num_workers: int,
    trace_rank_concurrency: int | None = None,
) -> None:
    """Fork one child per non-empty lane. Each child runs all jobs
    assigned to its lane (in order) and exits. Parent ``waitpid``s and
    reads each child's status file to detect failures.

    Forking once per lane (rather than once per job) amortizes the
    meta-swap cost across the lane's jobs.
    """
    if not jobs:
        return

    lanes = _partition_round_robin(jobs, num_workers)
    workdir = tempfile.mkdtemp(prefix=f"trace_pool_rank{parent_rank}_")
    try:
        child_pids: dict[int, int] = {}
        result_paths: dict[int, str] = {}
        for lane_idx, lane_jobs in enumerate(lanes):
            if not lane_jobs:
                continue
            rp = os.path.join(workdir, f"lane{lane_idx}.status")
            result_paths[lane_idx] = rp
            pid = os.fork()
            if pid == 0:
                # Child: run target and exit. Use os._exit to skip
                # atexit handlers (which would otherwise try to clean
                # up parent state we still want).
                try:
                    _run_throttled_child(
                        lane_idx,
                        parent_rank,
                        lane_jobs,
                        rp,
                        trace_rank_concurrency,
                    )
                    os._exit(0)
                except BaseException:
                    try:
                        with open(rp, "w") as f:
                            f.write("ERROR\n" + traceback.format_exc())
                    except Exception:
                        pass
                    os._exit(1)
            else:
                child_pids[lane_idx] = pid

        # Poll our own lane PIDs so we can early-abort surviving lanes
        # the moment one fails.
        pending = dict(child_pids)  # lane_idx -> pid
        completed: dict[int, tuple[int, int]] = {}  # lane_idx -> (pid, exit_code)
        first_failure: str | None = None

        def _reap(lane_idx: int, pid: int, status_word: int) -> None:
            nonlocal first_failure
            exit_code = os.WEXITSTATUS(status_word) if os.WIFEXITED(status_word) else -1
            completed[lane_idx] = (pid, exit_code)
            child_status, child_err = _read_status_file(result_paths[lane_idx])
            if exit_code != 0 or child_status != "OK":
                msg = (
                    f"lane={lane_idx} pid={pid} exit_code={exit_code} "
                    f"status={child_status} err={child_err}"
                )
                if first_failure is None:
                    first_failure = msg
                    logger.error(
                        "Parallel trace lane failed; aborting siblings: %s", msg
                    )

        while pending:
            for lane_idx in list(pending):
                pid = pending[lane_idx]
                try:
                    result_pid, status_word = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    # Already reaped (shouldn't happen in normal flow,
                    # but treat as exit_code=-1 so we still surface it).
                    pending.pop(lane_idx, None)
                    completed[lane_idx] = (pid, -1)
                    continue
                if result_pid == 0:
                    continue  # still running
                pending.pop(lane_idx, None)
                _reap(lane_idx, pid, status_word)
            if first_failure is not None and pending:
                # Early abort: SIGTERM remaining children, give them a
                # short grace period to flush their status files, then
                # SIGKILL stragglers so the workdir cleanup can run.
                _abort_remaining(pending, completed, _reap)
                break
            if pending:
                time.sleep(0.1)

        if first_failure is not None:
            raise RuntimeError(
                f"Parallel trace fork failed (rank={parent_rank}): {first_failure}"
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


_ABORT_GRACE_PERIOD_S = 5.0
"""Time we give SIGTERM'd children to flush their status files before
escalating to SIGKILL. Tracing children spend most of their time inside
torch_xla / Dynamo C extensions that don't always honour SIGTERM
promptly, so escalation is needed to make progress on a real failure;
the grace period is long enough that a child mid-write has time to
finish."""


def _abort_remaining(
    pending: dict[int, int],
    completed: dict[int, tuple[int, int]],
    reap: Callable[[int, int, int], None],
) -> None:
    """Kill the still-running lane children after another lane failed.

    Sends SIGTERM, polls briefly so cooperating children can flush
    their status files, then SIGKILLs stragglers. Reaps every PID
    via the supplied ``reap`` callback so the parent doesn't leave
    zombies behind. Mutates ``pending`` in place.
    """
    for lane_idx, pid in list(pending.items()):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + _ABORT_GRACE_PERIOD_S
    while pending and time.monotonic() < deadline:
        for lane_idx in list(pending):
            pid = pending[lane_idx]
            try:
                result_pid, status_word = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pending.pop(lane_idx, None)
                completed[lane_idx] = (pid, -1)
                continue
            if result_pid == 0:
                continue
            pending.pop(lane_idx, None)
            reap(lane_idx, pid, status_word)
        if pending:
            time.sleep(0.05)

    # Stragglers — SIGKILL and reap synchronously. Blocking waitpid is
    # safe here: SIGKILL guarantees prompt exit, and we own the PID.
    for lane_idx, pid in list(pending.items()):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            _, status_word = os.waitpid(pid, 0)
        except ChildProcessError:
            pending.pop(lane_idx, None)
            completed[lane_idx] = (pid, -1)
            continue
        pending.pop(lane_idx, None)
        reap(lane_idx, pid, status_word)


def _partition_round_robin(items: list, n: int) -> list[list]:
    """Return n lanes; items round-robin so each lane runs ceil(len/n) jobs."""
    lanes: list[list] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        lanes[i % n].append(item)
    return lanes


# ---------------------------------------------------------------------------
# Per-lane child entrypoint + status I/O
# ---------------------------------------------------------------------------


def _run_throttled_child(
    lane_idx: int,
    parent_rank: int,
    jobs_slice: list[Job],
    result_path: str,
    trace_rank_concurrency: int | None,
) -> None:
    """Acquire the host slot before importing or mutating trace state."""
    stage = "preflight" if envs.VLLM_NEURON_TRACE_PREFLIGHT_ONLY else "normal"
    emit_trace_milestone(
        "host_slot_waiting",
        parent_rank=parent_rank,
        stage=stage,
        lane=lane_idx,
        host_limit=trace_rank_concurrency,
    )
    with host_trace_slot(
        trace_rank_concurrency,
        parent_rank=parent_rank,
        lane_idx=lane_idx,
    ) as slot:
        emit_trace_milestone(
            "host_slot_acquired",
            parent_rank=parent_rank,
            stage=stage,
            lane=lane_idx,
            slot=slot,
        )
        _fork_child_main(lane_idx, parent_rank, jobs_slice, result_path)
    emit_trace_milestone(
        "host_slot_released",
        parent_rank=parent_rank,
        stage=stage,
        lane=lane_idx,
        slot=slot,
    )


def _fork_child_main(
    lane_idx: int,
    parent_rank: int,
    jobs_slice: list[Job],
    result_path: str,
) -> None:
    """Run inside the forked child.

    1. Tag log records with [trace lane=N rank=R] so interleaved output
       is readable.
    2. Set ``VLLM_NEURON_CPU_COMPILE=1`` so the capture backend's
       device validator accepts meta inputs.
    3. Meta-swap the unique underlying nn.Modules referenced by this
       lane's jobs (once each, regardless of how many jobs reuse them).
    4. Run each job: ``callable(**kwargs)``. The capture backend raises
       ``CaptureComplete`` after writing the HLO — swallowed here.
    5. Write a status file the parent reads after waitpid.
    """
    from vllm_neuron.compile.capture_backend import CaptureComplete

    status = "OK"
    err: str | None = None
    failing_job: int | None = None
    stage = "preflight" if envs.VLLM_NEURON_TRACE_PREFLIGHT_ONLY else "normal"
    try:
        prefix = f"[trace lane={lane_idx} rank={parent_rank}] "

        class _Prefixer(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                if not getattr(record, "_trace_prefixed", False):
                    record.msg = prefix + str(record.msg)
                    record._trace_prefixed = True  # type: ignore[attr-defined]
                return True

        for h in logging.getLogger().handlers:
            h.addFilter(_Prefixer())

        os.environ["VLLM_NEURON_CPU_COMPILE"] = "1"

        # A fork inherits Dynamo's process-local code/guard caches and any
        # FakeTensorMode mappings populated by the parent.  Those mappings may
        # retain the pre-swap device tensors even when every model attribute is
        # subsequently replaced.  Reset before inspecting the callable roots
        # or entering Dynamo so the audit describes the state this child will
        # actually trace.
        torch.compiler.reset()

        emit_trace_milestone(
            "meta_swap_started",
            parent_rank=parent_rank,
            stage=stage,
            lane=lane_idx,
            jobs=len(jobs_slice),
        )
        _swap_unique_models_to_meta([model for model, _ in jobs_slice])
        _audit_jobs_on_meta(jobs_slice)
        emit_trace_milestone(
            "meta_swap_completed",
            parent_rank=parent_rank,
            stage=stage,
            lane=lane_idx,
            jobs=len(jobs_slice),
        )

        for j_idx, (model, kwargs) in enumerate(jobs_slice):
            failing_job = j_idx
            job_started = time.perf_counter()
            emit_trace_milestone(
                "job_started",
                parent_rank=parent_rank,
                stage=stage,
                lane=lane_idx,
                job_index=j_idx,
            )
            try:
                model(**kwargs)
            except CaptureComplete:
                # Successful trace — capture backend signals "done"
                # this way after writing the HLO.
                pass
            emit_trace_milestone(
                "job_completed",
                parent_rank=parent_rank,
                stage=stage,
                lane=lane_idx,
                job_index=j_idx,
                elapsed_seconds=time.perf_counter() - job_started,
            )
        failing_job = None
    except BaseException as e:
        status = "ERROR"
        emit_trace_milestone(
            "job_failed" if failing_job is not None else "child_failed",
            parent_rank=parent_rank,
            stage=stage,
            lane=lane_idx,
            job_index=failing_job,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        err = (
            f"job_index={failing_job}\n{e}\n{traceback.format_exc()}"
            if failing_job is not None
            else f"{e}\n{traceback.format_exc()}"
        )

    tmp_path = result_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(f"{status}\n")
        if err:
            f.write(err)
    os.rename(tmp_path, result_path)


def _read_status_file(path: str) -> tuple[str, str | None]:
    if not os.path.exists(path):
        return ("ERROR", "no status file written")
    with open(path) as f:
        content = f.read()
    if not content:
        return ("ERROR", "empty status file")
    lines = content.split("\n", 1)
    return (lines[0].strip(), lines[1] if len(lines) > 1 else None)


# ---------------------------------------------------------------------------
# Pre-fork input validation
# ---------------------------------------------------------------------------


def _validate_jobs_on_meta(jobs: list[Job]) -> None:
    """Fail fast if any kwargs tensor is not on the meta device.

    The capture backend would catch a non-meta input later
    (``compile/backend.py::_validate_inputs_on_device``), but only after
    fork — by which point a NRT_STATE_CHILD allocation will have already
    crashed the child. Raising here keeps the diagnostic local to the
    call site that built the inputs, and names the offending kwarg path
    so locating the leak doesn't require bisection.
    """
    for j_idx, (_, kwargs) in enumerate(jobs):
        for path, t in _walk_tensors(kwargs):
            if t.device.type != "meta":
                raise ValueError(
                    f"parallel_trace.parallel_trace: jobs[{j_idx}] kwarg "
                    f"{path!r} is on device {t.device} (expected meta) "
                    f"shape={tuple(t.shape)} dtype={t.dtype}. Build "
                    f"synthetic inputs with device='meta' before passing "
                    f"them to parallel_trace."
                )


def _walk_tensors(obj: Any, path: str = ""):
    """Yield ``(path, tensor)`` pairs for every ``torch.Tensor``
    reachable from ``obj`` via dict / list / tuple / dataclass-like
    attributes. ``path`` is a dotted/bracketed accessor ("a.b[3].c")
    so the validator's error message can name the offending field.
    """
    yield from _walk_tensors_seen(obj, path, set())


def _walk_tensors_seen(obj: Any, path: str, seen: set[int]):
    if isinstance(obj, torch.Tensor):
        yield path, obj
        return
    if isinstance(obj, torch.nn.Module):
        return

    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if isinstance(obj, dict):
        for k, v in obj.items():
            sep = "" if not path else "."
            yield from _walk_tensors_seen(v, f"{path}{sep}{k}", seen)
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_tensors_seen(v, f"{path}[{i}]", seen)
        return
    # Dataclass-like (e.g. AttentionMetadata). Skip primitives, modules,
    # and anything without a sensible __dict__.
    if hasattr(obj, "__dict__") and not isinstance(obj, torch.nn.Module):
        type_name = type(obj).__name__
        for k, v in vars(obj).items():
            sep = "" if not path else "."
            yield from _walk_tensors_seen(v, f"{path}{sep}<{type_name}>.{k}", seen)


@dataclasses.dataclass
class _TensorAuditRecord:
    """One tensor identity and every path by which the audit reached it."""

    tensor: torch.Tensor
    owner_paths: set[str] = dataclasses.field(default_factory=set)


def _tensor_alias_root(tensor: torch.Tensor) -> torch.Tensor:
    """Return the root of PyTorch's explicit ``Tensor._base`` view chain."""
    root = tensor
    seen: set[int] = set()
    while isinstance(getattr(root, "_base", None), torch.Tensor):
        if id(root) in seen:
            break
        seen.add(id(root))
        root = root._base
    return root


def _tensor_audit_metadata(tensor: torch.Tensor) -> str:
    """Format metadata that is safe to query without executing the tensor."""
    return (
        f"device={tensor.device} shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} stride={tuple(tensor.stride())} "
        f"storage_offset={tensor.storage_offset()} "
        f"identity=0x{id(tensor):x} "
        f"alias_root=0x{id(_tensor_alias_root(tensor)):x}"
    )


class _CallableTensorAudit:
    """Collect tensors reachable from trace callables without executing code.

    The traversal follows owned Python state and the exact callable references
    that affect invocation.  In particular, a function contributes only the
    globals/nonlocals named by its bytecode through ``inspect.getclosurevars``;
    its entire ``__globals__`` namespace is never scanned.  Imported module
    objects and classes are terminal leaves for the same reason.

    Each object's relative tensor descendants are memoized.  Reusing those
    suffixes under every incoming edge reports all finite alias paths without
    repeatedly expanding a large ``OptimizedModule`` object graph.  ``_active``
    terminates cycles; paths that repeatedly circle a cycle are intentionally
    omitted because that set is infinite and adds no ownership information.
    """

    def __init__(self) -> None:
        self.records: dict[int, _TensorAuditRecord] = {}
        self._active: set[int] = set()
        self._descendants: dict[
            int,
            tuple[Any, list[tuple[str, torch.Tensor]]],
        ] = {}

    def collect(self, obj: Any, path: str) -> None:
        for suffix, tensor in self._scan(obj):
            record = self.records.setdefault(id(tensor), _TensorAuditRecord(tensor))
            record.owner_paths.add(f"{path}{suffix}")

    def _scan(self, obj: Any) -> list[tuple[str, torch.Tensor]]:
        if isinstance(obj, torch.Tensor):
            return [("", obj)]
        if obj is None or isinstance(
            obj,
            (str, bytes, bytearray, int, float, complex, bool, type),
        ):
            return []
        # A module namespace can contain thousands of unrelated tensors and is
        # not an ownership edge from the callable.  Referenced values selected
        # by getclosurevars are inspected individually instead.
        if isinstance(obj, types.ModuleType):
            return []

        obj_id = id(obj)
        cached = self._descendants.get(obj_id)
        if cached is not None:
            return cached[1]
        if obj_id in self._active:
            return []
        self._active.add(obj_id)
        descendants: list[tuple[str, torch.Tensor]] = []
        try:
            if isinstance(obj, weakref.ReferenceType):
                resolved = obj()
                if resolved is not None:
                    self._extend(descendants, "()", resolved)
            elif isinstance(obj, functools.partial):
                self._extend(descendants, ".func", obj.func)
                self._extend(descendants, ".args", obj.args)
                self._extend(descendants, ".keywords", obj.keywords or {})
                self._scan_object_state(obj, descendants)
            elif inspect.ismethod(obj):
                self._extend(descendants, ".__self__", obj.__self__)
                self._extend(descendants, ".__func__", obj.__func__)
            elif inspect.isfunction(obj):
                self._scan_function_state(obj, descendants)
                self._scan_object_state(obj, descendants)
            elif isinstance(obj, dict):
                for index, (key, value) in enumerate(obj.items()):
                    self._extend(descendants, f".keys[{index}]", key)
                    self._extend(descendants, f"[{key!r}]", value)
            elif isinstance(obj, (list, tuple)):
                for index, value in enumerate(obj):
                    self._extend(descendants, f"[{index}]", value)
            elif isinstance(obj, (set, frozenset)):
                for index, value in enumerate(obj):
                    self._extend(descendants, f"[set:{index}]", value)
            else:
                # OptimizedModule owns the original nn.Module through _orig_mod.
                # nn.Module stores it inside _modules, while lightweight wrappers
                # may keep it directly in __dict__.  Visit either representation
                # explicitly so the diagnostic has the stable public owner path.
                object_state = getattr(obj, "__dict__", {})
                registered_modules = object_state.get("_modules", {})
                orig_mod = registered_modules.get("_orig_mod")
                if orig_mod is None:
                    orig_mod = inspect.getattr_static(obj, "_orig_mod", None)
                if orig_mod is not None:
                    self._extend(descendants, "._orig_mod", orig_mod)

                self._scan_dataclass_fields(obj, descendants)
                self._scan_object_state(
                    obj,
                    descendants,
                    skip_names={"_orig_mod"},
                )
                self._scan_slots(obj, descendants)
        finally:
            self._active.remove(obj_id)

        # Retain the object with the memo entry so a short-lived container's
        # Python id cannot be reused for a different object during this audit.
        self._descendants[obj_id] = (obj, descendants)
        return descendants

    def _extend(
        self,
        descendants: list[tuple[str, torch.Tensor]],
        edge: str,
        child: Any,
    ) -> None:
        descendants.extend(
            (f"{edge}{suffix}", tensor) for suffix, tensor in self._scan(child)
        )

    def _scan_function_state(
        self,
        function: Any,
        descendants: list[tuple[str, torch.Tensor]],
    ) -> None:
        closure = function.__closure__ or ()
        freevars = function.__code__.co_freevars
        for name, cell in zip(freevars, closure, strict=True):
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            self._extend(descendants, f".__closure__[{name!r}]", value)

        try:
            closure_vars = inspect.getclosurevars(function)
        except (TypeError, ValueError):
            return
        for name, value in closure_vars.nonlocals.items():
            self._extend(descendants, f".nonlocals[{name!r}]", value)
        for name, value in closure_vars.globals.items():
            self._extend(descendants, f".globals[{name!r}]", value)

    def _scan_dataclass_fields(
        self,
        obj: Any,
        descendants: list[tuple[str, torch.Tensor]],
    ) -> None:
        if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
            return
        for field in dataclasses.fields(obj):
            try:
                value = getattr(obj, field.name)
            except (AttributeError, RuntimeError):
                continue
            self._extend(descendants, f".{field.name}", value)

    def _scan_object_state(
        self,
        obj: Any,
        descendants: list[tuple[str, torch.Tensor]],
        *,
        skip_names: set[str] | None = None,
    ) -> None:
        try:
            state = vars(obj)
        except TypeError:
            return
        skipped = skip_names or set()
        for name, value in state.items():
            if name not in skipped:
                self._extend(descendants, f".{name}", value)

    def _scan_slots(
        self,
        obj: Any,
        descendants: list[tuple[str, torch.Tensor]],
    ) -> None:
        seen_names: set[str] = set()
        for cls in type(obj).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name in seen_names or name in ("__dict__", "__weakref__"):
                    continue
                seen_names.add(name)
                try:
                    value = getattr(obj, name)
                except (AttributeError, RuntimeError):
                    continue
                self._extend(descendants, f".{name}", value)


def _collect_job_tensor_owners(jobs: list[Job]) -> list[_TensorAuditRecord]:
    """Return tensor records reachable from every callable and kwarg root."""
    audit = _CallableTensorAudit()
    for job_index, (model, kwargs) in enumerate(jobs):
        audit.collect(model, f"jobs[{job_index}].callable")
        audit.collect(kwargs, f"jobs[{job_index}].kwargs")
    return list(audit.records.values())


def _audit_jobs_on_meta(jobs: list[Job]) -> None:
    """Fail before Dynamo if any callable-owned tensor is not meta.

    Unlike the parent-side kwargs validator, this child-side audit runs after
    the model swap and includes wrappers, bound callables, closures, referenced
    globals/nonlocals, partial arguments, dataclasses, slots, weak references,
    and ordinary object state.  Every owner path for one tensor identity is
    emitted together so a retained live alias can be fixed at its real source.
    """
    offenders = [
        record
        for record in _collect_job_tensor_owners(jobs)
        if record.tensor.device.type != "meta"
    ]
    if not offenders:
        return

    lines = [
        (
            "parallel trace pre-Dynamo tensor audit found "
            f"{len(offenders)} unexplained non-meta tensor identity(s):"
        )
    ]
    for record in sorted(offenders, key=lambda item: id(item.tensor)):
        lines.append(f"- {_tensor_audit_metadata(record.tensor)}")
        lines.extend(f"  owner={owner}" for owner in sorted(record.owner_paths))
    raise ValueError("\n".join(lines))


def _validate_models_on_meta(models: list[Any]) -> None:
    """Reject live device tensors still reachable from trace models.

    This runs immediately after the child meta swap and before Dynamo starts.
    The capture backend performs a similar device check much later, after the
    model has been Python-expanded. Naming the owner path here turns a late,
    expensive backend failure into an immediate diagnostic.
    """
    seen_modules: set[int] = set()
    for model_index, model in enumerate(models):
        underlying = _underlying_module(model)
        if not isinstance(underlying, torch.nn.Module):
            continue
        if id(underlying) in seen_modules:
            continue
        seen_modules.add(id(underlying))

        for module_path, submod in underlying.named_modules():
            owner = f"models[{model_index}]"
            if module_path:
                owner += f".{module_path}"
            for name, param in submod.named_parameters(recurse=False):
                if param.device.type != "meta":
                    raise ValueError(
                        f"parallel trace model tensor {owner}.{name} remains "
                        f"on {param.device} after meta swap "
                        f"shape={tuple(param.shape)} dtype={param.dtype}"
                    )
            for name, buffer in submod.named_buffers(recurse=False):
                if buffer.device.type != "meta":
                    raise ValueError(
                        f"parallel trace model tensor {owner}.{name} remains "
                        f"on {buffer.device} after meta swap "
                        f"shape={tuple(buffer.shape)} dtype={buffer.dtype}"
                    )
            for name, value in vars(submod).items():
                if name in ("_parameters", "_buffers", "_modules"):
                    continue
                for nested_path, tensor in _walk_tensors(value, f"{owner}.{name}"):
                    if tensor.device.type != "meta":
                        raise ValueError(
                            f"parallel trace model tensor {nested_path} remains "
                            f"on {tensor.device} after meta swap "
                            f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                        )


# ---------------------------------------------------------------------------
# Meta swap — move inherited models to meta inside the child without
# freeing the parent's neuron-device storage (NRT_STATE_CHILD forbids
# deallocations).
# ---------------------------------------------------------------------------


def _swap_unique_models_to_meta(models: list[Any]) -> None:
    """Apply ``_swap_to_meta_no_free`` to each unique underlying nn.Module
    referenced by ``models``. Sibling torch.compile wrappers sharing the
    same ``_orig_mod`` get swapped exactly once.
    """
    seen: set[int] = set()
    for m in models:
        underlying = _underlying_module(m)
        if underlying is None or id(underlying) in seen:
            continue
        if not isinstance(underlying, torch.nn.Module):
            continue
        seen.add(id(underlying))
        _swap_to_meta_no_free(underlying)


def _swap_to_meta_no_free(module: torch.nn.Module) -> None:
    """Replace every parameter, buffer, and tensor-valued attribute in
    ``module`` (recursive) with a meta-device counterpart, *without
    freeing* the original neuron storage.

    ``nn.Module.to("meta")`` would re-assign each parameter via ``_apply``,
    which decrements the refcount of the old neuron-device tensor. When
    that refcount hits zero, ``nrt_tensor_free`` runs — and fails in
    NRT_STATE_CHILD. We keep the originals alive in a module-level list
    so destructors don't fire until child exit.

    Beyond parameters/buffers, this also swaps tensors held by plain
    attributes, including tensors nested in dict/list/tuple containers.
    K3 binds its paged MLA caches this way (for example ``_kv_caches``),
    and those caches live on the runtime device.

    Alias preservation: when two attribute slots are views descended from
    the same source tensor (the typical KV cache pattern:
    ``typed_tensor[0]`` / ``typed_tensor[1]`` both come from a shared raw
    buffer in ``initialize_kv_cache``), the meta replacements must also share
    storage. Neuron/XLA tensors expose one placeholder StorageImpl across
    independent lazy allocations, so neither ``id(untyped_storage())`` nor
    ``untyped_storage()._cdata`` can identify allocation ownership. We instead
    walk PyTorch's explicit ``Tensor._base`` view chain and key on the root
    tensor object. Independent tensors stay independent; real views and exact
    Python aliases stay shared. The capture backend's input dedup
    (``compile/backend.py::_detect_duplicate_inputs``) keys on
    ``(alias_root, storage_offset, shape, stride, dtype)``,
    so any two slots with that same key collapse to a single FX
    placeholder. If we were to allocate a fresh meta storage per slot,
    those keys would diverge — placeholder count balloons (e.g. KV
    caches go from 12 → 24 in GPT-OSS-20B), HBM usage doubles, and the
    HLO verifier rejects the graph.

    The ``storage_to_meta`` cache below maps each unique source alias root
    to a single meta storage; per-slot meta tensors are then constructed
    as views over that storage matching the source's offset / shape /
    stride / dtype. ``id_to_meta`` further dedupes by Python identity.
    """
    storage_to_meta: dict[int, torch.UntypedStorage] = {}
    meta_storage_to_source: dict[int, int] = {}
    id_to_meta: dict[int, torch.Tensor] = {}
    container_to_meta: dict[int, Any] = {}
    containers_in_progress: set[int] = set()

    def _replacement_for(src: torch.Tensor) -> torch.Tensor:
        cached = id_to_meta.get(id(src))
        if cached is not None:
            return cached
        storage_id = id(_tensor_alias_root(src))
        source_storage_nbytes = src.untyped_storage().nbytes()
        source_offset = src.storage_offset()
        source_shape = tuple(src.shape)
        source_stride = tuple(src.stride())
        if any(stride < 0 for stride in source_stride):
            raise ValueError(
                "cannot reconstruct unsupported negative-stride tensor view: "
                f"{_tensor_audit_metadata(src)}"
            )
        if src.numel() == 0:
            if (
                source_offset < 0
                or source_offset * src.element_size() > source_storage_nbytes
            ):
                raise ValueError(
                    "cannot reconstruct empty tensor view outside source storage: "
                    f"{_tensor_audit_metadata(src)} "
                    f"storage_nbytes={source_storage_nbytes}"
                )
        else:
            minimum_index = source_offset + sum(
                (size - 1) * min(stride, 0)
                for size, stride in zip(source_shape, source_stride, strict=True)
            )
            maximum_index = source_offset + sum(
                (size - 1) * max(stride, 0)
                for size, stride in zip(source_shape, source_stride, strict=True)
            )
            if (
                minimum_index < 0
                or (maximum_index + 1) * src.element_size() > source_storage_nbytes
            ):
                raise ValueError(
                    "cannot reconstruct tensor view outside source storage: "
                    f"{_tensor_audit_metadata(src)} "
                    f"storage_nbytes={source_storage_nbytes} "
                    f"index_bounds=({minimum_index}, {maximum_index})"
                )
        meta_storage = storage_to_meta.get(storage_id)
        if meta_storage is None:
            meta_storage = torch.UntypedStorage(source_storage_nbytes, device="meta")
            storage_to_meta[storage_id] = meta_storage
            meta_storage_id = meta_storage._cdata
            prior_source = meta_storage_to_source.setdefault(
                meta_storage_id,
                storage_id,
            )
            if prior_source != storage_id:
                raise AssertionError(
                    "independent source alias roots received one meta storage: "
                    f"source_alias_root=0x{storage_id:x} "
                    f"prior_alias_root=0x{prior_source:x} "
                    f"meta_storage=0x{meta_storage_id:x}"
                )
        repl = torch.empty(0, dtype=src.dtype, device="meta").set_(
            meta_storage,
            source_offset,
            source_shape,
            source_stride,
        )
        parity = (
            tuple(repl.shape) == source_shape
            and repl.dtype == src.dtype
            and tuple(repl.stride()) == source_stride
            and repl.storage_offset() == source_offset
            and repl.untyped_storage().nbytes() == source_storage_nbytes
            and repl.untyped_storage()._cdata == meta_storage._cdata
        )
        if not parity:
            raise AssertionError(
                "meta tensor reconstruction changed view metadata: "
                f"source=({_tensor_audit_metadata(src)}) "
                f"replacement=({_tensor_audit_metadata(repl)}) "
                f"source_storage_nbytes={source_storage_nbytes} "
                f"replacement_storage_nbytes={repl.untyped_storage().nbytes()}"
            )
        id_to_meta[id(src)] = repl
        # Hold a strong reference to the source so its storage survives
        # until child exit (no nrt_tensor_free calls).
        _META_PARAM_KEEPALIVE.append(src)
        return repl

    def _replace_nested_tensors(value: Any) -> Any:
        """Replace tensors in supported attribute containers.

        Dicts and lists are updated in place so shared-container aliases and
        self-references remain intact. Tuples are rebuilt because they are
        immutable, with completed replacements memoized for repeated aliases.
        A pathological cycle that re-enters an in-progress tuple is left at
        that edge rather than recursing forever; ordinary mutable-container
        cycles are fully supported.
        """
        if isinstance(value, torch.Tensor):
            if value.device.type == "meta":
                return value
            return _replacement_for(value)
        if isinstance(value, torch.nn.Module):
            return value

        value_id = id(value)
        cached = container_to_meta.get(value_id)
        if cached is not None:
            return cached

        if isinstance(value, dict):
            container_to_meta[value_id] = value
            for key, item in list(value.items()):
                value[key] = _replace_nested_tensors(item)
            return value
        if isinstance(value, list):
            container_to_meta[value_id] = value
            for index, item in enumerate(value):
                value[index] = _replace_nested_tensors(item)
            return value
        if isinstance(value, tuple):
            if value_id in containers_in_progress:
                return value
            containers_in_progress.add(value_id)
            try:
                items = tuple(_replace_nested_tensors(item) for item in value)
                if hasattr(value, "_fields"):
                    replacement = type(value)(*items)
                else:
                    replacement = items
                container_to_meta[value_id] = replacement
                return replacement
            finally:
                containers_in_progress.remove(value_id)
        return value

    for submod in module.modules():
        for name, param in list(submod._parameters.items()):
            if param is None or param.device.type == "meta":
                continue
            meta_t = _replacement_for(param)
            submod._parameters[name] = torch.nn.Parameter(
                meta_t, requires_grad=param.requires_grad
            )
        for name, buf in list(submod._buffers.items()):
            if buf is None or buf.device.type == "meta":
                continue
            submod._buffers[name] = _replacement_for(buf)
        # Plain tensor attributes and supported nested containers (e.g.
        # K3's _kv_caches dict bound after initialize_kv_cache). Skip the
        # special _parameters / _buffers / _modules dicts already handled.
        for name, val in list(submod.__dict__.items()):
            if name in ("_parameters", "_buffers", "_modules"):
                continue
            submod.__dict__[name] = _replace_nested_tensors(val)


_META_PARAM_KEEPALIVE: list = []
"""Holds references to the original neuron parameters/buffers we replaced
during the in-child meta swap. The Python destructor for a neuron
tensor calls ``nrt_tensor_free``, which fails in NRT_STATE_CHILD. By
keeping a strong reference, we defer the free until child exit (where
the OS reaps process memory directly without going through NRT)."""


def _underlying_module(model: Any) -> Any:
    """Return the underlying nn.Module of a torch.compile-wrapped model.

    OptimizedModule keeps a reference to the original module under
    ``_orig_mod``. Mutations on that propagate to all sibling
    OptimizedModule wrappers that share the same underlying module.
    """
    if model is None:
        return None
    return getattr(model, "_orig_mod", model)
