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

import logging
import os
import shutil
import signal
import tempfile
import time
import traceback
from collections.abc import Callable
from typing import Any

import torch

from vllm_neuron import envs
from vllm_neuron.compile.trace_throttle import host_trace_slot

logger = logging.getLogger(__name__)


# A trace job: a callable plus the kwargs that drive its forward pass.
Job = tuple[Callable[..., Any], dict[str, Any]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


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
    logger.info(
        "Parallel trace: jobs=%d, lanes=%d, parent_rank=%d, host_limit=%s",
        len(jobs),
        num_workers,
        parent_rank,
        trace_rank_concurrency,
    )
    t0 = time.perf_counter()
    if num_workers == 1:
        _run_fresh_children_sequentially(
            jobs, parent_rank, trace_rank_concurrency
        )
    else:
        _run_pool_fork(
            jobs, parent_rank, num_workers, trace_rank_concurrency
        )
    elapsed = time.perf_counter() - t0
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
    with host_trace_slot(
        trace_rank_concurrency,
        parent_rank=parent_rank,
        lane_idx=lane_idx,
    ):
        _fork_child_main(lane_idx, parent_rank, jobs_slice, result_path)


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

        _swap_unique_models_to_meta([model for model, _ in jobs_slice])

        for j_idx, (model, kwargs) in enumerate(jobs_slice):
            failing_job = j_idx
            try:
                model(**kwargs)
            except CaptureComplete:
                # Successful trace — capture backend signals "done"
                # this way after writing the HLO.
                pass
        failing_job = None
    except BaseException as e:
        status = "ERROR"
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
    if isinstance(obj, torch.Tensor):
        yield path, obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            sep = "" if not path else "."
            yield from _walk_tensors(v, f"{path}{sep}{k}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_tensors(v, f"{path}[{i}]")
        return
    # Dataclass-like (e.g. AttentionMetadata). Skip primitives, modules,
    # and anything without a sensible __dict__.
    if hasattr(obj, "__dict__") and not isinstance(obj, torch.nn.Module):
        type_name = type(obj).__name__
        for k, v in vars(obj).items():
            sep = "" if not path else "."
            yield from _walk_tensors(v, f"{path}{sep}<{type_name}>.{k}")


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

    Beyond parameters/buffers, this also swaps plain tensor attributes
    (``__dict__`` entries that are torch.Tensor) — KV caches like
    ``self.k_cache`` / ``self.v_cache`` are bound this way via
    ``bind_kv_cache`` and live on the runtime device.

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
    id_to_meta: dict[int, torch.Tensor] = {}

    def _alias_root_id(src: torch.Tensor) -> int:
        root = src
        seen: set[int] = set()
        while isinstance(getattr(root, "_base", None), torch.Tensor):
            if id(root) in seen:
                break
            seen.add(id(root))
            root = root._base
        return id(root)

    def _replacement_for(src: torch.Tensor) -> torch.Tensor:
        cached = id_to_meta.get(id(src))
        if cached is not None:
            return cached
        storage_id = _alias_root_id(src)
        meta_storage = storage_to_meta.get(storage_id)
        if meta_storage is None:
            meta_storage = torch.UntypedStorage(
                src.untyped_storage().nbytes(), device="meta"
            )
            storage_to_meta[storage_id] = meta_storage
        repl = torch.empty(0, dtype=src.dtype, device="meta").set_(
            meta_storage,
            src.storage_offset(),
            src.shape,
            src.stride(),
        )
        id_to_meta[id(src)] = repl
        # Hold a strong reference to the source so its storage survives
        # until child exit (no nrt_tensor_free calls).
        _META_PARAM_KEEPALIVE.append(src)
        return repl

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
        # Plain tensor attributes (e.g. k_cache / v_cache bound after
        # initialize_kv_cache). Skip the special _parameters / _buffers
        # / _modules dicts — already handled above.
        for name, val in list(submod.__dict__.items()):
            if name in ("_parameters", "_buffers", "_modules"):
                continue
            if not isinstance(val, torch.Tensor):
                continue
            if val.device.type == "meta":
                continue
            submod.__dict__[name] = _replacement_for(val)


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
