# SPDX-License-Identifier: Apache-2.0
"""Container-wide concurrency throttle for forked graph-trace children."""

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path("/dev/shm/vllm-neuron-trace-rank-throttle")
_POLL_INTERVAL_S = 0.1


def _open_lock_file(path: Path) -> int:
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags, 0o600)


def _read_configured_limit(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Trace throttle state is corrupt at {path}: {value!r}"
        ) from exc
    if not 1 <= limit <= 4096:
        raise RuntimeError(
            f"Trace throttle state is out of range at {path}: {value!r}"
        )
    return limit


def _write_configured_limit(path: Path, limit: int) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("w", encoding="ascii") as stream:
            stream.write(f"{limit}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _ensure_limit(
    fcntl: Any,
    root: Path,
    requested_limit: int,
) -> None:
    """Establish one limit, rejecting changes while any slot is active."""
    config_path = root / "limit"
    configured_limit = _read_configured_limit(config_path)
    if configured_limit is None:
        _write_configured_limit(config_path, requested_limit)
        return
    if configured_limit == requested_limit:
        return

    probe_fds: list[int] = []
    try:
        for slot in range(configured_limit):
            fd = _open_lock_file(root / f"slot-{slot}")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(fd)
                raise RuntimeError(
                    "Trace throttle limit mismatch while slots are active: "
                    f"configured={configured_limit}, requested={requested_limit}"
                ) from exc
            probe_fds.append(fd)
        _write_configured_limit(config_path, requested_limit)
    finally:
        for fd in probe_fds:
            os.close(fd)


@contextmanager
def host_trace_slot(
    limit: int | None,
    *,
    parent_rank: int,
    lane_idx: int,
    root: Path = _DEFAULT_ROOT,
) -> Iterator[int | None]:
    """Lease one container-wide graph-construction slot.

    When ``limit`` is ``None`` this is a no-op. Otherwise every cooperating
    child uses one of ``limit`` advisory ``flock`` leases under ``/dev/shm``.
    Waiting happens before graph-related imports or model mutation. Closing the
    descriptor releases the lease, including on process exit or fatal signal.
    """
    if limit is None:
        yield None
        return
    if os.name != "posix":
        raise RuntimeError(
            "VLLM_NEURON_TRACE_RANK_CONCURRENCY requires Linux/POSIX flock"
        )

    import fcntl

    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Unsafe trace throttle root: {root}")
    acquired_fd: int | None = None
    acquired_slot: int | None = None
    wait_started = time.monotonic()

    while acquired_fd is None:
        coordinator_fd = _open_lock_file(root / "coordinator")
        try:
            fcntl.flock(coordinator_fd, fcntl.LOCK_EX)
            _ensure_limit(fcntl, root, limit)
            for slot in range(limit):
                candidate_fd = _open_lock_file(root / f"slot-{slot}")
                try:
                    fcntl.flock(
                        candidate_fd,
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError:
                    os.close(candidate_fd)
                    continue
                acquired_fd = candidate_fd
                acquired_slot = slot
                break
        finally:
            os.close(coordinator_fd)
        if acquired_fd is None:
            time.sleep(_POLL_INTERVAL_S)

    try:
        waited = time.monotonic() - wait_started
        logger.info(
            "Trace throttle acquired: slot=%d/%d parent_rank=%d lane=%d wait=%.2fs",
            acquired_slot,
            limit,
            parent_rank,
            lane_idx,
            waited,
        )
        yield acquired_slot
    finally:
        os.close(acquired_fd)
        logger.info(
            "Trace throttle released: slot=%d/%d parent_rank=%d lane=%d",
            acquired_slot,
            limit,
            parent_rank,
            lane_idx,
        )
