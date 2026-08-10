# SPDX-License-Identifier: Apache-2.0
"""Host-wide cross-rank semaphore bounding live neuronx-cc compiler processes.

vLLM_NEURON_PARALLEL_COMPILE_WORKERS caps each rank's compile pool, but ranks
independent processes (and containers) never see each other, so a 64-rank
launch can spawn far more concurrent neuronx-cc processes than the host has
RAM for. This module adds a GLOBAL bound: every neuronx-cc spawn acquires one
of ``cap`` advisory flock slots on a shared filesystem directory before the
compiler process is started, and releases it when the compiler exits.

Mechanism: one flock token FILE per slot (not byte-range locks on a single
file). POSIX fcntl byte-range locks are per-process, not per-fd, and are
dropped when ANY descriptor to the file closes in the process — unsafe with
the ThreadPoolExecutor used by parallel_compile. flock(LOCK_EX) on separate
token files is per-open-file-description, composes safely with threads, and
is released automatically on fd close — including process kill/OOM death, so
a dead compiler can never leak its slot.

Config (read lazily at acquire time; explicit keyword args override, which is
what the unit tests use):
    VLLM_NEURON_COMPILE_MAX_GLOBAL  unset/0 = disabled (byte-identical legacy
                                    behavior); when set must be in [16, 24].
    VLLM_NEURON_COMPILE_SEM_DIR     slot directory, shared by host + containers.
                                    Fallback: $NEURON_COMPILED_ARTIFACTS/compile-sem,
                                    then the system temp dir with a loud warning
                                    (temp dirs are usually NOT shared across
                                    containers, which silently breaks the bound).
    VLLM_NEURON_COMPILE_SEM_TIMEOUT seconds to wait for a slot before raising
                                    TimeoutError; unset/0 = wait forever.

This module is stdlib-only at import time so unit tests can load it standalone
without the heavy vllm_neuron package __init__.
"""

import logging
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MIN_CAP = 16
_MAX_CAP = 24
_POLL_INTERVAL_S = 0.25
_HEARTBEAT_INTERVAL_S = 60.0

# Slot files must be openable from containers running as different uids.
_DIR_MODE = 0o777
_FILE_MODE = 0o666


def _open_slot_file(path: Path) -> int:
    flags = os.O_CREAT | os.O_RDWR
    # O_CLOEXEC keeps the lease out of the spawned compiler; Python's
    # subprocess also closes fds by default, this is belt and braces.
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags, _FILE_MODE)


def parse_max_global(value: Optional[str]) -> Optional[int]:
    """Parse VLLM_NEURON_COMPILE_MAX_GLOBAL.

    Returns None when disabled (unset, empty, or "0"). An enabled value must
    be in [16, 24]; anything else raises instead of silently ignoring a
    safety bound the operator thought was active.
    """
    if value is None or value == "":
        return None
    try:
        converted = int(value)
    except ValueError as exc:
        raise ValueError(
            "VLLM_NEURON_COMPILE_MAX_GLOBAL must be 0 (disabled) or an "
            f"integer in [{_MIN_CAP}, {_MAX_CAP}], got {value!r}"
        ) from exc
    if converted == 0:
        return None
    if not _MIN_CAP <= converted <= _MAX_CAP:
        raise ValueError(
            "VLLM_NEURON_COMPILE_MAX_GLOBAL must be 0 (disabled) or an "
            f"integer in [{_MIN_CAP}, {_MAX_CAP}], got {value!r}"
        )
    return converted


def resolve_sem_dir(sem_dir: Optional[str] = None) -> Path:
    """Resolve the slot directory shared by host and containers."""
    if sem_dir:
        return Path(sem_dir)
    env_dir = os.environ.get("VLLM_NEURON_COMPILE_SEM_DIR")
    if env_dir:
        return Path(env_dir)
    artifacts = os.environ.get("NEURON_COMPILED_ARTIFACTS")
    if artifacts:
        return Path(artifacts) / "compile-sem"
    fallback = Path(tempfile.gettempdir()) / "vllm-neuron-compile-sem"
    logger.warning(
        "VLLM_NEURON_COMPILE_SEM_DIR is unset and NEURON_COMPILED_ARTIFACTS is "
        "unset; falling back to %s. Temp dirs are normally NOT shared between "
        "host and containers, so the global compile bound will only apply "
        "per-filesystem-view. Set VLLM_NEURON_COMPILE_SEM_DIR to a directory "
        "every rank can reach.",
        fallback,
    )
    return fallback


def _read_config() -> tuple[Optional[int], Optional[str], float]:
    """Read config from vllm_neuron.envs, deferred so tests can stay standalone."""
    from vllm_neuron import envs

    return (
        envs.VLLM_NEURON_COMPILE_MAX_GLOBAL,
        envs.VLLM_NEURON_COMPILE_SEM_DIR,
        float(envs.VLLM_NEURON_COMPILE_SEM_TIMEOUT),
    )


def _is_rank0() -> bool:
    return os.environ.get("RANK", "0") in ("", "0")


@contextmanager
def global_compile_slot(
    cap: Optional[int] = None,
    sem_dir: Optional[str] = None,
    timeout_s: Optional[float] = None,
    heartbeat_s: float = _HEARTBEAT_INTERVAL_S,
) -> Iterator[Optional[int]]:
    """Lease one host-wide neuronx-cc slot for the duration of a compile.

    When the bound is disabled (cap None/0) this is a no-op that touches no
    filesystem state — legacy behavior is preserved byte-for-byte. Otherwise
    blocks until one of ``cap`` flock token files under the slot directory is
    free, yields the slot index, and releases the lease on exit. The lease fd
    is O_CLOEXEC and is closed in a finally block, so the slot is released on
    compiler exit, exception, or holder death.
    """
    if cap is None and sem_dir is None and timeout_s is None:
        cfg_cap, cfg_dir, cfg_timeout = _read_config()
        cap, sem_dir, timeout_s = cfg_cap, cfg_dir, cfg_timeout
    elif cap is None:
        cap = parse_max_global(os.environ.get("VLLM_NEURON_COMPILE_MAX_GLOBAL"))
    if not cap:
        yield None
        return
    if os.name != "posix":
        raise RuntimeError(
            "VLLM_NEURON_COMPILE_MAX_GLOBAL requires Linux/POSIX flock"
        )
    if timeout_s is None:
        timeout_s = float(os.environ.get("VLLM_NEURON_COMPILE_SEM_TIMEOUT") or 0)

    import fcntl

    root = resolve_sem_dir(sem_dir)
    root.mkdir(mode=_DIR_MODE, parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"Unsafe compile semaphore dir: {root}")

    if _is_rank0():
        logger.info(
            "Global compile bound active: cap=%d live neuronx-cc processes, "
            "slot dir=%s, timeout=%ss",
            cap,
            root,
            timeout_s or "none",
        )

    acquired_fd: Optional[int] = None
    acquired_slot: Optional[int] = None
    wait_started = time.monotonic()
    last_heartbeat = wait_started

    while acquired_fd is None:
        for slot in range(cap):
            candidate_fd = _open_slot_file(root / f"slot-{slot}")
            try:
                fcntl.flock(candidate_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(candidate_fd)
                continue
            acquired_fd = candidate_fd
            acquired_slot = slot
            break
        if acquired_fd is not None:
            break
        now = time.monotonic()
        if timeout_s and now - wait_started >= timeout_s:
            raise TimeoutError(
                f"Timed out after {timeout_s}s waiting for a global compile "
                f"slot (cap={cap}, dir={root}). Increase "
                "VLLM_NEURON_COMPILE_SEM_TIMEOUT or check for stalled "
                "neuronx-cc processes holding slots."
            )
        if now - last_heartbeat >= heartbeat_s:
            last_heartbeat = now
            logger.info(
                "Waiting %.0fs for a global compile slot (cap=%d, dir=%s, "
                "pid=%d, rank=%s): all %d slots held by other compilers",
                now - wait_started,
                cap,
                root,
                os.getpid(),
                os.environ.get("RANK", "?"),
                cap,
            )
        time.sleep(_POLL_INTERVAL_S)

    try:
        waited = time.monotonic() - wait_started
        (logger.info if waited >= 1.0 else logger.debug)(
            "Global compile slot acquired: slot=%d/%d wait=%.2fs pid=%d",
            acquired_slot,
            cap,
            waited,
            os.getpid(),
        )
        yield acquired_slot
    finally:
        os.close(acquired_fd)
