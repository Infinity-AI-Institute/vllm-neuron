# SPDX-License-Identifier: Apache-2.0
"""Persistent disk cache for NKI kernel compile results.

Caches NKICompileResult to disk so that warm starts skip NKI BIR
compilation. Compatible with vLLM Neuron's local + remote cache layout.

Storage: JSON files in {cache_dir}/nki/{key}.json
Multi-process safety: filelock.FileLock per cache key
"""

import base64
import dataclasses
import enum
import errno
import hashlib
import inspect
import json
import logging
import os
import platform
import shutil
import sys
import threading
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

from filelock import FileLock, Timeout

from vllm_neuron import envs

from ..compile.platform import (
    get_neuronxcc_version,
    get_nki_version,
    get_platform_target,
    get_torch_neuronx_version,
)
from .nki_compile import NKICompileResult
from .nki_dtype import str_to_torch_dtype

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2
_KEY_SCHEMA_VERSION = 2
_NKI_CACHE_SUBDIR = "nki"
_NKI_BINARY_SUBDIR = "binaries"
_LOCK_TIMEOUT = 300  # seconds
_SOURCE_IDENTITY_ENV = "VLLM_NEURON_NKI_SOURCE_IDENTITY"
_MAX_KEY_DEPTH = 64
_SOURCE_DIGEST_CACHE: dict[tuple[str, int, int, int], str] = {}
_SOURCE_DIGEST_LOCK = threading.Lock()

# One K3 prefill graph can contain tens of thousands of calls to the same NKI
# kernel configurations.  The persistent cache prevents recompilation, but a
# hit still opens and parses the same JSON record (and validates its binary)
# for every FX node.  Keep successful results in the tracing process after the
# first persistent lookup. Source, shape, dtype, constants, grid, platform,
# versions, and device-dump mode retain the disk cache's invalidation contract.
# Local and remote roots are additional process-local namespace components
# because dumped_config embeds the resolved path to the materialized binary.
_ProcessCacheKey = tuple[str, str, str]


@dataclass(frozen=True)
class _ProcessCacheEntry:
    result: NKICompileResult
    binary_path: Optional[str]


_PROCESS_CACHE: dict[_ProcessCacheKey, _ProcessCacheEntry] = {}
_PROCESS_CACHE_LOCK = threading.Lock()
_PROCESS_CACHE_HITS = 0
_PROCESS_CACHE_MISSES = 0
_PROCESS_CACHE_DISABLED: Optional[bool] = None
_PROCESS_CACHE_GENERATION = 0
_PROCESS_CACHE_PID = os.getpid()


class _UnsafeCacheKey(ValueError):
    """Raised when semantic key material cannot be serialized safely."""


def _reset_process_cache_after_fork_locked() -> None:
    """Drop inherited entries when the current PID changes after ``fork``."""

    global _PROCESS_CACHE_DISABLED, _PROCESS_CACHE_GENERATION, _PROCESS_CACHE_PID
    global _PROCESS_CACHE_HITS, _PROCESS_CACHE_MISSES
    current_pid = os.getpid()
    if current_pid == _PROCESS_CACHE_PID:
        return
    _PROCESS_CACHE.clear()
    _PROCESS_CACHE_HITS = 0
    _PROCESS_CACHE_MISSES = 0
    _PROCESS_CACHE_DISABLED = None
    _PROCESS_CACHE_GENERATION += 1
    _PROCESS_CACHE_PID = current_pid


def clear_nki_process_cache() -> None:
    """Clear process-local NKI results and counters.

    Production code should not need this; it is an explicit seam for tests and
    long-lived tooling that intentionally changes cache roots or kernel state.
    """

    global _PROCESS_CACHE_DISABLED, _PROCESS_CACHE_GENERATION, _PROCESS_CACHE_PID
    global _PROCESS_CACHE_HITS, _PROCESS_CACHE_MISSES
    with _PROCESS_CACHE_LOCK:
        _reset_process_cache_after_fork_locked()
        _PROCESS_CACHE.clear()
        _PROCESS_CACHE_HITS = 0
        _PROCESS_CACHE_MISSES = 0
        _PROCESS_CACHE_DISABLED = None
        _PROCESS_CACHE_GENERATION += 1
        _PROCESS_CACHE_PID = os.getpid()


def nki_process_cache_stats() -> dict[str, int]:
    """Return cheap instrumentation for repeated in-process cache lookups."""

    with _PROCESS_CACHE_LOCK:
        _reset_process_cache_after_fork_locked()
        return {
            "entries": len(_PROCESS_CACHE),
            "hits": _PROCESS_CACHE_HITS,
            "misses": _PROCESS_CACHE_MISSES,
        }


def _sync_nki_process_cache_mode(disabled: bool) -> int:
    """Discard entries whenever compile-cache bypass mode changes."""

    global _PROCESS_CACHE_DISABLED, _PROCESS_CACHE_GENERATION
    with _PROCESS_CACHE_LOCK:
        _reset_process_cache_after_fork_locked()
        if _PROCESS_CACHE_DISABLED is None:
            _PROCESS_CACHE_DISABLED = disabled
        elif _PROCESS_CACHE_DISABLED != disabled:
            _PROCESS_CACHE.clear()
            _PROCESS_CACHE_DISABLED = disabled
            _PROCESS_CACHE_GENERATION += 1
        return _PROCESS_CACHE_GENERATION


def _get_nki_process_cache(
    process_key: _ProcessCacheKey,
) -> Optional[NKICompileResult]:
    global _PROCESS_CACHE_HITS, _PROCESS_CACHE_MISSES
    with _PROCESS_CACHE_LOCK:
        _reset_process_cache_after_fork_locked()
        entry = _PROCESS_CACHE.get(process_key)
        if entry is None:
            _PROCESS_CACHE_MISSES += 1
            return None

        # Persistent reads fail closed when the materialized BIR/KLIR file is
        # removed. Preserve that contract while still avoiding JSON I/O and
        # repeated backend-config decoding on every FX node.
        if entry.binary_path is not None and not os.path.isfile(entry.binary_path):
            _PROCESS_CACHE.pop(process_key, None)
            _PROCESS_CACHE_MISSES += 1
            return None

        _PROCESS_CACHE_HITS += 1
        return entry.result


def _put_nki_process_cache(
    process_key: _ProcessCacheKey,
    result: NKICompileResult,
    generation: int,
) -> NKICompileResult:
    entry = _ProcessCacheEntry(
        result=result,
        binary_path=_kernel_binary_path(result.dumped_config),
    )
    with _PROCESS_CACHE_LOCK:
        _reset_process_cache_after_fork_locked()
        # A clear or cache-mode transition may have occurred while this caller
        # compiled or waited on the file lock. Do not resurrect an entry from
        # the previous generation.
        if generation != _PROCESS_CACHE_GENERATION:
            return result
        # Preserve the first complete result if concurrent tracing lanes ever
        # share a process.  The persistent key guarantees equivalence.
        return _PROCESS_CACHE.setdefault(process_key, entry).result


def _normalize_cache_root(path: Optional[str]) -> str:
    if path is None:
        return ""
    return os.path.normcase(os.path.abspath(path))


def _nki_process_cache_key(
    cache_key: str,
    local_dir: str,
    remote_dir: Optional[str],
) -> _ProcessCacheKey:
    return (
        _normalize_cache_root(local_dir),
        _normalize_cache_root(remote_dir),
        cache_key,
    )


def create_nki_cache_key(
    func: Callable,
    args: dict[str, Any],
    grid: tuple[int, ...],
) -> Optional[str]:
    """Generate a persistent cache key for an NKI kernel invocation.

    Returns None if any semantic input cannot be represented deterministically
    or if required version/platform information cannot be obtained. Returning
    None deliberately bypasses both persistent and process-local caches.
    """
    try:
        hashed_args = []

        # Kernel identity
        kernel_module = getattr(func, "__module__", None)
        kernel_qualname = getattr(func, "__qualname__", None)

        # Kernel source hash — invalidates cache when kernel code is edited
        kernel_semantic_digest = _hash_kernel_source(func)

        # Input shapes/dtypes
        for name, v in args.items():
            h, is_hashable = _hashable_arg(v, name)
            if not is_hashable:
                return None
            hashed_args.append([name, h])

        # Grid/LNC
        hashed_grid = _canonical_value(grid, "grid", set(), 0)

        # Platform + versions for cache invalidation
        versions = {
            "nki": get_nki_version(),
            "neuronxcc": get_neuronxcc_version(),
            "torch_neuronx": get_torch_neuronx_version(),
            "python": platform.python_version(),
        }

        # Device dump inserts DevicePrint ops into the kernel, producing a different binary.
        device_dump = envs.VLLM_NEURON_KERNEL_DEVICE_DUMP

        payload = {
            "key_schema": _KEY_SCHEMA_VERSION,
            "kernel": {
                "module": kernel_module,
                "qualname": kernel_qualname,
                "semantic_digest": kernel_semantic_digest,
            },
            "args": hashed_args,
            "grid": hashed_grid,
            "platform": get_platform_target(),
            "versions": versions,
            "device_dump": device_dump,
            "source_identity": _source_overlay_identity(func),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:32]
    except Exception as e:  # noqa: BLE001 - cache generation must fail closed
        logger.debug("Cannot generate NKI cache key: %s", e)
        return None


def get_nki_cache(cache_key: str) -> Optional[NKICompileResult]:
    """Load a cached NKICompileResult from local disk cache.

    Returns None on miss. Does not check remote — use
    ``_fetch_from_remote`` inside a locked block for that.
    """
    return _get_nki_cache_from_dir(cache_key, _get_local_nki_cache_dir())


def _get_nki_cache_from_dir(
    cache_key: str, local_dir: str
) -> Optional[NKICompileResult]:
    local_path = os.path.join(local_dir, f"{cache_key}.json")

    result = _read_cache_file(local_path)
    if result is not None:
        logger.debug("NKI cache hit (local): %s", cache_key)
        return result

    return None


def put_nki_cache(cache_key: str, result: NKICompileResult) -> None:
    """Store an NKICompileResult to the local disk cache.

    Uses atomic write (tmp file + replace) for safety.
    """
    _put_nki_cache_in_dir(cache_key, result, _get_local_nki_cache_dir())


def _put_nki_cache_in_dir(
    cache_key: str,
    result: NKICompileResult,
    local_dir: str,
) -> NKICompileResult:
    os.makedirs(local_dir, exist_ok=True)

    local_path = os.path.join(local_dir, f"{cache_key}.json")
    tmp_path = f"{local_path}.tmp.{os.getpid()}"
    cached_result, kernel_binary = _persist_kernel_binary(cache_key, result, local_dir)

    data = {
        "schema_version": _SCHEMA_VERSION,
        "dumped_config": cached_result.dumped_config,
        "return_types": [
            [str(dtype), list(shape)] for dtype, shape in cached_result.return_types
        ],
        "operand_output_aliases": {
            str(k): v for k, v in cached_result.operand_output_aliases.items()
        },
    }
    if kernel_binary is not None:
        data["kernel_binary"] = kernel_binary

    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, sort_keys=True, separators=(",", ":"))
            f.flush()
        os.replace(tmp_path, local_path)
        logger.debug("NKI cache stored: %s", cache_key)
    except OSError as e:
        logger.warning("Failed to write NKI cache entry %s: %s", cache_key, e)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return cached_result


def compile_with_cache(
    cache_key: Optional[str],
    compile_fn: Callable[[], NKICompileResult],
) -> NKICompileResult:
    """Check cache, or compile under lock if miss.

    Args:
        cache_key: Persistent cache key (None disables caching).
        compile_fn: Zero-arg callable that produces the NKICompileResult.

    Returns:
        NKICompileResult from cache or fresh compilation.
    """
    cache_disabled = envs.VLLM_NEURON_DISABLE_COMPILE_CACHE
    process_generation = _sync_nki_process_cache_mode(cache_disabled)
    if cache_key is None or cache_disabled:
        return compile_fn()

    # Snapshot roots once so an environment mutation cannot mix namespaces,
    # lock files, reads, and writes from different cache roots in one call.
    local_dir = _get_local_nki_cache_dir()
    remote_dir = _get_remote_nki_cache_dir()
    process_key = _nki_process_cache_key(cache_key, local_dir, remote_dir)

    cached = _get_nki_process_cache(process_key)
    if cached is not None:
        return cached

    # Fast path: lockless local read
    cached = _get_nki_cache_from_dir(cache_key, local_dir)
    if cached is not None:
        return _put_nki_process_cache(process_key, cached, process_generation)

    # Slow path: acquire lock, try remote fetch, then compile
    os.makedirs(local_dir, exist_ok=True)
    lock_path = os.path.join(local_dir, f"{cache_key}.lock")

    try:
        with FileLock(lock_path, timeout=_LOCK_TIMEOUT):
            # Double-check local after acquiring lock
            cached = _get_nki_cache_from_dir(cache_key, local_dir)
            if cached is not None:
                return _put_nki_process_cache(process_key, cached, process_generation)

            # Try remote fetch (only lock winner hits remote filesystem)
            cached = _fetch_from_remote_dirs(cache_key, local_dir, remote_dir)
            if cached is not None:
                return _put_nki_process_cache(process_key, cached, process_generation)

            result = compile_fn()
            persisted = _put_nki_cache_in_dir(cache_key, result, local_dir)
            return _put_nki_process_cache(process_key, persisted, process_generation)
    except Timeout:
        logger.warning(
            "NKI cache lock timeout for %s, compiling without cache", cache_key
        )
        return _put_nki_process_cache(process_key, compile_fn(), process_generation)


def save_nki_cache_to_remote(
    local_cache_dir: Optional[str] = None,
    remote_cache_dir: Optional[str] = None,
) -> None:
    """Promote local NKI cache entries to the remote cache atomically.

    Copies the entire local nki/ subdirectory into a per-process staging
    directory on the remote filesystem, then atomically renames it to the
    final destination. If the remote nki/ directory already exists
    (another node promoted first), this is a no-op.

    Args:
        local_cache_dir: Override local cache root. Defaults to get_neuron_compile_cache_dir().
        remote_cache_dir: Override remote cache root. Defaults to VLLM_NEURON_REMOTE_CACHE.
    """
    import socket

    if local_cache_dir is None:
        local_nki_dir = _get_local_nki_cache_dir()
    else:
        local_nki_dir = os.path.join(local_cache_dir, _NKI_CACHE_SUBDIR)

    if remote_cache_dir is None:
        remote_nki_dir = _get_remote_nki_cache_dir()
    else:
        remote_nki_dir = os.path.join(remote_cache_dir, _NKI_CACHE_SUBDIR)

    if not remote_nki_dir:
        return

    if not os.path.isdir(local_nki_dir):
        return

    # Fast path: remote already exists
    if os.path.exists(remote_nki_dir):
        logger.debug("Remote NKI cache already exists, skipping promotion")
        return

    remote_parent = os.path.dirname(remote_nki_dir)
    os.makedirs(remote_parent, exist_ok=True)
    staging_dir = os.path.join(
        remote_parent,
        f"{_NKI_CACHE_SUBDIR}.tmp.{socket.gethostname()}.{os.getpid()}",
    )

    try:
        shutil.copytree(local_nki_dir, staging_dir)
        os.rename(staging_dir, remote_nki_dir)
        entry_count = sum(1 for f in os.listdir(remote_nki_dir) if f.endswith(".json"))
        logger.info("Promoted NKI cache to remote (%d entries)", entry_count)
    except OSError as e:
        if e.errno in (errno.EEXIST, errno.ENOTEMPTY):
            logger.debug("Remote NKI cache already exists (concurrent promoter won)")
        else:
            logger.warning("Failed to promote NKI cache to remote: %s", e)
    finally:
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)


def _get_local_nki_cache_dir() -> str:
    return os.path.join(envs.get_neuron_compile_cache_dir(), _NKI_CACHE_SUBDIR)


def _get_remote_nki_cache_dir() -> Optional[str]:
    remote = envs.VLLM_NEURON_REMOTE_CACHE
    if remote:
        return os.path.join(remote, _NKI_CACHE_SUBDIR)
    return None


def _fetch_from_remote(cache_key: str) -> Optional[NKICompileResult]:
    """Fetch a single NKI cache entry from the remote cache into local.

    Returns the deserialized result on hit, None on miss. Copies the
    remote file to the local cache for future fast-path hits.
    """
    return _fetch_from_remote_dirs(
        cache_key,
        _get_local_nki_cache_dir(),
        _get_remote_nki_cache_dir(),
    )


def _fetch_from_remote_dirs(
    cache_key: str,
    local_dir: str,
    remote_dir: Optional[str],
) -> Optional[NKICompileResult]:
    if not remote_dir:
        return None

    remote_path = os.path.join(remote_dir, f"{cache_key}.json")
    result = _read_cache_file(remote_path)
    if result is not None:
        logger.debug("NKI cache hit (remote): %s", cache_key)
        local_path = os.path.join(local_dir, f"{cache_key}.json")
        remote_binary = _kernel_binary_path(result.dumped_config)
        if remote_binary is not None:
            relative_binary = os.path.relpath(remote_binary, remote_dir)
            if not os.path.isabs(relative_binary) and not (
                relative_binary == os.pardir
                or relative_binary.startswith(os.pardir + os.sep)
            ):
                _copy_to_local(
                    remote_binary,
                    os.path.join(local_dir, relative_binary),
                )
        _copy_to_local(remote_path, local_path)
        local_result = _read_cache_file(local_path)
        return local_result if local_result is not None else result

    return None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hashable_arg(value: Any, name: str) -> tuple[Any, bool]:
    """Return deterministic semantic key material for a kernel argument."""

    try:
        return _canonical_value(value, name, set(), 0), True
    except _UnsafeCacheKey as error:
        logger.debug("NKI cache disabled for argument %s: %s", name, error)
        return None, False


def _canonical_value(
    value: Any,
    owner: str,
    active: set[int],
    depth: int,
) -> Any:
    if depth > _MAX_KEY_DEPTH:
        raise _UnsafeCacheKey(f"{owner} exceeds maximum key depth")

    if value is None or isinstance(value, (bool, int, str)):
        return [type(value).__name__, value]
    if value is Ellipsis or value is NotImplemented:
        return ["singleton", repr(value)]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, complex):
        return ["complex", value.real.hex(), value.imag.hex()]
    if isinstance(value, bytes):
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, range):
        return ["range", value.start, value.stop, value.step]
    if isinstance(value, slice):
        return [
            "slice",
            _canonical_value(value.start, f"{owner}.start", active, depth + 1),
            _canonical_value(value.stop, f"{owner}.stop", active, depth + 1),
            _canonical_value(value.step, f"{owner}.step", active, depth + 1),
        ]
    if isinstance(value, enum.Enum):
        return [
            "enum",
            value.__class__.__module__,
            value.__class__.__qualname__,
            value.name,
            _canonical_value(value.value, f"{owner}.value", active, depth + 1),
        ]
    if type(value).__module__ == "torch" and type(value).__name__ in {
        "device",
        "dtype",
        "layout",
        "memory_format",
    }:
        return ["torch_value", type(value).__name__, str(value)]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return _tensor_key_material(value, owner, active, depth + 1)
    if isinstance(value, types.CodeType):
        return ["code", _code_semantics(value, owner, active, depth + 1)]
    if inspect.ismethod(value) or inspect.isfunction(value):
        return ["callable", _callable_semantics(value, owner, active, depth + 1)]
    if inspect.isbuiltin(value):
        return [
            "builtin",
            getattr(value, "__module__", None),
            getattr(value, "__qualname__", getattr(value, "__name__", None)),
        ]
    if isinstance(value, types.ModuleType):
        return ["module", _module_semantics(value)]
    if inspect.isclass(value):
        return ["class", _class_semantics(value)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = []
        _enter_key_object(value, owner, active)
        try:
            for field in dataclasses.fields(value):
                fields.append(
                    [
                        field.name,
                        _canonical_value(
                            getattr(value, field.name),
                            f"{owner}.{field.name}",
                            active,
                            depth + 1,
                        ),
                    ]
                )
        finally:
            active.remove(id(value))
        return [
            "dataclass",
            value.__class__.__module__,
            value.__class__.__qualname__,
            fields,
        ]
    if isinstance(value, tuple):
        return [
            "tuple",
            _canonical_sequence(value, owner, active, depth + 1),
        ]
    if isinstance(value, list):
        return [
            "list",
            _canonical_sequence(value, owner, active, depth + 1),
        ]
    if isinstance(value, dict):
        _enter_key_object(value, owner, active)
        try:
            items = []
            for key, item in value.items():
                canonical_key = _canonical_value(key, f"{owner}.key", active, depth + 1)
                canonical_item = _canonical_value(
                    item, f"{owner}[{key!r}]", active, depth + 1
                )
                items.append([canonical_key, canonical_item])
            items.sort(key=lambda pair: _canonical_json(pair[0]))
        finally:
            active.remove(id(value))
        return ["dict", items]
    if isinstance(value, (set, frozenset)):
        _enter_key_object(value, owner, active)
        try:
            items = [
                _canonical_value(item, f"{owner}.item", active, depth + 1)
                for item in value
            ]
            items.sort(key=_canonical_json)
        finally:
            active.remove(id(value))
        return [type(value).__name__, items]

    raise _UnsafeCacheKey(
        f"{owner} has unsupported nondeterministic type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _enter_key_object(value: Any, owner: str, active: set[int]) -> None:
    identity = id(value)
    if identity in active:
        raise _UnsafeCacheKey(f"{owner} contains a recursive value")
    active.add(identity)


def _canonical_sequence(
    value: tuple[Any, ...] | list[Any],
    owner: str,
    active: set[int],
    depth: int,
) -> list[Any]:
    _enter_key_object(value, owner, active)
    try:
        return [
            _canonical_value(item, f"{owner}[{index}]", active, depth + 1)
            for index, item in enumerate(value)
        ]
    finally:
        active.remove(id(value))


def _tensor_key_material(
    value: Any,
    owner: str,
    active: set[int],
    depth: int,
) -> list[Any]:
    try:
        shape = tuple(value.shape)
        dtype = str(value.dtype)
        layout = str(value.layout)
        device = getattr(value, "fake_device", value.device)
        stride_attribute = value.stride
        stride = stride_attribute() if callable(stride_attribute) else stride_attribute
        offset_attribute = value.storage_offset
        storage_offset = (
            offset_attribute() if callable(offset_attribute) else offset_attribute
        )
    except Exception as error:
        raise _UnsafeCacheKey(
            f"{owner} tensor metadata is incomplete: {error}"
        ) from error

    device_role = {
        "type": getattr(device, "type", None),
        "index": getattr(device, "index", None),
        "value": str(device),
        "fake": hasattr(value, "fake_device"),
    }
    return [
        "tensor",
        _canonical_value(shape, f"{owner}.shape", active, depth + 1),
        dtype,
        layout,
        _canonical_value(tuple(stride), f"{owner}.stride", active, depth + 1),
        _canonical_value(storage_offset, f"{owner}.storage_offset", active, depth + 1),
        device_role,
    ]


def _code_semantics(
    code: types.CodeType,
    owner: str,
    active: set[int],
    depth: int,
) -> dict[str, Any]:
    constants = [
        _canonical_value(constant, f"{owner}.const", active, depth + 1)
        for constant in code.co_consts
    ]
    return {
        "bytecode": base64.b64encode(code.co_code).decode("ascii"),
        "constants": constants,
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
        "exceptiontable": base64.b64encode(
            getattr(code, "co_exceptiontable", b"")
        ).decode("ascii"),
    }


def _callable_semantics(
    func: Callable,
    owner: str,
    active: set[int],
    depth: int,
) -> dict[str, Any]:
    if inspect.ismethod(func):
        bound_self = _canonical_value(
            func.__self__, f"{owner}.__self__", active, depth + 1
        )
        func = func.__func__
    else:
        bound_self = None

    try:
        unwrapped = inspect.unwrap(func)
    except ValueError as error:
        raise _UnsafeCacheKey(f"{owner} has a cyclic __wrapped__ chain") from error
    code = getattr(unwrapped, "__code__", None)
    if not isinstance(code, types.CodeType):
        raise _UnsafeCacheKey(f"{owner} has no inspectable Python code")

    reference = {
        "module": getattr(unwrapped, "__module__", None),
        "qualname": getattr(unwrapped, "__qualname__", None),
    }
    identity = id(unwrapped)
    if identity in active:
        return {"recursive_reference": reference}
    active.add(identity)
    try:
        try:
            closure = inspect.getclosurevars(unwrapped)
        except TypeError as error:
            raise _UnsafeCacheKey(f"{owner} closure cannot be inspected") from error
        globals_payload = [
            [
                name,
                _canonical_value(value, f"{owner}.global[{name}]", active, depth + 1),
            ]
            for name, value in sorted(closure.globals.items())
        ]
        nonlocals_payload = [
            [
                name,
                _canonical_value(value, f"{owner}.nonlocal[{name}]", active, depth + 1),
            ]
            for name, value in sorted(closure.nonlocals.items())
        ]
        defaults = _canonical_value(
            getattr(unwrapped, "__defaults__", None),
            f"{owner}.__defaults__",
            active,
            depth + 1,
        )
        kwdefaults = _canonical_value(
            getattr(unwrapped, "__kwdefaults__", None),
            f"{owner}.__kwdefaults__",
            active,
            depth + 1,
        )
        attributes = _canonical_value(
            getattr(unwrapped, "__dict__", {}),
            f"{owner}.__dict__",
            active,
            depth + 1,
        )
        return {
            **reference,
            "code": _code_semantics(code, owner, active, depth + 1),
            "bound_self": bound_self,
            "defaults": defaults,
            "kwdefaults": kwdefaults,
            "attributes": attributes,
            "globals": globals_payload,
            "nonlocals": nonlocals_payload,
            "builtins": sorted(closure.builtins),
            # inspect.getclosurevars reports attribute names here as well as
            # truly unresolved globals. Preserve them as code identity instead
            # of treating ordinary ``tensor.shape`` access as unsafe.
            "unbound_names": sorted(closure.unbound),
        }
    finally:
        active.remove(identity)


def _module_semantics(module: types.ModuleType) -> dict[str, Any]:
    version = getattr(module, "__version__", None)
    if version is not None and not isinstance(version, (bool, int, float, str)):
        raise _UnsafeCacheKey(f"module {module.__name__} has nondeterministic version")
    source_digest = None
    source_file = getattr(module, "__file__", None)
    if isinstance(source_file, str):
        try:
            source_digest = _sealed_source_digest(source_file)
        except OSError:
            source_digest = None
    root_name = module.__name__.partition(".")[0]
    sdk_sealed = root_name in {"nki", "neuronxcc", "numpy", "torch", "torch_xla"}
    if (
        version is None
        and source_digest is None
        and root_name not in sys.stdlib_module_names
        and not sdk_sealed
    ):
        raise _UnsafeCacheKey(
            f"module {module.__name__} has no deterministic source or version identity"
        )
    return {
        "name": module.__name__,
        "version": version,
        "source_digest": source_digest,
        "sdk_version_sealed": sdk_sealed,
    }


def _class_semantics(cls: type[Any]) -> dict[str, Any]:
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        source = None
    return {
        "module": cls.__module__,
        "qualname": cls.__qualname__,
        "source_digest": (
            hashlib.sha256(source.encode("utf-8")).hexdigest() if source else None
        ),
    }


def _source_overlay_identity(func: Callable) -> dict[str, Any]:
    explicit = os.getenv(_SOURCE_IDENTITY_ENV)
    if explicit is not None and (
        not explicit or len(explicit) > 512 or "\x00" in explicit
    ):
        raise _UnsafeCacheKey(
            f"{_SOURCE_IDENTITY_ENV} must be a nonempty deterministic string"
        )

    source_file = inspect.getsourcefile(inspect.unwrap(func))
    source_digest = None
    source_name = None
    if source_file is not None:
        source_name = os.path.basename(source_file)
        try:
            source_digest = _sealed_source_digest(source_file)
        except OSError:
            # Dynamically generated kernels are fully represented by their code,
            # defaults, closures, and referenced globals below.
            source_digest = None
    return {
        "explicit": explicit,
        "module": getattr(func, "__module__", None),
        "source_name": source_name,
        "sealed_source_digest": source_digest,
    }


def _sealed_source_digest(source_file: str) -> str:
    absolute = os.path.abspath(source_file)
    stat = os.stat(absolute)
    cache_key = (absolute, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
    with _SOURCE_DIGEST_LOCK:
        cached = _SOURCE_DIGEST_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with open(absolute, "rb") as source_stream:
        digest = hashlib.sha256(source_stream.read()).hexdigest()
    with _SOURCE_DIGEST_LOCK:
        if len(_SOURCE_DIGEST_CACHE) >= 256:
            _SOURCE_DIGEST_CACHE.clear()
        _SOURCE_DIGEST_CACHE[cache_key] = digest
    return digest


def _hash_kernel_source(func: Callable) -> str:
    """Hash code plus transitive helpers and specialization-time values."""

    payload = _callable_semantics(func, "kernel", set(), 0)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _read_cache_file(path: str) -> Optional[NKICompileResult]:
    """Read and deserialize a cache JSON file. Returns None on any failure."""
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)

        if data.get("schema_version") != _SCHEMA_VERSION:
            return None

        dumped_config = data["dumped_config"]
        kernel_binary = data.get("kernel_binary")
        if kernel_binary is not None:
            binary_path = os.path.abspath(
                os.path.join(os.path.dirname(path), kernel_binary)
            )
            if not os.path.isfile(binary_path):
                logger.debug(
                    "NKI cache entry %s references missing kernel binary %s",
                    path,
                    binary_path,
                )
                return None
            dumped_config = _replace_kernel_binary_path(dumped_config, binary_path)
        else:
            # Entries produced without a materialized binary are valid only
            # while their backend-config path still exists. This also makes
            # stale schema-v1 entries fail closed if a copy was interrupted.
            binary_path = _kernel_binary_path(dumped_config)
            if binary_path is not None and not os.path.isfile(binary_path):
                logger.debug(
                    "NKI cache entry %s references missing kernel binary %s",
                    path,
                    binary_path,
                )
                return None

        return_types = tuple(
            (str_to_torch_dtype(dtype_str), tuple(shape))
            for dtype_str, shape in data["return_types"]
        )

        operand_output_aliases = {
            int(k): v for k, v in data["operand_output_aliases"].items()
        }

        return NKICompileResult(
            dumped_config=dumped_config,
            return_types=return_types,
            operand_output_aliases=operand_output_aliases,
        )
    except (json.JSONDecodeError, KeyError, OSError, ValueError) as e:
        logger.debug("Failed to read NKI cache file %s: %s", path, e)
        return None


def _decode_backend_config(dumped_config: str) -> dict[str, Any]:
    payload = base64.b64decode(dumped_config.encode("ascii"), validate=True)
    config = json.loads(payload.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("NKI backend config must be a JSON object")
    return config


def _encode_backend_config(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(payload).decode("ascii")


def _kernel_binary_path(dumped_config: str) -> Optional[str]:
    """Return the BIR/KLIR file referenced by an NKI backend config."""
    try:
        config = _decode_backend_config(dumped_config)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return None
    klir_binary = config.get("klir_binary")
    if not isinstance(klir_binary, dict):
        return None
    binary_path = klir_binary.get("binary")
    return binary_path if isinstance(binary_path, str) else None


def _replace_kernel_binary_path(dumped_config: str, binary_path: str) -> str:
    config = _decode_backend_config(dumped_config)
    klir_binary = config.get("klir_binary")
    if not isinstance(klir_binary, dict):
        raise ValueError("NKI backend config has no klir_binary object")
    klir_binary["binary"] = binary_path
    return _encode_backend_config(config)


def _persist_kernel_binary(
    cache_key: str,
    result: NKICompileResult,
    local_dir: str,
) -> tuple[NKICompileResult, Optional[str]]:
    """Copy the ephemeral NKI binary into the persistent compile cache.

    ``CompileKernel`` emits backend config that names a file below
    ``/var/tmp/nki-intermediate-cache``. Persisting only the base64 config
    makes an HLO cache hit unusable after its container exits. Store the
    referenced file with the cache record and make the record relocatable.
    """
    source = _kernel_binary_path(result.dumped_config)
    if source is None or not os.path.isfile(source):
        if source is not None:
            logger.warning(
                "NKI kernel binary does not exist while caching %s: %s",
                cache_key,
                source,
            )
        return result, None

    extension = os.path.splitext(source)[1] or ".bin"
    relative_path = os.path.join(_NKI_BINARY_SUBDIR, f"{cache_key}{extension}")
    destination = os.path.join(local_dir, relative_path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    tmp_destination = f"{destination}.tmp.{os.getpid()}"
    try:
        shutil.copy2(source, tmp_destination)
        os.replace(tmp_destination, destination)
        persisted = NKICompileResult(
            dumped_config=_replace_kernel_binary_path(
                result.dumped_config, os.path.abspath(destination)
            ),
            return_types=result.return_types,
            operand_output_aliases=result.operand_output_aliases,
        )
        return persisted, relative_path
    except OSError as error:
        logger.warning(
            "Failed to persist NKI kernel binary for %s: %s",
            cache_key,
            error,
        )
        try:
            os.unlink(tmp_destination)
        except OSError:
            pass
        return result, None


def _copy_to_local(remote_path: str, local_path: str) -> None:
    """Best-effort copy of a remote cache entry to local."""
    try:
        local_dir = os.path.dirname(local_path)
        os.makedirs(local_dir, exist_ok=True)
        tmp_path = f"{local_path}.tmp.{os.getpid()}"
        shutil.copy2(remote_path, tmp_path)
        os.replace(tmp_path, local_path)
    except OSError as e:
        logger.debug("Failed to copy NKI cache entry to local: %s", e)
