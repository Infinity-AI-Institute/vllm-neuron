# SPDX-License-Identifier: Apache-2.0
"""Persistent disk cache for NKI kernel compile results.

Caches NKICompileResult to disk so that warm starts skip NKI BIR
compilation. Compatible with vLLM Neuron's local + remote cache layout.

Storage: JSON files in {cache_dir}/nki/{key}.json
Multi-process safety: filelock.FileLock per cache key
"""

import base64
import errno
import hashlib
import inspect
import json
import logging
import os
import shutil
import threading
from collections.abc import Callable, Hashable
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

_SCHEMA_VERSION = 1
_NKI_CACHE_SUBDIR = "nki"
_NKI_BINARY_SUBDIR = "binaries"
_LOCK_TIMEOUT = 300  # seconds

# One K3 prefill graph can contain tens of thousands of calls to the same NKI
# kernel configurations.  The persistent cache prevents recompilation, but a
# hit still opens and parses the same JSON record (and validates its binary)
# for every FX node.  Keep successful results in the tracing process after the
# first persistent lookup.  This is deliberately below create_nki_cache_key():
# source, shape, dtype, constants, grid, platform, versions, and device-dump
# mode therefore retain exactly the same invalidation contract as the disk
# cache.
_PROCESS_CACHE: dict[str, NKICompileResult] = {}
_PROCESS_CACHE_LOCK = threading.Lock()
_PROCESS_CACHE_HITS = 0
_PROCESS_CACHE_MISSES = 0


def clear_nki_process_cache() -> None:
    """Clear process-local NKI results and counters.

    Production code should not need this; it is an explicit seam for tests and
    long-lived tooling that intentionally changes cache roots or kernel state.
    """

    global _PROCESS_CACHE_HITS, _PROCESS_CACHE_MISSES
    with _PROCESS_CACHE_LOCK:
        _PROCESS_CACHE.clear()
        _PROCESS_CACHE_HITS = 0
        _PROCESS_CACHE_MISSES = 0


def nki_process_cache_stats() -> dict[str, int]:
    """Return cheap instrumentation for repeated in-process cache lookups."""

    with _PROCESS_CACHE_LOCK:
        return {
            "entries": len(_PROCESS_CACHE),
            "hits": _PROCESS_CACHE_HITS,
            "misses": _PROCESS_CACHE_MISSES,
        }


def _get_nki_process_cache(cache_key: str) -> Optional[NKICompileResult]:
    global _PROCESS_CACHE_HITS, _PROCESS_CACHE_MISSES
    with _PROCESS_CACHE_LOCK:
        cached = _PROCESS_CACHE.get(cache_key)
        if cached is None:
            _PROCESS_CACHE_MISSES += 1
        else:
            _PROCESS_CACHE_HITS += 1
        return cached


def _put_nki_process_cache(
    cache_key: str, result: NKICompileResult
) -> NKICompileResult:
    with _PROCESS_CACHE_LOCK:
        # Preserve the first complete result if concurrent tracing lanes ever
        # share a process.  The persistent key guarantees equivalence.
        return _PROCESS_CACHE.setdefault(cache_key, result)


def create_nki_cache_key(
    func: Callable,
    args: dict[str, Any],
    grid: tuple[int, ...],
) -> Optional[str]:
    """Generate a persistent cache key for an NKI kernel invocation.

    Returns None if any argument is unhashable or if required version/platform
    information cannot be obtained (e.g. no Neuron runtime available).
    """
    try:
        key_parts = []

        # Kernel identity
        key_parts.append(os.path.basename(func.__code__.co_filename))
        key_parts.append(func.__qualname__)

        # Kernel source hash — invalidates cache when kernel code is edited
        key_parts.append(_hash_kernel_source(func))

        # Input shapes/dtypes
        for name, v in args.items():
            h, is_hashable = _hashable_arg(v, name)
            if not is_hashable:
                return None
            key_parts.append(h)

        # Grid/LNC
        key_parts.append(grid)

        # Platform + versions for cache invalidation
        key_parts.append(get_platform_target())
        key_parts.append(get_nki_version())
        key_parts.append(get_neuronxcc_version())
        key_parts.append(get_torch_neuronx_version())

        # Device dump inserts DevicePrint ops into the kernel, producing a different binary.
        key_parts.append(envs.VLLM_NEURON_KERNEL_DEVICE_DUMP)

        return hashlib.md5("|".join(str(p) for p in key_parts).encode()).hexdigest()[
            :32
        ]
    except Exception as e:
        logger.debug("Cannot generate NKI cache key: %s", e)
        return None


def get_nki_cache(cache_key: str) -> Optional[NKICompileResult]:
    """Load a cached NKICompileResult from local disk cache.

    Returns None on miss. Does not check remote — use
    ``_fetch_from_remote`` inside a locked block for that.
    """
    local_dir = _get_local_nki_cache_dir()
    local_path = os.path.join(local_dir, f"{cache_key}.json")

    result = _read_cache_file(local_path)
    if result is not None:
        logger.debug("NKI cache hit (local): %s", cache_key)
        return result

    return None


def put_nki_cache(cache_key: str, result: NKICompileResult) -> None:
    """Store an NKICompileResult to the local disk cache.

    Uses atomic write (tmp file + rename) for safety.
    """
    local_dir = _get_local_nki_cache_dir()
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
        os.rename(tmp_path, local_path)
        logger.debug("NKI cache stored: %s", cache_key)
    except OSError as e:
        logger.warning("Failed to write NKI cache entry %s: %s", cache_key, e)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


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
    if cache_key is None or envs.VLLM_NEURON_DISABLE_COMPILE_CACHE:
        return compile_fn()

    cached = _get_nki_process_cache(cache_key)
    if cached is not None:
        return cached

    # Fast path: lockless local read
    cached = get_nki_cache(cache_key)
    if cached is not None:
        return _put_nki_process_cache(cache_key, cached)

    # Slow path: acquire lock, try remote fetch, then compile
    local_dir = _get_local_nki_cache_dir()
    os.makedirs(local_dir, exist_ok=True)
    lock_path = os.path.join(local_dir, f"{cache_key}.lock")

    try:
        with FileLock(lock_path, timeout=_LOCK_TIMEOUT):
            # Double-check local after acquiring lock
            cached = get_nki_cache(cache_key)
            if cached is not None:
                return _put_nki_process_cache(cache_key, cached)

            # Try remote fetch (only lock winner hits remote filesystem)
            cached = _fetch_from_remote(cache_key)
            if cached is not None:
                return _put_nki_process_cache(cache_key, cached)

            result = compile_fn()
            put_nki_cache(cache_key, result)
            return _put_nki_process_cache(cache_key, result)
    except Timeout:
        logger.warning(
            "NKI cache lock timeout for %s, compiling without cache", cache_key
        )
        return _put_nki_process_cache(cache_key, compile_fn())


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
    remote_dir = _get_remote_nki_cache_dir()
    if not remote_dir:
        return None

    remote_path = os.path.join(remote_dir, f"{cache_key}.json")
    result = _read_cache_file(remote_path)
    if result is not None:
        logger.debug("NKI cache hit (remote): %s", cache_key)
        local_dir = _get_local_nki_cache_dir()
        local_path = os.path.join(local_dir, f"{cache_key}.json")
        _copy_to_local(remote_path, local_path)
        return result

    return None


def _hashable_arg(value: Any, name: str) -> tuple[Any, bool]:
    """Convert a kernel argument to a hashable representation for cache keying.

    Tensors are represented by (shape, dtype). Lists/dicts are recursively
    converted. Returns (hashable_value, True) on success, (None, False) if
    the argument cannot be hashed.
    """
    if isinstance(value, (tuple, list)):
        parts = []
        for i, e in enumerate(value):
            h, is_hashable = _hashable_arg(e, f"{name}[{i}]")
            if not is_hashable:
                return None, False
            parts.append(h)
        return tuple(parts), True
    if isinstance(value, dict):
        parts = []
        for k, v in sorted(value.items()):
            h, is_hashable = _hashable_arg(v, f"{name}[{k}]")
            if not is_hashable:
                return None, False
            parts.append((k, h))
        return tuple(parts), True
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = tuple(value.shape) if hasattr(value.shape, "__iter__") else value.shape
        return (shape, str(value.dtype)), True
    if isinstance(value, Hashable):
        return value, True
    return None, False


def _hash_kernel_source(func: Callable) -> str:
    """Hash the source code of a kernel function for cache invalidation.

    Falls back to hashing the bytecode if source is unavailable (e.g.
    dynamically generated kernels).
    """
    try:
        source = inspect.getsource(func)
        return hashlib.sha256(source.encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return hashlib.sha256(func.__code__.co_code).hexdigest()[:16]


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
        os.rename(tmp_path, local_path)
    except OSError as e:
        logger.debug("Failed to copy NKI cache entry to local: %s", e)
