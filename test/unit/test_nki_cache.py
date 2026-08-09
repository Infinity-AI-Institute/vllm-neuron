"""Regression tests for portable NKI compile-cache records."""

import base64
import enum
import importlib.util
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.nki import nki_cache
from vllm_neuron.nki.nki_compile import NKICompileResult


def _backend_config(binary_path) -> str:
    config = {
        "func_name": "test.kernel",
        "kernel_format": "bir",
        "klir_binary": {
            "binary": str(binary_path),
            "input_names": ["input"],
            "output_names": ["output"],
        },
    }
    return base64.b64encode(json.dumps(config).encode("utf-8")).decode("ascii")


def _binary_path(dumped_config: str) -> str:
    config = json.loads(base64.b64decode(dumped_config).decode("utf-8"))
    return config["klir_binary"]["binary"]


@pytest.fixture
def stable_key_environment(monkeypatch):
    monkeypatch.setattr(nki_cache, "get_platform_target", lambda: "trn2")
    monkeypatch.setattr(nki_cache, "get_nki_version", lambda: "nki-test")
    monkeypatch.setattr(nki_cache, "get_neuronxcc_version", lambda: "cc-test")
    monkeypatch.setattr(
        nki_cache, "get_torch_neuronx_version", lambda: "torch-neuronx-test"
    )
    monkeypatch.setattr(nki_cache.envs, "VLLM_NEURON_KERNEL_DEVICE_DUMP", False)
    monkeypatch.delenv("VLLM_NEURON_NKI_SOURCE_IDENTITY", raising=False)


def _dynamic_kernel_namespace():
    namespace = {"GLOBAL_SCALE": 2}
    exec(  # noqa: S102 - isolated namespace models mutable kernel helpers
        "def helper(value):\n"
        "    return value + 1\n"
        "def kernel(value):\n"
        "    return helper(value) * GLOBAL_SCALE\n",
        namespace,
    )
    return namespace


def _key(kernel, value, grid=(2,)):
    return nki_cache.create_nki_cache_key(kernel, {"value": value}, grid)


def test_nki_cache_materializes_ephemeral_kernel_binary(monkeypatch, tmp_path):
    cache_dir = tmp_path / "compile-cache" / "nki"
    ephemeral_binary = tmp_path / "ephemeral" / "kernel.json"
    ephemeral_binary.parent.mkdir()
    ephemeral_binary.write_text('{"kernel": "bir"}')
    monkeypatch.setattr(nki_cache, "_get_local_nki_cache_dir", lambda: str(cache_dir))
    result = NKICompileResult(
        dumped_config=_backend_config(ephemeral_binary),
        return_types=((torch.bfloat16, (1, 8)),),
        operand_output_aliases={},
    )

    nki_cache.put_nki_cache("portable", result)
    ephemeral_binary.unlink()
    cached = nki_cache.get_nki_cache("portable")

    assert cached is not None
    persisted_binary = _binary_path(cached.dumped_config)
    assert persisted_binary != str(ephemeral_binary)
    assert (cache_dir / "binaries" / "portable.json").read_text() == '{"kernel": "bir"}'


def test_nki_cache_rewrites_binary_path_after_cache_move(monkeypatch, tmp_path):
    first_cache = tmp_path / "first" / "nki"
    ephemeral_binary = tmp_path / "kernel.json"
    ephemeral_binary.write_text('{"kernel": "bir"}')
    monkeypatch.setattr(nki_cache, "_get_local_nki_cache_dir", lambda: str(first_cache))
    result = NKICompileResult(
        dumped_config=_backend_config(ephemeral_binary),
        return_types=((torch.bfloat16, (1,)),),
        operand_output_aliases={},
    )
    nki_cache.put_nki_cache("movable", result)

    second_cache = tmp_path / "second" / "nki"
    shutil.copytree(first_cache, second_cache)
    monkeypatch.setattr(
        nki_cache, "_get_local_nki_cache_dir", lambda: str(second_cache)
    )
    cached = nki_cache.get_nki_cache("movable")

    assert cached is not None
    assert _binary_path(cached.dumped_config).startswith(str(second_cache))


def test_nki_cache_rejects_missing_materialized_binary(monkeypatch, tmp_path):
    cache_dir = tmp_path / "compile-cache" / "nki"
    ephemeral_binary = tmp_path / "kernel.json"
    ephemeral_binary.write_text('{"kernel": "bir"}')
    monkeypatch.setattr(nki_cache, "_get_local_nki_cache_dir", lambda: str(cache_dir))
    result = NKICompileResult(
        dumped_config=_backend_config(ephemeral_binary),
        return_types=((torch.bfloat16, (1,)),),
        operand_output_aliases={},
    )
    nki_cache.put_nki_cache("missing", result)
    (cache_dir / "binaries" / "missing.json").unlink()

    assert nki_cache.get_nki_cache("missing") is None


def test_compile_with_cache_reuses_result_without_reopening_disk(monkeypatch, tmp_path):
    nki_cache.clear_nki_process_cache()
    cache_dir = tmp_path / "compile-cache" / "nki"
    monkeypatch.setattr(nki_cache, "_get_local_nki_cache_dir", lambda: str(cache_dir))
    result = NKICompileResult(
        dumped_config="cached",
        return_types=((torch.bfloat16, (1,)),),
        operand_output_aliases={},
    )
    disk_reads = 0
    compiles = 0

    def read_once(cache_key, local_dir):
        nonlocal disk_reads
        assert cache_key == "same-key"
        assert local_dir == str(cache_dir)
        disk_reads += 1
        return result

    def must_not_compile():
        nonlocal compiles
        compiles += 1
        return result

    monkeypatch.setattr(nki_cache, "_get_nki_cache_from_dir", read_once)

    first = nki_cache.compile_with_cache("same-key", must_not_compile)
    second = nki_cache.compile_with_cache("same-key", must_not_compile)

    assert first is result
    assert second is result
    assert disk_reads == 1
    assert compiles == 0
    assert nki_cache.nki_process_cache_stats() == {
        "entries": 1,
        "hits": 1,
        "misses": 1,
    }


def test_disabled_compile_cache_does_not_use_process_cache(monkeypatch):
    nki_cache.clear_nki_process_cache()
    monkeypatch.setattr(nki_cache.envs, "VLLM_NEURON_DISABLE_COMPILE_CACHE", True)
    calls = 0

    def compile_each_time():
        nonlocal calls
        calls += 1
        return NKICompileResult(
            dumped_config=str(calls),
            return_types=((torch.bfloat16, (1,)),),
            operand_output_aliases={},
        )

    first = nki_cache.compile_with_cache("disabled", compile_each_time)
    second = nki_cache.compile_with_cache("disabled", compile_each_time)

    assert first.dumped_config == "1"
    assert second.dumped_config == "2"
    assert nki_cache.nki_process_cache_stats() == {
        "entries": 0,
        "hits": 0,
        "misses": 0,
    }


def test_process_cache_is_scoped_to_local_cache_root(monkeypatch, tmp_path):
    nki_cache.clear_nki_process_cache()
    roots = {
        "first": tmp_path / "first" / "nki",
        "second": tmp_path / "second" / "nki",
    }
    active_root = "first"
    monkeypatch.setattr(
        nki_cache,
        "_get_local_nki_cache_dir",
        lambda: str(roots[active_root]),
    )

    first_result = NKICompileResult(
        dumped_config="first-root",
        return_types=((torch.bfloat16, (1,)),),
        operand_output_aliases={},
    )
    second_result = NKICompileResult(
        dumped_config="second-root",
        return_types=((torch.bfloat16, (1,)),),
        operand_output_aliases={},
    )
    nki_cache.put_nki_cache("shared-key", first_result)
    active_root = "second"
    nki_cache.put_nki_cache("shared-key", second_result)

    active_root = "first"
    first = nki_cache.compile_with_cache("shared-key", lambda: first_result)
    active_root = "second"
    second = nki_cache.compile_with_cache("shared-key", lambda: second_result)

    assert first.dumped_config == "first-root"
    assert second.dumped_config == "second-root"
    assert nki_cache.nki_process_cache_stats()["entries"] == 2


def test_process_cache_revalidates_materialized_binary(monkeypatch, tmp_path):
    nki_cache.clear_nki_process_cache()
    cache_dir = tmp_path / "compile-cache" / "nki"
    ephemeral_binary = tmp_path / "ephemeral" / "kernel.json"
    ephemeral_binary.parent.mkdir()
    ephemeral_binary.write_text('{"kernel": "first"}')
    monkeypatch.setattr(nki_cache, "_get_local_nki_cache_dir", lambda: str(cache_dir))
    first_result = NKICompileResult(
        dumped_config=_backend_config(ephemeral_binary),
        return_types=((torch.bfloat16, (1,)),),
        operand_output_aliases={},
    )

    first = nki_cache.compile_with_cache("revalidate", lambda: first_result)
    materialized_binary = _binary_path(first.dumped_config)
    assert materialized_binary != str(ephemeral_binary)
    (cache_dir / "binaries" / "revalidate.json").unlink()

    replacement = NKICompileResult(
        dumped_config="replacement",
        return_types=((torch.bfloat16, (2,)),),
        operand_output_aliases={},
    )
    second = nki_cache.compile_with_cache("revalidate", lambda: replacement)

    assert second is replacement


def test_remote_hit_materializes_binary_in_local_root(monkeypatch, tmp_path):
    nki_cache.clear_nki_process_cache()
    remote_dir = tmp_path / "remote" / "nki"
    local_dir = tmp_path / "local" / "nki"
    ephemeral_binary = tmp_path / "ephemeral" / "kernel.json"
    ephemeral_binary.parent.mkdir()
    ephemeral_binary.write_text('{"kernel": "remote"}')
    monkeypatch.setattr(nki_cache, "_get_local_nki_cache_dir", lambda: str(remote_dir))
    nki_cache.put_nki_cache(
        "remote-hit",
        NKICompileResult(
            dumped_config=_backend_config(ephemeral_binary),
            return_types=((torch.bfloat16, (1,)),),
            operand_output_aliases={},
        ),
    )

    monkeypatch.setattr(nki_cache, "_get_local_nki_cache_dir", lambda: str(local_dir))
    monkeypatch.setattr(nki_cache, "_get_remote_nki_cache_dir", lambda: str(remote_dir))

    def must_not_compile():
        raise AssertionError("unexpected compile")

    cached = nki_cache.compile_with_cache("remote-hit", must_not_compile)

    binary_path = _binary_path(cached.dumped_config)
    assert binary_path.startswith(str(local_dir))
    assert (local_dir / "binaries" / "remote-hit.json").is_file()


def test_compile_cache_mode_change_invalidates_process_entries(monkeypatch, tmp_path):
    nki_cache.clear_nki_process_cache()
    cache_dir = tmp_path / "compile-cache" / "nki"
    monkeypatch.setattr(nki_cache, "_get_local_nki_cache_dir", lambda: str(cache_dir))
    monkeypatch.setattr(nki_cache.envs, "VLLM_NEURON_DISABLE_COMPILE_CACHE", False)
    original = NKICompileResult(
        dumped_config="original",
        return_types=((torch.bfloat16, (1,)),),
        operand_output_aliases={},
    )
    nki_cache.put_nki_cache("mode-change", original)
    assert nki_cache.compile_with_cache("mode-change", lambda: original) is not None

    monkeypatch.setattr(nki_cache.envs, "VLLM_NEURON_DISABLE_COMPILE_CACHE", True)
    bypassed = NKICompileResult(
        dumped_config="bypassed",
        return_types=((torch.bfloat16, (2,)),),
        operand_output_aliases={},
    )
    assert nki_cache.compile_with_cache("mode-change", lambda: bypassed) is bypassed
    replacement = NKICompileResult(
        dumped_config="replacement",
        return_types=((torch.bfloat16, (3,)),),
        operand_output_aliases={},
    )
    nki_cache.put_nki_cache("mode-change", replacement)

    monkeypatch.setattr(nki_cache.envs, "VLLM_NEURON_DISABLE_COMPILE_CACHE", False)
    reenabled = nki_cache.compile_with_cache("mode-change", lambda: original)

    assert reenabled.dumped_config == "replacement"


def test_concurrent_same_key_compiles_once(monkeypatch, tmp_path):
    nki_cache.clear_nki_process_cache()
    cache_dir = tmp_path / "compile-cache" / "nki"
    monkeypatch.setattr(nki_cache, "_get_local_nki_cache_dir", lambda: str(cache_dir))
    compile_calls = 0

    def compile_once():
        nonlocal compile_calls
        compile_calls += 1
        return NKICompileResult(
            dumped_config="thread-safe",
            return_types=((torch.bfloat16, (1,)),),
            operand_output_aliases={},
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: nki_cache.compile_with_cache("threaded", compile_once),
                range(4),
            )
        )

    assert compile_calls == 1
    assert {result.dumped_config for result in results} == {"thread-safe"}


def test_semantically_identical_kernel_invocations_hit_same_key(
    stable_key_environment,
):
    namespace = _dynamic_kernel_namespace()
    tensor = torch.empty((2, 3), dtype=torch.bfloat16)

    first = _key(namespace["kernel"], tensor)
    second = _key(namespace["kernel"], tensor)

    assert first is not None
    assert first == second


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (torch.empty((2, 3)), torch.empty((2, 4))),
        (
            torch.empty((2, 3), dtype=torch.float32),
            torch.empty((2, 3), dtype=torch.bfloat16),
        ),
        (torch.empty((2, 3)), torch.empty((3, 2)).t()),
        (
            torch.empty(16).as_strided((2, 3), (3, 1), 0),
            torch.empty(16).as_strided((2, 3), (3, 1), 1),
        ),
        (torch.empty((2, 3), device="cpu"), torch.empty((2, 3), device="meta")),
    ],
    ids=["shape", "dtype", "stride", "storage-offset", "device-role"],
)
def test_tensor_semantic_changes_miss(
    stable_key_environment,
    first,
    second,
):
    kernel = _dynamic_kernel_namespace()["kernel"]

    assert _key(kernel, first) != _key(kernel, second)


def test_tensor_layout_change_misses(stable_key_environment):
    kernel = _dynamic_kernel_namespace()["kernel"]

    def descriptor(layout):
        return SimpleNamespace(
            shape=(2, 3),
            dtype=torch.bfloat16,
            layout=layout,
            device=torch.device("meta"),
            stride=lambda: (3, 1),
            storage_offset=lambda: 0,
        )

    assert _key(kernel, descriptor("torch.strided")) != _key(
        kernel, descriptor("torch.sparse_coo")
    )


def test_grid_change_misses(stable_key_environment):
    namespace = _dynamic_kernel_namespace()
    tensor = torch.empty((2, 3), device="meta")

    assert _key(namespace["kernel"], tensor, (1,)) != _key(
        namespace["kernel"], tensor, (2,)
    )


def test_transitive_helper_mutation_misses(stable_key_environment):
    namespace = _dynamic_kernel_namespace()
    tensor = torch.empty((2, 3), device="meta")
    first = _key(namespace["kernel"], tensor)
    exec(  # noqa: S102 - isolated namespace models a helper source edit
        "def helper(value):\n    return value + 2\n", namespace
    )

    second = _key(namespace["kernel"], tensor)

    assert first is not None
    assert second is not None
    assert first != second


def test_specialization_global_mutation_misses(stable_key_environment):
    namespace = _dynamic_kernel_namespace()
    tensor = torch.empty((2, 3), device="meta")
    first = _key(namespace["kernel"], tensor)
    namespace["GLOBAL_SCALE"] = 3

    second = _key(namespace["kernel"], tensor)

    assert first is not None
    assert second is not None
    assert first != second


def test_source_revision_change_misses(stable_key_environment, monkeypatch):
    namespace = _dynamic_kernel_namespace()
    tensor = torch.empty((2, 3), device="meta")
    monkeypatch.setenv("VLLM_NEURON_NKI_SOURCE_IDENTITY", "revision-a")
    first = _key(namespace["kernel"], tensor)
    monkeypatch.setenv("VLLM_NEURON_NKI_SOURCE_IDENTITY", "revision-b")

    second = _key(namespace["kernel"], tensor)

    assert first is not None
    assert second is not None
    assert first != second


def test_sealed_source_file_mutation_misses(stable_key_environment, tmp_path):
    module_path = tmp_path / "kernel_source.py"
    module_path.write_text("def kernel(value):\n    return value + 1\n")
    spec = importlib.util.spec_from_file_location("cache_key_test_kernel", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    tensor = torch.empty((2, 3), device="meta")
    first = _key(module.kernel, tensor)
    module_path.write_text("def kernel(value):\n    return value + 1\n# revision two\n")

    second = _key(module.kernel, tensor)

    assert first is not None
    assert second is not None
    assert first != second


def test_nondeterministic_specialization_value_fails_closed(
    stable_key_environment,
):
    namespace = _dynamic_kernel_namespace()

    assert _key(namespace["kernel"], object()) is None


def test_cache_identity_does_not_execute_mapping_key_repr(stable_key_environment):
    class HostileEnum(enum.Enum):
        VALUE = 1

        def __repr__(self):
            raise AssertionError("mapping-key repr must not execute")

    namespace = _dynamic_kernel_namespace()

    assert _key(namespace["kernel"], {HostileEnum.VALUE: 1}) is not None


def test_nondeterministic_global_fails_closed(stable_key_environment):
    namespace = _dynamic_kernel_namespace()
    namespace["GLOBAL_SCALE"] = object()

    assert _key(namespace["kernel"], torch.empty(1, device="meta")) is None


def test_dynamic_class_global_fails_closed(stable_key_environment):
    namespace = _dynamic_kernel_namespace()
    namespace["DynamicClass"] = type("DynamicClass", (), {})
    exec(  # noqa: S102 - isolated namespace models generated kernel code
        "def kernel(value):\n    return DynamicClass, value\n", namespace
    )

    assert _key(namespace["kernel"], torch.empty(1, device="meta")) is None


def test_schema_v1_entry_is_not_reused(tmp_path):
    cache_path = tmp_path / "legacy.json"
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dumped_config": "legacy",
                "return_types": [["torch.bfloat16", [1]]],
                "operand_output_aliases": {},
            }
        )
    )

    assert nki_cache._read_cache_file(str(cache_path)) is None


def test_pid_change_drops_inherited_process_cache(monkeypatch):
    nki_cache.clear_nki_process_cache()
    process_key = ("local", "remote", "pid-change")
    result = NKICompileResult(
        dumped_config="parent",
        return_types=((torch.bfloat16, (1,)),),
        operand_output_aliases={},
    )
    generation = nki_cache._PROCESS_CACHE_GENERATION
    nki_cache._put_nki_process_cache(process_key, result, generation)
    parent_pid = nki_cache._PROCESS_CACHE_PID

    monkeypatch.setattr(nki_cache.os, "getpid", lambda: parent_pid + 1)

    assert nki_cache.nki_process_cache_stats() == {
        "entries": 0,
        "hits": 0,
        "misses": 0,
    }


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_child_drops_inherited_process_cache():
    nki_cache.clear_nki_process_cache()
    process_key = ("local", "remote", "fork-key")
    result = NKICompileResult(
        dumped_config="fork-parent",
        return_types=((torch.bfloat16, (1,)),),
        operand_output_aliases={},
    )
    generation = nki_cache._PROCESS_CACHE_GENERATION
    nki_cache._put_nki_process_cache(process_key, result, generation)
    assert nki_cache.nki_process_cache_stats()["entries"] == 1

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(read_fd)
            child_stats = nki_cache.nki_process_cache_stats()
            os.write(write_fd, json.dumps(child_stats).encode("utf-8"))
        finally:
            os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    child_payload = os.read(read_fd, 4096)
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)

    assert status == 0
    assert json.loads(child_payload) == {"entries": 0, "hits": 0, "misses": 0}
    assert nki_cache.nki_process_cache_stats()["entries"] == 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_child_reinitializes_an_inherited_locked_cache_lock():
    nki_cache.clear_nki_process_cache()
    parent_pid = os.getpid()
    read_fd, write_fd = os.pipe()
    nki_cache._PROCESS_CACHE_LOCK.acquire()
    try:
        pid = os.fork()
    finally:
        # Only the parent still owns the original lock. The registered child
        # hook replaced its inherited copy before Python resumed after fork.
        if os.getpid() == parent_pid:
            nki_cache._PROCESS_CACHE_LOCK.release()

    if pid == 0:
        try:
            os.close(read_fd)
            child_stats = nki_cache.nki_process_cache_stats()
            os.write(write_fd, json.dumps(child_stats).encode("utf-8"))
        finally:
            os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    child_payload = os.read(read_fd, 4096)
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)

    assert status == 0
    assert json.loads(child_payload) == {"entries": 0, "hits": 0, "misses": 0}


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_fork_child_reinitializes_an_inherited_source_digest_lock():
    parent_pid = os.getpid()
    read_fd, write_fd = os.pipe()
    nki_cache._SOURCE_DIGEST_LOCK.acquire()
    try:
        pid = os.fork()
    finally:
        if os.getpid() == parent_pid:
            nki_cache._SOURCE_DIGEST_LOCK.release()

    if pid == 0:
        try:
            os.close(read_fd)
            digest = nki_cache._sealed_source_digest(__file__)
            os.write(write_fd, digest.encode("ascii"))
        finally:
            os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    child_payload = os.read(read_fd, 4096)
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)

    assert status == 0
    assert len(child_payload) == 64
