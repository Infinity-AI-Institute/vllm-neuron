"""Regression tests for portable NKI compile-cache records."""

import base64
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

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
