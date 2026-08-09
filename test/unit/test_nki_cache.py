"""Regression tests for portable NKI compile-cache records."""

import base64
import json
import multiprocessing
import os
import shutil
import time

import torch
from _nki_test_loader import load_nki_modules

nki_cache, nki_compile, _ = load_nki_modules()
NKICompileResult = nki_compile.NKICompileResult


def _cache_result(value: str = "compiled") -> NKICompileResult:
    dumped_config = base64.b64encode(
        json.dumps({"value": value}).encode("utf-8")
    ).decode("ascii")
    return NKICompileResult(
        dumped_config=dumped_config,
        return_types=((torch.bfloat16, (1,)),),
        operand_output_aliases={},
    )


def _compile_cache_process_worker(
    cache_dir: str,
    compile_marker: str,
    start_event,
    result_queue,
) -> None:
    """Exercise the real file lock from an independently spawned process."""
    os.environ.pop("VLLM_NEURON_DISABLE_COMPILE_CACHE", None)
    os.environ.pop("VLLM_NEURON_REMOTE_CACHE", None)
    nki_cache._get_local_nki_cache_dir = lambda: cache_dir
    start_event.wait(timeout=10)

    def compile_once() -> NKICompileResult:
        with open(compile_marker, "a", encoding="utf-8") as marker:
            marker.write(f"{os.getpid()}\n")
            marker.flush()
        time.sleep(0.2)
        return _cache_result()

    result = nki_cache.compile_with_cache("shared-process-key", compile_once)
    result_queue.put(result.dumped_config)


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


def test_compile_cache_hit_skips_recompilation_and_emits_events(
    monkeypatch, tmp_path, caplog
):
    cache_dir = tmp_path / "compile-cache" / "nki"
    monkeypatch.setattr(nki_cache, "_get_local_nki_cache_dir", lambda: str(cache_dir))
    monkeypatch.setattr(nki_cache, "_fetch_from_remote", lambda _key: None)
    compile_calls = 0

    def compile_once() -> NKICompileResult:
        nonlocal compile_calls
        compile_calls += 1
        return _cache_result()

    caplog.set_level("DEBUG", logger="vllm_neuron.nki.nki_compile")
    first = nki_cache.compile_with_cache("warm-key", compile_once)
    second = nki_cache.compile_with_cache("warm-key", compile_once)

    assert first == second
    assert compile_calls == 1
    events = [
        json.loads(record.getMessage().split("NKI_COMPILE_EVENT ", 1)[1])
        for record in caplog.records
        if "NKI_COMPILE_EVENT " in record.getMessage()
    ]
    assert any(
        event["event"] == "cache_lookup"
        and event["stage"] == "initial"
        and event["outcome"] == "miss"
        for event in events
    )
    assert any(
        event["event"] == "cache_write"
        and event["outcome"] == "stored"
        and event["cache_key"] == "warm-key"
        for event in events
    )
    assert any(
        event["event"] == "cache_lookup"
        and event["stage"] == "initial"
        and event["outcome"] == "hit"
        for event in events
    )
    assert all(event["duration_ms"] >= 0 for event in events if "duration_ms" in event)


def test_compile_cache_serializes_spawned_processes(tmp_path):
    """Only the lock winner compiles; waiters consume its persisted result."""
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue()
    cache_dir = tmp_path / "compile-cache" / "nki"
    compile_marker = tmp_path / "compile-calls.txt"
    processes = [
        context.Process(
            target=_compile_cache_process_worker,
            args=(str(cache_dir), str(compile_marker), start_event, result_queue),
        )
        for _ in range(3)
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=30)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(timeout=5)

    assert not alive
    assert [process.exitcode for process in processes] == [0, 0, 0]
    assert len(compile_marker.read_text().splitlines()) == 1
    results = [result_queue.get(timeout=5) for _ in processes]
    assert results == [results[0]] * len(processes)
