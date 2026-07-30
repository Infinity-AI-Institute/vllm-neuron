"""Regression tests for portable NKI compile-cache records."""

import base64
import json
import shutil

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
