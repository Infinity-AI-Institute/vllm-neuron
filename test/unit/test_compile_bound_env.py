# SPDX-License-Identifier: Apache-2.0
"""Validation tests for the global compile concurrency env config."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def _load_envs():
    path = ROOT / "vllm_neuron/envs.py"
    spec = importlib.util.spec_from_file_location("compile_bound_envs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compile_max_global_disabled_when_unset_or_zero(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_COMPILE_MAX_GLOBAL", raising=False)
    assert _load_envs().VLLM_NEURON_COMPILE_MAX_GLOBAL is None
    monkeypatch.setenv("VLLM_NEURON_COMPILE_MAX_GLOBAL", "0")
    assert _load_envs().VLLM_NEURON_COMPILE_MAX_GLOBAL is None


@pytest.mark.parametrize("value", ["16", "20", "24"])
def test_compile_max_global_accepts_bounded_values(monkeypatch, value):
    monkeypatch.setenv("VLLM_NEURON_COMPILE_MAX_GLOBAL", value)
    assert _load_envs().VLLM_NEURON_COMPILE_MAX_GLOBAL == int(value)


@pytest.mark.parametrize("value", ["1", "15", "25", "-1", "1.5", "lots"])
def test_compile_max_global_rejects_out_of_range(monkeypatch, value):
    monkeypatch.setenv("VLLM_NEURON_COMPILE_MAX_GLOBAL", value)
    with pytest.raises(ValueError, match=r"\[16, 24\]"):
        _ = _load_envs().VLLM_NEURON_COMPILE_MAX_GLOBAL


def test_compile_sem_dir_passthrough(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_COMPILE_SEM_DIR", raising=False)
    assert _load_envs().VLLM_NEURON_COMPILE_SEM_DIR is None
    monkeypatch.setenv("VLLM_NEURON_COMPILE_SEM_DIR", "/shared/compile-sem")
    assert _load_envs().VLLM_NEURON_COMPILE_SEM_DIR == "/shared/compile-sem"


def test_compile_sem_timeout_defaults_to_zero(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_COMPILE_SEM_TIMEOUT", raising=False)
    assert _load_envs().VLLM_NEURON_COMPILE_SEM_TIMEOUT == 0
    monkeypatch.setenv("VLLM_NEURON_COMPILE_SEM_TIMEOUT", "300")
    assert _load_envs().VLLM_NEURON_COMPILE_SEM_TIMEOUT == 300
