"""Validation tests for the opt-in trace-rank concurrency limit."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def _load_envs():
    path = ROOT / "vllm_neuron/envs.py"
    spec = importlib.util.spec_from_file_location("trace_throttle_envs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_rank_concurrency_is_disabled_only_when_unset(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_TRACE_RANK_CONCURRENCY", raising=False)
    assert _load_envs().VLLM_NEURON_TRACE_RANK_CONCURRENCY is None


@pytest.mark.parametrize("value", ["1", "64", "4096"])
def test_trace_rank_concurrency_accepts_bounded_positive_integers(
    monkeypatch,
    value,
):
    monkeypatch.setenv("VLLM_NEURON_TRACE_RANK_CONCURRENCY", value)
    assert _load_envs().VLLM_NEURON_TRACE_RANK_CONCURRENCY == int(value)


@pytest.mark.parametrize("value", ["", "0", "-1", "4097", "1.5", "no"])
def test_trace_rank_concurrency_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("VLLM_NEURON_TRACE_RANK_CONCURRENCY", value)
    with pytest.raises(ValueError, match=r"must be an integer in \[1, 4096\]"):
        _ = _load_envs().VLLM_NEURON_TRACE_RANK_CONCURRENCY


def test_trace_leader_only_is_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_TRACE_LEADER_ONLY", raising=False)
    assert _load_envs().VLLM_NEURON_TRACE_LEADER_ONLY is False

    monkeypatch.setenv("VLLM_NEURON_TRACE_LEADER_ONLY", "1")
    assert _load_envs().VLLM_NEURON_TRACE_LEADER_ONLY is True
