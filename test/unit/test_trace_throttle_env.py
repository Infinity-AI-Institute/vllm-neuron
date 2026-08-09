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


def test_trace_preflight_is_disabled_only_when_rank_is_unset(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_TRACE_PREFLIGHT_RANK", raising=False)
    assert _load_envs().VLLM_NEURON_TRACE_PREFLIGHT_RANK is None


@pytest.mark.parametrize("value", ["0", "1", "4095"])
def test_trace_preflight_rank_accepts_bounded_nonnegative_values(
    monkeypatch, value
):
    monkeypatch.setenv("VLLM_NEURON_TRACE_PREFLIGHT_RANK", value)
    assert _load_envs().VLLM_NEURON_TRACE_PREFLIGHT_RANK == int(value)


@pytest.mark.parametrize("value", ["", "-1", "4096", "1.5", "no"])
def test_trace_preflight_rank_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("VLLM_NEURON_TRACE_PREFLIGHT_RANK", value)
    with pytest.raises(ValueError, match=r"must be an integer in \[0, 4095\]"):
        _ = _load_envs().VLLM_NEURON_TRACE_PREFLIGHT_RANK


@pytest.mark.parametrize("value", ["0", "-1", "4097", "all"])
def test_trace_preflight_job_limit_fails_closed(monkeypatch, value):
    monkeypatch.setenv("VLLM_NEURON_TRACE_PREFLIGHT_JOBS", value)
    with pytest.raises(ValueError, match=r"must be an integer in \[1, 4096\]"):
        _ = _load_envs().VLLM_NEURON_TRACE_PREFLIGHT_JOBS


def test_trace_preflight_control_defaults_are_multi_hour(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_TRACE_PREFLIGHT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv(
        "VLLM_NEURON_TRACE_PREFLIGHT_HEARTBEAT_SECONDS", raising=False
    )
    envs = _load_envs()

    assert envs.VLLM_NEURON_TRACE_PREFLIGHT_TIMEOUT_SECONDS == 14400
    assert envs.VLLM_NEURON_TRACE_PREFLIGHT_HEARTBEAT_SECONDS == 300


@pytest.mark.parametrize("value", ["0", "-1", "86401", "1.5", "no"])
def test_trace_preflight_timeout_fails_closed(monkeypatch, value):
    monkeypatch.setenv("VLLM_NEURON_TRACE_PREFLIGHT_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match=r"must be an integer in \[1, 86400\]"):
        _ = _load_envs().VLLM_NEURON_TRACE_PREFLIGHT_TIMEOUT_SECONDS


@pytest.mark.parametrize("value", ["0", "-1", "3601", "1.5", "no"])
def test_trace_preflight_heartbeat_fails_closed(monkeypatch, value):
    monkeypatch.setenv("VLLM_NEURON_TRACE_PREFLIGHT_HEARTBEAT_SECONDS", value)
    with pytest.raises(ValueError, match=r"must be an integer in \[1, 3600\]"):
        _ = _load_envs().VLLM_NEURON_TRACE_PREFLIGHT_HEARTBEAT_SECONDS
