import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[3] / "vllm_neuron" / "compile" / "rank_invariant_trace.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("rank_invariant_trace", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_rank_extracts_by_default():
    policy = _load_module()
    assert all(
        policy.should_extract_graphs(rank=rank, leader_only=False) for rank in range(64)
    )


def test_only_rank_zero_extracts_when_explicitly_enabled():
    policy = _load_module()
    extracting = [
        rank
        for rank in range(64)
        if policy.should_extract_graphs(rank=rank, leader_only=True)
    ]
    assert extracting == [0]


@pytest.mark.parametrize("rank", [-1, True, 1.5, "0", None])
def test_invalid_rank_fails_closed(rank):
    policy = _load_module()
    with pytest.raises(ValueError, match="rank must be a non-negative integer"):
        policy.should_extract_graphs(rank=rank, leader_only=True)
