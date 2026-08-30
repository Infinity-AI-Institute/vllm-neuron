"""Adversarial gates for the GLM-5.3-Flash Trn2 DSA pool selector."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch


def _load_topk_helper():
    """Load the CPU-safe wrapper module without importing top-level vLLM."""
    root = Path(__file__).parents[3]
    for name, path in (
        ("vllm_neuron", root / "vllm_neuron"),
        ("vllm_neuron.model", root / "vllm_neuron" / "model"),
        (
            "vllm_neuron.model.glm53_flash",
            root / "vllm_neuron" / "model" / "glm53_flash",
        ),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules.setdefault(name, package)
    return importlib.import_module(
        "vllm_neuron.model.glm53_flash.neuron_wrapper"
    )._dsa_pool_topk


_dsa_pool_topk = _load_topk_helper()


def _fake_topk_module(monkeypatch: pytest.MonkeyPatch, *, can_use: bool):
    functional = types.ModuleType("vllm_neuron.functional")
    functional.__path__ = []
    module = types.ModuleType("vllm_neuron.functional.topk")

    def topk(tensor, k, *, dim, gather_dim):
        return torch.topk(tensor, k, dim=dim, largest=True, sorted=True)

    module.topk = topk
    module._can_use_nki_topk = lambda *args, **kwargs: can_use
    monkeypatch.setitem(sys.modules, "vllm_neuron.functional", functional)
    monkeypatch.setitem(sys.modules, "vllm_neuron.functional.topk", module)


def test_dsa_pool_topk_preserves_values_indices_and_ties_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Trn2 seam must preserve torch.topk's observable contract on CPU."""
    scores = torch.tensor(
        [
            [[4.0, 4.0, 1.0, 3.0, 4.0, -2.0]],
            [[0.0, 2.0, 2.0, 2.0, -1.0, 5.0]],
        ],
        dtype=torch.float32,
    )
    expected_values, expected_indices = torch.topk(
        scores, k=3, dim=-1, largest=True, sorted=True
    )

    _fake_topk_module(monkeypatch, can_use=True)
    values, indices = _dsa_pool_topk(scores, k=3)

    torch.testing.assert_close(values, expected_values, rtol=0.0, atol=0.0)
    assert torch.equal(indices, expected_indices)
    assert indices.dtype == torch.int64


def test_dsa_pool_topk_all_pools_avoids_topk_and_preserves_pool_order() -> None:
    """S128's k==pools case must not lower to a full-vocabulary sort."""
    scores = torch.tensor([[[4.0, 4.0, 1.0, 4.0]]], dtype=torch.float32)

    values, indices = _dsa_pool_topk(scores, k=scores.shape[-1])

    torch.testing.assert_close(values, scores, rtol=0.0, atol=0.0)
    assert torch.equal(indices, torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int64))


def test_dsa_pool_topk_hardware_path_fails_closed_when_nki_shape_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never reintroduce an implicit XLA torch.topk→sort fallback."""
    _fake_topk_module(monkeypatch, can_use=False)
    topk_module = sys.modules["vllm_neuron.functional.topk"]

    class _FakeXlaScores:
        device = "xla:0"
        shape = (1, 1, 32)

    monkeypatch.setattr(topk_module, "_can_use_nki_topk", lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match="refusing torch.topk fallback"):
        _dsa_pool_topk(_FakeXlaScores(), k=8)  # type: ignore[arg-type]


def test_compiled_dsa_hlo_contains_no_unsupported_sort() -> None:
    """Validate an actual emitted HLO when ``GLM53_DSA_HLO_PATH`` is supplied.

    The environment-gated form keeps ordinary source CI Neuron-free; the
    host compile packet runs this same assertion against its retained HLO.
    """
    value = os.environ.get("GLM53_DSA_HLO_PATH")
    if not value:
        pytest.skip("set GLM53_DSA_HLO_PATH to gate a compiled DSA HLO")

    hlo = Path(value).read_bytes()
    assert b"%sort." not in hlo
    assert b'op_type="aten__topk"' not in hlo
