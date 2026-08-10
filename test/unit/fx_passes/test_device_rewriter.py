# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the FX device-rewriting pass."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.fx import Graph, GraphModule

ROOT = Path(__file__).parents[3]


def _load_module(monkeypatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def device_rewriter(monkeypatch):
    """Load the focused pass without importing vLLM or Neuron runtimes."""
    package = ModuleType("vllm_neuron")
    package.__path__ = [str(ROOT / "vllm_neuron")]
    monkeypatch.setitem(sys.modules, "vllm_neuron", package)

    fx_package = ModuleType("vllm_neuron.fx_passes")
    fx_package.__path__ = [str(ROOT / "vllm_neuron/fx_passes")]
    monkeypatch.setitem(sys.modules, "vllm_neuron.fx_passes", fx_package)
    _load_module(
        monkeypatch,
        "vllm_neuron.fx_passes.base",
        ROOT / "vllm_neuron/fx_passes/base.py",
    )
    return _load_module(
        monkeypatch,
        "vllm_neuron.fx_passes.device_rewriter",
        ROOT / "vllm_neuron/fx_passes/device_rewriter.py",
    )


def _graph_with_to(*to_args, **to_kwargs):
    graph = Graph()
    tensor = graph.placeholder("tensor")
    other = graph.placeholder("other")
    resolved_args = tuple(
        tensor if value == "__tensor__" else other if value == "__other__" else value
        for value in to_args
    )
    to_node = graph.call_method(
        "to", args=(tensor, *resolved_args), kwargs=to_kwargs
    )
    graph.output(to_node)
    return GraphModule({}, graph), to_node, tensor, other


def test_rewrites_positional_string_device_without_changing_other_semantics(
    device_rewriter,
):
    gm, node, tensor, _ = _graph_with_to(
        "meta", torch.bfloat16, True, True, memory_format=torch.contiguous_format
    )
    original_kwargs = dict(node.kwargs)

    _, metadata = device_rewriter.DeviceRewriterPass().run(gm)

    assert node.args == (tensor, "xla", torch.bfloat16, True, True)
    assert node.kwargs == original_kwargs
    assert metadata["rewrite_count"] == 1


def test_rewrites_positional_torch_device(device_rewriter):
    gm, node, tensor, _ = _graph_with_to(torch.device("meta"), non_blocking=True)

    device_rewriter.DeviceRewriterPass().run(gm)

    assert node.args == (tensor, torch.device("xla", index=0))
    assert node.kwargs == {"non_blocking": True}


def test_preserves_already_xla_positional_device(device_rewriter):
    xla_device = torch.device("xla", index=0)
    gm, node, tensor, _ = _graph_with_to(xla_device, copy=True)

    _, metadata = device_rewriter.DeviceRewriterPass().run(gm)

    assert node.args == (tensor, xla_device)
    assert node.kwargs == {"copy": True}
    assert metadata["rewrite_count"] == 1


def test_preserves_dtype_only_overload(device_rewriter):
    gm, node, tensor, _ = _graph_with_to(torch.bfloat16, non_blocking=True, copy=True)

    _, metadata = device_rewriter.DeviceRewriterPass().run(gm)

    assert node.args == (tensor, torch.bfloat16)
    assert node.kwargs == {"non_blocking": True, "copy": True}
    assert metadata["rewrite_count"] == 0


def test_preserves_tensor_to_tensor_overload(device_rewriter):
    gm, node, tensor, other = _graph_with_to("__other__", non_blocking=True, copy=True)

    _, metadata = device_rewriter.DeviceRewriterPass().run(gm)

    assert node.args == (tensor, other)
    assert node.kwargs == {"non_blocking": True, "copy": True}
    assert metadata["rewrite_count"] == 0


def test_existing_keyword_device_rewrite_is_preserved(device_rewriter):
    gm, node, tensor, _ = _graph_with_to(
        device=torch.device("meta"), dtype=torch.bfloat16, copy=True
    )

    _, metadata = device_rewriter.DeviceRewriterPass().run(gm)

    assert node.args == (tensor,)
    assert node.kwargs == {
        "device": torch.device("xla", index=0),
        "dtype": torch.bfloat16,
        "copy": True,
    }
    assert metadata["rewrite_count"] == 1
