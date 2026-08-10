# SPDX-License-Identifier: Apache-2.0
import importlib.util
import operator
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[3]


def _load_module(monkeypatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    """Load the focused pass without importing vLLM or Neuron runtimes."""
    package = ModuleType("vllm_neuron")
    package.__path__ = [str(ROOT / "vllm_neuron")]
    monkeypatch.setitem(sys.modules, "vllm_neuron", package)

    utils_package = ModuleType("vllm_neuron.utils")
    utils_package.__path__ = [str(ROOT / "vllm_neuron/utils")]
    monkeypatch.setitem(sys.modules, "vllm_neuron.utils", utils_package)
    metrics = ModuleType("vllm_neuron.utils.trace_metrics")
    metrics.TraceMetrics = object
    metrics.render_code = lambda gm, trace_metrics=None: gm.code
    metrics.render_graph = lambda gm, trace_metrics=None: str(gm.graph)
    monkeypatch.setitem(sys.modules, "vllm_neuron.utils.trace_metrics", metrics)

    fx_package = ModuleType("vllm_neuron.fx_passes")
    fx_package.__path__ = [str(ROOT / "vllm_neuron/fx_passes")]
    monkeypatch.setitem(sys.modules, "vllm_neuron.fx_passes", fx_package)
    _load_module(
        monkeypatch,
        "vllm_neuron.fx_passes.base",
        ROOT / "vllm_neuron/fx_passes/base.py",
    )
    inplace = _load_module(
        monkeypatch,
        "vllm_neuron.fx_passes.inplace_rewrite_pass",
        ROOT / "vllm_neuron/fx_passes/inplace_rewrite_pass.py",
    )
    manager = _load_module(
        monkeypatch,
        "vllm_neuron.fx_passes.pass_manager",
        ROOT / "vllm_neuron/fx_passes/pass_manager.py",
    )
    return SimpleNamespace(inplace=inplace, manager=manager)


def _mutation_chain() -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    before = graph.call_method("neg", args=(x,))
    first = graph.call_method("add_", args=(x, y))
    # Unrelated nodes make sure the optimization does not depend on adjacent
    # users or mutations.
    unrelated = before
    for _ in range(20):
        unrelated = graph.call_method("neg", args=(unrelated,))
    second = graph.call_method("clamp_", args=(x,), kwargs={"min": 0.0})
    nested = graph.call_function(torch.cat, args=([x, second],))
    graph.output((before, first, unrelated, nested))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _nested_kwarg_value(*, payload):
    return payload["value"]


def test_rewrites_only_actual_downstream_users_and_preserves_order(
    modules, monkeypatch
):
    gm = _mutation_chain()
    replacements = []
    original = torch.fx.Node.replace_input_with

    def counted(node, old_input, new_input):
        replacements.append((node, old_input, new_input))
        return original(node, old_input, new_input)

    monkeypatch.setattr(torch.fx.Node, "replace_input_with", counted)
    gm, _ = modules.inplace.InPlaceToOutOfPlacePass().run(gm)
    gm.graph.lint()

    x = torch.tensor([-2.0, 1.0])
    y = torch.tensor([3.0, -4.0])
    before, first, _, nested = gm(x, y)
    torch.testing.assert_close(before, torch.tensor([2.0, -1.0]))
    # Returning the original in-place value observes the later clamp, matching
    # the alias/mutation order of eager execution.
    torch.testing.assert_close(first, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(nested, torch.tensor([1.0, 0.0, 1.0, 0.0]))

    # The unrelated chain is never visited.  Four real user nodes are
    # rewritten: first mutation -> second mutation + output, then second
    # mutation -> cat + output.
    assert len(replacements) == 4


def test_setitem_rewrite_updates_only_following_aliases(modules):
    graph = torch.fx.Graph()
    state = graph.placeholder("state")
    before = graph.call_method("clone", args=(state,))
    graph.call_function(operator.setitem, args=(state, 1, 7.0))
    after = graph.call_method("clone", args=(state,))
    graph.output((before, after))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    gm, _ = modules.inplace.InPlaceToOutOfPlacePass().run(gm)
    gm.graph.lint()
    before_value, after_value = gm(torch.zeros(3))

    torch.testing.assert_close(before_value, torch.zeros(3))
    torch.testing.assert_close(after_value, torch.tensor([0.0, 7.0, 0.0]))


def test_chained_copy_rewrites_do_not_cycle_inserted_nodes(modules):
    graph = torch.fx.Graph()
    state = graph.placeholder("state")
    first_value = graph.placeholder("first_value")
    second_value = graph.placeholder("second_value")
    first = graph.call_method("copy_", args=(state, first_value))
    second = graph.call_method("copy_", args=(state, second_value))
    graph.output((first, second, state))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    gm, _ = modules.inplace.InPlaceToOutOfPlacePass().run(gm)
    gm.graph.lint()
    outputs = gm(torch.zeros(2), torch.ones(2), torch.full((2,), 3.0))

    for output in outputs:
        torch.testing.assert_close(output, torch.full((2,), 3.0))


def test_downstream_nested_kwarg_and_view_use_latest_value(modules):
    graph = torch.fx.Graph()
    state = graph.placeholder("state")
    delta = graph.placeholder("delta")
    graph.call_method("add_", args=(state, delta))
    view = graph.call_method("view", args=(state, -1))
    nested = graph.call_function(
        _nested_kwarg_value,
        kwargs={"payload": {"value": view}},
    )
    graph.output((state, nested))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    gm, _ = modules.inplace.InPlaceToOutOfPlacePass().run(gm)
    gm.graph.lint()
    state_value, nested_value = gm(torch.zeros(2), torch.ones(2))

    torch.testing.assert_close(state_value, torch.ones(2))
    torch.testing.assert_close(nested_value, torch.ones(2))


def test_dynamic_setitem_rewrites_following_users(modules):
    graph = torch.fx.Graph()
    state = graph.placeholder("state")
    index = graph.placeholder("index")
    index.meta["example_value"] = torch.tensor([0], dtype=torch.int64)
    value = graph.placeholder("value")
    graph.call_function(operator.setitem, args=(state, index, value))
    after = graph.call_method("clone", args=(state,))
    graph.output(after)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    gm, _ = modules.inplace.InPlaceToOutOfPlacePass().run(gm)
    gm.graph.lint()
    result = gm(torch.zeros(3), torch.tensor([1]), torch.tensor([9.0]))

    torch.testing.assert_close(result, torch.tensor([0.0, 9.0, 0.0]))


def test_manager_recompiles_inplace_pass_exactly_once(modules, monkeypatch):
    gm = _mutation_chain()
    calls = 0
    original = gm.recompile

    def counted():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(gm, "recompile", counted)
    manager = modules.manager.FXPassManager()
    manager.add_pass(modules.inplace.InPlaceToOutOfPlacePass())
    manager.run_passes(gm)

    assert calls == 1


def test_direct_run_still_recompiles_once(modules, monkeypatch):
    gm = _mutation_chain()
    calls = 0
    original = gm.recompile

    def counted():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(gm, "recompile", counted)
    modules.inplace.InPlaceToOutOfPlacePass().run(gm)

    assert calls == 1
