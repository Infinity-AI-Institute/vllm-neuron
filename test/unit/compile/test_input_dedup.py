"""Regression tests for graph-input storage identity."""

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[3]


def _load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def compile_modules(monkeypatch):
    """Load the two modules without importing the vLLM platform plugin."""
    package = ModuleType("vllm_neuron")
    package.__path__ = [str(ROOT / "vllm_neuron")]
    package.envs = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "vllm_neuron", package)

    compile_package = ModuleType("vllm_neuron.compile")
    compile_package.__path__ = [str(ROOT / "vllm_neuron/compile")]
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile", compile_package)

    cache = ModuleType("vllm_neuron.compile.cache")
    cache.get_neff_filename = lambda *_args, **_kwargs: "graph.neff"
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile.cache", cache)

    timer_module = ModuleType("vllm_neuron.utils.timer")
    timer_module.timer = nullcontext
    monkeypatch.setitem(sys.modules, "vllm_neuron.utils.timer", timer_module)

    milestones = ModuleType("vllm_neuron.compile.trace_milestones")
    milestones.emit_trace_milestone = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile.trace_milestones", milestones)

    throttle = ModuleType("vllm_neuron.compile.trace_throttle")
    throttle.host_trace_slot = nullcontext
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile.trace_throttle", throttle)

    backend = _load(
        monkeypatch,
        "vllm_neuron.compile.backend",
        ROOT / "vllm_neuron/compile/backend.py",
    )
    parallel_trace = _load(
        monkeypatch,
        "vllm_neuron.compile.parallel_trace",
        ROOT / "vllm_neuron/compile/parallel_trace.py",
    )
    return SimpleNamespace(backend=backend, parallel_trace=parallel_trace)


def test_input_dedup_keeps_independent_equal_shape_allocations(compile_modules):
    shared = torch.zeros(8)
    shared_view = shared.view_as(shared)
    independent = torch.zeros(8)

    keep_mask, dupe_map = compile_modules.backend._detect_duplicate_inputs(
        [shared, shared_view, shared, independent]
    )

    assert keep_mask == [True, False, False, True]
    assert dupe_map == [0, 0, 0, 1]


def test_meta_swap_preserves_real_aliases_without_aliasing_layers(compile_modules):
    class CacheOwner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_0_cache = torch.zeros(4, 2)
            self.layer_1_cache = torch.zeros(4, 2)
            self.layer_0_alias = self.layer_0_cache.view_as(self.layer_0_cache)

    owner = CacheOwner()
    compile_modules.parallel_trace._swap_to_meta_no_free(owner)

    layer_0_storage = owner.layer_0_cache.untyped_storage()._cdata
    layer_1_storage = owner.layer_1_cache.untyped_storage()._cdata
    alias_storage = owner.layer_0_alias.untyped_storage()._cdata
    assert owner.layer_0_cache.device.type == "meta"
    assert owner.layer_1_cache.device.type == "meta"
    assert layer_0_storage != layer_1_storage
    assert alias_storage == layer_0_storage


def test_meta_swap_preserves_distinct_offset_views_of_one_cache(compile_modules):
    class CacheOwner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            full_cache = torch.zeros(2, 4, 2)
            self.k_cache = full_cache[0]
            self.v_cache = full_cache[1]

    owner = CacheOwner()
    compile_modules.parallel_trace._swap_to_meta_no_free(owner)

    assert (
        owner.k_cache.untyped_storage()._cdata == owner.v_cache.untyped_storage()._cdata
    )
    assert owner.k_cache.storage_offset() == 0
    assert owner.v_cache.storage_offset() == owner.k_cache.numel()

    keep_mask, dupe_map = compile_modules.backend._detect_duplicate_inputs(
        [owner.k_cache, owner.v_cache]
    )
    assert keep_mask == [True, True]
    assert dupe_map == [0, 1]


def test_meta_swap_replaces_tensors_nested_in_k3_cache_containers(
    compile_modules,
):
    class CacheOwner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            cache_root = torch.zeros(2, 4, 2)
            k_cache = cache_root[0]
            v_cache = cache_root[1]
            self._kv_caches = {"prefill": [k_cache, (v_cache, "unchanged")]}
            self._kv_cache_roots = {"prefill": cache_root}
            self._kv_cache_allocation_roots = {"prefill": cache_root}

    owner = CacheOwner()
    source_root = owner._kv_cache_roots["prefill"]
    source_k = owner._kv_caches["prefill"][0]
    source_v = owner._kv_caches["prefill"][1][0]

    compile_modules.parallel_trace._swap_to_meta_no_free(owner)

    root = owner._kv_cache_roots["prefill"]
    allocation_root = owner._kv_cache_allocation_roots["prefill"]
    k_cache = owner._kv_caches["prefill"][0]
    v_cache = owner._kv_caches["prefill"][1][0]
    assert all(tensor.device.type == "meta" for tensor in (root, k_cache, v_cache))
    assert allocation_root is root
    assert owner._kv_caches["prefill"][1][1] == "unchanged"
    assert (
        len(
            {
                root.untyped_storage()._cdata,
                k_cache.untyped_storage()._cdata,
                v_cache.untyped_storage()._cdata,
            }
        )
        == 1
    )
    assert k_cache.storage_offset() == source_k.storage_offset()
    assert v_cache.storage_offset() == source_v.storage_offset()
    keepalive = compile_modules.parallel_trace._META_PARAM_KEEPALIVE
    assert any(source is source_root for source in keepalive)
    assert any(source is source_k for source in keepalive)
    assert any(source is source_v for source in keepalive)


def test_meta_swap_preserves_shared_container_aliases_and_mutable_cycles(
    compile_modules,
):
    class CacheOwner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            shared = {"cache": [torch.zeros(4)]}
            shared["self"] = shared
            self.primary = shared
            self.alias = shared

    owner = CacheOwner()
    compile_modules.parallel_trace._swap_to_meta_no_free(owner)

    assert owner.primary is owner.alias
    assert owner.primary["self"] is owner.primary
    assert owner.primary["cache"][0].device.type == "meta"
    compile_modules.parallel_trace._validate_models_on_meta([owner])


def test_post_swap_audit_names_tensor_in_unsupported_object(compile_modules):
    class CacheWrapper:
        def __init__(self):
            self.cache = torch.zeros(8039, 1, 32, 576)

    class CacheOwner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wrapper = CacheWrapper()

    owner = CacheOwner()
    compile_modules.parallel_trace._swap_to_meta_no_free(owner)

    with pytest.raises(
        ValueError,
        match=(
            r"models\[0\]\.wrapper\.<CacheWrapper>\.cache remains on cpu.*"
            r"8039, 1, 32, 576"
        ),
    ):
        compile_modules.parallel_trace._validate_models_on_meta([owner])
