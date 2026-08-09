"""CPU-only coverage for the fail-closed pre-Dynamo tensor audit."""

import dataclasses
import functools
import importlib.util
import sys
import types
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[3]


@pytest.fixture
def parallel_trace_module(monkeypatch):
    """Load parallel_trace without importing the vLLM platform plugin."""
    package = ModuleType("vllm_neuron")
    package.__path__ = [str(ROOT / "vllm_neuron")]
    package.envs = SimpleNamespace(VLLM_NEURON_TRACE_PREFLIGHT_ONLY=False)
    monkeypatch.setitem(sys.modules, "vllm_neuron", package)

    compile_package = ModuleType("vllm_neuron.compile")
    compile_package.__path__ = [str(ROOT / "vllm_neuron/compile")]
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile", compile_package)

    milestones = ModuleType("vllm_neuron.compile.trace_milestones")
    milestones.emit_trace_milestone = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile.trace_milestones", milestones)

    throttle = ModuleType("vllm_neuron.compile.trace_throttle")
    throttle.host_trace_slot = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile.trace_throttle", throttle)

    name = "vllm_neuron.compile.parallel_trace"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "vllm_neuron/compile/parallel_trace.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _owners_by_tensor(module, jobs):
    return {
        id(record.tensor): record for record in module._collect_job_tensor_owners(jobs)
    }


def test_audit_follows_optimized_module_original_and_all_alias_paths(
    parallel_trace_module,
):
    tensor = torch.zeros(2, 3).T

    class Original(torch.nn.Module):
        def __init__(self):
            super().__init__()
            shared = {"tensor": tensor}
            self.first = tensor
            self.nested = {"second": [tensor]}
            self.shared_a = shared
            self.shared_b = shared

    class OptimizedModuleLike:
        def __init__(self):
            self._orig_mod = Original()

        def __call__(self, **_kwargs):
            return None

    records = _owners_by_tensor(
        parallel_trace_module,
        [(OptimizedModuleLike(), {"input": tensor})],
    )

    assert records[id(tensor)].owner_paths == {
        "jobs[0].callable._orig_mod.first",
        "jobs[0].callable._orig_mod.nested['second'][0]",
        "jobs[0].callable._orig_mod.shared_a['tensor']",
        "jobs[0].callable._orig_mod.shared_b['tensor']",
        "jobs[0].kwargs['input']",
    }
    metadata = parallel_trace_module._tensor_audit_metadata(tensor)
    assert "device=cpu" in metadata
    assert "shape=(3, 2)" in metadata
    assert "dtype=torch.float32" in metadata
    assert "stride=(1, 3)" in metadata
    assert "storage_offset=0" in metadata
    assert f"identity=0x{id(tensor):x}" in metadata
    assert "alias_root=0x" in metadata


def test_audit_follows_real_torch_optimized_module(parallel_trace_module):
    class Original(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cache = torch.ones(2)

        def forward(self, value):
            return value + self.cache

    optimized = torch.compile(Original(), backend="eager")
    try:
        records = _owners_by_tensor(
            parallel_trace_module,
            [(optimized, {"value": torch.empty(2, device="meta")})],
        )
        source = optimized._orig_mod.cache
        assert any(
            path == "jobs[0].callable._orig_mod.cache"
            for path in records[id(source)].owner_paths
        ), records[id(source)].owner_paths
    finally:
        torch.compiler.reset()


def test_audit_follows_bound_method_partial_closure_and_referenced_globals(
    parallel_trace_module,
):
    bound_tensor = torch.ones(1)
    partial_arg = torch.ones(2)
    partial_kwarg = torch.ones(3)
    closure_tensor = torch.ones(4)
    global_tensor = torch.ones(5)
    unrelated_global = torch.ones(6)

    namespace = {
        "used_global": global_tensor,
        "unrelated_global": unrelated_global,
    }
    exec(  # noqa: S102 - isolated namespace verifies referenced globals only
        "def referenced_global():\n    return used_global\n",
        namespace,
    )
    referenced_global = namespace["referenced_global"]

    def closure_function():
        return closure_tensor, referenced_global()

    class BoundOwner:
        def __init__(self):
            self.tensor = bound_tensor

        def invoke(self, *_args, **_kwargs):
            return closure_function()

    callable_root = functools.partial(
        BoundOwner().invoke,
        partial_arg,
        named=partial_kwarg,
    )
    records = _owners_by_tensor(
        parallel_trace_module,
        [(callable_root, {})],
    )

    assert id(bound_tensor) in records
    assert id(partial_arg) in records
    assert id(partial_kwarg) in records
    assert id(closure_tensor) in records
    assert id(global_tensor) in records
    assert id(unrelated_global) not in records
    assert any(
        ".__self__.tensor" in path for path in records[id(bound_tensor)].owner_paths
    )
    assert any(".args[0]" in path for path in records[id(partial_arg)].owner_paths)
    assert any(
        ".keywords['named']" in path for path in records[id(partial_kwarg)].owner_paths
    )
    assert any(
        "closure_tensor" in path for path in records[id(closure_tensor)].owner_paths
    )
    assert any(
        "globals['used_global']" in path
        for path in records[id(global_tensor)].owner_paths
    )


def test_audit_follows_dataclass_slots_resolved_weakrefs_and_cycles(
    parallel_trace_module,
):
    dataclass_tensor = torch.ones(1)
    slot_tensor = torch.ones(2)
    weak_tensor = torch.ones(3)

    @dataclasses.dataclass
    class Payload:
        tensor: torch.Tensor

    class Slotted:
        __slots__ = ("__weakref__", "cycle", "tensor")

        def __init__(self):
            self.tensor = slot_tensor
            self.cycle = self

    class WeakOwner:
        pass

    weak_owner = WeakOwner()
    weak_owner.tensor = weak_tensor
    kwargs = {
        "dataclass": Payload(dataclass_tensor),
        "slots": Slotted(),
        "weak": weakref.ref(weak_owner),
    }

    records = _owners_by_tensor(
        parallel_trace_module,
        [(lambda **_kwargs: None, kwargs)],
    )

    assert any(
        path.endswith("kwargs['dataclass'].tensor")
        for path in records[id(dataclass_tensor)].owner_paths
    )
    assert any(
        path.endswith("kwargs['slots'].tensor")
        for path in records[id(slot_tensor)].owner_paths
    )
    assert any(
        path.endswith("kwargs['weak']().tensor")
        for path in records[id(weak_tensor)].owner_paths
    )


def test_audit_ignores_dead_weak_references(parallel_trace_module):
    class WeakOwner:
        pass

    weak_owner = WeakOwner()
    dead_reference = weakref.ref(weak_owner)
    del weak_owner

    assert (
        parallel_trace_module._collect_job_tensor_owners(
            [(lambda: None, {"weak": dead_reference})]
        )
        == []
    )


def test_audit_does_not_execute_mapping_key_repr(parallel_trace_module):
    tensor = torch.ones(1)

    class HostileKey:
        def __repr__(self):
            raise AssertionError("mapping-key repr must not execute")

    records = _owners_by_tensor(
        parallel_trace_module,
        [(lambda: None, {HostileKey(): tensor})],
    )

    [owner_path] = records[id(tensor)].owner_paths
    assert "kwargs[key:0<" in owner_path
    assert owner_path.endswith("HostileKey>]")


def test_audit_does_not_scan_whole_function_globals_or_module_namespaces(
    parallel_trace_module,
):
    unrelated = torch.ones(7)
    namespace = {"module": torch, "unrelated": unrelated}
    exec(  # noqa: S102 - isolated namespace carries an unreferenced sentinel
        "def callable_root():\n    return module.float32\n",
        namespace,
    )

    records = parallel_trace_module._collect_job_tensor_owners(
        [(namespace["callable_root"], {})]
    )

    assert records == []


def test_audit_failure_aggregates_paths_and_exact_tensor_metadata(
    parallel_trace_module,
):
    tensor = torch.arange(12, dtype=torch.float32).view(3, 4)[:, 1:]

    class CallableRoot:
        def __init__(self):
            self.direct = tensor
            self.alias = [tensor]

        def __call__(self):
            return None

    with pytest.raises(ValueError) as exc_info:
        parallel_trace_module._audit_jobs_on_meta([(CallableRoot(), {})])

    message = str(exc_info.value)
    assert "1 unexplained non-meta tensor identity(s)" in message
    assert "device=cpu" in message
    assert "shape=(3, 3)" in message
    assert "stride=(4, 1)" in message
    assert "storage_offset=1" in message
    assert f"identity=0x{id(tensor):x}" in message
    assert "alias_root=0x" in message
    assert "owner=jobs[0].callable.direct" in message
    assert "owner=jobs[0].callable.alias[0]" in message


def test_audit_accepts_only_meta_tensor_roots(parallel_trace_module):
    meta = torch.empty(2, 3, device="meta")
    parallel_trace_module._audit_jobs_on_meta(
        [(functools.partial(lambda value: value, meta), {"value": meta})]
    )


def test_production_audit_does_not_retain_clean_meta_path_suffixes(
    parallel_trace_module,
):
    meta = torch.empty(2, 3, device="meta")
    jobs = [(lambda **_kwargs: None, {"nested": [{"value": meta}]})]

    assert (
        parallel_trace_module._collect_job_tensor_owners(jobs, include_meta=False) == []
    )


def test_child_resets_compiler_before_swap_audit_and_trace(
    monkeypatch,
    tmp_path,
    parallel_trace_module,
):
    events = []

    capture_backend = ModuleType("vllm_neuron.compile.capture_backend")

    class CaptureComplete(Exception):
        pass

    capture_backend.CaptureComplete = CaptureComplete
    monkeypatch.setitem(
        sys.modules,
        "vllm_neuron.compile.capture_backend",
        capture_backend,
    )
    monkeypatch.setattr(
        parallel_trace_module.torch.compiler,
        "reset",
        lambda: events.append("reset"),
    )
    monkeypatch.setattr(
        parallel_trace_module,
        "_swap_unique_models_to_meta",
        lambda _models: events.append("swap"),
    )
    monkeypatch.setattr(
        parallel_trace_module,
        "_audit_jobs_on_meta",
        lambda _jobs: events.append("audit"),
    )

    def trace_callable(**_kwargs):
        events.append("trace")
        raise CaptureComplete

    result_path = tmp_path / "child.status"
    parallel_trace_module._fork_child_main(
        0,
        0,
        [(trace_callable, {})],
        str(result_path),
    )

    assert events == ["reset", "swap", "audit", "trace"]
    assert result_path.read_text().splitlines() == ["OK"]


def test_module_objects_are_terminal_even_when_directly_owned(
    parallel_trace_module,
):
    module = types.ModuleType("tensor_filled_module")
    module.tensor = torch.ones(8)

    records = parallel_trace_module._collect_job_tensor_owners(
        [(lambda: None, {"module": module})]
    )

    assert records == []
