"""Adversarial qualification for the opt-in safe fast-trace subset."""

from __future__ import annotations

import importlib.util
import shlex
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
    """Load the focused modules without importing vLLM or Neuron runtimes."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

    package = ModuleType("vllm_neuron")
    package.__path__ = [str(ROOT / "vllm_neuron")]
    monkeypatch.setitem(sys.modules, "vllm_neuron", package)
    envs = _load_module(monkeypatch, "vllm_neuron.envs", ROOT / "vllm_neuron/envs.py")
    package.envs = envs

    utils_package = ModuleType("vllm_neuron.utils")
    utils_package.__path__ = [str(ROOT / "vllm_neuron/utils")]
    monkeypatch.setitem(sys.modules, "vllm_neuron.utils", utils_package)
    timer = _load_module(
        monkeypatch, "vllm_neuron.utils.timer", ROOT / "vllm_neuron/utils/timer.py"
    )
    metrics = _load_module(
        monkeypatch,
        "vllm_neuron.utils.trace_metrics",
        ROOT / "vllm_neuron/utils/trace_metrics.py",
    )

    compile_package = ModuleType("vllm_neuron.compile")
    compile_package.__path__ = [str(ROOT / "vllm_neuron/compile")]
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile", compile_package)
    platform = _load_module(
        monkeypatch,
        "vllm_neuron.compile.platform",
        ROOT / "vllm_neuron/compile/platform.py",
    )

    backend = ModuleType("vllm_neuron.compile.backend")
    backend._parse_compiler_args = lambda value: (
        shlex.split(value) if isinstance(value, str) else list(value or [])
    )
    backend._apply_platform_compiler_args = lambda options: options
    backend.preprocess_and_validate_inputs = lambda gm, inputs: (gm, inputs)
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile.backend", backend)

    cache = _load_module(
        monkeypatch, "vllm_neuron.compile.cache", ROOT / "vllm_neuron/compile/cache.py"
    )
    monkeypatch.setattr(cache, "get_torch_neuronx_version", lambda: "test-tnx")
    monkeypatch.setattr(cache, "get_neuronxcc_version", lambda: "test-ncc")
    monkeypatch.setattr(cache, "get_nki_version", lambda: "test-nki")

    fx_package = ModuleType("vllm_neuron.fx_passes")
    fx_package.__path__ = [str(ROOT / "vllm_neuron/fx_passes")]
    monkeypatch.setitem(sys.modules, "vllm_neuron.fx_passes", fx_package)
    base = _load_module(
        monkeypatch,
        "vllm_neuron.fx_passes.base",
        ROOT / "vllm_neuron/fx_passes/base.py",
    )
    manager = _load_module(
        monkeypatch,
        "vllm_neuron.fx_passes.pass_manager",
        ROOT / "vllm_neuron/fx_passes/pass_manager.py",
    )
    fx_package.get_default_pass_manager = manager.FXPassManager

    capture = _load_module(
        monkeypatch,
        "vllm_neuron.compile.capture_backend",
        ROOT / "vllm_neuron/compile/capture_backend.py",
    )
    hlo = ModuleType("vllm_neuron.compile.hlo")
    hlo.convert_fx_to_hlo = lambda *args, **kwargs: (object(), [], False)
    monkeypatch.setitem(sys.modules, "vllm_neuron.compile.hlo", hlo)

    return SimpleNamespace(
        base=base,
        cache=cache,
        capture=capture,
        envs=envs,
        hlo=hlo,
        manager=manager,
        metrics=metrics,
        platform=platform,
        timer=timer,
    )


class _Toy(torch.nn.Module):
    def forward(self, value):
        return value + 1


def _graph_module():
    return torch.fx.symbolic_trace(_Toy())


def test_cache_hash_renders_normalized_graph_once(modules, monkeypatch):
    gm = _graph_module()
    inputs = [torch.tensor([1.0])]
    renders = 0
    original = torch.fx.Graph.__str__

    def counted(graph):
        nonlocal renders
        renders += 1
        return original(graph)

    monkeypatch.setattr(torch.fx.Graph, "__str__", counted)
    first = modules.cache.create_cache_hash(gm, inputs, {})
    second = modules.cache.create_cache_hash(gm, inputs, {})

    assert first == second
    assert renders == 2, "each hash call must render its normalized graph once"


def test_fast_success_suppresses_dumps_but_preserves_every_recompile(
    modules, monkeypatch, tmp_path
):
    class AliasingPass(modules.base.FXPass):
        @property
        def name(self):
            return "aliasing_output_rewrite"

        def run(self, gm, **kwargs):
            gm.recompile()  # pass-local recompile must remain
            return gm, {"io_map": None, "original_output_count": 1}

    def run(fast: bool, directory: Path):
        gm = _graph_module()
        calls = 0
        original = gm.recompile

        def counted():
            nonlocal calls
            calls += 1
            return original()

        monkeypatch.setattr(gm, "recompile", counted)
        manager = modules.manager.FXPassManager()
        manager.add_pass(AliasingPass())
        monkeypatch.setattr(
            modules.capture, "get_default_pass_manager", lambda: manager
        )
        metrics = modules.metrics.TraceMetrics(fast_trace=fast)
        result = modules.capture._run_fx_passes(
            gm, {}, str(directory), trace_metrics=metrics
        )
        return calls, metrics, result

    baseline_calls, baseline_metrics, baseline_result = run(
        False, tmp_path / "baseline"
    )
    fast_calls, fast_metrics, fast_result = run(True, tmp_path / "fast")

    # Pass-local, pass-manager, and final pipeline recompiles all remain.
    assert baseline_calls == fast_calls == 3
    assert baseline_result[1:] == fast_result[1:] == (None, 1)
    assert baseline_metrics.graph_dump_files == 2
    assert fast_metrics.graph_dump_files == 0
    assert fast_metrics.graph_dump_files_suppressed == 2
    assert list((tmp_path / "baseline/passes").glob("*.txt"))
    assert not (tmp_path / "fast/passes").exists()


def test_fast_pipeline_writes_one_failure_diagnostic_and_keeps_options_clean(
    modules, monkeypatch, tmp_path
):
    monkeypatch.setenv("VLLM_NEURON_FAST_TRACE", "1")
    options = {"target_device": "xla"}
    gm = _graph_module()
    monkeypatch.setattr(
        modules.capture,
        "_run_fx_passes",
        lambda gm, options, workdir, trace_metrics=None: (gm, None, 1),
    )

    def fail_convert(*args, **kwargs):
        raise ValueError("ordinary failure")

    modules.hlo.convert_fx_to_hlo = fail_convert

    with pytest.raises(ValueError, match="ordinary failure"):
        modules.capture.run_fx_to_hlo_pipeline(
            gm, [torch.tensor([1.0])], options, str(tmp_path)
        )

    assert options == {"target_device": "xla"}
    assert not (tmp_path / "fxgraph.txt").exists()
    diagnostics = list(tmp_path.glob("*failure*.txt"))
    assert diagnostics == [tmp_path / "fxgraph_failure.txt"]
    receipt = (tmp_path / "trace_metrics.json").read_text()
    assert '"failure_diagnostics": 1' in receipt
    assert '"graph_string_renders": 1' in receipt
    assert '"graph_code_renders": 1' in receipt
    assert '"success": false' in receipt


@pytest.mark.parametrize(
    "error",
    [
        MemoryError("low memory"),
        KeyboardInterrupt(),
        SystemExit(2),
    ],
)
def test_fast_pipeline_never_renders_fatal_or_low_memory_failures(
    modules, monkeypatch, tmp_path, error
):
    monkeypatch.setenv("VLLM_NEURON_FAST_TRACE", "1")
    monkeypatch.setattr(
        modules.capture,
        "render_graph",
        lambda *args, **kwargs: pytest.fail("graph rendering must not run"),
    )
    monkeypatch.setattr(
        modules.capture,
        "render_code",
        lambda *args, **kwargs: pytest.fail("code rendering must not run"),
    )

    def fail_passes(*args, **kwargs):
        if isinstance(error, MemoryError):
            raise RuntimeError("wrapped") from error  # noqa: TRY004
        raise error

    monkeypatch.setattr(modules.capture, "_run_fx_passes", fail_passes)

    expected = (
        pytest.raises(RuntimeError, match="wrapped")
        if isinstance(error, MemoryError)
        else pytest.raises(type(error))
    )
    with expected:
        modules.capture.run_fx_to_hlo_pipeline(
            _graph_module(), [torch.tensor([1.0])], {}, str(tmp_path)
        )

    assert not (tmp_path / "fxgraph_failure.txt").exists()


def test_reused_options_get_fresh_metrics_and_no_stale_failure_state(
    modules, monkeypatch, tmp_path
):
    monkeypatch.setenv("VLLM_NEURON_FAST_TRACE", "1")
    options = {"target_device": "xla"}
    seen = []
    calls = 0

    monkeypatch.setattr(
        modules.capture,
        "_run_fx_passes",
        lambda gm, options, workdir, trace_metrics=None: (gm, None, 1),
    )

    class Hlo:
        pass

    def convert(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("first graph fails")
        return Hlo(), [], False

    modules.hlo.convert_fx_to_hlo = convert

    def record_write(self, workdir):
        seen.append(self)
        return str(Path(workdir) / "trace_metrics.json")

    monkeypatch.setattr(modules.metrics.TraceMetrics, "write", record_write)

    with pytest.raises(ValueError, match="first graph fails"):
        modules.capture.run_fx_to_hlo_pipeline(
            _graph_module(), [torch.tensor([1.0])], options, str(tmp_path / "first")
        )
    result = modules.capture.run_fx_to_hlo_pipeline(
        _graph_module(), [torch.tensor([1.0])], options, str(tmp_path / "second")
    )

    assert len(seen) == 2
    assert seen[0] is not seen[1]
    assert seen[0].success is False
    assert seen[1].success is True
    assert seen[1].error_type is None
    assert seen[1].failure_diagnostics == 0
    assert options == {"target_device": "xla"}
    assert result[1:5] == ([], False, None, 1)
    assert result[-1] >= 0
    assert not (tmp_path / "second/fxgraph.txt").exists()
    assert (tmp_path / "second/example_inputs.txt").is_file()


def test_normalized_hash_copy_still_recompiles_after_device_rewrite(
    modules, monkeypatch
):
    graph = torch.fx.Graph()
    output = graph.call_function(
        torch.empty, args=((1,),), kwargs={"device": "neuron:7"}
    )
    graph.output(output)
    gm = torch.fx.GraphModule({}, graph)
    recompiles = 0
    prepared_copy = modules.cache.copy.deepcopy(gm)
    original_recompile = prepared_copy.recompile

    def counted():
        nonlocal recompiles
        recompiles += 1
        return original_recompile()

    prepared_copy.recompile = counted
    monkeypatch.setattr(
        torch.fx.GraphModule,
        "__deepcopy__",
        lambda self, memo: prepared_copy,
    )
    normalized = modules.cache._normalize_neuron_devices_for_hashing(gm)

    normalized_device = next(
        node.kwargs["device"]
        for node in normalized.graph.nodes
        if "device" in node.kwargs
    )
    assert normalized_device == "neuron:0"
    assert recompiles == 1
