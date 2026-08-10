import hashlib
import importlib.util
import inspect
import json
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[3]


def _stub_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_hlo(monkeypatch):
    xla_builder = _stub_module(monkeypatch, "torch_xla.core.xla_builder")
    core = _stub_module(monkeypatch, "torch_xla.core", xla_builder=xla_builder)
    _stub_module(monkeypatch, "torch_xla", core=core)

    structure = _stub_module(monkeypatch, "torch_neuronx.xla_impl.structure")
    xla_impl = _stub_module(monkeypatch, "torch_neuronx.xla_impl", structure=structure)
    _stub_module(monkeypatch, "torch_neuronx", xla_impl=xla_impl)

    proto_type = type("Proto", (), {})
    hlo_pb2 = _stub_module(
        monkeypatch,
        "torch_neuronx.pyhlo.hlo_pb2",
        HloModuleProto=proto_type,
        HloInputOutputAliasProto=proto_type,
        HloInstructionProto=proto_type,
        Kind=SimpleNamespace(MUST_ALIAS=1),
    )
    xla_data_pb2 = _stub_module(
        monkeypatch,
        "torch_neuronx.pyhlo.xla_data_pb2",
        ProgramShapeProto=proto_type,
        ShapeProto=proto_type,
        LiteralProto=proto_type,
    )
    _stub_module(
        monkeypatch,
        "torch_neuronx.pyhlo",
        hlo_pb2=hlo_pb2,
        xla_data_pb2=xla_data_pb2,
    )
    enum_utils = _stub_module(
        monkeypatch,
        "torch_neuronx.xla_impl.xla_hlo_tools.xla_primitive_enum_utils",
        XlaPrimitiveProperties=type("XlaPrimitiveProperties", (), {}),
    )
    tools = _stub_module(
        monkeypatch,
        "torch_neuronx.xla_impl.xla_hlo_tools",
        xla_primitive_enum_utils=enum_utils,
    )
    xla_impl.xla_hlo_tools = tools

    path = ROOT / "vllm_neuron" / "compile" / "hlo.py"
    spec = importlib.util.spec_from_file_location("hlo_diagnostics_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_disabled_by_default_and_rejects_ambiguous_value(monkeypatch):
    hlo = _load_hlo(monkeypatch)
    monkeypatch.delenv("VLLM_NEURON_HLO_FORWARD_DIAGNOSTICS", raising=False)
    assert hlo._forward_diagnostics_enabled() is False

    monkeypatch.setenv("VLLM_NEURON_HLO_FORWARD_DIAGNOSTICS", "maybe")
    with pytest.raises(ValueError, match="must be one of"):
        hlo._forward_diagnostics_enabled()


def test_operator_overload_target_is_exact_and_value_free(monkeypatch):
    hlo = _load_hlo(monkeypatch)
    assert hlo._safe_target_name(torch.ops.aten.add.Tensor) == "aten.add.Tensor"


def test_atomic_receipt_maps_lines_and_omits_values(monkeypatch, tmp_path):
    hlo = _load_hlo(monkeypatch)

    class Module(torch.nn.Module):
        def forward(self, value):
            doubled = value + value
            return doubled.relu()

    gm = torch.fx.symbolic_trace(Module())
    secret_value = torch.tensor([123456.0, 789012.0], dtype=torch.float32)
    destination = hlo._persist_forward_diagnostics(
        gm, (secret_value, {"secret-key": "secret-value"}), f"{tmp_path}/"
    )

    assert destination == tmp_path / "fx_forward_diagnostics"
    assert sorted(path.name for path in destination.iterdir()) == [
        "generated_line_to_fx_node.jsonl",
        "gm.code.py",
        "manifest.json",
    ]
    code = (destination / "gm.code.py").read_text(encoding="utf-8")
    manifest = json.loads((destination / "manifest.json").read_text())
    mappings = [
        json.loads(line)
        for line in (destination / "generated_line_to_fx_node.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert code == gm.code
    assert (
        manifest["graph"]["code_sha256"]
        == hashlib.sha256(code.encode("utf-8")).hexdigest()
    )
    assert manifest["graph"]["graph_sha256"] == manifest["graph"]["code_sha256"]
    assert manifest["graph"]["node_count"] == len(list(gm.graph.nodes))
    assert manifest["inputs"][0] == {
        "dtype": "torch.float32",
        "kind": "tensor",
        "shape": [2],
    }
    assert manifest["inputs"][1] == {
        "kind": "dict",
        "items": [{"kind": "opaque", "type": "str"}],
    }
    assert mappings
    assert all(
        {"generated_line", "node_index", "node_name", "op", "target", "parents"}
        <= mapping.keys()
        for mapping in mappings
    )
    assert any(mapping["parents"] for mapping in mappings)
    receipt_text = "\n".join(
        path.read_text(encoding="utf-8") for path in destination.iterdir()
    )
    assert "123456" not in receipt_text
    assert "789012" not in receipt_text
    assert "secret-key" not in receipt_text
    assert "secret-value" not in receipt_text
    assert not list(tmp_path.glob(".fx_forward_diagnostics.*"))


def test_existing_receipt_fails_closed_without_partial_publish(monkeypatch, tmp_path):
    hlo = _load_hlo(monkeypatch)
    gm = torch.fx.symbolic_trace(lambda value: value + 1)
    hlo._persist_forward_diagnostics(gm, (torch.zeros(1),), f"{tmp_path}/")

    with pytest.raises(FileExistsError, match="already exists"):
        hlo._persist_forward_diagnostics(gm, (torch.zeros(1),), f"{tmp_path}/")

    assert not list(tmp_path.glob(".fx_forward_diagnostics.*"))


def test_explicit_diagnostic_write_failure_is_fatal_before_forward(
    monkeypatch, tmp_path
):
    hlo = _load_hlo(monkeypatch)
    gm = torch.fx.symbolic_trace(lambda value: value + 1)
    forward_called = False

    def fail_receipt(*args, **kwargs):
        raise OSError("disk full")

    def record_forward(*args, **kwargs):
        nonlocal forward_called
        forward_called = True

    monkeypatch.setattr(hlo, "_persist_forward_diagnostics", fail_receipt)
    monkeypatch.setattr(gm, "forward", record_forward)

    assert (
        hlo._prepare_forward_diagnostics(False, gm, (torch.zeros(1),), str(tmp_path))
        is None
    )
    with pytest.raises(RuntimeError, match="explicitly enabled"):
        hlo._prepare_forward_diagnostics(True, gm, (torch.zeros(1),), str(tmp_path))

    assert forward_called is False


def test_diagnostic_markers_are_opt_in(monkeypatch, caplog):
    hlo = _load_hlo(monkeypatch)
    with caplog.at_level(logging.INFO):
        hlo._forward_diagnostic_marker(False, "before gm execution")
        hlo._forward_diagnostic_marker(True, "before gm execution")

    assert caplog.messages == ["FX-forward diagnostic marker: before gm execution"]


def test_requested_markers_bracket_forward_extract_build_and_serialize(monkeypatch):
    hlo = _load_hlo(monkeypatch)
    source = inspect.getsource(hlo.convert_fx_to_hlo)
    ordered_fragments = [
        '"before gm execution"',
        "outputs = gm(*xla_placeholders)",
        '"after gm execution"',
        '"before structure.extract"',
        "structure.extract(outputs)",
        '"after structure.extract"',
        '"before LoweringContext.build"',
        "context.build(tensors)",
        '"after LoweringContext.build"',
        '"before LoweringContext.serialize"',
        "hlo = context.hlo()",
        '"after LoweringContext.serialize"',
    ]

    positions = [source.index(fragment) for fragment in ordered_fragments]
    assert positions == sorted(positions)
