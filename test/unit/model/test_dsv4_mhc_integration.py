# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[3]
DSV4 = ROOT / "vllm_neuron" / "model" / "dsv4_flash"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, DSV4 / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for package_name, package_path in (
    ("vllm_neuron", ROOT / "vllm_neuron"),
    ("vllm_neuron.model", ROOT / "vllm_neuron" / "model"),
    ("vllm_neuron.model.dsv4_flash", DSV4),
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]
    sys.modules.setdefault(package_name, package)

CONFIG = _load("vllm_neuron.model.dsv4_flash.config", "config.py")
CONVERTER = _load(
    "vllm_neuron.model.dsv4_flash.checkpoint_convert", "checkpoint_convert.py"
)
MHC = _load("dsv4_mhc_integration_contract", "mhc_contract.py")
WRAPPER_PATH = DSV4 / "neuron_wrapper.py"


def _small_source() -> SimpleNamespace:
    return SimpleNamespace(hidden_size=3, hc_mult=4)


def _mhc_state() -> dict[str, torch.Tensor]:
    state = {
        "hc_head_fn": torch.arange(48, dtype=torch.float32).reshape(4, 12),
        "hc_head_base": torch.arange(4, dtype=torch.float32),
        "hc_head_scale": torch.ones(1, dtype=torch.float32),
    }
    for stem in ("hc_attn", "hc_ffn"):
        prefix = f"layers.0.{stem}"
        state[f"{prefix}_fn"] = torch.arange(24 * 12, dtype=torch.float32).reshape(
            24, 12
        )
        state[f"{prefix}_base"] = torch.arange(24, dtype=torch.float32)
        state[f"{prefix}_scale"] = torch.ones(3, dtype=torch.float32)
    return state


def test_lossless_checkpoint_identity() -> None:
    state = _mhc_state()
    source = _small_source()
    converted = {
        **CONVERTER._convert_mhc_head(state, source),
        **CONVERTER._convert_mhc_layer(state, 0, source),
    }
    assert set(converted) == set(state)
    for key, value in converted.items():
        assert value is state[key]
        assert value.dtype == torch.float32


@pytest.mark.parametrize("mode", ["missing", "dtype", "shape"])
def test_checkpoint_mhc_drift_fails_before_materialization(mode: str) -> None:
    state = _mhc_state()
    key = "layers.0.hc_ffn_fn"
    if mode == "missing":
        state.pop(key)
        expected = KeyError
    elif mode == "dtype":
        state[key] = state[key].to(torch.bfloat16)
        expected = TypeError
    else:
        state[key] = state[key][:-1]
        expected = ValueError
    with pytest.raises(expected):
        CONVERTER._convert_mhc_layer(state, 0, _small_source())


def test_source_config_json_roundtrip_preserves_mhc_contract() -> None:
    source = CONFIG.DeepseekV4FlashInferenceConfig()
    payload = json.loads(json.dumps(source.to_dict(), sort_keys=True))
    restored = CONFIG.DeepseekV4FlashInferenceConfig.from_configs(payload)
    assert restored.to_dict() == source.to_dict()
    assert (restored.hc_mult, restored.hc_sinkhorn_iters, restored.hc_eps) == (
        4,
        20,
        1e-6,
    )
    payload["hc_eps"] = 1e-5
    with pytest.raises(ValueError, match="hc_eps"):
        CONFIG.DeepseekV4FlashInferenceConfig.from_configs(payload)


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _call_count(node: ast.AST, name: str) -> int:
    return sum(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == name
        for item in ast.walk(node)
    )


def test_wrapper_structurally_wraps_both_branches_and_head_once() -> None:
    tree = ast.parse(WRAPPER_PATH.read_text(encoding="utf-8"))
    layer = _class_node(tree, "DeepseekV4FlashLayer")
    model = _class_node(tree, "_NeuronDeepseekV4FlashModel")
    assert _call_count(layer, "mhc_pre") == 2
    assert _call_count(layer, "mhc_post") == 2
    assert _call_count(model, "mhc_head") == 1

    text = WRAPPER_PATH.read_text(encoding="utf-8")
    assert ".unsqueeze(2)" in text
    assert ".expand(-1, -1, 4, -1)" in text
    assert "DeepseekV4FlashInferenceConfig.from_configs" in text


def test_four_stream_equations_remain_closed_across_all_43_layers() -> None:
    torch.manual_seed(29)
    streams = torch.randn(1, 2, 4, 3, dtype=torch.bfloat16)
    for _layer in range(43):
        for branch_scale in (0.5, -0.25):  # attention, then MoE
            fn = torch.randn(24, 12, dtype=torch.float32)
            scale = torch.randn(3, dtype=torch.float32)
            base = torch.randn(24, dtype=torch.float32)
            collapsed, post, comb = MHC.mhc_pre(streams, fn, scale, base, norm_eps=1e-6)
            branch = (collapsed * branch_scale).to(streams.dtype)
            streams = MHC.mhc_post(branch, streams, post, comb)
            assert streams.shape == (1, 2, 4, 3)
            assert streams.dtype == torch.bfloat16

    head = MHC.mhc_head(
        streams,
        torch.randn(4, 12, dtype=torch.float32),
        torch.randn(1, dtype=torch.float32),
        torch.randn(4, dtype=torch.float32),
        norm_eps=1e-6,
    )
    assert head.shape == (1, 2, 3)
    assert head.dtype == torch.bfloat16
