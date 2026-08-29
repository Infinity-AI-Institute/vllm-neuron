# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).parents[3]
    / "vllm_neuron"
    / "model"
    / "dsv4_flash"
    / "mhc_contract.py"
)
SPEC = importlib.util.spec_from_file_location("dsv4_mhc_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MHC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MHC
SPEC.loader.exec_module(MHC)


def _reference_split(mixes, scale, base):
    pre = torch.sigmoid(mixes[..., :4] * scale[0] + base[:4]) + 1e-6
    post = 2 * torch.sigmoid(mixes[..., 4:8] * scale[1] + base[4:8])
    comb = (mixes[..., 8:] * scale[2] + base[8:]).reshape(*mixes.shape[:-1], 4, 4)
    comb = comb.softmax(-1) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    return pre, post, comb


def test_exact_261_tensor_contract_and_header_validation() -> None:
    specs = MHC.build_dsv4_mhc_tensor_contract()
    assert len(specs) == 261
    assert sum(item.name.startswith("hc_head_") for item in specs) == 3
    assert sum(item.name.startswith("layers.") for item in specs) == 258
    assert {item.dtype for item in specs} == {"F32"}
    assert {item.ownership for item in specs} == {"replicated_on_all_tp32_ranks"}
    headers = {
        item.name: {"dtype": item.dtype, "shape": list(item.shape)} for item in specs
    }
    assert len(MHC.validate_dsv4_mhc_headers(headers)) == 64


def test_exact_inherited_compile_boundary_is_fail_closed() -> None:
    boundary = MHC.build_dsv4_mhc_integration_boundary()
    assert boundary["source_parent_commit"] == (
        "2dc3d6a2a125cad006426d77a2998c5dd4b7bd13"
    )
    assert boundary["topology"] == {
        "hardware": "trn2.48xlarge",
        "tp_degree": 32,
        "logical_neuroncore_config": 2,
        "ctx_batch_size": 1,
        "tkg_batch_size": 1,
        "sequence_buckets": [4096],
    }
    assert boundary["emitted_state"]["compute_dtype"] == "bfloat16"
    assert boundary["emitted_state"]["speculative_decode"] is False
    assert boundary["compile_execution_policy"] == {
        "ownership_marker": "/mnt/compile/OWNERSHIP.md",
        "max_active_compiles": 2,
        "systemd_run_required": True,
        "unit_name_required": True,
        "nice": 15,
        "scope_forbidden": True,
        "docker_network": "none",
        "atomic_output_suffix": ".partial",
    }
    assert not any(boundary["claims"].values())
    assert len(MHC.validate_dsv4_mhc_integration_boundary(boundary)) == 64


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("topology", "tp_degree", 16),
        ("topology", "logical_neuroncore_config", 1),
        ("emitted_state", "compute_dtype", "float16"),
        ("emitted_state", "speculative_decode", True),
        ("compile_execution_policy", "max_active_compiles", 3),
        ("compile_execution_policy", "nice", 0),
        ("compile_execution_policy", "scope_forbidden", False),
        ("compile_execution_policy", "docker_network", "host"),
        ("claims", "compile_permitted", True),
    ],
)
def test_integration_boundary_drift_fails_closed(
    section: str, field: str, value: object
) -> None:
    boundary = MHC.build_dsv4_mhc_integration_boundary()
    boundary[section][field] = value
    with pytest.raises(MHC.Dsv4MhcContractError, match="boundary drift"):
        MHC.validate_dsv4_mhc_integration_boundary(boundary)


@pytest.mark.parametrize("mode", ["missing", "extra", "dtype", "shape"])
def test_header_contract_drift_fails_closed(mode: str) -> None:
    specs = MHC.build_dsv4_mhc_tensor_contract()
    headers = {
        item.name: {"dtype": item.dtype, "shape": list(item.shape)} for item in specs
    }
    key = "layers.7.hc_ffn_fn"
    if mode == "missing":
        headers.pop(key)
    elif mode == "extra":
        headers["layers.7.hc_unknown"] = {"dtype": "F32", "shape": [1]}
    elif mode == "dtype":
        headers[key]["dtype"] = "BF16"
    else:
        headers[key]["shape"] = [24, 16383]
    with pytest.raises(MHC.Dsv4MhcContractError):
        MHC.validate_dsv4_mhc_headers(headers)


def test_split_sinkhorn_matches_independent_official_equations() -> None:
    torch.manual_seed(4)
    mixes = torch.randn(2, 3, 24, dtype=torch.float32)
    scale = torch.randn(3, dtype=torch.float32)
    base = torch.randn(24, dtype=torch.float32)
    actual = MHC.hc_split_sinkhorn(mixes, scale, base)
    expected = _reference_split(mixes, scale, base)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.all(actual[0] > 0)
    assert torch.all(actual[1] > 0)


def test_pre_post_and_head_match_independent_formulas() -> None:
    torch.manual_seed(5)
    residual = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    fn = torch.randn(24, 32, dtype=torch.float32)
    scale = torch.randn(3, dtype=torch.float32)
    base = torch.randn(24, dtype=torch.float32)
    reduced, post, comb = MHC.mhc_pre(residual, fn, scale, base, norm_eps=1e-6)
    flat = residual.flatten(2).float()
    mixes = torch.nn.functional.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + 1e-6
    )
    pre_ref, post_ref, comb_ref = _reference_split(mixes, scale, base)
    reduced_ref = torch.sum(pre_ref.unsqueeze(-1) * residual.float(), dim=2).to(
        residual.dtype
    )
    torch.testing.assert_close(reduced, reduced_ref, rtol=0, atol=0)
    torch.testing.assert_close(post, post_ref, rtol=0, atol=0)
    torch.testing.assert_close(comb, comb_ref, rtol=0, atol=0)

    branch = torch.randn_like(reduced)
    post_out = MHC.mhc_post(branch, residual, post, comb)
    post_ref_out = post.to(branch.dtype).unsqueeze(-1) * branch.unsqueeze(
        -2
    ) + torch.matmul(comb.to(branch.dtype).transpose(-1, -2), residual)
    torch.testing.assert_close(post_out, post_ref_out, rtol=0, atol=0)

    head_fn = torch.randn(4, 32, dtype=torch.float32)
    head_scale = torch.randn(1, dtype=torch.float32)
    head_base = torch.randn(4, dtype=torch.float32)
    head = MHC.mhc_head(
        residual,
        head_fn,
        head_scale,
        head_base,
        norm_eps=1e-6,
    )
    head_mixes = torch.nn.functional.linear(flat, head_fn) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + 1e-6
    )
    head_pre = torch.sigmoid(head_mixes * head_scale + head_base) + 1e-6
    head_ref = torch.sum(head_pre.unsqueeze(-1) * residual.float(), dim=2).to(
        residual.dtype
    )
    torch.testing.assert_close(head, head_ref, rtol=0, atol=0)


def test_parameter_bearings_are_exact_and_fp32() -> None:
    layer = MHC.Dsv4MhcLayerMixer(hidden_size=8)
    head = MHC.Dsv4MhcHeadMixer(hidden_size=8)
    assert list(dict(layer.named_parameters())) == [
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
    ]
    assert list(dict(head.named_parameters())) == [
        "hc_head_fn",
        "hc_head_base",
        "hc_head_scale",
    ]
    assert all(value.dtype == torch.float32 for value in layer.parameters())
    assert all(value.dtype == torch.float32 for value in head.parameters())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mixes_dtype", torch.bfloat16),
        ("mixes_width", 23),
        ("scale_width", 2),
        ("base_width", 23),
        ("hc_mult", 2),
        ("sinkhorn_iters", 19),
        ("eps", 1e-5),
    ],
)
def test_split_contract_rejects_precision_shape_and_algorithm_drift(
    field: str, value: object
) -> None:
    mixes = torch.randn(1, 1, 24, dtype=torch.float32)
    scale = torch.randn(3, dtype=torch.float32)
    base = torch.randn(24, dtype=torch.float32)
    kwargs = {}
    if field == "mixes_dtype":
        mixes = mixes.to(value)
    elif field == "mixes_width":
        mixes = mixes[..., : int(value)]
    elif field == "scale_width":
        scale = scale[: int(value)]
    elif field == "base_width":
        base = base[: int(value)]
    else:
        kwargs[field] = value
    with pytest.raises(MHC.Dsv4MhcContractError):
        MHC.hc_split_sinkhorn(mixes, scale, base, **kwargs)


def test_no_mutation_of_inputs() -> None:
    torch.manual_seed(6)
    tensors = [
        torch.randn(1, 2, 24, dtype=torch.float32),
        torch.randn(3, dtype=torch.float32),
        torch.randn(24, dtype=torch.float32),
    ]
    before = copy.deepcopy(tensors)
    MHC.hc_split_sinkhorn(*tensors)
    for actual, expected in zip(tensors, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
