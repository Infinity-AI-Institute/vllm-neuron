# SPDX-License-Identifier: Apache-2.0
"""Exact DeepSeek-V4-Flash mHC source and host-reference contract.

This module closes the ambiguity around the 261 main-model Hyper-Connection
parameters without claiming that the current NxDI model consumes them.  It
defines their exact names, shapes, dtype and TP ownership and implements the
official FP32 equations as a CPU-portable reference.  Compile remains blocked
until the NxDI model tree uses this contract at all 43 layers and at the head.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

DSV4_MHC_SCHEMA = "dsv4-main-model-mhc-contract-v1"
DSV4_MHC_CHECKPOINT_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DSV4_MHC_CONFIG_SHA256 = (
    "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
)
DSV4_MHC_INDEX_SHA256 = (
    "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
)
DSV4_MHC_SOURCE_KEYS_SHA256 = (
    "ad07358ebd20fce2a30d3ddd889880250d5280c523eb474a226a2982833d58d6"
)
DSV4_MHC_SOURCE_PARENT_COMMIT = "2dc3d6a2a125cad006426d77a2998c5dd4b7bd13"
DSV4_MHC_AUTH_PACKET_GIT_BLOB_SHA256 = (
    "b9cdcfdaabccfcd807a6fd5cf9cc19f03730368796f5b0d9dde86bb7c5986822"
)
DSV4_MHC_COMPILER_IMAGE = (
    "public.ecr.aws/neuron/pytorch-inference-neuronx@sha256:"
    "011d49c7495457fc2932dedd3fbecf67d28833a3f12c147377cee4d72889ebc1"
)
DSV4_MHC_HIDDEN_SIZE = 4096
DSV4_MHC_MULTIPLIER = 4
DSV4_MHC_MIX_ROWS = 24
DSV4_MHC_LAYERS = 43
DSV4_MHC_SINKHORN_ITERS = 20
DSV4_MHC_EPS = 1e-6


class Dsv4MhcContractError(ValueError):
    """The exact main-model mHC contract is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Dsv4MhcContractError(message)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Dsv4MhcTensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    ownership: str = "replicated_on_all_tp32_ranks"

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * 4

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "ownership": self.ownership,
        }


def build_dsv4_mhc_tensor_contract() -> tuple[Dsv4MhcTensorSpec, ...]:
    """Return the 261 exact, replicated FP32 main-model mHC parameters."""
    specs = [
        Dsv4MhcTensorSpec("hc_head_base", "F32", (DSV4_MHC_MULTIPLIER,)),
        Dsv4MhcTensorSpec(
            "hc_head_fn",
            "F32",
            (DSV4_MHC_MULTIPLIER, DSV4_MHC_MULTIPLIER * DSV4_MHC_HIDDEN_SIZE),
        ),
        Dsv4MhcTensorSpec("hc_head_scale", "F32", (1,)),
    ]
    for layer in range(DSV4_MHC_LAYERS):
        for stem in ("hc_attn", "hc_ffn"):
            prefix = f"layers.{layer}.{stem}"
            specs.extend(
                (
                    Dsv4MhcTensorSpec(f"{prefix}_base", "F32", (DSV4_MHC_MIX_ROWS,)),
                    Dsv4MhcTensorSpec(
                        f"{prefix}_fn",
                        "F32",
                        (
                            DSV4_MHC_MIX_ROWS,
                            DSV4_MHC_MULTIPLIER * DSV4_MHC_HIDDEN_SIZE,
                        ),
                    ),
                    Dsv4MhcTensorSpec(f"{prefix}_scale", "F32", (3,)),
                )
            )
    result = tuple(sorted(specs, key=lambda item: item.name))
    _require(len(result) == 261, "mHC tensor count drift")
    _require(len({item.name for item in result}) == 261, "duplicate mHC tensor name")
    _require(
        _canonical_sha256([item.name for item in result])
        == DSV4_MHC_SOURCE_KEYS_SHA256,
        "mHC source-key identity drift",
    )
    return result


def validate_dsv4_mhc_headers(headers: Mapping[str, Any]) -> str:
    """Validate exact dtype/shape for the 261 main-model mHC source headers.

    ``headers`` may contain the entire checkpoint. MTP layer 43 and later are
    intentionally outside this no-spec main-model contract.
    """
    expected = {item.name: item for item in build_dsv4_mhc_tensor_contract()}
    observed = {
        key: value
        for key, value in headers.items()
        if key in expected or ("hc_" in key and not key.startswith("layers.43."))
    }
    _require(set(observed) == set(expected), "mHC source key-set drift")
    for name, spec in expected.items():
        row = observed[name]
        dtype = row.dtype if hasattr(row, "dtype") else row.get("dtype")
        shape = row.shape if hasattr(row, "shape") else row.get("shape")
        _require(dtype == spec.dtype, f"mHC dtype drift: {name}")
        _require(tuple(shape) == spec.shape, f"mHC shape drift: {name}")
    return _canonical_sha256([item.to_mapping() for item in expected.values()])


def build_dsv4_mhc_integration_boundary() -> dict[str, Any]:
    """Return the exact inherited compile boundary; it authorizes no execution."""
    return {
        "schema": "dsv4-main-model-mhc-integration-boundary-v1",
        "source_parent_commit": DSV4_MHC_SOURCE_PARENT_COMMIT,
        "compile_authorization_git_blob_sha256": (DSV4_MHC_AUTH_PACKET_GIT_BLOB_SHA256),
        "model": {
            "repo_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
            "revision": DSV4_MHC_CHECKPOINT_REVISION,
            "config_sha256": DSV4_MHC_CONFIG_SHA256,
            "index_sha256": DSV4_MHC_INDEX_SHA256,
        },
        "compiler_image": DSV4_MHC_COMPILER_IMAGE,
        "topology": {
            "hardware": "trn2.48xlarge",
            "tp_degree": 32,
            "logical_neuroncore_config": 2,
            "ctx_batch_size": 1,
            "tkg_batch_size": 1,
            "sequence_buckets": [4096],
        },
        "emitted_state": {
            "rank_count": 32,
            "rank_checkpoint_dtype": "bfloat16",
            "compute_dtype": "bfloat16",
            "cache_dtype": "bfloat16",
            "runtime_weight_quantized": False,
            "speculative_decode": False,
            "mtp": False,
        },
        "compile_execution_policy": {
            "ownership_marker": "/mnt/compile/OWNERSHIP.md",
            "max_active_compiles": 2,
            "systemd_run_required": True,
            "unit_name_required": True,
            "nice": 15,
            "scope_forbidden": True,
            "docker_network": "none",
            "atomic_output_suffix": ".partial",
        },
        "claims": {
            "compile_permitted": False,
            "runtime_permitted": False,
            "correctness": False,
            "performance": False,
            "tokenomics": False,
        },
    }


def validate_dsv4_mhc_integration_boundary(candidate: Mapping[str, Any]) -> str:
    """Fail closed unless a future integration preserves every frozen field."""
    expected = build_dsv4_mhc_integration_boundary()
    _require(dict(candidate) == expected, "mHC integration boundary drift")
    _require(
        not any(expected["claims"].values()),
        "host-only mHC contract cannot authorize claims",
    )
    return _canonical_sha256(expected)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    hc_mult: int = DSV4_MHC_MULTIPLIER,
    sinkhorn_iters: int = DSV4_MHC_SINKHORN_ITERS,
    eps: float = DSV4_MHC_EPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Host reference for the pinned official mHC split/Sinkhorn kernel."""
    mix_rows = (2 + hc_mult) * hc_mult
    _require(mixes.dtype == torch.float32, "mHC mixes must be FP32")
    _require(mixes.ndim == 3 and mixes.shape[-1] == mix_rows, "mHC mixes shape drift")
    _require(scale.dtype == torch.float32 and scale.shape == (3,), "mHC scale drift")
    _require(
        base.dtype == torch.float32 and base.shape == (mix_rows,), "mHC base drift"
    )
    _require(hc_mult == 4, "DeepSeek-V4-Flash requires hc_mult=4")
    _require(sinkhorn_iters == 20, "DeepSeek-V4-Flash requires 20 Sinkhorn iterations")
    _require(eps == 1e-6, "DeepSeek-V4-Flash requires hc_eps=1e-6")

    pre = torch.sigmoid(mixes[..., :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :] * scale[2] + base[2 * hc_mult :]
    comb = comb.unflatten(-1, (hc_mult, hc_mult)).softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce four HC streams to one model stream and return post state."""
    _require(residual.ndim == 4 and residual.shape[-2] == 4, "mHC residual shape drift")
    flat = residual.flatten(2).float()
    _require(fn.dtype == torch.float32, "mHC fn must be FP32")
    _require(fn.shape == (24, flat.shape[-1]), "mHC fn shape drift")
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + norm_eps
    )
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base)
    reduced = torch.sum(pre.unsqueeze(-1) * residual.float(), dim=2)
    return reduced.to(residual.dtype), post, comb


def mhc_post(
    branch: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Expand one branch result back to four HC streams."""
    _require(branch.ndim == 3, "mHC branch shape drift")
    _require(residual.ndim == 4 and residual.shape[-2] == 4, "mHC residual shape drift")
    _require(post.shape == residual.shape[:-1], "mHC post shape drift")
    _require(comb.shape == (*residual.shape[:-2], 4, 4), "mHC comb shape drift")
    result = post.unsqueeze(-1) * branch.float().unsqueeze(-2)
    result = result + torch.sum(
        comb.unsqueeze(-1) * residual.float().unsqueeze(-2), dim=2
    )
    return result.to(branch.dtype)


def mhc_head(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    norm_eps: float,
    hc_eps: float = DSV4_MHC_EPS,
) -> torch.Tensor:
    """Collapse four HC streams before final normalization and LM head."""
    _require(residual.ndim == 4 and residual.shape[-2] == 4, "mHC head input drift")
    flat = residual.flatten(2).float()
    _require(
        fn.dtype == torch.float32 and fn.shape == (4, flat.shape[-1]),
        "mHC head fn drift",
    )
    _require(
        scale.dtype == torch.float32 and scale.shape == (1,), "mHC head scale drift"
    )
    _require(base.dtype == torch.float32 and base.shape == (4,), "mHC head base drift")
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + norm_eps
    )
    pre = torch.sigmoid(mixes * scale + base) + hc_eps
    return torch.sum(pre.unsqueeze(-1) * residual.float(), dim=2).to(residual.dtype)


class Dsv4MhcLayerMixer(nn.Module):
    """Parameter-bearing host primitive for one attention/FFN HC pair."""

    def __init__(self, hidden_size: int = DSV4_MHC_HIDDEN_SIZE) -> None:
        super().__init__()
        _require(
            type(hidden_size) is int and hidden_size > 0, "invalid mHC hidden size"
        )
        width = DSV4_MHC_MULTIPLIER * hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(24, width, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(24, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(24, width, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(24, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))


class Dsv4MhcHeadMixer(nn.Module):
    """Parameter-bearing host primitive for the final HC collapse."""

    def __init__(self, hidden_size: int = DSV4_MHC_HIDDEN_SIZE) -> None:
        super().__init__()
        _require(
            type(hidden_size) is int and hidden_size > 0, "invalid mHC hidden size"
        )
        self.hc_head_fn = nn.Parameter(
            torch.empty(4, 4 * hidden_size, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(4, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))


__all__ = [
    "DSV4_MHC_AUTH_PACKET_GIT_BLOB_SHA256",
    "DSV4_MHC_CHECKPOINT_REVISION",
    "DSV4_MHC_COMPILER_IMAGE",
    "DSV4_MHC_CONFIG_SHA256",
    "DSV4_MHC_EPS",
    "DSV4_MHC_HIDDEN_SIZE",
    "DSV4_MHC_INDEX_SHA256",
    "DSV4_MHC_LAYERS",
    "DSV4_MHC_MIX_ROWS",
    "DSV4_MHC_MULTIPLIER",
    "DSV4_MHC_SCHEMA",
    "DSV4_MHC_SINKHORN_ITERS",
    "DSV4_MHC_SOURCE_KEYS_SHA256",
    "DSV4_MHC_SOURCE_PARENT_COMMIT",
    "Dsv4MhcContractError",
    "Dsv4MhcHeadMixer",
    "Dsv4MhcLayerMixer",
    "Dsv4MhcTensorSpec",
    "build_dsv4_mhc_integration_boundary",
    "build_dsv4_mhc_tensor_contract",
    "hc_split_sinkhorn",
    "mhc_head",
    "mhc_post",
    "mhc_pre",
    "validate_dsv4_mhc_headers",
    "validate_dsv4_mhc_integration_boundary",
]
