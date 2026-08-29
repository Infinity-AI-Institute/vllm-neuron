# SPDX-License-Identifier: Apache-2.0
"""Fail-closed conversion primitives for the GLM-5.3-Flash checkpoint.

This module deliberately does not reuse ``glm52_moe_dsa``.  GLM-5.3 is a
``Glm5NextForConditionalGeneration`` checkpoint with a nested text backbone,
34 KDA layers, 11 DSA layers, mHC parameters, a vision tower, one MTP layer,
and reciprocal 128x128 block-FP8 scales.  The GLM-5.2 converter assumes a
different architecture and per-tensor conversion contract.

The entry-point here is metadata-only and therefore safe on CPU-only hosts.
It validates the immutable production config/index before a future streaming
writer reads any 306 GiB weight shard.  Tensor conversion helpers cover the
two correctness-sensitive transforms already required by that writer:
reciprocal block-FP8 dequantization and per-head KDA Q/K/V convolution layout.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

GLM53_CHECKPOINT_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
GLM53_CONFIG_SHA256 = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
GLM53_INDEX_SHA256 = "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05"
GLM53_ARCHITECTURE = "Glm5NextForConditionalGeneration"
GLM53_MODEL_TYPE = "glm5_next"
GLM53_TEXT_MODEL_TYPE = "glm5_next_text"
GLM53_BLOCK_SIZE = (128, 128)
GLM53_DSA_LAYERS = tuple(range(3, 45, 4))
GLM53_KDA_LAYERS = tuple(i for i in range(45) if i not in GLM53_DSA_LAYERS)
GLM53_EXPECTED_TENSORS = 76_108
GLM53_EXPECTED_SCALES = 37_338
GLM53_EXPECTED_VISION_TENSORS = 347
GLM53_EXPECTED_MTP_TENSORS = 1_760
GLM53_EXPECTED_INDEXER_TENSORS = 84
SCALE_SUFFIX = ".weight_scale_inv"

TensorPolicy = Literal[
    "block_fp8_weight",
    "block_fp8_scale",
    "bf16_holdout",
    "drop_mtp",
    "drop_vision",
]


class Glm53ArchitectureMismatch(ValueError):
    """Raised when GLM-5.2 or another architecture reaches this adapter."""


@dataclass(frozen=True)
class Glm53CheckpointReport:
    architecture: str
    model_type: str
    text_model_type: str
    tensor_count: int
    block_scale_count: int
    vision_tensor_count: int
    mtp_tensor_count: int
    indexer_tensor_count: int
    dsa_layers: tuple[int, ...]
    kda_layers: tuple[int, ...]
    weight_block_size: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Glm53ArchitectureMismatch(message)


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("text_config")
    _require(isinstance(value, Mapping), "GLM-5.3 requires nested text_config")
    return value


def _validate_config(
    config: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    architectures = tuple(config.get("architectures", ()))
    _require(
        GLM53_ARCHITECTURE in architectures,
        "expected Glm5NextForConditionalGeneration; Glm52MoeDsa is not convertible by this adapter",
    )
    _require(
        config.get("model_type") == GLM53_MODEL_TYPE, "expected model_type=glm5_next"
    )
    text = _text_config(config)
    _require(
        text.get("model_type") == GLM53_TEXT_MODEL_TYPE,
        "expected text_config.model_type=glm5_next_text",
    )
    _require(text.get("num_hidden_layers") == 45, "expected exactly 45 text layers")

    layer_types = tuple(text.get("layer_types", ()))
    _require(len(layer_types) == 45, "layer_types must contain 45 entries")
    dsa = tuple(
        i for i, kind in enumerate(layer_types) if kind == "deepseek_sparse_attention"
    )
    kda = tuple(i for i, kind in enumerate(layer_types) if kind == "linear_attention")
    _require(dsa == GLM53_DSA_LAYERS, f"unexpected DSA schedule: {dsa}")
    _require(kda == GLM53_KDA_LAYERS, f"unexpected KDA schedule: {kda}")

    mlp_types = tuple(text.get("mlp_layer_types", ()))
    _require(
        mlp_types == ("dense",) * 3 + ("sparse",) * 42,
        "expected 3 dense then 42 sparse MLP layers",
    )

    quant = config.get("quantization_config")
    _require(isinstance(quant, Mapping), "missing quantization_config")
    _require(quant.get("quant_method") == "fp8", "expected quant_method=fp8")
    _require(quant.get("fmt") == "e4m3", "expected fmt=e4m3")
    _require(
        quant.get("activation_scheme") == "dynamic",
        "expected dynamic activation scaling",
    )
    raw_block_size = quant.get("weight_block_size")
    _require(
        isinstance(raw_block_size, (list, tuple)),
        "expected reciprocal 128x128 block-FP8 scales",
    )
    block_size = tuple(raw_block_size)
    _require(
        block_size == GLM53_BLOCK_SIZE, "expected reciprocal 128x128 block-FP8 scales"
    )
    return dsa, kda


def _layer_number(key: str) -> int | None:
    match = re.match(r"^model\.language_model\.layers\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def classify_tensor(key: str, weight_map: Mapping[str, str]) -> TensorPolicy:
    """Classify one immutable-checkpoint tensor without GLM-5.2 heuristics."""
    if key.startswith("model.visual."):
        return "drop_vision"
    if _layer_number(key) == 45:
        return "drop_mtp"
    if key.endswith(SCALE_SUFFIX):
        base = key[: -len("_scale_inv")]
        if base not in weight_map:
            raise Glm53ArchitectureMismatch(f"orphan reciprocal block scale: {key}")
        return "block_fp8_scale"
    if f"{key}_scale_inv" in weight_map:
        return "block_fp8_weight"
    return "bf16_holdout"


def preflight_checkpoint_metadata(
    config: Mapping[str, Any], weight_map: Mapping[str, str]
) -> Glm53CheckpointReport:
    """Validate the exact production architecture and complete index.

    This intentionally rejects partial downloads and GLM-5.2 indices.  A
    converter may only proceed after this report is produced successfully.
    """
    dsa, kda = _validate_config(config)
    keys = tuple(weight_map)
    _require(
        len(keys) == GLM53_EXPECTED_TENSORS,
        f"expected {GLM53_EXPECTED_TENSORS} indexed tensors; got {len(keys)}",
    )

    scales = tuple(key for key in keys if key.endswith(SCALE_SUFFIX))
    vision = tuple(key for key in keys if key.startswith("model.visual."))
    mtp = tuple(key for key in keys if _layer_number(key) == 45)
    indexer = tuple(key for key in keys if ".self_attn.indexer." in key)
    _require(
        len(scales) == GLM53_EXPECTED_SCALES,
        f"expected {GLM53_EXPECTED_SCALES} reciprocal scales; got {len(scales)}",
    )
    _require(
        len(vision) == GLM53_EXPECTED_VISION_TENSORS,
        f"expected {GLM53_EXPECTED_VISION_TENSORS} vision tensors; got {len(vision)}",
    )
    _require(
        len(mtp) == GLM53_EXPECTED_MTP_TENSORS,
        f"expected {GLM53_EXPECTED_MTP_TENSORS} MTP tensors; got {len(mtp)}",
    )
    _require(
        len(indexer) == GLM53_EXPECTED_INDEXER_TENSORS,
        f"expected {GLM53_EXPECTED_INDEXER_TENSORS} DSA-indexer tensors; got {len(indexer)}",
    )
    for scale_key in scales:
        classify_tensor(scale_key, weight_map)

    return Glm53CheckpointReport(
        architecture=GLM53_ARCHITECTURE,
        model_type=GLM53_MODEL_TYPE,
        text_model_type=GLM53_TEXT_MODEL_TYPE,
        tensor_count=len(keys),
        block_scale_count=len(scales),
        vision_tensor_count=len(vision),
        mtp_tensor_count=len(mtp),
        indexer_tensor_count=len(indexer),
        dsa_layers=dsa,
        kda_layers=kda,
        weight_block_size=GLM53_BLOCK_SIZE,
    )


def preflight_checkpoint_dir(checkpoint_dir: str | Path) -> Glm53CheckpointReport:
    """Bind an on-disk HF snapshot to the immutable revision and metadata.

    Hugging Face cache snapshots resolve to a directory named by their full
    commit SHA.  Requiring that name and hashing both metadata files prevents
    a shape-compatible fabricated index from inheriting this checkpoint's
    approval.
    """
    root = Path(checkpoint_dir).resolve(strict=True)
    if root.name != GLM53_CHECKPOINT_REVISION:
        raise Glm53ArchitectureMismatch(
            f"checkpoint directory must resolve to pinned revision "
            f"{GLM53_CHECKPOINT_REVISION}; got {root.name!r}"
        )
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    _require_file_sha256(config_path, GLM53_CONFIG_SHA256)
    _require_file_sha256(index_path, GLM53_INDEX_SHA256)
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    with index_path.open(encoding="utf-8") as stream:
        index = json.load(stream)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise Glm53ArchitectureMismatch(
            "model.safetensors.index.json has no weight_map"
        )
    return preflight_checkpoint_metadata(config, weight_map)


def _require_file_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise Glm53ArchitectureMismatch(
            f"missing pinned metadata file: {path.name}"
        ) from exc
    actual = digest.hexdigest()
    if actual != expected:
        raise Glm53ArchitectureMismatch(
            f"{path.name} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def dequantize_block_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int] = GLM53_BLOCK_SIZE,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize ``weight`` using reciprocal per-block scales.

    GLM-5.3 stores ``weight_scale_inv``: multiplication is required.  The
    function validates exact block geometry and never broadcast-guesses.
    """
    if weight.ndim < 2 or scale_inv.ndim < 2:
        raise ValueError("weight and scale_inv must both be at least rank 2")
    if not torch.is_floating_point(scale_inv) or scale_inv.numel() == 0:
        raise TypeError("weight_scale_inv must be a non-empty floating tensor")
    if not torch.isfinite(scale_inv).all() or torch.any(scale_inv <= 0):
        raise ValueError("weight_scale_inv must be finite and strictly positive")
    block_out, block_in = block_size
    if block_out <= 0 or block_in <= 0:
        raise ValueError(f"invalid block_size={block_size}")
    out_features, in_features = weight.shape[-2:]
    expected = (math.ceil(out_features / block_out), math.ceil(in_features / block_in))
    if tuple(scale_inv.shape[-2:]) != expected:
        raise ValueError(
            f"scale shape {tuple(scale_inv.shape)} does not match weight {tuple(weight.shape)} "
            f"for block_size={block_size}; expected trailing shape {expected}"
        )
    expanded = scale_inv.to(torch.float32)
    expanded = expanded.repeat_interleave(block_out, -2).repeat_interleave(block_in, -1)
    expanded = expanded[..., :out_features, :in_features]
    return (weight.to(torch.float32) * expanded).to(out_dtype)


def kda_conv1d_per_head_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Fuse Q/K/V depthwise kernels as ``[head][q,k,v][channel]``.

    A plain ``torch.cat((q, k, v), dim=0)`` is stream-major and silently
    disagrees with the Glm5Next KDA forward's ``view(..., heads, 3*dim)``.
    """
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"q/k/v convolution shapes differ: {q.shape}, {k.shape}, {v.shape}"
        )
    expected_channels = num_heads * head_dim
    if q.ndim != 3 or q.shape[0] != expected_channels:
        raise ValueError(
            f"expected [num_heads*head_dim, 1, kernel], got {tuple(q.shape)}"
        )
    shaped = [
        tensor.reshape(num_heads, head_dim, *tensor.shape[1:]) for tensor in (q, k, v)
    ]
    return (
        torch.stack(shaped, dim=1)
        .reshape(num_heads * 3 * head_dim, *q.shape[1:])
        .contiguous()
    )


__all__ = [
    "GLM53_CHECKPOINT_REVISION",
    "GLM53_CONFIG_SHA256",
    "GLM53_INDEX_SHA256",
    "Glm53ArchitectureMismatch",
    "Glm53CheckpointReport",
    "classify_tensor",
    "dequantize_block_fp8",
    "kda_conv1d_per_head_layout",
    "preflight_checkpoint_dir",
    "preflight_checkpoint_metadata",
]
