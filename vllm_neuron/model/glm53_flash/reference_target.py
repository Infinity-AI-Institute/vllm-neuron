"""Bounded reader for one immutable GLM-5.3 reference target.

The reader is deliberately row-oriented.  A reference producer may publish
full-vocabulary rows from the original target checkpoint, but this module does
not load the 600+ GiB checkpoint or infer that a BF16/FP32 file is canonical.
The manifest must bind the pinned checkpoint metadata, explicit weight
semantics, row dtype/shape, and every row's bytes.  It is therefore useful on
the CPU host before a device capture exists while remaining fail-closed when
no genuine original-target producer is available.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from .checkpoint_converter import (
        GLM53_CHECKPOINT_REVISION,
        GLM53_CONFIG_SHA256,
        GLM53_INDEX_SHA256,
    )
    from .raw_capture import GLM53_RAW_CAPTURE_VOCAB_SIZE, GLM53_REFERENCE_SEMANTICS
except ImportError:  # pragma: no cover - direct-file qualification tests
    GLM53_CHECKPOINT_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
    GLM53_CONFIG_SHA256 = (
        "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
    )
    GLM53_INDEX_SHA256 = (
        "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05"
    )
    GLM53_RAW_CAPTURE_VOCAB_SIZE = 154_880
    GLM53_REFERENCE_SEMANTICS = frozenset(
        {
            "native-block-fp8",
            "native-block-fp8-dequantized-bfloat16",
            "original-checkpoint-cpu-fp32",
        }
    )

GLM53_REFERENCE_TARGET_SCHEMA = "glm53-reference-target-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DTYPES = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}


class Glm53ReferenceTargetError(ValueError):
    """A reference target is absent, ambiguous, or not bound to GLM-5.3."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Glm53ReferenceTargetError(message)


def _sha(value: Any, name: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256",
    )
    return value


def _text(value: Any, name: str) -> str:
    _require(
        isinstance(value, str) and value and value.strip() == value,
        f"{name} must be non-empty trimmed text",
    )
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate reference manifest key: {key!r}")
        result[key] = value
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise Glm53ReferenceTargetError(
            f"cannot read reference manifest: {path}"
        ) from exc
    _require(isinstance(value, dict), "reference manifest root must be an object")
    return value


@dataclass(frozen=True)
class Glm53ReferenceTarget:
    """One explicitly selected reference bank, addressed by capture row."""

    reference_id: str
    checkpoint_revision: str
    config_sha256: str
    index_sha256: str
    semantics: str
    dtype: str
    vocab_size: int
    manifest_path: Path
    rows: dict[tuple[int, str, int], dict[str, Any]]
    loader_versions: dict[str, str] | None = None

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> Glm53ReferenceTarget:
        path = Path(manifest_path).resolve(strict=True)
        manifest = _load_manifest(path)
        _require(
            manifest.get("schema") == GLM53_REFERENCE_TARGET_SCHEMA,
            "reference target schema drift",
        )
        reference_id = _text(manifest.get("reference_id"), "reference_id")
        _require(
            manifest.get("checkpoint_revision") == GLM53_CHECKPOINT_REVISION,
            "reference checkpoint revision is not the pinned GLM-5.3 revision",
        )
        _require(
            manifest.get("config_sha256") == GLM53_CONFIG_SHA256,
            "reference config identity drift",
        )
        _require(
            manifest.get("index_sha256") == GLM53_INDEX_SHA256,
            "reference index identity drift",
        )
        semantics = manifest.get("semantics")
        _require(
            semantics in GLM53_REFERENCE_SEMANTICS,
            "reference semantics are not an explicitly supported target",
        )
        dtype = manifest.get("dtype")
        _require(dtype in _DTYPES, "reference dtype must be explicitly BF16 or FP32")
        _require(
            manifest.get("vocab_size") == GLM53_RAW_CAPTURE_VOCAB_SIZE,
            "reference vocabulary width drift",
        )
        raw_rows = manifest.get("rows")
        _require(
            isinstance(raw_rows, list) and raw_rows,
            "reference target must contain at least one row",
        )
        rows: dict[tuple[int, str, int], dict[str, Any]] = {}
        for row in raw_rows:
            _require(isinstance(row, dict), "reference row must be an object")
            slot = row.get("slot")
            prompt_id = row.get("prompt_id")
            position = row.get("position")
            _require(type(slot) is int and slot >= 0, "reference slot is invalid")
            _text(prompt_id, "reference prompt_id")
            _require(
                type(position) is int and position >= 0, "reference position is invalid"
            )
            key = (slot, prompt_id, position)
            _require(key not in rows, f"duplicate reference row: {key!r}")
            _require(row.get("dtype") == dtype, f"reference row dtype drift: {key!r}")
            _require(
                row.get("shape") == [GLM53_RAW_CAPTURE_VOCAB_SIZE],
                f"reference row shape drift: {key!r}",
            )
            _sha(row.get("raw_sha256"), f"reference row hash {key!r}")
            relative = row.get("relative_path")
            _require(
                isinstance(relative, str)
                and not Path(relative).is_absolute()
                and ".." not in Path(relative).parts,
                f"reference row path is unsafe: {key!r}",
            )
            rows[key] = dict(row)
        raw_loader_versions = manifest.get("loader_versions")
        loader_versions = None
        if raw_loader_versions is not None:
            _require(
                isinstance(raw_loader_versions, dict) and raw_loader_versions,
                "loader_versions must be a non-empty object",
            )
            loader_versions = {}
            for key, value in raw_loader_versions.items():
                _text(key, "loader version key")
                loader_versions[key] = _text(value, f"loader version {key}")
        return cls(
            reference_id,
            GLM53_CHECKPOINT_REVISION,
            GLM53_CONFIG_SHA256,
            GLM53_INDEX_SHA256,
            semantics,
            dtype,
            GLM53_RAW_CAPTURE_VOCAB_SIZE,
            path,
            rows,
            loader_versions,
        )

    def load_row(self, *, slot: int, prompt_id: str, position: int) -> torch.Tensor:
        """Load and verify exactly one full-vocabulary row."""

        key = (slot, prompt_id, position)
        row = self.rows.get(key)
        _require(row is not None, f"reference row is missing: {key!r}")
        path = self.manifest_path.parent / row["relative_path"]
        _require(
            path.is_file() and not path.is_symlink(),
            f"reference row is missing: {path}",
        )
        raw = path.read_bytes()
        _require(
            hashlib.sha256(raw).hexdigest() == row["raw_sha256"],
            f"reference row hash drift: {key!r}",
        )
        dtype = _DTYPES[self.dtype]
        expected_bytes = (
            GLM53_RAW_CAPTURE_VOCAB_SIZE * torch.empty((), dtype=dtype).element_size()
        )
        _require(len(raw) == expected_bytes, f"reference row byte count drift: {key!r}")
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
        _require(
            tuple(tensor.shape) == (GLM53_RAW_CAPTURE_VOCAB_SIZE,),
            f"reference row shape drift after load: {key!r}",
        )
        _require(
            bool(torch.isfinite(tensor).all().item()),
            f"reference row is non-finite: {key!r}",
        )
        return tensor


__all__ = [
    "GLM53_REFERENCE_TARGET_SCHEMA",
    "Glm53ReferenceTarget",
    "Glm53ReferenceTargetError",
]
