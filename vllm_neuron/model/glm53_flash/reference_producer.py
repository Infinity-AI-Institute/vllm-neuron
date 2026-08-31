"""CPU-only producer contract for an original/native GLM-5.3 target bank.

This module is the producer-side seam, not a model implementation.  It binds
the exact released checkpoint, explicit native weight semantics, loader
versions, and the campaign's four-prompts-by-ten-positions feedback matrix.
The caller supplies the already-qualified original-target loader and runner;
the producer validates and writes one full-vocabulary row at a time.  It never
assumes that FP32 output means FP32 execution and never substitutes Q4 data.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

import torch

try:
    from .checkpoint_converter import (
        GLM53_CHECKPOINT_REVISION,
        GLM53_CONFIG_SHA256,
        GLM53_INDEX_SHA256,
    )
    from .raw_capture import (
        GLM53_RAW_CAPTURE_VOCAB_SIZE,
        GLM53_REFERENCE_SEMANTICS,
    )
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

GLM53_REFERENCE_PRODUCER_SCHEMA = "glm53-reference-producer-v1"
_DTYPES = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}


class Glm53ReferenceProducerError(ValueError):
    """The original-target producer input or emitted bank is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Glm53ReferenceProducerError(message)


def _text(value: Any, name: str) -> str:
    _require(
        isinstance(value, str) and value and value.strip() == value,
        f"{name} must be non-empty trimmed text",
    )
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bounded_rows(values: Iterable[torch.Tensor], expected: int) -> list[torch.Tensor]:
    """Materialize exactly ``expected`` rows, detecting overflow without hanging."""

    rows = list(islice(iter(values), expected + 1))
    _require(
        len(rows) == expected,
        f"runner returned wrong position count: expected {expected}, got {len(rows)}",
    )
    return rows


@dataclass(frozen=True)
class Glm53OriginalTargetProducerSpec:
    """Immutable producer identity and exact feedback coverage."""

    reference_id: str
    checkpoint_dir: Path
    loader_versions: Mapping[str, str]
    semantics: str
    output_dtype: str = "torch.float32"
    prompt_ids: tuple[str, ...] = (
        "feedback-0",
        "feedback-1",
        "feedback-2",
        "feedback-3",
    )
    positions: tuple[int, ...] = tuple(range(10))
    vocab_size: int = GLM53_RAW_CAPTURE_VOCAB_SIZE
    checkpoint_revision: str = GLM53_CHECKPOINT_REVISION
    config_sha256: str = GLM53_CONFIG_SHA256
    index_sha256: str = GLM53_INDEX_SHA256
    tokenizer_versions: Mapping[str, str] = field(default_factory=dict)
    prompt_token_ids: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.reference_id, "reference_id")
        _require(
            self.checkpoint_revision == GLM53_CHECKPOINT_REVISION,
            "producer checkpoint revision is not pinned",
        )
        _require(
            self.config_sha256 == GLM53_CONFIG_SHA256, "producer config identity drift"
        )
        _require(
            self.index_sha256 == GLM53_INDEX_SHA256, "producer index identity drift"
        )
        _require(
            self.semantics in GLM53_REFERENCE_SEMANTICS,
            "producer semantics are not explicitly supported",
        )
        _require(
            self.output_dtype in _DTYPES, "producer output dtype must be BF16 or FP32"
        )
        _require(
            self.vocab_size == GLM53_RAW_CAPTURE_VOCAB_SIZE,
            "producer vocabulary width drift",
        )
        _require(
            len(self.prompt_ids) == 4 and len(set(self.prompt_ids)) == 4,
            "producer requires exactly four unique feedback prompts",
        )
        _require(
            all(_text(prompt, "prompt_id") for prompt in self.prompt_ids),
            "producer prompt ids are invalid",
        )
        _require(
            self.positions == tuple(range(10)),
            "producer requires exactly positions 0 through 9",
        )
        _require(
            isinstance(self.loader_versions, Mapping) and self.loader_versions,
            "producer requires explicit loader versions",
        )
        for key, value in self.loader_versions.items():
            _text(key, "loader version key")
            _text(value, f"loader version {key}")
        has_tokenizer_versions = bool(self.tokenizer_versions)
        has_prompt_token_ids = bool(self.prompt_token_ids)
        _require(
            has_tokenizer_versions == has_prompt_token_ids,
            "tokenizer versions and prompt token ids must be bound together",
        )
        if has_tokenizer_versions:
            for key, value in self.tokenizer_versions.items():
                _text(key, "tokenizer version key")
                _text(value, f"tokenizer version {key}")
            _require(
                set(self.prompt_token_ids) == set(self.prompt_ids),
                "prompt token ids must bind exactly the four feedback prompts",
            )
            for prompt_id in self.prompt_ids:
                token_ids = self.prompt_token_ids[prompt_id]
                _require(
                    isinstance(token_ids, tuple)
                    and token_ids
                    and all(
                        type(token_id) is int and token_id >= 0
                        for token_id in token_ids
                    ),
                    f"prompt token ids are invalid for {prompt_id}",
                )

    @property
    def expected_rows(self) -> tuple[tuple[int, str, int], ...]:
        return tuple(
            (0, prompt_id, position)
            for prompt_id in self.prompt_ids
            for position in self.positions
        )


class Glm53OriginalTargetProducer:
    """Run an injected original-target implementation and publish 40 rows."""

    def __init__(self, spec: Glm53OriginalTargetProducerSpec) -> None:
        self.spec = spec

    def produce(
        self,
        *,
        loader: Callable[[Path], Any],
        run_prompt: Callable[..., Iterable[torch.Tensor]],
        output_dir: str | Path,
    ) -> Path:
        """Emit a verified manifest and row files, or nothing publishable."""

        checkpoint_dir = self.spec.checkpoint_dir.resolve(strict=True)
        _require(
            checkpoint_dir.name == self.spec.checkpoint_revision,
            "producer checkpoint path does not name the pinned revision",
        )
        destination = Path(output_dir).resolve()
        _require(
            not destination.exists(),
            "producer output already exists; refusing overwrite",
        )
        partial = destination.with_name(
            f".{destination.name}.partial-{uuid.uuid4().hex}"
        )
        partial.mkdir(parents=True)
        rows: list[dict[str, Any]] = []
        try:
            model = loader(checkpoint_dir)
            for prompt_id in self.spec.prompt_ids:
                if self.spec.prompt_token_ids:
                    values = run_prompt(
                        model,
                        prompt_id,
                        self.spec.positions,
                        self.spec.prompt_token_ids[prompt_id],
                    )
                else:
                    values = run_prompt(model, prompt_id, self.spec.positions)
                values = _bounded_rows(values, len(self.spec.positions))
                for position, logits in zip(self.spec.positions, values, strict=True):
                    _require(
                        isinstance(logits, torch.Tensor),
                        f"runner returned non-tensor for {prompt_id}:{position}",
                    )
                    _require(
                        tuple(logits.shape) == (self.spec.vocab_size,),
                        f"runner row is not full vocabulary for {prompt_id}:{position}",
                    )
                    _require(
                        str(logits.dtype) == self.spec.output_dtype,
                        f"runner dtype drift for {prompt_id}:{position}",
                    )
                    _require(
                        bool(torch.isfinite(logits).all().item()),
                        f"runner row is non-finite for {prompt_id}:{position}",
                    )
                    raw = (
                        logits.detach()
                        .cpu()
                        .contiguous()
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    )
                    relative = Path("rows") / f"slot0-{prompt_id}-{position}.bin"
                    row_path = partial / relative
                    row_path.parent.mkdir(exist_ok=True)
                    row_path.write_bytes(raw)
                    rows.append(
                        {
                            "slot": 0,
                            "prompt_id": prompt_id,
                            "position": position,
                            "dtype": self.spec.output_dtype,
                            "shape": [self.spec.vocab_size],
                            "relative_path": relative.as_posix(),
                            "raw_sha256": _sha256(raw),
                        }
                    )
            _require(len(rows) == 40, "producer did not emit the required 4x10 matrix")
            manifest = {
                "schema": "glm53-reference-target-v1",
                "producer_schema": GLM53_REFERENCE_PRODUCER_SCHEMA,
                "reference_id": self.spec.reference_id,
                "checkpoint_revision": self.spec.checkpoint_revision,
                "config_sha256": self.spec.config_sha256,
                "index_sha256": self.spec.index_sha256,
                "semantics": self.spec.semantics,
                "dtype": self.spec.output_dtype,
                "vocab_size": self.spec.vocab_size,
                "loader_versions": dict(self.spec.loader_versions),
                "rows": rows,
            }
            if self.spec.prompt_token_ids:
                manifest["tokenizer_versions"] = dict(self.spec.tokenizer_versions)
                manifest["prompt_token_ids"] = {
                    prompt_id: list(self.spec.prompt_token_ids[prompt_id])
                    for prompt_id in self.spec.prompt_ids
                }
            (partial / "reference.json").write_text(
                json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            partial.replace(destination)
        except Exception:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        return destination / "reference.json"


__all__ = [
    "GLM53_REFERENCE_PRODUCER_SCHEMA",
    "Glm53OriginalTargetProducer",
    "Glm53OriginalTargetProducerSpec",
    "Glm53ReferenceProducerError",
]
