"""Fail-closed full-vocabulary capture contract for GLM-5.3.

This module prepares a later device capture; it does not run a model, choose a
reference bank, or authorize a card load.  Each row is validated and reduced to
metadata immediately, so a caller can persist the raw row without retaining a
full ``slots x prompts x positions x vocab`` tensor in the host process.

The candidate checkpoint is the released native block-FP8 checkpoint after the
reviewed converter's reciprocal-scale-to-BF16 transform.  That is explicitly
different from both a Q4 diagnostic bank and an FP32-storage/FP32-execution
claim.  A canonical reference is therefore optional at preparation time and
must be bound consistently if supplied later.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import torch

try:
    from .checkpoint_converter import GLM53_CHECKPOINT_REVISION
except ImportError:  # pragma: no cover - direct file qualification tests
    # Keep direct-file qualification independent of the repository's optional
    # vLLM import surface.  The value is the same immutable pinned revision.
    GLM53_CHECKPOINT_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"

GLM53_RAW_CAPTURE_SCHEMA = "glm53-full-vocab-raw-capture-v1"
GLM53_RAW_CAPTURE_MODEL = "GLM-5.3-Flash"
GLM53_RAW_CAPTURE_VOCAB_SIZE = 154_880
GLM53_CANDIDATE_WEIGHT_SEMANTICS = "native-block-fp8-dequantized-bfloat16"
GLM53_REFERENCE_SEMANTICS = frozenset(
    {
        "native-block-fp8",
        "native-block-fp8-dequantized-bfloat16",
        "original-checkpoint-cpu-fp32",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class Glm53RawCaptureError(ValueError):
    """A capture row or its immutable identity contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Glm53RawCaptureError(message)


def _sha256_hex(value: str, name: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256",
    )
    return value


def _text(value: Any, name: str) -> str:
    _require(
        isinstance(value, str) and value and value.strip() == value,
        f"{name} must be a non-empty trimmed string",
    )
    return value


@dataclass(frozen=True)
class Glm53RawCapturePlan:
    """Immutable row shape and provenance contract for one capture."""

    checkpoint_revision: str
    emitted_config_sha256: str
    rank_bundle_sha256: str
    prompt_ids: tuple[str, ...]
    positions: tuple[int, ...]
    slot_count: int
    vocab_size: int = GLM53_RAW_CAPTURE_VOCAB_SIZE
    output_dtype: str = "torch.bfloat16"
    candidate_weight_semantics: str = GLM53_CANDIDATE_WEIGHT_SEMANTICS
    canonical_reference_id: str | None = None
    canonical_reference_semantics: str | None = None

    def __post_init__(self) -> None:
        _require(
            self.checkpoint_revision == GLM53_CHECKPOINT_REVISION,
            "capture checkpoint revision is not the pinned GLM-5.3 revision",
        )
        _sha256_hex(self.emitted_config_sha256, "emitted_config_sha256")
        _sha256_hex(self.rank_bundle_sha256, "rank_bundle_sha256")
        _require(
            self.prompt_ids and len(set(self.prompt_ids)) == len(self.prompt_ids),
            "prompt_ids must be non-empty and unique",
        )
        _require(
            all(_text(item, "prompt_id") for item in self.prompt_ids),
            "prompt_ids must be text",
        )
        _require(
            self.positions and len(set(self.positions)) == len(self.positions),
            "positions must be non-empty and unique",
        )
        _require(
            all(type(item) is int and item >= 0 for item in self.positions),
            "positions must be non-negative integers",
        )
        _require(
            type(self.slot_count) is int and self.slot_count > 0,
            "slot_count must be positive",
        )
        _require(
            self.vocab_size == GLM53_RAW_CAPTURE_VOCAB_SIZE,
            "GLM-5.3 full-vocabulary width drift",
        )
        _require(
            self.output_dtype in {"torch.bfloat16", "torch.float32"},
            "capture output dtype must be explicitly BF16 or FP32",
        )
        _require(
            self.candidate_weight_semantics == GLM53_CANDIDATE_WEIGHT_SEMANTICS,
            "candidate weight semantics must remain native block-FP8 dequantized BF16",
        )
        if self.canonical_reference_id is not None:
            _text(self.canonical_reference_id, "canonical_reference_id")
            _require(
                self.canonical_reference_semantics in GLM53_REFERENCE_SEMANTICS,
                "canonical reference semantics must be explicitly supported",
            )
        else:
            _require(
                self.canonical_reference_semantics is None,
                "reference semantics cannot be supplied without a reference id",
            )

    @property
    def expected_rows(self) -> frozenset[tuple[int, str, int]]:
        return frozenset(
            (slot, prompt_id, position)
            for slot in range(self.slot_count)
            for prompt_id in self.prompt_ids
            for position in self.positions
        )


@dataclass(frozen=True)
class Glm53RawCaptureRow:
    slot: int
    prompt_id: str
    position: int
    dtype: str
    shape: tuple[int, ...]
    raw_sha256: str
    argmax: int
    reference_id: str | None


class Glm53RawCapture:
    """Validate full-vocabulary rows while retaining only row metadata."""

    def __init__(self, plan: Glm53RawCapturePlan) -> None:
        self.plan = plan
        self._rows: dict[tuple[int, str, int], Glm53RawCaptureRow] = {}

    def record_logits(
        self,
        *,
        slot: int,
        prompt_id: str,
        position: int,
        logits: torch.Tensor,
        reference_id: str | None = None,
    ) -> Glm53RawCaptureRow:
        """Validate one complete row before the caller persists its raw bytes."""

        key = (slot, prompt_id, position)
        _require(
            key in self.plan.expected_rows, f"row is outside capture plan: {key!r}"
        )
        _require(key not in self._rows, f"duplicate capture row: {key!r}")
        _require(
            reference_id == self.plan.canonical_reference_id,
            "row reference identity does not match the selected canonical reference",
        )
        _require(isinstance(logits, torch.Tensor), "logits must be a torch.Tensor")
        _require(
            tuple(logits.shape) == (self.plan.vocab_size,),
            "capture row must be exactly one full-vocabulary vector",
        )
        _require(
            str(logits.dtype) == self.plan.output_dtype,
            f"capture row dtype drift: expected {self.plan.output_dtype}, got {logits.dtype}",
        )
        _require(
            bool(torch.isfinite(logits).all().item()),
            "capture row contains non-finite values",
        )
        raw = logits.detach().cpu().contiguous()
        raw_bytes = raw.view(torch.uint8).numpy().tobytes()
        row = Glm53RawCaptureRow(
            slot=slot,
            prompt_id=prompt_id,
            position=position,
            dtype=str(logits.dtype),
            shape=tuple(logits.shape),
            raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            argmax=int(torch.argmax(logits).item()),
            reference_id=reference_id,
        )
        self._rows[key] = row
        return row

    def finalize(self) -> dict[str, Any]:
        """Return a non-authorizing receipt only after every planned row exists."""

        missing = sorted(self.plan.expected_rows - self._rows.keys())
        _require(not missing, f"capture is missing rows: {missing[:4]!r}")
        rows = [self._rows[key] for key in sorted(self._rows)]
        return {
            "schema": GLM53_RAW_CAPTURE_SCHEMA,
            "model": GLM53_RAW_CAPTURE_MODEL,
            "checkpoint_revision": self.plan.checkpoint_revision,
            "emitted_config_sha256": self.plan.emitted_config_sha256,
            "rank_bundle_sha256": self.plan.rank_bundle_sha256,
            "candidate_weight_semantics": self.plan.candidate_weight_semantics,
            "output_dtype": self.plan.output_dtype,
            "vocab_size": self.plan.vocab_size,
            "slot_count": self.plan.slot_count,
            "prompt_ids": list(self.plan.prompt_ids),
            "positions": list(self.plan.positions),
            "rows": [row.__dict__ for row in rows],
            "coverage": {
                "row_count": len(rows),
                "expected_row_count": len(self.plan.expected_rows),
                "all_slots": True,
                "full_vocabulary": True,
            },
            "canonical_reference": {
                "bound": self.plan.canonical_reference_id is not None,
                "id": self.plan.canonical_reference_id,
                "semantics": self.plan.canonical_reference_semantics,
            },
            "claims": {
                "raw_capture_complete": True,
                "runtime_permitted": False,
                "correctness_40_of_40": False,
                "performance": False,
            },
        }

    def compare_against_reference(
        self,
        *,
        slot: int,
        prompt_id: str,
        position: int,
        logits: torch.Tensor,
        reference_target: Any,
    ) -> dict[str, Any]:
        """Compare one candidate row with the already-selected target bank.

        ``reference_target`` is intentionally duck-typed so this capture seam
        does not import or construct a reference model.  The target must expose
        the immutable ``reference_id`` and bounded ``load_row`` API supplied by
        ``reference_target.Glm53ReferenceTarget``.  Metrics are evidence only;
        this method never turns a row comparison into a 40/40 claim.
        """

        _require(
            getattr(reference_target, "reference_id", None)
            == self.plan.canonical_reference_id,
            "reference target identity does not match the selected canonical reference",
        )
        _require(
            getattr(reference_target, "semantics", None)
            == self.plan.canonical_reference_semantics,
            "reference target semantics do not match the selected canonical reference",
        )
        _require(
            callable(getattr(reference_target, "load_row", None)),
            "reference target lacks the bounded load_row API",
        )
        _require(
            isinstance(logits, torch.Tensor), "candidate logits must be a torch.Tensor"
        )
        _require(
            tuple(logits.shape) == (self.plan.vocab_size,),
            "candidate comparison row must be full vocabulary",
        )
        _require(
            bool(torch.isfinite(logits).all().item()), "candidate row is non-finite"
        )
        target = reference_target.load_row(
            slot=slot, prompt_id=prompt_id, position=position
        )
        _require(
            tuple(target.shape) == (self.plan.vocab_size,),
            "reference comparison row must be full vocabulary",
        )
        candidate_f32 = logits.detach().to(torch.float32)
        target_f32 = target.detach().to(torch.float32)
        denominator = torch.linalg.vector_norm(
            candidate_f32
        ) * torch.linalg.vector_norm(target_f32)
        _require(
            bool(torch.isfinite(denominator).item()) and denominator.item() > 0,
            "reference comparison cosine denominator is zero or non-finite",
        )
        return {
            "reference_id": reference_target.reference_id,
            "slot": slot,
            "prompt_id": prompt_id,
            "position": position,
            "candidate_argmax": int(torch.argmax(candidate_f32).item()),
            "reference_argmax": int(torch.argmax(target_f32).item()),
            "argmax_equal": bool(
                torch.argmax(candidate_f32).item() == torch.argmax(target_f32).item()
            ),
            "max_abs_error": float(
                torch.max(torch.abs(candidate_f32 - target_f32)).item()
            ),
            "cosine": float(
                torch.dot(candidate_f32, target_f32).div(denominator).item()
            ),
            "correctness_authorized": False,
        }


__all__ = [
    "GLM53_CANDIDATE_WEIGHT_SEMANTICS",
    "GLM53_RAW_CAPTURE_MODEL",
    "GLM53_RAW_CAPTURE_SCHEMA",
    "GLM53_RAW_CAPTURE_VOCAB_SIZE",
    "GLM53_REFERENCE_SEMANTICS",
    "Glm53RawCapture",
    "Glm53RawCaptureError",
    "Glm53RawCapturePlan",
    "Glm53RawCaptureRow",
]
