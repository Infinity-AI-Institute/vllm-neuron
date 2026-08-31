"""Explicit CPU reference provider for the released GLM-5.3-Flash model.

The provider is intentionally thin: Transformers owns the model and processor
implementation, while this module binds the exact snapshot, preserves the
checkpoint's serialized dtype, and supplies the producer's token-bound,
greedy, full-vocabulary runner.  It does not dequantize weights, substitute
Q4, enable MTP, or use speculative decoding.

The module is lazy-imported so the Neuron package remains usable without the
optional Transformers reference environment.  ``configure`` must be called
before tokenization, loading, or running; the CLI's ``--dry-run`` calls it
after its metadata-only checkpoint preflight and therefore loads processor
metadata but never model weights.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

GLM53_CHECKPOINT_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
GLM53_VOCAB_SIZE = 154_880
GLM53_POSITIONS = tuple(range(10))
GLM53_NATIVE_BLOCK_FP8 = "native-block-fp8"
GLM53_CONVERTED_BF16 = "native-block-fp8-dequantized-bfloat16"
GLM53_PROVIDER_SCHEMA = "glm53-transformers-original-provider-v1"
GLM53_TRANSFORMERS_MINIMUM = "5.14.1"

_processor: Any | None = None
_checkpoint_dir: Path | None = None
_semantics: str | None = None


class Glm53OriginalProviderError(ValueError):
    """Raised when the exact reference-provider contract is not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Glm53OriginalProviderError(message)


def _torch_xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    is_available = getattr(xpu, "is_available", None)
    return bool(callable(is_available) and is_available())


def _root(path: str | Path) -> Path:
    root = Path(path).resolve(strict=True)
    _require(root.is_dir(), "checkpoint path must be a directory")
    _require(
        root.name == GLM53_CHECKPOINT_REVISION,
        f"checkpoint directory must name pinned revision {GLM53_CHECKPOINT_REVISION}",
    )
    return root


def _require_config_identity(root: Path) -> None:
    """Require the repo's metadata preflight when called outside the CLI."""

    try:
        from .checkpoint_converter import preflight_checkpoint_dir
    except ImportError as exc:  # pragma: no cover - direct standalone import
        raise Glm53OriginalProviderError(
            "repository checkpoint preflight is unavailable"
        ) from exc
    preflight_checkpoint_dir(root)


def configure(checkpoint_dir: str | Path, semantics: str) -> dict[str, str]:
    """Bind processor metadata and explicit serialized precision semantics.

    Native block-FP8 and explicitly requested BF16 conversion are accepted.
    The latter is never selected implicitly: the caller must pass the
    ``native-block-fp8-dequantized-bfloat16`` label.
    """

    global _checkpoint_dir, _processor, _semantics
    _require(
        semantics in (GLM53_NATIVE_BLOCK_FP8, GLM53_CONVERTED_BF16),
        "this provider requires native-block-fp8 or explicitly declared "
        "native-block-fp8-dequantized-bfloat16 semantics",
    )
    root = _root(checkpoint_dir)
    _require_config_identity(root)
    try:
        from transformers import AutoProcessor
    except ImportError as exc:  # pragma: no cover - exercised in deployment env
        raise Glm53OriginalProviderError(
            "Transformers >= 5.14.1 is required for Glm5NextProcessor"
        ) from exc
    try:
        processor = AutoProcessor.from_pretrained(
            str(root),
            local_files_only=True,
            revision=GLM53_CHECKPOINT_REVISION,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise Glm53OriginalProviderError(
            "pinned GLM-5.3 processor/tokenizer metadata could not be loaded"
        ) from exc
    processor_name = type(processor).__name__
    _require(
        processor_name == "Glm5NextProcessor",
        f"unexpected processor class: {processor_name}",
    )
    _checkpoint_dir = root
    _processor = processor
    _semantics = semantics
    return {
        "provider_schema": GLM53_PROVIDER_SCHEMA,
        "checkpoint_revision": GLM53_CHECKPOINT_REVISION,
        "processor_class": processor_name,
        "semantics": semantics,
    }


def _require_configured() -> tuple[Path, Any, str]:
    _require(
        _checkpoint_dir is not None and _processor is not None and _semantics,
        "provider.configure(checkpoint_dir, semantics) must run first",
    )
    return _checkpoint_dir, _processor, _semantics  # type: ignore[return-value]


def tokenize(prompt: str) -> tuple[int, ...]:
    """Apply the official GLM chat template and return bound input IDs."""

    _, processor, _ = _require_configured()
    _require(isinstance(prompt, str) and prompt, "prompt must be non-empty text")
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    try:
        encoded = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as exc:
        raise Glm53OriginalProviderError(
            "official GLM-5.3 chat-template tokenization failed"
        ) from exc
    input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else None
    _require(isinstance(input_ids, torch.Tensor), "processor did not return input_ids")
    _require(
        input_ids.ndim == 2 and input_ids.shape[0] == 1 and input_ids.shape[1] > 0,
        f"processor input_ids must have shape [1, sequence], got {tuple(input_ids.shape)}",
    )
    _require(
        input_ids.dtype in (torch.int64, torch.int32),
        f"processor input_ids must be integer, got {input_ids.dtype}",
    )
    ids = tuple(int(value) for value in input_ids[0].tolist())
    _require(
        all(0 <= value < GLM53_VOCAB_SIZE for value in ids),
        "token ID outside vocabulary",
    )
    return ids


def load(checkpoint_dir: str | Path) -> Any:
    """Load the exact upstream class without overriding serialized dtypes."""

    root, _, semantics = _require_configured()
    requested = _root(checkpoint_dir)
    _require(requested == root, "loader path differs from configured checkpoint")
    if semantics == GLM53_NATIVE_BLOCK_FP8:
        _require(
            torch.cuda.is_available() or _torch_xpu_available(),
            "native block-FP8 Transformers execution requires CUDA or XPU; "
            "CPU execution must explicitly request converted-BF16 semantics",
        )
    try:
        from transformers import AutoModelForMultimodalLM
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextForConditionalGeneration,
        )
    except ImportError as exc:  # pragma: no cover - deployment environment
        raise Glm53OriginalProviderError(
            "Transformers >= 5.14.1 with Glm5Next is required"
        ) from exc
    try:
        kwargs: dict[str, Any] = {
            "revision": GLM53_CHECKPOINT_REVISION,
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": "auto",
        }
        if semantics == GLM53_CONVERTED_BF16:
            from transformers.utils.quantization_config import FineGrainedFP8Config

            kwargs["quantization_config"] = FineGrainedFP8Config(
                activation_scheme="dynamic",
                weight_block_size=(128, 128),
                dequantize=True,
            )
        model = AutoModelForMultimodalLM.from_pretrained(
            str(root),
            **kwargs,
        )
    except Exception as exc:
        raise Glm53OriginalProviderError(
            "pinned GLM-5.3 model weights could not be loaded with serialized dtypes"
        ) from exc
    _require(
        isinstance(model, Glm5NextForConditionalGeneration),
        "AutoModelForMultimodalLM did not return Glm5NextForConditionalGeneration",
    )
    model.eval()
    return model


def run(
    model: Any,
    prompt_id: str,
    positions: Sequence[int],
    input_ids: Sequence[int],
) -> Iterator[torch.Tensor]:
    """Yield ten own-greedy FP32 rows after native model execution.

    FP32 here is only the serialized comparison row after the model's native
    forward/projection.  It is not a claim that the model executed in FP32.
    ``use_cache=False`` is intentional for a small, deterministic reference
    harness and does not introduce MTP or speculative decoding.
    """

    _, _, semantics = _require_configured()
    _require(
        semantics in (GLM53_NATIVE_BLOCK_FP8, GLM53_CONVERTED_BF16),
        "precision binding was lost",
    )
    _require(
        tuple(positions) == GLM53_POSITIONS, "runner requires positions 0 through 9"
    )
    bound = tokenize(prompt_id)
    supplied = tuple(int(value) for value in input_ids)
    _require(
        supplied == bound, "runner input IDs do not match official tokenizer binding"
    )
    tokens = torch.tensor([supplied], dtype=torch.long)
    with torch.inference_mode():
        for _position in GLM53_POSITIONS:
            outputs = model(
                input_ids=tokens,
                attention_mask=torch.ones_like(tokens),
                use_cache=False,
                return_dict=True,
            )
            logits = getattr(outputs, "logits", None)
            _require(
                isinstance(logits, torch.Tensor)
                and logits.ndim == 3
                and logits.shape[0] == 1
                and logits.shape[-1] == GLM53_VOCAB_SIZE,
                "GLM-5.3 model did not return [1, sequence, 154880] logits",
            )
            row = logits[0, -1].detach().cpu().to(torch.float32).contiguous()
            _require(
                bool(torch.isfinite(row).all().item()), "model logits are non-finite"
            )
            yield row
            next_token = torch.argmax(logits[0, -1], dim=-1).to(torch.long)
            tokens = torch.cat((tokens, next_token.reshape(1, 1)), dim=1)


__all__ = [
    "GLM53_CHECKPOINT_REVISION",
    "GLM53_CONVERTED_BF16",
    "GLM53_NATIVE_BLOCK_FP8",
    "GLM53_POSITIONS",
    "GLM53_PROVIDER_SCHEMA",
    "GLM53_TRANSFORMERS_MINIMUM",
    "GLM53_VOCAB_SIZE",
    "Glm53OriginalProviderError",
    "configure",
    "load",
    "run",
    "tokenize",
]
