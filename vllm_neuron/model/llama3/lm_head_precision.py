# SPDX-License-Identifier: Apache-2.0
"""Fail-closed dtype boundary for Llama language-model logits."""

import torch

_SUPPORTED_LM_HEAD_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_lm_head_dtype(
    model_dtype: torch.dtype, requested_dtype: str | None
) -> torch.dtype:
    """Resolve the serialized lm_head dtype without accepting aliases."""
    if requested_dtype is None:
        return model_dtype
    try:
        return _SUPPORTED_LM_HEAD_DTYPES[requested_dtype]
    except KeyError as exc:  # Defensive: NeuronConfig normally rejects this first.
        raise ValueError(
            "lm_head_dtype must be null, 'bfloat16', or 'float32'; "
            f"got {requested_dtype!r}"
        ) from exc


def prepare_lm_head_input(
    hidden_states: torch.Tensor, lm_head_dtype: torch.dtype
) -> torch.Tensor:
    """Cast before projection so precision is not merely relabelled afterward."""
    return hidden_states.to(lm_head_dtype)


def require_lm_head_output_dtype(
    logits: torch.Tensor, lm_head_dtype: torch.dtype
) -> torch.Tensor:
    """Reject a backend/module that did not emit the contracted logits dtype."""
    if logits.dtype != lm_head_dtype:
        raise RuntimeError(
            "lm_head output dtype contract violated: "
            f"expected {lm_head_dtype}, got {logits.dtype}"
        )
    return logits


def require_lm_head_weight_dtype(
    weight: torch.Tensor, lm_head_dtype: torch.dtype
) -> None:
    """Reject checkpoint loading that replaced the typed lm_head parameter."""
    if weight.dtype != lm_head_dtype:
        raise RuntimeError(
            "lm_head weight dtype contract violated after checkpoint load: "
            f"expected {lm_head_dtype}, got {weight.dtype}"
        )
