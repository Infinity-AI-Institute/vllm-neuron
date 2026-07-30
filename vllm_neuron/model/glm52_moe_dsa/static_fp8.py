# SPDX-License-Identifier: Apache-2.0
"""Explicit GLM-5.2 static-FP8 on-disk weight-format contract."""

from __future__ import annotations

from typing import Any

import torch

OCP_E4M3FN_QMAX448 = "ocp_e4m3fn_qmax448"
NEURON_LEGACY_E4M3FN_QMAX240 = "neuron_legacy_e4m3fn_qmax240"
STATIC_FP8_WEIGHT_FORMATS = frozenset(
    {
        OCP_E4M3FN_QMAX448,
        NEURON_LEGACY_E4M3FN_QMAX240,
    }
)
OCP_E4M3_MAX = 448.0
NEURON_LEGACY_E4M3_MAX = 240.0
WEIGHT_DOWNSCALE = NEURON_LEGACY_E4M3_MAX / OCP_E4M3_MAX
SCALE_COMPENSATION = OCP_E4M3_MAX / NEURON_LEGACY_E4M3_MAX
DIRECT_RANGE_CHECK_CHUNK_ELEMENTS = 16 * 1024**2


def normalize_static_fp8_weight_format(value: Any) -> str:
    """Validate a declared format; ``None`` preserves the original OCP ABI."""

    if value is None:
        return OCP_E4M3FN_QMAX448
    if not isinstance(value, str) or value not in STATIC_FP8_WEIGHT_FORMATS:
        raise ValueError(
            "static_fp8_weight_format must be one of "
            f"{sorted(STATIC_FP8_WEIGHT_FORMATS)!r}, got {value!r}"
        )
    return value


def is_direct_neuron_legacy_format(value: Any) -> bool:
    return (
        normalize_static_fp8_weight_format(value)
        == NEURON_LEGACY_E4M3FN_QMAX240
    )


def static_fp8_qmax(value: Any) -> float:
    return (
        NEURON_LEGACY_E4M3_MAX
        if is_direct_neuron_legacy_format(value)
        else OCP_E4M3_MAX
    )


def static_fp8_scale_multiplier(value: Any) -> float:
    return 1.0 if is_direct_neuron_legacy_format(value) else SCALE_COMPENSATION


def static_fp8_weight_multiplier(value: Any) -> float:
    return 1.0 if is_direct_neuron_legacy_format(value) else WEIGHT_DOWNSCALE


def prepare_static_fp8_weight(
    weight: torch.Tensor,
    weight_format: Any,
) -> torch.Tensor:
    """Prepare a checkpoint FP8 tensor for the qmax-240 Neuron kernel.

    Direct-legacy tensors are returned value- and byte-preserving after a
    fail-closed range check. Original OCP tensors retain the qualified second
    FP8 rounding and paired scale compensation.
    """

    normalized = normalize_static_fp8_weight_format(weight_format)
    if normalized == NEURON_LEGACY_E4M3FN_QMAX240:
        if weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "direct GLM-5.2 static-FP8 weights must use "
                f"torch.float8_e4m3fn, got {weight.dtype}"
            )
        if weight.numel():
            if weight.ndim == 0:
                chunks = (weight.reshape(1),)
            else:
                trailing_elements = max(weight[0].numel(), 1)
                rows_per_chunk = max(
                    DIRECT_RANGE_CHECK_CHUNK_ELEMENTS // trailing_elements,
                    1,
                )
                chunks = (
                    weight[start : start + rows_per_chunk]
                    for start in range(0, weight.shape[0], rows_per_chunk)
                )
            for chunk in chunks:
                fp32_chunk = chunk.to(torch.float32)
                if not bool(torch.isfinite(fp32_chunk).all()):
                    raise ValueError(
                        "direct Neuron-legacy artifact contains a non-finite "
                        "FP8 value"
                    )
                if bool(
                    (fp32_chunk.abs() > NEURON_LEGACY_E4M3_MAX).any()
                ):
                    raise ValueError(
                        "direct Neuron-legacy artifact contains an FP8 value "
                        "outside the declared qmax-240 range"
                    )
        return weight.contiguous()
    return (
        (weight.to(torch.float32) * WEIGHT_DOWNSCALE)
        .clamp(-NEURON_LEGACY_E4M3_MAX, NEURON_LEGACY_E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )


def static_fp8_manifest_contract(value: Any) -> dict[str, Any]:
    normalized = normalize_static_fp8_weight_format(value)
    direct = normalized == NEURON_LEGACY_E4M3FN_QMAX240
    return {
        "storage_format": normalized,
        "format": (
            "Neuron legacy E4M3FN" if direct else "OCP E4M3FN"
        ),
        "qmax": NEURON_LEGACY_E4M3_MAX if direct else OCP_E4M3_MAX,
        "loader_compensation": {
            "weight_multiplier": 1.0 if direct else WEIGHT_DOWNSCALE,
            "scale_multiplier": 1.0 if direct else SCALE_COMPENSATION,
            "neuron_kernel_qmax": NEURON_LEGACY_E4M3_MAX,
        },
    }
