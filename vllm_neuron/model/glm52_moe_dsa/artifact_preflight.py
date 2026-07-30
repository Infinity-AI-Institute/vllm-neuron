# SPDX-License-Identifier: Apache-2.0
"""Bind GLM-5.2 runtime configuration to its checkpoint metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .static_fp8 import (
    NEURON_LEGACY_E4M3FN_QMAX240,
    OCP_E4M3FN_QMAX448,
    normalize_static_fp8_weight_format,
)

INDEX_FILENAME = "model.safetensors.index.json"
_JSON_LIMITS = {
    "config": 16 * 1024**2,
    "manifest": 64 * 1024**2,
    "index": 256 * 1024**2,
}
_HASH_CHUNK_BYTES = 16 * 1024**2
_VERSIONS = {
    ("fp8", OCP_E4M3FN_QMAX448): "glm52-trn2-static-fp8-v1",
    (
        "bfloat16",
        OCP_E4M3FN_QMAX448,
    ): "glm52-trn2-static-fp8-bf16-shared-v1",
    (
        "fp8",
        NEURON_LEGACY_E4M3FN_QMAX240,
    ): "glm52-trn2-static-fp8-direct-legacy-v1",
    (
        "bfloat16",
        NEURON_LEGACY_E4M3FN_QMAX240,
    ): "glm52-trn2-static-fp8-direct-legacy-bf16-shared-v1",
}
_DIRECT_VERSIONS = frozenset(
    version
    for (shared_dtype, weight_format), version in _VERSIONS.items()
    if weight_format == NEURON_LEGACY_E4M3FN_QMAX240
)


def _read_json(path: Path, *, kind: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"GLM-5.2 checkpoint is missing {kind}: {path}")
    size = path.stat().st_size
    limit = _JSON_LIMITS[kind]
    if size <= 0 or size > limit:
        raise ValueError(
            f"GLM-5.2 {kind} size {size} is outside the bounded limit "
            f"(1..{limit} bytes)"
        )
    try:
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
        if len(payload) > limit:
            raise ValueError(
                f"GLM-5.2 {kind} grew beyond the bounded read limit"
            )
        parsed = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid GLM-5.2 {kind}: {path}") from error
    if not isinstance(parsed, Mapping):
        raise ValueError(f"GLM-5.2 {kind} must contain a JSON object")
    return parsed


def _sha256_file(path: Path, *, maximum_bytes: int) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(
                    "GLM-5.2 index grew beyond the bounded hash limit"
                )
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"GLM-5.2 {label} metadata is missing")
    return value


def _config_markers(
    config: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    str,
    object,
    tuple[object, object, object],
]:
    artifact = _mapping(config.get("glm52_artifact"), label="artifact")
    shared_dtype = str(
        config.get(
            "shared_expert_dtype",
            artifact.get("shared_expert_dtype", "fp8"),
        )
    )
    if shared_dtype not in ("fp8", "bfloat16"):
        raise ValueError("GLM-5.2 shared_expert_dtype is invalid")
    quantization_root = _mapping(
        config.get("quantization_config"),
        label="quantization",
    )
    details = quantization_root.get("quantization")
    if isinstance(details, Mapping):
        quantization = details
    else:
        quantization = quantization_root
    return (
        artifact,
        shared_dtype,
        quantization_root.get("artifact_version"),
        (
            config.get("static_fp8_weight_format"),
            artifact.get("static_fp8_weight_format"),
            quantization.get("weight_format"),
        ),
    )


def _expected_version(shared_dtype: str, weight_format: str) -> str:
    try:
        return _VERSIONS[(shared_dtype, weight_format)]
    except KeyError as error:
        raise ValueError(
            "unsupported GLM-5.2 shared dtype/static-FP8 format combination"
        ) from error


def _safe_declared_file(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"explicit GLM-5.2 artifact must declare {label}")
    relative = Path(value)
    if relative.is_absolute() or relative.name != value:
        raise ValueError(f"GLM-5.2 {label} must be a root-local file name")
    return root / relative


def _reject_direct_metadata(
    *,
    config_artifact: Mapping[str, Any],
    index: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> None:
    candidates: list[object] = [config_artifact.get("artifact_version")]
    index_metadata = index.get("metadata")
    if isinstance(index_metadata, Mapping):
        candidates.extend(
            (
                index_metadata.get("artifact_version"),
                index_metadata.get("static_fp8_weight_format"),
            )
        )
    if manifest is not None:
        candidates.extend(
            (
                manifest.get("artifact_version"),
                manifest.get("static_fp8_weight_format"),
            )
        )
    if any(
        value == NEURON_LEGACY_E4M3FN_QMAX240
        or value in _DIRECT_VERSIONS
        for value in candidates
    ):
        raise ValueError(
            "markerless OCP GLM-5.2 config cannot load direct qmax-240 metadata"
        )


def preflight_checkpoint_artifact(
    checkpoint_path: str | Path,
    *,
    expected_weight_format: str,
    expected_shared_expert_dtype: str,
) -> str:
    """Verify config/manifest/index identity before opening any tensor shard.

    Pre-marker OCP artifacts retain their qualified compatibility behavior.
    Any explicitly marked artifact—and every direct qmax-240 artifact—must
    close over one consistent config, manifest, and safetensors index.
    """

    expected_format = normalize_static_fp8_weight_format(
        expected_weight_format
    )
    if expected_shared_expert_dtype not in ("fp8", "bfloat16"):
        raise ValueError("runtime GLM-5.2 shared_expert_dtype is invalid")
    root = Path(checkpoint_path)
    if not root.is_dir():
        if expected_format == NEURON_LEGACY_E4M3FN_QMAX240:
            raise ValueError(
                "direct qmax-240 GLM-5.2 artifacts require a local attested "
                "checkpoint directory"
            )
        return OCP_E4M3FN_QMAX448

    config = _read_json(root / "config.json", kind="config")
    artifact, shared_dtype, quantization_version, markers = _config_markers(config)
    if shared_dtype != expected_shared_expert_dtype:
        raise ValueError(
            "GLM-5.2 checkpoint shared_expert_dtype does not match runtime"
        )
    index_path = root / INDEX_FILENAME
    index = _read_json(index_path, kind="index")
    any_declared = any(marker is not None for marker in markers)

    if not any_declared:
        if expected_format != OCP_E4M3FN_QMAX448:
            raise ValueError(
                "direct qmax-240 GLM-5.2 runtime requires explicit artifact "
                "format declarations"
            )
        expected_version = _expected_version(shared_dtype, expected_format)
        if artifact.get("artifact_version") != expected_version:
            raise ValueError(
                "markerless OCP GLM-5.2 artifact_version is incompatible"
            )
        if (
            quantization_version is not None
            and quantization_version != expected_version
        ):
            raise ValueError(
                "markerless OCP GLM-5.2 quantization artifact_version is "
                "incompatible"
            )
        if artifact.get("shared_expert_dtype", "fp8") != shared_dtype:
            raise ValueError(
                "markerless OCP GLM-5.2 artifact shared dtype is incompatible"
            )
        manifest = None
        declared_manifest = artifact.get("manifest_file")
        if isinstance(declared_manifest, str):
            manifest_path = _safe_declared_file(
                root,
                declared_manifest,
                label="manifest_file",
            )
            if manifest_path.is_file():
                manifest = _read_json(manifest_path, kind="manifest")
        _reject_direct_metadata(
            config_artifact=artifact,
            index=index,
            manifest=manifest,
        )
        index_metadata = index.get("metadata")
        if isinstance(index_metadata, Mapping):
            index_version = index_metadata.get("artifact_version")
            if index_version is not None and index_version != expected_version:
                raise ValueError(
                    "markerless OCP GLM-5.2 index artifact_version is incompatible"
                )
            index_format = index_metadata.get("static_fp8_weight_format")
            if (
                index_format is not None
                and index_format != OCP_E4M3FN_QMAX448
            ):
                raise ValueError(
                    "markerless OCP GLM-5.2 index declares a different format"
                )
            index_shared = index_metadata.get("shared_expert_dtype")
            if index_shared is not None and index_shared != shared_dtype:
                raise ValueError(
                    "markerless OCP GLM-5.2 index shared dtype is incompatible"
                )
        return OCP_E4M3FN_QMAX448

    if any(marker is None for marker in markers):
        raise ValueError(
            "GLM-5.2 static-FP8 format must be declared in config, artifact, "
            "and quantization metadata"
        )
    weight_format = normalize_static_fp8_weight_format(markers[0])
    if any(marker != weight_format for marker in markers):
        raise ValueError("GLM-5.2 config contains mixed static-FP8 formats")
    if weight_format != expected_format:
        raise ValueError("GLM-5.2 checkpoint format does not match runtime")
    expected_version = _expected_version(shared_dtype, weight_format)
    if artifact.get("artifact_version") != expected_version:
        raise ValueError("GLM-5.2 config artifact_version is incompatible")
    if quantization_version != expected_version:
        raise ValueError(
            "GLM-5.2 quantization artifact_version is incompatible"
        )
    if artifact.get("shared_expert_dtype", "fp8") != shared_dtype:
        raise ValueError("GLM-5.2 artifact shared dtype is incompatible")
    if artifact.get("loader_ready") is not True:
        raise ValueError("GLM-5.2 artifact is not loader_ready")
    if artifact.get("index_closure_status") != "passed":
        raise ValueError("GLM-5.2 artifact index closure has not passed")

    manifest_path = _safe_declared_file(
        root,
        artifact.get("manifest_file"),
        label="manifest_file",
    )
    manifest = _read_json(manifest_path, kind="manifest")
    if manifest.get("artifact_version") != expected_version:
        raise ValueError("GLM-5.2 manifest artifact_version is incompatible")
    if manifest.get("static_fp8_weight_format") != weight_format:
        raise ValueError("GLM-5.2 manifest static-FP8 format is incompatible")
    if manifest.get("shared_expert_dtype") != shared_dtype:
        raise ValueError("GLM-5.2 manifest shared dtype is incompatible")
    manifest_quantization = _mapping(
        manifest.get("quantization"),
        label="manifest quantization",
    )
    if manifest_quantization.get("storage_format") != weight_format:
        raise ValueError(
            "GLM-5.2 manifest quantization format is incompatible"
        )
    loader_validation = _mapping(
        manifest.get("loader_validation"),
        label="manifest loader_validation",
    )
    if loader_validation.get("loader_ready") is not True:
        raise ValueError("GLM-5.2 manifest is not loader_ready")
    if loader_validation.get("required_artifact_version") != expected_version:
        raise ValueError(
            "GLM-5.2 manifest loader artifact_version is incompatible"
        )
    if (
        loader_validation.get("required_static_fp8_weight_format")
        != weight_format
    ):
        raise ValueError("GLM-5.2 manifest loader format is incompatible")
    source = _mapping(manifest.get("source"), label="manifest source")
    index_closure = _mapping(
        source.get("index_closure"),
        label="manifest index closure",
    )
    if index_closure.get("status") != "passed":
        raise ValueError("GLM-5.2 manifest index closure has not passed")

    output = _mapping(manifest.get("output"), label="manifest output")
    if output.get("index_file") != INDEX_FILENAME:
        raise ValueError("GLM-5.2 manifest names an unexpected index file")
    expected_index_sha256 = output.get("index_sha256")
    if (
        not isinstance(expected_index_sha256, str)
        or len(expected_index_sha256) != 64
        or _sha256_file(
            index_path,
            maximum_bytes=_JSON_LIMITS["index"],
        )
        != expected_index_sha256
    ):
        raise ValueError("GLM-5.2 safetensors index SHA-256 is stale or mismatched")

    index_metadata = _mapping(index.get("metadata"), label="index")
    if index_metadata.get("artifact_version") != expected_version:
        raise ValueError("GLM-5.2 index artifact_version is incompatible")
    if index_metadata.get("static_fp8_weight_format") != weight_format:
        raise ValueError("GLM-5.2 index static-FP8 format is incompatible")
    if index_metadata.get("shared_expert_dtype") != shared_dtype:
        raise ValueError("GLM-5.2 index shared dtype is incompatible")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("GLM-5.2 safetensors index has no weight_map")
    return weight_format
