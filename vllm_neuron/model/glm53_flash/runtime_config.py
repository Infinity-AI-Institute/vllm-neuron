# SPDX-License-Identifier: Apache-2.0
"""Strict emitted runtime-configuration contract for GLM-5.3-Flash.

This module deliberately supplies no compile-profile defaults.  A caller must
provide every field, then validate the configuration actually emitted by the
runtime against the same contract.  Unknown, missing, duplicate, or silently
dropped fields fail closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

GLM53_RUNTIME_CONFIG_SCHEMA = "glm53-emitted-runtime-config-v2"
GLM53_ARCHITECTURE = "Glm5NextForConditionalGeneration"
GLM53_CHECKPOINT_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"

_HEX40 = re.compile(r"[0-9a-f]{40}")
_SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}")
_DIGEST = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")


class Glm53RuntimeConfigError(ValueError):
    """The requested or emitted runtime configuration is not reviewable."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Glm53RuntimeConfigError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise Glm53RuntimeConfigError(f"{name} must be a positive integer")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise Glm53RuntimeConfigError(
            f"{name} must be a non-empty, whitespace-trimmed string"
        )
    return value


def _strictly_increasing_ints(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise Glm53RuntimeConfigError(f"{name} must be a non-empty JSON array")
    result = tuple(_positive_int(item, f"{name}[]") for item in value)
    if tuple(sorted(set(result))) != result:
        raise Glm53RuntimeConfigError(
            f"{name} must contain unique, strictly increasing integers"
        )
    return result


@dataclass(frozen=True)
class Glm53RuntimeConfig:
    """Requested or emitted compile/runtime profile with no implicit fields."""

    schema: str
    architecture: str
    checkpoint_revision: str
    source_commit: str
    source_tree: str
    runtime_adapter: str
    compiler_image_id: str
    compiler_image_digest: str
    compiler_version: str
    runtime_packages: tuple[tuple[str, str], ...]
    compiler_flags: tuple[str, ...]
    tensor_parallel_degree: int
    logical_neuron_cores: int
    batch_size: int
    max_sequence_length: int
    context_encoding_buckets: tuple[int, ...]
    token_generation_buckets: tuple[int, ...]
    weight_dtype: str
    cache_dtype: str
    runtime_quantization: str
    sampling_mode: str
    output_logits: bool
    speculative_decode: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Glm53RuntimeConfig:
        if not isinstance(value, Mapping):
            raise Glm53RuntimeConfigError("runtime config must be a JSON object")
        expected = {item.name for item in fields(cls)}
        observed = set(value)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise Glm53RuntimeConfigError(
                f"runtime config field-set mismatch: missing={missing}, extra={extra}"
            )

        strings = {
            name: _nonempty_string(value[name], name)
            for name in (
                "schema",
                "architecture",
                "checkpoint_revision",
                "source_commit",
                "source_tree",
                "runtime_adapter",
                "compiler_image_id",
                "compiler_image_digest",
                "compiler_version",
                "weight_dtype",
                "cache_dtype",
                "runtime_quantization",
                "sampling_mode",
            )
        }
        if strings["schema"] != GLM53_RUNTIME_CONFIG_SCHEMA:
            raise Glm53RuntimeConfigError("unsupported GLM-5.3 runtime config schema")
        if strings["architecture"] != GLM53_ARCHITECTURE:
            raise Glm53RuntimeConfigError("wrong GLM-5.3 architecture")
        if strings["checkpoint_revision"] != GLM53_CHECKPOINT_REVISION:
            raise Glm53RuntimeConfigError("wrong GLM-5.3 checkpoint revision")
        for name in ("checkpoint_revision", "source_commit", "source_tree"):
            if _HEX40.fullmatch(strings[name]) is None:
                raise Glm53RuntimeConfigError(f"{name} must be lowercase 40-hex")
        if _SHA256_ID.fullmatch(strings["compiler_image_id"]) is None:
            raise Glm53RuntimeConfigError(
                "compiler_image_id must be an exact sha256 image ID"
            )
        if _DIGEST.fullmatch(strings["compiler_image_digest"]) is None:
            raise Glm53RuntimeConfigError(
                "compiler_image_digest must be an exact repository digest"
            )

        packages_value = value["runtime_packages"]
        if not isinstance(packages_value, Mapping) or not packages_value:
            raise Glm53RuntimeConfigError(
                "runtime_packages must be a non-empty package-to-identity object"
            )
        packages = tuple(
            sorted(
                (
                    _nonempty_string(name, "runtime_packages key"),
                    _nonempty_string(identity, f"runtime_packages[{name!r}]"),
                )
                for name, identity in packages_value.items()
            )
        )

        flags_value = value["compiler_flags"]
        if not isinstance(flags_value, list):
            raise Glm53RuntimeConfigError("compiler_flags must be a JSON array")
        flags = tuple(
            _nonempty_string(flag, "compiler_flags[]") for flag in flags_value
        )
        if len(set(flags)) != len(flags):
            raise Glm53RuntimeConfigError("compiler_flags contains duplicates")

        integer_names = (
            "tensor_parallel_degree",
            "logical_neuron_cores",
            "batch_size",
            "max_sequence_length",
        )
        integers = {name: _positive_int(value[name], name) for name in integer_names}
        if integers["tensor_parallel_degree"] not in (32, 64):
            raise Glm53RuntimeConfigError("GLM-5.3 rank plan requires TP32 or TP64")
        context_buckets = _strictly_increasing_ints(
            value["context_encoding_buckets"], "context_encoding_buckets"
        )
        token_buckets = _strictly_increasing_ints(
            value["token_generation_buckets"], "token_generation_buckets"
        )
        maximum = integers["max_sequence_length"]
        if context_buckets[-1] > maximum or token_buckets[-1] > maximum:
            raise Glm53RuntimeConfigError(
                "compile buckets cannot exceed max_sequence_length"
            )
        if strings["sampling_mode"] != "greedy":
            raise Glm53RuntimeConfigError("formal gate requires greedy sampling")
        output_logits = value["output_logits"]
        if type(output_logits) is not bool:
            raise Glm53RuntimeConfigError("output_logits must be a boolean")
        if integers["tensor_parallel_degree"] == 64 and output_logits is not True:
            raise Glm53RuntimeConfigError(
                "TP64 profile requires output_logits=true for the full-vocabulary gate"
            )
        if type(value["speculative_decode"]) is not bool:
            raise Glm53RuntimeConfigError("speculative_decode must be a boolean")
        if value["speculative_decode"]:
            raise Glm53RuntimeConfigError(
                "speculative decode is outside the GLM-5.3 formal gate"
            )

        return cls(
            **strings,
            runtime_packages=packages,
            compiler_flags=flags,
            **integers,
            context_encoding_buckets=context_buckets,
            token_generation_buckets=token_buckets,
            output_logits=output_logits,
            speculative_decode=False,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the exact JSON-compatible emitted representation."""
        return {
            "schema": self.schema,
            "architecture": self.architecture,
            "checkpoint_revision": self.checkpoint_revision,
            "source_commit": self.source_commit,
            "source_tree": self.source_tree,
            "runtime_adapter": self.runtime_adapter,
            "compiler_image_id": self.compiler_image_id,
            "compiler_image_digest": self.compiler_image_digest,
            "compiler_version": self.compiler_version,
            "runtime_packages": dict(self.runtime_packages),
            "compiler_flags": list(self.compiler_flags),
            "tensor_parallel_degree": self.tensor_parallel_degree,
            "logical_neuron_cores": self.logical_neuron_cores,
            "batch_size": self.batch_size,
            "max_sequence_length": self.max_sequence_length,
            "context_encoding_buckets": list(self.context_encoding_buckets),
            "token_generation_buckets": list(self.token_generation_buckets),
            "weight_dtype": self.weight_dtype,
            "cache_dtype": self.cache_dtype,
            "runtime_quantization": self.runtime_quantization,
            "sampling_mode": self.sampling_mode,
            "output_logits": self.output_logits,
            "speculative_decode": self.speculative_decode,
        }

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                self.to_mapping(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def assert_emitted(self, emitted: Mapping[str, Any]) -> None:
        """Require the runtime-emitted profile to equal the reviewed request."""
        observed = type(self).from_mapping(emitted)
        if observed != self:
            drift = [
                item.name
                for item in fields(self)
                if getattr(observed, item.name) != getattr(self, item.name)
            ]
            raise Glm53RuntimeConfigError(
                f"runtime emitted configuration drifted in fields: {drift}"
            )

    @classmethod
    def load_canonical(cls, payload: bytes) -> Glm53RuntimeConfig:
        """Reload canonical bytes with duplicate-key and byte-identity checks."""
        if not isinstance(payload, bytes):
            raise Glm53RuntimeConfigError("canonical runtime config must be bytes")
        try:
            text = payload.decode("utf-8")
            raw = json.loads(text, object_pairs_hook=_strict_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Glm53RuntimeConfigError(
                "invalid canonical runtime config JSON"
            ) from exc
        config = cls.from_mapping(raw)
        if config.canonical_bytes() != payload:
            raise Glm53RuntimeConfigError(
                "runtime config bytes are not the canonical serialized form"
            )
        return config


__all__ = [
    "GLM53_ARCHITECTURE",
    "GLM53_CHECKPOINT_REVISION",
    "GLM53_RUNTIME_CONFIG_SCHEMA",
    "Glm53RuntimeConfig",
    "Glm53RuntimeConfigError",
]
