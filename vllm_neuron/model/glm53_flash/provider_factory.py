# SPDX-License-Identifier: Apache-2.0
"""Fail-closed provider-registration scaffold for future GLM-5.3 TP64 service.

This module intentionally contains no execution bridge.  It authenticates a
paired CTE/TKG package and the explicit B1 state-slot/reset contract, then
refuses model construction until a separately reviewed bridge exists.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SERVICE_SCHEMA = "glm53-tp64-service-package-v1"
PHASE_SCHEMA = "glm53-tp64-compiled-phase-v1"
STATE_SCHEMA = "glm53-hybrid-state-slot-v1"
ARCHITECTURE = "Glm5NextForConditionalGeneration"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMAGE = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")


class Glm53ProviderAdmissionError(ValueError):
    """The future service package is not exactly admissible."""


class Glm53ProviderExecutionBridgeUnavailable(RuntimeError):
    """Admission passed, but no reviewed runtime execution bridge exists."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Glm53ProviderAdmissionError(message)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise Glm53ProviderAdmissionError(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None,
             f"{name} must be lowercase SHA-256")
    return value


@dataclass(frozen=True)
class Glm53PhaseAdmission:
    phase: str
    manifest_path: str
    manifest_sha256: str
    artifact_manifest_sha256: str


@dataclass(frozen=True)
class Glm53ServicePackageAdmission:
    package_path: str
    package_sha256: str
    runtime_config_sha256: str
    rank_bundle_sha256: str
    state_abi_sha256: str
    phases: tuple[Glm53PhaseAdmission, Glm53PhaseAdmission]

    @classmethod
    def load(cls, package_path: str | Path) -> "Glm53ServicePackageAdmission":
        unresolved = Path(package_path)
        _require(unresolved.is_file() and not unresolved.is_symlink(),
                 "service package must be a regular non-symlink file")
        path = unresolved.resolve(strict=True)
        root = path.parent
        package = _load_json(path)
        _require(set(package) == {
            "schema", "architecture", "service_ready", "topology", "workload",
            "runtime_config_sha256", "rank_bundle_sha256", "state_contract", "phases",
        }, "service package field-set mismatch")
        _require(package["schema"] == SERVICE_SCHEMA, "service package schema mismatch")
        _require(package["architecture"] == ARCHITECTURE, "service architecture mismatch")
        _require(package["service_ready"] is False,
                 "scaffold accepts only explicitly not-ready packages")
        _require(package["topology"] == {
            "tensor_parallel_degree": 64, "logical_neuron_cores": 2,
            "cards": 16, "physical_neuron_cores": 64, "rank_count": 64,
        }, "service topology must be one TP64/LNC2/all-16 engine")
        _require(package["workload"] == {
            "batch_size": 1, "context_encoding_tokens": 2048,
            "total_context_capacity": 2560, "token_generation_step_tokens": 1,
        }, "service workload contract mismatch")
        runtime_config_sha = _digest(package["runtime_config_sha256"], "runtime config")
        rank_bundle_sha = _digest(package["rank_bundle_sha256"], "rank bundle")

        state = package["state_contract"]
        _require(state == {
            "schema": STATE_SCHEMA,
            "owner": "shared_device_resident_cte_tkg",
            "slots": 1,
            "slot_input": {"name": "state_slot", "dtype": "int32", "shape": [1]},
            "reset_input": {"name": "state_reset", "dtype": "bool", "shape": [1]},
            "cte_initial_reset_required": True,
            "tkg_reset_forbidden": True,
            "finish_invalidates_slot": True,
            "reuse_requires_reset": True,
            "preemption_supported": False,
            "prefix_caching_supported": False,
            "async_scheduling_supported": False,
        }, "explicit B1 hybrid-state slot/reset contract mismatch")
        state_abi_sha = hashlib.sha256(
            json.dumps(state, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()

        phases = package["phases"]
        _require(isinstance(phases, dict) and set(phases) == {"cte", "tkg"},
                 "exactly paired CTE and TKG phase manifests are required")
        admitted: list[Glm53PhaseAdmission] = []
        for key, expected_phase in (("cte", "CTE"), ("tkg", "TKG")):
            reference = phases[key]
            _require(isinstance(reference, dict) and set(reference) == {"path", "sha256"},
                     f"{expected_phase} phase reference field-set mismatch")
            relative = Path(reference["path"])
            _require(not relative.is_absolute() and len(relative.parts) == 1
                     and relative.name == reference["path"],
                     f"{expected_phase} manifest must be root-local")
            phase_path = root / relative
            _require(phase_path.is_file() and not phase_path.is_symlink(),
                     f"{expected_phase} manifest missing or symlinked")
            observed_sha = _sha256_file(phase_path)
            _require(_digest(reference["sha256"], f"{expected_phase} manifest") == observed_sha,
                     f"{expected_phase} manifest hash mismatch")
            phase = _load_json(phase_path)
            _require(set(phase) == {
                "schema", "phase", "tensor_parallel_degree", "logical_neuron_cores",
                "rank_count", "runtime_config_sha256", "rank_bundle_sha256",
                "state_abi_sha256", "artifact_manifest_sha256", "compiler_image_digest",
            }, f"{expected_phase} manifest field-set mismatch")
            _require(phase["schema"] == PHASE_SCHEMA and phase["phase"] == expected_phase,
                     f"{expected_phase} phase identity mismatch")
            _require(phase["tensor_parallel_degree"] == 64
                     and phase["logical_neuron_cores"] == 2
                     and phase["rank_count"] == 64,
                     f"{expected_phase} refuses TP32/r5 or incomplete ranks")
            _require(phase["runtime_config_sha256"] == runtime_config_sha,
                     f"{expected_phase} runtime config drift")
            _require(phase["rank_bundle_sha256"] == rank_bundle_sha,
                     f"{expected_phase} rank bundle drift")
            _require(phase["state_abi_sha256"] == state_abi_sha,
                     f"{expected_phase} state ABI drift")
            artifact_sha = _digest(phase["artifact_manifest_sha256"],
                                   f"{expected_phase} artifact manifest")
            _require(isinstance(phase["compiler_image_digest"], str)
                     and _IMAGE.fullmatch(phase["compiler_image_digest"]) is not None,
                     f"{expected_phase} compiler image digest invalid")
            admitted.append(Glm53PhaseAdmission(
                phase=expected_phase, manifest_path=relative.name,
                manifest_sha256=observed_sha, artifact_manifest_sha256=artifact_sha,
            ))
        return cls(
            package_path=str(path), package_sha256=_sha256_file(path),
            runtime_config_sha256=runtime_config_sha,
            rank_bundle_sha256=rank_bundle_sha, state_abi_sha256=state_abi_sha,
            phases=(admitted[0], admitted[1]),
        )


class Glm53FlashProviderForCausalLM:
    """Registry-facing admission scaffold; deliberately non-executable."""

    @classmethod
    def from_configs(cls, hf_config: Any, neuron_config: Any) -> Any:
        _require(neuron_config is not None, "explicit neuron_config required")
        architectures = getattr(hf_config, "architectures", None)
        _require(isinstance(architectures, (list, tuple)) and ARCHITECTURE in architectures,
                 "HF architecture does not identify GLM-5.3-Flash")
        _require(getattr(neuron_config, "tp_degree", None) == 64,
                 "GLM-5.3 service refuses TP32/r5")
        _require(getattr(neuron_config, "logical_nc_config", None) == 2,
                 "GLM-5.3 service requires LNC2")
        _require(getattr(neuron_config, "ctx_batch_size", None) == 1
                 and getattr(neuron_config, "tkg_batch_size", None) == 1,
                 "GLM-5.3 scaffold is B1 only")
        _require(getattr(neuron_config, "is_prefix_caching", False) is False,
                 "prefix caching is not state-qualified")
        _require(getattr(neuron_config, "async_mode", False) is False,
                 "async scheduling is not state-qualified")
        package_path = getattr(neuron_config, "glm53_service_package_path", None)
        _require(isinstance(package_path, str) and package_path,
                 "sealed GLM-5.3 service package path required")
        expected_package_sha = getattr(
            neuron_config, "glm53_service_package_sha256", None
        )
        _digest(expected_package_sha, "externally pinned service package")
        admission = Glm53ServicePackageAdmission.load(package_path)
        _require(admission.package_sha256 == expected_package_sha,
                 "externally pinned service package hash mismatch")
        raise Glm53ProviderExecutionBridgeUnavailable(
            "TP64 CTE/TKG package and state contract admitted at "
            f"{admission.package_sha256}, but provider execution bridge ABI is unresolved; "
            "refusing model construction rather than falling back to the CPU oracle"
        )


__all__ = [
    "Glm53FlashProviderForCausalLM", "Glm53ProviderAdmissionError",
    "Glm53ProviderExecutionBridgeUnavailable", "Glm53ServicePackageAdmission",
]
