"""Verify the GLM TKG/CTE serialized weight and state handoff on the host.

The two GLM phases are compiled independently.  TKG serializes a
``LayoutTransformation`` weight loader while CTE serializes Neuron's
``_parallel_load`` path.  This module proves that the difference is phase-local
and that both serialized models expose the same hybrid KDA/DSA state-key
schema, shared emitted config, and immutable source contract.

It never loads a Torch model, opens a Neuron device, or authorizes runtime.
Only small TorchScript code members and a streaming scan of the serialized
state-key table are read from each ``model.pt`` ZIP archive.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

PHASE_HANDOFF_SCHEMA = "glm53-phase-handoff-v1"
_STATE_KEY = re.compile(rb"past_key_values\.[0-9]+")
_PHASES = {
    "tkg": ("token_generation_model", "LayoutTransformation"),
    "cte": ("context_encoding_model", "_parallel_load"),
}
_COMMON_SHAPE = {
    "model": "GLM-5.3-Flash",
    "tp": 32,
    "lnc": 2,
    "resident_batch": 1,
    "sequence": 128,
    "max_model_len": 128,
    "blockwise_use_shard_on_intermediate_dynamic_while": True,
    "kv_cache_quant_requested": False,
}
_SOURCE_KEYS = (
    "source_commit",
    "source_tree",
    "nxdi_commit",
    "checkpoint_revision",
)


class Glm53PhaseHandoffError(ValueError):
    """The independently serialized GLM phases cannot be joined exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Glm53PhaseHandoffError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Glm53PhaseHandoffError(f"cannot read JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required(root: Path, relative: str) -> Path:
    path = root / relative
    _require(path.is_file() and not path.is_symlink(), f"missing file: {relative}")
    return path


def _locate(root: Path, suffix: str) -> Path:
    # Original compile roots put cache beside ``artifacts``; the immutable
    # Trn staging bundle nests the same cache under ``artifacts``.  Accept both
    # layouts, but still require exactly one non-symlink match.
    cache_roots = (root / "cache", root / "artifacts" / "cache")
    candidates = sorted(
        path
        for cache_root in cache_roots
        if cache_root.is_dir()
        for path in cache_root.rglob(suffix)
    )
    _require(len(candidates) == 1, f"expected one {suffix} in artifact cache")
    _require(not candidates[0].is_symlink(), f"{suffix} must not be a symlink")
    return candidates[0]


def _scan_serialized_model(path: Path, phase: str) -> dict[str, Any]:
    expected_model, loader_marker = _PHASES[phase]
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise Glm53PhaseHandoffError(f"invalid serialized model: {path}") from exc

    code_names = [
        name
        for name in archive.namelist()
        if name.endswith("/neuronx_distributed/trace/spmd.py")
    ]
    _require(len(code_names) == 1, f"{phase} model has no unique spmd code member")
    code = archive.read(code_names[0])
    _require(expected_model.encode() in code, f"{phase} model tag drift")
    _require(loader_marker.encode() in code, f"{phase} loader marker missing")
    if phase == "tkg":
        _require(
            b"ops.neuron._parallel_load" not in code,
            "TKG unexpectedly uses CTE _parallel_load",
        )
    else:
        _require(
            b"LayoutTransformation" not in code,
            "CTE unexpectedly uses TKG LayoutTransformation",
        )

    # The state-key table lives in data.pkl.  Stream it so the bounded probe
    # never materializes the 40--160 MiB serialized model payload in memory.
    state_keys: set[bytes] = set()
    data_names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
    _require(len(data_names) == 1, f"{phase} model has no unique data.pkl member")
    with archive.open(data_names[0], "r") as stream:
        carry = b""
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            window = carry + chunk
            state_keys.update(_STATE_KEY.findall(window))
            carry = window[-32:]
    _require(state_keys, f"{phase} model has no serialized past_key_values keys")
    ordered_keys = tuple(sorted(key.decode("ascii") for key in state_keys))
    archive.close()
    return {
        "phase": phase,
        "model": expected_model,
        "loader": loader_marker,
        "state_key_count": len(ordered_keys),
        "state_keys_sha256": hashlib.sha256(
            ("\n".join(ordered_keys) + "\n").encode("ascii")
        ).hexdigest(),
    }


def _phase_record(root: Path, phase: str) -> dict[str, Any]:
    shape_path = _required(root, "artifacts/effective-shape.json")
    shape = _load_json(shape_path)
    expected_model, _ = _PHASES[phase]
    expected_shape = dict(_COMMON_SHAPE)
    expected_shape.update(
        {
            "emit_phases": phase.upper(),
            "models_compiled": [expected_model],
        }
    )
    _require(
        {key: shape.get(key) for key in expected_shape} == expected_shape,
        f"{phase} effective shape drift",
    )
    config_path = _required(root, "artifacts/model/neuron_config.json")
    compile_path = _required(root, "artifacts/compile-result.json")
    compile_result = _load_json(compile_path)
    _require(
        compile_result.get("neff_count") == 1
        and compile_result.get("neuron_config_has_float8_e4m3fn") is False
        and compile_result.get("neuron_config_has_bfloat16") is True,
        f"{phase} compile-result contract drift",
    )
    model_path = _required(root, "artifacts/model/model.pt")
    model_scan = _scan_serialized_model(model_path, phase)
    hlo = _locate(root, "model.hlo_module.pb")
    neff = _locate(root, "model.neff")
    return {
        "root": str(root),
        "phase": phase,
        "effective_shape_sha256": _sha256_file(shape_path),
        "emitted_config_sha256": _sha256_file(config_path),
        "compile_result_sha256": _sha256_file(compile_path),
        "model_pt": {
            "relative_path": "artifacts/model/model.pt",
            "bytes": model_path.stat().st_size,
            "sha256": _sha256_file(model_path),
        },
        "hlo_sha256": _sha256_file(hlo),
        "neff_sha256": _sha256_file(neff),
        "neff_bytes": neff.stat().st_size,
        "serialized": model_scan,
    }


def _source_identity(
    tkg_root: Path, cte_root: Path, compose: dict[str, Any]
) -> dict[str, Any]:
    tkg_identity = _load_json(_required(tkg_root, "artifacts/source-identity.json"))
    shared = compose.get("shared_contract")
    _require(isinstance(shared, dict), "compose receipt lacks shared_contract")
    _require(
        {key: tkg_identity.get(key) for key in _SOURCE_KEYS}
        == {key: shared.get(key) for key in _SOURCE_KEYS},
        "TKG source identity disagrees with shared contract",
    )
    cte_identity_path = cte_root / "artifacts/source-identity.json"
    if cte_identity_path.is_file() and not cte_identity_path.is_symlink():
        cte_identity = _load_json(cte_identity_path)
        _require(
            {key: cte_identity.get(key) for key in _SOURCE_KEYS}
            == {key: shared.get(key) for key in _SOURCE_KEYS},
            "CTE source identity disagrees with shared contract",
        )
    launch_path = cte_root / "artifacts/launch-receipt.json"
    if launch_path.is_file() and not launch_path.is_symlink():
        launch = _load_json(launch_path)
        launch_source = launch.get("source")
        _require(isinstance(launch_source, dict), "CTE launch receipt lacks source")
        _require(
            {key: launch_source.get(key) for key in _SOURCE_KEYS}
            == {key: shared.get(key) for key in _SOURCE_KEYS},
            "CTE launch source disagrees with shared contract",
        )
    return {key: shared.get(key) for key in _SOURCE_KEYS}


def _verify_compose_phase(
    compose: dict[str, Any], phase: str, record: dict[str, Any]
) -> None:
    artifacts = compose.get("artifacts")
    _require(isinstance(artifacts, dict), "compose receipt lacks artifacts")
    entry = artifacts.get(phase)
    _require(isinstance(entry, dict), f"compose receipt lacks {phase} artifact")
    for key in (
        "compile_result_sha256",
        "effective_shape_sha256",
        "emitted_config_sha256",
        "hlo_sha256",
        "neff_sha256",
    ):
        _require(entry.get(key) == record[key], f"compose {phase} {key} drift")


def inspect_phase_handoff(
    tkg_artifact_root: str | Path,
    cte_artifact_root: str | Path,
    *,
    compose_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the serialized TKG→CTE handoff and return a non-authorizing receipt."""

    tkg_root = Path(tkg_artifact_root).resolve(strict=True)
    cte_root = Path(cte_artifact_root).resolve(strict=True)
    compose_path = (
        Path(compose_receipt_path).resolve(strict=True)
        if compose_receipt_path is not None
        else cte_root / "artifacts/tkg-cte-compose-receipt.json"
    )
    compose = _load_json(compose_path)
    _require(
        compose.get("schema") == "glm53-tkg-cte-compose-v1", "compose schema drift"
    )
    tkg = _phase_record(tkg_root, "tkg")
    cte = _phase_record(cte_root, "cte")
    source = _source_identity(tkg_root, cte_root, compose)
    _verify_compose_phase(compose, "tkg", tkg)
    _verify_compose_phase(compose, "cte", cte)
    _require(
        tkg["emitted_config_sha256"] == cte["emitted_config_sha256"],
        "TKG/CTE emitted config differs",
    )
    _require(
        tkg["serialized"]["state_key_count"] == cte["serialized"]["state_key_count"]
        and tkg["serialized"]["state_keys_sha256"]
        == cte["serialized"]["state_keys_sha256"],
        "TKG/CTE serialized state-key schema differs",
    )
    _require(
        compose.get("shared_contract", {}).get("dtype") == "bfloat16"
        and compose.get("shared_contract", {}).get("quantization") == "none"
        and compose.get("shared_contract", {}).get("speculation") is False,
        "shared runtime contract is not BF16/no-quant/no-spec",
    )
    _require(
        compose.get("reuse_policy", {}).get("tkg_neff_reused_unchanged") is True
        and compose.get("reuse_policy", {}).get("cte_neff_reused_unchanged") is True,
        "compose receipt does not bind unchanged NEFF reuse",
    )
    return {
        "schema": PHASE_HANDOFF_SCHEMA,
        "source": source,
        "tkg": tkg,
        "cte": cte,
        "handoff": {
            "weight_loader_difference_is_phase_local": True,
            "shared_state_schema": True,
            "shared_emitted_config": True,
            "model_pt_bound_for_both_phases": True,
            "card_launch_authorized": False,
            "runtime_permitted": False,
            "correctness_40_of_40": False,
            "performance": False,
        },
    }


def canonical_bytes(receipt: dict[str, Any]) -> bytes:
    return (json.dumps(receipt, separators=(",", ":"), sort_keys=True) + "\n").encode()


__all__ = [
    "PHASE_HANDOFF_SCHEMA",
    "Glm53PhaseHandoffError",
    "canonical_bytes",
    "inspect_phase_handoff",
]
