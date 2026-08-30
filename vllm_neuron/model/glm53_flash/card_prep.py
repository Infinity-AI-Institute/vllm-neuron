# SPDX-License-Identifier: Apache-2.0
"""Host-only preparation receipt for a GLM-5.3 TKG artifact.

The TKG NEFF is deliberately treated as immutable.  This module only inspects
the emitted graph/config and the pinned checkpoint headers, then reports the
exact BF16 rank payload that the existing streaming rank plan would emit.  It
does not compile, load Neuron devices, or write checkpoint shards.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GLM53_CARD_PREP_SCHEMA = "glm53-tkg-card-prep-v1"
GLM53_TP = 32
GLM53_LNC = 2
GLM53_BATCH = 1
GLM53_SEQUENCE = 128
GLM53_BUCKET = 128


class Glm53CardPreparationError(ValueError):
    """The TKG artifact or its card-preparation inputs are not exact."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Glm53CardPreparationError(message)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Glm53CardPreparationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise Glm53CardPreparationError(f"cannot read JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_file(root: Path, relative: str) -> Path:
    path = root / relative
    _require(path.is_file() and not path.is_symlink(), f"missing artifact file: {relative}")
    return path


@dataclass(frozen=True)
class Glm53CheckpointMemory:
    """Metadata-only source and emitted-rank memory accounting."""

    source_shard_count: int
    source_tensor_count: int
    source_indexed_payload_bytes: int
    source_payload_bytes_loaded_during_audit: int
    rank_tensor_bytes: tuple[int, ...]
    rank_bfloat16_bytes: tuple[int, ...]
    rank_non_bfloat16_bytes: tuple[int, ...]

    @property
    def rank_count(self) -> int:
        return len(self.rank_tensor_bytes)

    @property
    def rank_tensor_bytes_min(self) -> int:
        return min(self.rank_tensor_bytes)

    @property
    def rank_tensor_bytes_max(self) -> int:
        return max(self.rank_tensor_bytes)

    @property
    def emitted_bfloat16_bytes_total(self) -> int:
        return sum(self.rank_bfloat16_bytes)

    @property
    def emitted_non_bfloat16_bytes_total(self) -> int:
        return sum(self.rank_non_bfloat16_bytes)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "source": {
                "shard_count": self.source_shard_count,
                "tensor_count": self.source_tensor_count,
                "indexed_payload_bytes": self.source_indexed_payload_bytes,
                "payload_bytes_loaded_during_audit": (
                    self.source_payload_bytes_loaded_during_audit
                ),
            },
            "emitted_rank_payload": {
                "rank_count": self.rank_count,
                "tensor_bytes_min": self.rank_tensor_bytes_min,
                "tensor_bytes_max": self.rank_tensor_bytes_max,
                "tensor_bytes_by_rank": list(self.rank_tensor_bytes),
                "bfloat16_bytes_by_rank": list(self.rank_bfloat16_bytes),
                "non_bfloat16_bytes_by_rank": list(self.rank_non_bfloat16_bytes),
                "bfloat16_bytes_total": self.emitted_bfloat16_bytes_total,
                "non_bfloat16_bytes_total": self.emitted_non_bfloat16_bytes_total,
            },
        }


@dataclass(frozen=True)
class Glm53CardPreparation:
    """Fail-closed, non-authorizing card-preparation receipt."""

    artifact_root: str
    effective_shape_sha256: str
    compile_result_sha256: str
    hlo_path: str
    hlo_sha256: str
    neff_path: str
    neff_sha256: str
    emitted_config_sha256: str
    emit_phases: str
    models_compiled: tuple[str, ...]
    cte_artifact_present: bool
    rank_bundle_present: bool
    fresh_prompt_ready: bool
    continuation_tkg_ready: bool
    blockers: tuple[str, ...]
    memory: Glm53CheckpointMemory | None

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": GLM53_CARD_PREP_SCHEMA,
            "artifact": {
                "root": self.artifact_root,
                "effective_shape_sha256": self.effective_shape_sha256,
                "compile_result_sha256": self.compile_result_sha256,
                "hlo": self.hlo_path,
                "hlo_sha256": self.hlo_sha256,
                "neff": self.neff_path,
                "neff_sha256": self.neff_sha256,
                "emitted_config_sha256": self.emitted_config_sha256,
            },
            "graph": {
                "emit_phases": self.emit_phases,
                "models_compiled": list(self.models_compiled),
                "cte_artifact_present": self.cte_artifact_present,
                "rank_bundle_present": self.rank_bundle_present,
            },
            "readiness": {
                "fresh_prompt_ready": self.fresh_prompt_ready,
                "continuation_tkg_ready": self.continuation_tkg_ready,
                "card_launch_authorized": False,
            },
            "blockers": list(self.blockers),
            "memory": self.memory.to_mapping() if self.memory else None,
        }

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(self.to_mapping(), separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _locate_neff(root: Path, compile_result: dict[str, Any]) -> Path:
    neffs = compile_result.get("neffs")
    _require(isinstance(neffs, list) and len(neffs) == 1, "artifact must contain exactly one NEFF")
    candidates = sorted(path for path in (root / "cache").rglob("model.neff"))
    _require(len(candidates) == 1, "artifact cache must contain exactly one model.neff")
    listed = neffs[0]
    _require(isinstance(listed, dict), "compile-result NEFF row must be an object")
    _require(listed.get("bytes") == candidates[0].stat().st_size, "NEFF size drift")
    return candidates[0]


def _locate_hlo(root: Path) -> Path:
    candidates = sorted(path for path in (root / "cache").rglob("model.hlo_module.pb"))
    _require(len(candidates) == 1, "artifact cache must contain exactly one retained HLO")
    return candidates[0]


def _inspect_emitted_config(root: Path) -> tuple[dict[str, Any], str]:
    path = _required_file(root, "artifacts/model/neuron_config.json")
    config = _load_json(path)
    emitted = config.get("neuron_config")
    _require(isinstance(emitted, dict), "emitted config has no neuron_config object")
    exact = {
        "tp_degree": GLM53_TP,
        "world_size": GLM53_TP,
        "logical_nc_config": GLM53_LNC,
        "batch_size": GLM53_BATCH,
        "ctx_batch_size": GLM53_BATCH,
        "tkg_batch_size": GLM53_BATCH,
        "buckets": [GLM53_BUCKET],
        "context_encoding_buckets": None,
        "token_generation_buckets": None,
        "seq_len": GLM53_SEQUENCE,
        "max_context_length": GLM53_SEQUENCE,
        "torch_dtype": "bfloat16",
        "kv_cache_quant": False,
        "quantized": False,
        "enable_eagle_speculation": False,
        "enable_fused_speculation": False,
    }
    for key, value in exact.items():
        _require(emitted.get(key) == value, f"emitted config drift: {key}")
    return emitted, _sha256_file(path)


def measure_checkpoint_memory(
    checkpoint_dir: str | Path,
    *,
    rank_plan_builder: Callable[..., Any] | None = None,
    reader_factory: Callable[[str | Path], Any] | None = None,
) -> Glm53CheckpointMemory:
    """Audit source headers and calculate exact BF16 TP32 output bytes.

    ``IndexedTensorReader`` audits SafeTensors headers only.  The rank plans
    describe emitted tensor shapes/dtypes without materialising a checkpoint.
    Optional factories make the invariant easy to test without production
    dependencies or payload files.
    """

    if rank_plan_builder is None or reader_factory is None:
        from .rank_plan import build_glm53_rank_plan
        from .streaming_rank_writer import IndexedTensorReader

        rank_plan_builder = rank_plan_builder or build_glm53_rank_plan
        reader_factory = reader_factory or IndexedTensorReader
    reader = reader_factory(checkpoint_dir)
    audit = reader.audit_report
    plans = tuple(
        rank_plan_builder(
            checkpoint_dir, rank=rank, tp_degree=GLM53_TP, max_chunk_bytes=64 * 1024 * 1024
        )
        for rank in range(GLM53_TP)
    )
    tensor_bytes: list[int] = []
    bfloat16_bytes: list[int] = []
    non_bfloat16_bytes: list[int] = []
    for plan in plans:
        total = 0
        bf16 = 0
        for spec in plan.inventory.tensors:
            total += spec.nbytes
            if str(spec.dtype) == "torch.bfloat16":
                bf16 += spec.nbytes
        tensor_bytes.append(total)
        bfloat16_bytes.append(bf16)
        non_bfloat16_bytes.append(total - bf16)
    return Glm53CheckpointMemory(
        source_shard_count=audit.shard_count,
        source_tensor_count=audit.tensor_count,
        source_indexed_payload_bytes=audit.indexed_payload_bytes,
        source_payload_bytes_loaded_during_audit=audit.payload_bytes_loaded_during_audit,
        rank_tensor_bytes=tuple(tensor_bytes),
        rank_bfloat16_bytes=tuple(bfloat16_bytes),
        rank_non_bfloat16_bytes=tuple(non_bfloat16_bytes),
    )


def inspect_tkg_artifact(
    artifact_root: str | Path,
    *,
    checkpoint_dir: str | Path | None = None,
    rank_dir: str | Path | None = None,
    cte_artifact_root: str | Path | None = None,
) -> Glm53CardPreparation:
    """Inspect a TKG artifact and return a non-authorizing preparation receipt."""

    root = Path(artifact_root).resolve(strict=True)
    effective_shape_path = _required_file(root, "artifacts/effective-shape.json")
    shape = _load_json(effective_shape_path)
    _require(shape.get("tp") == GLM53_TP, "effective shape TP drift")
    _require(shape.get("lnc") == GLM53_LNC, "effective shape LNC drift")
    _require(shape.get("resident_batch") == GLM53_BATCH, "effective shape batch drift")
    _require(shape.get("sequence") == GLM53_SEQUENCE, "effective shape sequence drift")
    _require(shape.get("emit_phases") == "TKG", "artifact is not the qualified TKG profile")
    models = shape.get("models_compiled")
    _require(models == ["token_generation_model"], "artifact is not TKG-only")
    _, emitted_sha = _inspect_emitted_config(root)
    compile_result_path = _required_file(root, "artifacts/compile-result.json")
    compile_result = _load_json(compile_result_path)
    hlo = _locate_hlo(root)
    neff = _locate_neff(root, compile_result)

    cte_present = False
    if cte_artifact_root is not None:
        cte_root = Path(cte_artifact_root).resolve(strict=True)
        cte_shape = _load_json(_required_file(cte_root, "artifacts/effective-shape.json"))
        cte_present = "context_encoding_model" in cte_shape.get("models_compiled", [])

    rank_present = False
    if rank_dir is not None:
        rank_root = Path(rank_dir).resolve(strict=True)
        rank_present = all(
            (rank_root / f"tp{rank}_sharded_checkpoint.safetensors").is_file()
            and (rank_root / f"tp{rank}_sharded_checkpoint.safetensors.manifest.json").is_file()
            for rank in range(GLM53_TP)
        )

    blockers: list[str] = []
    if not cte_present:
        blockers.append("fresh prompt requires a context_encoding_model artifact")
    if not rank_present:
        blockers.append("card load requires all 32 TP32 sharded BF16 rank files and manifests")
    memory = measure_checkpoint_memory(checkpoint_dir) if checkpoint_dir is not None else None
    fresh_ready = cte_present and rank_present
    continuation_ready = rank_present
    return Glm53CardPreparation(
        artifact_root=str(root),
        effective_shape_sha256=_sha256_file(effective_shape_path),
        compile_result_sha256=_sha256_file(compile_result_path),
        hlo_path=str(hlo),
        hlo_sha256=_sha256_file(hlo),
        neff_path=str(neff),
        neff_sha256=_sha256_file(neff),
        emitted_config_sha256=emitted_sha,
        emit_phases=shape["emit_phases"],
        models_compiled=tuple(models),
        cte_artifact_present=cte_present,
        rank_bundle_present=rank_present,
        fresh_prompt_ready=fresh_ready,
        continuation_tkg_ready=continuation_ready,
        blockers=tuple(blockers),
        memory=memory,
    )


__all__ = [
    "GLM53_CARD_PREP_SCHEMA",
    "Glm53CardPreparation",
    "Glm53CardPreparationError",
    "Glm53CheckpointMemory",
    "inspect_tkg_artifact",
    "measure_checkpoint_memory",
]
