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
GLM53_RANK_BYTES = 64 * 1024 * 1024


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
    _require(
        path.is_file() and not path.is_symlink(), f"missing artifact file: {relative}"
    )
    return path


def _require_sha256(value: Any, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value),
        f"{name} must be a lowercase SHA-256",
    )
    return value


def _bound_file(root: Path, entry: Any, name: str) -> Path:
    _require(isinstance(entry, dict), f"{name} packet entry is missing")
    relative = entry.get("relative_path")
    _require(
        isinstance(relative, str)
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        f"{name} packet path is unsafe",
    )
    path = _required_file(root, relative)
    expected_sha = _require_sha256(entry.get("sha256"), f"{name} packet hash")
    _require(_sha256_file(path) == expected_sha, f"{name} hash drift")
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
    rank_inventory_sha256: tuple[str, ...]
    rank_plan_sha256: tuple[str, ...]

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
                "inventory_sha256_by_rank": list(self.rank_inventory_sha256),
                "plan_sha256_by_rank": list(self.rank_plan_sha256),
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
            json.dumps(self.to_mapping(), separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _locate_neff(root: Path, compile_result: dict[str, Any]) -> Path:
    neffs = compile_result.get("neffs")
    _require(
        isinstance(neffs, list) and len(neffs) == 1,
        "artifact must contain exactly one NEFF",
    )
    candidates = sorted(path for path in (root / "cache").rglob("model.neff"))
    _require(len(candidates) == 1, "artifact cache must contain exactly one model.neff")
    _require(not candidates[0].is_symlink(), "retained NEFF must not be symlink")
    listed = neffs[0]
    _require(isinstance(listed, dict), "compile-result NEFF row must be an object")
    _require(listed.get("bytes") == candidates[0].stat().st_size, "NEFF size drift")
    return candidates[0]


def _locate_hlo(root: Path) -> Path:
    candidates = sorted(path for path in (root / "cache").rglob("model.hlo_module.pb"))
    _require(
        len(candidates) == 1, "artifact cache must contain exactly one retained HLO"
    )
    _require(not candidates[0].is_symlink(), "retained HLO must not be symlink")
    return candidates[0]


def _validate_retained_packet(
    root: Path,
    packet_path: Path,
    *,
    shape: dict[str, Any],
    effective_shape_path: Path,
    emitted_config_path: Path,
    emitted_config_sha256: str,
    compile_result_path: Path,
    compile_result: dict[str, Any],
    hlo: Path,
    neff: Path,
) -> dict[str, Any]:
    packet = _load_json(packet_path)
    _require(
        packet.get("schema") == "glm53-retained-tkg-artifact-v1",
        "retained artifact packet schema drift",
    )
    _require(
        packet.get("artifact_role")
        == "retained compiler evidence; not a fresh compile",
        "retained artifact packet role drift",
    )
    topology = packet.get("topology")
    _require(isinstance(topology, dict), "retained topology is missing")
    topology_map = {
        "tp": GLM53_TP,
        "lnc": GLM53_LNC,
        "batch": GLM53_BATCH,
        "sequence": GLM53_SEQUENCE,
        "dtype": "bfloat16",
        "kv_cache_quant": False,
        "quantized": False,
        "speculation": False,
        "emit_phases": "TKG",
        "models_compiled": ["token_generation_model"],
    }
    _require(
        {key: topology.get(key) for key in topology_map} == topology_map,
        "retained topology contract drift",
    )
    _require(
        {
            key: shape.get(key)
            for key in (
                "tp",
                "lnc",
                "resident_batch",
                "sequence",
                "emit_phases",
                "models_compiled",
            )
        }
        == {
            "tp": GLM53_TP,
            "lnc": GLM53_LNC,
            "resident_batch": GLM53_BATCH,
            "sequence": GLM53_SEQUENCE,
            "emit_phases": "TKG",
            "models_compiled": ["token_generation_model"],
        },
        "effective shape contract drift",
    )
    for entry, actual, name in (
        (packet.get("effective_shape"), effective_shape_path, "effective shape"),
        (packet.get("emitted_config"), emitted_config_path, "emitted config"),
        (packet.get("compile_result"), compile_result_path, "compile result"),
    ):
        bound = _bound_file(root, entry, name)
        _require(bound.resolve() == actual.resolve(), f"{name} path drift")
    model_entry = packet.get("model")
    model_path = _bound_file(root, model_entry, "serialized model")
    _require(
        model_path.resolve() == (root / "artifacts/model/model.pt").resolve(),
        "serialized model path drift",
    )
    _require(
        model_path.stat().st_size == model_entry.get("bytes"),
        "serialized model size drift",
    )
    compiler = packet.get("compiler_evidence")
    _require(isinstance(compiler, dict), "compiler evidence is missing")
    hlo_relative = compiler.get("hlo_relative_path")
    neff_relative = compiler.get("neff_relative_path")
    _require(
        isinstance(hlo_relative, str)
        and Path(hlo_relative).as_posix() == hlo.relative_to(root).as_posix(),
        "retained HLO path drift",
    )
    _require(
        isinstance(neff_relative, str)
        and Path(neff_relative).as_posix() == neff.relative_to(root).as_posix(),
        "retained NEFF path drift",
    )
    _require(
        _sha256_file(hlo)
        == _require_sha256(compiler.get("hlo_sha256"), "HLO packet hash"),
        "retained HLO hash drift",
    )
    _require(
        _sha256_file(neff)
        == _require_sha256(compiler.get("neff_sha256"), "NEFF packet hash"),
        "retained NEFF hash drift",
    )
    _require(
        neff.stat().st_size == compiler.get("neff_bytes"),
        "retained NEFF size drift",
    )
    _require(
        emitted_config_sha256
        == _require_sha256(
            packet.get("emitted_config", {}).get("sha256"),
            "emitted config packet hash",
        ),
        "emitted config packet hash drift",
    )
    _require(
        compile_result.get("neff_count") == 1
        and compile_result.get("neuron_config_has_float8_e4m3fn") is False
        and compile_result.get("neuron_config_has_bfloat16") is True,
        "compile-result contract drift",
    )
    hlo_bytes = hlo.read_bytes()
    _require(b"%sort." not in hlo_bytes, "retained HLO contains unsupported %sort.")
    _require(
        b"aten__topk" not in hlo_bytes,
        "retained HLO contains unsupported aten__topk",
    )
    source_entry = packet.get("source_identity")
    source_identity_path = _bound_file(root, source_entry, "source identity")
    source_identity = _load_json(source_identity_path)
    artifact_source = packet.get("artifact_source")
    _require(isinstance(artifact_source, dict), "artifact source identity is missing")
    _require(
        {
            key: source_identity.get(key)
            for key in (
                "source_commit",
                "source_tree",
                "nxdi_commit",
                "checkpoint_revision",
            )
        }
        == {
            key: artifact_source.get(key)
            for key in (
                "source_commit",
                "source_tree",
                "nxdi_commit",
                "checkpoint_revision",
            )
        },
        "artifact source identity drift",
    )
    return packet


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


def _inspect_cte_artifact(root: Path, packet: dict[str, Any]) -> None:
    """Require a bound CTE companion, not a shape-only placeholder."""
    root = root.resolve(strict=True)
    shape_path = _required_file(root, "artifacts/effective-shape.json")
    shape = _load_json(shape_path)
    expected = {
        "model": "GLM-5.3-Flash",
        "tp": GLM53_TP,
        "lnc": GLM53_LNC,
        "resident_batch": GLM53_BATCH,
        "sequence": GLM53_SEQUENCE,
        "emit_phases": "CTE",
        "models_compiled": ["context_encoding_model"],
        "blockwise_use_shard_on_intermediate_dynamic_while": True,
        "kv_cache_quant_requested": False,
    }
    _require(
        {key: shape.get(key) for key in expected} == expected,
        "CTE effective shape contract drift",
    )
    config, config_sha256 = _inspect_emitted_config(root)
    _require(
        config_sha256 == packet["emitted_config"]["sha256"],
        "CTE emitted config is not bound to TKG config",
    )
    source_path = _required_file(root, "artifacts/source-identity.json")
    source = _load_json(source_path)
    artifact_source = packet["artifact_source"]
    _require(
        {
            key: source.get(key)
            for key in (
                "source_commit",
                "source_tree",
                "nxdi_commit",
                "checkpoint_revision",
            )
        }
        == {
            key: artifact_source.get(key)
            for key in (
                "source_commit",
                "source_tree",
                "nxdi_commit",
                "checkpoint_revision",
            )
        },
        "CTE source identity drift",
    )
    compile_result_path = _required_file(root, "artifacts/compile-result.json")
    compile_result = _load_json(compile_result_path)
    hlo = _locate_hlo(root)
    neff = _locate_neff(root, compile_result)
    _require(
        compile_result.get("neff_count") == 1
        and compile_result.get("neuron_config_has_float8_e4m3fn") is False
        and compile_result.get("neuron_config_has_bfloat16") is True,
        "CTE compile-result contract drift",
    )
    hlo_bytes = hlo.read_bytes()
    _require(b"%sort." not in hlo_bytes, "CTE HLO contains unsupported %sort.")
    _require(b"aten__topk" not in hlo_bytes, "CTE HLO contains unsupported aten__topk")
    _ = config
    _require(source_path.is_file(), "CTE source identity is missing")


def _rank_bundle_is_valid(
    rank_dir: Path,
    checkpoint_dir: Path | None,
    memory: Glm53CheckpointMemory | None,
) -> bool:
    """Validate every emitted rank and manifest before exposing readiness."""
    if checkpoint_dir is None or memory is None or memory.rank_count != GLM53_TP:
        return False
    try:
        for rank in range(GLM53_TP):
            filename = f"tp{rank}_sharded_checkpoint.safetensors"
            rank_path = _required_file(rank_dir, filename)
            manifest_path = _required_file(rank_dir, f"{filename}.manifest.json")
            manifest = _load_json(manifest_path)
            _require(
                manifest.get("schema") == "glm53-streaming-rank-v1",
                "rank manifest schema drift",
            )
            _require(manifest.get("rank") == rank, "rank manifest rank drift")
            _require(manifest.get("tp_degree") == GLM53_TP, "rank manifest TP drift")
            source = manifest.get("source")
            _require(isinstance(source, dict), "rank manifest source is missing")
            _require(
                {
                    key: source.get(key)
                    for key in ("model", "revision", "config_sha256", "index_sha256")
                }
                == {
                    "model": "GLM-5.3-Flash",
                    "revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
                    "config_sha256": "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f",
                    "index_sha256": "3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05",
                },
                "rank manifest checkpoint identity drift",
            )
            checkpoint = manifest.get("checkpoint")
            _require(
                isinstance(checkpoint, dict), "rank manifest checkpoint is missing"
            )
            _require(checkpoint.get("path") == filename, "rank manifest filename drift")
            _require(
                checkpoint.get("bytes") == rank_path.stat().st_size > 0,
                "rank payload size drift",
            )
            _require(
                checkpoint.get("sha256") == _sha256_file(rank_path),
                "rank payload hash drift",
            )
            resource = manifest.get("resource_bound")
            _require(isinstance(resource, dict), "rank resource bound is missing")
            _require(
                resource.get("configured_max_chunk_bytes") == GLM53_RANK_BYTES
                and isinstance(resource.get("observed_max_chunk_bytes"), int)
                and 0 < resource["observed_max_chunk_bytes"] <= GLM53_RANK_BYTES
                and resource.get("full_rank_tensor_bytes")
                == memory.rank_tensor_bytes[rank],
                "rank resource contract drift",
            )
            _require(
                isinstance(resource.get("chunks_written"), int)
                and resource["chunks_written"] > 0,
                "rank transaction completion is missing",
            )
            _require(
                manifest.get("rank_inventory_sha256")
                == memory.rank_inventory_sha256[rank]
                and manifest.get("rank_plan_sha256") == memory.rank_plan_sha256[rank]
                and _require_sha256(
                    memory.rank_inventory_sha256[rank], "rank inventory"
                )
                and _require_sha256(memory.rank_plan_sha256[rank], "rank plan"),
                "rank plan identity drift",
            )
            tensors = manifest.get("tensors")
            _require(
                isinstance(tensors, list) and tensors,
                "rank tensor inventory is missing",
            )
            by_dtype: dict[str, int] = {}
            for tensor in tensors:
                _require(isinstance(tensor, dict), "rank tensor entry is invalid")
                dtype = tensor.get("dtype")
                _require(
                    dtype in ("torch.bfloat16", "torch.float32"),
                    "rank tensor dtype drift",
                )
                nbytes = tensor.get("nbytes")
                _require(
                    isinstance(nbytes, int) and nbytes > 0,
                    "rank tensor byte count drift",
                )
                by_dtype[dtype] = by_dtype.get(dtype, 0) + nbytes
            _require(
                sum(by_dtype.values()) == memory.rank_tensor_bytes[rank]
                and by_dtype.get("torch.bfloat16", 0)
                == memory.rank_bfloat16_bytes[rank]
                and by_dtype.get("torch.float32", 0)
                == memory.rank_non_bfloat16_bytes[rank],
                "rank tensor dtype/size inventory drift",
            )
    except (OSError, KeyError, TypeError, ValueError, Glm53CardPreparationError):
        return False
    return True


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
            checkpoint_dir,
            rank=rank,
            tp_degree=GLM53_TP,
            max_chunk_bytes=64 * 1024 * 1024,
        )
        for rank in range(GLM53_TP)
    )
    tensor_bytes: list[int] = []
    bfloat16_bytes: list[int] = []
    non_bfloat16_bytes: list[int] = []
    inventory_sha256: list[str] = []
    plan_sha256: list[str] = []
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
        inventory_sha256.append(getattr(plan.inventory, "contract_sha256", ""))
        plan_sha256.append(getattr(plan, "contract_sha256", ""))
    return Glm53CheckpointMemory(
        source_shard_count=audit.shard_count,
        source_tensor_count=audit.tensor_count,
        source_indexed_payload_bytes=audit.indexed_payload_bytes,
        source_payload_bytes_loaded_during_audit=audit.payload_bytes_loaded_during_audit,
        rank_tensor_bytes=tuple(tensor_bytes),
        rank_bfloat16_bytes=tuple(bfloat16_bytes),
        rank_non_bfloat16_bytes=tuple(non_bfloat16_bytes),
        rank_inventory_sha256=tuple(inventory_sha256),
        rank_plan_sha256=tuple(plan_sha256),
    )


def inspect_tkg_artifact(
    artifact_root: str | Path,
    *,
    checkpoint_dir: str | Path | None = None,
    rank_dir: str | Path | None = None,
    cte_artifact_root: str | Path | None = None,
    retained_packet_path: str | Path | None = None,
) -> Glm53CardPreparation:
    """Inspect a bound TKG artifact and return a non-authorizing receipt."""

    root = Path(artifact_root).resolve(strict=True)
    effective_shape_path = _required_file(root, "artifacts/effective-shape.json")
    shape = _load_json(effective_shape_path)
    _require(shape.get("tp") == GLM53_TP, "effective shape TP drift")
    _require(shape.get("lnc") == GLM53_LNC, "effective shape LNC drift")
    _require(shape.get("resident_batch") == GLM53_BATCH, "effective shape batch drift")
    _require(shape.get("sequence") == GLM53_SEQUENCE, "effective shape sequence drift")
    _require(
        shape.get("emit_phases") == "TKG", "artifact is not the qualified TKG profile"
    )
    models = shape.get("models_compiled")
    _require(models == ["token_generation_model"], "artifact is not TKG-only")
    emitted_config_path = _required_file(root, "artifacts/model/neuron_config.json")
    _, emitted_sha = _inspect_emitted_config(root)
    compile_result_path = _required_file(root, "artifacts/compile-result.json")
    compile_result = _load_json(compile_result_path)
    hlo = _locate_hlo(root)
    neff = _locate_neff(root, compile_result)
    packet_path = (
        Path(retained_packet_path).resolve(strict=True)
        if retained_packet_path is not None
        else Path(__file__).with_name("RETAINED-TKG-ARTIFACT-PACKET.json")
    )
    packet = _validate_retained_packet(
        root,
        packet_path,
        shape=shape,
        effective_shape_path=effective_shape_path,
        emitted_config_path=emitted_config_path,
        emitted_config_sha256=emitted_sha,
        compile_result_path=compile_result_path,
        compile_result=compile_result,
        hlo=hlo,
        neff=neff,
    )

    cte_present = False
    if cte_artifact_root is not None:
        try:
            _inspect_cte_artifact(Path(cte_artifact_root), packet)
            cte_present = True
        except (OSError, KeyError, TypeError, ValueError, Glm53CardPreparationError):
            cte_present = False

    rank_present = False
    memory = (
        measure_checkpoint_memory(checkpoint_dir)
        if checkpoint_dir is not None
        else None
    )
    if rank_dir is not None:
        rank_root = Path(rank_dir).resolve(strict=True)
        rank_present = _rank_bundle_is_valid(
            rank_root, Path(checkpoint_dir) if checkpoint_dir else None, memory
        )

    blockers: list[str] = []
    if not cte_present:
        blockers.append("fresh prompt requires a context_encoding_model artifact")
    if not rank_present:
        blockers.append(
            "card load requires all 32 TP32 sharded BF16 rank files and manifests"
        )
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
