# SPDX-License-Identifier: Apache-2.0
"""Host-only runtime-artifact factory for qualified GLM-5.3 TP bundles.

The factory joins three already-reviewed boundaries without constructing or
loading a model: the immutable checkpoint metadata, the transactional rank
plan, and the requested-versus-emitted runtime configuration.  It emits a
canonical bundle contract only after every rank file and manifest verifies.
Compile and runtime remain explicitly unauthorized.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint_converter import (
    GLM53_CHECKPOINT_REVISION,
    GLM53_CONFIG_SHA256,
    GLM53_INDEX_SHA256,
)
from .rank_plan import build_glm53_rank_plan
from .runtime_config import GLM53_ARCHITECTURE, Glm53RuntimeConfig

GLM53_RUNTIME_ADAPTER = (
    "vllm_neuron.model.glm53_flash.runtime_factory.Glm53RuntimeFactory"
)
GLM53_RUNTIME_BUNDLE_SCHEMA = "glm53-runtime-artifact-bundle-v1"
GLM53_RUNTIME_FACTORY_ABI = (
    "glm53-runtime-factory-v1|checkpoint=04c4e9e95c5d|tp=32|"
    "rank-manifest=glm53-streaming-rank-v1|emitted-config=v2|no-spec|no-runtime-quant"
)
GLM53_TP64_RUNTIME_FACTORY_ABI = (
    "glm53-runtime-factory-v2|class=R3|checkpoint=04c4e9e95c5d|tp=64|"
    "rank-manifest=glm53-streaming-rank-v1+rank-plan-copy[32,4096]|"
    "emitted-config=v2|lnc=2|b1-prefill-s2048-decode-total-s2560|"
    "no-spec|no-runtime-quant"
)
GLM53_TP32_RANK_COUNT = 32
GLM53_TP64_RANK_COUNT = 64
# Compatibility alias: existing TP32 callers remain bound to 32 ranks.
GLM53_RANK_COUNT = GLM53_TP32_RANK_COUNT
GLM53_MAX_CHUNK_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_UNIT = re.compile(r"glm53-compile-[A-Za-z0-9_.-]{1,96}")


class Glm53RuntimeFactoryError(ValueError):
    """The reviewed runtime bundle cannot be assembled exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Glm53RuntimeFactoryError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Glm53RuntimeFactoryError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _rank_count_for(tp_degree: int) -> int:
    if tp_degree == 32:
        return GLM53_TP32_RANK_COUNT
    if tp_degree == 64:
        return GLM53_TP64_RANK_COUNT
    raise Glm53RuntimeFactoryError(f"unsupported tensor parallelism: TP{tp_degree}")


def _factory_abi_for(tp_degree: int) -> str:
    if tp_degree == 32:
        return GLM53_RUNTIME_FACTORY_ABI
    if tp_degree == 64:
        return GLM53_TP64_RUNTIME_FACTORY_ABI
    raise Glm53RuntimeFactoryError(f"unsupported tensor parallelism: TP{tp_degree}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object
        )
    except json.JSONDecodeError as exc:
        raise Glm53RuntimeFactoryError(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


@dataclass(frozen=True)
class Glm53CompileLaunchPolicy:
    """Explicit future launch policy; this object itself never authorizes fire."""

    ownership_path: str
    active_compile_cap: int
    systemd_unit: str
    systemd_nice: int
    systemd_scope: bool
    network_mode: str
    atomic_staging_suffix: str
    compile_permitted: bool

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> Glm53CompileLaunchPolicy:
        expected = {
            "ownership_path",
            "active_compile_cap",
            "systemd_unit",
            "systemd_nice",
            "systemd_scope",
            "network_mode",
            "atomic_staging_suffix",
            "compile_permitted",
        }
        _require(set(value) == expected, "compile launch policy field-set mismatch")
        _require(
            value["ownership_path"] == "/mnt/compile/OWNERSHIP.md",
            "compile policy requires exact OWNERSHIP.md path",
        )
        _require(value["active_compile_cap"] == 2, "compile cap must be exactly two")
        _require(
            isinstance(value["systemd_unit"], str)
            and _UNIT.fullmatch(value["systemd_unit"]) is not None,
            "compile requires a bounded named systemd unit",
        )
        _require(value["systemd_nice"] == 15, "compile requires systemd nice=15")
        _require(value["systemd_scope"] is False, "systemd --scope is forbidden")
        _require(value["network_mode"] == "none", "compile network must be none")
        _require(
            value["atomic_staging_suffix"] == ".partial-<run-id>",
            "atomic staging suffix drift",
        )
        _require(
            value["compile_permitted"] is False, "factory cannot authorize compile"
        )
        return cls(**value)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "ownership_path": self.ownership_path,
            "active_compile_cap": self.active_compile_cap,
            "systemd_unit": self.systemd_unit,
            "systemd_nice": self.systemd_nice,
            "systemd_scope": self.systemd_scope,
            "network_mode": self.network_mode,
            "atomic_staging_suffix": self.atomic_staging_suffix,
            "compile_permitted": self.compile_permitted,
        }


@dataclass(frozen=True)
class Glm53RuntimeRank:
    rank: int
    checkpoint_path: str
    checkpoint_bytes: int
    checkpoint_sha256: str
    manifest_path: str
    manifest_sha256: str
    inventory_sha256: str
    plan_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "checkpoint": {
                "path": self.checkpoint_path,
                "bytes": self.checkpoint_bytes,
                "sha256": self.checkpoint_sha256,
            },
            "manifest": {
                "path": self.manifest_path,
                "sha256": self.manifest_sha256,
            },
            "inventory_sha256": self.inventory_sha256,
            "plan_sha256": self.plan_sha256,
        }


@dataclass(frozen=True)
class Glm53RuntimeArtifactBundle:
    requested_config_sha256: str
    emitted_config_sha256: str
    tensor_parallel_degree: int
    output_logits: bool
    compile_policy: Glm53CompileLaunchPolicy
    ranks: tuple[Glm53RuntimeRank, ...]

    def to_mapping(self) -> dict[str, Any]:
        rank_rows = [rank.to_mapping() for rank in self.ranks]
        return {
            "schema": GLM53_RUNTIME_BUNDLE_SCHEMA,
            "architecture": GLM53_ARCHITECTURE,
            "checkpoint": {
                "revision": GLM53_CHECKPOINT_REVISION,
                "config_sha256": GLM53_CONFIG_SHA256,
                "index_sha256": GLM53_INDEX_SHA256,
            },
            "factory": {
                "adapter": GLM53_RUNTIME_ADAPTER,
                "abi": _factory_abi_for(self.tensor_parallel_degree),
            },
            "topology": {
                "tp_degree": self.tensor_parallel_degree,
                "rank_count": len(rank_rows),
                "indexer_weights_proj_ownership": (
                    "replicated-copy[32,4096]"
                    if self.tensor_parallel_degree == 64
                    else "tp32-sharded-gathered"
                ),
                "output_logits": self.output_logits,
            },
            "runtime_config": {
                "requested_sha256": self.requested_config_sha256,
                "emitted_sha256": self.emitted_config_sha256,
                "equal": self.requested_config_sha256 == self.emitted_config_sha256,
            },
            "compile_policy": self.compile_policy.to_mapping(),
            "ranks": rank_rows,
            "rank_bundle_sha256": _canonical_sha256(rank_rows),
            "claims": {
                "rank_files_verified": True,
                "compile_permitted": False,
                "runtime_permitted": False,
                "correctness_40_of_40": False,
                "performance": False,
                "tokenomics": False,
            },
        }

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(self.to_mapping(), separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class Glm53RuntimeFactory:
    """Verify and join exact host-produced inputs; never instantiate a model."""

    @classmethod
    def from_paths(
        cls,
        *,
        checkpoint_dir: str | Path,
        rank_dir: str | Path,
        requested_config: str | Path,
        emitted_config: str | Path,
        compile_policy: dict[str, Any],
    ) -> Glm53RuntimeArtifactBundle:
        checkpoint_root = Path(checkpoint_dir).resolve(strict=True)
        rank_root = Path(rank_dir).resolve(strict=True)
        requested_path = Path(requested_config).resolve(strict=True)
        emitted_path = Path(emitted_config).resolve(strict=True)
        requested = Glm53RuntimeConfig.load_canonical(requested_path.read_bytes())
        emitted = Glm53RuntimeConfig.load_canonical(emitted_path.read_bytes())
        requested.assert_emitted(emitted.to_mapping())
        _require(
            requested.runtime_adapter == GLM53_RUNTIME_ADAPTER,
            "runtime adapter identity drift",
        )
        tp_degree = requested.tensor_parallel_degree
        _require(tp_degree in (32, 64), "runtime factory requires TP32 or TP64")
        _require(
            requested.logical_neuron_cores == 2,
            "runtime factory requires LNC2",
        )
        _require(requested.weight_dtype == "bfloat16", "rank weight dtype must be BF16")
        _require(requested.cache_dtype == "bfloat16", "cache dtype must be BF16")
        _require(
            requested.runtime_quantization == "none",
            "runtime quantization must be explicitly none",
        )
        _require(requested.sampling_mode == "greedy", "runtime sampling must be greedy")
        _require(
            requested.speculative_decode is False, "speculative decode is forbidden"
        )
        if tp_degree == 64:
            _require(
                requested.output_logits is True,
                "TP64 factory requires output_logits=true",
            )
            _require(requested.batch_size == 1, "TP64 factory requires B1")
            _require(
                requested.max_sequence_length == 2560,
                "TP64 factory requires total S2560",
            )
            _require(
                requested.context_encoding_buckets == (2048,),
                "TP64 factory requires one CTE S2048 bucket",
            )
            _require(
                requested.token_generation_buckets == (2560,),
                "TP64 factory requires one TKG total-S2560 bucket",
            )
        policy = Glm53CompileLaunchPolicy.from_mapping(compile_policy)
        ranks = tuple(
            cls._validate_rank(checkpoint_root, rank_root, rank, tp_degree=tp_degree)
            for rank in range(_rank_count_for(tp_degree))
        )
        return Glm53RuntimeArtifactBundle(
            requested_config_sha256=requested.sha256(),
            emitted_config_sha256=emitted.sha256(),
            tensor_parallel_degree=tp_degree,
            output_logits=requested.output_logits,
            compile_policy=policy,
            ranks=ranks,
        )

    @staticmethod
    def _validate_rank(
        checkpoint_root: Path, rank_root: Path, rank: int, *, tp_degree: int
    ) -> Glm53RuntimeRank:
        checkpoint = rank_root / f"tp{rank}_sharded_checkpoint.safetensors"
        manifest_path = checkpoint.with_suffix(checkpoint.suffix + ".manifest.json")
        _require(
            checkpoint.is_file() and not checkpoint.is_symlink(), f"rank {rank} missing"
        )
        _require(
            manifest_path.is_file() and not manifest_path.is_symlink(),
            f"rank {rank} manifest missing",
        )
        manifest = _load_json(manifest_path)
        expected_plan = build_glm53_rank_plan(
            checkpoint_root,
            rank=rank,
            tp_degree=tp_degree,
            max_chunk_bytes=GLM53_MAX_CHUNK_BYTES,
        )
        _require(
            manifest.get("schema") == "glm53-streaming-rank-v1", "rank schema drift"
        )
        _require(manifest.get("rank") == rank, f"rank manifest identity drift: {rank}")
        _require(manifest.get("tp_degree") == tp_degree, "rank manifest TP drift")
        source = manifest.get("source", {})
        _require(
            {
                key: source.get(key)
                for key in ("model", "revision", "config_sha256", "index_sha256")
            }
            == {
                "model": "GLM-5.3-Flash",
                "revision": GLM53_CHECKPOINT_REVISION,
                "config_sha256": GLM53_CONFIG_SHA256,
                "index_sha256": GLM53_INDEX_SHA256,
            },
            "rank source identity drift",
        )
        _require(
            manifest.get("rank_inventory_sha256")
            == expected_plan.inventory.contract_sha256,
            "rank inventory identity drift",
        )
        _require(
            manifest.get("rank_plan_sha256") == expected_plan.contract_sha256,
            "rank plan identity drift",
        )
        if tp_degree == 64:
            indexer_ops = [
                operation
                for operation in getattr(expected_plan, "operations", ())
                if operation.target.name.endswith("indexer.weights_proj.weight")
            ]
            _require(
                len(indexer_ops) == 11,
                "TP64 rank plan lacks all 11 indexer weights_proj entries",
            )
            _require(
                all(
                    operation.kind == "copy" and operation.target.shape == (32, 4096)
                    for operation in indexer_ops
                ),
                "TP64 rank plan must copy full indexer weights_proj [32,4096]",
            )
        checkpoint_row = manifest.get("checkpoint", {})
        _require(
            checkpoint_row.get("path") == checkpoint.name, "rank checkpoint path drift"
        )
        _require(
            checkpoint_row.get("bytes") == checkpoint.stat().st_size,
            "rank checkpoint size drift",
        )
        checkpoint_sha = _sha256_file(checkpoint)
        _require(
            checkpoint_row.get("sha256") == checkpoint_sha, "rank checkpoint hash drift"
        )
        resource = manifest.get("resource_bound", {})
        _require(
            resource.get("configured_max_chunk_bytes") == GLM53_MAX_CHUNK_BYTES,
            "rank configured chunk bound drift",
        )
        observed = resource.get("observed_max_chunk_bytes")
        _require(
            type(observed) is int and 0 < observed <= GLM53_MAX_CHUNK_BYTES,
            "rank observed chunk bound invalid",
        )
        _require(
            resource.get("full_rank_tensor_bytes")
            == expected_plan.inventory.total_tensor_bytes,
            "rank tensor byte inventory drift",
        )
        tensors = manifest.get("tensors")
        expected_tensors = [
            {
                "name": spec.name,
                "dtype": str(spec.dtype),
                "shape": list(spec.shape),
                "nbytes": spec.nbytes,
            }
            for spec in expected_plan.inventory.tensors
        ]
        _require(isinstance(tensors, list), "rank tensor manifest missing")
        observed_tensors = [
            {key: row.get(key) for key in ("name", "dtype", "shape", "nbytes")}
            for row in tensors
        ]
        _require(observed_tensors == expected_tensors, "rank tensor manifest drift")
        inventory_sha = manifest["rank_inventory_sha256"]
        plan_sha = manifest["rank_plan_sha256"]
        _require(_SHA256.fullmatch(inventory_sha) is not None, "invalid inventory SHA")
        _require(_SHA256.fullmatch(plan_sha) is not None, "invalid plan SHA")
        return Glm53RuntimeRank(
            rank=rank,
            checkpoint_path=checkpoint.name,
            checkpoint_bytes=checkpoint.stat().st_size,
            checkpoint_sha256=checkpoint_sha,
            manifest_path=manifest_path.name,
            manifest_sha256=_sha256_file(manifest_path),
            inventory_sha256=inventory_sha,
            plan_sha256=plan_sha,
        )


def get_runtime_factories() -> list[tuple[str, type[Glm53RuntimeFactory]]]:
    """Return only the host-side artifact factory; no model is registered."""
    return [(GLM53_ARCHITECTURE, Glm53RuntimeFactory)]


__all__ = [
    "GLM53_RUNTIME_ADAPTER",
    "GLM53_RUNTIME_BUNDLE_SCHEMA",
    "GLM53_RUNTIME_FACTORY_ABI",
    "GLM53_TP64_RUNTIME_FACTORY_ABI",
    "Glm53CompileLaunchPolicy",
    "Glm53RuntimeArtifactBundle",
    "Glm53RuntimeFactory",
    "Glm53RuntimeFactoryError",
    "Glm53RuntimeRank",
    "get_runtime_factories",
]
