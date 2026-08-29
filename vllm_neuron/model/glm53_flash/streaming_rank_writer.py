# SPDX-License-Identifier: Apache-2.0
"""Transactional, resource-bounded per-rank checkpoint writer.

The writer emits a valid SafeTensors file from bounded tensor chunks.  It
predeclares the complete rank inventory in the SafeTensors header, writes each
chunk directly to its final byte range, rejects overlap and gaps, and publishes
the output atomically only after every byte is covered.  It never constructs a
full converted model or a full-rank tensor dictionary.

``IndexedTensorReader`` independently audits the immutable source index against
the actual shard headers.  Missing shards, extra shards, missing tensors,
misrouted tensors, and orphan tensors all fail before conversion begins.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import torch

from .checkpoint_converter import (
    GLM53_CHECKPOINT_REVISION,
    GLM53_CONFIG_SHA256,
    GLM53_INDEX_SHA256,
    Glm53CheckpointReport,
    classify_tensor,
    dequantize_block_fp8,
    preflight_checkpoint_dir,
)

_TORCH_TO_SAFE_DTYPE: dict[torch.dtype, str] = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}
_SAFE_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}
for _torch_name, _safe_name in (
    ("float8_e4m3fn", "F8_E4M3"),
    ("float8_e5m2", "F8_E5M2"),
):
    if hasattr(torch, _torch_name):
        _TORCH_TO_SAFE_DTYPE[getattr(torch, _torch_name)] = _safe_name


class Glm53StreamingError(RuntimeError):
    """Raised when source auditing or transactional rank writing fails."""


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name or self.name == "__metadata__":
            raise ValueError(f"invalid tensor name: {self.name!r}")
        if self.dtype not in _TORCH_TO_SAFE_DTYPE:
            raise TypeError(
                f"unsupported SafeTensors dtype for {self.name}: {self.dtype}"
            )
        if not self.shape or any(
            not isinstance(dim, int) or dim <= 0 for dim in self.shape
        ):
            raise ValueError(
                f"{self.name} requires a non-empty positive shape; got {self.shape}"
            )

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def element_size(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    @property
    def nbytes(self) -> int:
        return self.numel * self.element_size


@dataclass(frozen=True)
class RankInventory:
    rank: int
    tp_degree: int
    tensors: tuple[TensorSpec, ...]

    def __post_init__(self) -> None:
        if self.tp_degree <= 0 or self.rank < 0 or self.rank >= self.tp_degree:
            raise ValueError(
                f"invalid rank/TP pair: rank={self.rank}, tp_degree={self.tp_degree}"
            )
        if not self.tensors:
            raise ValueError("rank inventory must contain at least one tensor")
        names = [spec.name for spec in self.tensors]
        if len(names) != len(set(names)):
            raise ValueError("rank inventory contains duplicate tensor names")

    @property
    def total_tensor_bytes(self) -> int:
        return sum(spec.nbytes for spec in self.tensors)

    @property
    def contract_sha256(self) -> str:
        payload = {
            "rank": self.rank,
            "tp_degree": self.tp_degree,
            "tensors": [
                {
                    "name": spec.name,
                    "dtype": str(spec.dtype),
                    "shape": list(spec.shape),
                    "nbytes": spec.nbytes,
                }
                for spec in self.tensors
            ],
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TensorChunk:
    tensor_name: str
    start_element: int
    tensor: torch.Tensor


@dataclass(frozen=True)
class SourceAuditReport:
    shard_count: int
    tensor_count: int
    indexed_payload_bytes: int
    payload_bytes_loaded_during_audit: int


@dataclass(frozen=True)
class SourceTensorSpec:
    safe_dtype: str
    shape: tuple[int, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_open():
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - deployment dependency gate
        raise Glm53StreamingError(
            "safetensors is required for GLM-5.3 source shard auditing"
        ) from exc
    return safe_open


class IndexedTensorReader:
    """Audit and lazily read the exact immutable GLM-5.3 shard set."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.root = Path(checkpoint_dir).resolve(strict=True)
        self.preflight_report = preflight_checkpoint_dir(self.root)
        index = json.loads(
            (self.root / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping):
            raise Glm53StreamingError("immutable index has no weight_map")
        self.weight_map = {str(key): str(value) for key, value in weight_map.items()}
        self.source_specs: dict[str, SourceTensorSpec] = {}
        self.max_source_group_bytes = 0
        self.source_groups_loaded = 0
        self.audit_report = self._audit_shards()

    def _audit_shards(self) -> SourceAuditReport:
        safe_open = _safe_open()
        expected_by_shard: dict[str, set[str]] = {}
        for key, shard_name in self.weight_map.items():
            shard_path = Path(shard_name)
            if shard_path.is_absolute() or shard_path.name != shard_name:
                raise Glm53StreamingError(f"unsafe shard path in index: {shard_name!r}")
            expected_by_shard.setdefault(shard_name, set()).add(key)

        referenced = set(expected_by_shard)
        actual_files = {path.name for path in self.root.glob("*.safetensors")}
        missing_files = sorted(referenced - actual_files)
        extra_files = sorted(actual_files - referenced)
        if missing_files or extra_files:
            raise Glm53StreamingError(
                f"source shard inventory mismatch: missing={missing_files[:4]} "
                f"extra={extra_files[:4]}"
            )

        seen: set[str] = set()
        indexed_payload_bytes = 0
        for shard_name in sorted(referenced):
            shard_path = self.root / shard_name
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                actual_keys = set(handle.keys())
                header_specs = {
                    key: (
                        handle.get_slice(key).get_dtype(),
                        handle.get_slice(key).get_shape(),
                    )
                    for key in actual_keys
                }
            expected_keys = expected_by_shard[shard_name]
            missing_keys = sorted(expected_keys - actual_keys)
            orphan_keys = sorted(actual_keys - expected_keys)
            if missing_keys or orphan_keys:
                raise Glm53StreamingError(
                    f"tensor inventory mismatch in {shard_name}: "
                    f"missing={missing_keys[:4]} orphan={orphan_keys[:4]}"
                )
            duplicate = seen & actual_keys
            if duplicate:
                raise Glm53StreamingError(
                    f"tensor appears in multiple shards: {sorted(duplicate)[:4]}"
                )
            seen.update(actual_keys)
            for key, (safe_dtype, shape) in header_specs.items():
                self.source_specs[key] = SourceTensorSpec(
                    safe_dtype=safe_dtype, shape=tuple(shape)
                )
                dtype_bytes = _SAFE_DTYPE_BYTES.get(safe_dtype)
                if dtype_bytes is None:
                    raise Glm53StreamingError(
                        f"unsupported source dtype {safe_dtype!r} for {key}"
                    )
                indexed_payload_bytes += math.prod(shape) * dtype_bytes
                scale_key = f"{key}_scale_inv"
                if safe_dtype.startswith("F8_") and scale_key not in self.weight_map:
                    raise Glm53StreamingError(
                        f"orphan FP8 weight has no reciprocal scale: {key}"
                    )
                if scale_key in self.weight_map and not safe_dtype.startswith("F8_"):
                    raise Glm53StreamingError(
                        f"reciprocal scale is paired with a non-FP8 weight: "
                        f"{key}={safe_dtype}"
                    )
                if key.endswith("_scale_inv"):
                    base_key = key[: -len("_scale_inv")]
                    if base_key not in self.weight_map:
                        raise Glm53StreamingError(
                            f"orphan reciprocal scale has no weight: {key}"
                        )
                    if safe_dtype not in {"F16", "BF16", "F32", "F64"}:
                        raise Glm53StreamingError(
                            f"reciprocal scale must be floating point: {key}={safe_dtype}"
                        )
        if seen != set(self.weight_map):
            raise Glm53StreamingError(
                "audited shard keys do not exactly equal immutable index"
            )
        return SourceAuditReport(
            shard_count=len(referenced),
            tensor_count=len(seen),
            indexed_payload_bytes=indexed_payload_bytes,
            payload_bytes_loaded_during_audit=0,
        )

    def _load(self, key: str) -> torch.Tensor:
        shard_name = self.weight_map.get(key)
        if shard_name is None:
            raise Glm53StreamingError(f"tensor is not in immutable index: {key}")
        safe_open = _safe_open()
        with safe_open(self.root / shard_name, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            if key not in actual_keys:
                raise Glm53StreamingError(
                    f"indexed tensor disappeared from shard: {key}"
                )
            return handle.get_tensor(key)

    def _load_slice(self, key: str, slices: tuple[slice, ...]) -> torch.Tensor:
        shard_name = self.weight_map.get(key)
        spec = self.source_specs.get(key)
        if shard_name is None or spec is None:
            raise Glm53StreamingError(
                f"tensor is not in audited immutable index: {key}"
            )
        if len(slices) != len(spec.shape):
            raise Glm53StreamingError(
                f"slice rank for {key} is {len(slices)}; expected {len(spec.shape)}"
            )
        safe_open = _safe_open()
        with safe_open(self.root / shard_name, framework="pt", device="cpu") as handle:
            return handle.get_slice(key)[slices]

    @staticmethod
    def _normalize_slice(value: slice, size: int, key: str) -> tuple[int, int]:
        start, stop, step = value.indices(size)
        if step != 1 or stop <= start:
            raise Glm53StreamingError(
                f"{key} requires a non-empty unit-stride slice; got {value!r}"
            )
        return start, stop

    def read_converted_slice(
        self,
        key: str,
        slices: tuple[slice, ...],
        *,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Read and convert one bounded source slice.

        Block-FP8 slices may start inside a 128x128 tile. Only the intersecting
        reciprocal-scale cells are loaded, expanded, and cropped to the exact
        requested window; no full source tensor is materialized.
        """
        policy = classify_tensor(key, self.weight_map)
        if policy in ("drop_mtp", "drop_vision", "block_fp8_scale"):
            raise Glm53StreamingError(
                f"tensor policy {policy} cannot be emitted directly: {key}"
            )
        spec = self.source_specs.get(key)
        if spec is None:
            raise Glm53StreamingError(
                f"tensor is not in audited immutable index: {key}"
            )
        bounds = [
            self._normalize_slice(value, size, key)
            for value, size in zip(slices, spec.shape, strict=True)
        ]
        weight = self._load_slice(key, slices)
        loaded_bytes = weight.numel() * weight.element_size()
        if policy == "block_fp8_weight":
            if len(spec.shape) < 2:
                raise Glm53StreamingError(f"block-FP8 tensor is not rank >=2: {key}")
            scale_key = f"{key}_scale_inv"
            scale_spec = self.source_specs.get(scale_key)
            if scale_spec is None:
                raise Glm53StreamingError(
                    f"missing audited reciprocal scale: {scale_key}"
                )
            block_out, block_in = (128, 128)
            expected_scale_shape = spec.shape[:-2] + (
                math.ceil(spec.shape[-2] / block_out),
                math.ceil(spec.shape[-1] / block_in),
            )
            if scale_spec.shape != expected_scale_shape:
                raise Glm53StreamingError(
                    f"reciprocal scale shape drift for {scale_key}: expected "
                    f"{expected_scale_shape}, got {scale_spec.shape}"
                )
            scale_slices = list(slices[:-2])
            row_start, row_stop = bounds[-2]
            col_start, col_stop = bounds[-1]
            scale_slices.extend(
                (
                    slice(row_start // block_out, math.ceil(row_stop / block_out)),
                    slice(col_start // block_in, math.ceil(col_stop / block_in)),
                )
            )
            scale = self._load_slice(scale_key, tuple(scale_slices))
            loaded_bytes += scale.numel() * scale.element_size()
            expanded = scale.to(torch.float32)
            expanded = expanded.repeat_interleave(block_out, -2).repeat_interleave(
                block_in, -1
            )
            row_offset = row_start % block_out
            col_offset = col_start % block_in
            expanded = expanded[
                ...,
                row_offset : row_offset + (row_stop - row_start),
                col_offset : col_offset + (col_stop - col_start),
            ]
            result = (weight.to(torch.float32) * expanded).to(out_dtype)
        else:
            if spec.safe_dtype.startswith("F8_"):
                raise Glm53StreamingError(
                    f"unscaled FP8 tensor is not a valid holdout: {key}"
                )
            result = weight.to(out_dtype)
        self.max_source_group_bytes = max(self.max_source_group_bytes, loaded_bytes)
        self.source_groups_loaded += 1
        return result

    def read_converted(
        self, key: str, *, out_dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        policy = classify_tensor(key, self.weight_map)
        if policy in ("drop_mtp", "drop_vision", "block_fp8_scale"):
            raise Glm53StreamingError(
                f"tensor policy {policy} cannot be emitted directly: {key}"
            )
        weight = self._load(key)
        loaded_bytes = weight.numel() * weight.element_size()
        if policy == "block_fp8_weight":
            scale_key = f"{key}_scale_inv"
            scale = self._load(scale_key)
            loaded_bytes += scale.numel() * scale.element_size()
            result = dequantize_block_fp8(weight, scale, out_dtype=out_dtype)
        else:
            float8_types = tuple(
                dtype
                for name in ("float8_e4m3fn", "float8_e5m2")
                if (dtype := getattr(torch, name, None)) is not None
            )
            if weight.dtype in float8_types:
                raise Glm53StreamingError(
                    f"unscaled FP8 tensor is not a valid BF16 holdout: {key}"
                )
            result = weight.to(out_dtype)
        self.max_source_group_bytes = max(self.max_source_group_bytes, loaded_bytes)
        self.source_groups_loaded += 1
        return result


class StreamingRankWriter:
    """Write one rank directly into a transactional SafeTensors file."""

    def __init__(
        self,
        output_path: str | Path,
        inventory: RankInventory,
        *,
        source_report: Glm53CheckpointReport | Any,
        max_chunk_bytes: int,
        source_metadata: Mapping[str, str] | None = None,
        manifest_schema: str = "glm53-streaming-rank-v1",
        plan_contract_sha256: str | None = None,
    ) -> None:
        if max_chunk_bytes <= 0:
            raise ValueError("max_chunk_bytes must be positive")
        self.output_path = Path(output_path)
        self.inventory = inventory
        self.source_report = source_report
        self.max_chunk_bytes = max_chunk_bytes
        self.source_metadata = dict(
            source_metadata
            or {
                "model": "GLM-5.3-Flash",
                "revision": GLM53_CHECKPOINT_REVISION,
                "config_sha256": GLM53_CONFIG_SHA256,
                "index_sha256": GLM53_INDEX_SHA256,
            }
        )
        required_metadata = {"model", "revision", "config_sha256", "index_sha256"}
        if set(self.source_metadata) != required_metadata or not all(
            isinstance(value, str) and value for value in self.source_metadata.values()
        ):
            raise ValueError(
                "source_metadata must contain exactly model, revision, "
                "config_sha256, and index_sha256"
            )
        if not manifest_schema:
            raise ValueError("manifest_schema must be non-empty")
        self.manifest_schema = manifest_schema
        if plan_contract_sha256 is not None and (
            len(plan_contract_sha256) != 64
            or any(char not in "0123456789abcdef" for char in plan_contract_sha256)
        ):
            raise ValueError("plan_contract_sha256 must be a lowercase SHA-256")
        self.plan_contract_sha256 = plan_contract_sha256
        if self.output_path.exists():
            raise FileExistsError(
                f"refusing to overwrite rank checkpoint: {self.output_path}"
            )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        self.partial_path = self.output_path.with_name(
            f".{self.output_path.name}.partial-{token}"
        )
        self.manifest_path = self.output_path.with_suffix(
            self.output_path.suffix + ".manifest.json"
        )
        if self.manifest_path.exists():
            raise FileExistsError(
                f"refusing to overwrite rank manifest: {self.manifest_path}"
            )

        self._specs = {spec.name: spec for spec in inventory.tensors}
        self._offsets: dict[str, tuple[int, int]] = {}
        self._coverage: dict[str, list[tuple[int, int]]] = {
            spec.name: [] for spec in inventory.tensors
        }
        self.chunks_written = 0
        self.max_observed_chunk_bytes = 0
        self._closed = False
        self._published = False
        self._handle = self.partial_path.open("x+b")
        self._data_start = self._write_header_and_reserve()

    def _write_header_and_reserve(self) -> int:
        cursor = 0
        header: dict[str, Any] = {
            "__metadata__": {
                "format": "pt",
                **self.source_metadata,
                "rank": str(self.inventory.rank),
                "tp_degree": str(self.inventory.tp_degree),
                "plan_contract_sha256": self.plan_contract_sha256 or "unbound",
            }
        }
        for spec in self.inventory.tensors:
            start, end = cursor, cursor + spec.nbytes
            self._offsets[spec.name] = (start, end)
            header[spec.name] = {
                "dtype": _TORCH_TO_SAFE_DTYPE[spec.dtype],
                "shape": list(spec.shape),
                "data_offsets": [start, end],
            }
            cursor = end
        encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        padding = (-len(encoded)) % 8
        encoded += b" " * padding
        self._handle.write(struct.pack("<Q", len(encoded)))
        self._handle.write(encoded)
        data_start = 8 + len(encoded)
        self._handle.truncate(data_start + cursor)
        return data_start

    def write_chunk(self, chunk: TensorChunk) -> None:
        if self._closed:
            raise Glm53StreamingError("rank writer is closed")
        spec = self._specs.get(chunk.tensor_name)
        if spec is None:
            raise Glm53StreamingError(
                f"chunk targets tensor outside rank inventory: {chunk.tensor_name}"
            )
        tensor = chunk.tensor.detach()
        if tensor.device.type != "cpu":
            raise Glm53StreamingError("rank chunks must be CPU tensors")
        if tensor.dtype != spec.dtype:
            raise Glm53StreamingError(
                f"dtype mismatch for {spec.name}: expected {spec.dtype}, got {tensor.dtype}"
            )
        tensor = tensor.contiguous()
        chunk_bytes = tensor.numel() * tensor.element_size()
        if chunk_bytes <= 0 or chunk_bytes > self.max_chunk_bytes:
            raise Glm53StreamingError(
                f"chunk size for {spec.name} is {chunk_bytes} bytes; bound is {self.max_chunk_bytes}"
            )
        if chunk.start_element < 0:
            raise Glm53StreamingError("chunk start_element must be non-negative")
        byte_start = chunk.start_element * spec.element_size
        byte_end = byte_start + chunk_bytes
        if byte_end > spec.nbytes:
            raise Glm53StreamingError(
                f"chunk exceeds tensor {spec.name}: [{byte_start},{byte_end}) > {spec.nbytes}"
            )
        intervals = self._coverage[spec.name]
        if any(byte_start < end and byte_end > start for start, end in intervals):
            raise Glm53StreamingError(
                f"overlapping chunk for {spec.name}: [{byte_start},{byte_end})"
            )
        tensor_offset = self._offsets[spec.name][0]
        self._handle.seek(self._data_start + tensor_offset + byte_start)
        byte_view = tensor.view(torch.uint8).numpy()
        self._handle.write(memoryview(byte_view).cast("B"))
        intervals.append((byte_start, byte_end))
        intervals.sort()
        self.chunks_written += 1
        self.max_observed_chunk_bytes = max(self.max_observed_chunk_bytes, chunk_bytes)

    def _assert_complete(self) -> None:
        incomplete: list[str] = []
        for spec in self.inventory.tensors:
            intervals = self._coverage[spec.name]
            cursor = 0
            for start, end in intervals:
                if start != cursor:
                    break
                cursor = end
            if cursor != spec.nbytes:
                incomplete.append(f"{spec.name}:{cursor}/{spec.nbytes}")
        if incomplete:
            raise Glm53StreamingError(
                f"rank inventory is incomplete: {', '.join(incomplete[:8])}"
            )

    def finalize(
        self, *, source_reader: IndexedTensorReader | None = None
    ) -> dict[str, Any]:
        if self._closed:
            raise Glm53StreamingError("rank writer is already closed")
        try:
            self._assert_complete()
        except Exception:
            self.abort()
            raise
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._closed = True
        checkpoint_sha256 = _sha256_file(self.partial_path)
        os.replace(self.partial_path, self.output_path)
        self._published = True
        manifest = {
            "schema": self.manifest_schema,
            "source": {
                **self.source_metadata,
                "preflight": self.source_report.to_dict(),
            },
            "rank": self.inventory.rank,
            "tp_degree": self.inventory.tp_degree,
            "rank_inventory_sha256": self.inventory.contract_sha256,
            "rank_plan_sha256": self.plan_contract_sha256,
            "checkpoint": {
                "path": self.output_path.name,
                "sha256": checkpoint_sha256,
                "bytes": self.output_path.stat().st_size,
            },
            "resource_bound": {
                "configured_max_chunk_bytes": self.max_chunk_bytes,
                "observed_max_chunk_bytes": self.max_observed_chunk_bytes,
                "full_rank_tensor_bytes": self.inventory.total_tensor_bytes,
                "chunks_written": self.chunks_written,
                "source_max_group_bytes": (
                    source_reader.max_source_group_bytes if source_reader else None
                ),
            },
            "source_audit": (
                asdict(source_reader.audit_report) if source_reader else None
            ),
            "tensors": [
                {
                    **asdict(spec),
                    "dtype": str(spec.dtype),
                    "nbytes": spec.nbytes,
                    "chunks": len(self._coverage[spec.name]),
                }
                for spec in self.inventory.tensors
            ],
        }
        manifest_tmp = self.manifest_path.with_name(
            f".{self.manifest_path.name}.partial-{uuid.uuid4().hex}"
        )
        manifest_tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(manifest_tmp, self.manifest_path)
        return manifest

    def abort(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True
        if not self._published:
            self.partial_path.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self._published:
            self.abort()


def stream_rank_checkpoint(
    checkpoint_dir: str | Path,
    output_path: str | Path,
    inventory: RankInventory,
    chunk_factory: Callable[[IndexedTensorReader], Iterable[TensorChunk]],
    *,
    max_chunk_bytes: int,
) -> dict[str, Any]:
    """Audit the source and transactionally stream one complete TP rank."""
    reader = IndexedTensorReader(checkpoint_dir)
    with StreamingRankWriter(
        output_path,
        inventory,
        source_report=reader.preflight_report,
        max_chunk_bytes=max_chunk_bytes,
    ) as writer:
        for chunk in chunk_factory(reader):
            writer.write_chunk(chunk)
        return writer.finalize(source_reader=reader)


__all__ = [
    "Glm53StreamingError",
    "IndexedTensorReader",
    "RankInventory",
    "SourceAuditReport",
    "SourceTensorSpec",
    "StreamingRankWriter",
    "TensorChunk",
    "TensorSpec",
    "stream_rank_checkpoint",
]
