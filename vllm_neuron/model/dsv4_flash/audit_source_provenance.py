#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit a pinned DSv4 checkpoint without reading tensor payload bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
CONFIG_SHA256 = "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
INDEX_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
TOKENIZER_SHA256 = "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"
SOURCE_COMMIT = "266af97d16cb8621514d6ae77741ce17ece40d9c"
VALIDATOR_MERGE_SHA = "2543fe18c70f7d92a5bf498e4adaccacde425728"
VALIDATOR_MERGE_TREE_SHA = "e51e1e648726a054b8906b6b2338b7fb75b795ea"
EXPECTED_SHARDS = 48
EXPECTED_TENSORS = 72317
EXPECTED_PAYLOAD_BYTES = 166886535336
EXPECTED_PAYLOAD_MANIFEST_SHA256 = (
    "ac05cbd738cb8866595257aec855bdb231b677280bbdcb6bd65950c356f68a8d"
)
SHA_LINE = re.compile(r"^([0-9a-f]{64})  \./([^/]+)$")


class AuditError(ValueError):
    """The checkpoint or its prior payload-hash receipt is inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_payload_manifest(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    require(
        sha256_bytes(raw) == EXPECTED_PAYLOAD_MANIFEST_SHA256,
        "payload manifest identity drift",
    )
    entries: dict[str, str] = {}
    for line in raw.decode("utf-8").splitlines():
        match = SHA_LINE.fullmatch(line)
        require(match is not None, f"malformed payload manifest line: {line!r}")
        digest, name = match.groups()
        require(name not in entries, f"duplicate payload manifest entry: {name}")
        entries[name] = digest
    return entries


def read_header(path: Path) -> tuple[bytes, dict[str, Any]]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, f"truncated SafeTensors prefix: {path.name}")
        (length,) = struct.unpack("<Q", prefix)
        require(
            2 <= length <= path.stat().st_size - 8,
            f"invalid header length: {path.name}",
        )
        body = stream.read(length)
        require(len(body) == length, f"truncated SafeTensors header: {path.name}")
    try:
        header = json.loads(body, object_pairs_hook=unique_object)
    except json.JSONDecodeError as error:
        raise AuditError(f"invalid SafeTensors header JSON: {path.name}") from error
    require(isinstance(header, dict), f"header is not an object: {path.name}")
    return prefix + body, header


def tensor_keys_and_extent(
    header: dict[str, Any], payload_bytes: int, shard_name: str
) -> tuple[set[str], int]:
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    keys = set(tensors)
    require(len(keys) == len(tensors), f"duplicate JSON keys in {shard_name}")
    intervals: list[tuple[int, int, str]] = []
    for key, descriptor in tensors.items():
        require(isinstance(descriptor, dict), f"bad descriptor for {key}")
        require(
            set(descriptor) == {"dtype", "shape", "data_offsets"},
            f"descriptor fields drift for {key}",
        )
        require(isinstance(descriptor["dtype"], str), f"bad dtype for {key}")
        shape = descriptor["shape"]
        require(
            isinstance(shape, list)
            and all(type(dim) is int and dim >= 0 for dim in shape),
            f"bad shape for {key}",
        )
        offsets = descriptor["data_offsets"]
        require(
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(type(offset) is int for offset in offsets),
            f"bad offsets for {key}",
        )
        start, end = offsets
        require(0 <= start <= end <= payload_bytes, f"out-of-range offsets for {key}")
        intervals.append((start, end, key))
    intervals.sort()
    previous_end = 0
    for start, end, key in intervals:
        require(start >= previous_end, f"overlapping payload interval at {key}")
        previous_end = end
    require(previous_end == payload_bytes, f"payload extent drift in {shard_name}")
    return keys, previous_end


def audit(
    model_dir: Path,
    payload_manifest: Path,
    output_dir: Path,
    audit_tool_sha256: str,
) -> dict[str, Any]:
    model_dir = model_dir.resolve(strict=True)
    require(
        model_dir.name.endswith("--7872f01b1d1fe23"),
        "checkpoint directory revision suffix drift",
    )
    complete = model_dir / ".complete"
    require(
        complete.is_file() and complete.stat().st_size == 0,
        "missing or non-empty .complete marker",
    )
    expected_small = {
        "config.json": CONFIG_SHA256,
        "model.safetensors.index.json": INDEX_SHA256,
        "tokenizer.json": TOKENIZER_SHA256,
    }
    model_files: dict[str, dict[str, int | str]] = {}
    for name, expected in expected_small.items():
        path = model_dir / name
        require(path.is_file(), f"missing model file: {name}")
        actual = sha256_file(path)
        require(actual == expected, f"model file hash drift: {name}")
        model_files[name] = {"sha256": actual, "bytes": path.stat().st_size}

    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
    )
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict), "index weight_map missing")
    require(len(weight_map) == EXPECTED_TENSORS, "index tensor census drift")
    require(
        all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in weight_map.items()
        ),
        "malformed index route",
    )
    expected_names = [
        f"model-{index:05d}-of-00048.safetensors" for index in range(1, 49)
    ]
    require(
        sorted(set(weight_map.values())) == expected_names,
        "index shard inventory drift",
    )
    actual_names = sorted(
        path.name for path in model_dir.glob("model-*-of-00048.safetensors")
    )
    require(actual_names == expected_names, "actual shard inventory drift")

    payload_hashes = load_payload_manifest(payload_manifest)
    require(
        all(name in payload_hashes for name in expected_names),
        "payload manifest lacks a shard",
    )
    expected_by_shard: dict[str, set[str]] = defaultdict(set)
    for key, name in weight_map.items():
        expected_by_shard[name].add(key)

    shards: list[dict[str, Any]] = []
    seen: set[str] = set()
    missing: list[str] = []
    orphan: list[str] = []
    misrouted: list[str] = []
    total_bytes = 0
    for name in expected_names:
        path = model_dir / name
        size = path.stat().st_size
        raw_header, header = read_header(path)
        header_bytes = len(raw_header)
        actual_keys, payload_extent = tensor_keys_and_extent(
            header, size - header_bytes, name
        )
        expected_keys = expected_by_shard[name]
        missing.extend(sorted(expected_keys - actual_keys))
        for key in sorted(actual_keys - expected_keys):
            if key in weight_map:
                misrouted.append(key)
            else:
                orphan.append(key)
        require(not (seen & actual_keys), f"tensor duplicated across headers: {name}")
        seen.update(actual_keys)
        route_identity = {"shard": name, "tensor_names": sorted(actual_keys)}
        shards.append(
            {
                "name": name,
                "lfs_sha256": payload_hashes[name],
                "size": size,
                "header_sha256": sha256_bytes(raw_header),
                "header_bytes": header_bytes,
                "tensor_count": len(actual_keys),
                "routing_sha256": canonical_sha256(route_identity),
            }
        )
        require(
            payload_extent == size - header_bytes, f"payload extent mismatch: {name}"
        )
        total_bytes += size

    require(total_bytes == EXPECTED_PAYLOAD_BYTES, "payload byte total drift")
    require(len(seen) == EXPECTED_TENSORS, "header tensor census drift")
    require(seen == set(weight_map), "header/index identity drift")
    require(not missing and not orphan and not misrouted, "routing audit failed")
    require(
        len({row["lfs_sha256"] for row in shards}) == EXPECTED_SHARDS,
        "duplicate payload identities",
    )
    require(
        len({row["header_sha256"] for row in shards}) == EXPECTED_SHARDS,
        "duplicate header identities",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    copied_manifest = output_dir / "checkpoint-all-files.sha256"
    copied_manifest.write_bytes(payload_manifest.read_bytes())
    routing = {"revision": REVISION, "index_sha256": INDEX_SHA256, "shards": shards}
    routing_path = output_dir / "routing-manifest.json"
    routing_path.write_text(
        json.dumps(routing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    source = {
        "source_commit": SOURCE_COMMIT,
        "validator_merge_sha": VALIDATOR_MERGE_SHA,
        "validator_merge_tree_sha": VALIDATOR_MERGE_TREE_SHA,
        "validator_merged_to_agent_main": True,
        "revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "model_files": model_files,
        "shards": shards,
        "payload_bytes_read_during_header_audit": 0,
        "missing": missing,
        "orphan": orphan,
        "misrouted": misrouted,
        "canonical_shard_inventory_sha256": canonical_sha256(shards),
        "routing_manifest": {
            "path": routing_path.name,
            "sha256": sha256_file(routing_path),
        },
        "payload_manifest": {
            "path": copied_manifest.name,
            "sha256": sha256_file(copied_manifest),
            "entry_count": len(payload_hashes),
            "shard_entries": EXPECTED_SHARDS,
            "shard_bytes": total_bytes,
            "prior_hash_verification": "48/48_PASS",
        },
        "complete_marker": {
            "name": ".complete",
            "bytes": 0,
            "sha256": sha256_file(complete),
        },
        "audit": {
            "tool_path": "audit_source_provenance.py",
            "tool_sha256": audit_tool_sha256,
            "method": "safetensors_8_byte_prefix_plus_json_header_only",
            "tensor_payload_bytes_read": 0,
            "rank_conversion_performed": False,
            "large_outputs_created": False,
        },
    }
    source_path = output_dir / "source-provenance.json"
    source_path.write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "source_provenance_sha256": sha256_file(source_path),
        "routing_manifest_sha256": sha256_file(routing_path),
        "payload_manifest_sha256": sha256_file(copied_manifest),
        "canonical_shard_inventory_sha256": source["canonical_shard_inventory_sha256"],
        "shards": len(shards),
        "tensors": len(seen),
        "payload_bytes": total_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--payload-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-tool-sha256", required=True)
    args = parser.parse_args()
    try:
        result = audit(
            args.model_dir,
            args.payload_manifest,
            args.output_dir,
            args.audit_tool_sha256,
        )
    except (AuditError, OSError, json.JSONDecodeError) as error:
        print(f"REFUSED: {error}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
