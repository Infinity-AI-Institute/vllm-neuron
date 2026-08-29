# SPDX-License-Identifier: Apache-2.0
"""Header-only DeepSeek-V4-Flash TP32 target-rank inventory audit.

The audit reads the exact SafeTensors headers and immutable index, never tensor
payloads.  It models the merged converter's TP ownership rules and fails closed
when source tensors cannot be routed into the merged wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
CONFIG_SHA256 = "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
INDEX_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
SOURCE_RECEIPT_SHA256 = (
    "6d8adb42de7664cbc2f76fb481aaab12ca1ba8deded4d9fb8daabb40f788a965"
)
ROUTING_RECEIPT_SHA256 = (
    "93c516df53d1fc5e88e77acb043f68c9d37b7eb12a75f70d8469ab284a663547"
)
VALIDATOR_MERGE = "157abe3661fec04cbdf69501aae23ed7fe22b93c"
VALIDATOR_MERGE_TREE = "827993165838800aaa01eb1334aa2c713ca9bfae"
TP = 32
SHARDS = 48
SOURCE_TENSORS = 72317
MAX_CHUNK_BYTES = 64 * 1024 * 1024
SAFE_BYTES = {
    "BF16": 2,
    "F32": 4,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "I32": 4,
    "I64": 8,
}


class RankPlanError(ValueError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise RankPlanError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


@dataclass(frozen=True)
class HeaderSpec:
    dtype: str
    shape: tuple[int, ...]
    shard: str

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * SAFE_BYTES[self.dtype]


@dataclass(frozen=True)
class TargetSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    ownership: str

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * {"BF16": 2, "F32": 4, "I32": 4}[self.dtype]

    def canonical(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "ownership": self.ownership,
        }


def read_headers(
    root: Path, *, test_only_allow_unpinned: bool = False
) -> tuple[dict[str, str], dict[str, HeaderSpec], int]:
    if not test_only_allow_unpinned:
        require(
            root.name.endswith("7872f01b1d1fe23"),
            "checkpoint directory identity drift",
        )
    require(sha256(root / "config.json") == CONFIG_SHA256, "config identity drift")
    index_path = root / "model.safetensors.index.json"
    require(sha256(index_path) == INDEX_SHA256, "index identity drift")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict), "index weight_map missing")
    require(len(weight_map) == SOURCE_TENSORS, "source tensor count drift")
    shard_names = sorted(set(weight_map.values()))
    require(len(shard_names) == SHARDS, "source shard count drift")
    headers: dict[str, HeaderSpec] = {}
    header_bytes = 0
    for shard in shard_names:
        path = root / shard
        require(path.is_file(), f"missing shard: {shard}")
        with path.open("rb") as stream:
            length_raw = stream.read(8)
            require(len(length_raw) == 8, f"truncated header length: {shard}")
            length = struct.unpack("<Q", length_raw)[0]
            raw = stream.read(length)
        header_bytes += 8 + length
        header = json.loads(raw, object_pairs_hook=reject_duplicate_json_keys)
        actual = set(header) - {"__metadata__"}
        expected = {key for key, route in weight_map.items() if route == shard}
        require(actual == expected, f"index/header routing drift: {shard}")
        for key in sorted(actual):
            require(key not in headers, f"duplicate source key: {key}")
            row = header[key]
            dtype = row["dtype"]
            require(
                dtype in SAFE_BYTES, f"unsupported SafeTensors dtype: {key}={dtype}"
            )
            shape = tuple(row["shape"])
            offsets = row["data_offsets"]
            require(
                offsets[1] - offsets[0] == math.prod(shape) * SAFE_BYTES[dtype],
                f"header byte-count drift: {key}",
            )
            headers[key] = HeaderSpec(dtype, shape, shard)
    require(set(headers) == set(weight_map), "missing or orphan source keys")
    return weight_map, headers, header_bytes


def build_targets() -> list[TargetSpec]:
    hidden, vocab, heads, head_dim = 4096, 129280, 64, 512
    q_lora, o_groups, o_lora = 1024, 8, 1024
    experts, inter, top_k = 256, 2048, 6
    targets: list[TargetSpec] = []

    def add(name: str, dtype: str, shape: tuple[int, ...], ownership: str) -> None:
        targets.append(TargetSpec(name, dtype, shape, ownership))

    add("embed_tokens.weight", "BF16", (vocab // TP, hidden), "tp_shard_dim0")
    add("final_norm_weight", "BF16", (hidden,), "replicated")
    add("lm_head.weight", "BF16", (vocab // TP, hidden), "tp_shard_dim0")
    mqa = {
        "wq_a.weight": (q_lora // TP, hidden, "tp_shard_dim0"),
        "wq_b.weight": (heads * head_dim // TP, q_lora, "tp_shard_dim0"),
        "wkv.weight": (head_dim, hidden, "replicated"),
        "wo_a.weight": (
            o_groups * o_lora // TP,
            (heads * head_dim) // o_groups,
            "tp_shard_dim0",
        ),
        "wo_b.weight": (hidden, o_groups * o_lora // TP, "tp_shard_dim1"),
        "q_norm.weight": (q_lora, "replicated"),
        "kv_norm.weight": (head_dim, "replicated"),
        "attn_sink": (heads, "replicated"),
    }
    for layer in range(43):
        prefix = f"layers.{layer}."
        for name, values in mqa.items():
            *shape, ownership = values
            add(f"{prefix}attn.mqa.{name}", "BF16", tuple(shape), ownership)
        add(f"{prefix}attn_norm.weight", "BF16", (hidden,), "replicated")
        if layer >= 2 and layer % 2 == 0:
            ratio = 4
            for name, shape in {
                "wkv.weight": (2 * head_dim, hidden),
                "wgate.weight": (2 * head_dim, hidden),
                "ape": (ratio, 2 * head_dim),
                "norm.weight": (head_dim,),
            }.items():
                add(f"{prefix}attn.compressor.{name}", "BF16", shape, "replicated")
            for name, shape in {
                "wkv.weight": (256, hidden),
                "wgate.weight": (256, hidden),
                "ape": (ratio, 256),
                "norm.weight": (128,),
            }.items():
                add(
                    f"{prefix}attn.indexer.compressor.{name}",
                    "BF16",
                    shape,
                    "replicated",
                )
            add(
                f"{prefix}attn.indexer.weights_proj.weight",
                "BF16",
                (64, hidden),
                "replicated",
            )
            add(
                f"{prefix}attn.indexer.wq_b.weight",
                "BF16",
                (8192, q_lora),
                "replicated",
            )
        elif layer >= 3:
            ratio = 128
            for name, shape in {
                "wkv.weight": (head_dim, hidden),
                "wgate.weight": (head_dim, hidden),
                "ape": (ratio, head_dim),
                "norm.weight": (head_dim,),
            }.items():
                add(f"{prefix}attn.compressor.{name}", "BF16", shape, "replicated")
        add(f"{prefix}ffn_norm.weight", "BF16", (hidden,), "replicated")
        add(f"{prefix}mlp.router.weight", "F32", (experts, hidden), "replicated_router")
        if layer < 3:
            add(f"{prefix}mlp.tid2eid", "I32", (vocab, top_k), "replicated_hash_route")
        else:
            add(
                f"{prefix}mlp.e_score_correction_bias",
                "F32",
                (experts,),
                "replicated_router",
            )
        add(
            f"{prefix}mlp.shared_expert.gate_proj.weight",
            "BF16",
            (inter // TP, hidden),
            "tp_shard_dim0",
        )
        add(
            f"{prefix}mlp.shared_expert.up_proj.weight",
            "BF16",
            (inter // TP, hidden),
            "tp_shard_dim0",
        )
        add(
            f"{prefix}mlp.shared_expert.down_proj.weight",
            "BF16",
            (hidden, inter // TP),
            "tp_shard_dim1",
        )
        add(
            f"{prefix}mlp.expert_mlps.mlp_op.gate_up_proj.weight",
            "BF16",
            (experts, hidden, 2 * inter // TP),
            "expert_axis_replicated_intermediate_tp_sharded",
        )
        add(
            f"{prefix}mlp.expert_mlps.mlp_op.down_proj.weight",
            "BF16",
            (experts, inter // TP, hidden),
            "expert_axis_replicated_intermediate_tp_sharded",
        )
    require(len(targets) == 1024, "target tensor count drift")
    require(len({item.name for item in targets}) == len(targets), "duplicate targets")
    return targets


def classify_sources(headers: dict[str, HeaderSpec]) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {
        "routable": [],
        "support_scale": [],
        "dropped_mtp_or_speculation": [],
        "unmapped_mhc": [],
        "incompatible_hash_route_dtype": [],
        "orphan": [],
    }
    top = {"embed.weight", "norm.weight", "head.weight"}
    for key, spec in sorted(headers.items()):
        if key.startswith(("mtp.", "layers.43.")) or "dspark" in key:
            categories["dropped_mtp_or_speculation"].append(key)
        elif "hc_" in key:
            categories["unmapped_mhc"].append(key)
        elif key.endswith("ffn.gate.tid2eid") and spec.dtype != "I32":
            categories["incompatible_hash_route_dtype"].append(key)
        elif key.endswith(".scale"):
            weight = f"{key[: -len('.scale')]}.weight"
            if weight in headers:
                categories["support_scale"].append(key)
            else:
                categories["orphan"].append(key)
        elif key in top or key.startswith("layers."):
            categories["routable"].append(key)
        else:
            categories["orphan"].append(key)
    return categories


def audit(root: Path, tool_sha256: str) -> dict[str, Any]:
    weight_map, headers, header_bytes = read_headers(root)
    targets = build_targets()
    categories = classify_sources(headers)
    blockers = {
        key: values
        for key, values in categories.items()
        if key in {"unmapped_mhc", "incompatible_hash_route_dtype", "orphan"} and values
    }
    target_rows = [item.canonical() for item in targets]
    inventory_bytes = sum(item.nbytes for item in targets)
    ownership = Counter(item.ownership for item in targets)
    rank_rows = []
    for rank in range(TP):
        rank_rows.append(
            {
                "rank": rank,
                "tensor_count": len(targets),
                "expected_bytes": inventory_bytes,
                "inventory_sha256": canonical_sha256(
                    {"rank": rank, "tp_degree": TP, "tensors": target_rows}
                ),
            }
        )
    return {
        "schema": "dsv4-tp32-header-rank-plan-audit-v1",
        "status": "HOLD_UNROUTED_SOURCE_CONTRACT" if blockers else "PASS_RANK_PLAN",
        "complete": not blockers,
        "compile_permitted": False,
        "source": {
            "revision": REVISION,
            "config_sha256": CONFIG_SHA256,
            "index_sha256": INDEX_SHA256,
            "validator_merge": VALIDATOR_MERGE,
            "validator_merge_tree": VALIDATOR_MERGE_TREE,
            "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
            "routing_receipt_sha256": ROUTING_RECEIPT_SHA256,
            "tool_sha256": tool_sha256,
            "shard_count": len(set(weight_map.values())),
            "tensor_count": len(headers),
            "header_bytes_read": header_bytes,
            "tensor_payload_bytes_read": 0,
            "header_inventory_sha256": canonical_sha256(
                {
                    key: {
                        "dtype": spec.dtype,
                        "shape": list(spec.shape),
                        "shard": spec.shard,
                    }
                    for key, spec in sorted(headers.items())
                }
            ),
        },
        "routing": {
            "source_integrity": {
                "index_missing_keys": 0,
                "header_orphan_keys": 0,
                "duplicate_keys": 0,
                "misrouted_keys": 0,
            },
            "source_category_counts": {
                key: len(value) for key, value in categories.items()
            },
            "source_category_sha256": {
                key: canonical_sha256(value) for key, value in categories.items()
            },
            "target_tensor_count_per_rank": len(targets),
            "target_bytes_per_rank": inventory_bytes,
            "target_plan_sha256": canonical_sha256(target_rows),
            "ownership_counts": dict(sorted(ownership.items())),
            "moe_ownership": {
                "expert_count": 256,
                "expert_axis": "replicated_on_all_32_ranks",
                "expert_axis_partitioned_by_ep": False,
                "intermediate_size": 2048,
                "local_intermediate_per_rank": 64,
                "router_and_hash_or_bias_state": "replicated",
            },
            "max_writer_chunk_bytes": MAX_CHUNK_BYTES,
        },
        "ranks": rank_rows,
        "blockers": {
            key: {
                "count": len(value),
                "keys_sha256": canonical_sha256(value),
                "examples": value[:8],
            }
            for key, value in blockers.items()
        },
        "claims": {
            "rank_conversion_performed": False,
            "rank_files_materialized": False,
            "neuron_compile": False,
            "trn2_used": False,
            "runtime_correctness": False,
            "performance": False,
            "tokenomics": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tool-sha256", required=True)
    args = parser.parse_args()
    try:
        result = audit(args.checkpoint.resolve(strict=True), args.tool_sha256)
    except (OSError, KeyError, TypeError, json.JSONDecodeError, RankPlanError) as error:
        print(f"REFUSED: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if not result["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
