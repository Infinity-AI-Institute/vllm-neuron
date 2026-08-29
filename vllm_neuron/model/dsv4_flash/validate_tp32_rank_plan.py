# SPDX-License-Identifier: Apache-2.0
"""Fail-closed validator for the DeepSeek-V4 TP32 symbolic rank-plan receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "evidence/tp32-rank-plan-audit.json"
AUDIT_TOOL = HERE / "audit_tp32_rank_plan.py"

RANK_HASHES = (
    "cb2651152e2fca2e3a82807a0d29a23dd49dd0c3b1a4ac78f4ee0e8d8b8b0a85",
    "6aa298c57e25903cfc5fc4c9a661d51a06fe6d744bd0729b8b6e5aadde49baa6",
    "48fd8eb2448076f24f2253b284e2ef13ae191507234a5d7816b1e5ee338d732e",
    "4bc4ed484825166ce331724fba8205e276f1084b9e1613c0f12fa372eeb172f4",
    "89597cb18e5131ced960f1bed414d495b9528577ea4876b2481b987e644f5296",
    "157dce58d87efe6b8a4e8ff53de1ee4773ed690d17283e9dfe6c285b12dc30da",
    "656f4d5cab547056c02fffbb45850e9f35c8b10c8569fc880dad1607a9386a0c",
    "635dc57e63a120c0fa33d8011f74dcc968dc6fdd0e449643720829fc76364f1f",
    "87b3bec1ee0b2d8e1f6017440a5ca69299fd73d78ec579530dee4ce63168fdac",
    "8379c16fabbdcc7d0883c5962950630ff6b23f574fbcaaa57d24720d40235633",
    "7437a61d047b8e1295074b7426dc18f94f6b01b5e12135a8be44b17a98fed185",
    "8fafc41381949a15fc99fa9177f479624b7c052bffe9b76fc9f6e1ef6c2a5d4f",
    "04469b1a9eaefbeb63bd367e101a18a881eb1f11513aec2622edadf230697340",
    "67ab4b27507186d4da851867cc5396f732a080c2dd235103efa540f4d4eb505c",
    "54b9dfbd5b7938366cfc0b12c52ea671bf4957664a9aa336d5533dbb1bfc53a5",
    "3684a2566e2a4508c3627fa11e88fcdabc267560f6606e8497f3d113c71a85fd",
    "ae3d0706f92300a645ed05bb04f715cdaa896985b386c87c3b8d87cc9d7f7635",
    "864eaaaa6a0a51496225ff44f9cada0f59eb00ba3ccf14ba25170162c95f5bfb",
    "e58c1c47da0e94756baea860d3ea37571328dcbcdfbf493be9fe2196bc40b2e1",
    "7cb9da41bee81bbe528dcda93ea730e9f427382d9efc117d030c2dd70a43db32",
    "33fcefd119700cde8353501c098b695046fc80142b2797be9d265597a065244f",
    "7b4d365195d208af46d49eb67fa3060783a2f9fe3be89f1bba7f716c1d041537",
    "51e8d5113aa7276b43470ac629396f1e2cce8a392794bc9f2dde4e06c2679c93",
    "f070012caefb150ed4eb65f528611d557d839da0f2f4fe3fa7dea5f3dbc14735",
    "2fcfd4fcf15e28dc2816d1e13caac44f9c62d1fc5e5783f05674c33180c8710a",
    "b3b65c0b8a101df2df4319bd37a031f49b426d8864460ed1262d35f9e409ab5a",
    "961dcfa82477de2fdbacfa10b512ba313dccf270ff0d67ca7041f19c1f7a0fb0",
    "978b4e40adeeb297e3f7ce68a3557a290dab181805b6ead301d533ffe0ae4b88",
    "2cb902c677a71ad0062e46443392eadf849088138432c566d79f33bb82791b67",
    "5890e89d7fba4cad399f72cad81ae2a3afdcca59b5934f8b02a4aa92ac1accad",
    "84695008aa3960918935c0223b961a214b4b848ed0c27266fa44c8161b1ee894",
    "7fba49815e624c7757db51089c3e26bb45f7d030290970e7b6e8236af4f70e7d",
)


class RankPlanValidationError(ValueError):
    """The symbolic receipt is incomplete, mutated, or overclaims readiness."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RankPlanValidationError(message)


def _sha256(path: Path) -> str:
    # Git stores these reviewed text tools with LF. Normalize a Windows
    # autocrlf checkout back to the canonical blob bytes before binding it.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_receipt(receipt: Mapping[str, Any], tool_path: Path = AUDIT_TOOL) -> None:
    """Require the exact reviewed host-only routed plan and every rank identity."""
    _require(
        set(receipt)
        == {
            "schema",
            "status",
            "complete",
            "compile_permitted",
            "source",
            "routing",
            "ranks",
            "blockers",
            "claims",
        },
        "receipt field drift",
    )
    _require(receipt["schema"] == "dsv4-tp32-header-rank-plan-audit-v1", "schema drift")
    _require(receipt["status"] == "PASS_RANK_PLAN", "status drift")
    _require(receipt["complete"] is True, "symbolic rank plan must be complete")
    _require(receipt["compile_permitted"] is False, "compile must remain forbidden")

    source = receipt["source"]
    expected_source = {
        "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        "config_sha256": "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
        "index_sha256": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
        "validator_merge": "157abe3661fec04cbdf69501aae23ed7fe22b93c",
        "validator_merge_tree": "827993165838800aaa01eb1334aa2c713ca9bfae",
        "source_receipt_sha256": "6d8adb42de7664cbc2f76fb481aaab12ca1ba8deded4d9fb8daabb40f788a965",
        "routing_receipt_sha256": "93c516df53d1fc5e88e77acb043f68c9d37b7eb12a75f70d8469ab284a663547",
        "tool_sha256": "5cefb3a2e6ba4be617aa607ecaabec4ad79b9c08d2bd6750c2ced07facd45e15",
        "shard_count": 48,
        "tensor_count": 72317,
        "header_bytes_read": 7998896,
        "tensor_payload_bytes_read": 0,
        "header_inventory_sha256": "762f20dc104c926e30026c2fad1e096a82b1fcac582cf4a2e5c58c4ca63bec57",
    }
    _require(source == expected_source, "source identity or census drift")
    _require(
        tool_path.is_file() and _sha256(tool_path) == source["tool_sha256"],
        "audit tool drift",
    )

    routing = receipt["routing"]
    _require(
        routing
        == {
            "source_integrity": {
                "index_missing_keys": 0,
                "header_orphan_keys": 0,
                "duplicate_keys": 0,
                "misrouted_keys": 0,
            },
            "source_category_counts": {
                "converted_hash_route_i64_to_i32": 3,
                "dropped_mtp_or_speculation": 4705,
                "incompatible_hash_route_dtype": 0,
                "orphan": 0,
                "replicated_mhc": 261,
                "routable": 33959,
                "support_scale": 33389,
            },
            "source_category_sha256": {
                "dropped_mtp_or_speculation": "8222ff3f67bb3eceb230d36f787506416629a0beafaaf1ee1f56ab19a9cf5e9f",
                "converted_hash_route_i64_to_i32": "36e2d1f0d5a59f9565d376ad8d7e1f53e68d461ca1939c6b6758b667e78851b2",
                "incompatible_hash_route_dtype": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                "orphan": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                "replicated_mhc": "ad07358ebd20fce2a30d3ddd889880250d5280c523eb474a226a2982833d58d6",
                "routable": "cd37aeea3bf2dc21d5155c60ef7019a6a2923117562db38ba4d2e5bf342d84b5",
                "support_scale": "d8a59e9afc77307fd08f536139638853f27da375dc18df7a7b0bbe2ad22f4023",
            },
            "target_tensor_count_per_rank": 1285,
            "target_bytes_per_rank": 19210553052,
            "target_plan_sha256": "ac020ab0a03693273aff2525d88e2cf23486eb324c12d9bbeccf6f44ab728bc3",
            "ownership_counts": {
                "expert_axis_replicated_intermediate_tp_sharded": 86,
                "replicated": 549,
                "replicated_hash_route": 3,
                "replicated_mhc_fp32": 261,
                "replicated_router": 83,
                "tp_shard_dim0": 217,
                "tp_shard_dim1": 86,
            },
            "moe_ownership": {
                "expert_count": 256,
                "expert_axis": "replicated_on_all_32_ranks",
                "expert_axis_partitioned_by_ep": False,
                "intermediate_size": 2048,
                "local_intermediate_per_rank": 64,
                "router_and_hash_or_bias_state": "replicated",
            },
            "max_writer_chunk_bytes": 67108864,
        },
        "routing plan drift",
    )

    ranks = receipt["ranks"]
    _require(
        isinstance(ranks, list) and len(ranks) == 32, "exact 32-rank inventory required"
    )
    expected_ranks = [
        {
            "rank": rank,
            "tensor_count": 1285,
            "expected_bytes": 19210553052,
            "inventory_sha256": digest,
        }
        for rank, digest in enumerate(RANK_HASHES)
    ]
    _require(ranks == expected_ranks, "rank order, count, bytes, or inventory drift")
    _require(
        len({row["inventory_sha256"] for row in ranks}) == 32,
        "duplicate rank inventory",
    )

    _require(
        receipt["blockers"] == {},
        "source-contract blocker drift",
    )
    claims = receipt["claims"]
    _require(
        isinstance(claims, Mapping) and claims and not any(claims.values()),
        "claims must remain false",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument("--audit-tool", type=Path, default=AUDIT_TOOL)
    args = parser.parse_args()
    value = json.loads(args.receipt.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), "receipt must be an object")
    validate_receipt(value, args.audit_tool)
    print("PASS: exact TP32 symbolic rank plan is routed; compile_permitted=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
