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
    "fe62e396a2214565cce7866af4f681732a7f8020442f40f3dedaee3162de507c",
    "13069ca63445de19c7e2c928b00194419f3812aa764b049153a817a1779176a4",
    "746fec55966f9c8480f06caaeab71a9ae3a15dcf395c7bb60512f8b85fd0a6c5",
    "739d6d624f508ef2c7bea36a142c77742bcb1f065c8243a5019e6ef73f3974ff",
    "2a160fc1b65de248b02accf86fcc14f48574f5def519b2c3787d37ad6ff8f163",
    "be57ac2cf3a4b7350c13c570ad5532530c06a7774e8a256cc12042469d2770ed",
    "e0b5cc98a9673ae15c4cf42560c6c96cf7efb538adce975874be887fd24d464f",
    "7287022fb981127c8b5bef55443c7104bc3f8605f90a84cbfc3831d0d2e6aeea",
    "b6e9134a1512df60ecbe35259477a21c85f743e6107b8ea1fc13b9987564dae6",
    "47f51e67aadad093f81746f83f2dcfe12f3d3298b8b6d0508b9b517aa420b4c7",
    "da2717d9660392da8717247a552d148b0553f8b284d19c99903c2e1f30d69389",
    "0071070be05acf6e7acd8392c8bb472984481bd327b926c0b9b4524674a4f927",
    "22826c25ec6b4e546d3b06a5c9a3295be75087811e6a884b4c21ee622b739220",
    "175bc806a730aba9dca0db085a1c38045322c4478541d8178c17e3d830f5dc6c",
    "712181e06e0a8fc005cc912450a0b5add3c9a9523bdb871bd0a303095ac11bdd",
    "1e1a05660af58bfd0acc80014c29a4554c1f647859cbcfb6d728a2eeabc2ab2f",
    "046541071e8a360ba168ec2c36a6a7bbcd3c57f310937aa0522858c79a8f8f5c",
    "f6f26901b8fc535f220dd1c3a3023ae60c81ee363354433c24992b7b2d087125",
    "8aa566046eb20a54a41a2ca7004cacb3a35ea737bbabde9da3383b7d818fd311",
    "6696a0df77e7dfc4448adb16ae9e8ecb856ac8274b492a9f6789250985b9b1ab",
    "136afd181b522a3b75f3a5dc20e0d835d345606a9efd1b48984a918a88d624c9",
    "a0eb5806f553d7c5b4c68dbe7e5b0b65bc26ac89db67dde830bc80e69ebd7ae9",
    "bd60079e2df6639a3412639fc2fff58444b543d3581401230e1299978f9dd666",
    "a6101e243f0f11ec57f7bf1d12e0236799cb1b7a6c11439930443855b6fe454b",
    "a96556b8564339a02740b5790276dfa6c78307f40aca6d4cd196a7a8fd72a4c7",
    "ca90737a64a6eaab158b227d578ab8282cc9b3e8ee32aee4dfa1c2ff22c9e77c",
    "2943f4c59331bfa8a9bd5ff9cce6631c00f5654d14e774157e11f1056edbbfb0",
    "aba7734777708b56e0f991672807dd0a89ee2d7c00a705a398204637d7bbbd26",
    "8544451f77b339170a9370d9ed62418f9b45a736291fe3d289cbd3c8d7fbcf4b",
    "013d71c32cc68e2ce44ad62ac7832c7416fdd928c859f343869b9ea34606a908",
    "0445c2868617426055d552cff52f42f8159bb59b74e4b965342abc2d1179eb11",
    "0de07b99e8b70362e1da75d3b3fb4a93af40ebfbfe72b9f56f9b974b77c07352",
)


class RankPlanValidationError(ValueError):
    """The symbolic receipt is incomplete, mutated, or overclaims readiness."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RankPlanValidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_receipt(receipt: Mapping[str, Any], tool_path: Path = AUDIT_TOOL) -> None:
    """Require the exact reviewed host-only HOLD and every rank identity."""
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
    _require(receipt["status"] == "HOLD_UNROUTED_SOURCE_CONTRACT", "status drift")
    _require(receipt["complete"] is False, "receipt must remain incomplete")
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
        "tool_sha256": "f6f661eee4cf0c38225a15e41304c753cbebff22d83e1fbbe3bf2e73be3cb6d1",
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
                "dropped_mtp_or_speculation": 4705,
                "incompatible_hash_route_dtype": 3,
                "orphan": 0,
                "routable": 33959,
                "support_scale": 33389,
                "unmapped_mhc": 261,
            },
            "source_category_sha256": {
                "dropped_mtp_or_speculation": "8222ff3f67bb3eceb230d36f787506416629a0beafaaf1ee1f56ab19a9cf5e9f",
                "incompatible_hash_route_dtype": "36e2d1f0d5a59f9565d376ad8d7e1f53e68d461ca1939c6b6758b667e78851b2",
                "orphan": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                "routable": "cd37aeea3bf2dc21d5155c60ef7019a6a2923117562db38ba4d2e5bf342d84b5",
                "support_scale": "d8a59e9afc77307fd08f536139638853f27da375dc18df7a7b0bbe2ad22f4023",
                "unmapped_mhc": "ad07358ebd20fce2a30d3ddd889880250d5280c523eb474a226a2982833d58d6",
            },
            "target_tensor_count_per_rank": 1024,
            "target_bytes_per_rank": 19075015296,
            "target_plan_sha256": "110c5a4b1e3844d0aaf6e43132219e21ac0de333d41732152a4c82116fa2ba12",
            "ownership_counts": {
                "expert_axis_replicated_intermediate_tp_sharded": 86,
                "replicated": 549,
                "replicated_hash_route": 3,
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
            "tensor_count": 1024,
            "expected_bytes": 19075015296,
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
        receipt["blockers"]
        == {
            "incompatible_hash_route_dtype": {
                "count": 3,
                "keys_sha256": "36e2d1f0d5a59f9565d376ad8d7e1f53e68d461ca1939c6b6758b667e78851b2",
                "examples": [
                    "layers.0.ffn.gate.tid2eid",
                    "layers.1.ffn.gate.tid2eid",
                    "layers.2.ffn.gate.tid2eid",
                ],
            },
            "unmapped_mhc": {
                "count": 261,
                "keys_sha256": "ad07358ebd20fce2a30d3ddd889880250d5280c523eb474a226a2982833d58d6",
                "examples": [
                    "hc_head_base",
                    "hc_head_fn",
                    "hc_head_scale",
                    "layers.0.hc_attn_base",
                    "layers.0.hc_attn_fn",
                    "layers.0.hc_attn_scale",
                    "layers.0.hc_ffn_base",
                    "layers.0.hc_ffn_fn",
                ],
            },
        },
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
    print("PASS: exact TP32 symbolic rank-plan HOLD is bound; compile_permitted=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
