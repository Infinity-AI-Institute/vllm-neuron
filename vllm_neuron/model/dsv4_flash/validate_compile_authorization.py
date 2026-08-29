# SPDX-License-Identifier: Apache-2.0
"""Fail-closed validator for the DeepSeek-V4-Flash TP32 compile packet."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PACKET = HERE / "tp32_compile_authorization.json"
EXPECTED_BLOCKERS = {
    "source_validator_merged",
    "checkpoint_headers_and_payload_identity",
    "tp32_rank_inventory",
    "compiler_inventory",
    "cpu_reference_bank",
    "emitted_contract_receipt",
}


class AuthorizationError(ValueError):
    """The compile packet or supplied evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorizationError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def validate_packet(packet: Mapping[str, Any]) -> None:
    _require(packet.get("schema_version") == 1, "schema drift")
    _require(
        packet.get("revision") == "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
        "revision drift",
    )
    _require(
        packet.get("config_sha256")
        == "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
        "config drift",
    )
    _require(
        packet.get("index_sha256")
        == "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
        "index drift",
    )
    _require(packet.get("source_tensor_count") == 72317, "tensor census drift")
    _require(packet.get("source_shard_count") == 48, "shard census drift")
    topology = packet.get("topology", {})
    _require(topology.get("tp_degree") == 32, "TP must be 32")
    _require(topology.get("logical_neuroncore_config") == 2, "LNC drift")
    _require(topology.get("sequence_buckets") == [4096], "sequence drift")
    emitted = packet.get("emitted_contract", {})
    _require(emitted.get("rank_count") == 32, "rank count drift")
    _require(emitted.get("rank_checkpoint_dtype") == "bfloat16", "weight dtype drift")
    _require(emitted.get("compute_dtype") == "bfloat16", "compute dtype drift")
    _require(emitted.get("cache_dtype") == "bfloat16", "cache dtype drift")
    _require(emitted.get("fp8_kv") is False, "FP8 KV is forbidden")
    _require(
        emitted.get("runtime_weight_quantized") is False, "weight quantization drift"
    )
    _require(emitted.get("sampler") == "greedy_argmax", "sampler drift")
    _require(emitted.get("speculative_decode") is False, "speculation is forbidden")
    _require(
        emitted.get("mtp") is False and emitted.get("dspark") is False,
        "MTP/DSpark forbidden",
    )
    blockers = packet.get("blockers")
    _require(isinstance(blockers, list), "blockers must be a list")
    _require(
        {item.get("id") for item in blockers} == EXPECTED_BLOCKERS, "blocker drift"
    )
    _require(
        all(item.get("satisfied") is False for item in blockers),
        "static packet may not pre-authorize blockers",
    )
    _require(
        all(item.get("machine_check") for item in blockers),
        "every blocker needs a machine check",
    )
    claims = packet.get("claims", {})
    _require(
        isinstance(claims, Mapping) and not any(claims.values()),
        "claims must remain false",
    )


def validate_evidence(packet: Mapping[str, Any], root: Path) -> list[str]:
    """Validate evidence when present; absence remains a normal HOLD."""
    required = {
        "source_validator_merged": root / "source-provenance.json",
        "checkpoint_headers_and_payload_identity": root / "source-provenance.json",
        "tp32_rank_inventory": root / "rank-inventory.json",
        "compiler_inventory": root / "compiler-provenance.json",
        "cpu_reference_bank": root / "cpu-reference-manifest.json",
        "emitted_contract_receipt": root / "emitted-contract.json",
    }
    missing = sorted({item.name for item in required.values() if not item.is_file()})
    if missing:
        return [f"missing:{name}" for name in missing]

    source = _load(required["source_validator_merged"])
    _require(
        source.get("validator_merged_to_agent_main") is True,
        "source is not validator-merged",
    )
    _require(source.get("revision") == packet["revision"], "source revision drift")
    _require(
        source.get("config_sha256") == packet["config_sha256"], "source config drift"
    )
    _require(source.get("index_sha256") == packet["index_sha256"], "source index drift")
    _require(source.get("shard_count") == 48, "source shard audit drift")
    _require(source.get("tensor_count") == 72317, "source tensor audit drift")
    _require(
        source.get("header_audit_payload_bytes") == 0, "header audit loaded payload"
    )
    _require(
        source.get("missing") == []
        and source.get("orphan") == []
        and source.get("misrouted") == [],
        "source routing audit failed",
    )

    inventory = _load(required["tp32_rank_inventory"])
    ranks = inventory.get("ranks")
    _require(
        isinstance(ranks, list)
        and [item.get("rank") for item in ranks] == list(range(32)),
        "rank inventory must cover 0..31",
    )
    _require(all(item.get("tp_degree") == 32 for item in ranks), "rank TP drift")
    _require(all(item.get("dtype") == "bfloat16" for item in ranks), "rank dtype drift")
    _require(
        all(
            item.get("observed_max_chunk_bytes", 1 << 63) <= 67108864 for item in ranks
        ),
        "rank chunk bound exceeded",
    )
    _require(
        all(
            item.get("inventory_sha256") and item.get("checkpoint_sha256")
            for item in ranks
        ),
        "rank identities missing",
    )

    compiler = _load(required["compiler_inventory"])
    required_compiler = {
        "image_digest",
        "image_config_digest",
        "neuronx_cc",
        "torch_neuronx",
        "nxdi",
        "nxd",
        "nki",
        "torch",
        "runtime",
    }
    _require(
        required_compiler <= set(compiler)
        and all(compiler[key] for key in required_compiler),
        "compiler inventory incomplete",
    )

    reference = _load(required["cpu_reference_bank"])
    _require(reference.get("prompts_sha16") == "99f702c72d2fafcc", "prompt bank drift")
    _require(
        reference.get("prompt_count") == 4 and reference.get("tokens_per_prompt") == 10,
        "4x10 bank incomplete",
    )
    _require(
        reference.get("token_id_count") == 40
        and reference.get("full_vocab_logit_count") == 40,
        "reference evidence incomplete",
    )
    _require(reference.get("manifest_sha256"), "reference manifest identity missing")

    emitted = _load(required["emitted_contract_receipt"])
    _require(
        emitted
        == {
            "topology": packet["topology"],
            "emitted_contract": packet["emitted_contract"],
        },
        "emitted contract drift",
    )
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=PACKET)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--require-compile-permitted", action="store_true")
    args = parser.parse_args()
    packet = _load(args.packet)
    validate_packet(packet)
    holds = [item["id"] for item in packet["blockers"]]
    if args.evidence_root is not None:
        holds = validate_evidence(packet, args.evidence_root)
    permitted = not holds
    print(json.dumps({"compile_permitted": permitted, "holds": holds}, sort_keys=True))
    return 2 if args.require_compile_permitted and not permitted else 0


if __name__ == "__main__":
    raise SystemExit(main())
