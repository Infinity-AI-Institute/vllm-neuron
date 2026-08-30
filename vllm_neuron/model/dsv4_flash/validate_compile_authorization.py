# SPDX-License-Identifier: Apache-2.0
"""Fail-closed validator for the DeepSeek-V4-Flash TP32 compile packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
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
EXPECTED_SATISFIED = {
    "source_validator_merged": True,
    "checkpoint_headers_and_payload_identity": True,
    "tp32_rank_inventory": False,
    "compiler_inventory": False,
    "cpu_reference_bank": False,
    "emitted_contract_receipt": True,
}
PROMPTS = [
    ("prompt00", "Hi, what can you help me with?"),
    ("prompt01", "What is 84 * 3 / 2?"),
    ("prompt02", "Tell me an interesting fact about the universe!"),
    ("prompt03", "Explain quantum computing in simple terms."),
]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
PAYLOAD_LINE_RE = re.compile(r"^([0-9a-f]{64})  \./([^/]+)$")


class AuthorizationError(ValueError):
    """The compile packet or supplied evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorizationError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_git_text(path: Path) -> str:
    """Hash committed text as canonical LF bytes across autocrlf checkouts."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _require_git_oid(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and GIT_OID_RE.fullmatch(value) is not None,
        f"{label} must be a 40-character Git object ID",
    )
    return value


def _safe_artifact(root: Path, relative: Any, label: str) -> Path:
    _require(isinstance(relative, str) and relative, f"{label} path missing")
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), f"{label} escapes evidence root")
    _require(path.is_file(), f"{label} file missing: {relative}")
    return path


def _git_output(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(
        result.returncode == 0,
        f"Git command failed in {repository}: {' '.join(arguments)}",
    )
    return result.stdout.strip()


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
    _require(
        packet.get("tokenizer_sha256")
        == "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
        "tokenizer drift",
    )
    _require(packet.get("source_tensor_count") == 72317, "tensor census drift")
    _require(packet.get("source_shard_count") == 48, "shard census drift")
    _require(
        packet.get("source_total_bytes") == 166886535336, "source byte census drift"
    )
    _require_sha256(
        str(packet.get("compiler_image", "")).rsplit("sha256:", 1)[-1],
        "compiler image digest",
    )
    topology = packet.get("topology", {})
    _require(
        topology
        == {
            "hardware": "trn2.48xlarge",
            "tp_degree": 32,
            "logical_neuroncore_config": 2,
            "ctx_batch_size": 1,
            "tkg_batch_size": 1,
            "sequence_buckets": [4096],
            "continuous_batching": True,
        },
        "topology drift",
    )
    emitted = packet.get("emitted_contract", {})
    _require(
        emitted
        == {
            "rank_count": 32,
            "rank_checkpoint_dtype": "bfloat16",
            "compute_dtype": "bfloat16",
            "cache_dtype": "bfloat16",
            "fp8_kv": False,
            "runtime_weight_quantized": False,
            "sampler": "greedy_argmax",
            "speculative_decode": False,
            "mtp": False,
            "dspark": False,
            "max_writer_chunk_bytes": 67108864,
        },
        "emitted contract drift",
    )
    blockers = packet.get("blockers")
    _require(isinstance(blockers, list), "blockers must be a list")
    _require(
        {item.get("id") for item in blockers} == EXPECTED_BLOCKERS, "blocker drift"
    )
    blocker_state = {item["id"]: item.get("satisfied") for item in blockers}
    _require(blocker_state == EXPECTED_SATISFIED, "production receipt state drift")
    _require(
        all(item.get("machine_check") for item in blockers),
        "every blocker requires a machine check",
    )
    production = packet.get("production_evidence")
    expected_production = {
        "source_provenance": (
            "evidence/source-provenance.json",
            "6d8adb42de7664cbc2f76fb481aaab12ca1ba8deded4d9fb8daabb40f788a965",
        ),
        "routing_manifest": (
            "evidence/routing-manifest.json",
            "93c516df53d1fc5e88e77acb043f68c9d37b7eb12a75f70d8469ab284a663547",
        ),
        "checkpoint_payload_manifest": (
            "evidence/checkpoint-all-files.sha256",
            "ac05cbd738cb8866595257aec855bdb231b677280bbdcb6bd65950c356f68a8d",
        ),
        "audit_tool": (
            "audit_source_provenance.py",
            "d7c6acf9ae3340ff01ef8aeac3b1ea2f93d53de456c310aab6b83c78ced6ab3a",
        ),
        "tp32_rank_plan_audit": (
            "evidence/tp32-rank-plan-audit.json",
            "b329dab1c85dd7396fcd121814efb7da3b18d849977b5ff0dcdbff96406f3f39",
        ),
        "tp32_rank_plan_validator": (
            "validate_tp32_rank_plan.py",
            "c23d5a4f7b5d6e499663b564ae113fea7112f88c0f6c44ae7209935736e0124d",
        ),
        "host_inventory": (
            "evidence/host-inventory-20260829.json",
            "7ef2c0d54c8c5e7a995e7d3c9231e14e2dad31d4a4dbf3f0d68861f1b5966e31",
        ),
        "emitted_contract_receipt": (
            "evidence/emitted-contract.json",
            "3de8e16ee2722c6947e4827493040675f553c00ea2f11a0e573f9d6d74a413c2",
        ),
    }
    _require(
        isinstance(production, Mapping) and set(production) == set(expected_production),
        "production evidence inventory drift",
    )
    for name, (path, digest) in expected_production.items():
        identity = production[name]
        _require(
            isinstance(identity, Mapping)
            and identity == {"path": path, "sha256": digest},
            f"production evidence binding drift: {name}",
        )

    claims = packet.get("claims", {})
    _require(
        isinstance(claims, Mapping) and not any(claims.values()),
        "claims must remain false",
    )


def validate_compile_contract(
    packet: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    expected = {
        "contract_slug": "tp32-lnc2-b1c1-s4096-bf16-shard_intermediate-skip_dma-cont_batch",
        "model": {"repo_id": packet["model"], "revision": packet["revision"]},
        "stack": {"container_digest": packet["compiler_image"]},
        "compile": {
            "tp": 32,
            "logical_nc_config": 2,
            "ctx_batch_size": 1,
            "tkg_batch_size": 1,
            "sequence_buckets": [4096],
            "disable_argmax_kernel": False,
            "dry_run": False,
            "blockwise_matmul_config": {
                "use_shard_on_intermediate_dynamic_while": True
            },
        },
        "emitted_contract": packet["emitted_contract"],
    }
    _require(
        contract == expected, "effective compile contract is not exactly authorized"
    )


def _validate_source(
    packet: Mapping[str, Any], source: Mapping[str, Any], root: Path, repository: Path
) -> None:
    for key in ("source_commit", "validator_merge_sha"):
        _require_git_oid(source.get(key), key)
    _require_git_oid(source.get("validator_merge_tree_sha"), "validator merge tree")
    _require(
        source.get("validator_merged_to_agent_main") is True,
        "source is not validator-merged",
    )
    for key in ("revision", "config_sha256", "index_sha256"):
        _require(source.get(key) == packet[key], f"source {key} drift")
    model_files = source.get("model_files")
    expected_model_hashes = {
        "config.json": packet["config_sha256"],
        "model.safetensors.index.json": packet["index_sha256"],
        "tokenizer.json": packet["tokenizer_sha256"],
    }
    _require(
        isinstance(model_files, Mapping)
        and set(model_files) == set(expected_model_hashes),
        "source model-file inventory drift",
    )
    for name, expected_hash in expected_model_hashes.items():
        identity = model_files[name]
        _require(
            isinstance(identity, Mapping) and set(identity) == {"sha256", "bytes"},
            f"source {name} identity fields drift",
        )
        _require(identity["sha256"] == expected_hash, f"source {name} hash drift")
        _require(
            isinstance(identity["bytes"], int) and identity["bytes"] > 0,
            f"source {name} byte count invalid",
        )
    command = ["git", "-C", str(repository), "merge-base", "--is-ancestor"]
    for ancestor, descendant in (
        (source["source_commit"], source["validator_merge_sha"]),
        (source["validator_merge_sha"], "origin/agent-main"),
    ):
        result = subprocess.run(
            [*command, ancestor, descendant], check=False, capture_output=True
        )
        _require(
            result.returncode == 0,
            f"source ancestry failed: {ancestor} -> {descendant}",
        )

    _require(
        _git_output(
            repository, "rev-parse", f"{source['validator_merge_sha']}^{{tree}}"
        )
        == source["validator_merge_tree_sha"],
        "validator merge tree evidence drift",
    )

    shards = source.get("shards")
    _require(
        isinstance(shards, list) and len(shards) == 48,
        "exact 48-entry shard inventory required",
    )
    expected_names = [
        f"model-{index:05d}-of-00048.safetensors" for index in range(1, 49)
    ]
    _require(
        [item.get("name") for item in shards] == expected_names,
        "shard names/order drift",
    )
    required_keys = {
        "name",
        "lfs_sha256",
        "size",
        "header_sha256",
        "header_bytes",
        "tensor_count",
        "routing_sha256",
    }
    _require(
        all(set(item) == required_keys for item in shards),
        "shard identity fields drift",
    )
    for index, item in enumerate(shards):
        _require_sha256(item["lfs_sha256"], f"shard {index} LFS")
        _require_sha256(item["header_sha256"], f"shard {index} header")
        _require_sha256(item["routing_sha256"], f"shard {index} routing")
        _require(
            isinstance(item["size"], int) and item["size"] > 0, "invalid shard size"
        )
        _require(
            isinstance(item["header_bytes"], int) and item["header_bytes"] > 0,
            "invalid header bytes",
        )
        _require(
            isinstance(item["tensor_count"], int) and item["tensor_count"] > 0,
            "invalid shard tensor count",
        )
    _require(
        len({item["lfs_sha256"] for item in shards}) == 48, "duplicate LFS identities"
    )
    _require(
        len({item["header_sha256"] for item in shards}) == 48,
        "duplicate header identities",
    )
    _require(
        sum(item["size"] for item in shards) == packet["source_total_bytes"],
        "source byte total drift",
    )
    _require(
        sum(item["tensor_count"] for item in shards) == packet["source_tensor_count"],
        "source tensor total drift",
    )
    _require(
        source.get("payload_bytes_read_during_header_audit") == 0,
        "header audit read payload",
    )
    _require(
        source.get("missing") == []
        and source.get("orphan") == []
        and source.get("misrouted") == [],
        "source routing audit failed",
    )
    _require(
        source.get("canonical_shard_inventory_sha256") == _canonical_sha256(shards),
        "canonical shard inventory drift",
    )
    routing = source.get("routing_manifest", {})
    routing_path = _safe_artifact(root, routing.get("path"), "routing manifest")
    _require(
        _sha256_git_text(routing_path)
        == _require_sha256(routing.get("sha256"), "routing manifest"),
        "routing manifest file drift",
    )
    routing_body = _load(routing_path)
    _require(
        routing_body
        == {
            "revision": packet["revision"],
            "index_sha256": packet["index_sha256"],
            "shards": shards,
        },
        "routing manifest body drift",
    )


def _validate_host_inventory(
    packet: Mapping[str, Any], inventory: Mapping[str, Any]
) -> None:
    _require(
        inventory.get("schema") == "dsv4-read-only-host-inventory-v1",
        "host inventory schema drift",
    )
    _require(
        inventory.get("captured_at_utc") == "2026-08-29T21:55:19Z",
        "host inventory capture time drift",
    )
    boundary = inventory.get("audit_boundary")
    _require(
        isinstance(boundary, Mapping)
        and set(boundary)
        == {
            "compiles_started",
            "devices_used",
            "docker_jobs_started_or_stopped",
            "downloads_started",
            "files_created_on_hosts",
            "services_started_or_stopped",
        }
        and not any(boundary.values()),
        "host inventory exceeded its read-only boundary",
    )

    compiler = inventory.get("compiler_stack")
    _require(isinstance(compiler, Mapping), "compiler host inventory missing")
    _require(
        compiler.get("image_reference") == packet["compiler_image"],
        "inventoried compiler image drift",
    )
    _require(
        compiler.get("image_manifest_digest")
        == "sha256:011d49c7495457fc2932dedd3fbecf67d28833a3f12c147377cee4d72889ebc1",
        "compiler manifest digest drift",
    )
    _require(
        compiler.get("image_config_digest")
        == "sha256:cffb98efd1c380b1f3a8092aacb070e78fa7cc712e0225a5569094b332cc218d",
        "compiler config digest drift",
    )
    _require(
        compiler.get("versions_from_immutable_image_history")
        == {
            "nki": "0.4.0+25940409122.gd30719f9",
            "neuron_runtime": "2.32.31.0-0234f5ed2",
            "neuronx-cc": "2.25.3371.0+f524f7f8",
            "neuronx-distributed": "0.19.28093+fc70b593",
            "neuronx-distributed-inference": "0.10.17970+8548ba25",
            "torch-neuronx": "2.9.0.2.14.27725+e2ff0410",
        },
        "compiler version inventory drift",
    )

    hosts = inventory.get("hosts")
    _require(
        isinstance(hosts, Mapping) and set(hosts) == {"r7i", "trn2"},
        "host set drift",
    )
    r7i = hosts["r7i"]
    trn2 = hosts["trn2"]
    _require(
        r7i.get("address") == "ubuntu@100.122.8.63" and r7i.get("cpu_count") == 48,
        "r7i identity drift",
    )
    _require(
        trn2.get("address") == "ubuntu@100.105.72.19" and trn2.get("cpu_count") == 192,
        "Trn2 identity drift",
    )
    checkpoint = trn2.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "checkpoint inventory missing")
    for key in ("revision", "config_sha256", "index_sha256", "tokenizer_sha256"):
        _require(checkpoint.get(key) == packet[key], f"checkpoint {key} drift")
    _require(
        checkpoint.get("shard_count") == packet["source_shard_count"]
        and checkpoint.get("shard_bytes") == packet["source_total_bytes"]
        and checkpoint.get("prior_payload_verification") == "48/48_PASS",
        "checkpoint completeness drift",
    )
    _require(
        checkpoint.get("complete_marker_sha256") == hashlib.sha256(b"").hexdigest(),
        "checkpoint completion marker drift",
    )

    capacity = trn2.get("rank_materialization_capacity")
    _require(isinstance(capacity, Mapping), "rank capacity inventory missing")
    expected_output = 19_210_553_052 * 32
    free_bytes = trn2["filesystem"]["available_bytes"]
    reserve = capacity.get("safety_reserve_bytes")
    _require(
        capacity.get("rank_count") == 32
        and capacity.get("target_bytes_per_rank") == 19_210_553_052
        and capacity.get("output_bytes") == expected_output,
        "TP32 output capacity arithmetic drift",
    )
    _require(
        isinstance(reserve, int)
        and reserve == 200 * 1024**3
        and capacity.get("post_materialization_free_bytes")
        == free_bytes - expected_output
        and capacity.get("usable_headroom_after_output_and_reserve_bytes")
        == free_bytes - expected_output - reserve
        and capacity.get("fits_with_reserve") is True,
        "TP32 guarded capacity drift",
    )
    _require(
        r7i["filesystem"]["available_bytes"] < expected_output,
        "r7i must not be claimed as a safe TP32 materialization target",
    )

    holds = inventory.get("formal_holds")
    _require(isinstance(holds, Mapping), "formal hold inventory missing")
    _require(
        holds["tp32_rank_inventory"].get("resolved") is False
        and holds["tp32_rank_inventory"].get("materialized_rank_files_found") == 0,
        "rank materialization claim drift",
    )
    _require(
        holds["compiler_inventory"].get("resolved") is False,
        "compiler hold was overclaimed",
    )
    _require(
        holds["cpu_reference_bank"].get("resolved") is False
        and holds["cpu_reference_bank"]["existing_non_authorizing_asset"].get("weights")
        == "UD-Q3_K_XL GGUF",
        "CPU reference hold was overclaimed",
    )
    _require(
        holds["emitted_contract_receipt"]
        == {"path": "emitted-contract.json", "resolved": True},
        "emitted-contract hold drift",
    )


def validate_production_source_evidence(
    packet: Mapping[str, Any], package_root: Path, repository: Path
) -> None:
    """Validate the two committed production receipts independently of later gates."""
    production = packet["production_evidence"]
    resolved: dict[str, Path] = {}
    for name, identity in production.items():
        path = (package_root / identity["path"]).resolve()
        _require(path.is_relative_to(package_root.resolve()), f"{name} escapes package")
        _require(path.is_file(), f"production evidence missing: {name}")
        _require(
            _sha256_git_text(path) == identity["sha256"],
            f"production evidence hash drift: {name}",
        )
        resolved[name] = path

    _validate_host_inventory(packet, _load(resolved["host_inventory"]))
    _require(
        _load(resolved["emitted_contract_receipt"])
        == {
            "topology": packet["topology"],
            "emitted_contract": packet["emitted_contract"],
        },
        "committed emitted-contract receipt drift",
    )

    source_path = resolved["source_provenance"]
    source_root = source_path.parent
    source = _load(source_path)
    _require(
        source.get("source_commit") == "266af97d16cb8621514d6ae77741ce17ece40d9c",
        "reviewed source head drift",
    )
    _require(
        source.get("validator_merge_sha") == "2543fe18c70f7d92a5bf498e4adaccacde425728",
        "validator merge SHA drift",
    )
    _require(
        source.get("validator_merge_tree_sha")
        == "e51e1e648726a054b8906b6b2338b7fb75b795ea",
        "validator merge tree drift",
    )
    _validate_source(packet, source, source_root, repository)
    _require(
        (source_root / source["routing_manifest"]["path"]).resolve()
        == resolved["routing_manifest"],
        "routing path binding drift",
    )

    payload = source.get("payload_manifest")
    _require(
        isinstance(payload, Mapping)
        and payload
        == {
            "path": "checkpoint-all-files.sha256",
            "sha256": production["checkpoint_payload_manifest"]["sha256"],
            "entry_count": 56,
            "shard_entries": 48,
            "shard_bytes": packet["source_total_bytes"],
            "prior_hash_verification": "48/48_PASS",
        },
        "payload manifest receipt drift",
    )
    _require(
        (source_root / payload["path"]).resolve()
        == resolved["checkpoint_payload_manifest"],
        "payload manifest path binding drift",
    )
    entries: dict[str, str] = {}
    for line in (
        resolved["checkpoint_payload_manifest"].read_text(encoding="utf-8").splitlines()
    ):
        match = PAYLOAD_LINE_RE.fullmatch(line)
        _require(match is not None, "malformed checkpoint payload manifest")
        digest, name = match.groups()
        _require(name not in entries, f"duplicate payload manifest entry: {name}")
        entries[name] = digest
    _require(len(entries) == 56, "payload manifest entry census drift")
    for shard in source["shards"]:
        _require(
            entries.get(shard["name"]) == shard["lfs_sha256"],
            f"payload identity mismatch: {shard['name']}",
        )
    for name, identity in source["model_files"].items():
        _require(
            entries.get(name) == identity["sha256"],
            f"model-file manifest identity mismatch: {name}",
        )

    _require(
        source.get("complete_marker")
        == {
            "name": ".complete",
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        },
        "checkpoint completion marker drift",
    )
    _require(
        source.get("audit")
        == {
            "tool_path": "audit_source_provenance.py",
            "tool_sha256": production["audit_tool"]["sha256"],
            "method": "safetensors_8_byte_prefix_plus_json_header_only",
            "tensor_payload_bytes_read": 0,
            "rank_conversion_performed": False,
            "large_outputs_created": False,
        },
        "host-only audit boundary drift",
    )
    _require(
        resolved["audit_tool"] == package_root / source["audit"]["tool_path"],
        "audit tool path binding drift",
    )


def _validate_ranks(
    packet: Mapping[str, Any], inventory: Mapping[str, Any], root: Path
) -> None:
    ranks = inventory.get("ranks")
    _require(
        isinstance(ranks, list)
        and [item.get("rank") for item in ranks] == list(range(32)),
        "rank inventory must cover 0..31",
    )
    inventory_hashes: set[str] = set()
    checkpoint_hashes: set[str] = set()
    for item in ranks:
        _require(
            set(item)
            == {
                "rank",
                "tp_degree",
                "dtype",
                "inventory_sha256",
                "checkpoint",
                "manifest",
            },
            "rank receipt fields drift",
        )
        _require(
            item["tp_degree"] == 32 and item["dtype"] == "bfloat16",
            "rank contract drift",
        )
        inventory_hashes.add(
            _require_sha256(item["inventory_sha256"], "rank inventory")
        )
        checkpoint = item["checkpoint"]
        manifest_ref = item["manifest"]
        _require(
            checkpoint.get("path") == f"ranks/tp{item['rank']}.safetensors",
            "rank checkpoint path must bind to rank",
        )
        _require(
            manifest_ref.get("path") == f"ranks/tp{item['rank']}.manifest.json",
            "rank manifest path must bind to rank",
        )
        checkpoint_path = _safe_artifact(
            root, checkpoint.get("path"), "rank checkpoint"
        )
        checkpoint_sha = _require_sha256(checkpoint.get("sha256"), "rank checkpoint")
        _require(
            _sha256_file(checkpoint_path) == checkpoint_sha,
            "rank checkpoint hash drift",
        )
        _require(
            checkpoint_path.stat().st_size == checkpoint.get("bytes") > 0,
            "rank checkpoint byte drift",
        )
        checkpoint_hashes.add(checkpoint_sha)
        manifest_path = _safe_artifact(root, manifest_ref.get("path"), "rank manifest")
        _require(
            _sha256_file(manifest_path)
            == _require_sha256(manifest_ref.get("sha256"), "rank manifest"),
            "rank manifest hash drift",
        )
        manifest = _load(manifest_path)
        _require(
            manifest.get("schema") == "dsv4-streaming-rank-v1",
            "rank manifest schema drift",
        )
        _require(
            manifest.get("rank") == item["rank"] and manifest.get("tp_degree") == 32,
            "rank manifest topology drift",
        )
        _require(
            manifest.get("rank_inventory_sha256") == item["inventory_sha256"],
            "rank inventory binding drift",
        )
        _require(
            manifest.get("checkpoint", {}).get("sha256") == checkpoint_sha,
            "rank checkpoint binding drift",
        )
        _require(
            manifest.get("resource_bound", {}).get("observed_max_chunk_bytes", 1 << 63)
            <= packet["emitted_contract"]["max_writer_chunk_bytes"],
            "rank chunk bound exceeded",
        )
    _require(len(inventory_hashes) == 32, "duplicate rank inventory identities")
    _require(len(checkpoint_hashes) == 32, "duplicate rank checkpoint identities")
    _require(
        inventory.get("canonical_rank_inventory_sha256") == _canonical_sha256(ranks),
        "canonical rank inventory drift",
    )


def _validate_launch_model(
    packet: Mapping[str, Any], source: Mapping[str, Any], model_dir: Path
) -> None:
    model_dir = model_dir.resolve()
    _require(model_dir.is_dir(), f"launch model directory missing: {model_dir}")
    model_files = source["model_files"]
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        path = model_dir / name
        _require(path.is_file(), f"launch model file missing: {name}")
        identity = model_files[name]
        _require(path.stat().st_size == identity["bytes"], f"launch {name} byte drift")
        _require(_sha256_file(path) == identity["sha256"], f"launch {name} hash drift")

    reviewed_shards = source["shards"]
    expected_names = [item["name"] for item in reviewed_shards]
    actual_names = sorted(
        path.name for path in model_dir.glob("model-*-of-00048.safetensors")
    )
    _require(actual_names == expected_names, "launch model shard set/order drift")
    for item in reviewed_shards:
        path = model_dir / item["name"]
        _require(
            path.stat().st_size == item["size"], f"launch {item['name']} byte drift"
        )
        _require(
            _sha256_file(path) == item["lfs_sha256"],
            f"launch {item['name']} payload hash drift",
        )
    _require(
        len(actual_names) == packet["source_shard_count"], "launch shard count drift"
    )


def _validate_launch_ranks(inventory: Mapping[str, Any], rank_source: Path) -> None:
    rank_source = rank_source.resolve()
    _require(rank_source.is_dir(), f"launch rank directory missing: {rank_source}")
    ranks = inventory["ranks"]
    expected_names = [f"tp{rank}_sharded_checkpoint.safetensors" for rank in range(32)]
    actual_names = sorted(
        (path.name for path in rank_source.glob("tp*_sharded_checkpoint.safetensors")),
        key=lambda name: int(name.removeprefix("tp").split("_", 1)[0]),
    )
    _require(actual_names == expected_names, "launch rank file set/order drift")
    for item in ranks:
        path = rank_source / f"tp{item['rank']}_sharded_checkpoint.safetensors"
        checkpoint = item["checkpoint"]
        _require(
            path.stat().st_size == checkpoint["bytes"],
            f"launch rank {item['rank']} byte drift",
        )
        _require(
            _sha256_file(path) == checkpoint["sha256"],
            f"launch rank {item['rank']} hash drift",
        )


def _validate_launch_source(source: Mapping[str, Any], source_dir: Path) -> None:
    source_dir = source_dir.resolve()
    _require(source_dir.is_dir(), f"launch source directory missing: {source_dir}")
    _require(
        _git_output(source_dir, "status", "--porcelain=v1", "--untracked-files=all")
        == "",
        "launch source checkout is dirty",
    )
    head = _git_output(source_dir, "rev-parse", "HEAD")
    _require(head == source["validator_merge_sha"], "launch source HEAD drift")
    tree = _git_output(source_dir, "rev-parse", "HEAD^{tree}")
    _require(tree == source["validator_merge_tree_sha"], "launch source tree drift")


def validate_launch_bindings(
    packet: Mapping[str, Any],
    root: Path,
    model_dir: Path,
    compile_run_root: Path,
    rank_source: Path,
    source_dir: Path,
) -> None:
    """Bind the reviewed evidence to every path consumed by command.sh."""
    compile_run_root = compile_run_root.resolve()
    _require(
        rank_source.resolve() == compile_run_root / "weights",
        "launch rank source is not COMPILE_RUN_ROOT/weights",
    )
    source = _load(root / "source-provenance.json")
    inventory = _load(root / "rank-inventory.json")
    _validate_launch_model(packet, source, model_dir)
    _validate_launch_ranks(inventory, rank_source)
    _validate_launch_source(source, source_dir)


def _validate_compiler(packet: Mapping[str, Any], compiler: Mapping[str, Any]) -> None:
    _require(
        compiler.get("image_digest") == packet["compiler_image"], "compiler image drift"
    )
    _require_sha256(compiler.get("image_config_digest"), "image config")
    packages = compiler.get("packages")
    names = {"neuronx-cc", "torch-neuronx", "nxdi", "nxd", "nki", "torch", "runtime"}
    _require(
        isinstance(packages, Mapping) and set(packages) == names,
        "compiler package inventory drift",
    )
    for name, identity in packages.items():
        _require(
            set(identity) == {"version", "artifact_sha256", "source_commit"},
            f"{name} identity fields drift",
        )
        _require(
            isinstance(identity["version"], str) and identity["version"],
            f"{name} version missing",
        )
        _require_sha256(identity["artifact_sha256"], f"{name} artifact")
        _require_git_oid(identity["source_commit"], f"{name} source")
    _require(
        compiler.get("canonical_inventory_sha256") == _canonical_sha256(packages),
        "compiler canonical digest drift",
    )


def _validate_reference(
    packet: Mapping[str, Any], reference: Mapping[str, Any], root: Path
) -> None:
    for key in ("revision", "config_sha256", "index_sha256", "tokenizer_sha256"):
        _require(reference.get(key) == packet[key], f"CPU reference {key} drift")
    _require(reference.get("prompts_sha16") == "99f702c72d2fafcc", "prompt bank drift")
    prompts = reference.get("prompts")
    _require(
        isinstance(prompts, list) and len(prompts) == 4, "four prompt receipts required"
    )
    logit_hashes: set[str] = set()
    for expected, prompt in zip(PROMPTS, prompts, strict=True):
        _require(
            prompt.get("id") == expected[0] and prompt.get("text") == expected[1],
            "prompt identity drift",
        )
        _require(
            isinstance(prompt.get("prompt_token_ids"), list)
            and all(isinstance(value, int) for value in prompt["prompt_token_ids"]),
            "prompt tokens missing",
        )
        _require(
            isinstance(prompt.get("generated_token_ids"), list)
            and len(prompt["generated_token_ids"]) == 10
            and all(isinstance(value, int) for value in prompt["generated_token_ids"]),
            "generated 4x10 IDs incomplete",
        )
        logits = prompt.get("logits")
        _require(
            isinstance(logits, list) and len(logits) == 10,
            "ten logit receipts required per prompt",
        )
        for position, receipt in enumerate(logits):
            _require(
                receipt.get("position") == position
                and receipt.get("dtype") == "float32"
                and receipt.get("vocab_size") == 129280
                and receipt.get("bytes") == 129280 * 4
                and receipt.get("finite") is True,
                "logit contract drift",
            )
            path = _safe_artifact(root, receipt.get("path"), "logit artifact")
            _require(
                path.stat().st_size == receipt["bytes"], "logit artifact byte drift"
            )
            digest = _require_sha256(receipt.get("sha256"), "logit artifact")
            _require(_sha256_file(path) == digest, "logit artifact hash drift")
            logit_hashes.add(digest)
    _require(len(logit_hashes) == 40, "duplicate or missing full-logit identities")
    body = dict(reference)
    claimed = body.pop("manifest_sha256", None)
    _require(claimed == _canonical_sha256(body), "CPU reference canonical digest drift")


def validate_evidence(
    packet: Mapping[str, Any], root: Path, repository: Path
) -> list[str]:
    required = {
        "source": root / "source-provenance.json",
        "ranks": root / "rank-inventory.json",
        "compiler": root / "compiler-provenance.json",
        "reference": root / "cpu-reference-manifest.json",
        "emitted": root / "emitted-contract.json",
    }
    missing = sorted(path.name for path in required.values() if not path.is_file())
    if missing:
        return [f"missing:{name}" for name in missing]
    _validate_source(packet, _load(required["source"]), root, repository)
    _validate_ranks(packet, _load(required["ranks"]), root)
    _validate_compiler(packet, _load(required["compiler"]))
    _validate_reference(packet, _load(required["reference"]), root)
    _require(
        _load(required["emitted"])
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
    parser.add_argument("--compile-contract", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--compile-run-root", type=Path)
    parser.add_argument("--rank-source", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--require-compile-permitted", action="store_true")
    args = parser.parse_args()
    packet = _load(args.packet)
    validate_packet(packet)
    repository = args.source_dir or HERE.parents[2]
    validate_production_source_evidence(packet, HERE, repository)
    if args.compile_contract is not None:
        validate_compile_contract(packet, _load(args.compile_contract))
    holds = [item["id"] for item in packet["blockers"] if not item["satisfied"]]
    if args.evidence_root is not None:
        _require(args.source_dir is not None, "--source-dir is required with evidence")
        _require(args.model_dir is not None, "--model-dir is required with evidence")
        _require(
            args.compile_run_root is not None,
            "--compile-run-root is required with evidence",
        )
        _require(
            args.rank_source is not None, "--rank-source is required with evidence"
        )
        _require(
            args.compile_contract is not None,
            "--compile-contract is required with evidence",
        )
        holds = validate_evidence(packet, args.evidence_root, args.source_dir)
        if not holds:
            validate_launch_bindings(
                packet,
                args.evidence_root,
                args.model_dir,
                args.compile_run_root,
                args.rank_source,
                args.source_dir,
            )
    permitted = not holds
    print(json.dumps({"compile_permitted": permitted, "holds": holds}, sort_keys=True))
    return 2 if args.require_compile_permitted and not permitted else 0


if __name__ == "__main__":
    raise SystemExit(main())
