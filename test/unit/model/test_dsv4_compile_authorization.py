from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
PACKAGE = ROOT / "vllm_neuron" / "model" / "dsv4_flash"
SPEC = importlib.util.spec_from_file_location(
    "dsv4_compile_authorization", PACKAGE / "validate_compile_authorization.py"
)
assert SPEC and SPEC.loader
AUTH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTH)


def _packet() -> dict:
    return json.loads(
        (PACKAGE / "tp32_compile_authorization.json").read_text(encoding="utf-8")
    )


def _contract() -> dict:
    return json.loads(
        (PACKAGE / "first_fire_contract.json").read_text(encoding="utf-8")
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _git_repository(root: Path) -> tuple[Path, str, str]:
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    (repo / "source").write_text("source", encoding="utf-8")
    subprocess.run(["git", "add", "source"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=repo, check=True)
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    (repo / "merge").write_text("merge", encoding="utf-8")
    subprocess.run(["git", "add", "merge"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "merge"], cwd=repo, check=True)
    merge = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/agent-main", merge],
        cwd=repo,
        check=True,
    )
    return repo, source, merge


def _complete_evidence(tmp_path: Path) -> tuple[Path, Path]:
    packet = _packet()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    repo, source_commit, merge_sha = _git_repository(tmp_path)

    shards = []
    for index in range(48):
        shards.append(
            {
                "name": f"model-{index + 1:05d}-of-00048.safetensors",
                "lfs_sha256": _sha_bytes(f"lfs-{index}".encode()),
                "size": 1 if index < 47 else packet["source_total_bytes"] - 47,
                "header_sha256": _sha_bytes(f"header-{index}".encode()),
                "header_bytes": 8 + index,
                "tensor_count": 1 if index < 47 else packet["source_tensor_count"] - 47,
                "routing_sha256": _sha_bytes(f"routing-{index}".encode()),
            }
        )
    routing = {
        "revision": packet["revision"],
        "index_sha256": packet["index_sha256"],
        "shards": shards,
    }
    routing_path = evidence / "routing-manifest.json"
    _write_json(routing_path, routing)
    source = {
        "source_commit": source_commit,
        "validator_merge_sha": merge_sha,
        "validator_merged_to_agent_main": True,
        "revision": packet["revision"],
        "config_sha256": packet["config_sha256"],
        "index_sha256": packet["index_sha256"],
        "shards": shards,
        "payload_bytes_read_during_header_audit": 0,
        "missing": [],
        "orphan": [],
        "misrouted": [],
        "canonical_shard_inventory_sha256": AUTH._canonical_sha256(shards),
        "routing_manifest": {
            "path": routing_path.name,
            "sha256": AUTH._sha256_file(routing_path),
        },
    }
    _write_json(evidence / "source-provenance.json", source)

    ranks = []
    for rank in range(32):
        checkpoint_path = evidence / "ranks" / f"tp{rank}.safetensors"
        checkpoint_path.parent.mkdir(exist_ok=True)
        checkpoint_path.write_bytes(f"checkpoint-{rank}".encode())
        checkpoint_sha = AUTH._sha256_file(checkpoint_path)
        inventory_sha = _sha_bytes(f"inventory-{rank}".encode())
        manifest = {
            "schema": "dsv4-streaming-rank-v1",
            "rank": rank,
            "tp_degree": 32,
            "rank_inventory_sha256": inventory_sha,
            "checkpoint": {"sha256": checkpoint_sha},
            "resource_bound": {"observed_max_chunk_bytes": 1024},
        }
        manifest_path = evidence / "ranks" / f"tp{rank}.manifest.json"
        _write_json(manifest_path, manifest)
        ranks.append(
            {
                "rank": rank,
                "tp_degree": 32,
                "dtype": "bfloat16",
                "inventory_sha256": inventory_sha,
                "checkpoint": {
                    "path": checkpoint_path.relative_to(evidence).as_posix(),
                    "sha256": checkpoint_sha,
                    "bytes": checkpoint_path.stat().st_size,
                },
                "manifest": {
                    "path": manifest_path.relative_to(evidence).as_posix(),
                    "sha256": AUTH._sha256_file(manifest_path),
                },
            }
        )
    _write_json(
        evidence / "rank-inventory.json",
        {
            "ranks": ranks,
            "canonical_rank_inventory_sha256": AUTH._canonical_sha256(ranks),
        },
    )

    packages = {
        name: {
            "version": "1.0",
            "artifact_sha256": _sha_bytes(f"artifact-{name}".encode()),
            "source_commit": hashlib.sha1(f"source-{name}".encode()).hexdigest(),
        }
        for name in (
            "neuronx-cc",
            "torch-neuronx",
            "nxdi",
            "nxd",
            "nki",
            "torch",
            "runtime",
        )
    }
    _write_json(
        evidence / "compiler-provenance.json",
        {
            "image_digest": packet["compiler_image"],
            "image_config_digest": _sha_bytes(b"image-config"),
            "packages": packages,
            "canonical_inventory_sha256": AUTH._canonical_sha256(packages),
        },
    )

    prompts = []
    for prompt_index, (prompt_id, text) in enumerate(AUTH.PROMPTS):
        logits = []
        for position in range(10):
            path = evidence / "logits" / f"{prompt_id}-{position}.fp32"
            path.parent.mkdir(exist_ok=True)
            prefix = f"logits-{prompt_index}-{position}".encode()
            path.write_bytes(prefix + bytes(129280 * 4 - len(prefix)))
            logits.append(
                {
                    "position": position,
                    "path": path.relative_to(evidence).as_posix(),
                    "sha256": AUTH._sha256_file(path),
                    "bytes": path.stat().st_size,
                    "dtype": "float32",
                    "vocab_size": 129280,
                    "finite": True,
                }
            )
        prompts.append(
            {
                "id": prompt_id,
                "text": text,
                "prompt_token_ids": [prompt_index + 1],
                "generated_token_ids": list(range(10)),
                "logits": logits,
            }
        )
    reference = {
        "revision": packet["revision"],
        "config_sha256": packet["config_sha256"],
        "index_sha256": packet["index_sha256"],
        "tokenizer_sha256": packet["tokenizer_sha256"],
        "prompts_sha16": "99f702c72d2fafcc",
        "prompts": prompts,
    }
    reference["manifest_sha256"] = AUTH._canonical_sha256(reference)
    _write_json(evidence / "cpu-reference-manifest.json", reference)
    _write_json(
        evidence / "emitted-contract.json",
        {
            "topology": packet["topology"],
            "emitted_contract": packet["emitted_contract"],
        },
    )
    return evidence, repo


def test_static_packet_and_exact_compile_contract_are_valid() -> None:
    packet = _packet()
    AUTH.validate_packet(packet)
    AUTH.validate_compile_contract(packet, _contract())
    assert not any(packet["claims"].values())


def test_driver_validates_the_same_contract_before_any_side_effect() -> None:
    driver = (PACKAGE / "command.sh").read_text(encoding="utf-8")
    validator = driver.index("validate_compile_authorization.py")
    mkdir = driver.index('mkdir -p "$COMPILE_RUN_ROOT"')
    docker = driver.index("sudo docker run")
    assert validator < mkdir < docker
    assert '--compile-contract "$COMPILE_CONTRACT"' in driver
    assert '--evidence-root "$AUTH_EVIDENCE_ROOT"' in driver
    assert '--repository "$SRC_DIR"' in driver
    assert (
        "// 32" not in driver and "// 4096" not in driver and "// false" not in driver
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("stack", "container_digest"), "bad"),
        (("compile", "tp"), 16),
        (("compile", "logical_nc_config"), 1),
        (("compile", "ctx_batch_size"), 2),
        (("compile", "tkg_batch_size"), 2),
        (("compile", "sequence_buckets"), [8192]),
        (("emitted_contract", "rank_checkpoint_dtype"), "float16"),
        (("emitted_contract", "compute_dtype"), "float16"),
        (("emitted_contract", "cache_dtype"), "float16"),
        (("emitted_contract", "fp8_kv"), True),
        (("emitted_contract", "sampler"), "sample"),
        (("emitted_contract", "speculative_decode"), True),
        (("emitted_contract", "mtp"), True),
        (("emitted_contract", "dspark"), True),
    ],
)
def test_every_launch_field_drift_fails_closed(
    path: tuple[str, str], value: object
) -> None:
    contract = copy.deepcopy(_contract())
    contract[path[0]][path[1]] = value
    with pytest.raises(AUTH.AuthorizationError, match="not exactly authorized"):
        AUTH.validate_compile_contract(_packet(), contract)


def test_complete_proof_bearing_evidence_passes(tmp_path: Path) -> None:
    evidence, repo = _complete_evidence(tmp_path)
    assert AUTH.validate_evidence(_packet(), evidence, repo) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_shard",
        "duplicate_lfs",
        "duplicate_rank",
        "missing_logit",
        "bad_ancestry",
    ],
)
def test_incomplete_mutated_or_duplicate_evidence_fails(
    tmp_path: Path, mutation: str
) -> None:
    evidence, repo = _complete_evidence(tmp_path)
    if mutation in {"missing_shard", "duplicate_lfs", "bad_ancestry"}:
        path = evidence / "source-provenance.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "missing_shard":
            value["shards"].pop()
        elif mutation == "duplicate_lfs":
            value["shards"][1]["lfs_sha256"] = value["shards"][0]["lfs_sha256"]
        else:
            value["source_commit"] = "0" * 40
        _write_json(path, value)
    elif mutation == "duplicate_rank":
        path = evidence / "rank-inventory.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["ranks"][1]["inventory_sha256"] = value["ranks"][0]["inventory_sha256"]
        _write_json(path, value)
    else:
        path = evidence / "cpu-reference-manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["prompts"][0]["logits"].pop()
        _write_json(path, value)
    with pytest.raises(AUTH.AuthorizationError):
        AUTH.validate_evidence(_packet(), evidence, repo)


def test_missing_evidence_is_a_bounded_hold(tmp_path: Path) -> None:
    holds = AUTH.validate_evidence(_packet(), tmp_path, tmp_path)
    assert len(holds) == 5
    assert all(item.startswith("missing:") for item in holds)
