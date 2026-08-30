from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
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


def _git_repository(root: Path) -> tuple[Path, str, str, str]:
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
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True
    ).strip()
    return repo, source, merge, tree


def _complete_evidence(
    tmp_path: Path,
    *,
    packet: dict | None = None,
    shards: list[dict] | None = None,
    model_files: dict | None = None,
) -> tuple[Path, Path]:
    packet = packet or _packet()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    repo, source_commit, merge_sha, merge_tree_sha = _git_repository(tmp_path)

    if shards is None:
        shards = []
        for index in range(48):
            shards.append(
                {
                    "name": f"model-{index + 1:05d}-of-00048.safetensors",
                    "lfs_sha256": _sha_bytes(f"lfs-{index}".encode()),
                    "size": 1 if index < 47 else packet["source_total_bytes"] - 47,
                    "header_sha256": _sha_bytes(f"header-{index}".encode()),
                    "header_bytes": 8 + index,
                    "tensor_count": (
                        1 if index < 47 else packet["source_tensor_count"] - 47
                    ),
                    "routing_sha256": _sha_bytes(f"routing-{index}".encode()),
                }
            )
    if model_files is None:
        model_files = {
            "config.json": {"sha256": packet["config_sha256"], "bytes": 1},
            "model.safetensors.index.json": {
                "sha256": packet["index_sha256"],
                "bytes": 1,
            },
            "tokenizer.json": {"sha256": packet["tokenizer_sha256"], "bytes": 1},
        }
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
        "validator_merge_tree_sha": merge_tree_sha,
        "validator_merged_to_agent_main": True,
        "revision": packet["revision"],
        "config_sha256": packet["config_sha256"],
        "index_sha256": packet["index_sha256"],
        "model_files": model_files,
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


def _launch_fixture(
    tmp_path: Path,
) -> tuple[dict, Path, Path, Path, Path, Path]:
    packet = copy.deepcopy(_packet())
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_files = {}
    packet_fields = {
        "config.json": "config_sha256",
        "model.safetensors.index.json": "index_sha256",
        "tokenizer.json": "tokenizer_sha256",
    }
    for name, packet_field in packet_fields.items():
        path = model_dir / name
        path.write_bytes(f"reviewed-{name}".encode())
        identity = {"sha256": AUTH._sha256_file(path), "bytes": path.stat().st_size}
        model_files[name] = identity
        packet[packet_field] = identity["sha256"]

    shards = []
    tensor_counts = [1] * 47 + [packet["source_tensor_count"] - 47]
    for index in range(48):
        path = model_dir / f"model-{index + 1:05d}-of-00048.safetensors"
        path.write_bytes(f"reviewed-shard-{index}".encode())
        shards.append(
            {
                "name": path.name,
                "lfs_sha256": AUTH._sha256_file(path),
                "size": path.stat().st_size,
                "header_sha256": _sha_bytes(f"header-{index}".encode()),
                "header_bytes": 8 + index,
                "tensor_count": tensor_counts[index],
                "routing_sha256": _sha_bytes(f"routing-{index}".encode()),
            }
        )
    packet["source_total_bytes"] = sum(item["size"] for item in shards)
    evidence, repo = _complete_evidence(
        tmp_path, packet=packet, shards=shards, model_files=model_files
    )

    rank_source = tmp_path / "run" / "weights"
    rank_source.mkdir(parents=True)
    inventory = json.loads(
        (evidence / "rank-inventory.json").read_text(encoding="utf-8")
    )
    for rank in inventory["ranks"]:
        source = evidence / rank["checkpoint"]["path"]
        destination = rank_source / f"tp{rank['rank']}_sharded_checkpoint.safetensors"
        shutil.copyfile(source, destination)
    return packet, evidence, repo, model_dir, rank_source, rank_source.parent


def test_static_packet_and_exact_compile_contract_are_valid() -> None:
    packet = _packet()
    AUTH.validate_packet(packet)
    AUTH.validate_compile_contract(packet, _contract())
    assert not any(packet["claims"].values())


def test_committed_read_only_host_inventory_is_capacity_safe() -> None:
    packet = _packet()
    inventory = json.loads(
        (PACKAGE / "evidence" / "host-inventory-20260829.json").read_text(
            encoding="utf-8"
        )
    )
    AUTH._validate_host_inventory(packet, inventory)
    capacity = inventory["hosts"]["trn2"]["rank_materialization_capacity"]
    assert capacity["fits_with_reserve"] is True
    assert (
        capacity["usable_headroom_after_output_and_reserve_bytes"] > 1_000_000_000_000
    )
    assert (
        inventory["hosts"]["r7i"]["filesystem"]["available_bytes"]
        < capacity["output_bytes"]
    )


@pytest.mark.parametrize(
    "mutation",
    ["host_write", "checkpoint_revision", "capacity", "compiler", "oracle_overclaim"],
)
def test_host_inventory_tampering_fails_closed(mutation: str) -> None:
    inventory = json.loads(
        (PACKAGE / "evidence" / "host-inventory-20260829.json").read_text(
            encoding="utf-8"
        )
    )
    if mutation == "host_write":
        inventory["audit_boundary"]["files_created_on_hosts"] = 1
    elif mutation == "checkpoint_revision":
        inventory["hosts"]["trn2"]["checkpoint"]["revision"] = "0" * 40
    elif mutation == "capacity":
        inventory["hosts"]["trn2"]["rank_materialization_capacity"]["output_bytes"] -= 1
    elif mutation == "compiler":
        inventory["compiler_stack"]["versions_from_immutable_image_history"][
            "neuronx-cc"
        ] = "unknown"
    else:
        inventory["formal_holds"]["cpu_reference_bank"]["resolved"] = True
    with pytest.raises(AUTH.AuthorizationError):
        AUTH._validate_host_inventory(_packet(), inventory)


def test_driver_validates_the_same_contract_before_any_side_effect() -> None:
    driver = (PACKAGE / "command.sh").read_text(encoding="utf-8")
    validator = driver.index("validate_compile_authorization.py")
    mkdir = driver.index('mkdir -p "$COMPILE_RUN_ROOT"')
    copy = driver.index('cp -al "$COMPILE_RUN_ROOT/weights/')
    docker = driver.index("sudo docker run")
    assert validator < mkdir < copy < docker
    assert '--compile-contract "$COMPILE_CONTRACT"' in driver
    assert '--evidence-root "$AUTH_EVIDENCE_ROOT"' in driver
    assert '--model-dir "$MODEL_DIR"' in driver
    assert '--compile-run-root "$COMPILE_RUN_ROOT"' in driver
    assert '--rank-source "$RANK_SOURCE_DIR"' in driver
    assert '--source-dir "$SRC_DIR"' in driver
    assert "--repository " not in driver
    assert 'RANK_SOURCE_DIR="$(realpath "$COMPILE_RUN_ROOT/weights")"' in driver
    assert "${MODEL_DIR:-" not in driver
    assert "${SRC_DIR:-" not in driver
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


def test_exact_reviewed_launch_inputs_pass(tmp_path: Path) -> None:
    packet, evidence, repo, model_dir, rank_source, _ = _launch_fixture(tmp_path)
    assert AUTH.validate_evidence(packet, evidence, repo) == []
    AUTH.validate_launch_bindings(
        packet, evidence, model_dir, rank_source.parent, rank_source, repo
    )


@pytest.mark.parametrize(
    "mutation",
    ["swapped_model_dir", "substituted_rank", "different_head", "dirty_checkout"],
)
def test_adversarial_launch_input_rejects_before_side_effect(
    tmp_path: Path, mutation: str
) -> None:
    packet, evidence, repo, model_dir, rank_source, run_root = _launch_fixture(tmp_path)
    forbidden = [
        run_root / "cache",
        run_root / "work",
        run_root / "artifacts",
        run_root / "logs",
    ]
    assert not any(path.exists() for path in forbidden)

    if mutation == "swapped_model_dir":
        swapped = tmp_path / "swapped-model"
        shutil.copytree(model_dir, swapped)
        (swapped / "config.json").write_bytes(b"substituted-config")
        model_dir = swapped
    elif mutation == "substituted_rank":
        (rank_source / "tp7_sharded_checkpoint.safetensors").write_bytes(
            b"substituted-rank"
        )
    elif mutation == "different_head":
        (repo / "different").write_text("different", encoding="utf-8")
        subprocess.run(["git", "add", "different"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "different"], cwd=repo, check=True)
    else:
        (repo / "untracked").write_text("dirty", encoding="utf-8")

    with pytest.raises(AUTH.AuthorizationError):
        AUTH.validate_launch_bindings(
            packet, evidence, model_dir, run_root, rank_source, repo
        )
    assert not any(path.exists() for path in forbidden)


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


def test_rank_evidence_binds_paths_to_rank_identity(tmp_path: Path) -> None:
    evidence, repo = _complete_evidence(tmp_path)
    rank0 = evidence / "ranks/tp0.safetensors"
    rank1 = evidence / "ranks/tp1.safetensors"
    rank0_bytes = rank0.read_bytes()
    rank0.write_bytes(rank1.read_bytes())
    rank1.write_bytes(rank0_bytes)

    inventory_path = evidence / "rank-inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["ranks"][0]["checkpoint"]["path"] = "ranks/tp1.safetensors"
    inventory["ranks"][1]["checkpoint"]["path"] = "ranks/tp0.safetensors"
    inventory["canonical_rank_inventory_sha256"] = AUTH._canonical_sha256(
        inventory["ranks"]
    )
    _write_json(inventory_path, inventory)

    with pytest.raises(AUTH.AuthorizationError, match="path must bind"):
        AUTH.validate_evidence(_packet(), evidence, repo)


def test_missing_evidence_is_a_bounded_hold(tmp_path: Path) -> None:
    holds = AUTH.validate_evidence(_packet(), tmp_path, tmp_path)
    assert len(holds) == 5
    assert all(item.startswith("missing:") for item in holds)
