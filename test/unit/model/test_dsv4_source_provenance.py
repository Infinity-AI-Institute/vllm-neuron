from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
PACKAGE = ROOT / "vllm_neuron" / "model" / "dsv4_flash"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load("dsv4_source_audit", PACKAGE / "audit_source_provenance.py")
AUTH = _load("dsv4_source_authorization", PACKAGE / "validate_compile_authorization.py")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_tensor(key: str, payload: bytes = b"x") -> bytes:
    header = json.dumps(
        {
            key: {
                "dtype": "U8",
                "shape": [len(payload)],
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode()
    return struct.pack("<Q", len(header)) + header + payload


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    model = tmp_path / "DeepSeek-V4-Flash-0731--7872f01b1d1fe23"
    model.mkdir()
    (model / ".complete").write_bytes(b"")
    small = {
        "config.json": b"config",
        "tokenizer.json": b"tokenizer",
    }
    weight_map: dict[str, str] = {}
    shard_bytes: dict[str, bytes] = {}
    for index in range(1, 49):
        name = f"model-{index:05d}-of-00048.safetensors"
        key = f"model.tensor.{index:05d}"
        weight_map[key] = name
        shard_bytes[name] = _safe_tensor(key)
    small["model.safetensors.index.json"] = json.dumps(
        {"weight_map": weight_map}, separators=(",", ":")
    ).encode()
    for name, value in small.items():
        (model / name).write_bytes(value)
    for name, value in shard_bytes.items():
        (model / name).write_bytes(value)

    lines = [
        f"{_sha(value)}  ./{name}"
        for name, value in sorted({**small, **shard_bytes}.items())
    ]
    manifest = tmp_path / "all-files.sha256"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(AUDIT, "CONFIG_SHA256", _sha(small["config.json"]))
    monkeypatch.setattr(
        AUDIT, "INDEX_SHA256", _sha(small["model.safetensors.index.json"])
    )
    monkeypatch.setattr(AUDIT, "TOKENIZER_SHA256", _sha(small["tokenizer.json"]))
    monkeypatch.setattr(AUDIT, "EXPECTED_TENSORS", 48)
    monkeypatch.setattr(
        AUDIT,
        "EXPECTED_PAYLOAD_BYTES",
        sum(len(value) for value in shard_bytes.values()),
    )
    monkeypatch.setattr(
        AUDIT, "EXPECTED_PAYLOAD_MANIFEST_SHA256", AUDIT.sha256_file(manifest)
    )
    return model, manifest


def test_committed_production_source_receipt_passes() -> None:
    packet = json.loads(
        (PACKAGE / "tp32_compile_authorization.json").read_text(encoding="utf-8")
    )
    AUTH.validate_packet(packet)
    AUTH.validate_production_source_evidence(packet, PACKAGE, ROOT)
    assert {
        item["id"]: item["satisfied"] for item in packet["blockers"]
    } == AUTH.EXPECTED_SATISFIED
    assert not any(packet["claims"].values())


def test_header_only_audit_produces_complete_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, manifest = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "output"
    result = AUDIT.audit(model, manifest, output, "a" * 64)
    source = json.loads((output / "source-provenance.json").read_text())
    assert result["shards"] == 48
    assert result["tensors"] == 48
    assert source["payload_bytes_read_during_header_audit"] == 0
    assert source["audit"]["rank_conversion_performed"] is False
    assert source["audit"]["large_outputs_created"] is False
    assert source["missing"] == source["orphan"] == source["misrouted"] == []
    assert len({item["header_sha256"] for item in source["shards"]}) == 48


@pytest.mark.parametrize(
    "mutation",
    [
        "index_misroute",
        "missing_shard",
        "duplicate_payload",
        "duplicate_header_key",
        "header_truncation",
    ],
)
def test_header_audit_rejects_adversarial_checkpoint_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    model, manifest = _fixture(tmp_path, monkeypatch)
    if mutation == "index_misroute":
        index_path = model / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        first, second = list(index["weight_map"])[:2]
        index["weight_map"][first], index["weight_map"][second] = (
            index["weight_map"][second],
            index["weight_map"][first],
        )
        index_path.write_text(json.dumps(index, separators=(",", ":")))
        monkeypatch.setattr(AUDIT, "INDEX_SHA256", AUDIT.sha256_file(index_path))
    elif mutation == "missing_shard":
        (model / "model-00048-of-00048.safetensors").unlink()
    elif mutation == "duplicate_payload":
        lines = manifest.read_text().splitlines()
        first_hash = lines[3].split()[0]
        digest, _, name = lines[4].partition("  ./")
        assert digest != first_hash
        lines[4] = f"{first_hash}  ./{name}"
        manifest.write_text("\n".join(lines) + "\n")
        monkeypatch.setattr(
            AUDIT, "EXPECTED_PAYLOAD_MANIFEST_SHA256", AUDIT.sha256_file(manifest)
        )
    elif mutation == "duplicate_header_key":
        descriptor = '{"dtype":"U8","shape":[1],"data_offsets":[0,1]}'
        body = (
            '{"model.tensor.00001":'
            + descriptor
            + ',"model.tensor.00001":'
            + descriptor
            + "}"
        ).encode()
        (model / "model-00001-of-00048.safetensors").write_bytes(
            struct.pack("<Q", len(body)) + body + b"x"
        )
    else:
        (model / "model-00001-of-00048.safetensors").write_bytes(b"short")
    with pytest.raises(AUDIT.AuditError):
        AUDIT.audit(model, manifest, tmp_path / "output", "a" * 64)


@pytest.mark.parametrize(
    ("artifact", "mutation"),
    [
        ("source_provenance", b"\n"),
        ("routing_manifest", b"\n"),
        ("checkpoint_payload_manifest", b"\n"),
        ("audit_tool", b"\n# mutation\n"),
        ("tp32_rank_plan_audit", b"\n"),
        ("tp32_rank_plan_validator", b"\n# mutation\n"),
    ],
)
def test_every_bound_production_artifact_rejects_mutation(
    tmp_path: Path, artifact: str, mutation: bytes
) -> None:
    package = tmp_path / "package"
    evidence = package / "evidence"
    evidence.mkdir(parents=True)
    for source in (PACKAGE / "evidence").iterdir():
        (evidence / source.name).write_bytes(source.read_bytes())
    (package / "audit_source_provenance.py").write_bytes(
        (PACKAGE / "audit_source_provenance.py").read_bytes()
    )
    (package / "audit_tp32_rank_plan.py").write_bytes(
        (PACKAGE / "audit_tp32_rank_plan.py").read_bytes()
    )
    (package / "validate_tp32_rank_plan.py").write_bytes(
        (PACKAGE / "validate_tp32_rank_plan.py").read_bytes()
    )
    packet = json.loads(
        (PACKAGE / "tp32_compile_authorization.json").read_text(encoding="utf-8")
    )
    relative = packet["production_evidence"][artifact]["path"]
    path = package / relative
    path.write_bytes(path.read_bytes() + mutation)
    with pytest.raises(AUTH.AuthorizationError, match="hash drift"):
        AUTH.validate_production_source_evidence(packet, package, ROOT)


def test_four_unrelated_production_receipts_remain_holds() -> None:
    packet = json.loads(
        (PACKAGE / "tp32_compile_authorization.json").read_text(encoding="utf-8")
    )
    holds = {item["id"] for item in packet["blockers"] if not item["satisfied"]}
    assert holds == {
        "tp32_rank_inventory",
        "compiler_inventory",
        "cpu_reference_bank",
        "emitted_contract_receipt",
    }
    assert packet["claims"]["compile_permitted"] is False
