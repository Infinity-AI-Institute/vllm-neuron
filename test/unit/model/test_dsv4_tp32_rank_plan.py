from __future__ import annotations

import copy
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "vllm_neuron/model/dsv4_flash/audit_tp32_rank_plan.py"
SPEC = importlib.util.spec_from_file_location("dsv4_rank_plan", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

VALIDATOR_SCRIPT = ROOT / "vllm_neuron/model/dsv4_flash/validate_tp32_rank_plan.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "dsv4_rank_plan_validator", VALIDATOR_SCRIPT
)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)
RECEIPT = ROOT / "vllm_neuron/model/dsv4_flash/evidence/tp32-rank-plan-audit.json"


def _write_fixture(root: Path, rows: dict[str, tuple[str, list[int]]]) -> None:
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    weight_map = {key: "model-00001-of-00001.safetensors" for key in rows}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    cursor = 0
    header = {}
    for key, (dtype, shape) in rows.items():
        size = MODULE.SAFE_BYTES[dtype]
        for dim in shape:
            size *= dim
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    encoded = json.dumps(header).encode()
    with (root / "model-00001-of-00001.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)


def test_target_plan_is_exact_tp32_and_rank_bound() -> None:
    targets = MODULE.build_targets()
    assert len(targets) == 1024
    assert len({row.name for row in targets}) == 1024
    gate = next(
        row
        for row in targets
        if row.name.endswith("expert_mlps.mlp_op.gate_up_proj.weight")
    )
    down = next(
        row
        for row in targets
        if row.name.endswith("expert_mlps.mlp_op.down_proj.weight")
    )
    assert gate.shape == (256, 4096, 128)
    assert down.shape == (256, 64, 4096)
    assert gate.ownership == "expert_axis_replicated_intermediate_tp_sharded"
    rows = [item.canonical() for item in targets]
    assert MODULE.canonical_sha256(
        {"rank": 0, "tp_degree": 32, "tensors": rows}
    ) != MODULE.canonical_sha256({"rank": 31, "tp_degree": 32, "tensors": rows})


def test_source_classifier_rejects_mhc_and_i64_hash_routes() -> None:
    headers = {
        "embed.weight": MODULE.HeaderSpec("BF16", (129280, 4096), "s"),
        "layers.0.hc_attn_base": MODULE.HeaderSpec("F32", (24,), "s"),
        "layers.0.ffn.gate.tid2eid": MODULE.HeaderSpec("I64", (129280, 6), "s"),
        "layers.0.attn.wq_a.weight": MODULE.HeaderSpec("F8_E4M3", (1024, 4096), "s"),
        "layers.0.attn.wq_a.scale": MODULE.HeaderSpec("F8_E8M0", (8, 32), "s"),
        "mtp.0.attn_norm.weight": MODULE.HeaderSpec("BF16", (4096,), "s"),
    }
    result = MODULE.classify_sources(headers)
    assert result["unmapped_mhc"] == ["layers.0.hc_attn_base"]
    assert result["incompatible_hash_route_dtype"] == ["layers.0.ffn.gate.tid2eid"]
    assert result["support_scale"] == ["layers.0.attn.wq_a.scale"]
    assert result["dropped_mtp_or_speculation"] == ["mtp.0.attn_norm.weight"]


def test_header_reader_rejects_duplicate_route_and_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "snapshot"
    _write_fixture(root, {"weight": ("BF16", [2, 2])})
    monkeypatch.setattr(MODULE, "CONFIG_SHA256", MODULE.sha256(root / "config.json"))
    monkeypatch.setattr(
        MODULE, "INDEX_SHA256", MODULE.sha256(root / "model.safetensors.index.json")
    )
    monkeypatch.setattr(MODULE, "SOURCE_TENSORS", 1)
    monkeypatch.setattr(MODULE, "SHARDS", 1)
    with pytest.raises(MODULE.RankPlanError, match="directory identity"):
        MODULE.read_headers(root)
    _, headers, _ = MODULE.read_headers(root, test_only_allow_unpinned=True)
    assert headers["weight"].nbytes == 8
    shard = root / "model-00001-of-00001.safetensors"
    data = bytearray(shard.read_bytes())
    body = json.loads(data[8:])
    body["weight"]["data_offsets"] = [0, 7]
    encoded = json.dumps(body).encode()
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    with pytest.raises(MODULE.RankPlanError, match="byte-count drift"):
        MODULE.read_headers(root, test_only_allow_unpinned=True)


def test_duplicate_header_json_key_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    index = {"weight_map": {"weight": "model-00001-of-00001.safetensors"}}
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    row = '{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}'
    header = f'{{"weight":{row},"weight":{row}}}'.encode()
    with (root / "model-00001-of-00001.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
    monkeypatch.setattr(MODULE, "CONFIG_SHA256", MODULE.sha256(root / "config.json"))
    monkeypatch.setattr(
        MODULE, "INDEX_SHA256", MODULE.sha256(root / "model.safetensors.index.json")
    )
    monkeypatch.setattr(MODULE, "SOURCE_TENSORS", 1)
    monkeypatch.setattr(MODULE, "SHARDS", 1)
    with pytest.raises(MODULE.RankPlanError, match="duplicate JSON key"):
        MODULE.read_headers(root, test_only_allow_unpinned=True)


def test_exact_host_only_hold_receipt_passes() -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    VALIDATOR.validate_receipt(receipt)
    assert receipt["complete"] is False
    assert receipt["compile_permitted"] is False
    assert not any(receipt["claims"].values())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("source", "revision"), "0" * 40),
        (("source", "tensor_payload_bytes_read"), 1),
        (("routing", "target_tensor_count_per_rank"), 1023),
        (("routing", "target_bytes_per_rank"), 1),
        (("routing", "ownership_counts", "replicated"), 548),
        (("routing", "moe_ownership", "expert_axis_partitioned_by_ep"), True),
        (("blockers", "unmapped_mhc", "count"), 0),
        (("blockers", "incompatible_hash_route_dtype", "count"), 0),
        (("claims", "rank_files_materialized"), True),
        (("compile_permitted",), True),
    ],
)
def test_receipt_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    target = receipt
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    with pytest.raises(VALIDATOR.RankPlanValidationError):
        VALIDATOR.validate_receipt(receipt)


def test_rank_reorder_and_tool_substitution_fail_closed(tmp_path: Path) -> None:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    swapped = copy.deepcopy(receipt)
    swapped["ranks"][0], swapped["ranks"][1] = swapped["ranks"][1], swapped["ranks"][0]
    with pytest.raises(VALIDATOR.RankPlanValidationError, match="rank order"):
        VALIDATOR.validate_receipt(swapped)
    substituted = tmp_path / "audit.py"
    substituted.write_text("# substituted\n", encoding="utf-8")
    with pytest.raises(VALIDATOR.RankPlanValidationError, match="tool drift"):
        VALIDATOR.validate_receipt(receipt, substituted)
