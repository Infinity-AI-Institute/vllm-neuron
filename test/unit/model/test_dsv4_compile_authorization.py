from __future__ import annotations

import copy
import importlib.util
import json
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


def test_static_packet_is_valid_and_fail_closed() -> None:
    packet = _packet()
    AUTH.validate_packet(packet)
    assert not any(packet["claims"].values())
    assert all(not item["satisfied"] for item in packet["blockers"])


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda p: p["topology"].update(tp_degree=16), "TP must be 32"),
        (lambda p: p["emitted_contract"].update(fp8_kv=True), "FP8 KV"),
        (lambda p: p["blockers"][0].update(satisfied=True), "pre-authorize"),
        (lambda p: p["claims"].update(compiled=True), "claims must remain false"),
    ],
)
def test_semantic_drift_fails_closed(mutation, match: str) -> None:
    packet = copy.deepcopy(_packet())
    mutation(packet)
    with pytest.raises(AUTH.AuthorizationError, match=match):
        AUTH.validate_packet(packet)


def test_missing_evidence_is_a_bounded_hold(tmp_path: Path) -> None:
    holds = AUTH.validate_evidence(_packet(), tmp_path)
    assert len(holds) == 5
    assert all(item.startswith("missing:") for item in holds)
