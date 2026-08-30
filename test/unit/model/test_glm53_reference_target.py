from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "vllm_neuron/model/glm53_flash/reference_target.py"
SPEC = importlib.util.spec_from_file_location("glm53_reference_target", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

VOCAB = MODULE.GLM53_RAW_CAPTURE_VOCAB_SIZE
COMMON = {
    "schema": MODULE.GLM53_REFERENCE_TARGET_SCHEMA,
    "reference_id": "original-target-test",
    "checkpoint_revision": MODULE.GLM53_CHECKPOINT_REVISION,
    "config_sha256": MODULE.GLM53_CONFIG_SHA256,
    "index_sha256": MODULE.GLM53_INDEX_SHA256,
    "semantics": "original-checkpoint-cpu-fp32",
    "dtype": "torch.float32",
    "vocab_size": VOCAB,
}


def _manifest(
    tmp_path: Path, *, payload: bytes | None = None, **changes: object
) -> Path:
    if payload is None:
        payload = torch.arange(VOCAB, dtype=torch.float32).numpy().tobytes()
    row_path = tmp_path / "rows/0-p0-0.bin"
    row_path.parent.mkdir()
    row_path.write_bytes(payload)
    row = {
        "slot": 0,
        "prompt_id": "p0",
        "position": 0,
        "dtype": "torch.float32",
        "shape": [VOCAB],
        "relative_path": "rows/0-p0-0.bin",
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
    }
    data = {**COMMON, "rows": [row], **changes}
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_original_target_manifest_loads_one_verified_full_vocab_row(
    tmp_path: Path,
) -> None:
    target = MODULE.Glm53ReferenceTarget.from_manifest(_manifest(tmp_path))
    row = target.load_row(slot=0, prompt_id="p0", position=0)
    assert tuple(row.shape) == (VOCAB,)
    assert row.dtype is torch.float32
    assert row[-1].item() == VOCAB - 1


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"checkpoint_revision": "0" * 40}, "checkpoint revision"),
        ({"semantics": "q4-diagnostic"}, "semantics"),
        ({"vocab_size": VOCAB - 1}, "vocabulary"),
    ],
)
def test_reference_manifest_rejects_identity_or_semantic_drift(
    tmp_path: Path, changes: dict[str, object], match: str
) -> None:
    with pytest.raises(MODULE.Glm53ReferenceTargetError, match=match):
        MODULE.Glm53ReferenceTarget.from_manifest(_manifest(tmp_path, **changes))


def test_reference_row_hash_drift_is_fail_closed(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    target = MODULE.Glm53ReferenceTarget.from_manifest(path)
    (tmp_path / "rows/0-p0-0.bin").write_bytes(b"corrupt")
    with pytest.raises(MODULE.Glm53ReferenceTargetError, match="hash drift"):
        target.load_row(slot=0, prompt_id="p0", position=0)
