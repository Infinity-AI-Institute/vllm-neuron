from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "vllm_neuron/model/glm53_flash/raw_capture.py"
SPEC = importlib.util.spec_from_file_location("glm53_raw_capture", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE

# raw_capture imports the converter through its package-relative path.  Load
# the package normally for this test so its real checkpoint identity is used.
SPEC.loader.exec_module(MODULE)


def _plan(**overrides):
    values = {
        "checkpoint_revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
        "emitted_config_sha256": "a" * 64,
        "rank_bundle_sha256": "b" * 64,
        "prompt_ids": ("prompt-a", "prompt-b"),
        "positions": (0, 1),
        "slot_count": 2,
        "vocab_size": 154_880,
    }
    values.update(overrides)
    return MODULE.Glm53RawCapturePlan(**values)


def _fill(capture, *, reference_id=None):
    logits = torch.full((154_880,), -100.0, dtype=torch.bfloat16)
    logits[123] = 100.0
    for slot in range(2):
        for prompt in ("prompt-a", "prompt-b"):
            for position in (0, 1):
                capture.record_logits(
                    slot=slot,
                    prompt_id=prompt,
                    position=position,
                    logits=logits,
                    reference_id=reference_id,
                )


def test_complete_capture_requires_every_slot_and_preserves_full_vocab():
    capture = MODULE.Glm53RawCapture(_plan())
    _fill(capture)
    receipt = capture.finalize()
    assert receipt["coverage"] == {
        "row_count": 8,
        "expected_row_count": 8,
        "all_slots": True,
        "full_vocabulary": True,
    }
    assert receipt["canonical_reference"]["bound"] is False
    assert receipt["claims"]["correctness_40_of_40"] is False


def test_peaked_logits_are_valid_but_nonfinite_and_truncated_rows_fail():
    capture = MODULE.Glm53RawCapture(
        _plan(slot_count=1, prompt_ids=("p",), positions=(0,))
    )
    logits = torch.full((154_880,), -100.0, dtype=torch.bfloat16)
    logits[17] = 100.0
    capture.record_logits(slot=0, prompt_id="p", position=0, logits=logits)
    with pytest.raises(MODULE.Glm53RawCaptureError, match="duplicate"):
        capture.record_logits(slot=0, prompt_id="p", position=0, logits=logits)

    bad = torch.zeros(154_879, dtype=torch.bfloat16)
    with pytest.raises(MODULE.Glm53RawCaptureError, match="full-vocabulary"):
        MODULE.Glm53RawCapture(
            _plan(slot_count=1, prompt_ids=("p",), positions=(1,))
        ).record_logits(slot=0, prompt_id="p", position=1, logits=bad)
    bad = torch.zeros(154_880, dtype=torch.bfloat16)
    bad[0] = float("nan")
    with pytest.raises(MODULE.Glm53RawCaptureError, match="non-finite"):
        MODULE.Glm53RawCapture(
            _plan(slot_count=1, prompt_ids=("p",), positions=(1,))
        ).record_logits(slot=0, prompt_id="p", position=1, logits=bad)


def test_reference_identity_is_single_bound_bank_and_missing_rows_fail():
    capture = MODULE.Glm53RawCapture(
        _plan(
            slot_count=1,
            prompt_ids=("p",),
            positions=(0,),
            canonical_reference_id="native-ref",
            canonical_reference_semantics="native-block-fp8",
        )
    )
    logits = torch.zeros(154_880, dtype=torch.bfloat16)
    with pytest.raises(MODULE.Glm53RawCaptureError, match="reference identity"):
        capture.record_logits(
            slot=0, prompt_id="p", position=0, logits=logits, reference_id="q4-ref"
        )
    capture.record_logits(
        slot=0, prompt_id="p", position=0, logits=logits, reference_id="native-ref"
    )
    assert capture.finalize()["canonical_reference"] == {
        "bound": True,
        "id": "native-ref",
        "semantics": "native-block-fp8",
    }

    incomplete = MODULE.Glm53RawCapture(
        _plan(slot_count=1, prompt_ids=("p",), positions=(0, 1))
    )
    incomplete.record_logits(slot=0, prompt_id="p", position=0, logits=logits)
    with pytest.raises(MODULE.Glm53RawCaptureError, match="missing rows"):
        incomplete.finalize()


def test_q4_reference_cannot_be_bound_as_the_glm_canonical_bank():
    with pytest.raises(MODULE.Glm53RawCaptureError, match="semantics"):
        _plan(
            canonical_reference_id="q4-diagnostic",
            canonical_reference_semantics="Q4_K_M",
        )


@pytest.mark.parametrize(
    "field,value", [("checkpoint_revision", "wrong"), ("vocab_size", 32000)]
)
def test_plan_rejects_target_identity_drift(field, value):
    with pytest.raises(MODULE.Glm53RawCaptureError):
        _plan(**{field: value})
