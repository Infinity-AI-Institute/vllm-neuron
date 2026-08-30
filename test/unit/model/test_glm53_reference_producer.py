from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "vllm_neuron/model/glm53_flash/reference_producer.py"
SPEC = importlib.util.spec_from_file_location("glm53_reference_producer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

TARGET_SPEC = {
    "reference_id": "original-target-test",
    "checkpoint_dir": None,
    "loader_versions": {
        "torch": "2.9.1-test",
        "transformers": "4.57.6-test",
        "neuronx_distributed_inference": "0.10.18399-test",
        "producer": "tiny-original-v1",
    },
    "semantics": "original-checkpoint-cpu-fp32",
}


def _spec(tmp_path: Path, **changes):
    checkpoint = tmp_path / "04c4e9e95c5da8862dced7e5056455116f83a7e0"
    checkpoint.mkdir()
    values = {**TARGET_SPEC, "checkpoint_dir": checkpoint, **changes}
    return MODULE.Glm53OriginalTargetProducerSpec(**values)


def test_tiny_original_producer_emits_verified_4x10_full_vocab_bank(tmp_path: Path):
    spec = _spec(tmp_path)
    producer = MODULE.Glm53OriginalTargetProducer(spec)
    seen = []

    def loader(path):
        seen.append(path)
        return object()

    def run_prompt(_model, prompt_id, positions):
        return [
            torch.nn.functional.one_hot(
                torch.tensor((len(prompt_id) + position) % 154_880), 154_880
            ).to(torch.float32)
            for position in positions
        ]

    manifest_path = producer.produce(
        loader=loader, run_prompt=run_prompt, output_dir=tmp_path / "bank"
    )
    assert seen == [spec.checkpoint_dir]
    assert len(list((tmp_path / "bank" / "rows").glob("*.bin"))) == 40
    reference_module_path = ROOT / "vllm_neuron/model/glm53_flash/reference_target.py"
    reference_spec = importlib.util.spec_from_file_location(
        "glm53_reference_target_for_producer", reference_module_path
    )
    assert reference_spec is not None and reference_spec.loader is not None
    reference_module = importlib.util.module_from_spec(reference_spec)
    sys.modules[reference_spec.name] = reference_module
    reference_spec.loader.exec_module(reference_module)
    target = reference_module.Glm53ReferenceTarget.from_manifest(manifest_path)
    assert len(target.rows) == 40
    assert target.loader_versions["torch"] == "2.9.1-test"
    assert target.load_row(slot=0, prompt_id="feedback-0", position=0).shape == (
        154_880,
    )


def test_producer_passes_and_serializes_bound_prompt_tokens(tmp_path: Path):
    prompt_token_ids = {
        f"feedback-{index}": (101 + index, 201 + index) for index in range(4)
    }
    spec = _spec(
        tmp_path,
        tokenizer_versions={"tokenizer": "tiny-tokenizer-v1"},
        prompt_token_ids=prompt_token_ids,
    )
    seen = {}

    def run_prompt(_model, prompt_id, positions, token_ids):
        seen[prompt_id] = token_ids
        return [torch.zeros(154_880, dtype=torch.float32) for _ in positions]

    manifest_path = MODULE.Glm53OriginalTargetProducer(spec).produce(
        loader=lambda _path: object(),
        run_prompt=run_prompt,
        output_dir=tmp_path / "bound-bank",
    )
    manifest = __import__("json").loads(manifest_path.read_text())
    assert seen == prompt_token_ids
    assert manifest["tokenizer_versions"] == {"tokenizer": "tiny-tokenizer-v1"}
    assert manifest["prompt_token_ids"] == {
        prompt_id: list(token_ids) for prompt_id, token_ids in prompt_token_ids.items()
    }


def test_producer_rejects_wrong_shape_and_publishes_nothing(tmp_path: Path):
    spec = _spec(tmp_path)
    producer = MODULE.Glm53OriginalTargetProducer(spec)

    def run_prompt(_model, _prompt_id, positions):
        return [torch.zeros(154_879, dtype=torch.float32) for _ in positions]

    output = tmp_path / "bank"
    with pytest.raises(MODULE.Glm53ReferenceProducerError, match="full vocabulary"):
        producer.produce(
            loader=lambda _path: object(), run_prompt=run_prompt, output_dir=output
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".bank.partial-*"))


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"prompt_ids": ("only-one",)}, "exactly four"),
        ({"positions": (0, 1)}, "positions 0 through 9"),
        ({"loader_versions": {}}, "loader versions"),
        ({"semantics": "Q4_K_M"}, "semantics"),
        ({"tokenizer_versions": {"tokenizer": "v1"}}, "together"),
        ({"prompt_token_ids": {"feedback-0": (1,)}}, "together"),
    ],
)
def test_producer_spec_rejects_feedback_or_identity_drift(tmp_path, changes, match):
    with pytest.raises(MODULE.Glm53ReferenceProducerError, match=match):
        _spec(tmp_path, **changes)
