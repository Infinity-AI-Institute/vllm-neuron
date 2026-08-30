from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
MODULE_PATH = ROOT / "tools/glm53_reference_target_producer.py"
SPEC = importlib.util.spec_from_file_location(
    "glm53_reference_target_producer_cli", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _Spec:
    def __init__(self, **values):
        self.__dict__.update(values)
        self.expected_rows = tuple(range(40))


class _Producer:
    def __init__(self, _spec):
        pass


def _tokenize(prompt_id):
    return {"input_ids": [11, len(prompt_id)]}


def _load(_checkpoint):
    return object()


def _run(_model, _prompt_id, _positions, _token_ids):
    return []


def _bad_run(_model, _prompt_id, _positions):
    return []


def test_cli_dry_run_emits_exact_4x10_contract(tmp_path, capsys):
    checkpoint = tmp_path / "04c4e9e95c5da8862dced7e5056455116f83a7e0"
    checkpoint.mkdir()
    MODULE.main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "bank"),
            "--reference-id",
            "original-test",
            "--semantics",
            "native-block-fp8-dequantized-bfloat16",
            "--loader",
            f"{__name__}:_load",
            "--runner",
            f"{__name__}:_run",
            "--loader-version",
            "torch=2.9.1-test",
            "--loader-version",
            "producer=tiny-v1",
            "--dry-run",
        ],
        preflight=lambda _path: object(),
        producer_cls=_Producer,
        spec_cls=_Spec,
    )
    output = capsys.readouterr().out
    assert '"expected_rows": 40' in output
    assert '"vocab_size": 154880' in output
    assert '"device_used": false' in output
    assert '"weights_loaded": false' in output
    assert '"provider_bound": false' in output


def test_cli_rejects_duplicate_loader_version_before_provider_load(tmp_path):
    checkpoint = tmp_path / "04c4e9e95c5da8862dced7e5056455116f83a7e0"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="unique"):
        MODULE.main(
            [
                "--checkpoint-dir",
                str(checkpoint),
                "--output-dir",
                str(tmp_path / "bank"),
                "--reference-id",
                "original-test",
                "--semantics",
                "native-block-fp8-dequantized-bfloat16",
                "--loader",
                "test_loader:load",
                "--runner",
                "test_runner:run",
                "--loader-version",
                "torch=2.9.1-test",
                "--loader-version",
                "torch=2.9.1-other",
                "--dry-run",
            ],
            preflight=lambda _path: object(),
            producer_cls=_Producer,
            spec_cls=_Spec,
        )


def test_cli_dry_run_records_tokenizer_binding(tmp_path, capsys):
    checkpoint = tmp_path / "04c4e9e95c5da8862dced7e5056455116f83a7e0"
    checkpoint.mkdir()
    MODULE.main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "bank"),
            "--reference-id",
            "original-test",
            "--semantics",
            "native-block-fp8-dequantized-bfloat16",
            "--loader",
            f"{__name__}:_load",
            "--runner",
            f"{__name__}:_run",
            "--tokenizer",
            f"{__name__}:_tokenize",
            "--loader-version",
            "torch=2.9.1-test",
            "--tokenizer-version",
            "tokenizer=tiny-v1",
            "--dry-run",
        ],
        preflight=lambda _path: object(),
        producer_cls=_Producer,
        spec_cls=_Spec,
    )
    output = capsys.readouterr().out
    assert '"tokenizer_bound": true' in output
    assert '"provider_bound": true' in output
    assert '"tokenizer_versions": {\n    "tokenizer": "tiny-v1"' in output
    assert '"prompt_token_ids": {\n    "feedback-0": [\n      11,' in output


def test_cli_rejects_non_dry_run_without_tokenizer_binding(tmp_path):
    checkpoint = tmp_path / "04c4e9e95c5da8862dced7e5056455116f83a7e0"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="requires --tokenizer"):
        MODULE.main(
            [
                "--checkpoint-dir",
                str(checkpoint),
                "--output-dir",
                str(tmp_path / "bank"),
                "--reference-id",
                "original-test",
                "--semantics",
                "native-block-fp8-dequantized-bfloat16",
                "--loader",
                "test_loader:load",
                "--runner",
                "test_runner:run",
                "--loader-version",
                "torch=2.9.1-test",
            ],
            preflight=lambda _path: object(),
            producer_cls=_Producer,
            spec_cls=_Spec,
        )


def test_cli_dry_run_rejects_runner_signature_before_weights(tmp_path):
    checkpoint = tmp_path / "04c4e9e95c5da8862dced7e5056455116f83a7e0"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="runner provider"):
        MODULE.main(
            [
                "--checkpoint-dir",
                str(checkpoint),
                "--output-dir",
                str(tmp_path / "bank"),
                "--reference-id",
                "original-test",
                "--semantics",
                "native-block-fp8-dequantized-bfloat16",
                "--loader",
                f"{__name__}:_load",
                "--runner",
                f"{__name__}:_bad_run",
                "--tokenizer",
                f"{__name__}:_tokenize",
                "--loader-version",
                "torch=2.9.1-test",
                "--tokenizer-version",
                "tokenizer=tiny-v1",
                "--dry-run",
            ],
            preflight=lambda _path: object(),
            producer_cls=_Producer,
            spec_cls=_Spec,
        )
