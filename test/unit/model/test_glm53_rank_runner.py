from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
TOOL_PATH = ROOT / "tools" / "glm53_stream_rank_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("glm53_rank_runner", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _fake_streamer(calls: list[dict]) -> object:
    def stream(checkpoint, output, **kwargs):
        calls.append({"checkpoint": checkpoint, "output": output, **kwargs})
        return {"checkpoint_bytes": 123, "chunks_written": 1}

    return stream


def test_runner_emits_canonical_names_and_fixed_resource_bound(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "ranks"
    calls: list[dict] = []

    assert (
        RUNNER.main(
            [
                "--checkpoint-dir",
                str(checkpoint),
                "--output-dir",
                str(output),
                "--rank",
                "0",
                "--rank",
                "31",
            ],
            streamer=_fake_streamer(calls),
        )
        == 0
    )
    assert [call["output"].name for call in calls] == [
        "tp0_sharded_checkpoint.safetensors",
        "tp31_sharded_checkpoint.safetensors",
    ]
    assert all(call["tp_degree"] == 32 for call in calls)
    assert all(call["max_chunk_bytes"] == 64 * 1024 * 1024 for call in calls)


@pytest.mark.parametrize(
    "ranks",
    [[0, 0], [-1], [32]],
)
def test_runner_rejects_invalid_rank_set_before_streaming(
    tmp_path: Path, ranks: list[int]
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    calls: list[dict] = []
    rank_args = [item for rank in ranks for item in ("--rank", str(rank))]
    with pytest.raises(SystemExit) as exc:
        RUNNER.main(
            [
                "--checkpoint-dir",
                str(checkpoint),
                "--output-dir",
                str(tmp_path / "ranks"),
                *rank_args,
            ],
            streamer=_fake_streamer(calls),
        )
    assert exc.value.code == 2
    assert calls == []


def test_runner_rejects_existing_output_and_partial_before_any_rank(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "ranks"
    output.mkdir()
    (output / "tp0_sharded_checkpoint.safetensors").write_bytes(b"existing")
    (output / ".tp1_sharded_checkpoint.safetensors.partial-test").write_bytes(b"")
    calls: list[dict] = []
    with pytest.raises(SystemExit) as exc:
        RUNNER.main(
            [
                "--checkpoint-dir",
                str(checkpoint),
                "--output-dir",
                str(output),
                "--rank",
                "0",
                "--rank",
                "1",
            ],
            streamer=_fake_streamer(calls),
        )
    assert exc.value.code == 2
    assert calls == []


def test_runner_receipt_is_explicitly_non_authorizing(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    calls: list[dict] = []
    RUNNER.main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "ranks"),
            "--rank",
            "0",
        ],
        streamer=_fake_streamer(calls),
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["claims"] == {
        "canonical_rank_files_emitted": True,
        "card_launch_authorized": False,
        "correctness_40_of_40": False,
        "performance": False,
        "runtime_permitted": False,
    }
