import json

import pytest
import torch
from safetensors.torch import save_file

from vllm_neuron.utils.checkpoints import (
    SafetensorsCheckpoint,
    _LocalCheckpointSource,
)


class _SingleWeight(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(2, dtype=torch.bfloat16),
            requires_grad=False,
        )


def _write_duplicate_key_checkpoint(tmp_path) -> tuple[str, str]:
    base_name = "model-00001-of-00001.safetensors"
    overlay_name = "bf16-shared-layer-003.safetensors"
    save_file(
        {
            "weight": torch.tensor([1.0, 1.0], dtype=torch.bfloat16),
            "base_only": torch.tensor([3.0], dtype=torch.float32),
        },
        tmp_path / base_name,
    )
    save_file(
        {"weight": torch.tensor([2.0, 2.0], dtype=torch.bfloat16)},
        tmp_path / overlay_name,
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "weight": overlay_name,
                    "base_only": base_name,
                },
            }
        ),
        encoding="utf-8",
    )
    return base_name, overlay_name


@pytest.mark.parametrize("discover", [False, True])
def test_local_source_honors_safetensors_index_with_duplicate_keys(tmp_path, discover):
    base_name, overlay_name = _write_duplicate_key_checkpoint(tmp_path)

    source = _LocalCheckpointSource(str(tmp_path), ".safetensors")
    assert source.get_file_names() == [overlay_name, base_name]
    assert source.contains_tensor(overlay_name, "weight")
    assert not source.contains_tensor(base_name, "weight")

    checkpoint = SafetensorsCheckpoint(str(tmp_path))
    if discover:
        assert checkpoint.get_tensor_names() == {"weight", "base_only"}
    result = checkpoint.load_sharded(
        rank=0,
        world_size=1,
        model=_SingleWeight(),
        mappings={},
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        result.state_dict["weight"],
        torch.tensor([2.0, 2.0], dtype=torch.bfloat16),
    )
    assert result.missing_keys == []
    assert result.unexpected_keys == ["base_only"]


def _run_pipelined_duplicate_key_test(tmp_path, monkeypatch, *, discover: bool):
    _write_duplicate_key_checkpoint(tmp_path)
    store = torch.distributed.HashStore()
    monkeypatch.setattr(
        torch.distributed.distributed_c10d,
        "_get_default_store",
        lambda: store,
    )

    checkpoint = SafetensorsCheckpoint(str(tmp_path))
    if discover:
        assert checkpoint.get_tensor_names() == {"weight", "base_only"}
    result = checkpoint.load_sharded_pipelined(
        rank=0,
        world_size=1,
        model=_SingleWeight(),
        mappings={},
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        result.state_dict["weight"],
        torch.tensor([2.0, 2.0], dtype=torch.bfloat16),
    )
    assert result.missing_keys == []
    assert result.unexpected_keys == ["base_only"]


@pytest.mark.parametrize("discover", [False, True])
def test_pipelined_loader_honors_index_with_duplicate_keys(
    tmp_path, monkeypatch, discover
):
    _run_pipelined_duplicate_key_test(
        tmp_path,
        monkeypatch,
        discover=discover,
    )


def test_local_source_without_index_keeps_directory_scan_behavior(tmp_path):
    save_file(
        {"weight": torch.tensor([1.0, 1.0], dtype=torch.bfloat16)},
        tmp_path / "model.safetensors",
    )

    source = _LocalCheckpointSource(str(tmp_path), ".safetensors")

    assert source.get_file_names() == ["model.safetensors"]
    assert source.contains_tensor("model.safetensors", "weight")


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../outside.safetensors",
        r"..\outside.safetensors",
        "nested/shard.safetensors",
        "/absolute/shard.safetensors",
        "model.bin",
    ],
)
def test_local_source_rejects_unsafe_index_shard_paths(tmp_path, unsafe_name):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": unsafe_name}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsafe or non-safetensors"):
        _LocalCheckpointSource(str(tmp_path), ".safetensors")
