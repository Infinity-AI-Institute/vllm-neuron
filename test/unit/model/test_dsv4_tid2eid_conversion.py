from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[3]
PACKAGE = ROOT / "vllm_neuron" / "model" / "dsv4_flash"
PACKAGE_NAME = "_dsv4_tid2eid_test_package"


def _load(name: str):
    if PACKAGE_NAME not in sys.modules:
        package = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(PACKAGE)]
        sys.modules[PACKAGE_NAME] = package
    qualified = f"{PACKAGE_NAME}.{name}"
    spec = importlib.util.spec_from_file_location(qualified, PACKAGE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


_load("config")
CONVERTER = _load("checkpoint_convert")


def _convert(value: torch.Tensor, *, n_experts: int = 256):
    return CONVERTER._convert_hash_tid2eid(
        value,
        key="layers.0.ffn.gate.tid2eid",
        vocab_size=4,
        top_k=3,
        n_experts=n_experts,
    )


def test_i64_checkpoint_table_is_losslessly_normalized_to_i32() -> None:
    source = torch.tensor(
        [[0, 1, 255], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        dtype=torch.int64,
    )
    converted, report = _convert(source)
    assert converted.dtype == torch.int32
    assert converted.is_contiguous()
    assert torch.equal(converted.to(torch.int64), source)
    assert converted.data_ptr() != source.data_ptr()
    assert report == {
        "source_dtype": "torch.int64",
        "target_dtype": "torch.int32",
        "shape": (4, 3),
        "min_expert_index": 0,
        "max_expert_index": 255,
        "lossless_i64_to_i32": True,
    }


def test_i32_fixture_is_cloned_without_semantic_drift() -> None:
    source = torch.arange(12, dtype=torch.int32).reshape(4, 3)
    converted, report = _convert(source)
    assert torch.equal(converted, source)
    assert converted.data_ptr() != source.data_ptr()
    assert report["lossless_i64_to_i32"] is False


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (torch.zeros((3, 4), dtype=torch.int64), "shape"),
        (torch.zeros((4, 3), dtype=torch.float32), "not int32/int64"),
        (torch.tensor([[-1, 0, 1]] * 4, dtype=torch.int64), "out-of-range"),
        (torch.tensor([[0, 1, 256]] * 4, dtype=torch.int64), "out-of-range"),
    ],
)
def test_shape_dtype_and_expert_range_drift_fail_closed(
    value: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _convert(value)


def test_wrapper_i32_domain_is_checked_before_conversion() -> None:
    source = torch.zeros((4, 3), dtype=torch.int64)
    with pytest.raises(ValueError, match="cannot be represented"):
        _convert(source, n_experts=torch.iinfo(torch.int32).max + 2)


def test_empty_contract_dimensions_fail_before_tensor_reduction() -> None:
    source = torch.empty((0, 3), dtype=torch.int64)
    with pytest.raises(ValueError, match="invalid tid2eid contract dimensions"):
        CONVERTER._convert_hash_tid2eid(
            source,
            key="layers.0.ffn.gate.tid2eid",
            vocab_size=0,
            top_k=3,
            n_experts=256,
        )
