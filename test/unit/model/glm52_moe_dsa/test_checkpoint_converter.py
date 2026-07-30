# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_neuron.model.glm52_moe_dsa import checkpoint_converter
from vllm_neuron.model.glm52_moe_dsa.checkpoint_converter import (
    ARTIFACT_VERSION,
    BF16_SHARED_ARTIFACT_VERSION,
    COMPILE_CONSTANTS_FILENAME,
    COMPILE_STUB_MANIFEST_FILENAME,
    DIRECT_LEGACY_BF16_SHARED_ARTIFACT_VERSION,
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    convert_checkpoint,
    quantize_bf16_per_tensor_direct_legacy,
    quantize_bf16_per_tensor_ocp,
    required_activation_scale_keys,
    required_cache_quant_multiplier_keys,
    required_loader_source_keys,
    validate_static_fp8_artifact_metadata,
    validate_bf16_index_closure,
    write_compile_stub,
)
from vllm_neuron.model.glm52_moe_dsa.checkpoint_mapping import (
    build_checkpoint_contract,
)
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.model import _Glm52CompileStubCheckpoint
from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan
from vllm_neuron.model.glm52_moe_dsa.static_fp8 import (
    NEURON_LEGACY_E4M3FN_QMAX240,
    OCP_E4M3FN_QMAX448,
)


TARGET = "model.layers.0.self_attn.q_a_proj.weight"
ZERO_TARGET = "model.layers.0.mlp.gate_proj.weight"
UP_TARGET = "model.layers.0.mlp.up_proj.weight"
PRESERVED = "model.layers.0.input_layernorm.weight"
SHARED_TARGET = "model.layers.3.mlp.shared_experts.gate_proj.weight"
SHARED_UP_TARGET = "model.layers.3.mlp.shared_experts.up_proj.weight"
MTP_TARGET = "model.layers.78.self_attn.q_a_proj.weight"


def _write_source(root: Path) -> Path:
    root.mkdir()
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "dtype": "bfloat16",
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 78,
    }
    (root / "config.json").write_text(
        json.dumps(config, sort_keys=True),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        '{"model_max_length": 1024}\n',
        encoding="utf-8",
    )

    first = {
        TARGET: torch.tensor(
            [[-3.0, -1.0, 0.0], [0.5, 1.0, 3.0]],
            dtype=torch.bfloat16,
        ),
        PRESERVED: torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16),
    }
    second = {
        ZERO_TARGET: torch.zeros(2, 3, dtype=torch.bfloat16),
        UP_TARGET: torch.ones(2, 3, dtype=torch.bfloat16),
        SHARED_TARGET: torch.full((2, 3), 0.5, dtype=torch.bfloat16),
        SHARED_UP_TARGET: torch.full((2, 3), 0.75, dtype=torch.bfloat16),
        MTP_TARGET: torch.full((2, 3), 7.0, dtype=torch.bfloat16),
    }
    names = (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    save_file(first, root / names[0])
    save_file(second, root / names[1])
    weight_map = {
        **{key: names[0] for key in first},
        **{key: names[1] for key in second},
    }
    total_size = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (*first.values(), *second.values())
    )
    (root / INDEX_FILENAME).write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root


def _load_output_tensors(output: Path) -> dict[str, torch.Tensor]:
    index = json.loads((output / INDEX_FILENAME).read_text(encoding="utf-8"))
    by_shard: dict[str, list[str]] = {}
    for key, shard_name in index["weight_map"].items():
        by_shard.setdefault(shard_name, []).append(key)
    tensors = {}
    for shard_name, keys in by_shard.items():
        with safe_open(output / shard_name, framework="pt", device="cpu") as shard:
            tensors.update({key: shard.get_tensor(key) for key in keys})
    return tensors


def _write_full_contract_metadata(root: Path) -> Path:
    root.mkdir()
    config = asdict(Glm52MoeDsaConfig())
    config.pop("neuron_config")
    config.update(
        architectures=["GlmMoeDsaForCausalLM"],
        dtype="bfloat16",
        torch_dtype="bfloat16",
    )
    loader_keys = required_loader_source_keys(config)
    raw_keys = tuple(
        key
        for key in loader_keys
        if not key.endswith((".weight_scale", ".input_scale"))
    )
    (root / "config.json").write_text(
        json.dumps(config, sort_keys=True),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        '{"model_max_length": 2048}\n',
        encoding="utf-8",
    )
    (root / INDEX_FILENAME).write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1_506_659_919_872},
                "weight_map": {
                    key: "unmaterialized-bf16-source.safetensors" for key in raw_keys
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root


def _convert(source: Path, output: Path, **kwargs):
    source_revision = kwargs.pop(
        "source_revision",
        "b4734de4facf877f85769a911abafc5283eab3d9",
    )
    options = {
        "strict_index_closure": False,
        "max_shard_bytes": 16,
        "quantization_chunk_elements": 2,
    }
    options.update(kwargs)
    return convert_checkpoint(
        source,
        output,
        source_revision=source_revision,
        **options,
    )


def _interrupt_after_first_source_shard(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    output: Path,
    work_dir: Path,
    *,
    static_fp8_weight_format: str = NEURON_LEGACY_E4M3FN_QMAX240,
) -> None:
    original = checkpoint_converter.quantize_bf16_per_tensor
    calls = 0

    def interrupt_on_second_quantized_weight(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("test interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        checkpoint_converter,
        "quantize_bf16_per_tensor",
        interrupt_on_second_quantized_weight,
    )
    with pytest.raises(RuntimeError, match="test interruption"):
        _convert(
            source,
            output,
            work_dir=work_dir,
            static_fp8_weight_format=static_fp8_weight_format,
        )
    monkeypatch.setattr(
        checkpoint_converter,
        "quantize_bf16_per_tensor",
        original,
    )


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_compile_stub_is_metadata_only_and_never_loader_ready(
    tmp_path: Path,
) -> None:
    source = _write_full_contract_metadata(tmp_path / "metadata")
    output = tmp_path / "compile-stub"

    manifest = write_compile_stub(
        source,
        output,
        source_revision="b4734de4facf877f85769a911abafc5283eab3d9",
    )

    output_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    output_index = json.loads((output / INDEX_FILENAME).read_text(encoding="utf-8"))
    cache_keys = required_cache_quant_multiplier_keys(output_config)
    assert manifest["compile_stub"] is True
    assert manifest["serving_weights_materialized"] is False
    assert manifest["loader_ready"] is False
    assert output_config["glm52_artifact"]["compile_stub"] is True
    assert output_config["glm52_artifact"]["loader_ready"] is False
    assert (output / COMPILE_STUB_MANIFEST_FILENAME).is_file()
    assert {path.name for path in output.glob("*.safetensors")} == {
        COMPILE_CONSTANTS_FILENAME
    }
    assert set(cache_keys).issubset(output_index["weight_map"])
    with safe_open(
        output / COMPILE_CONSTANTS_FILENAME,
        framework="pt",
        device="cpu",
    ) as constants:
        assert set(constants.keys()) == set(cache_keys)
        assert all(constants.get_tensor(key).item() == 1 for key in cache_keys)
    checkpoint = _Glm52CompileStubCheckpoint(str(output))
    assert set(output_index["weight_map"]) == checkpoint.get_tensor_names()
    cache_slice = checkpoint._get_slice(cache_keys[0])
    assert tuple(cache_slice.get_shape()) == ()
    assert cache_slice[()].item() == 1


def test_hybrid_compile_stub_has_strict_bf16_shared_contract(
    tmp_path: Path,
) -> None:
    source = _write_full_contract_metadata(tmp_path / "metadata")
    output = tmp_path / "compile-stub"

    manifest = write_compile_stub(
        source,
        output,
        source_revision="b4734de4facf877f85769a911abafc5283eab3d9",
        shared_expert_dtype="bfloat16",
    )

    output_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    output_index = json.loads((output / INDEX_FILENAME).read_text(encoding="utf-8"))
    shared = "model.layers.3.mlp.shared_experts"
    assert manifest["artifact_version"] == BF16_SHARED_ARTIFACT_VERSION
    assert manifest["shared_expert_dtype"] == "bfloat16"
    assert output_config["shared_expert_dtype"] == "bfloat16"
    assert output_config["glm52_artifact"]["shared_expert_dtype"] == "bfloat16"
    assert (
        output_config["glm52_artifact"]["artifact_version"]
        == BF16_SHARED_ARTIFACT_VERSION
    )
    assert f"{shared}.gate_proj.weight" in output_index["weight_map"]
    assert f"{shared}.gate_proj.weight_scale" not in output_index["weight_map"]
    assert f"{shared}.gate_up_input_scale" not in output_index["weight_map"]


def test_zero_tensor_has_canonical_scalar_scale() -> None:
    quantized, scale = quantize_bf16_per_tensor_ocp(
        torch.zeros(3, 4, dtype=torch.bfloat16),
        chunk_elements=2,
    )

    assert quantized.dtype == torch.float8_e4m3fn
    assert not torch.count_nonzero(quantized.float())
    assert scale.dtype == torch.float32
    assert scale.shape == torch.Size([])
    assert scale.item() == 1.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_source_is_rejected(value: float) -> None:
    tensor = torch.tensor([[1.0, value]], dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="NaN or Inf"):
        quantize_bf16_per_tensor_ocp(tensor, chunk_elements=1)


def test_quantization_is_deterministic_and_uses_ocp_range() -> None:
    source = torch.tensor(
        [[-4.0, -1.25, 0.0, 1.25, 4.0]],
        dtype=torch.bfloat16,
    )

    first, first_scale = quantize_bf16_per_tensor_ocp(
        source,
        chunk_elements=2,
    )
    second, second_scale = quantize_bf16_per_tensor_ocp(
        source,
        chunk_elements=3,
    )

    assert torch.equal(first.view(torch.uint8), second.view(torch.uint8))
    assert torch.equal(first_scale, second_scale)
    assert first_scale.item() == pytest.approx(4.0 / 448.0)
    assert first.float().abs().max().item() == 448.0


def test_direct_legacy_quantization_uses_single_qmax240_rounding() -> None:
    source = torch.tensor(
        [[-4.0, -1.25, 0.0, 1.25, 4.0]],
        dtype=torch.bfloat16,
    )

    quantized, scale = quantize_bf16_per_tensor_direct_legacy(
        source,
        chunk_elements=2,
    )
    expected_scale = torch.tensor(4.0 / 240.0, dtype=torch.float32)
    expected = (
        (source.to(torch.float32) / expected_scale)
        .clamp(-240.0, 240.0)
        .to(torch.float8_e4m3fn)
    )

    assert scale.item() == pytest.approx(expected_scale.item())
    assert torch.equal(quantized.view(torch.uint8), expected.view(torch.uint8))
    assert quantized.float().abs().max().item() == 240.0


def test_conversion_excludes_mtp_and_emits_scalar_scales_and_provenance(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    output = tmp_path / "output"

    manifest = _convert(source, output)
    tensors = _load_output_tensors(output)
    index = json.loads((output / INDEX_FILENAME).read_text(encoding="utf-8"))
    output_config = json.loads((output / "config.json").read_text(encoding="utf-8"))

    assert MTP_TARGET not in index["weight_map"]
    assert manifest["exclusions"]["tensor_count"] == 1
    assert TARGET in tensors
    assert tensors[TARGET].dtype == torch.float8_e4m3fn
    assert tensors[f"{TARGET}_scale"].shape == torch.Size([])
    assert tensors[f"{ZERO_TARGET}_scale"].item() == 1.0
    assert tensors[PRESERVED].dtype == torch.bfloat16
    assert manifest["artifact_version"] == ARTIFACT_VERSION
    assert manifest["source"]["revision"].startswith("b4734de4")
    assert manifest["quantization"]["loader_compensation"] == {
        "weight_multiplier": 240.0 / 448.0,
        "scale_multiplier": 448.0 / 240.0,
        "neuron_kernel_qmax": 240.0,
    }
    assert manifest["calibration"]["status"] == "missing"
    assert not manifest["loader_validation"]["loader_ready"]
    assert manifest["source"]["declared_total_tensor_bytes"] == 78
    assert (output / MANIFEST_FILENAME).is_file()

    quantization = output_config["quantization_config"]
    assert quantization["quant_method"] == "modelopt"
    assert quantization["quantization"]["quant_algo"] == "FP8"
    assert quantization["quantization"]["kv_cache_quant_algo"] == "FP8"
    assert "lm_head" in quantization["quantization"]["exclude_modules"]
    assert not output_config["glm52_artifact"]["loader_ready"]
    assert output_config["dtype"] == "bfloat16"


def test_direct_legacy_hybrid_conversion_marks_contract_and_preserves_shared_bf16(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    output = tmp_path / "output"

    manifest = _convert(
        source,
        output,
        shared_expert_dtype="bfloat16",
        static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
    )
    tensors = _load_output_tensors(output)
    output_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    source_target = torch.tensor(
        [[-3.0, -1.0, 0.0], [0.5, 1.0, 3.0]],
        dtype=torch.bfloat16,
    )
    expected_scale = torch.tensor(3.0 / 240.0, dtype=torch.float32)
    expected_target = (
        (source_target.to(torch.float32) / expected_scale)
        .clamp(-240.0, 240.0)
        .to(torch.float8_e4m3fn)
    )

    assert manifest["artifact_version"] == (DIRECT_LEGACY_BF16_SHARED_ARTIFACT_VERSION)
    assert manifest["static_fp8_weight_format"] == NEURON_LEGACY_E4M3FN_QMAX240
    assert manifest["quantization"]["qmax"] == 240.0
    assert manifest["quantization"]["loader_compensation"] == {
        "weight_multiplier": 1.0,
        "scale_multiplier": 1.0,
        "neuron_kernel_qmax": 240.0,
    }
    assert torch.equal(
        tensors[TARGET].view(torch.uint8),
        expected_target.view(torch.uint8),
    )
    assert tensors[f"{TARGET}_scale"].item() == pytest.approx(expected_scale.item())
    assert tensors[SHARED_TARGET].dtype == torch.bfloat16
    assert tensors[SHARED_UP_TARGET].dtype == torch.bfloat16
    assert f"{SHARED_TARGET}_scale" not in tensors
    assert f"{SHARED_UP_TARGET}_scale" not in tensors
    assert output_config["static_fp8_weight_format"] == NEURON_LEGACY_E4M3FN_QMAX240
    assert (
        output_config["glm52_artifact"]["static_fp8_weight_format"]
        == NEURON_LEGACY_E4M3FN_QMAX240
    )
    assert (
        output_config["quantization_config"]["quantization"]["weight_format"]
        == NEURON_LEGACY_E4M3FN_QMAX240
    )
    assert (
        validate_static_fp8_artifact_metadata(manifest, output_config)
        == NEURON_LEGACY_E4M3FN_QMAX240
    )


def test_artifact_metadata_rejects_mixed_or_undeclared_direct_format(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    output = tmp_path / "output"
    manifest = _convert(
        source,
        output,
        shared_expert_dtype="bfloat16",
        static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
    )
    output_config = json.loads((output / "config.json").read_text(encoding="utf-8"))

    mixed = json.loads(json.dumps(output_config))
    mixed["glm52_artifact"]["static_fp8_weight_format"] = OCP_E4M3FN_QMAX448
    with pytest.raises(ValueError, match="mixed weight formats"):
        validate_static_fp8_artifact_metadata(manifest, mixed)

    undeclared = json.loads(json.dumps(output_config))
    undeclared["quantization_config"]["quantization"].pop("weight_format")
    with pytest.raises(ValueError, match="undeclared weight format"):
        validate_static_fp8_artifact_metadata(manifest, undeclared)


def test_hybrid_conversion_preserves_shared_expert_in_bf16(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    output = tmp_path / "output"

    manifest = _convert(
        source,
        output,
        shared_expert_dtype="bfloat16",
    )
    tensors = _load_output_tensors(output)
    output_config = json.loads((output / "config.json").read_text(encoding="utf-8"))

    assert manifest["artifact_version"] == BF16_SHARED_ARTIFACT_VERSION
    assert manifest["shared_expert_dtype"] == "bfloat16"
    assert tensors[SHARED_TARGET].dtype == torch.bfloat16
    assert tensors[SHARED_UP_TARGET].dtype == torch.bfloat16
    assert f"{SHARED_TARGET}_scale" not in tensors
    assert f"{SHARED_UP_TARGET}_scale" not in tensors
    assert output_config["shared_expert_dtype"] == "bfloat16"
    assert output_config["glm52_artifact"] == {
        **output_config["glm52_artifact"],
        "artifact_version": BF16_SHARED_ARTIFACT_VERSION,
        "shared_expert_dtype": "bfloat16",
    }
    excluded = output_config["quantization_config"]["quantization"]["exclude_modules"]
    assert "model.layers.*.mlp.shared_experts.*" in excluded
    required = required_activation_scale_keys(
        (TARGET, SHARED_TARGET),
        shared_expert_dtype="bfloat16",
    )
    assert "model.layers.3.mlp.shared_experts.gate_proj.input_scale" not in required


def test_conversion_is_byte_deterministic_and_has_bounded_output_buffer(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = _convert(source, first)
    second_manifest = _convert(source, second)

    assert (first / MANIFEST_FILENAME).read_bytes() == (
        second / MANIFEST_FILENAME
    ).read_bytes()
    assert (first / INDEX_FILENAME).read_bytes() == (
        second / INDEX_FILENAME
    ).read_bytes()
    assert first_manifest["artifact_id"] == second_manifest["artifact_id"]
    assert [shard["sha256"] for shard in first_manifest["output"]["shards"]] == [
        shard["sha256"] for shard in second_manifest["output"]["shards"]
    ]

    bounds = first_manifest["streaming_bounds"]
    assert bounds["source_tensors_loaded_at_once"] == 1
    assert bounds["source_shards_materialized_at_once"] == 1
    assert bounds["maximum_buffered_output_bytes"] <= max(
        bounds["configured_max_shard_bytes"],
        bounds["largest_output_group_bytes"],
    )
    assert len(first_manifest["output"]["shards"]) > 1


def test_direct_conversion_resumes_to_byte_identical_clean_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path / "source")
    clean = tmp_path / "clean"
    resumed = tmp_path / "resumed"
    work_dir = tmp_path / "resume-work"
    _convert(
        source,
        clean,
        static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
    )
    _interrupt_after_first_source_shard(
        monkeypatch,
        source,
        resumed,
        work_dir,
    )

    _convert(
        source,
        resumed,
        work_dir=work_dir,
        resume=True,
        static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
    )

    assert _artifact_bytes(resumed) == _artifact_bytes(clean)
    assert not (work_dir / "artifact").exists()


def test_resume_rolls_back_contiguous_unreceipted_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path / "source")
    clean = tmp_path / "clean"
    output = tmp_path / "output"
    work_dir = tmp_path / "resume-work"
    _convert(
        source,
        clean,
        static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
    )
    original_add = checkpoint_converter._StreamingShardWriter.add
    calls = 0

    def interrupt_after_unreceipted_flush(self, tensors):
        nonlocal calls
        calls += 1
        original_add(self, tensors)
        if calls == 4:
            raise RuntimeError("mid-shard interruption")

    monkeypatch.setattr(
        checkpoint_converter._StreamingShardWriter,
        "add",
        interrupt_after_unreceipted_flush,
    )
    with pytest.raises(RuntimeError, match="mid-shard interruption"):
        _convert(
            source,
            output,
            work_dir=work_dir,
            static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
        )
    monkeypatch.setattr(
        checkpoint_converter._StreamingShardWriter,
        "add",
        original_add,
    )

    _convert(
        source,
        output,
        work_dir=work_dir,
        resume=True,
        static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
    )
    assert _artifact_bytes(output) == _artifact_bytes(clean)


def test_resume_rebuilds_interrupted_hardlink_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path / "source")
    clean = tmp_path / "clean"
    output = tmp_path / "output"
    work_dir = tmp_path / "resume-work"
    _convert(
        source,
        clean,
        static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
    )
    original_link = checkpoint_converter.os.link
    links = 0

    def interrupt_second_link(source_path, destination_path):
        nonlocal links
        links += 1
        if links == 2:
            raise RuntimeError("publication interruption")
        return original_link(source_path, destination_path)

    monkeypatch.setattr(checkpoint_converter.os, "link", interrupt_second_link)
    with pytest.raises(RuntimeError, match="publication interruption"):
        _convert(
            source,
            output,
            work_dir=work_dir,
            static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
        )
    monkeypatch.setattr(checkpoint_converter.os, "link", original_link)

    _convert(
        source,
        output,
        work_dir=work_dir,
        resume=True,
        static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
    )
    assert _artifact_bytes(output) == _artifact_bytes(clean)


def test_resume_rejects_corrupted_receipt_and_output_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path / "source")
    output = tmp_path / "output"
    work_dir = tmp_path / "resume-work"
    _interrupt_after_first_source_shard(
        monkeypatch,
        source,
        output,
        work_dir,
    )
    part = next((work_dir / "artifact").glob(".part-*.safetensors"))
    contents = bytearray(part.read_bytes())
    contents[-1] ^= 1
    part.write_bytes(contents)
    with pytest.raises(ValueError, match="checksum changed"):
        _convert(
            source,
            output,
            work_dir=work_dir,
            resume=True,
            static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
        )

    part.write_bytes(bytes(contents[:-1]) + bytes([contents[-1] ^ 1]))
    receipt = work_dir / "receipts" / "source-00001.json"
    receipt_data = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_data["state"]["stats"]["output_tensor_count"] += 1
    receipt.write_text(json.dumps(receipt_data), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt checksum"):
        _convert(
            source,
            output,
            work_dir=work_dir,
            resume=True,
            static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
        )


def test_resume_identity_rejects_fp8_format_and_revision_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_source(tmp_path / "source")
    output = tmp_path / "output"
    work_dir = tmp_path / "resume-work"
    _interrupt_after_first_source_shard(
        monkeypatch,
        source,
        output,
        work_dir,
    )
    with pytest.raises(ValueError, match="resume identity mismatch"):
        _convert(
            source,
            output,
            work_dir=work_dir,
            resume=True,
            static_fp8_weight_format=OCP_E4M3FN_QMAX448,
        )
    with pytest.raises(ValueError, match="resume identity mismatch"):
        _convert(
            source,
            output,
            work_dir=work_dir,
            resume=True,
            source_revision="0123456789abcdef0123456789abcdef01234567",
            static_fp8_weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
        )


def _small_config() -> Glm52MoeDsaConfig:
    return Glm52MoeDsaConfig(
        vocab_size=16,
        hidden_size=4,
        num_hidden_layers=4,
        intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=4,
        kv_lora_rank=2,
        qk_head_dim=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        index_n_heads=2,
        index_head_dim=2,
        index_topk=2,
        index_skip_topk_offset=1,
        index_topk_freq=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        first_k_dense_replace=1,
    )


def test_index_preflight_proves_exact_loader_closure() -> None:
    config = _small_config()
    plan = RoutedExpertPlan(
        world_size=4,
        ep_degree=2,
        num_experts=4,
        expert_intermediate_size=8,
    )
    source_keys = set()
    for rank in range(4):
        contract = build_checkpoint_contract(config, plan, global_rank=rank)
        source_keys.update(
            key
            for key in contract.required_source_keys
            if not key.endswith((".weight_scale", ".input_scale"))
        )
    config_dict = asdict(config)
    config_dict["architectures"] = ["GlmMoeDsaForCausalLM"]

    result = validate_bf16_index_closure(
        config_dict,
        tuple(source_keys),
        world_size=4,
        ep_degree=2,
    )

    assert result["status"] == "passed"
    assert result["expected_non_mtp_key_count"] == len(source_keys)
    with pytest.raises(ValueError, match="does not close"):
        validate_bf16_index_closure(
            config_dict,
            tuple(source_keys - {next(iter(source_keys))}),
            world_size=4,
            ep_degree=2,
        )


def test_cache_calibration_contract_covers_main_and_full_indexer_caches() -> None:
    config = _small_config()
    config_dict = asdict(config)
    config_dict["architectures"] = ["GlmMoeDsaForCausalLM"]

    keys = required_cache_quant_multiplier_keys(config_dict)

    assert len(keys) == 2 * config.num_hidden_layers + len(
        config.full_indexer_layer_ids
    )
    assert "model.layers.0.self_attn.k_cache_quant_multiplier" in keys
    assert "model.layers.0.self_attn.v_cache_quant_multiplier" in keys
    assert "model.layers.0.self_attn.indexer.cache_quant_multiplier" in keys


def test_complete_calibration_emits_scalar_projection_and_cache_contract(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "source")
    source_index = json.loads((source / INDEX_FILENAME).read_text(encoding="utf-8"))
    projection_keys = required_activation_scale_keys(tuple(source_index["weight_map"]))
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    cache_keys = required_cache_quant_multiplier_keys(source_config)
    projection = {key: 0.25 for key in projection_keys}
    projection["model.layers.0.self_attn.q_a_proj.input_scale"] = 0.5
    cache = {key: 8.0 for key in cache_keys}
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "projection_input_scales": projection,
                "cache_quant_multipliers": cache,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    output = tmp_path / "output"
    manifest = _convert(
        source,
        output,
        activation_scales_path=calibration,
    )
    tensors = _load_output_tensors(output)
    output_config = json.loads((output / "config.json").read_text(encoding="utf-8"))

    assert manifest["calibration"]["status"] == "complete"
    assert manifest["output"]["generated_cache_scale_count"] == len(cache_keys)
    assert manifest["output"]["generated_input_scale_count"] == len(projection_keys)
    for key in (*projection_keys, *cache_keys):
        assert tensors[key].shape == torch.Size([])
    assert tensors["model.layers.0.self_attn.k_cache_quant_multiplier"].item() == 8.0
    assert output_config["glm52_artifact"]["cache_quant_multipliers"]["values"] == cache
    # Synthetic tests intentionally skip the exact 58,794-key closure. A
    # calibrated artifact is still rejected until that independent gate passes.
    assert not manifest["loader_validation"]["loader_ready"]


def test_hf_streaming_removes_ephemeral_snapshot_and_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = _write_source(tmp_path / "materialized")
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    shutil.copyfile(materialized / "config.json", metadata / "config.json")
    shutil.copyfile(
        materialized / INDEX_FILENAME,
        metadata / INDEX_FILENAME,
    )
    source_shards = {path.name: path for path in materialized.glob("*.safetensors")}
    cache_roots: list[Path] = []

    def fake_hf_hub_download(*, filename: str, cache_dir: str, **_kwargs):
        root = Path(cache_dir)
        cache_roots.append(root)
        blob = root / "models--test" / "blobs" / filename
        blob.parent.mkdir(parents=True)
        shutil.copyfile(source_shards[filename], blob)
        return str(blob)

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        fake_hf_hub_download,
    )
    output = tmp_path / "output"
    manifest = _convert(metadata, output, hf_streaming=True)

    assert manifest["streaming_bounds"]["hf_streaming"]
    assert all(shard["transient_download"] for shard in manifest["source"]["shards"])
    assert cache_roots
    assert all(not root.exists() for root in cache_roots)
