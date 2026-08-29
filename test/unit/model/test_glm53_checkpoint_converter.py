from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).parents[3]
    / "vllm_neuron"
    / "model"
    / "glm53_flash"
    / "checkpoint_converter.py"
)
SPEC = importlib.util.spec_from_file_location("glm53_checkpoint_converter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from glm53_checkpoint_converter import (
    GLM53_DSA_LAYERS,
    GLM53_EXPECTED_INDEXER_TENSORS,
    GLM53_EXPECTED_MTP_TENSORS,
    GLM53_EXPECTED_SCALES,
    GLM53_EXPECTED_TENSORS,
    GLM53_EXPECTED_VISION_TENSORS,
    GLM53_KDA_LAYERS,
    Glm53ArchitectureMismatch,
    classify_tensor,
    dequantize_block_fp8,
    kda_conv1d_per_head_layout,
    preflight_checkpoint_metadata,
)


def _config() -> dict:
    return {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": 45,
            "layer_types": [
                "deepseek_sparse_attention"
                if i in GLM53_DSA_LAYERS
                else "linear_attention"
                for i in range(45)
            ],
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
        },
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }


def _weight_map() -> dict[str, str]:
    result: dict[str, str] = {}
    # The immutable index categories overlap: layer 45 contains 871 FP8
    # weight/scale pairs and seven of its 18 BF16 tensors are indexer tensors.
    mtp_scale_pairs = 871
    mtp_indexer_tensors = 7
    for i in range(GLM53_EXPECTED_SCALES - mtp_scale_pairs):
        weight = (
            f"model.language_model.layers.{i % 45}.mlp.experts.{i}.gate_proj.weight"
        )
        result[weight] = "model-00001-of-00063.safetensors"
        result[f"{weight}_scale_inv"] = "model-00001-of-00063.safetensors"
    for i in range(mtp_scale_pairs):
        weight = f"model.language_model.layers.45.mlp.experts.{i}.gate_proj.weight"
        result[weight] = "model-00063-of-00063.safetensors"
        result[f"{weight}_scale_inv"] = "model-00063-of-00063.safetensors"
    for i in range(GLM53_EXPECTED_MTP_TENSORS - 2 * mtp_scale_pairs):
        suffix = (
            f"self_attn.indexer.synthetic.{i}"
            if i < mtp_indexer_tensors
            else f"bf16.synthetic.{i}"
        )
        result[f"model.language_model.layers.45.{suffix}"] = (
            "model-00063-of-00063.safetensors"
        )
    for i in range(GLM53_EXPECTED_VISION_TENSORS):
        result[f"model.visual.synthetic.{i}"] = "model-00062-of-00063.safetensors"
    for i in range(GLM53_EXPECTED_INDEXER_TENSORS - mtp_indexer_tensors):
        result[
            f"model.language_model.layers.{GLM53_DSA_LAYERS[i % 11]}.self_attn.indexer.synthetic.{i}"
        ] = "model-00001-of-00063.safetensors"
    remaining = GLM53_EXPECTED_TENSORS - len(result)
    for i in range(remaining):
        result[f"model.language_model.layers.{i % 45}.bf16.synthetic.{i}"] = (
            "model-00001-of-00063.safetensors"
        )
    assert len(result) == GLM53_EXPECTED_TENSORS
    return result


def test_exact_glm5_next_metadata_preflight_passes() -> None:
    report = preflight_checkpoint_metadata(_config(), _weight_map())
    assert report.tensor_count == 76_108
    assert report.block_scale_count == 37_338
    assert report.dsa_layers == GLM53_DSA_LAYERS
    assert report.kda_layers == GLM53_KDA_LAYERS
    assert report.weight_block_size == (128, 128)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("architectures",), ["Glm52MoeDsaForCausalLM"], "Glm52MoeDsa"),
        (("model_type",), "glm52_moe_dsa", "glm5_next"),
        (("quantization_config", "weight_block_size"), None, "128x128"),
        (
            ("text_config", "layer_types"),
            ["deepseek_sparse_attention"] * 45,
            "DSA schedule",
        ),
    ],
)
def test_preflight_rejects_glm52_and_shape_inference(path, value, match) -> None:
    config = copy.deepcopy(_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(Glm53ArchitectureMismatch, match=match):
        preflight_checkpoint_metadata(config, _weight_map())


def test_preflight_rejects_partial_index_and_orphan_scale() -> None:
    weight_map = _weight_map()
    weight_map.pop(next(iter(weight_map)))
    with pytest.raises(Glm53ArchitectureMismatch, match="indexed tensors"):
        preflight_checkpoint_metadata(_config(), weight_map)

    orphan = {"x.weight_scale_inv": "one.safetensors"}
    with pytest.raises(Glm53ArchitectureMismatch, match="orphan"):
        classify_tensor("x.weight_scale_inv", orphan)


def test_tensor_policy_does_not_treat_hyperconnection_scale_as_fp8() -> None:
    weight_map = {
        "model.language_model.layers.0.hc_attn_scale": "one.safetensors",
        "model.language_model.layers.3.self_attn.q_a_proj.weight": "one.safetensors",
        "model.language_model.layers.3.self_attn.q_a_proj.weight_scale_inv": "one.safetensors",
    }
    assert (
        classify_tensor("model.language_model.layers.0.hc_attn_scale", weight_map)
        == "bf16_holdout"
    )
    assert (
        classify_tensor(
            "model.language_model.layers.3.self_attn.q_a_proj.weight", weight_map
        )
        == "block_fp8_weight"
    )


def test_reciprocal_block_fp8_dequant_multiplies_and_handles_ragged_edges() -> None:
    weight = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    scale_inv = torch.tensor([[0.5, 2.0], [3.0, 4.0]], dtype=torch.float32)
    actual = dequantize_block_fp8(weight, scale_inv, (3, 4), out_dtype=torch.float32)
    expected_scale = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.5, 2.0, 2.0],
            [0.5, 0.5, 0.5, 0.5, 2.0, 2.0],
            [0.5, 0.5, 0.5, 0.5, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 3.0, 3.0, 4.0, 4.0],
        ]
    )
    torch.testing.assert_close(actual, weight * expected_scale)
    with pytest.raises(ValueError, match="expected trailing shape"):
        dequantize_block_fp8(weight, torch.ones(1, 1), (3, 4))


def test_kda_conv_layout_is_per_head_not_stream_major() -> None:
    q = torch.tensor([[[10.0]], [[11.0]], [[20.0]], [[21.0]]])
    k = torch.tensor([[[30.0]], [[31.0]], [[40.0]], [[41.0]]])
    v = torch.tensor([[[50.0]], [[51.0]], [[60.0]], [[61.0]]])
    fused = kda_conv1d_per_head_layout(q, k, v, num_heads=2, head_dim=2)
    assert fused.flatten().tolist() == [10, 11, 30, 31, 50, 51, 20, 21, 40, 41, 60, 61]
    assert fused.flatten().tolist() != torch.cat((q, k, v), dim=0).flatten().tolist()
