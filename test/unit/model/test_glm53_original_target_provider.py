from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).parents[3]
PROVIDER_PATH = ROOT / "vllm_neuron/model/glm53_flash/original_target_provider.py"
SPEC = importlib.util.spec_from_file_location(
    "glm53_original_target_provider", PROVIDER_PATH
)
assert SPEC is not None and SPEC.loader is not None
PROVIDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROVIDER
SPEC.loader.exec_module(PROVIDER)


pytest.importorskip("transformers")


def _tiny_model():
    from transformers.models.glm5_next.configuration_glm5_next import (
        Glm5NextConfig,
    )
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextForConditionalGeneration,
    )

    text = {
        "vocab_size": PROVIDER.GLM53_VOCAB_SIZE,
        "hidden_size": 16,
        "intermediate_size": 32,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "qk_rope_head_dim": 0,
        "qk_nope_head_dim": 8,
        "v_head_dim": 8,
        "linear_head_dim": 4,
        "linear_num_heads": 4,
        "mlp_layer_types": ["dense"],
        "layer_types": ["linear_attention"],
        "indexer_types": ["full"],
        "eos_token_id": [154820],
        "pad_token_id": 154820,
    }
    vision = {
        "depth": 1,
        "hidden_size": 8,
        "num_heads": 2,
        "image_size": 14,
        "patch_size": 14,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "out_hidden_size": 16,
        "intermediate_size": 16,
        "projection_intermediate_size": 16,
    }
    config = Glm5NextConfig(
        text_config=text,
        vision_config=vision,
        architectures=["Glm5NextForConditionalGeneration"],
    )
    return Glm5NextForConditionalGeneration(config).eval()


class _TinyProcessor:
    def apply_chat_template(self, _messages, **_kwargs):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}


def _bind_tiny_provider(monkeypatch):
    monkeypatch.setattr(
        PROVIDER,
        "_checkpoint_dir",
        Path(PROVIDER.GLM53_CHECKPOINT_REVISION),
    )
    monkeypatch.setattr(PROVIDER, "_processor", _TinyProcessor())
    monkeypatch.setattr(PROVIDER, "_semantics", PROVIDER.GLM53_NATIVE_BLOCK_FP8)


def test_real_upstream_glm5_next_tiny_model_returns_full_vocabulary(monkeypatch):
    _bind_tiny_provider(monkeypatch)
    model = _tiny_model()
    rows = list(PROVIDER.run(model, "feedback-0", range(10), (1, 2)))
    assert len(rows) == 10
    assert all(row.shape == (PROVIDER.GLM53_VOCAB_SIZE,) for row in rows)
    assert all(row.dtype is torch.float32 for row in rows)
    assert all(bool(torch.isfinite(row).all()) for row in rows)


def test_runner_rejects_unbound_token_ids(monkeypatch):
    _bind_tiny_provider(monkeypatch)
    with pytest.raises(PROVIDER.Glm53OriginalProviderError, match="do not match"):
        next(PROVIDER.run(_tiny_model(), "feedback-0", range(10), (2, 1)))


def test_configure_rejects_implicit_bfloat16_conversion(tmp_path):
    checkpoint = tmp_path / PROVIDER.GLM53_CHECKPOINT_REVISION
    checkpoint.mkdir()
    with pytest.raises(
        PROVIDER.Glm53OriginalProviderError, match="explicitly declared"
    ):
        PROVIDER.configure(checkpoint, "bfloat16")


def test_cpu_fp8_scaled_mm_is_available_but_native_provider_is_fail_closed(
    monkeypatch, tmp_path
):
    from transformers.quantizers import quantizer_finegrained_fp8
    from transformers.utils.quantization_config import FineGrainedFP8Config

    assert hasattr(torch, "_scaled_mm")
    activations = torch.randn(2, 4).to(torch.float8_e4m3fn)
    weights = torch.randn(3, 4).to(torch.float8_e4m3fn)
    output = torch._scaled_mm(
        activations,
        weights.t(),
        scale_a=torch.tensor(1.0),
        scale_b=torch.tensor(1.0),
        out_dtype=torch.float32,
    )
    assert output.shape == (2, 3)
    assert output.dtype is torch.float32

    # The actual Transformers quantizer, not a provider mock, rejects native
    # pre-quantized CPU execution.  Force only the optional accelerate check so
    # the test reaches the hardware gate in a minimal environment.
    quantizer = quantizer_finegrained_fp8.FineGrainedFP8HfQuantizer(
        FineGrainedFP8Config(weight_block_size=(128, 128)),
        pre_quantized=True,
    )
    monkeypatch.setattr(
        quantizer_finegrained_fp8, "is_accelerate_available", lambda: True
    )
    if not torch.cuda.is_available() and not PROVIDER._torch_xpu_available():
        quantizer.validate_environment()
        assert quantizer.quantization_config.dequantize is True

    checkpoint = tmp_path / PROVIDER.GLM53_CHECKPOINT_REVISION
    checkpoint.mkdir()
    monkeypatch.setattr(PROVIDER, "_checkpoint_dir", checkpoint)
    monkeypatch.setattr(PROVIDER, "_processor", _TinyProcessor())
    monkeypatch.setattr(PROVIDER, "_semantics", PROVIDER.GLM53_NATIVE_BLOCK_FP8)
    if not torch.cuda.is_available() and not PROVIDER._torch_xpu_available():
        with pytest.raises(PROVIDER.Glm53OriginalProviderError, match="requires CUDA"):
            PROVIDER.load(checkpoint)


def test_serialized_fp8_block_edges_dequantize_then_forward_on_cpu(tmp_path):
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    from transformers.quantizers.quantizer_finegrained_fp8 import (
        FineGrainedFP8HfQuantizer,
    )
    from transformers.utils.quantization_config import FineGrainedFP8Config

    quantizer = FineGrainedFP8HfQuantizer(
        FineGrainedFP8Config(weight_block_size=(128, 128), dequantize=True),
        pre_quantized=True,
    )
    weight = torch.ones((256, 256), dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1.0, 2.0], [4.0, 8.0]], dtype=torch.float32)
    serialized = tmp_path / "one-linear.safetensors"
    save_file({"weight": weight, "weight_scale_inv": scales}, str(serialized))
    loaded = load_file(str(serialized), device="cpu")

    target = torch.nn.Linear(256, 256, bias=False, dtype=torch.bfloat16)
    converted = Fp8Dequantize(quantizer).convert(
        {
            "weight$": [loaded["weight"]],
            "weight_scale_inv": [loaded["weight_scale_inv"]],
        },
        full_layer_name="weight",
        model=target,
    )["weight"]
    assert converted.dtype is torch.bfloat16
    assert torch.all(converted[:128, :128] == 1)
    assert torch.all(converted[:128, 128:] == 2)
    assert torch.all(converted[128:, :128] == 4)
    assert torch.all(converted[128:, 128:] == 8)

    target.weight.data.copy_(converted)
    output = target(torch.ones((1, 256), dtype=torch.bfloat16))
    assert output.shape == (1, 256)
    assert torch.all(output[0, :128] == 384)
    assert torch.all(output[0, 128:] == 1536)
