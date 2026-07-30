"""Contract tests for the native Gemma 4 vLLM-Neuron entry point."""

from types import SimpleNamespace

from vllm_neuron.model.gemma4.config import Gemma4Config
from vllm_neuron.model.gemma4.model import (
    Gemma4RMSNorm,
    Gemma4RotaryEmbedding,
    Gemma4ValueNorm,
)
from vllm_neuron.model.registry import get_models


def test_gemma4_is_registered():
    names = {name for name, _ in get_models()}
    assert "Gemma4MoeForCausalLM" in names


def test_nested_text_config_is_parsed():
    config = SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            vocab_size=1000,
        )
    )
    # The production HF config supplies to_dict(); this test uses the same
    # nested shape without importing Transformers model classes.
    config.to_dict = lambda: {"text_config": config.text_config.__dict__}
    parsed = Gemma4Config.from_configs(config)
    assert parsed.hidden_size == 512
    assert parsed.num_key_value_heads == 2
    assert parsed.head_dim == 64
    assert parsed.vocab_size == 1000


def test_rms_norm_matches_reference_formula():
    layer = Gemma4RMSNorm(8, dtype=None)
    layer.weight.data.zero_()
    x = __import__("torch").tensor([[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]])
    expected = x / (x.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    assert __import__("torch").allclose(layer(x), expected, atol=1e-5, rtol=1e-5)


def test_rotary_embeddings_have_local_and_global_shapes():
    torch = __import__("torch")
    positions = torch.arange(4)
    local = Gemma4RotaryEmbedding(256, 10_000.0)
    global_ = Gemma4RotaryEmbedding(512, 1_000_000.0)
    assert local(positions, torch.float32)[0].shape == (4, 256)
    assert global_(positions, torch.float32)[0].shape == (4, 512)


def test_value_norm_preserves_shape():
    torch = __import__("torch")
    values = torch.randn(2, 3, 64)
    assert Gemma4ValueNorm(64, dtype=torch.float32)(values).shape == values.shape
