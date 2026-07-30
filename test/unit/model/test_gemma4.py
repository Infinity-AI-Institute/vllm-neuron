"""Contract tests for the native Gemma 4 vLLM-Neuron entry point."""

from types import SimpleNamespace

from vllm_neuron.model.gemma4.config import Gemma4Config
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
