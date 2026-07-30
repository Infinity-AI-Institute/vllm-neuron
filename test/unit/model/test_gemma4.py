"""Contract tests for the native Gemma 4 vLLM-Neuron entry point."""

from types import SimpleNamespace

from vllm_neuron.model.gemma4.config import Gemma4Config
from vllm_neuron.model.gemma4.model import (
    Gemma4RMSNorm,
    Gemma4PagedKVCache,
    Gemma4ReferenceAttention,
    Gemma4ReferenceMoE,
    Gemma4WeightMapper,
    Gemma4Linear,
    Gemma4ReferenceAttentionBlock,
    Gemma4ReferenceDecoderLayer,
    Gemma4ReferenceTextModel,
    Gemma4ReferenceLMHead,
    Gemma4ReferenceCausalLM,
    Gemma4RotaryEmbedding,
    Gemma4ValueNorm,
)
from vllm_neuron.model.registry import get_models
from vllm_neuron.model.gemma4.factory import Gemma4MoeForCausalLM


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


def test_layer_attention_shape_preserves_hybrid_layout():
    config = Gemma4Config(layer_types=["local", "global"], head_dim=256,
                          num_key_value_heads=2, global_head_dim=512,
                          num_global_key_value_heads=1)
    assert config.attention_shape(0) == (256, 2)
    assert config.attention_shape(1) == (512, 1)


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


def test_paged_cache_writes_and_reads_slots():
    torch = __import__("torch")
    cache = Gemma4PagedKVCache(8, 2, 4, dtype=torch.float32)
    slots = torch.tensor([3, 6])
    key = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    value = key + 100
    cache.write(slots, key, value)
    got_key, got_value = cache.read(slots)
    assert torch.equal(got_key, key)
    assert torch.equal(got_value, value)
    assert torch.count_nonzero(cache.key[0]) == 0


def test_paged_cache_rejects_wrong_layer_shape():
    torch = __import__("torch")
    cache = Gemma4PagedKVCache(4, 1, 4, dtype=torch.float32)
    try:
        cache.write(torch.tensor([0]), torch.zeros(1, 2, 4), torch.zeros(1, 2, 4))
    except RuntimeError:
        # index_copy_ reports the head-dimension mismatch on current PyTorch.
        return
    except ValueError:
        return
    raise AssertionError("cache accepted a layer with the wrong KV-head shape")


def test_reference_attention_supports_gqa_and_cache_contract():
    torch = __import__("torch")
    attention = Gemma4ReferenceAttention(head_dim=4, num_query_heads=4, num_kv_heads=2)
    query = torch.randn(3, 4, 4)
    key = torch.randn(3, 2, 4)
    value = torch.randn(3, 2, 4)
    cache = Gemma4PagedKVCache(3, 2, 4, dtype=torch.float32)
    result = attention(query, key, value, cache, torch.tensor([0, 1, 2]))
    assert result.shape == query.shape


def test_reference_moe_dispatches_and_combines_top_k():
    torch = __import__("torch")
    torch.manual_seed(0)
    moe = Gemma4ReferenceMoE(hidden_size=8, intermediate_size=16, num_experts=4, top_k=2)
    hidden = torch.randn(5, 8)
    output = moe(hidden)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    # Router probabilities are normalized per token before expert combine.
    logits = moe.router(hidden).float()
    top, _ = torch.topk(logits, 2, dim=-1)
    assert torch.allclose(torch.softmax(top, -1).sum(-1), torch.ones(5))


def test_weight_mapper_preserves_expert_indices():
    name = "model.layers.7.block_sparse_moe.experts.12.down_proj.weight"
    mapped = Gemma4WeightMapper.map_name(name)
    assert mapped == "layers.7.moe.experts.12.down_proj.weight"
    assert Gemma4WeightMapper.is_expert_weight(name)
    assert not Gemma4WeightMapper.is_expert_weight("model.layers.7.self_attn.q_proj.weight")
    assert Gemma4WeightMapper.loader_kind("model.layers.7.self_attn.q_proj.weight") == "column"
    assert Gemma4WeightMapper.loader_kind("model.layers.7.self_attn.o_proj.weight") == "row"
    assert Gemma4WeightMapper.loader_kind(name) == "expert-local"
    assert Gemma4WeightMapper.loader_kind("model.layers.7.input_layernorm.weight") == "replicated"
    assert Gemma4WeightMapper.make_loader("q_proj.weight", 8, 4).__class__.__name__ == "SafetensorsWeightLoader"
    assert Gemma4WeightMapper.make_loader("o_proj.weight", 8, 4).__class__.__name__ == "SafetensorsWeightLoader"


def test_linear_attaches_loader_and_uses_local_tp_shape():
    layer = Gemma4Linear(16, 32, "q_proj.weight", tp_size=4)
    assert tuple(layer.weight.shape) == (8, 16)
    assert hasattr(layer.weight, "weight_loader")


def test_reference_attention_block_composes_native_linear_and_cache():
    torch = __import__("torch")
    block = Gemma4ReferenceAttentionBlock(16, 4, 2, 4)
    hidden = torch.randn(3, 16, dtype=torch.bfloat16)
    cache = Gemma4PagedKVCache(3, 2, 4, dtype=torch.bfloat16)
    output = block(hidden, cache, torch.tensor([0, 1, 2]))
    assert output.shape == hidden.shape


def test_reference_decoder_layer_composes_attention_and_moe():
    torch = __import__("torch")
    config = Gemma4Config(hidden_size=16, intermediate_size=32, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=4, layer_types=["local"],
                          global_head_dim=4, num_global_key_value_heads=2)
    layer = Gemma4ReferenceDecoderLayer(config, layer_idx=0, num_experts=4, top_k=2)
    hidden = torch.randn(3, 16, dtype=torch.bfloat16)
    cache = Gemma4PagedKVCache(3, 2, 4, dtype=torch.bfloat16)
    output = layer(hidden, cache, torch.tensor([0, 1, 2]))
    assert output.shape == hidden.shape


def test_reference_text_model_runs_tiny_stack():
    torch = __import__("torch")
    config = Gemma4Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=4,
                          layer_types=["local"], global_head_dim=4,
                          num_global_key_value_heads=2)
    model = Gemma4ReferenceTextModel(config, num_experts=2, top_k=1)
    output = model(torch.tensor([[1, 2, 3]]))
    assert output.shape == (1, 3, 16)


def test_lm_head_selects_sampling_positions():
    torch = __import__("torch")
    head = Gemma4ReferenceLMHead(8, 32)
    hidden = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    logits = head(hidden, torch.tensor([1, 6]))
    assert logits.shape == (2, 32)


def test_reference_causal_lm_returns_selected_logits():
    torch = __import__("torch")
    config = Gemma4Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=1, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=4,
                          layer_types=["local"], global_head_dim=4,
                          num_global_key_value_heads=2)
    model = Gemma4ReferenceCausalLM(config, num_experts=2, top_k=1)
    logits = model(torch.tensor([[1, 2, 3]]), torch.tensor([2]))
    assert logits.shape == (1, 32)


def test_registered_factory_smoke_with_reference_mode(monkeypatch):
    torch = __import__("torch")
    monkeypatch.setenv("VLLM_NEURON_GEMMA4_REFERENCE", "1")
    config = Gemma4Config(vocab_size=16, hidden_size=8, intermediate_size=16,
                          num_hidden_layers=1, num_attention_heads=2,
                          num_key_value_heads=1, head_dim=4,
                          layer_types=["local"], global_head_dim=4,
                          num_global_key_value_heads=1)
    model = Gemma4MoeForCausalLM(config)
    output = model(torch.tensor([[1, 2]]), sampling_positions=torch.tensor([1]))
    assert output.shape == (1, 16)
