"""Contract tests for the native Gemma 4 vLLM-Neuron entry point."""

from types import SimpleNamespace

from vllm_neuron.model.gemma4.config import Gemma4Config
from vllm_neuron.model.gemma4.reference import (
    Gemma4RMSNorm,
    Gemma4PagedKVCache,
    Gemma4ReferenceAttention,
    Gemma4ReferenceMoE,
    Gemma4Linear,
    Gemma4ReferenceAttentionBlock,
    Gemma4ReferenceDecoderLayer,
    Gemma4ReferenceTextModel,
    Gemma4ReferenceLMHead,
    Gemma4ReferenceCausalLM,
    Gemma4RotaryEmbedding,
    Gemma4ValueNorm,
)
from vllm_neuron.model.gemma4.weights import Gemma4WeightMapper
from vllm_neuron.model.registry import get_models
from vllm_neuron.model.gemma4.factory import Gemma4ForCausalLM
from vllm_neuron.model.gemma4.model import (
    Gemma4Experts,
    _expert_down_loader,
    _expert_gate_up_loader,
    _padded_kernel_size,
)
from vllm_neuron.vllm.platform import _uses_vision_config


def test_gemma4_is_registered():
    names = {name for name, _ in get_models()}
    assert "Gemma4ForCausalLM" in names
    assert "Gemma4ForConditionalGeneration" in names


def test_text_architecture_does_not_compile_outer_vision_config():
    text_model = SimpleNamespace(
        is_multimodal_model=False,
        hf_config=SimpleNamespace(vision_config=SimpleNamespace()),
    )
    multimodal_model = SimpleNamespace(
        is_multimodal_model=True,
        hf_config=SimpleNamespace(vision_config=SimpleNamespace()),
    )

    assert not _uses_vision_config(text_model)
    assert _uses_vision_config(multimodal_model)


def test_decode_uses_fused_all_expert_tkg(monkeypatch):
    torch = __import__("torch")
    experts = Gemma4Experts.__new__(Gemma4Experts)
    torch.nn.Module.__init__(experts)
    experts.dtype = torch.bfloat16
    experts.hidden_size = 8
    experts.kernel_hidden_size = 8
    experts.tp_group = SimpleNamespace(world_size=1)
    experts.world_size = 1
    experts.gate_up_proj_weight = torch.nn.Parameter(
        torch.zeros(4, 8, 2, 4, dtype=torch.bfloat16)
    )
    experts.down_proj_weight = torch.nn.Parameter(
        torch.zeros(4, 4, 8, dtype=torch.bfloat16)
    )
    experts.router_scale = torch.nn.Parameter(torch.ones(8))
    experts.router_weight = torch.nn.Parameter(torch.zeros(8, 4))
    experts.root_hidden_size = 8**-0.5
    experts.top_k = 2
    experts.eps = 1e-6
    captured = {}

    def fake_moe_block_tkg(**kwargs):
        captured.update(kwargs)
        return torch.zeros_like(kwargs["inp"].squeeze(0))

    monkeypatch.setattr(
        "vllm_neuron.model.gemma4.model.NF.moe_block_tkg",
        fake_moe_block_tkg,
    )
    output = experts(
        torch.ones(1, 8, dtype=torch.bfloat16),
        is_decode=True,
    )

    assert output.shape == (1, 8)
    assert captured["is_all_expert"] is True
    assert torch.equal(captured["rank_id"], torch.zeros(1, 1, dtype=torch.int32))
    assert captured["router_pre_norm"] is True
    assert captured["norm_topk_prob"] is True
    assert captured["hidden_actual"] == 8


def test_prefill_chunks_token_independent_experts_through_tkg(monkeypatch):
    torch = __import__("torch")
    experts = Gemma4Experts.__new__(Gemma4Experts)
    torch.nn.Module.__init__(experts)
    experts.hidden_size = 8
    experts.kernel_hidden_size = 8
    experts.world_size = 1
    experts.tp_group = SimpleNamespace(world_size=1)
    experts.prefill_tkg_chunk_size = 128
    chunks = []

    def fake_run_tkg(hidden_states):
        chunks.append(hidden_states.clone())
        return hidden_states + 1

    monkeypatch.setattr(experts, "_run_tkg", fake_run_tkg)
    hidden_states = torch.arange(256 * 8, dtype=torch.bfloat16).reshape(256, 8)
    output = experts(
        hidden_states,
        is_decode=False,
    )

    assert [chunk.shape for chunk in chunks] == [(128, 8), (128, 8)]
    assert torch.equal(torch.cat(chunks), hidden_states)
    assert torch.equal(output, hidden_states + 1)


def test_expert_weights_pad_hidden_width_for_neuron_moe_tiles():
    torch = __import__("torch")

    class TensorSlice:
        def __init__(self, tensor):
            self.tensor = tensor

        def get_shape(self):
            return self.tensor.shape

        def __getitem__(self, index):
            return self.tensor[index]

    padded_hidden = _padded_kernel_size(2816)
    assert padded_hidden == 3072
    gate_up = TensorSlice(torch.randn(2, 8, 2816))
    down = TensorSlice(torch.randn(2, 2816, 4))
    per_expert_scale = TensorSlice(torch.tensor([0.5, 1.5]))

    loaded_gate_up = _expert_gate_up_loader(
        intermediate_size=4,
        shard_size=4,
        tp_size=1,
        padded_hidden_size=padded_hidden,
    ).load([gate_up], rank=0)
    loaded_down = _expert_down_loader(
        intermediate_size=4,
        shard_size=4,
        tp_size=1,
        padded_hidden_size=padded_hidden,
    ).load([down, per_expert_scale], rank=0)

    assert loaded_gate_up.shape == (2, 3072, 2, 4)
    assert loaded_down.shape == (2, 4, 3072)
    assert torch.count_nonzero(loaded_gate_up[:, 2816:]) == 0
    assert torch.count_nonzero(loaded_down[:, :, 2816:]) == 0
    expected_down = down.tensor.transpose(1, 2)
    assert torch.allclose(loaded_down[0, :, :2816], expected_down[0] * 0.5)
    assert torch.allclose(loaded_down[1, :, :2816], expected_down[1] * 1.5)


def test_nested_text_config_is_parsed():
    config = SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=2816,
            intermediate_size=2112,
            moe_intermediate_size=704,
            num_hidden_layers=30,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=256,
            global_head_dim=512,
            num_global_key_value_heads=2,
            num_experts=128,
            top_k_experts=8,
            enable_moe_block=True,
            vocab_size=262144,
            dtype="bfloat16",
        )
    )
    # The production HF config supplies to_dict(); this test uses the same
    # nested shape without importing Transformers model classes.
    config.to_dict = lambda: {"text_config": config.text_config.__dict__}
    parsed = Gemma4Config.from_configs(config)
    assert parsed.hidden_size == 2816
    assert parsed.intermediate_size == 2112
    assert parsed.moe_intermediate_size == 704
    assert parsed.num_key_value_heads == 8
    assert parsed.head_dim == 256
    assert parsed.global_head_dim == 512
    assert parsed.num_global_key_value_heads == 2
    assert parsed.num_experts == 128
    assert parsed.top_k_experts == 8
    assert parsed.vocab_size == 262144
    assert parsed.torch_dtype == __import__("torch").bfloat16


def test_real_26b_a4b_architecture_defaults():
    config = Gemma4Config()
    assert config.hidden_size == 2816
    assert config.intermediate_size == 2112
    assert config.moe_intermediate_size == 704
    assert config.num_hidden_layers == 30
    assert config.num_experts == 128
    assert config.top_k_experts == 8
    assert config.enable_moe_block
    assert config.attention_k_eq_v
    assert config.final_logit_softcapping == 30.0


def test_layer_attention_shape_preserves_hybrid_layout():
    config = Gemma4Config(
        layer_types=["local", "global"],
        head_dim=256,
        num_key_value_heads=2,
        global_head_dim=512,
        num_global_key_value_heads=1,
    )
    assert config.attention_shape(0) == (256, 2)
    assert config.attention_shape(1) == (512, 1)


def test_rms_norm_matches_reference_formula():
    layer = Gemma4RMSNorm(8, dtype=None)
    layer.weight.data.fill_(1.0)
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
    moe = Gemma4ReferenceMoE(
        hidden_size=8, intermediate_size=16, num_experts=4, top_k=2
    )
    hidden = torch.randn(5, 8)
    output = moe(hidden)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    # Router probabilities are normalized per token before expert combine.
    probabilities, top_weights, _ = moe.router(hidden)
    assert torch.allclose(probabilities.sum(-1), torch.ones(5))
    assert torch.allclose(top_weights.sum(-1), torch.ones(5))


def test_weight_mapper_preserves_expert_indices():
    name = "model.layers.7.experts.down_proj"
    assert Gemma4WeightMapper.checkpoint_name(name) == (
        "model.language_model.layers.7.experts.down_proj"
    )
    assert (
        Gemma4WeightMapper.loader_kind("model.layers.7.self_attn.q_proj.weight")
        == "column"
    )
    assert (
        Gemma4WeightMapper.loader_kind("model.layers.7.self_attn.o_proj.weight")
        == "row"
    )
    assert Gemma4WeightMapper.loader_kind(name) == "expert-local"
    assert (
        Gemma4WeightMapper.loader_kind("model.layers.7.input_layernorm.weight")
        == "replicated"
    )
    assert (
        Gemma4WeightMapper.make_loader("q_proj.weight", 8, 4).__class__.__name__
        == "SafetensorsWeightLoader"
    )
    assert (
        Gemma4WeightMapper.make_loader("o_proj.weight", 8, 4).__class__.__name__
        == "SafetensorsWeightLoader"
    )


def test_linear_attaches_loader_and_uses_local_tp_shape():
    layer = Gemma4Linear(16, 32, "q_proj.weight", tp_size=4)
    assert tuple(layer.weight.shape) == (8, 16)
    assert hasattr(layer.weight, "weight_loader")


def test_reference_attention_block_composes_native_linear_and_cache():
    torch = __import__("torch")
    config = Gemma4Config(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        global_head_dim=4,
        num_global_key_value_heads=2,
        layer_types=["sliding_attention"],
        sliding_window=8,
    )
    block = Gemma4ReferenceAttentionBlock(config, layer_idx=0)
    hidden = torch.randn(3, 16, dtype=torch.bfloat16)
    cache = Gemma4PagedKVCache(3, 2, 4, dtype=torch.bfloat16)
    output = block(
        hidden,
        torch.tensor([0, 1, 2]),
        cache,
        torch.tensor([0, 1, 2]),
    )
    assert output.shape == hidden.shape


def test_reference_decoder_layer_composes_attention_and_moe():
    torch = __import__("torch")
    config = Gemma4Config(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["local"],
        global_head_dim=4,
        num_global_key_value_heads=2,
        moe_intermediate_size=8,
        num_experts=4,
        top_k_experts=2,
    )
    layer = Gemma4ReferenceDecoderLayer(config, layer_idx=0)
    hidden = torch.randn(3, 16, dtype=torch.bfloat16)
    cache = Gemma4PagedKVCache(3, 2, 4, dtype=torch.bfloat16)
    output = layer(
        hidden,
        torch.tensor([0, 1, 2]),
        cache,
        torch.tensor([0, 1, 2]),
    )
    assert output.shape == hidden.shape


def test_reference_text_model_runs_tiny_stack():
    torch = __import__("torch")
    config = Gemma4Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["local"],
        global_head_dim=4,
        num_global_key_value_heads=2,
        moe_intermediate_size=8,
        num_experts=2,
        top_k_experts=1,
    )
    model = Gemma4ReferenceTextModel(config)
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
    config = Gemma4Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["local"],
        global_head_dim=4,
        num_global_key_value_heads=2,
        moe_intermediate_size=8,
        num_experts=2,
        top_k_experts=1,
    )
    model = Gemma4ReferenceCausalLM(config)
    logits = model(torch.tensor([[1, 2, 3]]), torch.tensor([2]))
    assert logits.shape == (1, 32)


def test_registered_factory_smoke_with_reference_mode(monkeypatch):
    torch = __import__("torch")
    monkeypatch.setenv("VLLM_NEURON_GEMMA4_REFERENCE", "1")
    config = Gemma4Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["local"],
        global_head_dim=4,
        num_global_key_value_heads=1,
        moe_intermediate_size=4,
        num_experts=2,
        top_k_experts=1,
    )
    model = Gemma4ForCausalLM.from_configs(config)
    output = model(
        torch.tensor([[1, 2]]),
        positions=torch.tensor([0, 1]),
        sampling_positions=torch.tensor([1]),
    )
    assert output.shape == (1, 16)
    kv_spec = model.get_kv_spec()
    assert len(kv_spec.layers) == 1
    assert kv_spec.layers[0].name == "layers.0.self_attn"
    assert kv_spec.layers[0].head_size == 4


def test_native_model_requires_complete_kv_bindings(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_GEMMA4_REFERENCE", "1")
    config = Gemma4Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["local", "global"],
        global_head_dim=4,
        num_global_key_value_heads=1,
        moe_intermediate_size=4,
        num_experts=2,
        top_k_experts=1,
    )
    model = Gemma4ForCausalLM.from_configs(config)
    try:
        model.bind_kv_cache({"layers.0.self_attn": []})
    except ValueError as error:
        assert "layers.1.self_attn" in str(error)
    else:
        raise AssertionError("incomplete KV binding was accepted")


def test_reference_checkpoint_round_trip_uses_real_text_keys(monkeypatch, tmp_path):
    torch = __import__("torch")
    from safetensors.torch import save_file

    monkeypatch.setenv("VLLM_NEURON_GEMMA4_REFERENCE", "1")
    config = Gemma4Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        global_head_dim=4,
        num_global_key_value_heads=1,
        num_experts=2,
        top_k_experts=1,
        layer_types=["sliding_attention"],
        sliding_window=8,
        torch_dtype=torch.float32,
    )
    model = Gemma4ForCausalLM.from_configs(config)
    checkpoint_tensors = {}
    expected = {}
    for index, (name, parameter) in enumerate(model.named_parameters()):
        value = torch.full_like(parameter, float(index + 1))
        checkpoint_key = Gemma4WeightMapper.checkpoint_name(name)
        assert checkpoint_key.startswith("model.language_model.")
        checkpoint_tensors[checkpoint_key] = value
        expected[name] = value
        parameter.data.zero_()

    save_file(checkpoint_tensors, tmp_path / "model.safetensors")
    model.load_weights(str(tmp_path), torch.device("cpu"), cache_dir=None)

    loaded = dict(model.named_parameters())
    assert set(loaded) == set(expected)
    for name, value in expected.items():
        assert torch.equal(loaded[name], value), name
    assert model._last_checkpoint_load_result.missing_keys == []
    assert model._last_checkpoint_load_result.unexpected_keys == []
