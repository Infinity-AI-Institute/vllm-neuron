"""Contracts for the native Inkling-Small vLLM-Neuron entry point."""

import io
import json
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vllm_neuron.model.inkling import routing as routing_module
from vllm_neuron.model.inkling.config import InklingConfig
from vllm_neuron.model.inkling.factory import InklingForConditionalGeneration
from vllm_neuron.model.inkling.model import (
    InklingAttention,
    InklingPagedConv,
    _dense_gate_up_loader,
    _last_selected_row_or,
    _local_expert_down_loader,
    _local_expert_gate_up_loader,
    _shared_down_loader,
    _shared_gate_up_loader,
)
from vllm_neuron.model.inkling.model import (
    InklingForConditionalGeneration as NativeInklingForConditionalGeneration,
)
from vllm_neuron.model.inkling.routing import (
    inkling_route,
    inkling_route_from_logits,
)
from vllm_neuron.model.registry import get_models


def test_inkling_conditional_generation_is_registered():
    names = {name for name, _ in get_models()}
    assert "InklingForConditionalGeneration" in names
    assert "InklingForCausalLM" in names


def test_model_config_hook_registers_architecture_lazily():
    from vllm_neuron.vllm.platform import (
        _register_pre_model_config_architectures,
    )

    class Registry:
        def __init__(self):
            self.models = {}

        def get_supported_archs(self):
            return self.models.keys()

        def register_model(self, name, model):
            self.models[name] = model

    registry = Registry()
    _register_pre_model_config_architectures(registry)
    names = set(registry.models)
    assert {
        "InklingForConditionalGeneration",
        "InklingForCausalLM",
    } <= names
    assert all(
        model.endswith(":InklingForConditionalGeneration")
        for model in registry.models.values()
    )


def test_factory_exposes_vllm_text_generation_inspection_protocol():
    from vllm.model_executor.models.interfaces_base import (
        is_text_generation_model,
    )

    assert is_text_generation_model(InklingForConditionalGeneration)


def test_public_small_checkpoint_config_is_parsed():
    raw = {
        "architectures": ["InklingForConditionalGeneration"],
        "model_type": "inkling_mm_model",
        "eos_token_id": 200006,
        "text_config": {
            "model_max_length": 1048576,
            "torch_dtype": "bfloat16",
            "hidden_size": 4096,
            "num_hidden_layers": 42,
            "vocab_size": 201024,
            "unpadded_vocab_size": 200058,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "local_layer_ids": [0, 1, 3],
            "n_routed_experts": 256,
            "num_experts_per_tok": 6,
            "n_shared_experts": 2,
            "dense_mlp_idx": 2,
            "use_sconv": True,
            "sconv_kernel_size": 4,
        },
    }
    hf_config = SimpleNamespace(to_dict=lambda: raw)
    config = InklingConfig.from_configs(hf_config)

    assert config.hidden_size == 4096
    assert config.num_hidden_layers == 42
    assert config.max_position_embeddings == 1048576
    assert config.torch_dtype is torch.bfloat16
    assert config.eos_token_id == 200006
    assert config.padded_vocab_size == 201024
    assert config.unpadded_vocab_size == 200058
    assert config.layer_is_dense(0)
    assert not config.layer_is_dense(2)
    assert config.layer_is_local(3)
    assert config.conv_cache_heads == 66


def test_raw_checkpoint_config_wins_over_lossy_transformers_config(tmp_path):
    checkpoint_config = {
        "eos_token_id": 200006,
        "text_config": {
            "hidden_size": 4096,
            "dense_intermediate_size": 16384,
            "intermediate_size": 2048,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config))
    normalized = SimpleNamespace(
        name_or_path=str(tmp_path),
        to_dict=lambda: {
            "text_config": {
                "hidden_size": 4096,
                "intermediate_size": 16384,
                "moe_intermediate_size": 3072,
            }
        },
    )

    config = InklingConfig.from_configs(normalized)

    assert config.dense_intermediate_size == 16384
    assert config.intermediate_size == 2048
    assert config.eos_token_id == 200006


def test_router_matches_checkpoint_equations():
    logits = torch.tensor(
        [
            [0.7, -0.4, 1.1, 0.2, -0.3, 0.8],
            [-1.0, 0.3, 0.2, 0.9, 0.1, -0.6],
        ],
        dtype=torch.float32,
    )
    correction = torch.tensor([0.0, 0.4, -0.2, 0.1])
    global_scale = torch.tensor([0.75])

    affinities, indices, gammas = inkling_route_from_logits(
        logits,
        correction,
        global_scale,
        num_routed_experts=4,
        num_shared_experts=2,
        top_k=2,
        route_scale=8.0,
    )

    expected_indices = torch.topk(
        torch.sigmoid(logits[:, :4]) + correction, 2, dim=-1, sorted=False
    ).indices
    selected = logits[:, :4].gather(-1, expected_indices)
    active = torch.cat((selected, logits[:, 4:]), dim=-1)
    expected = torch.softmax(F.logsigmoid(active), dim=-1) * 6.0

    assert torch.equal(indices.long(), expected_indices)
    assert torch.allclose(affinities.gather(-1, indices.long()), expected[:, :2])
    assert torch.allclose(gammas, expected[:, 2:])
    assert torch.allclose(affinities.sum(-1) + gammas.sum(-1), torch.full((2,), 6.0))


def test_router_dispatches_selection_through_neuron_topk(monkeypatch):
    calls = []

    def record_topk(tensor, *, k, dim, gather_dim, process_group):
        calls.append((tensor.shape, k, dim, gather_dim, process_group))
        return torch.topk(tensor, k, dim=dim)

    monkeypatch.setattr(routing_module, "neuron_topk", record_topk)
    inkling_route_from_logits(
        torch.tensor([[0.2, -0.4, 0.7, 0.1, -0.3]]),
        correction_bias=torch.zeros(4),
        global_scale=torch.ones(1),
        num_routed_experts=4,
        num_shared_experts=1,
        top_k=2,
        route_scale=1.0,
    )

    assert calls == [(torch.Size([1, 4]), 2, -1, -1, None)]


def test_router_projection_is_fp32_and_keeps_shared_rows_out_of_topk():
    hidden = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [3.0, 3.0]], dtype=torch.bfloat16)
    logits, affinities, indices, gammas = inkling_route(
        hidden,
        weight,
        correction_bias=torch.zeros(2),
        global_scale=torch.ones(1),
        num_routed_experts=2,
        num_shared_experts=1,
        top_k=1,
        route_scale=1.0,
    )
    assert logits.dtype is torch.float32
    assert indices.item() == 1
    assert affinities.shape == (1, 2)
    assert gammas.shape == (1, 1)


def test_last_selected_cache_row_is_tensor_traceable_selection():
    rows = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
            [[7.0, 8.0]],
        ]
    )
    fallback = torch.tensor([[9.0, 10.0]])

    assert torch.equal(
        _last_selected_row_or(
            rows,
            torch.tensor([False, True, False, True]),
            fallback,
        ),
        rows[3],
    )
    assert torch.equal(
        _last_selected_row_or(
            rows,
            torch.zeros(4, dtype=torch.bool),
            fallback,
        ),
        fallback,
    )


def test_factory_rejects_approximate_router_contract():
    config = InklingConfig(use_gate_bias=False)
    try:
        InklingForConditionalGeneration._validate_config(config)
    except NotImplementedError as error:
        assert "sigmoid router contract" in str(error)
    else:
        raise AssertionError("approximate Inkling router contract was accepted")


def test_interleaved_gate_up_loaders_preserve_checkpoint_contract():
    hidden = 3
    intermediate = 4
    dense_source = torch.arange(2 * intermediate * hidden, dtype=torch.float32).view(
        2 * intermediate, hidden
    )
    dense = _dense_gate_up_loader(
        intermediate_size=intermediate,
        shard_size=2,
        tp_size=2,
    ).load([dense_source], rank=1)
    assert dense.shape == (hidden, 2, 2)
    for channel in range(hidden):
        for local_i in range(2):
            for projection in range(2):
                source_row = 2 * (2 + local_i) + projection
                assert (
                    dense[channel, local_i, projection]
                    == dense_source[source_row, channel]
                )

    expert_source = torch.arange(
        4 * 2 * intermediate * hidden, dtype=torch.float32
    ).view(4, 2 * intermediate, hidden)
    expert = _local_expert_gate_up_loader(
        num_local_experts=2,
        intermediate_size=intermediate,
    ).load([expert_source], rank=1)
    assert expert.shape == (2, hidden, 2, intermediate)
    assert expert[0, 1, 0, 3] == expert_source[2, 6, 1]
    assert expert[1, 2, 1, 1] == expert_source[3, 3, 2]

    shared_source = expert_source[:2]
    shared = _shared_gate_up_loader(
        intermediate_size=intermediate,
        shard_size=2,
        tp_size=2,
    ).load([shared_source], rank=1)
    assert shared.shape == (2, hidden, 2, 2)
    assert shared[1, 2, 1, 1] == shared_source[1, 7, 2]


def test_expert_down_loaders_preserve_checkpoint_contract():
    source = torch.arange(4 * 3 * 4, dtype=torch.float32).view(4, 3, 4)
    local = _local_expert_down_loader(num_local_experts=2).load([source], rank=1)
    assert local.shape == (2, 4, 3)
    assert local[0, 1, 2] == source[2, 2, 1]

    shared = _shared_down_loader(shard_size=2, tp_size=2).load([source[:2]], rank=1)
    assert shared.shape == (2, 2, 3)
    assert shared[1, 1, 2] == source[1, 2, 3]


def test_short_convolution_paged_state_matches_causal_reference():
    config = InklingConfig(
        hidden_size=4,
        head_dim=2,
        sconv_kernel_size=2,
    )
    conv = InklingPagedConv(config, layer_idx=0)
    conv.cache = torch.zeros(2, config.conv_cache_heads, 4, config.head_dim)
    metadata = {
        "block_size": 4,
        "slot_mapping": torch.tensor([0, 1, 2]),
        "block_table_tensor": torch.tensor([[0, 1]]),
    }
    states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weight = torch.tensor([[[10.0, 100.0]], [[20.0, 200.0]]])
    actual, cache = conv(
        states,
        torch.tensor([0, 1, 2]),
        weight,
        InklingPagedConv.K,
        metadata,
        conv.cache,
    )
    expected = states.clone()
    expected += states * torch.tensor([100.0, 200.0])
    expected[1:] += states[:-1] * torch.tensor([10.0, 20.0])
    assert torch.equal(actual, expected)

    decode_metadata = {
        **metadata,
        "slot_mapping": torch.tensor([3]),
    }
    decode_state = torch.tensor([[7.0, 8.0]])
    decode, cache = conv(
        decode_state,
        torch.tensor([3]),
        weight,
        InklingPagedConv.K,
        decode_metadata,
        cache,
    )
    decode_expected = (
        decode_state
        + decode_state * torch.tensor([100.0, 200.0])
        + states[-1:] * torch.tensor([10.0, 20.0])
    )
    assert torch.equal(decode, decode_expected)


def test_short_convolution_uses_physical_block_table_for_decode_history():
    config = InklingConfig(
        hidden_size=4,
        head_dim=2,
        sconv_kernel_size=2,
    )
    conv = InklingPagedConv(config, layer_idx=0)
    conv.cache = torch.zeros(4, config.conv_cache_heads, 4, config.head_dim)
    prefill_states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weight = torch.tensor([[[10.0, 100.0]], [[20.0, 200.0]]])
    metadata = {
        "block_size": 4,
        "slot_mapping": torch.tensor([8, 9, 10]),
        "block_table_tensor": torch.tensor([[2, -1]]),
    }
    _, cache = conv(
        prefill_states,
        torch.tensor([0, 1, 2]),
        weight,
        InklingPagedConv.K,
        metadata,
        conv.cache,
    )

    decode_state = torch.tensor([[7.0, 8.0]])
    actual, cache = conv(
        decode_state,
        torch.tensor([3]),
        weight,
        InklingPagedConv.K,
        {**metadata, "slot_mapping": torch.tensor([11])},
        cache,
    )
    expected = (
        decode_state
        + decode_state * torch.tensor([100.0, 200.0])
        + prefill_states[-1:] * torch.tensor([10.0, 20.0])
    )
    assert torch.equal(actual, expected)


def test_short_convolution_threads_all_four_stream_writes():
    config = InklingConfig(
        hidden_size=4,
        head_dim=2,
        sconv_kernel_size=2,
    )
    conv = InklingPagedConv(config, layer_idx=0)
    cache = torch.zeros(2, config.conv_cache_heads, 4, config.head_dim)
    metadata = {
        "block_size": 4,
        "slot_mapping": torch.tensor([0]),
        "block_table_tensor": torch.tensor([[0, 1]]),
    }
    weight = torch.zeros(4, 1, config.sconv_kernel_size)
    widths = (2, 2, 4, 4)

    for stream, width in enumerate(widths):
        states = torch.full((1, width), float(stream + 1))
        stream_weight = weight[:width]
        _, cache = conv(
            states,
            torch.tensor([0]),
            stream_weight,
            stream,
            metadata,
            cache,
        )

    expected = torch.tensor([1.0, 2.0, 3.0, 3.0, 4.0, 4.0])
    assert torch.equal(cache[0, :, 0, 0], expected)


def test_paged_decode_attention_matches_fresh_full_prefix():
    attention = object.__new__(InklingAttention)
    torch.nn.Module.__init__(attention)
    attention.head_dim = 2
    attention.num_kv_heads_per_rank = 1
    attention.sliding_window = None
    attention.relative_extent = 8
    attention.relative_projection = torch.nn.Parameter(torch.zeros(1, 8))
    attention.config = SimpleNamespace(log_scaling_n_floor=None)
    attention.kv_cache = torch.zeros(4, 2, 2, 2)

    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[1.0, 1.0]],
            [[2.0, -1.0]],
        ]
    )
    key = torch.tensor(
        [
            [[1.0, 0.5]],
            [[0.5, 1.0]],
            [[1.0, -0.5]],
            [[-0.5, 2.0]],
        ]
    )
    value = torch.tensor(
        [
            [[1.0, 2.0]],
            [[3.0, 4.0]],
            [[5.0, 6.0]],
            [[7.0, 8.0]],
        ]
    )
    positions = torch.arange(4)
    relative = torch.zeros(4, 1, 1)
    valid = torch.ones(4, dtype=torch.bool)
    metadata = {
        "block_size": 2,
        "slot_mapping": torch.tensor([4, 5, 6, 7]),
        "block_table_tensor": torch.tensor([[2, 3]]),
    }
    _, kv_cache = attention._write_cache(key, value, metadata)
    assert torch.equal(kv_cache[2:, 0], key.reshape(2, 2, 2))
    assert torch.equal(kv_cache[2:, 1], value.reshape(2, 2, 2))

    expected = attention._prefill_attention(
        query, key, value, relative, positions, valid
    )[-1:]
    actual = attention._decode_attention(
        query[-1:],
        relative[-1:],
        positions[-1:],
        valid[-1:],
        metadata,
        kv_cache,
    )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_kv_debug_snapshot_splits_page_major_packed_head_root():
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    root = torch.arange(3 * 2 * 2 * 2).view(3, 2, 2, 2)
    attention = SimpleNamespace(
        kv_cache=root,
        num_kv_heads_per_rank=1,
    )
    runner = object.__new__(NeuronModelRunner)
    runner.model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(attention=attention)])
    )

    snapshots = runner.get_kv_caches()
    payload = torch.load(
        io.BytesIO(snapshots["layers.0.self_attn"]),
        weights_only=True,
    )

    assert torch.equal(payload["k"], root[:, :1])
    assert torch.equal(payload["v"], root[:, 1:])


def test_hybrid_cache_binding_uses_one_canonical_allocation_root():
    native = object.__new__(NativeInklingForConditionalGeneration)
    torch.nn.Module.__init__(native)
    layers = []
    for layer_idx in range(6):
        layers.append(
            SimpleNamespace(
                attention=SimpleNamespace(
                    layer_name=f"layers.{layer_idx}.self_attn",
                    kv_cache=None,
                    kv_cache_shape=None,
                ),
                conv=SimpleNamespace(
                    layer_name=f"layers.{layer_idx}.conv_state",
                    cache_shape=None,
                ),
                cache_root=None,
                cache_allocation_index=None,
            )
        )
    native.model = SimpleNamespace(layers=layers)

    # Both shapes have 128 elements and reinterpret one canonical allocation.
    typed_root = torch.zeros(128)
    cache_views = {}
    allocation_roots = {}
    for layer in layers:
        cache_views[layer.attention.layer_name] = typed_root.view(2, 4, 1, 8, 2)
        cache_views[layer.conv.layer_name] = typed_root.view(2, 4, 4, 2, 2)
        allocation_roots[layer.attention.layer_name] = typed_root
        allocation_roots[layer.conv.layer_name] = typed_root

    native.bind_kv_cache_roots(cache_views)
    native.bind_kv_cache_allocation_roots(allocation_roots)

    assert {layer.cache_allocation_index for layer in layers} == {0}
    assert all(layer.cache_root is typed_root for layer in layers)
    assert all(layer.attention.kv_cache_shape == (4, 2, 8, 2) for layer in layers)
    assert all(layer.conv.cache_shape == (4, 8, 2, 2) for layer in layers)


def test_combined_kv_write_does_not_clobber_slot_zero_for_padding():
    attention = object.__new__(InklingAttention)
    torch.nn.Module.__init__(attention)
    attention.layer_idx = 0
    attention.head_dim = 2
    attention.num_kv_heads_per_rank = 1
    attention.kv_cache = torch.zeros(2, 2, 2, 2)
    attention.kv_cache[0, 0, 0] = torch.tensor([9.0, 10.0])
    attention.kv_cache[0, 1, 0] = torch.tensor([11.0, 12.0])
    key = torch.tensor([[[1.0, 2.0]], [[99.0, 99.0]]])
    value = torch.tensor([[[3.0, 4.0]], [[88.0, 88.0]]])
    metadata = {
        "block_size": 2,
        "slot_mapping": torch.tensor([1, -1]),
    }

    valid, kv_cache = attention._write_cache(key, value, metadata)

    assert torch.equal(valid, torch.tensor([True, False]))
    assert torch.equal(
        kv_cache[0, :, 0],
        torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
    )
    assert torch.equal(
        kv_cache[0, :, 1],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_relative_logits_use_query_conditioned_backward_distance():
    attention = object.__new__(InklingAttention)
    torch.nn.Module.__init__(attention)
    attention.relative_extent = 4
    attention.relative_projection = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    )
    relative = torch.tensor([[[1.0]], [[10.0]], [[100.0]]])
    actual = attention._relative_bias(
        relative,
        query_positions=torch.tensor([0, 1, 2]),
        key_positions=torch.tensor([0, 1, 2]),
    )
    expected = torch.tensor(
        [[[1.0, 0.0, 0.0], [20.0, 10.0, 0.0], [300.0, 200.0, 100.0]]]
    )
    assert torch.equal(actual, expected)


def test_swa_block_count_handles_hybrid_blocks_wider_than_pmax():
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    runner = object.__new__(NeuronModelRunner)
    runner._dcp_size = 1
    runner.max_model_len = 128

    assert runner._compute_swa_num_blocks(sliding_window=4, block_size=256) == 1


def test_swa_block_count_keeps_normal_pmax_alignment():
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    runner = object.__new__(NeuronModelRunner)
    runner._dcp_size = 1
    runner.max_model_len = 4096

    assert runner._compute_swa_num_blocks(sliding_window=512, block_size=16) == 40
