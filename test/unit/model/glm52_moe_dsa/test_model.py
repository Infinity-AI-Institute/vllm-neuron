# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from vllm_neuron.model.glm52_moe_dsa.cache_layout import Glm52CacheLayout
from vllm_neuron.model.glm52_moe_dsa.checkpoint_mapping import (
    build_checkpoint_contract,
)
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.indexer import Glm52IndexShareState
from vllm_neuron.model.glm52_moe_dsa.model import (
    Glm52DecoderLayer,
    Glm52MoeDsaForCausalLM,
    Glm52RotaryEmbedding,
)
from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan
from vllm_neuron.model.glm52_moe_dsa.sparse_mlp import (
    Glm52SparseMlp,
    glm52_rms_norm,
)
from vllm_neuron.model.neuron_config import NeuronConfig


class _Group:
    world_size = 1
    rank_in_group = 0

    @staticmethod
    def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def all_gather(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        del dim
        return tensor

    @staticmethod
    def reduce_scatter(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        del dim
        return tensor


class _Tp64Group:
    world_size = 64
    rank_in_group = 0


class _Ep16ExpertTpGroup:
    world_size = 4
    rank_in_group = 0


def _config(*, layers: int = 1) -> Glm52MoeDsaConfig:
    return Glm52MoeDsaConfig(
        vocab_size=16,
        hidden_size=4,
        num_hidden_layers=layers,
        intermediate_size=8,
        num_attention_heads=1,
        num_key_value_heads=1,
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
        first_k_dense_replace=layers,
        neuron_config=NeuronConfig(
            ep_degree=1,
            on_device_sampling_config=None,
        ),
        torch_dtype=torch.float32,
    )


def test_full_ep16_meta_model_matches_exact_rank_local_storage_contract() -> None:
    config = Glm52MoeDsaConfig(
        neuron_config=NeuronConfig(
            ep_degree=16,
            on_device_sampling_config=None,
        ),
        torch_dtype=torch.bfloat16,
    )
    model = Glm52MoeDsaForCausalLM(
        config,
        world_size=64,
        global_rank=0,
        tp_group=_Tp64Group(),
        expert_tp_group=_Ep16ExpertTpGroup(),
        cache_dtype=torch.float8_e4m3fn,
        device="meta",
    )

    assert len(model.state_dict()) == 2664
    assert all(tensor.device.type == "meta" for tensor in model.state_dict().values())
    assert (
        sum(
            tensor.numel() * tensor.element_size()
            for tensor in model.state_dict().values()
        )
        == 14_456_480_256
    )
    contract = build_checkpoint_contract(
        config,
        model.model.plan,
        global_rank=0,
    )
    assert set(model._loader_mappings(contract)) == set(model.state_dict())


def test_rotary_embedding_matches_frozen_default_formula() -> None:
    config = _config()
    config.qk_rope_head_dim = 4
    rotary = Glm52RotaryEmbedding(config)
    positions = torch.tensor([0, 3], dtype=torch.int32)

    cos, sin = rotary(positions, dtype=torch.float32)

    inv = 1.0 / (8_000_000 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4))
    frequencies = positions[:, None] * inv[None, :]
    expected = torch.cat((frequencies, frequencies), dim=-1)
    torch.testing.assert_close(cos, expected.cos())
    torch.testing.assert_close(sin, expected.sin())


class _Attention(nn.Module):
    cache_name = "layers.0.self_attn"
    indexer = None

    def forward_paged_decode(
        self,
        hidden_states,
        position_embeddings,
        positions,
        attn_metadata,
        **kwargs,
    ):
        del position_embeddings, positions, attn_metadata, kwargs
        state = Glm52IndexShareState(
            topk_indices=torch.zeros(
                hidden_states.shape[0],
                1,
                dtype=torch.int32,
            ),
            source_layer_idx=0,
        )
        return hidden_states * 2, state


class _Mlp(nn.Module):
    def __init__(self, config: Glm52MoeDsaConfig) -> None:
        super().__init__()
        self.config = config

    def forward_decode(self, hidden_states, *, norm_weight):
        return (
            glm52_rms_norm(
                hidden_states,
                norm_weight,
                eps=self.config.rms_norm_eps,
            )
            * 3
        )


def test_decoder_uses_two_pre_norm_residuals_in_checkpoint_order() -> None:
    config = _config()
    layout = Glm52CacheLayout.build(
        config,
        world_size=1,
        cache_dtype=torch.bfloat16,
    )
    layer = Glm52DecoderLayer(
        config,
        layer_idx=0,
        cache_layout=layout,
        plan=RoutedExpertPlan(1, 1, 4, 8),
        world_size=1,
        global_rank=0,
        tp_group=_Group(),
        expert_tp_group=_Group(),
        static_fp8=False,
        attention_module=_Attention(),
        mlp_module=_Mlp(config),
    )
    with torch.no_grad():
        layer.input_layernorm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        layer.post_attention_layernorm.weight.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    layer.key_cache = torch.empty(1)
    layer.value_cache = torch.empty(1)
    hidden = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    metadata = {
        "layers.0.self_attn": {
            "max_query_len": 1,
            "decode_token_threshold": 1,
        }
    }

    actual, state = layer(
        hidden,
        torch.tensor([0]),
        (torch.ones(1, 2), torch.zeros(1, 2)),
        metadata,
        None,
    )

    attention_input = glm52_rms_norm(
        hidden,
        layer.input_layernorm.weight,
        eps=config.rms_norm_eps,
    )
    after_attention = hidden + 2 * attention_input
    expected = after_attention + 3 * glm52_rms_norm(
        after_attention,
        layer.post_attention_layernorm.weight,
        eps=config.rms_norm_eps,
    )
    torch.testing.assert_close(actual, expected)
    assert state.source_layer_idx == 0


def test_causal_lm_binds_paired_caches_and_runs_cpu_decode() -> None:
    config = _config()
    model = Glm52MoeDsaForCausalLM(
        config,
        world_size=1,
        global_rank=0,
        tp_group=_Group(),
        expert_tp_group=_Group(),
        static_fp8=False,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.model.layers[0].input_layernorm.weight.fill_(1)
        model.model.layers[0].post_attention_layernorm.weight.fill_(1)
        model.model.layers[0].self_attn.q_a_layernorm.fill_(1)
        model.model.layers[0].self_attn.kv_a_layernorm.fill_(1)
        model.model.norm.weight.fill_(1)

    caches = {
        "layers.0.self_attn": (
            torch.zeros(1, 1, 2, config.qk_head_dim),
            torch.zeros(1, 1, 2, config.v_head_dim),
        ),
        "glm52.indexer_cache.0": (
            torch.zeros(1, 1, 2, config.index_head_dim),
            torch.zeros(1, 1, 2, config.index_head_dim),
        ),
    }
    model.bind_kv_cache(caches)
    assert model.model.layers[0].indexer_cache is caches["glm52.indexer_cache.0"][0]
    metadata = {
        "layers.0.self_attn": {
            "slot_mapping": torch.tensor([0]),
            "block_size": 2,
            "block_table_tensor": torch.tensor([[0]], dtype=torch.int32),
            "max_query_len": 1,
            "decode_token_threshold": 1,
        },
        "glm52.indexer_cache.0": {
            "slot_mapping": torch.tensor([0]),
            "block_size": 2,
            "block_table_tensor": torch.tensor([[0]], dtype=torch.int32),
        },
    }

    logits = model(
        torch.tensor([2]),
        torch.tensor([0]),
        attn_metadata=metadata,
        sampling_positions=torch.tensor([0]),
    )

    assert logits.shape == (1, config.vocab_size)
    torch.testing.assert_close(logits, torch.zeros_like(logits))


def test_bounded_prefill_reaches_logits_and_updates_paged_caches() -> None:
    config = _config()
    model = Glm52MoeDsaForCausalLM(
        config,
        world_size=1,
        global_rank=0,
        tp_group=_Group(),
        expert_tp_group=_Group(),
        static_fp8=False,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.1)
        model.model.layers[0].input_layernorm.weight.fill_(1)
        model.model.layers[0].post_attention_layernorm.weight.fill_(1)
        model.model.layers[0].self_attn.q_a_layernorm.fill_(1)
        model.model.layers[0].self_attn.kv_a_layernorm.fill_(1)
        model.model.norm.weight.fill_(1)
    caches = {
        "layers.0.self_attn": (
            torch.zeros(1, 1, 2, config.qk_head_dim),
            torch.zeros(1, 1, 2, config.v_head_dim),
        ),
        "glm52.indexer_cache.0": (
            torch.zeros(1, 1, 2, config.index_head_dim),
            torch.zeros(1, 1, 2, config.index_head_dim),
        ),
    }
    model.bind_kv_cache(caches)
    metadata = {
        "layers.0.self_attn": {
            "slot_mapping": torch.tensor([0, 1]),
            "block_size": 2,
            "block_table_tensor": torch.tensor([[0]], dtype=torch.int32),
            "max_query_len": 2,
            "decode_token_threshold": 1,
        },
        "glm52.indexer_cache.0": {
            "slot_mapping": torch.tensor([0, 1]),
            "block_size": 2,
            "block_table_tensor": torch.tensor([[0]], dtype=torch.int32),
        },
    }

    logits = model(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        attn_metadata=metadata,
        sampling_positions=torch.tensor([1]),
    )

    assert logits.shape == (1, config.vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.count_nonzero(caches["layers.0.self_attn"][0]) > 0
    assert torch.count_nonzero(caches["glm52.indexer_cache.0"][0]) > 0


def test_single_sequence_logits_preserve_runtime_position() -> None:
    config = _config()
    model = Glm52MoeDsaForCausalLM(
        config,
        world_size=1,
        global_rank=0,
        tp_group=_Group(),
        expert_tp_group=_Group(),
        static_fp8=False,
    )

    class _Backbone(nn.Module):
        def forward(self, *args, **kwargs):
            del args, kwargs
            return torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ]
            )

    model.model = _Backbone()
    model.lm_head = nn.Identity()
    hidden = model(
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        attn_metadata={"layers.0.self_attn": {}},
        # Short prefill requests are right-padded to a compiled bucket. The
        # requested row therefore cannot be replaced with a static last slice.
        sampling_positions=torch.tensor([0]),
    )

    torch.testing.assert_close(hidden, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))


def test_prefill_above_index_topk_remains_gated() -> None:
    config = _config()
    model = Glm52MoeDsaForCausalLM(
        config,
        world_size=1,
        global_rank=0,
        tp_group=_Group(),
        expert_tp_group=_Group(),
        static_fp8=False,
    )
    metadata = {
        "layers.0.self_attn": {
            "max_query_len": 3,
            "decode_token_threshold": 1,
        }
    }
    try:
        model(
            torch.tensor([1, 2, 3]),
            torch.tensor([0, 1, 2]),
            attn_metadata=metadata,
        )
    except NotImplementedError as error:
        assert "bounded" in str(error)
    else:
        raise AssertionError("prefill above index_topk was accepted")


def test_fp8_cache_layout_requires_calibrated_multipliers() -> None:
    config = _config()
    model = Glm52MoeDsaForCausalLM(
        config,
        world_size=1,
        global_rank=0,
        tp_group=_Group(),
        expert_tp_group=_Group(),
        cache_dtype=torch.float8_e4m3fn,
        static_fp8=False,
    )
    assert all(
        layer.dtype == torch.float8_e4m3fn for layer in model.get_kv_spec().layers
    )
    checkpoint = type(
        "Checkpoint",
        (),
        {"get_tensor_names": lambda self: set()},
    )()
    try:
        model._load_cache_quant_multipliers(checkpoint, torch.device("cpu"))
    except KeyError as error:
        assert "calibrated quantization multipliers" in str(error)
    else:
        raise AssertionError("uncalibrated FP8 cache artifact was accepted")


class _ScalarSlice:
    def __init__(self, value: float) -> None:
        self.tensor = torch.tensor(value)

    def get_shape(self) -> tuple[int, ...]:
        return ()

    def __getitem__(self, item) -> torch.Tensor:
        del item
        return self.tensor


def test_fp8_cache_multiplier_loader_uses_canonical_converter_keys() -> None:
    config = _config()
    model = Glm52MoeDsaForCausalLM(
        config,
        world_size=1,
        global_rank=0,
        tp_group=_Group(),
        expert_tp_group=_Group(),
        cache_dtype=torch.float8_e4m3fn,
        static_fp8=False,
    )
    values = {
        "model.layers.0.self_attn.k_cache_quant_multiplier": 11.0,
        "model.layers.0.self_attn.v_cache_quant_multiplier": 12.0,
        "model.layers.0.self_attn.indexer.cache_quant_multiplier": 13.0,
    }
    checkpoint = type(
        "Checkpoint",
        (),
        {
            "get_tensor_names": lambda self: set(values),
            "_get_slice": lambda self, key: _ScalarSlice(values[key]),
        },
    )()

    model._load_cache_quant_multipliers(checkpoint, torch.device("cpu"))

    attention = model.model.layers[0].self_attn
    assert attention.key_cache_quant_multiplier.item() == 11
    assert attention.value_cache_quant_multiplier.item() == 12
    assert attention.indexer.cache_quant_multiplier.item() == 13


class _ShapeGroup:
    world_size = 2
    rank_in_group = 0

    def __init__(self) -> None:
        self.gathers: list[tuple[tuple[int, ...], int]] = []
        self.scatters: list[tuple[tuple[int, ...], int]] = []

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        self.gathers.append((tuple(tensor.shape), dim))
        return torch.cat((tensor, tensor), dim=dim)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        self.scatters.append((tuple(tensor.shape), dim))
        return torch.chunk(tensor, self.world_size, dim=dim)[self.rank_in_group]


def test_prefill_preserves_sequence_parallel_shape_transitions() -> None:
    config = _config()
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    config.index_topk = 4
    group = _ShapeGroup()
    model = Glm52MoeDsaForCausalLM(
        config,
        world_size=2,
        global_rank=0,
        tp_group=group,
        expert_tp_group=group,
        static_fp8=False,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.model.layers[0].input_layernorm.weight.fill_(1)
        model.model.layers[0].post_attention_layernorm.weight.fill_(1)
        model.model.layers[0].self_attn.q_a_layernorm.fill_(1)
        model.model.layers[0].self_attn.kv_a_layernorm.fill_(1)
        model.model.norm.weight.fill_(1)
    caches = {
        "layers.0.self_attn": (
            torch.zeros(2, 1, 2, config.qk_head_dim),
            torch.zeros(2, 1, 2, config.v_head_dim),
        ),
        "glm52.indexer_cache.0": (
            torch.zeros(2, 1, 2, config.index_head_dim),
            torch.zeros(2, 1, 2, config.index_head_dim),
        ),
    }
    model.bind_kv_cache(caches)
    metadata = {
        "layers.0.self_attn": {
            "slot_mapping": torch.tensor([0, 1, 2, 3]),
            "block_size": 2,
            "block_table_tensor": torch.tensor([[0, 1]], dtype=torch.int32),
            "max_query_len": 4,
            "decode_token_threshold": 1,
        },
        "glm52.indexer_cache.0": {
            "slot_mapping": torch.tensor([0, 1, 2, 3]),
            "block_size": 2,
            "block_table_tensor": torch.tensor([[0, 1]], dtype=torch.int32),
        },
    }

    logits = model(
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([0, 1, 2, 3]),
        attn_metadata=metadata,
        sampling_positions=torch.tensor([3]),
    )

    assert logits.shape == (1, config.vocab_size)
    assert ((4, config.hidden_size), 0) in group.scatters
    assert ((2, config.hidden_size), 0) in group.gathers


class _RankedTp64Group:
    world_size = 64

    def __init__(self, rank: int) -> None:
        self.rank_in_group = rank


class _PrefillAttention(nn.Module):
    cache_name = "layers.0.self_attn"
    indexer = None
    world_size = 64

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.tp_group = _RankedTp64Group(rank)

    def forward_paged_prefill(
        self,
        hidden_states,
        position_embeddings,
        positions,
        attn_metadata,
        **kwargs,
    ):
        del position_embeddings, positions, attn_metadata, kwargs
        state = Glm52IndexShareState(
            topk_indices=torch.zeros(
                hidden_states.shape[0],
                1,
                dtype=torch.int32,
            ),
            source_layer_idx=0,
        )
        return torch.zeros_like(hidden_states), state


class _RecordingSparseMlp(Glm52SparseMlp):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.seen_slot_mapping: torch.Tensor | None = None

    def forward_prefill(self, hidden_states, *, norm_weight, slot_mapping):
        del norm_weight
        self.seen_slot_mapping = slot_mapping.clone()
        return torch.zeros_like(hidden_states)


def test_tp64_prefill_slices_production_2k_padding_slots_per_rank() -> None:
    """A 320-token request in the production 2K bucket leaves ranks 10-63 empty."""

    config = _config()
    config.num_attention_heads = 64
    config.num_key_value_heads = 64
    positions = torch.cat(
        (
            torch.arange(320, dtype=torch.int32),
            torch.full((1_728,), 319, dtype=torch.int32),
        )
    )
    slot_mapping = torch.cat(
        (
            torch.arange(320, dtype=torch.int64),
            torch.full((1_728,), -1, dtype=torch.int64),
        )
    )
    metadata = {
        "layers.0.self_attn": {
            "slot_mapping": slot_mapping,
            "max_query_len": 2_048,
            "decode_token_threshold": 1,
        }
    }

    for rank, expected in (
        (9, torch.arange(288, 320, dtype=torch.int64)),
        (10, torch.full((32,), -1, dtype=torch.int64)),
        (63, torch.full((32,), -1, dtype=torch.int64)),
    ):
        mlp = _RecordingSparseMlp()
        layer = Glm52DecoderLayer(
            config,
            layer_idx=0,
            cache_layout=Glm52CacheLayout.build(
                config,
                world_size=64,
                cache_dtype=torch.bfloat16,
            ),
            plan=RoutedExpertPlan(64, 16, 16, 64),
            world_size=64,
            global_rank=rank,
            tp_group=_RankedTp64Group(rank),
            expert_tp_group=_Group(),
            static_fp8=False,
            attention_module=_PrefillAttention(rank),
            mlp_module=mlp,
        )
        layer.key_cache = torch.empty(1)
        layer.value_cache = torch.empty(1)

        layer(
            torch.zeros(32, config.hidden_size),
            positions,
            (torch.ones(2_048, 2), torch.zeros(2_048, 2)),
            metadata,
            None,
        )

        assert mlp.seen_slot_mapping is not None
        torch.testing.assert_close(mlp.seen_slot_mapping, expected)
