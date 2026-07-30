# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_neuron.model.glm52_moe_dsa.checkpoint_mapping import (
    MTP_IGNORED_PREFIX,
    _to_neuron_legacy_fp8,
    build_checkpoint_contract,
    routed_down_scale_loader,
    routed_down_weight_loader,
    routed_gate_up_scale_loader,
    routed_gate_up_weight_loader,
)
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan
from vllm_neuron.model.glm52_moe_dsa.static_fp8 import (
    NEURON_LEGACY_E4M3FN_QMAX240,
)


class FakeSlice:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def get_shape(self) -> list[int]:
        return list(self.tensor.shape)

    def __getitem__(self, index):
        return self.tensor[index]


def _small_plan() -> RoutedExpertPlan:
    return RoutedExpertPlan(
        world_size=4,
        ep_degree=2,
        num_experts=4,
        expert_intermediate_size=8,
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


def test_contract_filters_experts_before_tp_sharding() -> None:
    contract = build_checkpoint_contract(
        _small_config(),
        _small_plan(),
        global_rank=2,
    )
    prefix = "model.layers.1.mlp.experts"

    assert contract.mappings[f"{prefix}.gate_up_proj"] == [
        "model.layers.1.mlp.experts.2.gate_proj.weight",
        "model.layers.1.mlp.experts.2.up_proj.weight",
        "model.layers.1.mlp.experts.3.gate_proj.weight",
        "model.layers.1.mlp.experts.3.up_proj.weight",
    ]
    assert contract.mappings[f"{prefix}.down_proj_scale"] == [
        "model.layers.1.mlp.experts.2.down_proj.weight_scale",
        "model.layers.1.mlp.experts.3.down_proj.weight_scale",
    ]
    assert all(
        not key.startswith("model.layers.78.") for key in contract.required_source_keys
    )


def test_contract_rejects_missing_required_tensor_and_tracks_mtp() -> None:
    contract = build_checkpoint_contract(
        _small_config(),
        _small_plan(),
        global_rank=0,
    )
    source_keys = set(contract.required_source_keys)
    missing = next(iter(source_keys))
    source_keys.remove(missing)

    try:
        contract.validate_source_keys(source_keys)
    except KeyError as error:
        assert missing in str(error)
    else:
        raise AssertionError("missing source tensor was accepted")

    mtp_key = f"{MTP_IGNORED_PREFIX}self_attn.q_a_proj.weight"
    source_keys.add(mtp_key)
    assert contract.ignored_source_keys(source_keys) == {mtp_key}


def test_contract_maps_attention_static_scales() -> None:
    contract = build_checkpoint_contract(
        _small_config(),
        _small_plan(),
        global_rank=0,
    )
    prefix = "model.layers.0.self_attn.q_a_proj"

    assert contract.mappings[f"{prefix}.weight_scale"] == (f"{prefix}.weight_scale")
    assert contract.mappings[f"{prefix}.input_scale"] == f"{prefix}.input_scale"


def test_contract_maps_dense_static_scales() -> None:
    contract = build_checkpoint_contract(
        _small_config(),
        _small_plan(),
        global_rank=0,
    )
    prefix = "model.layers.0.mlp"

    assert contract.mappings[f"{prefix}.gate_proj.weight_scale"] == (
        f"{prefix}.gate_proj.weight_scale"
    )
    assert contract.mappings[f"{prefix}.gate_up_input_scale"] == [
        f"{prefix}.gate_proj.input_scale",
        f"{prefix}.up_proj.input_scale",
    ]
    assert contract.mappings[f"{prefix}.down_input_scale"] == (
        f"{prefix}.down_proj.input_scale"
    )


def test_hybrid_contract_maps_bf16_shared_weights_without_scales() -> None:
    config = _small_config()
    config.shared_expert_dtype = "bfloat16"
    contract = build_checkpoint_contract(
        config,
        _small_plan(),
        global_rank=0,
    )
    shared = "model.layers.1.mlp.shared_experts"

    for projection in ("gate_proj", "up_proj", "down_proj"):
        weight = f"{shared}.{projection}.weight"
        assert contract.mappings[weight] == weight
        assert f"{weight}_scale" not in contract.mappings
    assert f"{shared}.gate_up_input_scale" not in contract.mappings
    assert f"{shared}.down_input_scale" not in contract.mappings


def test_routed_weight_loaders_match_tkg_and_cte_layouts() -> None:
    plan = _small_plan()
    hidden = 3
    gate_up_slices = []
    down_slices = []
    for expert in range(plan.experts_per_rank):
        gate = torch.arange(8 * hidden).reshape(8, hidden) + expert * 10
        up = gate + 100
        down = torch.arange(hidden * 8).reshape(hidden, 8) + expert * 20
        gate_up_slices.extend((FakeSlice(gate), FakeSlice(up)))
        down_slices.append(FakeSlice(down))

    gate_up = routed_gate_up_weight_loader(plan).load(gate_up_slices, rank=1)
    down = routed_down_weight_loader(plan).load(down_slices, rank=1)

    assert gate_up.shape == (2, hidden, 2, 4)
    assert down.shape == (2, 4, hidden)
    assert gate_up.dtype == torch.float8_e4m3fn
    assert down.dtype == torch.float8_e4m3fn
    scale = 240.0 / 448.0
    expected_gate = (gate_up_slices[0].tensor[4:8, :].T.float() * scale).to(
        torch.float8_e4m3fn
    )
    expected_up = (gate_up_slices[1].tensor[4:8, :].T.float() * scale).to(
        torch.float8_e4m3fn
    )
    expected_down = (down_slices[1].tensor[:, 4:8].T.float() * scale).to(
        torch.float8_e4m3fn
    )
    torch.testing.assert_close(gate_up[0, :, 0, :], expected_gate)
    torch.testing.assert_close(gate_up[0, :, 1, :], expected_up)
    torch.testing.assert_close(down[1], expected_down)


def test_routed_scalar_scales_broadcast_without_semantic_change() -> None:
    plan = _small_plan()
    gate_up_slices = [
        FakeSlice(torch.tensor(value, dtype=torch.float32))
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    down_slices = [
        FakeSlice(torch.tensor([5.0], dtype=torch.float32)),
        FakeSlice(torch.tensor([6.0], dtype=torch.float32)),
    ]

    gate_up = routed_gate_up_scale_loader(plan).load(gate_up_slices, rank=3)
    down = routed_down_scale_loader(plan, hidden_size=3).load(
        down_slices,
        rank=3,
    )

    assert gate_up.shape == (2, 2, 4)
    assert down.shape == (2, 3)
    compensation = 448.0 / 240.0
    torch.testing.assert_close(
        gate_up[0, 0],
        torch.full((4,), compensation),
    )
    torch.testing.assert_close(
        gate_up[0, 1],
        torch.full((4,), 2.0 * compensation),
    )
    torch.testing.assert_close(
        down[1],
        torch.full((3,), 6.0 * compensation),
    )


def test_absent_format_preserves_exact_ocp448_loader_behavior() -> None:
    source = torch.tensor(
        [-448.0, -256.0, -1.0, 0.0, 1.0, 256.0, 448.0],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    expected = (
        (source.to(torch.float32) * (240.0 / 448.0))
        .clamp(-240.0, 240.0)
        .to(torch.float8_e4m3fn)
    )

    actual = _to_neuron_legacy_fp8(source)

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_direct_legacy_loader_preserves_fp8_bytes_and_scales() -> None:
    source = torch.tensor(
        [-240.0, -128.0, -1.0, 0.0, 1.0, 128.0, 240.0],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)

    prepared = _to_neuron_legacy_fp8(
        source,
        NEURON_LEGACY_E4M3FN_QMAX240,
    )

    assert prepared.data_ptr() == source.data_ptr()
    assert torch.equal(prepared.view(torch.uint8), source.view(torch.uint8))

    plan = _small_plan()
    scale_slices = [
        FakeSlice(torch.tensor(value, dtype=torch.float32))
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    scales = routed_gate_up_scale_loader(
        plan,
        weight_format=NEURON_LEGACY_E4M3FN_QMAX240,
    ).load(scale_slices, rank=0)
    torch.testing.assert_close(scales[0, 0], torch.ones(4))
    torch.testing.assert_close(scales[0, 1], torch.full((4,), 2.0))


def test_direct_legacy_loader_rejects_out_of_contract_value() -> None:
    source = torch.tensor([256.0], dtype=torch.float32).to(torch.float8_e4m3fn)

    try:
        _to_neuron_legacy_fp8(source, NEURON_LEGACY_E4M3FN_QMAX240)
    except ValueError as error:
        assert "outside the declared qmax-240 range" in str(error)
    else:
        raise AssertionError("out-of-contract direct FP8 value was accepted")


def test_direct_legacy_loader_preserves_noncontiguous_bytes_contiguously() -> None:
    base = torch.tensor(
        [
            [-240.0, -128.0, -1.0],
            [0.0, 1.0, 128.0],
            [240.0, 16.0, -16.0],
        ],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    source = base.T
    assert not source.is_contiguous()

    prepared = _to_neuron_legacy_fp8(
        source,
        NEURON_LEGACY_E4M3FN_QMAX240,
    )

    assert prepared.is_contiguous()
    assert torch.equal(
        prepared.view(torch.uint8),
        source.contiguous().view(torch.uint8),
    )


def test_native_block_scale_is_rejected_instead_of_broadcast() -> None:
    plan = _small_plan()
    block_scales = [FakeSlice(torch.ones(2, 2)) for _ in range(4)]

    try:
        routed_gate_up_scale_loader(plan).load(block_scales, rank=0)
    except ValueError as error:
        assert "native 128x128 block-FP8 checkpoint must be converted" in str(error)
    else:
        raise AssertionError("native block scales were accepted as static scales")
