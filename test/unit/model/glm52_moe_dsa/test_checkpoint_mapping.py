# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_neuron.model.glm52_moe_dsa.checkpoint_mapping import (
    MTP_IGNORED_PREFIX,
    build_checkpoint_contract,
    routed_down_scale_loader,
    routed_down_weight_loader,
    routed_gate_up_scale_loader,
    routed_gate_up_weight_loader,
)
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig
from vllm_neuron.model.glm52_moe_dsa.parallelism import RoutedExpertPlan


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
        not key.startswith("model.layers.78.")
        for key in contract.required_source_keys
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


def test_routed_weight_loaders_match_tkg_and_cte_layouts() -> None:
    plan = _small_plan()
    hidden = 3
    gate_up_slices = []
    down_slices = []
    for expert in range(plan.experts_per_rank):
        gate = torch.arange(8 * hidden).reshape(8, hidden) + expert * 1_000
        up = gate + 100
        down = torch.arange(hidden * 8).reshape(hidden, 8) + expert * 2_000
        gate_up_slices.extend((FakeSlice(gate), FakeSlice(up)))
        down_slices.append(FakeSlice(down))

    gate_up = routed_gate_up_weight_loader(plan).load(gate_up_slices, rank=1)
    down = routed_down_weight_loader(plan).load(down_slices, rank=1)

    assert gate_up.shape == (2, hidden, 2, 4)
    assert down.shape == (2, 4, hidden)
    torch.testing.assert_close(
        gate_up[0, :, 0, :],
        gate_up_slices[0].tensor[4:8, :].T,
    )
    torch.testing.assert_close(
        gate_up[0, :, 1, :],
        gate_up_slices[1].tensor[4:8, :].T,
    )
    torch.testing.assert_close(down[1], down_slices[1].tensor[:, 4:8].T)


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
    torch.testing.assert_close(gate_up[0, 0], torch.ones(4))
    torch.testing.assert_close(gate_up[0, 1], torch.full((4,), 2.0))
    torch.testing.assert_close(down[1], torch.full((3,), 6.0))


def test_native_block_scale_is_rejected_instead_of_broadcast() -> None:
    plan = _small_plan()
    block_scales = [FakeSlice(torch.ones(2, 2)) for _ in range(4)]

    try:
        routed_gate_up_scale_loader(plan).load(block_scales, rank=0)
    except ValueError as error:
        assert "native 128x128 block-FP8 checkpoint must be converted" in str(error)
    else:
        raise AssertionError("native block scales were accepted as static scales")
