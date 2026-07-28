import torch

from examples.vllm_neuron.glm52.sparse_attention_probe import (
    SparseAttentionDenseProbe,
    SparseAttentionPagedProbe,
    _dense_inputs,
    _paged_inputs,
)
from vllm_neuron.model.glm52_moe_dsa.config import Glm52MoeDsaConfig


def test_dense_probe_reference_selects_only_topk_values() -> None:
    config = Glm52MoeDsaConfig(
        qk_head_dim=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        index_topk=4,
    )
    inputs, expected = _dense_inputs(config, context=8)

    output = SparseAttentionDenseProbe()(*inputs)

    torch.testing.assert_close(output, expected)


def test_paged_probe_reference_mutates_current_value() -> None:
    config = Glm52MoeDsaConfig(
        qk_head_dim=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        index_topk=4,
    )
    inputs, expected = _paged_inputs(config, context=128)

    output = SparseAttentionPagedProbe()(*inputs)

    torch.testing.assert_close(output, expected)
