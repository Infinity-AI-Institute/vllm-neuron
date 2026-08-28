# SPDX-License-Identifier: Apache-2.0

import math

import torch

from vllm_neuron.model.glm52_moe_dsa.attention import (
    apply_glm52_interleaved_rope,
    glm52_index_scores,
    glm52_index_topk,
    glm52_sparse_attention,
)


def test_interleaved_rope_matches_frozen_layout() -> None:
    query = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    key = torch.tensor([[[[5.0, 6.0, 7.0, 8.0]]]])
    cos = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])
    sin = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])

    query_rotated, key_rotated = apply_glm52_interleaved_rope(
        query,
        key,
        cos,
        sin,
    )

    torch.testing.assert_close(
        query_rotated,
        torch.tensor([[[[-2.0, 3.0, 1.0, 4.0]]]]),
    )
    torch.testing.assert_close(
        key_rotated,
        torch.tensor([[[[-6.0, 7.0, 5.0, 8.0]]]]),
    )


def test_indexer_relu_precedes_weighted_head_sum() -> None:
    query = torch.tensor([[[[1.0], [-2.0]]]])
    key_cache = torch.tensor([[[1.0]]])
    head_weights = torch.tensor([[[1.0, -1.0]]])

    scores = glm52_index_scores(query, key_cache, head_weights)

    torch.testing.assert_close(
        scores,
        torch.tensor([[[1.0 / math.sqrt(2.0)]]]),
    )


def test_index_topk_excludes_future_when_k_fits_history() -> None:
    scores = torch.tensor([[[1.0, 4.0, 3.0, 2.0]]])
    position_ids = torch.tensor([[1]])

    indices = glm52_index_topk(
        scores,
        top_k=2,
        position_ids=position_ids,
    )

    assert indices.dtype is torch.int64
    assert set(indices[0, 0].tolist()) == {0, 1}


def test_index_topk_clamps_to_context() -> None:
    scores = torch.tensor([[[1.0, 4.0, 3.0, 2.0]]])

    indices = glm52_index_topk(
        scores,
        top_k=2_048,
        position_ids=torch.tensor([[3]]),
    )

    assert indices.shape == (1, 1, 4)
    assert set(indices[0, 0].tolist()) == {0, 1, 2, 3}


def test_sparse_attention_gathers_only_selected_causal_tokens() -> None:
    query = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]]]],
    )
    key_cache = torch.tensor(
        [[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]],
    )
    value_cache = torch.tensor(
        [[[[10.0]], [[20.0]], [[30.0]]]],
    )
    topk_indices = torch.tensor([[[0, 2], [0, 1]]], dtype=torch.int32)

    output = glm52_sparse_attention(
        query,
        key_cache,
        value_cache,
        topk_indices,
        position_ids=torch.tensor([[0, 1]]),
        scaling=1.0,
    )

    torch.testing.assert_close(output[0, 0], torch.tensor([[10.0]]))
    expected = (
        torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
        * torch.tensor([10.0, 20.0])
    ).sum()
    torch.testing.assert_close(output[0, 1], expected.reshape(1, 1))
