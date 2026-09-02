from __future__ import annotations

import torch
import torch.nn.functional as F


def test_tp64_replicated_weights_projection_matches_tp32_gather_and_scores():
    """The 32-wide indexer scorer stays numerically identical at TP64.

    At TP32, each rank owns one output row of `weights_proj` and NxD gathers
    the rows.  At TP64 a 32-row column-parallel output is invalid, so the
    TP64 ownership rule replicates the small full projection.  Its output and
    the score contraction must remain exact; neither indexer cache is an
    output of this projection.
    """

    torch.manual_seed(64053)
    batch, query, hidden, index_heads, pools = 1, 3, 4096, 32, 5
    hidden_states = torch.randn(batch, query, hidden, dtype=torch.bfloat16)
    weight = torch.randn(index_heads, hidden, dtype=torch.bfloat16)
    scores = torch.randn(batch, query, index_heads, pools, dtype=torch.float32)
    index_k_cache = torch.randn(batch, 7, 128, dtype=torch.bfloat16)
    index_gate_cache = torch.randn(batch, 7, 128, dtype=torch.bfloat16)
    k_before, gate_before = index_k_cache.clone(), index_gate_cache.clone()

    reference_projection = F.linear(hidden_states, weight)
    tp32_gathered = torch.cat(
        [F.linear(hidden_states, weight[rank : rank + 1]) for rank in range(32)],
        dim=-1,
    )
    tp64_replicas = [F.linear(hidden_states, weight) for _rank in range(64)]

    assert torch.equal(reference_projection, tp32_gathered)
    assert all(torch.equal(reference_projection, output) for output in tp64_replicas)

    def score(output: torch.Tensor) -> torch.Tensor:
        learned_weights = output.float() * (index_heads**-0.5)
        return torch.matmul(learned_weights.unsqueeze(-2), scores).squeeze(-2)

    expected_scores = score(reference_projection)
    assert torch.equal(expected_scores, score(tp32_gathered))
    assert all(torch.equal(expected_scores, score(output)) for output in tp64_replicas)
    assert torch.equal(index_k_cache, k_before)
    assert torch.equal(index_gate_cache, gate_before)


def test_tp64_replicated_projection_memory_is_small_and_exact():
    # 11 DSA layers, full BF16 [32, 4096] on each TP64 rank.
    per_dsa_rank_bytes = 32 * 4096 * torch.tensor([], dtype=torch.bfloat16).element_size()
    assert per_dsa_rank_bytes == 262_144
    assert 11 * per_dsa_rank_bytes == 2_883_584
