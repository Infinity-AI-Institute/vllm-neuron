# GLM-5.3-Flash DSA indexer mapping fix (Round 7)

**Date:** 2026-08-28
**Branch:** `codex/glm53-flash-enablement`
**Scope:** wrapper-side rewrite (Python-only, no compile fired)
**Follows:** `NXDI-WRAPPER-ROUND6-STATUS-2026-08-28.md` §3 (recorded gap)
**Unblocks:** 10-token greedy exact gate on DSA layers once weights land

## TL;DR

Round 4 declared the DSA indexer's `q_proj` as a rank-3 param
`[pooled_index_heads, heads_per_rank * qk_head_dim, index_head_dim]` and its
`pool_weights` as a scalar `[index_kpool]`.  The HF checkpoint's actual
storage is different in a way that cannot be losslessly reformulated:

- **HF `indexer.wq_b.weight`** is `[n_heads * head_dim, q_lora_rank] =
  [4096, 1536]` — a **low-rank** projection off `q_lora` (post-Q_A +
  q_a_norm residual, before Q_B expansion).  Any Option-B reformulation
  through `Q_B` requires a right-inverse; `Q_B` has shape
  `[n_heads_main * qk_head_dim, q_lora_rank] = [16384, 1536]` and rank
  1536 < 16384 forecloses it.
- **HF `indexer.weights_proj.weight`** is `[n_heads, hidden_size]` — a
  per-token, per-head learned weight vector.  It cannot be reduced to a
  constant `[index_kpool]` pool weight tensor without discarding both the
  token axis and the head axis.

**Direction chosen: Option A** — reshape the wrapper's declared params to
match HF, and rewire the forward.  This is the only mathematically-exact
option; Option B is provably lossy.

Full paths (Windows local):

- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\neuron_wrapper.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\checkpoint_convert.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\stream_shard.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\tests\test_dsa_indexer_mapping.py`

## Direction of the fix

### Option A vs Option B — the derivation

**Golden CPU reference** (`harness-v2/staging/reference-sweep-20260826T2150Z/kernels/dsa_lightning_indexer.py`):

```
q_idx = einsum("bqf,hfd->bqhd", q_flat, indexer_q_proj)
        # q_flat = Q_B(q_lora) : [B, Q, num_heads * qk_head_dim]
        # indexer_q_proj       : [H_idx, num_heads * qk_head_dim, D_idx]
```

**HF `Glm5NextTextIndexer.forward`**
(`transformers/models/glm5_next/modeling_glm5_next.py:797`):

```
q = wq_b(q_resid).view(B, S, n_heads, head_dim)
    # wq_b.weight : [n_heads * head_dim, q_lora_rank]
    # q_resid     : q_lora latent, [B, S, q_lora_rank]
```

For these to be numerically equivalent one would need
`indexer_q_proj @ Q_B = wq_b` (in matrix form), i.e., some `M_golden`
such that `M_golden @ Q_B = M_hf`.  With `Q_B: [16384, 1536]` and rank
1536, there is no unique right-inverse; even in a best-fit sense, any
`M_golden` recovered by a least-squares solve loses information.  The
Round-4 rank-3 scaffold implicitly assumed this reformulation exists,
which it does not.

**Same story for `weights_proj`.**  HF stores
`weights_proj.weight = [n_heads, hidden_size]` and applies it *per token*
inside the forward.  The Round-4 wrapper had `pool_weights = [index_kpool]`
— a per-pool constant.  Collapsing a token-dependent per-head tensor to
a constant per-pool tensor is a lossy projection with no rebuild path.

**Round-6 gap:** the converter forwarded HF's tensors under the old
target names anyway; NxDI's `strict=False` loader silently accepted the
shape mismatch, leaving both `q_proj` and `pool_weights` populated with
the wrapper's `torch.empty(...)` sentinel.  DSA correctness would fail
at runtime with plausible-looking garbage.

### What changed in the wrapper

`_DSAIndexerBlock.__init__` now matches HF `Glm5NextTextIndexer` at the
param level:

| Param                          | Shape                              | Source of truth                    |
| ------------------------------ | ---------------------------------- | ---------------------------------- |
| `wq_b.weight`                  | `[n_heads * head_dim, q_lora_rank]`| HF `indexer.wq_b.weight`           |
| `wk.weight`                    | `[head_dim, hidden_size]`          | HF `indexer.wk.weight`             |
| `k_norm.weight`                | `[head_dim]`                       | HF `indexer.k_norm.weight`         |
| `k_norm.bias`                  | `[head_dim]`                       | HF `indexer.k_norm.bias`           |
| `weights_proj.weight`          | `[n_heads, hidden_size]`           | HF `indexer.weights_proj.weight`   |
| `index_kpool_compress_ape`     | `[index_kpool, head_dim]`          | HF `indexer.index_kpool_compress_ape` |
| `index_kpool_compress_gate`    | `[head_dim, hidden_size]`          | HF `indexer.index_kpool_compress_gate` |
| `cache_quant_multiplier` (buf) | `()` fp32                          | Round 4 (unchanged, FP8-KV guard)  |

Every projection uses `_NxdColumnParallelLinear(..., gather_output=True)`
so the state-dict key ends in `.weight` (matching HF) and the per-rank
compute is `[full-output-shard, all-gather]` — replicated behavior at
correctness time, sharded at storage time.  All indexer params
replicated across ranks; sparse selection is therefore bit-identical
across ranks by construction, no extra all-reduce needed.

`_DSAIndexerBlock.forward` now mirrors HF's math:

```
q      = wq_b(q_resid).view(B, L, n_heads, head_dim)
k      = k_norm(wk(hidden)).squeeze(-2)                  # [B, S, head_dim]  cached
gate   = F.linear(hidden, index_kpool_compress_gate)     # [B, S, head_dim]  cached
# pool over sequence: kpool consecutive positions, softmax-weighted
pool_keys = softmax(gate + kpool_compress_ape) @ k_grouped
scores    = relu( (q @ pool_keys.T) * (1/sqrt(head_dim)) )
weights   = weights_proj(hidden) * (n_heads ** -0.5)
index_scores = (weights.unsqueeze(-2) @ scores).squeeze(-2)
topk       = topk over pools, expanded back to raw token indices
```

`_NoPeMLABlock.project` was extended to return `q_latent` (the post-Q_A +
q_a_norm residual) as a 4th element, so `_DSABlock` can hand it to the
indexer without recomputing.  `_DSABlock.state_cache_specs` now returns
4 aliased tensors (was 3): `k_cache`, `v_cache`, `index_k_cache`,
`index_gate_cache`.  The `index_k_cache` shape changed to
`[B, S, head_dim]` (SINGLE-head, matching HF's `wk` output width).

### What changed in the converter

`_convert_dsa_layer` in `checkpoint_convert.py` now maps every HF indexer
tensor 1:1 into `indexer.<same-name>`.  The three-tensor Round-4 mapping
(`wk -> k_proj.weight`, `wq_b -> q_proj` [with reshape], `weights_proj
-> pool_weights`) is replaced by a straight-through carry of all seven
HF keys.  No reshape.  No unmapped-indexer bucket.

`stream_shard.py`'s per-rank emitter is updated symmetrically: `wq_b`,
`wk`, `weights_proj` sharded along dim=0 (`ColumnParallelLinear`
convention); `k_norm.{weight,bias}` and both compress params replicated.

## Numerical equivalence proof

`vllm_neuron/model/glm53_flash/tests/test_dsa_indexer_mapping.py` compares
`_DSAIndexerBlock.compute_index_scores` against an inline port of HF
`Glm5NextTextIndexer.forward` on identical synthetic input (batch=2,
seq_len=12, hidden=128, n_heads=8, head_dim=32, q_lora_rank=64,
index_kpool=4, index_topk=8, BF16 dtype).

**Result at both seeds:** `max_abs_error = 0.000e+00`
(bit-exact, not merely within tolerance).

```
$ python -m pytest vllm_neuron/model/glm53_flash/tests/test_dsa_indexer_mapping.py -v -s
vllm_neuron/model/glm53_flash/tests/test_dsa_indexer_mapping.py::test_wrapper_index_scores_match_hf_within_bf16_tolerance[0] [seed=0] wrapper vs HF index_scores max_abs_err = 0.000e+00
PASSED
vllm_neuron/model/glm53_flash/tests/test_dsa_indexer_mapping.py::test_wrapper_index_scores_match_hf_within_bf16_tolerance[17] [seed=17] wrapper vs HF index_scores max_abs_err = 0.000e+00
PASSED
vllm_neuron/model/glm53_flash/tests/test_dsa_indexer_mapping.py::test_wrapper_matches_hf_when_installed SKIPPED  # transformers.glm5_next not in this env
```

Every comparison runs through `degeneracy_guard.require_comparable` on
both sides before the max-abs-error check, so a NaN or all-zero output
cannot vacuously pass — the same guard Round 5's FP8 dequant test used.

`topk_indices` output also matches exactly (bit-equal `torch.equal`).

The second test (`test_wrapper_matches_hf_when_installed`) will
double-check against HF's own `Glm5NextTextIndexer` class when transformers
ships `glm5_next` — the uv archive carries it (transformers 5.14.x); the
standard Python 3.12 site-packages install in this env does not, so this
test currently skips.  The inline port in the primary test is the gate.

## Wrapper-tree key count

Layer 0 (KDA + dense MLP) is unchanged — the DSA indexer never lands
there.  DSA layers (indices 3, 7, ..., 43) now expose 8 indexer keys
(from 3):

```
DSA indexer state_dict keys (8):
  cache_quant_multiplier         : ()               fp32 buffer
  index_kpool_compress_ape       : (4, 32)          bf16
  index_kpool_compress_gate      : (32, 128)        bf16
  k_norm.bias                    : (32,)            bf16
  k_norm.weight                  : (32,)            bf16
  weights_proj.weight            : (8, 128)         bf16
  wk.weight                      : (32, 128)        bf16
  wq_b.weight                    : (256, 64)        bf16
```

DSA state cache spec goes from 3 to 4 slots:

```
_DSABlock.state_cache_specs (4 slots):
  k_cache             : (B, S, heads_per_rank, qk_nope_head_dim)
  v_cache             : (B, S, heads_per_rank, v_head_dim)
  index_k_cache       : (B, S, head_dim)                 # single-head now
  index_gate_cache    : (B, S, head_dim)                 # NEW in Round 7
```

Layer-0 wrapper-tree key count (27/27 from Round 5) is preserved by
inspection — layer 0 has no DSA path.

## Trade-offs and follow-ups

- **Compile impact.**  The state cache spec growing from 3 to 4 tensors
  per DSA layer means the aliased-parameter list also grows by 11 (one
  new tensor × 11 DSA layers).  The compile driver plumbing already
  keys off `state_cache_specs`, so no driver code needs to change.
  Compile-cache identity for DSA layers changes because the trace shape
  changes, but that is intended — the old cached NEFFs referred to the
  wrong math.
- **HBM.**  New `index_gate_cache` per DSA layer at `[B, S, head_dim]`.
  At S=1024, batch=1, head_dim=128, bf16 = 262 KiB per layer per rank ×
  11 layers = 2.9 MiB per rank total.  Negligible.
- **First-key = 0 assumption.**  HF's `Glm5NextTextIndexer` computes
  `first_key = argmax(valid_keys)` to support left-padded prompts; the
  Round-7 wrapper hard-codes `first_key = 0`.  For unpadded prompts —
  the campaign's smoke and 10-tok gate cases — this is exactly HF.  For
  a left-padded input, the wrapper's pool grouping starts at cache slot
  0 while HF's starts at the first real token.  Follow-up if the 10-tok
  gate needs left-padding support.
- **`weights_proj` fp32 keep.**  HF ships `_keep_in_fp32_modules =
  ["indexer.weights_proj"]`; this wrapper loads it in `torch_dtype`
  (bf16 by default) and up-casts to fp32 inside the forward via
  `.float()` before the matmul.  Mini-golden measured bit-exact agreement
  so the storage-dtype delta is invisible at the score boundary.

## Single next blocker

Real HF weights need to land through `stream_shard.py` at the new
per-layer key set; the streaming loader already emits every one of the
new HF keys 1:1 (see the `_convert_dsa_layer` and `stream_shard.py`
edits), but the actual per-rank output was not sharded end-to-end after
Round 6 hit disk contention.  Once the streaming loader can complete a
run against the real safetensors index — orthogonal to this fix, gated
on the Round-6 disk contention issue — the 10-token exact gate on DSA
layers unblocks.
