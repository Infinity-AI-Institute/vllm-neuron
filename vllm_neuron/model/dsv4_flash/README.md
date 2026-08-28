# `vllm_neuron.model.dsv4_flash` — DeepSeek-V4-Flash Neuron enablement (Round 8: PER-LAYER WIRING LANDED — READY FOR FIRST NEFF FIRE)

**Status:** Round 8 — `DeepseekV4FlashLayer` + per-layer dispatch loop
+ `input_ids` side channel + 84-entry state cache aggregation +
1024-key structural test all landed on top of Round 7's all-6-blocks
completion.  The full wrapper tree is now instantiable in
`_NeuronDeepseekV4FlashModel.init_model`; the `_convert_dsv4_checkpoint`
per-layer dispatch loop lands the full 1024-key state dict.  Wrapper
`forward` builds main + compress RoPE tables at model level and
threads `input_ids` through each layer as a REAL graph input (not an
attribute stash, per the module-level docstring above `_HashMoEBlock`).
NO NEFF compile fired yet — that's the immediate next lane on a
compile host (see `harness-v2/staging/reference-sweep-20260826T2150Z/lanes/deepseek-v4-flash/FIRST-NEFF-FIRE-PLAN-2026-08-28.md`).
**HF snapshot pinned:** `deepseek-ai/DeepSeek-V4-Flash-0731`
head SHA `7872f01b1d1fe23eabc4c98b48bffcef5a386062` (MIT-licensed).

The full enablement research + architecture-delta analysis lives at
`C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\deepseek-v4-flash\ENABLEMENT-DRAFT-2026-08-28.md`.

## Implementation-status matrix

| Component | State | Notes |
| --- | --- | --- |
| `config.py::DeepseekV4FlashInferenceConfig` | **Implemented** | Frozen constants from HF `config.json`; fail-loud validator. |
| `config.py::validate_ue8m0_scale` | **Implemented** | Refuses non-integer scales / out-of-range exponents. |
| `factory.py::DeepseekV4FlashForCausalLM` | **Implemented** | Fail-loud FP8-KV guard mirroring GLM-5.2 template; snapshot-SHA check. |
| `neuron_wrapper.py::build_neuron_config` | **Implemented** | Includes MoE `blockwise_matmul_config.use_shard_on_intermediate_dynamic_while` workaround for container `sha256:011d49c7…`. |
| `neuron_wrapper.py::DeepseekV4FlashNeuronInferenceConfig` | **Implemented** | Thin NxDI `InferenceConfig` subclass; frozen-field forwarding. |
| `neuron_wrapper.py::_MQABlock` | **Implemented (CPU-portable)** — max_abs_error_bf16 = 0.0 vs HF reference on real layer-0 tensors | Round 3. Shared K=V MQA + attention sink + partial RoPE + grouped output projection. 8 params match HF layer-0 subtree byte-for-byte. Verified via `tests/test_mqa_1tensor.py` against `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b` shard `model-00002-of-00048.safetensors`. |
| `neuron_wrapper.py::_SlidingOnlyAttentionBlock` | **Implemented (CPU-portable)** — max_abs_error_bf16 = 0.0 vs HF reference on real layer-0 tensors, sliding-window mask predicate exhaustive-match with HF | Round 6. Thin wrapper over `_MQABlock` + `build_sliding_window_causal_mask` (layers 0, 1 bootstrap). 8 params match HF layer-0 subtree byte-for-byte (identical shape to `_MQABlock.PARAM_KEYS` — no compressor, no indexer, no overlap state). Uses "main" RoPE (θ=10000), NOT the "compress" RoPE (θ=160000) that CSA/HCA share. Verified via `tests/test_sliding_only_1layer.py` against `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b` shard `model-00002-of-00048.safetensors`. Sliding-window mask arithmetic cross-checked two ways: (1) exhaustive predicate match against HF's `sliding_window_causal_mask_function` over the full 200×200 grid; (2) sufficient-condition real-forward cross-check where late queries (q≥128) MUST diverge from a full-causal forward (they do, max_diff=0.887) and early queries (q<128) MUST match (they do, max_diff=0.0). |
| `neuron_wrapper.py::build_sliding_window_causal_mask` | **Implemented (CPU-portable)** | Round 6. Additive-log-space sliding + causal mask helper (also the mask shared by CSA/HCA when they add the sliding-window branch alongside compressed KV in a future NEFF-side revision). Source-cited byte-for-byte against `masking_utils.py::sliding_window_causal_mask_function` (line 138: `and_masks(sliding_window_overlay, causal_mask_function)`) and its registration at `LAYER_TYPE_TO_MASK_CREATION_FUNCTION` (line 1478) for `"sliding_attention"` layers. |
| `neuron_wrapper.py::_CSABlock` | **Implemented (CPU-portable)** — max_abs_error_bf16 = 0.0 vs HF reference on real layer-2 tensors, multi-step state evolution PASS, Lightning Indexer top-K correctness PASS | Round 5. Composes `_MQABlock` hooks (project_q / project_kv / attend_and_project) + `_CSAOverlapCompressor` (Ca/Cb overlap-state, m=4, 2*head_dim projections) + `_LightningIndexerHead` (top-`index_topk=512` gating). 18 params match HF layer-2 subtree byte-for-byte. `state_cache_specs` declares 4 aliased pairs (`compressor_overlap_kv/gate` at head_dim=512, `indexer_overlap_kv/gate` at index_head_dim=128). Verified via `tests/test_csa_1layer.py` against `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b` shard `model-00004-of-00048.safetensors`. |
| `neuron_wrapper.py::_CSAOverlapCompressor` | **Implemented (CPU-portable)** | Round 5. Owns `wkv.weight` / `wgate.weight` / `ape` / `norm.weight` (2*head_dim projection widths). Ca of prior window → first half of new 2*compress_rate-wide window; Cb of current window → second half; window-0's first half stays zero-kv / -inf-gate on the first call (softmax weight 0), or receives the previous forward call's last-window Ca on subsequent calls. Source-cited byte-for-byte against `DeepseekV4CSACompressor` (transformers 5.15.1 `modeling_deepseek_v4.py:589-702`) + `DeepseekV4CSACache.update_overlap_state` (lines 286-300). |
| `neuron_wrapper.py::_LightningIndexerHead` | **Implemented (CPU-portable)** | Round 5. Owns its own inner `_CSAOverlapCompressor` at `index_head_dim=128` + `wq_b.weight` (FP8-UE8M0, `index_n_heads*index_head_dim × q_lora_rank`) + `weights_proj.weight` (`index_n_heads × hidden_size`). Scorer implements `∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)` with fp32 softmax scale = `index_head_dim**-0.5` and per-head weight scale = `index_n_heads**-0.5`; top-`min(index_topk=512, compressed_len)` selection with `-1` sentinel for causality-violating picks. Source-cited byte-for-byte against `DeepseekV4Indexer` (`modeling_deepseek_v4.py:462-586`) + `DeepseekV4IndexerScorer` (lines 446-459). |
| `neuron_wrapper.py::_HCABlock` | **Implemented (CPU-portable)** — max_abs_error_bf16 = 0.0 vs HF reference on real layer-3 tensors | Round 4. Composes `_MQABlock` hooks (project_q / project_kv / attend_and_project) + `_HCACompressor` (non-overlapping m'=128 pool, no indexer, no overlap state) + causal `block_bias` for compressed slots. 12 params match HF layer-3 subtree byte-for-byte. Verified via `tests/test_hca_1layer.py` against `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b` shard `model-00005-of-00048.safetensors`. |
| `neuron_wrapper.py::_HCACompressor` | **Implemented (CPU-portable)** | Round 4. Owns `wkv.weight` / `wgate.weight` / `ape` / `norm.weight`; softmax-gated per-window aggregation + compress-rope at window positions. Exposes `compress()` and `build_block_bias()` — the two pieces `_HCABlock` composes. Source-cited byte-for-byte against `DeepseekV4HCACompressor` (transformers 5.15.1 `modeling_deepseek_v4.py:362-443`). |
| `neuron_wrapper.py::_HashMoEBlock` | **Implemented (CPU-portable)** — max_abs_error_bf16 = 0.0 vs HF reference forward on synthetic + real-router weights; input-ids side channel bit-exact | Round 7. 6th and final DSv4-Flash block class. 7 wrapper-tree keys (same as `_RoutedMoEBlock` with `tid2eid` [vocab_size, num_experts_per_tok] int32 replacing `e_score_correction_bias`). Composes: `router` (fp32 `nn.Linear`) + `tid2eid` (frozen `nn.Parameter` requires_grad=False, dtype=int32) + `_HashMoESharedExpert` + `_HashMoEStackedExperts` (per-expert Python dispatch loop for CPU byte-cleanness; Round-8 NxDI subclass will swap in `_NxdExpertMLPs`). Forward signature `forward(hidden_states, input_ids)` — see class docstring for the `input_ids` side-channel plumbing decision (Round-8 NxDI wire-up extends decoder-layer forward to accept `input_ids` as a real graph input, NOT an attribute stash which would not survive `torch.export`/`torch_xla` lowering). Verified via `tests/test_hash_moe_1layer.py` against `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b` shard `model-00002-of-00048.safetensors` (router.weight + tid2eid pulled via HTTP-Range slice, ~5 MiB after dequant — no 3.6 GiB shard download). Side-channel gate proves the routing DIFFERENTIATES by input_ids: flipping one token's id changes ONLY that token's output while leaving other tokens byte-identical. |
| `neuron_wrapper.py::_HashMoESharedExpert` | **Implemented (CPU-portable)** | Round 7. Owns `gate_proj.weight` / `up_proj.weight` / `down_proj.weight` — three `nn.Linear` projections with the same on-disk spelling as `_MoESharedExpert` so state-dict is portable across the CPU-portable and NxDI paths. fp32 SwiGLU + ±swiglu_limit clamp per HF `Expert.forward` (`inference/model.py:601-611`). |
| `neuron_wrapper.py::_HashMoEStackedExperts` | **Implemented (CPU-portable)** | Round 7. Owns two stacked-expert tensors via `mlp_op.gate_up_proj.weight` `[E, hidden, 2*inter]` and `mlp_op.down_proj.weight` `[E, inter, hidden]` — same layout as `_RoutedMoEBlock`'s NxDI `ExpertMLPs` bearing, but wrapped in a CPU-portable Python per-expert dispatch loop (bit-exact against HF `MoE.forward` @ hash-MoE, `inference/model.py:634-649`). Round-8 NxDI subclass will rebind the two `mlp_op.<proj>_proj.weight` leaves onto `_NxdExpertMLPs` without renaming a single key. |
| `neuron_wrapper.py::_MoEBlock` | **Implemented via `_RoutedMoEBlock`** | Round 3. `sqrt(softplus(x))` scoring, 256×top-6, NxDI blockwise ExpertMLPs + separate shared branch, partial-sum all-reduce discipline. |
| `neuron_wrapper.py::NeuronDeepseekV4FlashForCausalLM` | **Implemented (Round 8)** — full per-layer dispatch, 1024-key wrapper tree, 84-entry state cache, `input_ids` side channel threaded through decoder-layer forward. Forward composes `embed_tokens + 43 x DeepseekV4FlashLayer + final_norm + lm_head`, builds main + compress RoPE tables at model level, feeds each layer the RoPE and mask tables its block class consumes. NO NEFF fired yet — that's the immediate next lane. |
| `neuron_wrapper.py::DeepseekV4FlashLayer` | **Implemented (Round 8)** — per-layer schedule dispatch (attn via `_SlidingOnlyAttentionBlock` / `_CSABlock` / `_HCABlock` based on `layer_types[i]`; mlp via `_HashMoEBlock` / `_RoutedMoEBlock` based on `mlp_layer_types[i]`). Owns `attn_norm.weight` + `ffn_norm.weight` at the layer level. `forward(hidden_states, position_ids, caches, input_ids, ...)` runs pre-attn norm → dispatched attn → residual → pre-mlp norm → dispatched mlp → residual. Hash-MoE MLPs read `input_ids` as second positional arg; routed-MoE MLPs ignore it. |
| `checkpoint_convert.py::dequantize_block_fp8_ue8m0` | **Implemented** | UE8M0 exponent → `torch.ldexp` multiplier. Byte-clean per `smoke_round1_one_tensor.py`. |
| `checkpoint_convert.py::dequantize_block_fp4_ue8m0` | **Implemented** — byte-exact vs HF reference (see `tests/test_fp4_dequant_1tensor.py`, max_abs_error_bf16 = 0.0). |
| `checkpoint_convert.py::_convert_routed_moe_layer` | **Implemented** — 7/7 wrapper-tree keys, 256/256 experts, router bit-exact vs HF (see `tests/test_routed_moe_1layer.py`). |
| `checkpoint_convert.py::_convert_mqa_block` | **Implemented** — 8 wrapper-tree keys + sibling `attn_norm` from HF layer subtree; FP8-UE8M0 dequant on 5 weights, dense pass-through on `q_norm`/`kv_norm`/`attn_sink`. Fail-loud on missing weight or scale. |
| `checkpoint_convert.py::_convert_hca_block` | **Implemented** — 12 wrapper-tree keys under `layers.<i>.attn.*` (8 MQA + 4 compressor) + sibling `attn_norm`. Layer-type guard refuses non-HCA layer indices. Verified via `tests/test_hca_1layer.py` on layer 3. |
| `checkpoint_convert.py::_convert_csa_block` | **Implemented** — 18 wrapper-tree keys under `layers.<i>.attn.*` (8 MQA + 4 CSA compressor + 4 indexer inner-compressor + 2 indexer projection [`wq_b` FP8-UE8M0 + `weights_proj` dense]) + sibling `attn_norm`. Layer-type guard refuses non-CSA layer indices. Verified via `tests/test_csa_1layer.py` on layer 2. |
| `checkpoint_convert.py::_convert_sliding_only_block` | **Implemented** — 8 wrapper-tree keys under `layers.<i>.attn.*` (same 8 MQA params — no compressor, no indexer) + sibling `attn_norm` = 9 tensors per layer. Delegates to `_convert_mqa_block` verbatim (identical on-disk parameter set). Layer-type guard refuses non-sliding layer indices. Verified via `tests/test_sliding_only_1layer.py` on layer 0. |
| `checkpoint_convert.py::_convert_hash_moe_block` | **Implemented** — 7 wrapper-tree keys under `layers.<i>.mlp.*` (`router.weight` fp32, `tid2eid` int32 [vocab_size, num_experts_per_tok], 3 shared_expert projections dequanted from FP8-UE8M0 block (128, 128), 2 stacked/fused expert_mlps tensors dequanted from FP4-UE8M0 block (1, 32)). Layer-type guard refuses non-hash-MoE layer indices. Refuses a checkpoint that carries `ffn.gate.bias` at a hash-MoE layer (schedule drift). Refuses an out-of-range `tid2eid`. Enforces int32 dtype on `tid2eid` for round-trip fidelity with HF. Verified via `tests/test_hash_moe_1layer.py` on layer 0. |
| `checkpoint_convert.py::_convert_dsv4_checkpoint` | **Implemented (Round 8)** — per-layer dispatch loop landed. Dispatches attention via `_convert_{sliding_only,csa,hca}_block` based on `layer_types[i]`; dispatches MLP via `_convert_{hash_moe,routed_moe}_block` based on `mlp_layer_types[i]`. Emits `ffn_norm.weight` sibling per layer alongside the `attn_norm.weight` from the attention converter. Sanity-checks converted key count = 1024 (full-shape). Records per-layer schedule report at `_conversion_report.per_layer_report`. |
| `stream_shard.py::stream_shard_dsv4_checkpoint` | **Implemented (Round 8)** | Mmap/layer-at-a-time HF safetensors conversion, TP=32 row/fused-axis slicing, replicated `tid2eid`/CSA-indexer state, per-rank safetensors output, and a queue-ready CLI. |
| `registry.py::get_models` / `registry_hook.py::register_dsv4_flash` | **Implemented** | Binds HF architecture id `DeepseekV4ForCausalLM` to the wrapper class. |
| `smoke_round1_one_tensor.py` | **Implemented** | FP8-UE8M0 dequant vs hand golden. Synthetic mode passes offline; HF-shard mode drops into any FP8 e4m3 non-expert weight + its `.scale` companion. |
| `tests/` | Round-8 adds `tests/test_full_wrapper_tree.py` — 9 structural tests covering schedule counts (2 sliding, 21 CSA, 20 HCA, 3 hash-MoE, 40 routed-MoE), PARAM_KEYS declaration on every block class, wrapper-tree total = 1024 keys, state cache = 84 aliased pairs, no degenerate output. Runs on CPU without NxDI. Prior gates continue: Round-7's 13/13 hash-MoE, Round-6's 8/8 sliding-only, Round-5's 7/7 CSA, Round-4's 6/6 HCA, Round-3's 5/5 MQA + 7/7 routed-MoE, Round-2's 8/8 FP4 dequant. |

## Verified

Nothing has been verified on-device yet.  Off-device CPU verification
against real HF tensors (no Trn2 host needed):

- **FP4-UE8M0 dequant** — `tests/test_fp4_dequant_1tensor.py`,
  max_abs_error_bf16 = 0.0 against `layers.3.ffn.experts.0.w2` from HF
  snapshot 7872f01b.  8/8 tests pass (byte-clean, sanity + edge cases).
- **Routed-MoE per-layer converter** — `tests/test_routed_moe_1layer.py`,
  7/7 wrapper-tree keys populated, 256/256 experts dequant, router bit-
  exact vs HF sqrt(softplus) reference.
- **MQA block forward** — `tests/test_mqa_1tensor.py`,
  **max_abs_error_bf16 = 0.0** against a hand-transcribed
  `DeepseekV4Attention.forward` reference on the real
  `layers.0.attn.*` tensors from HF snapshot 7872f01b, shard
  `model-00002-of-00048.safetensors`.  5/5 tests pass (wrapper-tree key
  set, real-HF byte-clean, synthetic shape, config guard, standalone
  RoPE helper).
- **HCA block forward** — `tests/test_hca_1layer.py`,
  **max_abs_error_bf16 = 0.0** against a hand-transcribed
  `DeepseekV4Attention.forward` + `DeepseekV4HCACompressor.forward`
  reference on the real `layers.3.attn.*` + `layers.3.attn.compressor.*`
  tensors from HF snapshot 7872f01b, shard
  `model-00005-of-00048.safetensors`.  6/6 tests pass (schedule
  confirmation, 12-param wrapper tree, layer-type guard, compress-rate
  guard, synthetic shape gate, real-HF byte-clean).  Compressor emits
  exactly `S // 128` compressed entries (2 for `S=256`, verified).
- **CSA block forward** — `tests/test_csa_1layer.py`,
  **max_abs_error_bf16 = 0.0** (first-step) and
  **max_abs_error_bf16 = 0.0** (multi-step state evolution for both
  outer CSA compressor and indexer inner compressor) against a
  hand-transcribed `DeepseekV4Attention.forward` +
  `DeepseekV4CSACompressor.forward` + `DeepseekV4Indexer.forward` +
  `DeepseekV4IndexerScorer.forward` reference on the real
  `layers.2.attn.*` + `layers.2.attn.compressor.*` +
  `layers.2.attn.indexer.*` tensors from HF snapshot 7872f01b, shard
  `model-00004-of-00048.safetensors`.  7/7 tests pass (schedule
  confirmation, 18-param wrapper tree, layer-type guard,
  state_cache_specs shape/dtype, synthetic shape gate, real-HF
  byte-clean forward with Lightning Indexer top-K bit-equal to reference
  + causality guard, multi-step state evolution byte-clean).  Compressor
  emits exactly `S // 4` compressed entries; indexer emits `min(512,
  compressed_len)` top-K indices per query, `-1` sentinel on invalid.
- **Sliding-only block forward** — `tests/test_sliding_only_1layer.py`,
  **max_abs_error_bf16 = 0.0** against a hand-transcribed
  `DeepseekV4Attention.forward` (sliding branch) +
  `sliding_window_causal_mask_function` predicate reference on the real
  `layers.0.attn.*` tensors from HF snapshot 7872f01b, shard
  `model-00002-of-00048.safetensors`.  8/8 tests pass (schedule
  confirmation, 8-param wrapper tree, block layer-type guard, converter
  layer-type guard, exhaustive sliding+causal mask predicate match
  against HF over the 200×200 grid, synthetic shape gate, real-HF
  byte-clean forward at S=200, sufficient-condition cross-check that
  the mask actually reaches the attention math — early queries q<128
  agree byte-for-byte with a full-causal forward (max_diff=0.0) and
  late queries q≥128 diverge (max_diff=0.887) proving the window
  clips past KV).

## First-fire blockers (mirrored from enablement draft §3)

1. HF checkpoint hydration (adapt Round-6 stream_shard).
2. FP4/UE8M0 dequant primitive. **DONE — Round 2, byte-exact.**
3. Wrapper block-class implementations:
   - `_MQABlock` — **DONE — Round 3, max_abs_error_bf16 = 0.0 vs HF ref on real layer-0 tensors.**
   - `_RoutedMoEBlock` — **DONE — Round 3 (routed-MoE, 256×top-6).**
   - `_HCABlock` + `_HCACompressor` — **DONE — Round 4, max_abs_error_bf16 = 0.0 vs HF ref on real layer-3 tensors.** Composes `_MQABlock` hooks; no overlap, no indexer, no input-id side channel.
   - `_CSABlock` + `_CSAOverlapCompressor` + `_LightningIndexerHead` — **DONE — Round 5, max_abs_error_bf16 = 0.0 vs HF ref on real layer-2 tensors, state-aliasing PASS, Lightning Indexer top-K PASS.** Composes `_MQABlock` hooks; adds Ca/Cb overlap state (aliased in graph via `state_cache_specs`, mirror of GLM-5.3-Flash KDA `_KDABlock.state_cache_specs`) and Lightning Indexer top-K gating.
   - `_SlidingOnlyAttentionBlock` — **DONE — Round 6, max_abs_error_bf16 = 0.0 vs HF ref on real layer-0 tensors, sliding-window mask predicate exhaustive-match, mask-actually-applies cross-check PASS.** Composes `_MQABlock` hooks + `build_sliding_window_causal_mask`; uses main RoPE (θ=10000).
   - `_HashMoEBlock` — Round 7 (last remaining block class), needs `input_ids` side channel through decoder forward (3 bootstrap MoE layers).
4. Attention-sink support in the NxDI flash-attn kernel — open. `_MQABlock` uses the eager path (byte-exact vs HF); NKI flash-attn kernel wrap comes at NEFF fire.
5. Compressor overlap-state aliasing — **DONE — Round 5.  Wired functionally via `_CSABlock.forward(overlap_state=...)`; NEFF-side `input_output_aliases` wiring lands with Round 6 model wrapper.**
6. Hash-MoE bootstrap side-channel through the decoder forward — Round 7 (last remaining block).

## Single next action

**First NEFF fire** on a Trn2 compile host — TP=32 LNC=2 seq=4096 B=1.
Contract, hydration status, expected wall (30-60 min), post-compile
emit-verification checklist, and correctness gate (token-0 cossim ≥
0.99 vs banked HF reference logits) are all documented at
`C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\deepseek-v4-flash\FIRST-NEFF-FIRE-PLAN-2026-08-28.md`.

Pre-fire prerequisites still pending (both are CPU-only, run on
r7i-class hosts):

  1. Bank HF reference logits at token-0 for the cossim gate.
  2. Wire `stream_shard_dsv4_checkpoint` per GLM-5.3-Flash Round 6's
     streaming-shard build pattern (334 GB bf16 fully-materialised
     checkpoint won't fit on rank 0; must stream per rank, one layer
     at a time).

## Files

- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\__init__.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\config.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\factory.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\neuron_wrapper.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\checkpoint_convert.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\stream_shard.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\registry.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\registry_hook.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\smoke_round1_one_tensor.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\tests\__init__.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\tests\test_fp4_dequant_1tensor.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\tests\test_routed_moe_1layer.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\tests\test_mqa_1tensor.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\tests\test_hca_1layer.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\tests\test_csa_1layer.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\tests\test_sliding_only_1layer.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\tests\test_full_wrapper_tree.py`
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\deepseek-v4-flash\FIRST-NEFF-FIRE-PLAN-2026-08-28.md`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\kernel_dispatch.py` — **NEW 2026-08-28** (DSA v2 slug advertiser, `resolve_dsa_impl_slug`, `get_emitted_kernel_slugs`)
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\dsv4_flash\tests\test_dsa_v2_wrapper_integration.py` — **NEW 2026-08-28** (13 tests, 13/13 PASS via direct-invoke)

## Kernel-slug env-flip (2026-08-28)

The DSv4-Flash CSA lane now advertises the DSA kernel slug via
`DSA_KERNEL_IMPL` env-flip. Wire-up commit `2fea306` on branch
`worker/dsv4-flash-enablement-scaffold-20260828`:

| Env-var value | Advertised slug | Notes |
|---|---|---|
| unset / empty / `cpu` / `cpu_golden` / `v0` | `nki_v0_reference_lightning_indexer` | default; existing behaviour |
| `nki_v2` / `v2` / `dsa_sparse_attention.nki_v2` | `dsa_sparse_attention.nki_v2` | v2 NKI device kernel (baremetal COMPILE PASS on r7i, 29.91s -- see `DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md` §3C) |
| any other value | -- | `ValueError` (rejects typos so a stale-NEFF class of bug can't land) |

Scope on DSv4-Flash: the env flip is a SLUG-ONLY advertisement -- DSv4
does not structurally call `dsa_sparse_attention_forward` (the indexer's
top-k is consumed by a `block_bias` mask feeding a dense softmax on the
extended KV axis). The wire-up:

1. Adds `_emitted_dsa_slug` to `_LightningIndexerHead.__init__` and
   `_CSABlock.__init__` (bound to the env-resolved slug at construction
   time; per-layer view for audit).
2. Re-exports `get_emitted_kernel_slugs`, `DSA_CPU_GOLDEN_SLUG`,
   `DSA_NKI_V2_SLUG` at the `neuron_wrapper` module top-level (single
   import surface for the compile driver, parallel to the GLM-5.3-Flash
   lane).
3. When a future revision inlines a fused sparse-gather kernel, the env
   flip is already the entrypoint.

DSv4 has NO KDA layers (attention families are MQA + CSA + HCA +
SlidingOnly), so there is NO KDA dispatch. `kernel_dispatch.py` deliberately
does not export a KDA resolver -- the integration test
(`test_dsv4_kernel_dispatch_has_no_kda_dispatch`) asserts against
copy-paste drift.

Parallel wire-up on GLM-5.3-Flash: `codex/glm53-flash-enablement@a3d56f6`.
Cross-model integrated-status receipt at
`C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\glm-5-2-5-3\KERNELS-INTEGRATED-2026-08-28.md`.
