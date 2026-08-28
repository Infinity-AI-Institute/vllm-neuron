# `vllm_neuron.model.dsv4_flash` — DeepSeek-V4-Flash Neuron enablement (Round 4: HCA composability landed)

**Status:** Round 4 — HCA attention block (`_HCABlock`) +
`_HCACompressor` implemented and verified byte-clean against real HF
tensors on the first-HCA layer (layer 3), on top of Round 3's MQA
attention block (`_MQABlock`) and routed-MoE block (`_RoutedMoEBlock`).
NO NEFF compile fired yet.
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
| `neuron_wrapper.py::_SlidingOnlyAttentionBlock` | **Stub** | Round 5. Layers 0, 1 (frozen schedule; layers 40-42 aren't sliding in this snapshot — the enablement doc's original claim was wrong). |
| `neuron_wrapper.py::_CSABlock` | **Stub** | Round 5. Lightning Indexer + compressor with overlap. |
| `neuron_wrapper.py::_HCABlock` | **Implemented (CPU-portable)** — max_abs_error_bf16 = 0.0 vs HF reference on real layer-3 tensors | Round 4. Composes `_MQABlock` hooks (project_q / project_kv / attend_and_project) + `_HCACompressor` (non-overlapping m'=128 pool, no indexer, no overlap state) + causal `block_bias` for compressed slots. 12 params match HF layer-3 subtree byte-for-byte. Verified via `tests/test_hca_1layer.py` against `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b` shard `model-00005-of-00048.safetensors`. |
| `neuron_wrapper.py::_HCACompressor` | **Implemented (CPU-portable)** | Round 4. Owns `wkv.weight` / `wgate.weight` / `ape` / `norm.weight`; softmax-gated per-window aggregation + compress-rope at window positions. Exposes `compress()` and `build_block_bias()` — the two pieces `_HCABlock` composes. Source-cited byte-for-byte against `DeepseekV4HCACompressor` (transformers 5.15.1 `modeling_deepseek_v4.py:362-443`). |
| `neuron_wrapper.py::_HashMoEBlock` | **Stub** | Round 2. `tid2eid[input_ids]` frozen lookup. |
| `neuron_wrapper.py::_MoEBlock` | **Stub** | Round 2. Fork of GLM-5.3-Flash `_MoEBlock` with `sqrt(softplus(x))` scoring, 256×top-6. |
| `neuron_wrapper.py::NeuronDeepseekV4FlashForCausalLM` | **Stub** — `init_model` raises | Round 2 wires the block scaffold. |
| `checkpoint_convert.py::dequantize_block_fp8_ue8m0` | **Implemented** | UE8M0 exponent → `torch.ldexp` multiplier. Byte-clean per `smoke_round1_one_tensor.py`. |
| `checkpoint_convert.py::dequantize_block_fp4_ue8m0` | **Implemented** — byte-exact vs HF reference (see `tests/test_fp4_dequant_1tensor.py`, max_abs_error_bf16 = 0.0). |
| `checkpoint_convert.py::_convert_routed_moe_layer` | **Implemented** — 7/7 wrapper-tree keys, 256/256 experts, router bit-exact vs HF (see `tests/test_routed_moe_1layer.py`). |
| `checkpoint_convert.py::_convert_mqa_block` | **Implemented** — 8 wrapper-tree keys + sibling `attn_norm` from HF layer subtree; FP8-UE8M0 dequant on 5 weights, dense pass-through on `q_norm`/`kv_norm`/`attn_sink`. Fail-loud on missing weight or scale. |
| `checkpoint_convert.py::_convert_hca_block` | **Implemented** — 12 wrapper-tree keys under `layers.<i>.attn.*` (8 MQA + 4 compressor) + sibling `attn_norm`. Layer-type guard refuses non-HCA layer indices. Verified via `tests/test_hca_1layer.py` on layer 3. |
| `checkpoint_convert.py::_convert_dsv4_checkpoint` | **Partial** — top-level tensors + routed-MoE per-layer + MQA-block per-layer + HCA-block per-layer available; per-attention-type composition and per-layer dispatch loop is Round 5. |
| `stream_shard.py::stream_shard_dsv4_checkpoint` | **Stub** | Round 2. Sharding-rule mapping described in the docstring. |
| `registry.py::get_models` / `registry_hook.py::register_dsv4_flash` | **Implemented** | Binds HF architecture id `DeepseekV4ForCausalLM` to the wrapper class. |
| `smoke_round1_one_tensor.py` | **Implemented** | FP8-UE8M0 dequant vs hand golden. Synthetic mode passes offline; HF-shard mode drops into any FP8 e4m3 non-expert weight + its `.scale` companion. |
| `tests/` | **Empty** | Round 2 lands `test_config_load.py`, `test_factory_validation.py`, `test_dequant_ue8m0.py`, and (post-Round-2) per-block dispatch tests. |

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

## First-fire blockers (mirrored from enablement draft §3)

1. HF checkpoint hydration (adapt Round-6 stream_shard).
2. FP4/UE8M0 dequant primitive. **DONE — Round 2, byte-exact.**
3. Wrapper block-class implementations:
   - `_MQABlock` — **DONE — Round 3, max_abs_error_bf16 = 0.0 vs HF ref on real layer-0 tensors.**
   - `_RoutedMoEBlock` — **DONE — Round 3 (routed-MoE, 256×top-6).**
   - `_HCABlock` + `_HCACompressor` — **DONE — Round 4, max_abs_error_bf16 = 0.0 vs HF ref on real layer-3 tensors.** Composes `_MQABlock` hooks; no overlap, no indexer, no input-id side channel.
   - `_SlidingOnlyAttentionBlock` — Round 5, thin wrapper over `_MQABlock` + sliding-window causal mask (layers 0-1).
   - `_CSABlock` — Round 5, `_MQABlock` + `_CSACompressor` + Lightning Indexer (needs overlap-state aliasing).
   - `_HashMoEBlock` — Round 5, needs `input_ids` side channel through decoder forward (3 bootstrap MoE layers).
4. Attention-sink support in the NxDI flash-attn kernel — open. `_MQABlock` uses the eager path (byte-exact vs HF); NKI flash-attn kernel wrap comes at NEFF fire.
5. Compressor overlap-state aliasing — Round 5 (`_CSACompressor`).
6. Hash-MoE bootstrap side-channel through the decoder forward — Round 5.

## Single next action

Pick the next attention/MLP block to land — each unlocks a different
piece of the per-layer dispatch loop `_convert_dsv4_checkpoint` will
need:

- **`_CSABlock`** — the last remaining attention family (compressed-
  sparse attention).  Adds two new mechanisms on top of `_HCABlock`:
  the CSA compressor's per-name overlap-state aliasing (Ca/Cb window
  scheme, `2 × head_dim` split) and the Lightning Indexer scoring head
  (paper §2.3.1 eq. 13-17) that gates the top-`index_topk=512`
  compressed entries per query.  Same `_MQABlock` hook boundary as
  `_HCABlock`.
- **`_SlidingOnlyAttentionBlock`** — thin wrapper over `_MQABlock` +
  sliding-window causal mask (layers 0, 1 only).  Simplest block that
  remains; small marginal value beyond the sliding-window mask math
  that `_CSABlock` also needs.
- **`_HashMoEBlock`** — first 3 MLP layers use the frozen
  `tid2eid[input_ids]` bootstrap lookup.  Needs the `input_ids` side
  channel threaded through the decoder forward — a new plumbing
  concern independent of the attention stack.

The pragmatic order: `_CSABlock` next (largest architectural unknown
remaining; blocks the full per-layer dispatch), then
`_SlidingOnlyAttentionBlock` (bootstrap layers 0-1 attention), then
`_HashMoEBlock` (bootstrap layers 0-2 MLP).  All three can then feed
the `_convert_dsv4_checkpoint` per-layer dispatch loop that Round 5
opens.

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
