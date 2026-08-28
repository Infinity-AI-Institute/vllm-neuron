# `vllm_neuron.model.dsv4_flash` — DeepSeek-V4-Flash Neuron enablement (Round 3: MQA + routed-MoE landed)

**Status:** Round 3 — MQA attention block (`_MQABlock`) + routed-MoE
block (`_RoutedMoEBlock`) implemented and verified byte-clean against
real HF tensors.  NO NEFF compile fired yet.
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
| `neuron_wrapper.py::_SlidingOnlyAttentionBlock` | **Stub** | Round 2. Layers 0, 1, 40-42. |
| `neuron_wrapper.py::_CSABlock` | **Stub** | Round 2. Lightning Indexer + compressor with overlap. |
| `neuron_wrapper.py::_HCABlock` | **Stub** | Round 2. Compressor without overlap. |
| `neuron_wrapper.py::_HashMoEBlock` | **Stub** | Round 2. `tid2eid[input_ids]` frozen lookup. |
| `neuron_wrapper.py::_MoEBlock` | **Stub** | Round 2. Fork of GLM-5.3-Flash `_MoEBlock` with `sqrt(softplus(x))` scoring, 256×top-6. |
| `neuron_wrapper.py::NeuronDeepseekV4FlashForCausalLM` | **Stub** — `init_model` raises | Round 2 wires the block scaffold. |
| `checkpoint_convert.py::dequantize_block_fp8_ue8m0` | **Implemented** | UE8M0 exponent → `torch.ldexp` multiplier. Byte-clean per `smoke_round1_one_tensor.py`. |
| `checkpoint_convert.py::dequantize_block_fp4_ue8m0` | **Implemented** — byte-exact vs HF reference (see `tests/test_fp4_dequant_1tensor.py`, max_abs_error_bf16 = 0.0). |
| `checkpoint_convert.py::_convert_routed_moe_layer` | **Implemented** — 7/7 wrapper-tree keys, 256/256 experts, router bit-exact vs HF (see `tests/test_routed_moe_1layer.py`). |
| `checkpoint_convert.py::_convert_mqa_block` | **Implemented** — 8 wrapper-tree keys + sibling `attn_norm` from HF layer subtree; FP8-UE8M0 dequant on 5 weights, dense pass-through on `q_norm`/`kv_norm`/`attn_sink`. Fail-loud on missing weight or scale. |
| `checkpoint_convert.py::_convert_dsv4_checkpoint` | **Partial** — top-level tensors + routed-MoE per-layer + MQA-block per-layer available; per-attention-type composition and per-layer dispatch loop is Round 4. |
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

## First-fire blockers (mirrored from enablement draft §3)

1. HF checkpoint hydration (adapt Round-6 stream_shard).
2. FP4/UE8M0 dequant primitive. **DONE — Round 2, byte-exact.**
3. Wrapper block-class implementations:
   - `_MQABlock` — **DONE — Round 3, max_abs_error_bf16 = 0.0 vs HF ref on real layer-0 tensors.**
   - `_RoutedMoEBlock` — **DONE — Round 3 (routed-MoE, 256×top-6).**
   - `_SlidingOnlyAttentionBlock` — Round 4, thin wrapper over `_MQABlock` + sliding-window causal mask.
   - `_HCABlock` — Round 4, `_MQABlock` + `_HCACompressor` (no indexer, no overlap; simplest CSA/HCA path).
   - `_CSABlock` — Round 5, `_MQABlock` + `_CSACompressor` + Lightning Indexer (needs overlap-state aliasing).
   - `_HashMoEBlock` — Round 5, needs `input_ids` side channel through decoder forward.
4. Attention-sink support in the NxDI flash-attn kernel — open. `_MQABlock` uses the eager path (byte-exact vs HF); NKI flash-attn kernel wrap comes at NEFF fire.
5. Compressor overlap-state aliasing — Round 5 (`_CSACompressor`).
6. Hash-MoE bootstrap side-channel through the decoder forward — Round 5.

## Single next action

Land `_HCABlock` (or `_SlidingOnlyAttentionBlock` if simpler first) —
`_HCABlock` is the smallest attention-family composition that exercises
the compressor plumbing without the CSA-only Lightning Indexer.  It
reuses `_MQABlock.project_q`, `_MQABlock.project_kv`, and
`_MQABlock.attend_and_project` and adds a `_HCACompressor` that closes
non-overlapping `compress_rate_hca=128` windows and cats one compressed
KV entry per closed window onto the KV axis before `attend_and_project`.
No overlap state, no indexer, no input-id side channel — it is the
cleanest composability check for the block boundary `_MQABlock` just
exposed.

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
