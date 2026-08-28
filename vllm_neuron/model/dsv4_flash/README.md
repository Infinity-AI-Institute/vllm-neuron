# `vllm_neuron.model.dsv4_flash` — DeepSeek-V4-Flash Neuron enablement (Round 1 scaffold)

**Status:** Round 1 scaffold only — NO NEFF compile fired.
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
| `neuron_wrapper.py::_MQABlock` | **Stub** — raises NotImplementedError | Round 2. Shared K=V MQA + attention sink + partial RoPE + grouped output projection. |
| `neuron_wrapper.py::_SlidingOnlyAttentionBlock` | **Stub** | Round 2. Layers 0, 1, 40-42. |
| `neuron_wrapper.py::_CSABlock` | **Stub** | Round 2. Lightning Indexer + compressor with overlap. |
| `neuron_wrapper.py::_HCABlock` | **Stub** | Round 2. Compressor without overlap. |
| `neuron_wrapper.py::_HashMoEBlock` | **Stub** | Round 2. `tid2eid[input_ids]` frozen lookup. |
| `neuron_wrapper.py::_MoEBlock` | **Stub** | Round 2. Fork of GLM-5.3-Flash `_MoEBlock` with `sqrt(softplus(x))` scoring, 256×top-6. |
| `neuron_wrapper.py::NeuronDeepseekV4FlashForCausalLM` | **Stub** — `init_model` raises | Round 2 wires the block scaffold. |
| `checkpoint_convert.py::dequantize_block_fp8_ue8m0` | **Implemented** | UE8M0 exponent → `torch.ldexp` multiplier. Byte-clean per `smoke_round1_one_tensor.py`. |
| `checkpoint_convert.py::dequantize_block_fp4_ue8m0` | **Stub** | Round 2. Deferred pending packing-layout verification against a real routed-expert shard. |
| `checkpoint_convert.py::_convert_dsv4_checkpoint` | **Partial** — top-level tensors only | Per-layer conversion (per attention type × MLP type) is Round 2. |
| `stream_shard.py::stream_shard_dsv4_checkpoint` | **Stub** | Round 2. Sharding-rule mapping described in the docstring. |
| `registry.py::get_models` / `registry_hook.py::register_dsv4_flash` | **Implemented** | Binds HF architecture id `DeepseekV4ForCausalLM` to the wrapper class. |
| `smoke_round1_one_tensor.py` | **Implemented** | FP8-UE8M0 dequant vs hand golden. Synthetic mode passes offline; HF-shard mode drops into any FP8 e4m3 non-expert weight + its `.scale` companion. |
| `tests/` | **Empty** | Round 2 lands `test_config_load.py`, `test_factory_validation.py`, `test_dequant_ue8m0.py`, and (post-Round-2) per-block dispatch tests. |

## Verified

Nothing has been verified on-device yet.  The `smoke_round1_one_tensor.py`
harness executes offline; running it against a real HF shard is the
first verification step (does not require a Trn2 host).

## First-fire blockers (mirrored from enablement draft §3)

1. HF checkpoint hydration (adapt Round-6 stream_shard).
2. FP4/UE8M0 dequant primitive.
3. Wrapper block-class implementations (`_MQABlock`, `_CSABlock`, `_HCABlock`, `_SlidingOnlyAttentionBlock`, `_HashMoEBlock`).
4. Attention-sink support in the NxDI flash-attn kernel (open question — may require a NKI binding).
5. Compressor overlap-state aliasing.
6. Hash-MoE bootstrap side-channel through the decoder forward.

## Single next action

Land the FP4-UE8M0 arithmetic and its hand-golden smoke against a real
routed-expert shard.  Every downstream lane is unblocked once the
dequant math is byte-clean.

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
