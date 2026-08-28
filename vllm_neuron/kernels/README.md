# NKI kernels

Five NKI kernel implementations for Trainium2 with CPU reference implementations, plus two Python-side mitigations, plus tests and status docs. Every NKI kernel body is a physical file under `_kernel_bodies/` that the shim loads by file-import so `inspect.getsource` can walk it at compile time (see `EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md`). Every CPU reference matches the vendor or model reference cited in its STATUS doc within the tolerance that doc records. On-device runs are pending the next Trn2 capacity window.

## What lives here

| File | What it does | Tests | Which models use it |
|---|---|---|---|
| `dsa_lightning_indexer.py` | CPU reference for DSA sparse-attention prefilter (top-k gather + attention over the selected keys). LSE base is natural log. | `test_dsa_lightning_indexer_correctness.py`, `test_dsa_lightning_indexer_speed.py`, `test_dsa_lse_accumulator.py` | DeepSeek-V4, GLM-5.2, GLM-5.3-Flash |
| `dsa_lightning_indexer_nki_v1.py` | NKI kernel for the same op. Body at `_kernel_bodies/dsa_lightning_indexer_nki_v1_body.py`. Falls back to the CPU reference on hosts without `neuronxcc.nki`. | `test_dsa_lightning_indexer_nki_v1_smoke.py` | same as above, once fired |
| `kda_state.py` | CPU reference for KDA linear-attention state, v1. Slug marks it as int8 study; missing the KDA per-channel gate. Kept for the documented gap only. | `test_kda_state_correctness.py`, `test_kda_state_speed.py` | (not for serving) |
| `kda_state_v2.py` | CPU reference for KDA linear-attention state, v2. Matches FLA v0.5.2 (KDA gate + Q/K L2-norm + Q scale + sigmoid-beta), bf16 state. | `test_kda_state_v2_correctness.py`, `test_kda_state_v2_speed.py` | Kimi-K3, GLM-5.3-Flash |
| `kda_state_nki_v2.py` | NKI kernel for KDA v2. Body at `_kernel_bodies/kda_state_nki_v2_body.py`. | `test_kda_state_v2_smoke.py`, `test_kda_state_nki_v2_smoke.py` | same as above, once fired |
| `kda_state_int8_study.py` | Re-export of v1 marked `SERVING_STATUS = "NOT_SERVED"` behind an ack guard. HBM-bandwidth study only. | `test_kda_state_correctness.py` | none |
| `dma_coalescing_transform.py` | CPU-side descriptor coalescing analyzer + KV-slab planner (Paths B and C from the DMA scaffold). | `test_dma_coalescing_smoke.py` | Qwen3.5-4B, Gemma-4, GLM-5.2 index-attn |
| `dma_coalescing_nki_v1.py` | NKI kernel for the K-way coalesced gather (Path A). Body at `_kernel_bodies/dma_coalescing_nki_v1_body.py`. | `test_dma_coalescing_nki_v1_smoke.py` | same as above, once fired |
| `moe_dispatch.py` | Fused router + expert-combine dispatcher plus a fail-loud fallback ladder. Sets `use_shard_on_intermediate_dynamic_while=True` to work around the missing `_call_shard_hidden_kernel` in container `sha256:011d49c7`. | `test_moe_dispatch_correctness.py` | Qwen3-30B-A3B, GPT-OSS-20B, Gemma-4-26B-A4B |
| `glm52_indexer_fp8_scale_fix.py` | Rescales `cache_quant_multiplier` by `240/448` so on-disk FP8 scale matches the Trn2 kernel's clamp, plus a load-time assertion. Analytical fix, no NKI. | `test_glm52_indexer_scale_audit.py` | GLM-5.2 |
| `gemma4_no_fallback_mitigations.py` | Python-side coverage for the two Gemma-4 26B-A4B CPU-fallback triggers PR #172 does not close: argmax kernel disable at high batch (trigger #3) and the `(GLU, GELU_TANH_APPROX)` activation branch guard (trigger #4). Triggers #1 and #2 are wired through the PR #172 adapter shim (`import_pr172_flash_attention`, `import_pr172_kv_cache_manager`); until PR #172 merges, they resolve against a local vendored snapshot. Renamed 2026-08-28 from `gemma4_cpu_fallback_replacement.py` — the name reflects intent (prevent fallback, never plan for it). | `test_no_cpu_fallback.py` (also universal — see below) | Gemma-4-26B-A4B |

## Slugs

Each NKI kernel exposes a slug constant that is the only cache identity for its compiled NEFF. Change the slug when the semantics change; a smoke test on each kernel checks its slug string.

| Kernel | Slug constant | Value |
|---|---|---|
| DSA Lightning Indexer v1 | `DSA_LIGHTNING_INDEXER_NKI_V1_KERNEL_SLUG` | `dsa_sparse_attention.nki_v1` |
| KDA state v2 | `KDA_STATE_NKI_V2_KERNEL_SLUG` | `kda_state.nki_v2` |
| DMA coalescing v1 | `DMA_COALESCING_NKI_V1_KERNEL_SLUG` | `dma_coalescing.nki_v1` |
| MoE dispatch | (Python-side, NKI slug reserved) | — |
| GLM-5.2 FP8 fix | (analytical, no NKI) | — |

## Tests

```
cd vllm_neuron/kernels
py -3 -m pytest -q
```

Last run on Windows Python 3.12.10 (this repo layout): 317 tests, 306 passed, 11 skipped, 165 s wall. The 11 skips are environmental: 2 want `GLM_FP8_INDEX_PATH` set to a live GLM-5.2 FP8 checkpoint, 5 want `neuronxcc.nki` on the host, 4 in `test_no_cpu_fallback.py` want a `--compile-log`, `--artifact-dir`, or `--runtime-probe` flag pointed at a real target. `pytest.ini` here sets `--import-mode=importlib` and pins `rootdir` at this directory so pytest does not walk up to `vllm_neuron/__init__.py`.

## Rules the kernels enforce

- **No CPU fallback.** Every compile lane on every model runs `tests/test_no_cpu_fallback.py` against its compile log and artifact dir. The test greps for canonical fallback markers (`falling back to cpu`, `torch_blockwise_matmul_inference`, `op fallback`, `emitting host code`, `partition cap exceeded`, and six more — see `CPU_FALLBACK_GREP_PATTERNS`) and fails loudly on any match. When Trn2 is reachable, `--runtime-probe` adds a live `neuron-top` sample check. This is a first-class campaign rule (Gemma-4-26B-A4B previously hit MFU 0.06% because a silent fallback slipped through). Invocation: `pytest kernels/tests/test_no_cpu_fallback.py --compile-log <path> --artifact-dir <dir>`.
- **No speculative decoding.** KDA v2 dispatch paths raise on `softmax_impl` requests (`test_dispatch_rejects_softmax_impl`, `test_prefill_shim_rejects_softmax`), and a source scan (`test_source_omits_spec_decode_branch`) fails if a spec-decode branch is added to any kernel.
- **LSE base is natural log.** Both DSA reference and NKI kernels expose `LSE_BASE_CONVENTION = "natural"`. Consumers that assume base-2 read wrong tokens. See `SGLANG-DSA-LSE-FIX-ANALYSIS-2026-08-28.md` for the SGLang cross-check.
- **File-import, not `exec()`.** NKI kernel bodies live under `_kernel_bodies/` and are loaded through `importlib.util.spec_from_file_location`. `exec(src, ns)` leaves the function without a real `__file__`, and `KernelRewriter.reparse_function` then raises `OSError: could not get source code` at compile time. See `EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md`.

## STATUS docs

Details per kernel and the two adjacent analyses:

- `DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md`
- `KDA-STATE-STATUS-2026-08-28.md` — v1 int8, algorithmic gap
- `KDA-STATE-V2-STATUS-2026-08-28.md` — v2 bf16 matching FLA v0.5.2
- `DMA-COALESCING-STATUS-2026-08-28.md`
- `MOE-DISPATCH-STATUS-2026-08-28.md`
- `GLM52-INDEXER-FP8-FIX-STATUS-2026-08-28.md`
- `GREEN-BOARD-2026-08-28.md` — aggregate regression board from the staging tree
- `EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md` — the file-import pattern
- `SGLANG-DSA-LSE-FIX-ANALYSIS-2026-08-28.md` — natural-log LSE cross-check
- `VLLM-KDA-KERNEL-FLAVOR-2026-08-28.md` — KDA reference-flavor gap analysis
- `GEMMA4-NO-FALLBACK-REFACTOR-2026-08-28.md` — Gemma-4 fallback-mitigation rename + refactor rationale (PR #172 supersession for triggers #1 and #2)

## On-device order

Once the next Trn2 capacity window opens:

1. DSA Lightning Indexer v1 — single kernel, single graph.
2. DMA coalescing v1 — single kernel, orthogonal to DSA.
3. KDA state v2 — Kimi-K3 and GLM-5.3-Flash shape presets get distinct cache slugs.

MoE dispatch and the GLM-5.2 FP8 fix are Python-side and effective without a NKI compile.

## License

Apache-2.0, inherited from `vllm-neuron`. Every file starts with `# SPDX-License-Identifier: Apache-2.0`. CPU references are original implementations validated against public references (FLA v0.5.2, SGLang for the LSE convention); no vendor code is copied.

Author: `fleet-a-nki-kernels-agent`. Per-kernel callsigns: `dsa-nki-v1-agent`, `kda-nki-v2-agent`, `dma-nki-v1-agent`, `moe-dispatch-v0-agent`, `glm52-fp8-fix-agent`, `gemma4-no-fallback-agent` (renamed 2026-08-28 from `gemma4-cpu-fallback-agent`).
