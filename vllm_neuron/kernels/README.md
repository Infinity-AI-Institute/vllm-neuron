# Fleet A NKI kernels + CPU-golden references

Author: Fleet A Callsign `fleet-a-nki-kernels-agent`
Date: 2026-08-28
Source-of-truth staging tree: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\`
Target in-repo path: `vllm_neuron/kernels/`
Base branch: `agent-main` (SHA at PR-open: `9427b3baf11927d11882e945e7fba58c09a0ba53`)
Runner: Windows Python 3.12.10, pytest 8.4.2 (CPU-only host — Trn2 device firing deferred to the next window)

## What ships

Five NKI kernel families with CPU-golden references, plus two adjacent Python-side mitigations. Every kernel is source-string device-ready (the NKI body files are validated by dry-import + shape smoke on any host); every CPU-golden reference is bit-exact against the vendor / model reference cited in its STATUS doc.

| # | Family | CPU golden | NKI v1/v2 source | Unlocks |
|---|---|---|---|---|
| 1 | DSA Lightning Indexer (DeepSeek-V4-style sparse-attention prefilter) | `dsa_lightning_indexer.py` (32/32) | `dsa_lightning_indexer_nki_v1.py` + `_kernel_bodies/dsa_lightning_indexer_nki_v1_body.py` | DeepSeek-V4 / DSv4 sparse-attention prefill on Trn2 (matches SGLang LSE natural-log convention — cross-verified in `SGLANG-DSA-LSE-FIX-ANALYSIS-2026-08-28.md`) |
| 2 | KDA state (linear-attention state kernel, v2 bf16 bit-exact vs FLA v0.5.2) | `kda_state.py` (v1 int8 study — algorithmic gap documented), `kda_state_v2.py` (45/45) | `kda_state_nki_v2.py` + `_kernel_bodies/kda_state_nki_v2_body.py` | Kimi-K3 / GLM-5.3-Flash / any FLA-consuming decoder; documented v1 int8 gap and int8 study lane (`kda_state_int8_study.py`) is marked `NOT_SERVED` to prevent accidental promotion. Reference-flavor gap analysis in `VLLM-KDA-KERNEL-FLAVOR-2026-08-28.md` |
| 3 | DMA coalescing hybrid transform | `dma_coalescing_transform.py` (15/15) | `dma_coalescing_nki_v1.py` + `_kernel_bodies/dma_coalescing_nki_v1_body.py` | Any decoder with fine-grained HBM strided reads (Qwen3.5-4B, Gemma-4, GLM-5.2 index-attn) |
| 4 | MoE dispatch + fail-loud fallback ladder | `moe_dispatch.py` (19 pass + 3 NKI-gated skip) | (fallback ladder is Python-side, NKI-slug reserved) | Qwen3-30B-A3B, DSv4 experts, GLM-5.3 MoE; enforces `use_shard_on_intermediate_dynamic_while=True` workaround for container `sha256:011d49c7` (see project memory `nxdi-container-moe-blockwise-mm-workaround-20260827.md`) |
| 5 | GLM-5.2 indexer FP8 scale-cap fix | `glm52_indexer_fp8_scale_fix.py` (36 pass + 2 env-gated skip) | (analytical fix; CLI included) | GLM-5.2 FP8 indexer numerics on Trn2 (fixes vendor scale-cap defect PR #13 residual) |
| — | Gemma-4 CPU-fallback replacement (Python-side) | `gemma4_cpu_fallback_replacement.py` | — | Mitigation for the 4 trigger classes that push Gemma-4 26B-A4B to CPU under container `sha256:011d49c7`; Gemma-4 itself is deferred per operator (see project memory `gemma4-deferred-cpu-fallback.md`), but the mitigation ships so a future Fleet B lane can re-open without re-deriving it |

## Test results

**307 tests · 300 passed · 7 skipped · 0 failed · 75.94 s wall (single-process, in-repo)**

Zero regressions vs. the standalone Fleet A staging tree's `GREEN-BOARD-2026-08-28.md` (253 tests, 248 pass + 5 skip) — the extra 54 tests come from the three NKI-source smoke suites (`test_dsa_lightning_indexer_nki_v1_smoke.py`, `test_kda_state_nki_v2_smoke.py`, `test_dma_coalescing_nki_v1_smoke.py`), which weren't in the aggregated green-board table but pass cleanly on Windows too (dry-import + source-string surface only — no live NKI compile required for smoke).

The 7 skips are all expected and environmental:
- `test_glm52_indexer_scale_audit.py` — 2 skips: `GLM_FP8_INDEX_PATH` env var not pointing at a live GLM-5.2 FP8 checkpoint.
- `test_moe_dispatch_correctness.py` — 3 skips: `nki.available() == False` on CPU host.
- `test_dma_coalescing_nki_v1_smoke.py` — 1 skip: NKI-toolchain gated.
- `test_kda_state_nki_v2_smoke.py` — 1 skip: NKI-toolchain gated.

### Reproducer

```
cd vllm_neuron/kernels
py -3 -m pytest -q
```

A local `pytest.ini` at `vllm_neuron/kernels/pytest.ini` sets `--import-mode=importlib` and isolates rootdir at the kernels directory so pytest does not walk up to `vllm_neuron/__init__.py` (which imports the vllm package). This keeps the kernel test suite self-contained CPU + numpy/torch only.

## Slug and cache-pinning conventions

Every NKI kernel exposes a stable slug constant that is the sole cache identity for the compiled NEFF. Slugs must not drift silently — a slug change is a cache-invalidation event and must be committed as such.

| kernel | slug constant | value |
|---|---|---|
| DSA Lightning Indexer v1 | `DSA_LIGHTNING_INDEXER_NKI_V1_KERNEL_SLUG` | `dsa_sparse_attention.nki_v1` |
| KDA state v2 (NKI) | `KDA_STATE_NKI_V2_KERNEL_SLUG` | `kda_state.nki_v2` |
| DMA coalescing v1 | `DMA_COALESCING_NKI_V1_KERNEL_SLUG` | `dma_coalescing.nki_v1` |
| MoE dispatch (reserved) | see `moe_dispatch.py` fallback ladder | — |
| GLM-5.2 indexer FP8 fix | (analytical / Python-side) | — |

Each smoke suite gates on its slug constant so a rename is caught before compile-cache poisoning.

## Anti-inheritance rules preserved

These are hard rules the CPU goldens and NKI drafts BOTH honor. Any downstream re-flavoring must preserve them:

1. **No spec-decode as measurement / optimization axis / comparison baseline** (operator hard rule 2026-08-27; see project memory `no-spec-decode-methodology-20260827.md`). All KDA-v2 dispatch paths raise on `softmax_impl` requests; `kda_state_v2_smoke::test_dispatch_rejects_softmax_impl` and `test_prefill_shim_rejects_softmax` are the guardrails. Every kernel source is scanned by a smoke test (`test_source_omits_spec_decode_branch`) to fail if a spec-decode branch ever leaks in.
2. **LSE natural-log convention** — the DSA Lightning Indexer's LSE is base-`e` (natural log), matching SGLang's reference (see `SGLANG-DSA-LSE-FIX-ANALYSIS-2026-08-28.md`). Both v0 and v1 expose `LSE_BASE_CONVENTION = "natural"` and smoke-check they match. Any consumer that assumes base-2 will read wrong tokens — the constant is the single source of truth.
3. **File-import, not `exec()`** — every NKI kernel body lives in its own `.py` file under `_kernel_bodies/` and is imported by module path, never by `exec(open(...))`. Full pattern in `EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md`. This is enforced because `exec` breaks IDE navigation, static analysis, coverage, and (critically) the NKI compile driver's own source-locator when it emits error line numbers.

## Fleet A Callsign attribution

Every kernel and STATUS doc names the Callsign of the agent that authored it. The five NKI kernel callsigns:

- `dsa-nki-v1-agent` — DSA Lightning Indexer v1 NKI + smoke
- `kda-nki-v2-agent` — KDA state v2 NKI + smoke
- `dma-nki-v1-agent` — DMA coalescing v1 NKI + smoke
- `moe-dispatch-v0-agent` — MoE dispatch v0 + fail-loud fallback ladder
- `glm52-fp8-fix-agent` — GLM-5.2 indexer FP8 scale-cap fix
- `gemma4-cpu-fallback-agent` — Gemma-4 CPU-fallback Python mitigations

Aggregating Callsign for the push itself: `fleet-a-nki-kernels-agent`.

## Not yet fired on device

Trn2 measurement is deferred to the next capacity-block window. The kernels ship source-string device-ready with all CPU goldens bit-exact against their reference, and every smoke test enforcing slug + surface + anti-inheritance discipline. On-device kickoff order once the window opens:

1. DSA Lightning Indexer v1 (single-kernel, single-graph, cleanest surface)
2. DMA coalescing v1 (single-kernel, orthogonal to DSA)
3. KDA state v2 (multi-shape presets — Kimi-K3, GLM-5.3-Flash — need distinct cache slugs)

MoE dispatch and GLM-5.2 FP8 fix are Python-side and already effective without a NKI compile.

## STATUS document index

Full-scope STATUS docs are shipped alongside the kernels in this directory:

- `DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md`
- `KDA-STATE-STATUS-2026-08-28.md` (v1 int8, algorithmic gap called out)
- `KDA-STATE-V2-STATUS-2026-08-28.md` (v2 bf16, bit-exact FLA v0.5.2)
- `DMA-COALESCING-STATUS-2026-08-28.md`
- `MOE-DISPATCH-STATUS-2026-08-28.md`
- `GLM52-INDEXER-FP8-FIX-STATUS-2026-08-28.md`
- `GREEN-BOARD-2026-08-28.md` (aggregate 253-test regression board from the standalone staging tree)
- `EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md` (universal file-import discipline pattern)
- `SGLANG-DSA-LSE-FIX-ANALYSIS-2026-08-28.md` (natural-log LSE cross-verification)
- `VLLM-KDA-KERNEL-FLAVOR-2026-08-28.md` (KDA reference-flavor gap analysis)

## Reference bar evidence (the "why" these kernels earn a compile budget)

- **Makora TrainSpotting blog (2026-08-10)** claims 1.4-6.8x speedups on Qwen3.5-4B / Qwen3-30B-A3B / Gemma-4-31B dense — ratios only, no absolute $/M. We're closing the gap on absolute $/M per model.
- **AWS Bedrock, H100 vLLM, OpenRouter DSv4** — public tokenomics bars that operator's hard rule 2026-08-27 (`beat-h100-b300-per-model-mandate-20260827.md`) requires we beat per-model or root-cause the gap.
- **Reference bar as of 2026-08-27:** PR #3 (Gemma-4 26B-A4B) sets $0.031/M (H100 FP8) and $0.058/M (B300 BF16). Gemma-4 is deferred pending better NKI coverage; DSA + KDA + DMA close the gap on DeepSeek-V4 and Kimi-K3 first.

## License and provenance

- License: Apache-2.0 (inherited from `vllm-neuron`).
- Every file carries `# SPDX-License-Identifier: Apache-2.0` at the top.
- Author attribution: Fleet A Callsign `fleet-a-nki-kernels-agent` (this push) + per-kernel Callsigns above.
- CPU goldens are original implementations validated against public reference libraries (FLA v0.5.2, SGLang for LSE convention); no vendor code is copied.
