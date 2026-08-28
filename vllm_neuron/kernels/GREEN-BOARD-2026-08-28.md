# Cross-kernel regression green board — 2026-08-28

Aggregated pass/fail across all 7 shipped Fleet A kernel decks (10 test files).

- Runner: `py -3 -m pytest -q` on Windows Python 3.12.10, pytest 8.4.2
- Working dir: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels`
- Wall-clock strategy: 10 pytest processes launched concurrently as background bash tasks; longest single suite (`test_kda_state_correctness.py`) dominates the true wall at 63.5 s. Serial wall is 149.65 s.
- Read-only: no code, kernel, or test file was modified this session.

## Aggregate line

**253 tests run · 248 passed · 0 failed · 5 skipped · 149.65 s serial wall (~65 s parallel wall) · STATUS = GREEN**

Zero regressions vs. prior STATUS baselines. Every suite matched its expected pass count exactly.

## Per-suite table

| # | file (absolute) | passed | failed | skipped | wall_s | baseline (prior STATUS doc) | delta |
|---|---|---:|---:|---:|---:|---|---|
| 1 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_dsa_lightning_indexer_correctness.py` | 24 | 0 | 0 | 5.22 | 10 gates T0..T9 parametrized over `K ∈ {2048,4096,8192,16384,32768}` × `topk ∈ {2048,4096}`, combined w/ speed = 77 pass per `DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md` | 0 |
| 2 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_dsa_lightning_indexer_speed.py` | 53 | 0 | 0 | 3.30 | 6 gates S0..S5 parametrized over 3 shapes × 5 L buckets; combined w/ correctness = 77 pass per `DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md` | 0 |
| 3 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_dsa_lse_accumulator.py` | 8 | 0 | 0 | 2.83 | 8 gates, all green in 3.14 s per `SGLANG-DSA-LSE-FIX-ANALYSIS-2026-08-28.md` | 0 |
| 4 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_kda_state_correctness.py` | 22 | 0 | 0 | 63.48 | 22 tests all passing per `KDA-STATE-STATUS-2026-08-28.md` | 0 |
| 5 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_kda_state_speed.py` | 23 | 0 | 0 | 0.18 | 23 tests all passing per `KDA-STATE-STATUS-2026-08-28.md` | 0 |
| 6 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_kda_state_v2_correctness.py` | 25 | 0 | 0 | 34.98 | 25 tests per `KDA-STATE-V2-STATUS-2026-08-28.md` | 0 |
| 7 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_kda_state_v2_speed.py` | 23 | 0 | 0 | 0.14 | 23 tests per `KDA-STATE-V2-STATUS-2026-08-28.md` | 0 |
| 8 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_dma_coalescing_smoke.py` | 15 | 0 | 0 | 0.08 | 15 unit tests per `DMA-COALESCING-STATUS-2026-08-28.md` | 0 |
| 9 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_glm52_indexer_scale_audit.py` | 36 | 0 | 2 | 0.21 | 36 pass + 2 env-var-gated skip per `GLM52-INDEXER-FP8-FIX-STATUS-2026-08-28.md` | 0 |
| 10 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_moe_dispatch_correctness.py` | 19 | 0 | 3 | 39.23 | 19 pass + 3 NKI-skip per `MOE-DISPATCH-STATUS-2026-08-28.md` | 0 |
| | **TOTAL** | **248** | **0** | **5** | **149.65** | 248 pass + 5 skip expected | **0** |

## Failures

None. No regression-since-baseline detected. No fix-hunt section required.

## Skipped tests (all expected, all environmental)

- `test_glm52_indexer_scale_audit.py` — 2 skips: `TestLiveCheckpointAudit` class, gated on `GLM_FP8_INDEX_PATH` env var pointing at a live GLM-5.2 FP8 checkpoint (not present on this Windows box). Baseline expected behavior per STATUS doc.
- `test_moe_dispatch_correctness.py` — 3 skips: NKI-toolchain-gated tests (`nki.available() == False` on Windows). Baseline expected behavior per STATUS doc.

## Baseline reference lines (per-kernel expected pass counts from prior STATUS docs)

| kernel | expected passed | expected skipped | source STATUS doc (absolute) |
|---|---:|---:|---|
| DSA lightning indexer (corr + speed combined) | 77 (24 + 53) | 0 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\DSA-LIGHTNING-INDEXER-STATUS-2026-08-28.md` |
| DSA LSE accumulator | 8 | 0 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\SGLANG-DSA-LSE-FIX-ANALYSIS-2026-08-28.md` |
| KDA state v1 (corr + speed) | 45 (22 + 23) | 0 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\KDA-STATE-STATUS-2026-08-28.md` |
| KDA state v2 (corr + speed) | 48 (25 + 23) | 0 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\KDA-STATE-V2-STATUS-2026-08-28.md` |
| DMA coalescing smoke | 15 | 0 | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\DMA-COALESCING-STATUS-2026-08-28.md` |
| GLM 5.2 indexer FP8 scale audit | 36 | 2 (env-gated) | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\GLM52-INDEXER-FP8-FIX-STATUS-2026-08-28.md` |
| MoE dispatch | 19 | 3 (NKI-gated) | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\MOE-DISPATCH-STATUS-2026-08-28.md` |
| **Total** | **248** | **5** | | 

Every observed count matches its baseline. Green board.

## Reproducer

```
cd C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels
py -3 -m pytest -q tests/test_dsa_lightning_indexer_correctness.py tests/test_dsa_lightning_indexer_speed.py tests/test_dsa_lse_accumulator.py tests/test_kda_state_correctness.py tests/test_kda_state_speed.py tests/test_kda_state_v2_correctness.py tests/test_kda_state_v2_speed.py tests/test_dma_coalescing_smoke.py tests/test_glm52_indexer_scale_audit.py tests/test_moe_dispatch_correctness.py
```

Raw per-suite pytest outputs saved this session at:
`C:\Users\apumu\AppData\Local\Temp\claude\C--Users-apumu-research-InfinityAI-gemma4-trn2-handoff\fe22bc9a-b8c7-4107-ac55-1d55ba4bd33a\scratchpad\green-board\`
