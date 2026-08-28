# DMA descriptor coalescing pass - status

**Date:** 2026-08-27 (deliverable dated -08-28 per the prompt)
**Author:** Fleet A worker agent, Trainium2 campaign
**Scaffold parent:** `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\NKI-DMA-COALESCING-SCAFFOLD-2026-08-27.md`

---

## 1. Unblock type

**Hybrid: NKI wrapper (Path A) + Python-level KV-slab pre-allocation reshape (Path B), plus a CPU-side descriptor-stream analyzer (Path C) that certifies mechanism before any device time is spent.**

HLO-level was evaluated and rejected as intractable on this box: NxDI lowers PyTorch to StableHLO via `torch-neuronxcc` inside the compile container (`sha256:be11c204f419a63e2487b2124005156dad091fb9edbfcadf42d81b745e284c12` for the vLLM-Neuron path, `sha256:011d49c7...` for the direct NxDI+MoE path per MEMORY.md). Intervening between StableHLO and NEFF requires either a compiler plugin (no public Trn2 API surface today) or hand-editing HLO IR post-lowering (no Python entry point). Both are single-shot rabbit holes; Paths A and B are both compositional and NEFF-diff verifiable.

## 2. Deliverables

| file (absolute) | role | test coverage | LOC |
|---|---|---|---|
| `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\dma_coalescing_transform.py` | Path A NKI wrapper (v0) + Path B KV-slab planner + Path C summary-json analyzer + NEFF-diff gate | 15/15 unit tests passing | 461 |
| `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\dma_coalescing_nki_v1.py` | **NKI v1 device kernel** - `@nki.jit`-decorated K-way coalescing body; source-string scaffold survives non-NKI hosts | 24/26 unit tests passing (2 skipped for NKI-not-present) | 285 |
| `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_dma_coalescing_smoke.py` | Golden smoke test (8 gate classes, 15 cases) | passes locally: `python -m unittest ... -v` | 235 |
| `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_dma_coalescing_nki_v1_smoke.py` | **NKI v1 smoke test** (7 gate classes, 26 cases: import + identity, source-scaffold patterns, signature contract, NKI-unavailable fallback, NKI-present compile smoke, first-fire lane manifest, v0 sibling consistency) | passes locally | 245 |
| `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\DMA-COALESCING-STATUS-2026-08-28.md` | this document | - | - |

Smoke test result on this Windows box (no NKI toolchain):
```
# v0 (Path A wrapper + Path B/C planners)
Ran 15 tests in 0.059s
OK

# v1 (NKI device kernel)
Ran 26 tests in 0.013s
OK (skipped=2)   # 2 NKI-runtime-only gates skip cleanly on Windows
```

---

## 2a. NKI v1

**Landing status (2026-08-27):** SOURCE-STRING scaffold + import-gated live `@nki.jit` function landed at
`C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\dma_coalescing_nki_v1.py`.

Slug: `dma_coalesced_gather.nki_v1`. Callsign: `dma-coalescing-nki-v1-agent`.

**Design:** single-source-of-truth. The kernel body lives in a module-level string constant
`DMA_COALESCED_GATHER_NKI_V1_SOURCE`. When `nki`/`nki.isa`/`nki.language` import cleanly (Trn2 compile host),
the string is `exec()`-ed into a private namespace and bound as the module-level `dma_coalesced_gather_nki_v1`.
When NKI is missing (this Windows box, most non-Trn2 hosts), a stub raises `NotImplementedError` with a
clear message pointing callers to v0's CPU-side planners. Prevents drift between the string and the live
function; ships to the compile host in exactly the form the Trn2 tracer needs.

**Address pattern (K>=2 K-group):**
```python
src=source_hbm.ap(
    [
        [B, K],   # outer: K packets of B bytes each; stride B, count K
        [1, B],   # inner: B contiguous bytes per packet
    ],
    offset=0,
    vector_offset=indices[g*K : (g+1)*K],
    indirect_dim=0,
)
```
Mirrors the K-way batched indirect DMA that attention_tkg already emits at
`harness-v2/staging/cycle630/remote-core.py:2421` (`bufs.k_prior_reshaped.ap([[stride, count], [1, stride]], vector_offset=cur_blks, indirect_dim=0)`).
The only new degree of freedom is the compile-time-baked K - not a new address-pattern primitive.
Risk surface is bounded to `.ap(...)` constant folding, not to any new NKI construct.

**Correctness gates preserved from v0:**
1. K=1 passthrough branch (uses the same `ap([[B,1],[1,B]], vector_offset=indices[i:i+1], indirect_dim=0)` shape shrunk to K=1).
2. `oob_mode.skip` default preserves KV-cache -1 sentinel semantics.
3. SBUF budget guard: `assert K*B <= SBUF_BUDGET_BYTES_PER_CALL_SITE` (2 MiB per call site).
4. Named per-group descriptors (`_g{g}`) for A/B provenance in `neuron-profile view`.

**Prep for Trn2 A/B window (GPT-OSS-20B TP=8 C=128 K=8 first-fire):**
- Manifest `FIRST_FIRE_LANE` in the module carries: `K=8`, `B=650`, `coalesced_bytes=5200`, `projected_multiplier=(1.4, 2.0)`, baseline receipt root `/mnt/scratch/tkg-profile-gpt-oss-20b-tp8-b128-20260827T052556Z/`, container `sha256:be11c204...`.
- Coalesced descriptor 5200 B is above `EFFICIENT_WINDOW_BYTES_MIN` (4096) and 400x under the SBUF budget (2 MiB).
- Fire sequence (per §4 of this document): Path B KV-slab reshape first (30 min), NEFF-content diff (5 s), verify_splice 10-token (15 min), Trn2 3-exec profile capture (15 min). If Path B lands, Path A NKI v1 fires second as the second-order verification that catches whatever Path B leaves on the table.

**On the Trn2 host, to smoke the v1 body:**
```python
from dma_coalescing_nki_v1 import (
    dma_coalesced_gather_nki_v1, FIRST_FIRE_LANE, is_available,
)
assert is_available(), "NKI toolchain required"
# Traceable through the standard NKI simulator or a real Trn2 device:
# dma_coalesced_gather_nki_v1(src_hbm, indices, out_sbuf,
#                              K=8, per_transfer_size=650, num_transfers=32)
```

## 3. Which lane fires first

**GPT-OSS-20B TP8 C=128** at K=8 (path A NKI wrapper primary; path B KV-slab reshape as compat A/B partner).

Rationale (source: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\gpt-oss-20b-tp8\PROFILE-C128-KNEE-2026-08-27.md`):

- 650 B HW-dynamic packets x 5.9M packets = the highest-absolute-tok/s knee measured (764.27 tok/s/card).
- K=7 (target 4 KiB, per `plan_coalesce_factor(650) = 7`) is moderate: 4.5 KiB coalesced descriptor sits inside SBUF budget with 4x headroom.
- Baseline NEFF + 3-exec profile receipt already banked at `/mnt/scratch/tkg-profile-gpt-oss-20b-tp8-b128-20260827T052556Z/` per the profile receipt cross-reference - a direct A/B does not require re-compiling the baseline.
- Projected uplift: 764 -> 1050-1090 tok/s/card (1.4-2.0x, from `PROFILE-AT-KNEE-SUMMARY-2026-08-27.md`).
- Confirmed with `build_coalescing_plan(...)`: multiplier bucket = (1.4, 2.0).

**Second fire (higher relative uplift, higher risk): Qwen3-32B TP8 C=16** at K=44 (path B pre-alloc reshape ONLY at first; path A K=44 exceeds a comfortable SBUF budget test window without empirical per-descriptor breakdown from `neuron-profile view` that Fleet A hasn't yet captured; scaffold GAP #9). Projected: 145 -> 260-350 tok/s/card (2.4-3.0x, largest relative headroom in campaign).

**Third fire: GPT-OSS-20B TP4 C=4** at K=5 (both paths, low risk). Projected: 135 -> 150-165 tok/s/card (1.15-1.30x).

An A/B partner contract already exists for the compiler-flag equivalent on Qwen3-32B: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\qwen3-32b-tp8\contract-s9216-b16-transferblocking.json` (delta = one line `--transfer-blocking-hint`). This means the mechanism can be smoke-checked via compiler flag first, and the Path A NKI wrapper fires as the second-order verification that catches whatever the compiler flag leaves on the table.

## 4. Recommended fire sequence (post-cliff, ~1-2 h window)

| step | action | time | gate | on failure |
|---|---|---|---|---|
| 1 | Path B: apply `plan_kv_slab_layout(...)` to GPT-OSS-20B TP8 C=128 NeuronConfig; recompile | 30 min | Tier-1 CPU battery + slug determinism | shrink K; go to step 2 |
| 2 | `run_neff_content_check(baseline, candidate, require_different=True)` | 5 s | TKG program bytes must differ | knob did not land; audit compile flags |
| 3 | `verify_splice --tokens 10` vs baseline | 15 min | 10/10 exact match | block lane; audit ap() shape emission |
| 4 | Trn2 fire: 3-exec `neuron-explorer capture` at same slug | 15 min | success gate: >= 30% `dma_active_time_percent` reduction | escalate to path A NKI wrapper |
| 5 | If step 4 lands: repeat 1-4 for Qwen3-32B TP8 C=16 (2.4-3x projected) | 60 min | same gates | fall back to path B only |

Total: ~2 h for GPT-OSS lane + Qwen3-32B lane back-to-back. Fits inside the next available Trn2 capacity block.

## 5. Open blockers

| # | blocker | severity | resolution path |
|---|---|---|---|
| B1 | NKI toolchain (`nki`, `neuronxcc`) unavailable on this Windows box | HIGH for Path A validation | container `sha256:be11c204...` on next Trn2 host - `nki.available()` gate in module auto-skips Path A when False |
| B2 | Trn2 device access — capacity block ends 2026-08-27T11:30Z per MEMORY.md `Trn2 11:30Z hard cliff` | HIGH for any fire | next 24 h+ window per operator; Fleet B queue picks up |
| B3 | `nisa.dma_copy` multi-source strided-descriptor API signature not publicly documented (scaffold GAP #5) | MEDIUM for Path A K>=2 | validate empirically on next Trn2 host with a 1-line smoke; the wrapper uses the same `ap(...)`/`vector_offset`/`indirect_dim=0` shape the existing attention_tkg already emits at `harness-v2/staging/cycle630/remote-core.py:2421` so the risk surface is bounded |
| B4 | NxDI local path to `HloTorchCompatibleAttentionBlockTkgKernel` shim gone with ondemand12 tear-down (scaffold GAP #7) | MEDIUM for integration | re-fetch via `gh api repos/aws-neuron/neuronx-distributed-inference/contents/...` post-cliff; the wrapper is drop-in either way |
| B5 | `neuron-profile view` per-descriptor breakdown (uniform vs strided K-groups) not captured in existing summary-jsons (scaffold GAP #9) | MEDIUM for Qwen3-32B path A K=44 | re-run `neuron-profile view` with an extended output-format on preserved NTFFs at `/mnt/scratch/tkg-profile-qwen3-32b-tp8-b16-20260827T054642Z/` |
| B6 | Descriptor-issue latency assumption 50-100 ns (scaffold GAP #1) drives the `plan_coalesce_factor` break-even; empirical value unknown | LOW - direction of lever is invariant, only bucket multiplier is affected | one 30-min NKI micro-bench in the next window per scaffold s.4.4 |
| B7 | KV-slab reshape (Path B) needs a runtime shim to convert the pre-existing KV tensor when `block_size` changes between checkpoints | LOW - fresh compile is the campaign norm | integration is trivial when the compile is being resubmitted anyway (which it will be for this lever) |

## 6. What is NOT this deliverable

- Not a merged NxDI patch (no code lands in `neuronx-distributed-inference` until the Path A wrapper survives its Trn2 A/B).
- Not a compiled NEFF (no compile host available on this Windows box; the NxDI compile requires the container's SDK 2.32).
- Not a Neuron Explorer capture (no device access this turn).
- Not a `verify_splice --tokens 10` run against a real baseline NEFF (needs the compile).

The next agent (or the same one, next window) picks up at step 1 of section 4 above.

## 7. Peer / competitor context

Per MEMORY.md `Makora TrainSpotting competitive context`: Makora claims 1.4-6.8x speedups on Qwen3.5-4B / Qwen3-30B-A3B / Gemma-4-31B; ratios only, no absolute $/M. If the DMA coalescing lever lands as projected (1.4-3.0x cross-knee), it puts Trn2 tokenomics inside Makora's claimed range without needing spec-decode (which is forbidden per MEMORY.md `No spec-decode methodology 2026-08-27`). This is a per-model beat, not a synthetic-benchmark beat - the Path A wrapper generalizes across all attention_tkg call sites, so it stacks with model-specific levers (KV int8, MoE grouped matmul, RoPE fusion) rather than competing with them.

## 8. Cross-reference

- Scaffold: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\NKI-DMA-COALESCING-SCAFFOLD-2026-08-27.md`
- Universal finding source: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\PROFILE-AT-KNEE-SUMMARY-2026-08-27.md`
- Existing attention_tkg NKI code: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\cycle630\remote-core.py` (lines 2313, 2371, 2421 are the batching precedent)
- GEMMA4-LESSONS gate reference: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\GEMMA4-LESSONS-GENERALIZED-2026-08-27.md` (A6 NEFF-diff, D3 verify_splice)
- Compiler-flag A/B partner: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\lanes\qwen3-32b-tp8\contract-s9216-b16-transferblocking.json`
