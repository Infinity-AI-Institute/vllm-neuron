# SGLang TileLang DSA "LSE fix" — analysis + Trn2 applicability

- author: Trn2 reference-sweep worker, 2026-08-27
- budget: 2 agent-hour unblock
- SUT: `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\dsa_lightning_indexer.py`
- deliverable: this file + patched SUT + new gate `test_dsa_lse_accumulator.py`
- inputs read:
  - `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\GLM-5-3-FLASH-ARCHITECTURE-2026-08-28.md` §12, §15
  - SGLang PR #31821 (DCP for DSA, OPEN) — `python/sglang/srt/layers/attention/dsa_backend.py`, `python/sglang/srt/layers/dcp/comm.py`
  - SGLang PR #35045 (base-2 to natural-log LSE in chunked-prefix MLA merge, OPEN)
  - SGLang HEAD `python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py`
  - SGLang HEAD `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py`

## TL;DR

- **Classification: GENERAL sparse-softmax LSE-plumbing issue, NOT a Blackwell-only numerical quirk.**
- **Root cause of the SGLang comment ("TileLang DSA DCP decode needs the LSE fix"):** the TileLang DSA split-K decode kernel writes per-split `Partial_Lse` but its combine kernel does not emit a **final combined LSE**. TRT-LLM DSA's `ComputeLSEFromMD` does. DCP's cross-rank combine (`cp_lse_ag_out_rs_mla`) needs per-rank final LSE, so DCP is trtllm-only until TileLang exposes final LSE.
- **Applicability to our Trn2 NKI DSA Lightning Indexer:** high — the same class of bug WILL bite any NKI kernel that (a) split-Ks softmax across neuron cores, (b) shards KV across TP/DCP-style groups, or (c) composes with any cross-shard sparse-attention combine. The math is provider-independent; the fix pattern is provider-independent.
- **Patch shipped:** `dsa_sparse_attention_forward` and `dsa_lightning_indexer_forward` now accept opt-in `return_lse=True` and return a natural-log LSE with the all-masked → `-inf` sentinel. `LSE_BASE_CONVENTION="natural"` documents the contract. 8 new gates in `test_dsa_lse_accumulator.py` (all green in 3.14 s; existing 24-case correctness suite unaffected).

## 1. The exact SGLang statement

From `harness-v2\staging\reference-sweep-20260826T2150Z\GLM-5-3-FLASH-ARCHITECTURE-2026-08-28.md` §12, row "SGLang":

> "**TileLang DSA** kernel (needs LSE fix in some cells) and **TRT-LLM DSA** kernel are both available. `TileLang DSA DCP decode needs the LSE fix`; `TRT-LLM DSA DCP decode returns the LSE natively`."

Not a rumor: the two mechanisms are visible in the SGLang tree at HEAD.

## 2. What the SGLang code actually does

### 2.1 LSE base convention is per-backend

`python/sglang/srt/layers/dcp/comm.py`:

```
def cp_lse_ag_out_rs_mla(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    is_lse_base_on_e: bool = False,
):
    """Merge DCP partial attention outputs with the LSE's actual log base.
    ...
    FlashInfer MLA returns base-2 LSE, while FlashMLA returns natural-log LSE.
    """
```

The `is_lse_base_on_e` flag threads the base convention from producer (attention kernel) to consumer (cross-rank Triton combine). Producer's `is_mla_dcp_lse_base_on_e()` returns:
- `True` for `flashmla` / `cutedsl_mla` → producer emits natural-log (`log(sum(exp(x)))`).
- `False` (default) for FlashInfer MLA and both DSA backends → producer emits base-2 (`log2(sum(exp2(x)))`).

Base-2 is chosen because SM90+ has a native `ex2.approx` intrinsic and skipping the `ln(2)` prefactor on the softmax hot loop is a real speed-up on GPU. It is a software convention, not a numerical property of the hardware.

### 2.2 TRT-LLM DSA emits LSE natively

`python/sglang/srt/layers/attention/dsa_backend.py` (from PR #31821 diff):

```
# Under DCP each rank attends its filtered top-k shard; the
# base-2 LSE (ComputeLSEFromMD) feeds the cross-rank combine.
return_lse=self.dcp_enabled,
lse=lse_buf,
```

TRT-LLM's trtllm-gen sparse decode kernel accepts a caller-supplied `lse_buf` and writes base-2 LSE directly into it. The DCP path then registers `lse_buf` in the symmetric-memory pool so the cross-rank `cp_lse_ag_out_rs_mla` all-gather + Triton combine needs no extra copy.

### 2.3 TileLang DSA writes per-split LSE but not final combined LSE

`python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py`:

The `sparse_mla_fwd_decode_partial` kernel (split-K decoder) writes per-split base-2 LSE:

```
sumexp[h_i] = T.if_then_else(
    sumexp[h_i] == 0.0,
    -(2**30),                                        # empty-split sentinel
    T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale,       # base-2 LSE
)
...
T.copy(sumexp, Partial_Lse[b_i, s_i, group_i, H0:H1])
```

with `sm_scale = 1.0/sqrt(D+D_tail) * 1.44269504` (that constant is `log2(e)`, so multiplying by it converts `exp(x*a)` to `exp2(x*a*log2(e))`, keeping the whole hot loop in `T.exp2()`).

The `sparse_mla_fwd_decode_combine` kernel consumes `Partial_Lse` to weighted-sum `Partial_O`:

```
lse_max = max_k(shared_lse)
lse_sum = sum_k(exp2(shared_lse[k] - lse_max))
scale   = exp2(shared_lse[k] - lse_max - log2(lse_sum))
acc_o   = sum_k(scale * Partial_O[k])
T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
```

**The combine writes `Output` only. There is no `Output + final_lse` write path.** That is the "LSE fix" the cookbook is pointing at.

### 2.4 The DCP DSA PR explicitly restricts to trtllm

`python/sglang/srt/layers/attention/dsa_backend.py` (PR #31821 diff):

```
if self.dcp_enabled:
    assert (
        self.dsa_decode_impl == "trtllm" and self.dsa_prefill_impl == "trtllm"
    ), "..."
```

Confirms that DCP DSA is trtllm-only in the current PR precisely because TileLang has no final-LSE output.

### 2.5 Related but separate: PR #35045 (base-2 vs natural-log MLA merge)

Different bug, same class. FlashInfer's ragged MLA prefill wrapper returned base-2 LSE while `merge_state_v2` assumed natural-log; the fix multiplies by `ln(2)` at the runner boundary. Peak error was 0.105 on a chunked-prefix merge → 0.0005 after fix. Documented for the reader: this is what a base-convention mismatch looks like when it goes uncaught — silent 10% peak error in the merged output.

## 3. Classification: general, not Blackwell-only

The SGLang comment in-context is describing a **plumbing gap** in one specific software backend (TileLang DSA's split-K combine), not a hardware numerical property of Blackwell.

Evidence:
- The math the LSE serves — `w_a * out_a + w_b * out_b` where `w_i = exp(lse_i - global_lse)` — is textbook cross-partition softmax combine. Provider-independent, hardware-independent.
- The base-2 vs natural-log choice is a software convention driven by the hardware's fast intrinsic (`ex2.approx` on SM90+). Trainium2 has no such intrinsic; the natural-log convention lowers cleanly to `nl.exp` / `nl.log`.
- The bug pattern — "producer doesn't emit LSE; consumer needs it" — has already been named across the SGLang tree in TWO backends (TileLang DSA needs the final-LSE emit; FlashInfer MLA had a base-mismatch in the merge). Provider-independent.
- The all-masked-row sentinel (`-inf`) that `cp_lse_ag_out_rs_mla` relies on is exactly what `fixup_zero_kv_rows` writes in PR #31821 (`vec_neginf_fill(lse + tok * lse_stride, lse_stride)`). Provider-independent.

## 4. Applicability to our Trn2 NKI DSA Lightning Indexer

The CPU golden reference `dsa_sparse_attention_forward` in `dsa_lightning_indexer.py`:
- is single-pass over the K axis (no split-K in the reference itself),
- does not emit LSE today (only the attended output),
- has no cross-shard combine plumbing.

A future NKI v1 device kernel almost certainly does:
1. **Split-K across cores.** SBUF fits one Q-tile × one K-block of scores; the whole top-K rarely fits. Split-K is the standard pattern (see the scaffold's `q_tile=16` and per-block SBUF math). Every split emits (partial_out, partial_lse), and a combine kernel weighted-sums with the LSE.
2. **KV shards across TP=8/16 groups.** The Trn2 topology already runs multi-card TP; if a future measurement lane splits KV across ranks (as DCP does on GPU) or reuses split-K semantics across cards, the same LSE plumbing applies.
3. **Cross-lane composition (KDA + DSA in GLM 5.3 Flash).** GLM 5.3 Flash has 11 DSA layers at [3,7,...,43] and 34 KDA layers elsewhere — a mixed-attention forward that ever merges partial outputs across layer types will need the same LSE contract.

The class of bug the SGLang cookbook is warning us about **will bite us the day we split-K on a Trn2 NKI kernel and don't emit LSE with a documented base and an all-masked sentinel.**

## 5. The patch we shipped, and why it stays "reference"

Patched files (both absolute):
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\dsa_lightning_indexer.py`
- `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\tests\test_dsa_lse_accumulator.py` (new)

Changes:

1. **New constant `LSE_BASE_CONVENTION = "natural"`** — makes the contract greppable and lets the NKI author's compile-cache slug pin the base to the reference. If any device kernel switches to `"base2"` (e.g. because a future NKI intrinsic makes it faster), that flip is a single-line change with a required audit at every combine site.

2. **`dsa_sparse_attention_forward(return_lse=False)`** — opt-in kwarg. When True, returns `(out, lse)` with `lse[B, Q, H]` fp32 = `log(sum(exp(masked_scores)))`. Default keeps existing 1-tensor return so the 24 existing correctness tests and every downstream consumer see zero API change. Same idiom as SGLang `return_lse=self.dcp_enabled` in PR #31821.

3. **All-masked sentinel `-inf`** — matches `fixup_zero_kv_rows` in PR #31821. Any downstream combine's `exp(lse - global_lse)` yields exactly 0, no NaN.

4. **`dsa_lightning_indexer_forward(return_lse=False)`** — plumbed through, returns a 3-tuple `(attn, topk_ret, lse)` when True. Existing 2-tuple return when False.

5. **Rationale for natural-log rather than base-2 on Trn2:**
   - NKI has `nl.exp` and `nl.log`; there is no documented native base-2 intrinsic on Trainium2 that would give the same 2-3× speed-up the SM90+ `ex2.approx` gives GPUs.
   - Downstream Trn2 combines (SBUF-resident split-combine) don't have a hardware reason to prefer base-2.
   - Choosing natural-log matches SGLang's `flashmla`/`cutedsl_mla` convention, so if we ever cross-verify against those references (they're the SBUF-analogue in the CUDA world) the base matches out of the box.

6. **`test_dsa_lse_accumulator.py`** — 8 gates, ~5 s wall, CPU only:
   - `L0` LSE base convention is `"natural"` (compile-time constant).
   - `L1` LSE matches `torch.logsumexp(masked_scores, dim=K)` to fp32 rel tol 1e-5.
   - `L2` Split-recombine invariant: two disjoint index shards' `(out_i, lse_i)` LSE-weighted recombine bit-exactly reproduces the single-pass output at the union topk. This IS the math `cp_lse_ag_out_rs_mla` performs in PR #31821.
   - `L3` All-masked sentinel: `lse = -inf`, `exp(lse - anything_finite) = 0`, no NaN.
   - `L4` Dtype invariant: LSE is fp32 regardless of input dtype.
   - Plus one-shot forward plumbing check (3 shape assertions).
   - Results: `8 passed in 3.14s`. Existing correctness gate: `24 passed in 2.69s` (unchanged).

## 6. What we did NOT do (deferred, and why)

- **Did NOT change the NKI stub `_nki_kernel_stub_dsa_lightning_indexer_forward`.** It still `raise NotImplementedError`s. The stub's docstring GAP-1..GAP-7 already cover the pre-NKI-compile audit; adding an LSE requirement there is a doc change that should land with the actual NKI author's design pass, not in this 2-hour investigation.
- **Did NOT implement a NKI/device split-K combine.** Out of budget scope and requires a live device.
- **Did NOT change `full_attention_reference` or `full_attention_at_indices_reference`.** They stay pristine golden oracles; L2's recombine uses `full_attention_at_indices_reference` as the reference so any drift in `dsa_sparse_attention_forward`'s LSE gets caught by an independent path.

## 7. Actionable next-agent bullets

1. When landing NKI v1 DSA, treat `return_lse=True` as a first-class kernel entry variant with its own compile-cache slug key (add to `DsaKernelConfig`), NOT as a runtime flag on the LSE-less kernel. Distinct NEFFs.
2. If any Trn2 topology lane ever splits KV across cards (DCP-style), write the cross-card combine against `test_dsa_lse_accumulator.py::test_L2_split_lse_recombine_reconstructs_single_pass_output` FIRST, then port up to device.
3. If a future NKI benchmark shows `nl.exp`/`nl.log` is the bottleneck and a base-2 intrinsic exists on next-gen Neuron, coordinate the base flip across producer + `LSE_BASE_CONVENTION` + every combine site in one PR. Recall PR #35045 (0.105 → 0.0005 max error) — mismatches are silent.
4. If KDA + DSA cross-layer merges ever combine attention outputs from different attention types (unlikely, but the Flash architecture doc §5 leaves room for this in the model), the LSE base MUST match across both kernels or a manual base conversion (multiply by `ln(2)` / `1/ln(2)`) must happen at the merge boundary.

## 8. References

- SGLang PR #31821 "[Feature] Decode context parallelism (DCP) for DSA models (DeepSeek V3.2, GLM-5.x)" — https://github.com/sgl-project/sglang/pull/31821 (OPEN)
- SGLang PR #35045 "fix(attention): convert flashinfer base-2 LSE to natural log in chunked-prefix MLA merge" — https://github.com/sgl-project/sglang/pull/35045 (OPEN)
- SGLang PR #14194 "[feature] implement dcp for deepseek_v2" — https://github.com/sgl-project/sglang/pull/14194 (MERGED 2026-06-25) — the MLA DCP baseline PR #31821 extends
- SGLang HEAD `python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py` — TileLang DSA kernel source (lines 973, 1052, 1868, 2011 all use base-2 LSE convention)
- SGLang HEAD `python/sglang/srt/layers/dcp/comm.py` — `cp_lse_ag_out_rs_mla` docstring: "FlashInfer MLA returns base-2 LSE, while FlashMLA returns natural-log LSE."
- SGLang HEAD `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py` — `is_mla_dcp_lse_base_on_e()` — per-backend base convention flag
