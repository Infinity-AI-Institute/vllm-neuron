# GLM-5.3-Flash NxDI wrapper — Round 4 status (2026-08-28)

Local path: `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\NXDI-WRAPPER-ROUND4-STATUS-2026-08-28.md`
Branch: `codex/glm53-flash-enablement` (local commits only, not pushed)
Compile host: `ec2-user@13.222.20.119` (r7i.12xlarge, 48 vCPU, 371 GiB RAM, **no Neuron device** — dry-run tracing + `neuronx-cc` only, hence `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` everywhere)
Container: `public.ecr.aws/neuron/pytorch-inference-neuronx@sha256:011d49c7495457fc2932dedd3fbecf67d28833a3f12c147377cee4d72889ebc1`
NxDI: `/mnt/compile/shared-models/src/nxdi-e05466c` (pinned `e05466c657dda846f860083493dc18436788d969`)

## Headline

**The 4-layer coverage smoke passes.** KDA + DSA + 288-expert routed MoE in one
traced graph at TP=16 / LNC=2, CTE *and* TKG HLOs generated, `Compiler status
PASS` in 58.97 s. Round 3 aborted here with `double free or corruption`.

**KDA state now persists across decode steps, and there is a test that fails
when it does not.** Two independent levels, each with its own negative control.

Getting there took five distinct root causes, only one of which was the
blocker Round 3 named. They are listed in "Root causes" below because four of
them are the kind that would otherwise be rediscovered.

**Two blockers found in Round 4 that Round 3 did not know about**, both of
which change what may honestly be fired — see "The CTE wall".

## Results

| gate | result |
|---|---|
| KDA torch-vs-numpy golden parity | PASS — 0.0 max abs err, seeds 0–3 |
| **KDA persistence, kernel level** | **PASS** — 2×1-token == 1×2-token bit-exact (0.0); zero-state restart differs by 5.4e-2 |
| DSA golden traceability edit is a no-op | PASS — bit-exact vs reconstructed pre-edit golden, both regimes |
| MoE dispatch identity (Tier-1 CPU battery) | PASS |
| Tiny CPU model builds (TP=1) | PASS |
| State wiring (alias list ↔ layer slices) | PASS |
| **Model persistence, all state** | **PASS** — 2.06% relative logit delta vs zero-state; repeat-run delta 0.0 |
| **Model persistence, KDA state only** | **PASS** — 0.81% relative logit delta; repeat-run delta 0.0 |
| 1-layer KDA+dense trace, TP=16 | PASS — `Compiler status PASS`, 13.05 s |
| 2-layer KDA+DSA trace, TP=16 | PASS — 26.47 s |
| **4-layer KDA+DSA+MoE trace, TP=16, s=128** | **PASS — 58.97 s** |
| 4-layer trace, s=256 / s=512 | PASS — 63.65 s / 73.16 s |
| **4-layer KDA+DSA+MoE trace, TP=32, s=128** | **PASS — 59.83 s** |
| Real 45-layer TKG HLO, s=2048 | PASS — 5.01 s |
| Real 45-layer CTE HLO, s=2048 | PASS — 709.12 s, ~87 GB RSS |
| Real 45-layer TKG NEFF, **TP=16** | **FAIL — `NCC_EVRF009`, 39.6 GB needed vs 24 GB** |
| Real 45-layer NEFF, TP=16, CTE+TKG traced | FAIL — same `NCC_EVRF009`, byte-identical 39,644,174,584 |

Receipts on the host under `/mnt/compile/runroot/glm53-round4/`:
`glm53-round4-<tag>.json` per run, `logs/<tag>.log`, and preserved
`neuronx-cc` logs under `logs/cc-<tag>/`.

## Blocker 1 — MoE dispatch: routed branch now runs on NxDI's `ExpertMLPs`

Round 3's `moe_gather_dispatch_torch` had the right FLOPs (O(top_k), not
O(288)) and the wrong memory: it gathered a full expert weight slab *per
token*, `[B*L*top_k, hidden, inter]` — ~2.1 GB per slab at 128 prefill tokens,
~34 GB at 2048. That is what aborted the tracer's allocator.

`_MoEBlock` now holds a `neuronx_distributed.modules.moe.expert_mlps.ExpertMLPs`
and lets it own the capacity dispatch. It picks its own inference path from the
token count (`expert_mlps_v2.py:1407-1500`):

* **TKG** (`seq_len == 1`): `T*top_k/E = 8/288 < 1.0` → `forward_selective_loading`,
  which loads only the 8 chosen expert slabs. Correct decode behaviour.
* **CTE** (`seq_len > 1`, `T*top_k >= block_size = 512`): `forward_blockwise`,
  i.e. the fused NKI kernel that `use_shard_on_intermediate_dynamic_while`
  selects. At the s=128 smoke, `128*8 = 1024 >= 512`, so the coverage smoke
  really does exercise the blockwise kernel and not a torch fallback.

Three GLM specifics had to be mapped onto NxDI's contract rather than
re-implemented:

* **SwiGLU with clamp.** GLM's `silu(clamp(gate, max=L)) * clamp(up, -L, L)`
  maps exactly onto `GLUType.GLU` + `hidden_act="silu"` +
  `hidden_act_scaling_factor=1` + the four clamp limits. Not an approximation:
  it is the same expression `moe.py` spells.
* **`norm_topk_prob`** → `normalize_top_k_affinities=True`, so NxDI's
  `get_expert_affinities_masked` does the L1 renormalisation over the selected
  experts. `glm53_route_affinities` therefore returns the *full-width raw
  sigmoid scores*, deliberately un-normalised.
* **`routed_scaling_factor`** is applied to the ExpertMLPs **output**, not to
  the affinities. Pre-scaling the affinities would be cancelled by that same
  L1 normalise; the expert combination `Σ_e a_e·MLP_e(x)` is linear in `a`, so
  scaling the output is exactly equivalent. This one is easy to get wrong in
  a way that silently changes the model's effective temperature.
* **The selection-only correction bias** is unchanged from Round 3: it is added
  to the top-k selection score and never to the returned affinities.

The TP reduce is now explicit. `Experts` constructs `down_proj` with
`reduce_output=False` (`experts.py`), and NxDI's own `MoE.forward` does a
delayed all-reduce afterwards (`modules/moe/model.py:238-245`). `_MoEBlock` is
not that class, so it performs the reduce itself — and adds the shared expert
*after* it, because the shared expert's `RowParallelLinear` already reduced.

## Blocker 2 — KDA state persistence, and the DSA cache Round 3 did not have

### What was actually wrong

Round 3's handoff said KV aliasing was already solved and only KDA needed the
same treatment. Reading the code, that was not the case:

* `Glm53FlashLayer.forward` called `self.self_attn(normalized, position_ids,
  key_lengths=...)` with **no cache argument at all**, and `_DSABlock.forward`'s
  `kv_cache` parameter defaulted to `None` on every call.
* The model's forward read `kv_mgr.past_key_values` and returned them
  **unchanged**. That satisfies the lowering contract and makes every alias a
  no-op write.

So the aliases existed and carried nothing. A TKG graph would have restarted
from zero KDA state *and* attended to a one-position DSA context, every step,
while compiling and benchmarking perfectly.

### The fix

`kv_mgr` is set to `None` and the model owns `self.past_key_values` directly —
the documented second branch of `DecoderModelInstance.get()`
(`model_wrapper.py:1614-1619`). `KVCacheManager` cannot describe this model:
34 of 45 layers are KDA, whose state is a fixed `[HV, V, K]` matrix plus a
short-conv history and does not grow with sequence length, and the 11 DSA
layers need a *third* buffer — the lightning indexer's index-K — that no
attention-shaped manager knows about.

Per-layer aliased state, in graph input/output order:

| layer kind | count | tensors |
|---|---|---|
| KDA (34) | 2 | `kda_state [B, HV/tp, V, K]`, `conv_state [B, 3·qkv/tp, k−1]` |
| DSA (11) | 3 | `k_cache`, `v_cache`, `index_k_cache [B, S, 32, 128]` (replicated) |

Index-K is cached because the indexer scores a query against the index-K of
every *past* position. Round 3 recomputed it from the current window only —
right for prefill-from-zero, silently wrong for decode.

`key_lengths` for DSA is now derived from `position_ids`, not from
`attention_mask`. The mask describes the current *window*; the sparse gather
needs the number of valid *cache* positions. At decode the mask says 1 while
the cache holds `p+1`. The mask is still consulted as a tighter bound during
prefill, where a right-padded window spans more positions than it has tokens.

### `_sequence_carry` — the reset and the read are the same operation

Every aliased buffer is multiplied by a factor that is `0` when the window
starts at position 0 and `1` otherwise. That does two jobs:

1. **Correct reset.** A prefill starting at position 0 must not inherit the
   previous sequence's state. Deriving this from `position_ids` rather than
   from the static bucket length also handles a window that legitimately
   continues a sequence.
2. **A read the compiler cannot fold away.** This is the non-obvious half. The
   alias list is appended to `input_parameter_numbers` **without** the `-1`
   filter (`hlo_conversion.py:490-496`), so every aliased parameter must appear
   in the lowering context. Writing the reset as `state * 0` or
   `torch.zeros_like(state)` does **not** satisfy that — a literal-zero multiply
   is algebraically foldable and `zeros_like` reads only metadata, so in both
   cases the parameter can vanish and the trace aborts with the same
   "parameter not found in lowering context" error Round 3 already chased once.
   A runtime-derived factor survives to the lowered HLO by construction.

The first Round-4 KDA+dense trace died exactly this way. It is recorded here
because the error message points at in-place ops and `.data`, and the actual
cause is neither.

### The persistence test

Two levels, each with a negative control, in `smoke_round4.py`:

**Kernel level** (`kda_persistence_kernel`). Threading state through two
1-token calls must reproduce a single 2-token call bit-for-bit — measured
`0.0` for both the output and the state. Then the *negative control*:
restarting step 2 from zero state must produce a different answer — measured
`5.4e-2`. Without the control the test would pass on a graph that drops the
state whenever the state happened to be small.

**Model level** (`model_persistence`). A real `_NeuronGlm53FlashModel` is built
on CPU at TP=1 — genuinely GLM-5.3-shaped (All-NoPE MLA at head dim 256,
IndexPool=4 with tail selection, mHC×4 with 20 Sinkhorn steps, sigmoid routing
with correction bias; only the sizes shrink) — then prefilled and stepped
twice, with the aliased parameters written back between steps exactly as
`input_output_aliases` does on device. Step 2 is then re-run with the state
zeroed.

* **Positive control**: re-running step 2 against the *same* state must be
  bit-identical. Measured `0.0`. Without this, a nonzero zero-state delta could
  be nondeterminism rather than state dependence.
* **Negative control, all state**: 2.06% relative logit delta.
* **Negative control, KDA state only** (DSA K/V/index-K left intact): 0.81%
  relative logit delta. This is the one that matters — it proves the KDA half
  cannot pass on the strength of the attention cache.

A zero-state restart fails all three of these.

## Root causes fixed this round

Five, found by bisecting the trace across named layer recipes
(`SMOKE_RECIPES` / `build_recipe_smoke_config`, added so a failure attributes
to a *specific* bound kernel rather than to "the 4-layer smoke"):

1. **MoE dispatch memory blowup** — the named blocker. Fixed by `ExpertMLPs`.
2. **Aliased-but-unread state parameters** — see `_sequence_carry` above.
3. **`torch.topk(..., sorted=False)` is not compilable on trn2.** The router's
   top-k lowered to a full `sort`, and `neuronx-cc` refused it:
   `[NCC_EVRF029] Operation sort is not supported on trn2. Use supported
   equivalent operation like TopK ...` (exit 70, `hlo2penguin` /
   `NeuronHloVerifier.cc:724`). Every in-tree NxDI router calls `torch.topk` at
   the default `sorted=True`. The index order is irrelevant here — `ExpertMLPs`
   turns the indices into a top-k-hot mask and the combination is a sum.
4. **`torch.topk(...).indices` fails under the profiler dispatch wrapper.**
   `torch_neuronx`'s `custom_op_name.__torch_dispatch__` returns a plain list
   rather than a `return_types.topk` namedtuple, so attribute access raises
   `'list' object has no attribute 'indices'`. Positional unpacking works for
   both.
5. **`Tensor.any()` inside a Python `if` cannot be traced.** Three
   `if all_neg_inf.any():` guards in the DSA golden forced a data-dependent
   bool; torch_neuronx surfaces this as
   `ValueError: Unknown custom-call API version enum value: 0
   (API_VERSION_UNSPECIFIED)` and the DSA graph could not be lowered at all.
   See the next section.

Two latent Round-3 defects surfaced by actually executing paths that had never
run:

* `_DSABlock.forward` unpacked `dsa_attend_from_scores(..., return_lse=False)`
  as a 2-tuple. It returns one value. This never fired in Round 3 because no
  Round-3 smoke ever executed the DSA forward — the 1-layer smoke is KDA+dense
  and the coverage smoke aborted in the MoE branch first.
* `NXDI_EMIT_PHASES` was parsed and stored on `self._emit_phases` and **never
  acted on**, so `NXDI_EMIT_PHASES=TKG` silently compiled both graphs. NxDI has
  no config switch for this — `NeuronBaseForCausalLM` calls
  `enable_context_encoding()` unconditionally (`model_base.py:3062`) and
  appends each wrapper to `self.models`, which `compile()` iterates. Pruning
  that list is the intervention point; `_apply_emit_phases` now does it.

## The DSA golden edit — and why it is provably a no-op

Three `if all_neg_inf.any():` guards in
`harness-v2/staging/reference-sweep-20260826T2150Z/kernels/dsa_lightning_indexer.py`
were made unconditional. The first guard's body was literally `pass` (dead
code, deleted); the other two wrap a `masked_fill`, which is by definition a
no-op when the mask is all-False.

That argument is not taken on faith. `dsa_traceability_parity_check` in
`nki_bindings.py` reconstructs the **pre-edit** function by textually
re-inserting the guards into the golden's own source, executes it as a separate
module, and compares the two implementations on identical inputs — including a
`key_lengths = 0` row, the degenerate case the guards exist for. Measured:

```
normal_out_max_abs_err            0.0
normal_lse_identical              True
fully_masked_row0_out_max_abs_err 0.0
fully_masked_row0_lse_identical   True
fully_masked_row_max_abs_out      0.0   (the guard still fires)
```

The last line is deliberate: without it, the comparison could be vacuously
satisfied by two equally-broken implementations.

A note on what *not* to compare against: the first version of this check
asserted the sparse forward equals the golden's dense
`full_attention_reference` and "failed" at `1.19e-07`. That is fp32
reassociation between two different summation orders, not evidence of a
behaviour change. The before/after comparison is the right test.

## Blocker 3 — the HBM budget, recomputed

Round 3's `306 GiB / 16 ≈ 19 GiB/chip` does not hold. Derivation script:
`C:\Users\apumu\AppData\Local\Temp\claude\C--Users-apumu-research-InfinityAI-gemma4-trn2-handoff\fe22bc9a-b8c7-4107-ac55-1d55ba4bd33a\scratchpad\hbm.py`

**Parameter inventory** (analytic, from the frozen config):

| group | dtype in checkpoint | params |
|---|---|---|
| routed experts (42 layers × 288 × gate/up/down) | FP8 | 304.406 G |
| shared experts | FP8 | 1.057 G |
| MLA `o_proj` / `q_b` / `q_a` / `kv_a` | FP8 | 1.107 G |
| dense MLP (layers 0–2) | FP8 | 0.453 G |
| **FP8 subtotal** | | **307.023 G** |
| KDA block (34 layers) | BF16 | 4.683 G |
| embed + lm_head | BF16 | 1.269 G |
| MLA `kv_b` | BF16 | 0.185 G |
| indexer K / Q | BF16 | 0.370 G |
| router / mHC / norms | BF16 | 0.086 G |
| **BF16 subtotal** | | **6.591 G** |
| **total** | | **313.614 G** |

Cross-check against ground truth: the analytic on-disk size (FP8 experts +
bf16 rest + fp32 `weight_scale_inv` at one per `[128,128]` block) is
**320.3 GB**, versus `model.safetensors.index.json`'s
`metadata.total_size = 328,326,771,576` = **328.3 GB**. −2.5%, which is the
residual from small tensors not itemised. The inventory is sound.

**HBM residency at TP=16:**

| scenario | total | per chip |
|---|---|---|
| Round-3 assumption (experts stay FP8 in HBM) | 298.21 GiB | **18.64 GiB** |
| **What this round actually emits (all weights bf16)** | 584.15 GiB | **36.51 GiB** |

The correction is **1.96×**. Two independent reasons the experts are bf16 in
HBM, not FP8:

1. `blockwise.py:~958` — a weight scale with more than 2 dims, which is exactly
   what a `weight_block_size=[128,128]` scale is, triggers
   `blockwise_scale_dequantize(...)` up to `hidden_states.dtype` with both
   scales set to `None`, *before* the expert matmul.
2. More decisively for this round: `ExpertMLPs` builds its
   `ExpertFusedColumnParallelLinear` / `ExpertFusedRowParallelLinear` at
   `dtype=bf16` with no quantisation wired, and `convert_hf_to_neuron_state_dict`
   dequantises the FP8 checkpoint at load. So the FP8 checkpoint buys disk and
   load bandwidth, not HBM residency or matmul throughput.

### TP=16 does not fit, and the compiler says so in the same numbers

The plan's TP=16 was fired and **neuronx-cc rejected it**:

```
[NCC_EVRF009] Size of total input and output tensors exceeds HBM limit of
Trainium2. Needed 39,644,174,584 bytes (36 GB) vs. available
25,769,803,776 bytes (24 GB).
Top three largest input tensors are:
  input200 of shape bf16[288,4096,256]
  input228 of shape bf16[288,4096,256]
  input256 of shape bf16[288,4096,256]
```
(`/mnt/compile/runroot/glm53-round4/logs/cc-real45-tkg-fire/`)

Three things this confirms at once:

* **The arithmetic above is right.** 39.644 GB measured vs 39.20 GB predicted
  for the weights plus 0.28 GB of state cache — within 1%.
* **The experts really are bf16 in HBM.** The three largest tensors are
  `bf16[288, 4096, 256]`, i.e. the fused `gate_up_proj` slab at TP=16
  (`intermediate*2/tp = 4096/16 = 256`). Had they stayed FP8 this would read
  `f8e4m3[...]` and be half the size.
* **The budget is 24 GB per LNC=2 logical core, not 96 GB per chip.**
  trn2.48xlarge is 16 chips × 8 NeuronCore-v3, paired by LNC=2 into 64 logical
  cores; 1.5 TB / 64 = 24 GB. A TP=16 job therefore addresses 16 of 64 logical
  cores — a quarter of the box's HBM — not all of it. This is the single most
  load-bearing number in the round and it is easy to get wrong by reasoning
  per-chip.

So the Round-3 figure was not a harmless bookkeeping error: at 18.64 GiB/chip
the model would have fit at TP=16, and at the true 36.51 GiB/chip it does not.

A second TP=16 run that traced CTE *and* TKG reported the **byte-identical**
`39,644,174,584`, which pins the cause: this is the resident weight set, not a
phase-specific activation peak. Prefill and decode fail at TP=16 for the same
reason and are fixed by the same change.

| TP | bf16 weights per logical core | fits under 24 GB? |
|---|---|---|
| 8 | 78.40 GB | no |
| **16** | **39.20 GB** | **no — confirmed by neuronx-cc** |
| 32 | 19.60 GB | yes, ~4 GB headroom |
| 64 | 9.80 GB | yes |

**The contract moves to TP=32** (or 64). All the divisibility gates still hold:
`moe_intermediate 2048/32 = 64` clears the `%16` Tier-1 wall, `num_attention_heads
64/32 = 2`, `KDA num_heads 64/32 = 2`, and the indexer's pooled head count is
replicated regardless.

This is now a **pre-fire gate** in the compile driver rather than a lesson: see
`/mnt/compile/shared-images/glm53-flash-command.sh`, which computes
`313.614 G × 2 B / TP` against a 90%-of-24 GB budget and refuses with the
smallest TP that fits. It costs a second; finding out from neuronx-cc costs the
whole trace.

**Hybrid state cache at B=1, S=2048, bf16, per chip:**

| | size |
|---|---|
| per KDA layer | 128.0 KiB state + 9.0 KiB conv |
| per DSA layer | 4.0 MiB K + 4.0 MiB V + **16.0 MiB index-K (replicated)** |
| **total (34 KDA + 11 DSA)** | **268.5 MiB = 0.262 GiB** |

### Why FP8-KV was deliberately NOT wired

The 4-requirement FP8-KV pattern (`kv_cache_quant=True` + a
`KVQuantizationConfig` **instance** at `NeuronConfig(...)` init +
`XLA_HANDLE_SPECIAL_SCALAR=1` + `UNSAFE_FP8FNCAST=1`) exists to halve a
multi-GiB KV cache. Here it would not apply and would actively mislead:

* **It has no effect on this model's cache.** NxDI's KV quantisation lives
  inside `KVCacheManager`, which this model replaces. Setting `kv_cache_quant`
  would emit a `neuron_config.json` advertising `float8_e4m3fn` KV while the
  actual aliased tensors stay bf16 — a requested-vs-emitted mismatch, which is
  exactly the artifact class this campaign refuses to publish.
* **The saving would be 0.15% of HBM.** The whole hybrid cache is 268.5 MiB per
  chip, because 34 of 45 layers are linear-attention and carry no per-position
  cache at all. Halving it saves 134 MiB against ~89 GiB.

The two env vars are still set on `docker run` (they are inert without FP8
casting), so the difference between this model and the Llama-3.3-70B-FP8
driver is one deliberate omission, documented here rather than silently
inherited.

## Verified emitted, not requested

From the emitted `neuron_config.json`
(`/mnt/compile/runroot/glm53-round4/artifacts/real45-tkg-fire/`), read back
rather than assumed:

| field | emitted value | why it matters |
|---|---|---|
| `blockwise_matmul_config.use_shard_on_intermediate_dynamic_while` | `True` | survived `MoENeuronConfig` construction all the way into the artifact — the container workaround is real, not merely requested |
| `blockwise_matmul_config.use_torch_block_wise` | `False` | no silent torch fallback in the expert path |
| `blockwise_matmul_config.logical_nc_config` | `LNC_2` | the only branch with a non-raising kernel |
| `kv_cache_quant` / `kv_quant_config` | `False` / `None` | matches the bf16 tensors the model actually allocates — no float8 claim without float8 storage |
| `quantized` | `False` | consistent with the bf16 expert weights the compiler reported |

`float8_e4m3fn` is **absent** from the emitted config. That is the correct
result for this round, not a miss — see "Why FP8-KV was deliberately NOT
wired".

## The CTE wall — two prefill-scaling limits found in Round 4

Both are properties of the **torch reference bindings**, not of NxDI, and both
are invisible at the smoke's s=128.

### 1. KDA prefill is an unrolled per-token scan

`kda_state_forward_torch` is a Python loop over the bucket length, so a CTE
graph unrolls `num_kda_layers × seq_len` recurrence steps. Measured CTE HLO
generation for the 4-layer recipe (3 KDA layers):

| seq | CTE HLO gen |
|---|---|
| 128 | 4.69 s |
| 256 | 8.38 s |
| 512 | 15.95 s |

Linear at ~0.0104 s per KDA-layer-token, so the real 45-layer model at s=2048
projects to ≈ 34 × 2048 × 0.0104 ≈ 12 min of HLO generation alone, before
`neuronx-cc` sees a graph with ~70,000 unrolled recurrence steps.

**Measured on the real 45-layer model at s=2048: 709.12 s (11.8 min) of CTE
HLO generation**, peaking at ~87 GB RSS. The projection from the 4-layer curve
was accurate to 2%, so the scan cost is genuinely linear in
`num_kda_layers × seq_len` and can be planned against.

TKG (L=1) is one step per layer and is unaffected — **5.01 s** for the full
45-layer model, and 0.46–0.54 s for the 4-layer recipe across every bucket
measured. The decode graph is not where this cost lives.

### 2. The DSA sparse gather is O(Q × topk)

`dsa_sparse_attention_forward` materialises the gathered K and V at
`[B, Q, topk, H_local, D]`. With `index_topk = 2048` clamped to the context
length:

| CTE Q | gathered K+V, **per DSA layer** |
|---|---|
| 128 | 0.062 GiB |
| 256 | 0.250 GiB |
| 512 | 1.000 GiB |
| 1024 | 4.000 GiB |
| **2048** | **16.000 GiB** |
| TKG Q=1 | 8.0 MiB |

This is the same failure *class* as the Round-3 MoE gather, relocated to DSA,
and it grows quadratically. At s=2048 it is 16 GiB of transient activation per
DSA layer during tracing.

**Consequence for the fire.** The TKG (decode) contract is clean at every
measured shape and is the graph that determines tokenomics. The CTE contract at
`max_model_len=2048` is where both limits bite. The two are fired and reported
separately rather than as one number, and a CTE bucket is chosen from
measurement rather than from the plan.

Neither limit is in NxDI or the compiler — both are properties of the torch
reference bindings, so both are fixable in Round 5 without touching the stack:

* **KDA prefill** wants the chunked-parallel delta-rule form (process the
  sequence in chunks of C with an intra-chunk parallel scan), which turns
  `O(seq_len)` unrolled steps into `O(seq_len / C)`. That is a
  correctness-critical rewrite and needs its own bit-exactness gate against
  the numpy golden, exactly like the per-token port got this round.
* **DSA prefill** wants the gather replaced by a masked score-then-attend that
  never materialises `[B, Q, topk, H, D]` — at `topk >= L` the selection is
  the degenerate dense case anyway, so a prefill-specific path is available
  without changing the model.

## Compile driver

`/mnt/compile/shared-images/glm53-flash-command.sh` is authored, contract-driven,
and mirrors `llama33-70b-fp8-command.sh`'s structure. Three deliberate
differences, each documented in its header rather than silently inherited:

1. **Model class.** GLM-5.3-Flash has no in-tree NxDI modeling module; the
   driver bind-mounts this campaign's wrapper (`/code`) and the qualified CPU
   goldens (`/kernels`) read-only and drives them directly.
2. **`MoENeuronConfig`, not `NeuronConfig`**, with an assertion that
   `blockwise_matmul_config.use_shard_on_intermediate_dynamic_while` survived
   construction, plus a hard refusal of `LNC != 2`.
3. **FP8-KV keys are refused, not ignored.** A contract that sets
   `kv_cache_quant` / `kv_quant_config` / `fp8_packed_kv` exits non-zero with an
   explanation, because honouring them would change `neuron_config.json` and
   not the tensors.

It also carries the two gates this round paid for:

* **HBM preflight** — `313.614 G × 2 B / TP` against 90% of 24 GB, refusing
  with the smallest TP that fits. This is what would have caught the TP=16
  fire in one second instead of one full trace.
* **CPU-fallback scan** — the compile log is grepped for
  `falling back to cpu`, `torch_blockwise_matmul_inference`,
  `use_torch_block_wise=True` and friends, and a hit *fails* the fire rather
  than warning. This is the failure mode that got Gemma-4 deferred.

Receipts written: `effective-shape.json` (resolved config, including
`models_compiled` so an emit-phase restriction is visible), `compile-result.json`
(emitted NEFF paths + sizes + emitted dtype flags), `terminal.json`.

## Files

| artifact | absolute path |
|---|---|
| wrapper | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\neuron_wrapper.py` |
| kernel bindings | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\nki_bindings.py` |
| Round-4 smoke driver | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\smoke_round4.py` |
| checkpoint converter | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\checkpoint_convert.py` |
| config | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\config.py` |
| DSA golden (edited) | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\dsa_lightning_indexer.py` |
| HBM derivation | `C:\Users\apumu\AppData\Local\Temp\claude\C--Users-apumu-research-InfinityAI-gemma4-trn2-handoff\fe22bc9a-b8c7-4107-ac55-1d55ba4bd33a\scratchpad\hbm.py` |
| Round-3 status | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\NXDI-WRAPPER-ROUND3-STATUS-2026-08-28.md` |
| this doc | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\NXDI-WRAPPER-ROUND4-STATUS-2026-08-28.md` |
| host: runner | `/mnt/compile/src/vllm-neuron-alpha/run-round4-smoke.sh` |
| host: receipts | `/mnt/compile/runroot/glm53-round4/` |
