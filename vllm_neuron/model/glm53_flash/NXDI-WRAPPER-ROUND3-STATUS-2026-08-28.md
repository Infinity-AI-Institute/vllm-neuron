# GLM-5.3-Flash NxDI wrapper — Round 3 status (2026-08-28)

Local path: `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\NXDI-WRAPPER-ROUND3-STATUS-2026-08-28.md`
Branch: `codex/glm53-flash-enablement` · commit `c6e10b0` (local only, not pushed)

## Headline

**Smoke = PASS. All 8 stages green, `Compiler status PASS`.** The 1-layer
GLM-5.3-Flash graph traces, lowers, and compiles under NxDI `e05466c` in
container `sha256:011d49c7`.

**The 4-layer coverage smoke that adds the DSA and MoE paths FAILS** (native
abort, `double free or corruption`, during CTE HLO generation). Root-caused to
my own token-major MoE dispatch materialising ~6 GB of expert slabs at 128
tokens — see "Coverage smoke" below. So: KDA + dense compiles; DSA + routed-MoE
does not yet.

Two results worth separating from the compile itself:

- **The KDA torch port is bit-exact against the numpy CPU golden** — max abs
  error `0.0` across 4 seeds, measured inside the container on the compile host.
- **The lowering failure that blocked this for most of the turn was misdiagnosed
  by its own error message.** See "Root cause" below; the fix is three lines and
  the wrong reading would have cost far more.

## Smoke result

Driver: `.../glm53_flash/smoke_round3.py`
Host: `ec2-user@13.222.20.119` (r7i.12xlarge — **no Neuron device**, so dry-run
tracing is the only thing possible here)
Container: `cffb98efd1c3` = digest `sha256:011d49c7495457fc2932dedd3fbecf67d28833a3f12c147377cee4d72889ebc1`
NxDI: `/mnt/compile/shared-models/src/nxdi-e05466c` (pinned rev `e05466c657dda846f860083493dc18436788d969`)
Receipt: `/mnt/compile/runroot/glm53-smoke/glm53-smoke-result.json`

| stage | result |
|---|---|
| 1. golden import (kda, dsa, moe) | PASS |
| 2. **KDA torch-vs-numpy parity** | **PASS — 0.0 max abs err** |
| 3. MoE dispatch identity (Tier-1 CPU battery) | PASS |
| 4a. source config | PASS |
| 4b. 1-layer reduced config | PASS |
| 4c. wrapper import | PASS |
| 5. wrapper construct | PASS |
| 6. dry-run compile | **PASS** — `Compiler status PASS`, HLOs saved |

Compile log evidence: HLOs generated for both `context_encoding_model` and
`token_generation_model` (1.52 s), priority HLO compiled in 14.77 s,
`Compilation Successfully Completed for model.MODULE_0f71aa9105a640304070+8f34053f`,
weight layout optimized, dry-run artifacts written.

## Coverage smoke (KDA + DSA + MoE in one graph) — FAIL, and the failure is useful

The passing smoke above is **1 layer of KDA + dense**. It says nothing about the
DSA or routed-MoE paths, so `build_kernel_coverage_smoke_config` traces the real
layer-0..3 prefix — 3 × KDA+dense, then layer 3 = DSA + 288-expert sparse MoE —
putting all three bound kernels in one graph.

**Result: native abort during CTE HLO generation.**

```
INFO:Neuron:generating HLO: context_encoding_model, input example shape = torch.Size([1, 128])
double free or corruption (!prev)
bash: line 1:     8 Aborted                 (core dumped) python .../smoke_round3.py
```

Receipt: `/mnt/compile/runroot/glm53-smoke/glm53-coverage-result.json`
(stages 1–5 PASS; the process aborted inside stage 6 before writing its result).

### Diagnosis: the routed-MoE dispatch does not scale to prefill

This is my own code, and the caveat was written into
`moe_gather_dispatch_torch`'s docstring before it was measured — now it is
measured. The dispatch is **token-major**: it gathers the `top_k` selected
expert weight slabs *per token*, materialising `[B*L*top_k, hidden, inter]`.

| shape | slab size (bf16, TP=8) |
|---|---|
| decode, `B*L = 1` | `8 × 4096 × 256` ≈ **16 MB** — fine |
| CTE, `B*L = 128` | `1024 × 4096 × 256` ≈ **2.1 GB per slab**, ×3 ≈ **6.4 GB** |
| CTE, `B*L = 2048` | ≈ **34 GB per slab** — hopeless |

The abort is the XLA tracer's allocator giving up on that. The asymptotics are
right (O(top_k), not O(288)), but the *constant* is a full weight-slab copy per
token, which prefill cannot afford.

### Fix (top code item for Round 4)

Stop hand-rolling the routed dispatch. Route it through NxDI's own
`ExpertMLPs` / blockwise-MoE module so the fused NKI kernel does the
capacity dispatch, which is what `moe_dispatch.py`'s `MoEDispatchConfig` and
`enable_moe_fused_dispatch` were always meant to drive — the config identity is
already built and Tier-1-validated at construction. The token-major gather
should survive only as the decode-path reference.

An expert-major masked loop is the obvious intermediate, but at 288 experts it
unrolls to 288 matmul pairs in the traced graph, so it trades an allocator abort
for a compile-time blowup. Not worth doing as a stepping stone.

## Root cause of the lowering failure (worth recording — the error message misleads)

For most of this turn stage 6 failed with:

```
ValueError: Unable to lower HLO: parameter not found in lowering context. This is
likely caused by an attempted in-place operation, or an attempted access of
nn.Parameter.data or nn.Buffer.data. These operations are not currently supported.
```

The message points at in-place ops and `.data`. **Neither was the cause**, and
the obvious second reading — unused example inputs — is also wrong. I probed
what NxDI generates and found a fixed **7-input** signature for both CTE and TKG
(`input_ids`, `attention_mask`, `position_ids`, `seq_ids`, `sampling_params`,
plus two more), of which this graph consumed two. That looked like the answer.
It was not: torch_neuronx **filters unused inputs into an `exclude` list with a
warning** (`hlo_conversion.py:465-485`) and they never reach `linearize_indices`.

The actual cause is the **KV-cache alias list**. `DecoderModelInstance.get()`
(`model_wrapper.py:1614-1619`) builds `input_output_aliases` from
`kv_mgr.past_key_values` — real `nn.Parameter`s — and that list is appended to
`input_parameter_numbers` **without** the `-1` filter
(`hlo_conversion.py:490-496`). A cache parameter that the graph aliases but
never reads therefore aborts lowering.

Every NxDI model in-tree sidesteps this by **not overriding `forward` at all**
(`NeuronQwen2Model`, `NeuronMistralModel`, and all 14 subclasses override only
`setup_attr_for_model` and `init_model`); the base `forward` reads the cache via
`kv_mgr.get_cache` and returns `outputs += updated_kv_cache`. GLM-5.3-Flash
needs its own `forward` — it is a hybrid KDA/DSA stack the base decode loop does
not model — so it now honours the same contract explicitly: read each aliased
cache parameter and return it directly after the logits, in alias order.

A useful negative result for the next agent: **do not spend time trying to
"anchor" unused inputs.** They are handled. If you see this error, look at the
alias list.

Cost of the misdiagnosis: roughly three failed smoke iterations. Recording it
here so the next hybrid-architecture port does not repeat it.

## What landed

### 1. `nki_bindings.py` (new) — the three kernel bindings

The Round-3 contract said "bind the CPU-golden references". Two of the three
goldens are **not directly bindable into a traced graph**, and saying so is part
of the deliverable:

| golden | impl | directly traceable? |
|---|---|---|
| `dsa_lightning_indexer.py` | torch | **yes** — called straight through |
| `kda_state_v2.py` | **numpy** | **no** — `np.einsum` on detached arrays is invisible to the XLA tracer |
| `moe_dispatch.py` | **`@nki.jit`** | **no CPU path at all** — it *is* the device kernel builder |

So:

- **KDA** — a line-for-line torch transcription of the numpy
  `_kda_delta_rule_step`, gated by `kda_reference_parity_check`, which runs the
  numpy golden and the torch port on identical inputs and returns the max abs
  error. **Measured 0.0 (bit-exact) for seeds 0–3.** All four FLA v0.5.2 parity
  pieces preserved, including the two that are easy to get subtly wrong:
  the L2-norm is spelled `x / sqrt(sum + eps)` and **not** `rsqrt` (they differ
  in the last mantissa bit), and the gate is written in the golden's
  `lower_bound / (1 + exp(-a_amp*g))` form rather than the algebraically equal
  `lower_bound * sigmoid(...)`, so the float rounding matches term-for-term.
  State layout is the vLLM one: `[num_slots, HV=64, V=128, K=128]`.
- **DSA** — the torch golden, split around a TP all-reduce. See "TP-correctness"
  below; this split is a correctness requirement, not an optimization.
- **MoE** — GLM routing (sigmoid scores, selection-only correction bias, so the
  bias moves *which* experts win but never leaks into the weights) plus an
  O(top_k) gather dispatch. `MoEDispatchConfig` is constructed and `validate()`d
  at module init, which runs the Tier-1 CPU battery (partition cap 288 < 16384,
  `I_TP = 2048/16 = 128` clears the `%16` wall, `top_k=8` in the tested set).

No-fallback discipline is enforced by `assert_impl_not_banned` on every entry
point; `softmax`/`full_attention`/`sdpa`/`flash_attn` are refused by name.

### 2. TP-correctness fixes to Round-2 declarations

Round 2's shapes were internally inconsistent in three ways that would each
have produced a model that runs and is quietly wrong:

- **KDA `conv1d`** was declared over the full `3 * qkv_dim` channels while
  consuming column-parallel Q/K/V, which only ever hold `qkv_dim_local`. A
  depthwise conv is per-channel, so the channel axis shards exactly like Q/K/V's
  output axis — now rank-local, no cross-rank communication needed.
- **MoE `gate`/`up`/`down`** were full-width replicated `nn.Parameter`s:
  `288 × 4096 × 2048 × 2 B = 4.8 GiB` *per slab, per layer, per rank*. Now
  sharded on the intermediate axis, with an explicit reduce on the routed output
  (the RowParallel contract the shared expert gets for free).
- **DSA indexer** used `index_n_heads % tp_degree` as its guard, which passes at
  TP=16 (32 % 16 == 0) but leaves 2 index heads per rank and then fails the
  IndexPool=4 collapse. The real constraint is pool divisibility. The index-K
  side is now `gather_output=True` (replicated) so the top-k is **identical on
  every rank** — ranks hold different head shards of the same KV and must gather
  the same sparse positions. The q-side contraction runs over the sharded
  main-attention head axis, so it is all-reduced before scoring; a per-rank
  partial top-k would select different positions per rank, which reads as mild
  quality loss rather than a crash.

### 3. Round-2 defects found by actually running the smoke

- `config.allow_reduced_shapes` gated only the `frozen` dict, never the
  layer-count checks — so `build_one_layer_smoke_config` **could not construct
  at all**. It sets `num_hidden_layers=1`, which tripped
  `"requires exactly 45 text layers"` before `allow_reduced_shapes` was read.
  Fixed, with a self-consistency check retained for the reduced path.
- `_NeuronGlm53FlashModel.forward` declared `(input_ids, positions, **kwargs)`
  against NxDI's 7 positional inputs — HLO generation could not even start.
- KDA state was a `persistent=False` buffer. Those are excluded from
  `state_dict()`, and NxDI derives the graph parameter set from the state dict,
  so reading one inside forward is itself a "not found in lowering context". A
  `persistent=True` buffer would instead demand a checkpoint tensor that does
  not exist. State is now materialised in-graph or passed as a real argument.

### 4. `checkpoint_convert.py` (new) — `convert_hf_to_neuron_state_dict`

Every name was read off the real `model.safetensors.index.json` for snapshot
`04c4e9e9…`; nothing is guessed. Facts that drove the implementation:

- 76,108 tensors; text prefix `model.language_model.`; `lm_head.weight` is the
  only key with no `model.` prefix; 347 vision tensors.
- **46 layer indices (0–45) but `num_hidden_layers = 45`. Layer 45 is the
  MTP/nextn module** — the only layer with `eh_proj`/`enorm`/`hnorm`/
  `shared_head` and the only one *without* `hc_*`. **Dropped**: MTP is a
  speculative-decode surface and the campaign forbids it.
- The only scale suffix that exists is **`weight_scale_inv`** (37,338 of them) —
  a *reciprocal* per-block scale under `weight_block_size=[128,128]`.
  `weight_scale` and `input_scale` are **absent** (activation scheme is
  `dynamic`, so no static activation scales are stored). Dequantization is
  `w * scale_inv`, not a divide.
- **The whole KDA block is BF16** — no KDA projection carries a scale. So is
  `kv_b_proj`, even though its MLA siblings `q_a_proj`/`q_b_proj`/
  `kv_a_proj_with_mqa`/`o_proj` all carry scales. "All MLA tensors are FP8" is
  wrong.
- `fused_qkvbfg_a_proj` and `qkv_proj` appear in the config's
  `modules_to_not_convert` but **do not exist as tensors**. KDA Q/K/V are
  separate, as are `q_conv1d`/`k_conv1d`/`v_conv1d` — concatenated on the
  channel axis into the wrapper's single depthwise conv (exact, because
  depthwise channels are independent).

**Two traps handled explicitly.** `hc_attn_scale` / `hc_ffn_scale` (45 each)
match a naive `*scale*` filter but are **hyper-connection parameters, not FP8
scales** — treating them as scales corrupts mHC. And the indexer tensors
(`wk`, `wq_b`, `k_norm.{weight,bias}`, `weights_proj`,
`index_kpool_compress_{ape,gate}`) carry no scales at all.

**Anti-inheritance held.** `normalize_static_fp8_weight_format()` is
deliberately *not* called — its OCP-448-when-`None` fallback is the defect this
port refuses to reproduce. Every scale field has an explicit non-`None` default
and a load-time `max(scale) <= 240.0` assertion via `validate_fp8_scale`.

## THE next blocker: KDA state does not survive across decode steps

This is the one thing standing between a passing smoke and a *trustworthy* TKG
artifact, and it is the reason no TKG compile was fired.

The KDA recurrence hands its updated state back on plain Python attributes,
because buffer mutation inside `forward` is precisely what torch_neuronx refuses
to lower. Making state persist on device needs NxDI's `input_output_aliases`
wireup — the same mechanism the KV cache uses, and the same one whose absence
produced the lowering bug above — so the state tensor becomes a real graph
input/output pair.

Stated plainly: **a CTE (prefill-from-zero) graph is exact. A multi-step TKG
graph would silently restart from zero KDA state every step.** It would compile,
run, and benchmark perfectly well while being wrong. That is exactly the failure
class this campaign refuses to ship, so the TKG contract must not be declared
correct until the aliasing lands.

The fix is well-scoped and the pattern is now known: register the KDA state as
an `nn.Parameter` per KDA layer (as `kv_cache_manager.py:152-162` does for the
KV cache), add it to `input_output_aliases`, read it at the top of the layer
forward and return it after the logits in alias order.

## Ordered next steps

1. ~~Resolve the `blockwise_matmul_config` rejection~~ — **done**, see above
   (`MoENeuronConfig`).
2. **Replace the hand-rolled routed-MoE dispatch** — see "Coverage smoke"
   below. This is now the top code item.
3. **Recompute the HBM budget** given that `[128,128]` block scales dequantize
   to bf16 in the expert path. The `306 GiB / 16 ranks ≈ 19 GiB/chip` figure in
   the Round-3 plan does not hold for the experts.
4. **Wire KDA state aliasing** (above). Blocks a correct TKG.
5. Then author `/mnt/compile/shared-images/glm53-flash-command.sh` and fire
   CTE + TKG.

## Not done (and why)

- **Stage 4 (TKG + CTE contracts, TP=16/LNC=2, fire)** — not fired. The smoke
  passes, but items 1–3 above are each sufficient on their own to make the
  resulting artifact wrong or un-compilable. Firing now would produce a slug
  and a NEFF path that look like progress and are not.
- `/mnt/compile/shared-images/glm53-flash-command.sh` — not authored, for the
  same reason. The FP8-KV 4-requirement pattern to mirror is confirmed present
  in `/mnt/compile/shared-images/llama33-70b-fp8-command.sh` (pop
  `kv_cache_quant` / `kv_quant_config` out of the extras dict *before* the
  `NeuronConfig` ctor so the post-init `setattr` loop cannot re-poison the FP8
  surface; `KVQuantizationConfig` must be an instance, not a dict;
  `XLA_HANDLE_SPECIAL_SCALAR=1`; `UNSAFE_FP8FNCAST=1`).
- **Correctness gate against reference logits** — out of scope for this turn by
  design; it gates after the NEFF lands.

## MoE blockwise findings — these change the memory plan

Three facts from reading the container's
`neuronx_distributed/modules/moe/blockwise.py` directly. All three are more
constraining than the standing workaround note implied.

### 1. `_call_shard_hidden_kernel` exists as an always-raising stub, and it is the *default* branch

The standing note says the symbol is missing. It is actually present and
unconditionally raises (`blockwise.py:267`):

```python
def _call_shard_hidden_kernel(args: BlockwiseMatmulArgs):
    raise NotImplementedError("_call_shard_hidden_kernel is not available - kernel not imported from nkilib")
```

The LNC=2 inference dispatch (`blockwise.py:1005-1017`) falls into it whenever
neither shard flag is set:

```python
elif logical_nc_config == 2:
    if use_shard_on_intermediate_dynamic_while:  ...
    elif use_shard_on_block_dynamic_while:       ...
    else:  output, ... = _call_shard_hidden_kernel(args)   # raises
else:
    raise NotImplementedError("LNC_1 kernels not available in nkilib")
```

So `use_shard_on_intermediate_dynamic_while=True` is **not a performance knob —
it is the only way to avoid a dead default branch.** There is a second escape
hatch worth holding in reserve: `use_shard_on_block_dynamic_while=True`
(`_call_bwmm_shard_on_block_kernel`), a real non-stub implementation. The two are
mutually exclusive (`assert` at `blockwise.py:927`).

### 2. LNC=2 is mandatory for any MoE path on this container

The `else` branch raises `"LNC_1 kernels not available in nkilib"`. The planned
TP=16 / LNC=2 contract is therefore forced, not chosen.

### 3. GLM's `[128,128]` FP8 block scales get DEQUANTIZED to bf16 before the expert matmul

At `blockwise.py:~958`, a weight scale with more than 2 dims — which is exactly
what a `weight_block_size=[128,128]` scale is — triggers:

> "Blockwise scaling is not supported in blockwise kernel for now and will be
> dequantized before the kernel"

followed by `blockwise_scale_dequantize(...)` up to `hidden_states.dtype` and
both scales set to `None`.

**Consequence, and it is a big one: on this container rev the FP8 checkpoint
buys disk and load bandwidth but NOT HBM residency or matmul throughput in the
expert path.** The 306 GiB FP8 weight set does not imply 306 GiB resident —
the MoE experts, which are the overwhelming bulk of the model, land in HBM as
bf16. Any TP=16 HBM budget that assumed `306 / 16 ≈ 19 GiB/chip` is wrong for
the expert path and must be recomputed before firing. This is the top thing to
verify before the full compile, and it is *measurement-pending*, not settled.

## Verified environment facts

- `use_shard_on_intermediate_dynamic_while` exists in NxDI `e05466c`
  (`modeling_qwen3_moe.py:276`) and in the container's `neuronx_distributed`
  (`moe_configs.py:56,72`). Canonical usage form, from
  `examples/generation_qwen3_moe_demo.py:43`:
  `blockwise_matmul_config={"use_shard_on_intermediate_dynamic_while": True, "skip_dma_token": True}`.
- The container warns at import:
  `Failed to import blockwise_mm_baseline_shard_hidden: No module named
  'neuronxcc.nki._private.blockwise_mm'` — the nkilib module behind the stub
  simply was not shipped in this build.
- **RESOLVED**: `NeuronConfig` was logging `Unexpected keyword arguments:
  {'blockwise_matmul_config': …}` and **silently dropping the flag** — which,
  per finding 1, would have driven the MoE compile straight into the raising
  stub. Cause: `blockwise_matmul_config` is popped and frozen at
  `models/config.py:837-839`, which is inside **`MoENeuronConfig`** (declared
  `:798`, next class `:849`) — *not* the base `NeuronConfig` (`:84`).
  `build_neuron_config` now constructs `MoENeuronConfig` and asserts the flag
  survived construction, so a future regression fails at config-build time
  instead of deep inside the compile.
- `skip_dma_token` changes the input-padding path
  (`augment_inputs_for_padded_blockwise_matmul`), not only DMA behaviour.
- Host disk: 574 GB free on `/` (shared with `/mnt/compile`); weights are 306 GB
  of the 1.5 TB used. `/tmp` is a 186 GB RAM-backed tmpfs — do not stage there.
- Host is `r7i.12xlarge` with **no Neuron device**, hence
  `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` on every invocation.

## Paths

| artifact | absolute path |
|---|---|
| kernel bindings | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\nki_bindings.py` |
| checkpoint converter | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\checkpoint_convert.py` |
| wrapper | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\neuron_wrapper.py` |
| smoke driver | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\smoke_round3.py` |
| config (fixed) | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\config.py` |
| this doc | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\NXDI-WRAPPER-ROUND3-STATUS-2026-08-28.md` |
| goldens (source of truth) | `C:\Users\apumu\research\InfinityAI\gemma4-trn2-handoff\harness-v2\staging\reference-sweep-20260826T2150Z\kernels\` |
| host: kernels | `/mnt/compile/src/glm53-kernels/` |
| host: code | `/mnt/compile/src/vllm-neuron-alpha/vllm_neuron/model/glm53_flash/` |
| host: smoke receipt | `/mnt/compile/runroot/glm53-smoke/glm53-smoke-result.json` |
