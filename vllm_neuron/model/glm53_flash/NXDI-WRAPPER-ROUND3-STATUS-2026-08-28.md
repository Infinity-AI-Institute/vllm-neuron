# GLM-5.3-Flash NxDI wrapper — Round 3 status (2026-08-28)

Local path: `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\NXDI-WRAPPER-ROUND3-STATUS-2026-08-28.md`
Branch: `codex/glm53-flash-enablement` · commit `c6e10b0` (local only, not pushed)

## Headline

**Smoke = FAIL at stage 6 (dry-run compile). Stages 1–5 PASS.** The failure is
real, reproducible, and localized; the traceback is below. No NEFF was produced,
no compile was fired, and nothing here claims a result it did not measure.

The single most valuable result of this turn is independent of the compile:
**the KDA torch port is bit-exact against the numpy CPU golden (max abs error
0.0 across 4 seeds)**, measured inside the NxDI container on the compile host.

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
| 6. dry-run compile | **FAIL** |

### Stage-6 traceback (verbatim, trimmed to the load-bearing frames)

```
File ".../neuronx_distributed_inference/models/application_base.py", line 305, in compile
    traced_model = self.get_builder(debug).trace(**trace_kwargs)
File ".../neuronx_distributed/trace/model_builder.py", line 904, in _generate_hlo
    hlo_artifacts = torch_neuronx.xla_impl.trace.generate_hlo(
File ".../torch_neuronx/xla_impl/hlo_conversion.py", line 545, in _xla_trace
    ) = hlo_opt.linearize_indices(
File ".../torch_neuronx/xla_impl/hlo_conversion.py", line 710, in linearize_indices
    raise ValueError(
ValueError: Unable to lower HLO: parameter not found in lowering context. This is
likely caused by an attempted in-place operation, or an attempted access of
nn.Parameter.data or nn.Buffer.data. These operations are not currently supported.
```

### Diagnosis

`linearize_indices` raises this exactly when `tensor_parameter_id(tensor)`
returns `-1` for one of the **example inputs**:

```python
# torch_neuronx/xla_impl/hlo_conversion.py:683-693
# NOTE: parameter_number = -1 when a tensor cannot be found in the lowering context
if parameter_number == -1:
    raise ValueError("Unable to lower HLO: parameter not found in lowering context. ...")
```

I probed what NxDI actually generates for this config. **Seven example inputs**,
for both CTE and TKG:

| # | CTE shape | TKG shape | dtype | identity |
|---|---|---|---|---|
| 0 | (1,128) | (1,1) | int32 | `input_ids` |
| 1 | (1,128) | (1,128) | int32 | `attention_mask` |
| 2 | (1,128) | (1,1) | int32 | `position_ids` |
| 3 | (1,) | (1,) | int32 | `seq_ids` |
| 4 | (1,3) | (1,3) | float32 | `sampling_params` |
| 5 | (1,) | (1,) | int32 | `num_queries` (continuous batching) |
| 6 | (1,) | (1,) | int32 | `computed_context_lens` (continuous batching) |

The Round-3 graph consumes **two** of the seven (`input_ids`, and
`position_ids` only on DSA layers — the 1-layer smoke is KDA-only, so in
practice just `input_ids`). The remaining inputs never enter the lowering
context, which is the `-1`.

This is a *wiring* gap, not a kernel or numerics defect: the model math traced
end-to-end successfully (parallel state initialised, `context_encoding_model`
HLO generation started and ran through the whole 1-layer forward) before
lowering rejected the unused parameters.

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

## Next blocker (single, specific)

**Wire the five unconsumed NxDI graph inputs**, in priority order:

1. `attention_mask` — should drive `key_lengths` for the DSA sparse gather and
   mask padded tokens. This is a real correctness improvement, not just an
   anchor: the DSA block currently assumes `key_lengths == context_len`.
2. `seq_ids` — selects the KDA state slot. This is the same work as the item
   below and should land with it.
3. `num_queries` / `computed_context_lens` — the continuous-batching contract.
   Setting `is_continuous_batching=False` removes them from the generated input
   list and is the cheaper path to a first NEFF.
4. `sampling_params` — only meaningful with on-device sampling, currently off.

## Second blocker, already visible and not yet addressed

**KDA state does not survive across decode steps.** The recurrence hands its
updated state back on plain Python attributes because buffer mutation inside
forward is exactly what torch_neuronx refuses to lower. Making state persist on
device needs NxDI's `input_output_aliases` wireup, so the state tensor becomes a
real graph input/output pair.

Consequence, stated plainly: **a CTE (prefill-from-zero) graph is exact, but a
multi-step TKG graph would silently restart from zero state every step.** The
TKG contract must not be declared correct until the aliasing lands. This is why
no TKG/CTE contract was authored or fired this turn — firing a TKG compile now
would produce an artifact that benchmarks fine and is wrong.

## Not done (and why)

- **Stage 4 (TKG + CTE contracts, TP=16/LNC=2, fire)** — gated on the smoke,
  which failed. Firing would have been fabricating progress.
- `/mnt/compile/shared-images/glm53-flash-command.sh` — not authored, for the
  same reason. The FP8-KV 4-requirement pattern to mirror is confirmed present
  in `/mnt/compile/shared-images/llama33-70b-fp8-command.sh` (pop
  `kv_cache_quant` / `kv_quant_config` out of the extras dict *before* the
  `NeuronConfig` ctor so the post-init `setattr` loop cannot re-poison the FP8
  surface; `KVQuantizationConfig` must be an instance, not a dict;
  `XLA_HANDLE_SPECIAL_SCALAR=1`; `UNSAFE_FP8FNCAST=1`).

## Verified environment facts

- `use_shard_on_intermediate_dynamic_while` **exists** in NxDI `e05466c`
  (`modeling_qwen3_moe.py:276`) and in the container's `neuronx_distributed`
  (`moe_configs.py:56,72`). The workaround is valid for this rev.
- `_call_shard_hidden_kernel` is **absent** everywhere — confirming the standing
  note. The container also warns at import:
  `Failed to import blockwise_mm_baseline_shard_hidden: No module named
  'neuronxcc.nki._private.blockwise_mm'`.
- `NeuronConfig` logs `Unexpected keyword arguments: {'blockwise_matmul_config': …}`,
  i.e. this rev does **not** accept it as a ctor kwarg on the base config —
  worth resolving before relying on the workaround at full MoE scale.
- Host disk: 574 GB free on `/` (shared with `/mnt/compile`); weights are 306 GB
  of the 1.5 TB used. `/tmp` is a 186 GB RAM-backed tmpfs — do not stage there.

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
