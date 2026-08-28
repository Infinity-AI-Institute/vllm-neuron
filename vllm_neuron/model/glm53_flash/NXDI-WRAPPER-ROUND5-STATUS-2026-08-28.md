# GLM-5.3-Flash NxDI wrapper — Round 5 status

Full path on the campaign box:
`C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\NXDI-WRAPPER-ROUND5-STATUS-2026-08-28.md`

Follows `NXDI-WRAPPER-ROUND4-STATUS-2026-08-28.md`; that document ended with
the concrete blocker: `checkpoint_convert.py` no longer matched the module
tree the Round 4 `_MoEBlock` rewrite installed, so `load_hf_model` still
raised and no real-weight compile had ever been attempted.

## What changed in Round 5

### 1. `checkpoint_convert.py` — full rewrite

- All FP8 block-quant weights (routed experts, shared expert, dense MLP,
  MLA q_a/q_b/kv_a/o) are now **dequantized to BF16** in the converter,
  because Round 4 declares `ExpertFused{Column,Row}ParallelLinear` at
  `dtype=neuron_config.torch_dtype` with no quantization type declared.
  `blockwise.py` will still dequant a block-scale-carrying tensor before
  the kernel internally (see driver header + `blockwise.py:~958`), but
  the state-dict handoff to NxDI's loader must already be the dtype the
  parameter declares.
- MoE routed experts are stacked + axis-flipped to match NxDI's canonical
  shapes:
    - `mlp.expert_mlps.mlp_op.gate_up_proj.weight` → `[E, hidden, 2*I]`
      with `[gate_full | up_full]` on the last axis (stride=2 fused; this
      is exactly what `torch.cat([gate, up], dim=1).transpose(1, 2)`
      produces in `models/dbrx/modeling_dbrx.py:82-83`).
    - `mlp.expert_mlps.mlp_op.down_proj.weight` → `[E, intermediate, hidden]`.
- The three per-stream KDA short convs are concatenated on the channel
  axis into the wrapper's single depthwise `conv1d` (exact for depthwise:
  no cross-channel mixing to preserve).
- Layer 45 (MTP) and the whole visual tower (`visual.*`, 347 tensors) are
  dropped explicitly.
- Every FP8 dequant validates the reciprocal scale is finite, positive,
  and `max ≤ 240.0` (native-E4M3 qmax) — an inherited OCP-448 scale
  would silently produce wrong dequantized values.
- Prefix handling: the converter accepts either the pre-strip
  (`model.language_model.*`) or post-strip (`language_model.*`) HF key
  namespace, and treats the visual tower prefix from whichever side.  On
  the NxDI compile path the framework strips `model.` → `""` before the
  converter runs (`NeuronApplicationBase._STATE_DICT_MODEL_PREFIX`), so
  the keys arrive as `language_model.layers.<i>...`, `lm_head.weight`,
  `visual.encoder...`.  This is what the converter's `TEXT_PREFIX` /
  `VISION_PREFIX` constants expect.
- Indexer traps handled: HF stores `indexer.wk` / `indexer.wq_b` /
  `indexer.weights_proj` (all BF16, no scales) plus four extra tensors
  (`indexer.k_norm.{weight,bias}`, `indexer.index_kpool_compress_{ape,gate}`)
  the Round 4 wrapper does not declare.  The mapped three go through; the
  four unmapped ones are **recorded** in the conversion report but NOT
  forwarded (they would land as "unexpected key" against the wrapper's
  actual parameter set).  Round 6 can decide whether to model them.

### 2. `neuron_wrapper.py` — fail-loud FP8-KV guard

Mirrors the GLM-5.2 guard at `vllm-neuron:apuroop/glm5-2-enablement:vllm_neuron/model/glm52_moe_dsa/factory.py:260-261`.
`build_neuron_config` now refuses any `extra` dict carrying
`fp8_packed_kv`, `kv_cache_quant`, or `kv_quant_config` with a clear
error, before any silent-drop is possible:

```
raise ValueError(
    "GLM-5.3-Flash refuses FP8-packed KV configuration: {offenders}. "
    "This wrapper replaces NxDI's KVCacheManager with its own hybrid state "
    "cache (KDA + DSA + indexer); the aliased state tensors are declared "
    "bf16 explicitly and any FP8-KV request would silently mismatch the "
    "emitted neuron_config.json against the actual tensor dtypes."
)
```

The compile-driver at `/mnt/compile/shared-images/glm53-flash-command.sh`
already carried the same refusal at the shell layer; this adds a Python-
side check for callers who instantiate the wrapper directly.  Structural
context (from Round 4): the four aliased state tensors per DSA layer
(`k_cache`, `v_cache`, `index_k_cache`) and per KDA layer (`kda_state`,
`conv_state`) are declared BF16 in `state_cache_specs()` — an FP8-KV
request would land a `neuron_config.json` advertising `float8_e4m3fn`
while the actual tensors stayed BF16.  That is the exact
"requested-vs-emitted" split the campaign refuses to inherit.

### 3. `load_hf_model` — directory-backed shell

Replaces the Round-4 `NotImplementedError` stub with a minimal
`nn.Module` shell whose `state_dict()` returns exactly what NxDI's own
`load_state_dict(directory)` would have returned.  On the compile path
NxDI never calls this method (the snapshot dir is an existing directory
so `get_state_dict` takes the `load_state_dict(dir)` branch directly), so
this is a safety hedge, not the main integration point.  Hub-name fetch
raises with a clear error — the campaign never does network fetches.

## Deliverables

### 1-tensor smoke — PASS

Local receipt: `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\smoke_round5_one_tensor.py`.
Compile-host receipt: `/mnt/compile/runroot/glm53-round5-1tensor-smoke.json`.

Ran twice: (1) synthetic FP8 tensor + block scale generated in-code (proves
the code path with no HF file dependency); (2) real HF FP8 routed-expert
weight (`model.language_model.layers.11.mlp.experts.204.down_proj.weight`,
`torch.float8_e4m3fn`, shape `[4096, 2048]`, scale shape `[32, 16]`).

Both compare `dequantize_block_fp8(..., torch.bfloat16)` output against a
hand-rolled reciprocal blockwise dequant:

```
w_bf16_lib    = dequantize_block_fp8(fp8, scale_inv, block=(128,128), torch.bfloat16)
w_bf16_golden = (fp8.to(fp32) * scale_inv.tile(bo,bi)).to(bf16)   # hand-rolled
assert (w_bf16_lib - w_bf16_golden).abs().max() < 1e-3
```

Real-shard receipt:

```json
{
  "mode": "hf-shard",
  "hf_key": "model.language_model.layers.11.mlp.experts.204.down_proj.weight",
  "hf_weight_dtype": "torch.float8_e4m3fn",
  "hf_scale_dtype": "torch.float32",
  "hf_weight_shape": [4096, 2048],
  "hf_scale_shape": [32, 16],
  "lib_shape": [4096, 2048],
  "lib_dtype": "torch.bfloat16",
  "max_abs_error_bf16": 0.0,
  "mean_abs_error_bf16": 0.0,
  "layer0_expected_keys": 27,
  "layer0_missing_keys": [],
  "converted_key_count": 27,
  "status": "PASS"
}
```

`0.0` bf16 max_abs is exact because fp32 arithmetic is exact for the
blockwise `w * scale_inv` and the final cast to bf16 rounds both sides
the same way.  27/27 expected layer-0 keys land under the wrapper's
Round-4 module tree spellings (`input_norm_weight`,
`self_attn.o_norm_weight`, `self_attn.conv1d.weight`, `hc_attn.base`,
`mlp.gate_proj.weight`, etc.).

### Real-weight compile fire — **BLOCKED on RAM ceiling**

The full-model conversion produces a state dict whose BF16 footprint is
**~611 GiB** — computed:

  ```
  per MoE layer routed weights:
      gate+up+down = 3 x 288 x 2048 x 4096 x 2 bytes  = 14.50 GiB
  42 MoE layers x 14.50 GiB                            = 609 GiB routed
  + shared expert + dense MLP + MLA + KDA + norms      ~ 2 GiB
                                              total   ~ 611 GiB
  ```

Compile-host RAM at time of measurement (`08:34Z`): **371 GiB total,
259 GiB available**.  The peer Round-4 CTE compile in flight has been
in `neuronx-cc` for ~40 min and will itself hit `shard_checkpoint` when
it exits the compiler, at which point it will also OOM (its old
`checkpoint_convert.py` module tree does not match the Round-4 wrapper —
`preprocess_checkpoint` will raise "missing key" for
`mlp.expert_mlps.mlp_op.gate_up_proj.weight` etc. before the OOM lands,
so the peer's failure mode is a Python KeyError, not an OS OOM).

The Round-4 status doc did **not** describe this constraint because
Round 4 never attempted a real-weight compile: its 4-layer coverage smoke
uses `initialize_model_weights=False` end-to-end, and its "TP=32 TKG NEFF
landed" receipt is a **shell** compile (10.28 MB NEFF, 43.97 MB model.pt
against a 313 G-param model — a real-weight `model.pt` would be much
larger by two orders of magnitude in trace size alone).

#### Why the RAM ceiling is inherent to the current design

NxDI's `NeuronBaseForCausalLM.shard_weights` calls
`get_builder().shard_checkpoint(...)` which:

  1. Calls `self.checkpoint_loader_fn()` → `get_state_dict(model_path,
     config)` → `load_state_dict(directory)` (loads all 62 shards, 306 GiB
     in FP8, into a single Python dict at once).
  2. Runs `convert_hf_to_neuron_state_dict(sd, config)` (our converter),
     which produces the ~611 GiB BF16 dict.
  3. Loops over ranks and calls `shard_weights_with_cache(rank, model,
     checkpoint, ...)` which does `checkpoint.copy()` (shallow — tensor
     references shared) then per-rank sliced + saves.

Steps 1–2 combined require ~611 GiB of tensor storage in RAM at peak, not
counting Python overhead.  Even with FP8 mmap (via `safe_open` +
per-tensor lazy read), the output dict from step 2 is dense — there is
no code path that streams from FP8 directly to per-rank sharded files.

#### Options for Round 6

  1. **Streaming checkpoint_loader override.**  Subclass
     `checkpoint_loader_fn` on the wrapper to open safetensors with
     `safe_open(mmap=True, framework="pt")`, and instead of returning a
     full dict, do the sharding inline: for each target parameter, read
     the FP8 slice, dequant, then per-rank slice + append to that rank's
     open safetensors writer.  Never materialise the full state dict.
     Rough sketch: a single-pass 76,108-tensor loop with `torch.chunk`
     for the ColumnParallel intermediate-axis split and `chunk` +
     `torch.cat` for the fused stride=2 gate|up ordering.
  2. **Two-host split.**  Load the HF checkpoint on a host with ≥800 GiB
     RAM and produce the sharded safetensors directly there.  Ship the
     sharded files (~19 GiB × 32 ranks = 611 GiB) back to the compile
     host, which then only needs to run `torch.jit.save` for the trace.
     Requires the runtime cluster to accept pre-sharded checkpoints,
     which NxDI's `load_weights` does when
     `save_sharded_checkpoint=True` and the files exist.
  3. **Compile without weights + defer real-weight materialisation to
     the runtime host.**  `neuron_config.skip_sharding = True` at compile
     time; the runtime host does the shard.  Same RAM ceiling as (1)
     applies but on the Trn2 host, which has 371 GiB (measured) — still
     too small.  Only helps if paired with (1) or (2).

**Round 6 recommendation:** Option 1.  It's the smallest incremental
change, and lands the same sharded artifacts NxDI's load path already
supports.  The bulk of the code is the axis-manipulation logic already
in `_convert_moe_layer`; the delta is to (a) not stack all experts, and
(b) not fuse gate|up in-memory — chunk them per rank and write directly.

## Discipline notes

- No compile was fired that would compete for RAM with the in-flight
  peer compile (`/mnt/compile/runroot/glm53-round4/`) — the RAM ceiling
  is documented as the blocker, not chased into an OOM.
- No `git push`.  Local commit only:
  `80206a6 GLM-5.3-Flash Round 5: checkpoint_convert rewrite for ExpertMLPs module tree`
  in the `vllm-neuron-codex-alpha` repo (branch:
  `codex/glm53-flash-enablement`).
- No spec-decode surface introduced.  Layer 45 (MTP) is dropped
  explicitly with a report entry.
- No CPU-fallback markers.  Every conversion path is either dequant
  (block-scale-carrying weights) or verbatim cast (BF16 hold-out).
- The FP8-KV wireup superset is deliberately not present.  See fail-loud
  guard above.  Rationale in the compile-driver header at
  `/mnt/compile/shared-images/glm53-flash-command.sh` §3.

## Files touched

Full absolute paths:

- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\checkpoint_convert.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\neuron_wrapper.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\smoke_round5_one_tensor.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\NXDI-WRAPPER-ROUND5-STATUS-2026-08-28.md`

Compile host mirror (via scp; matches the Round 4 file layout):

- `/mnt/compile/src/vllm-neuron-alpha/vllm_neuron/model/glm53_flash/checkpoint_convert.py`
- `/mnt/compile/src/vllm-neuron-alpha/vllm_neuron/model/glm53_flash/neuron_wrapper.py`
- `/mnt/compile/src/vllm-neuron-alpha/vllm_neuron/model/glm53_flash/smoke_round5_one_tensor.py`
- `/mnt/compile/runroot/glm53-round5-1tensor-smoke.json`  (1-tensor smoke receipt)

## Summary

| Deliverable | Status |
|-------------|--------|
| Rewritten `checkpoint_convert.py` matching Round-4 module tree | DONE |
| 1-tensor smoke against real HF FP8 shard | PASS (`0.0` bf16 max_abs, 27/27 keys) |
| 1-tensor smoke against synthetic input | PASS (local, Windows) |
| Fail-loud FP8-KV guard on `build_neuron_config` | DONE |
| `load_hf_model` no longer raises | DONE (directory-backed shell) |
| Real-weight compile: slug + NEFF + wall | BLOCKED — RAM ceiling ~611 GiB vs 259 GiB free |
| Emitted `neuron_config.json` + CPU-fallback grep | BLOCKED — no real-weight compile fired |
| 10-token gate verdict vs banked reference logits | BLOCKED — no runnable NEFF |

**Single next blocker:** streaming checkpoint loader.  The full BF16
dequantised state dict is ~611 GiB and the compile host has 259 GiB
available; a `checkpoint_loader_fn` override that produces per-rank
sharded safetensors directly (never materialising the full dict) is the
smallest change that unblocks the real-weight compile.  Option 1 above
is the recommended sketch.
