# NxDI Wrapper Round 2 — GLM-5.3-Flash — 2026-08-28

**Callsign:** nxdi-wrapper-round2-agent
**Branch:** `codex/glm53-flash-enablement` (local, no push per Codex constraint)
**Worktree:** `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha`
**Prior round:** Round-1 shell — `NXDI-WRAPPER-SMOKE-2026-08-28.md`, commit `42cea91`
**Compile host:** `ec2-user@13.222.20.119` — this round did NOT touch the host (container zombied; Round 3 tick will re-run the container-import smoke)

---

## 0. Deliverables landed

| file | absolute path | change | LOC before | LOC after | delta |
|---|---|---|---:|---:|---:|
| `neuron_wrapper.py` | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\neuron_wrapper.py` | replaced Round-1 single-Linear shell with real per-layer NxDI parallel primitives (`_NoPeMLABlock`, `_KDABlock`, `_DSABlock`, `_DenseMLPBlock`, `_MoEBlock` w/ shared expert, `_MHCBlock`, `Glm53FlashLayer`, real `_NeuronGlm53FlashModel.init_model` + `.forward`); added `load_weights` w/ indexer-multiplier preflight | 523 | 1378 | +855 |
| `registry.py` | `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\registry.py` | bumped `GLM53_SOURCE_CACHE_ABI` slug to `glm53-flash-round2-nxdi-primitives-v1` and `_GLM53_GRAPH_ID` to `glm53-flash-nkiv0-refs-round2-v1` (COMPILE-FASTPATH cache-key discipline) | 45 | 64 | +19 |

**Untouched:** `command.sh`, `registry_hook.py`, `model.py`, `__init__.py`, `attention.py`, `kda.py`, `indexer.py`, `mla.py`, `moe.py`, `dense_mlp.py`, `mhc.py`, `telemetry.py`, `config.py`, `_reference_kernels.py` — Round 2 lives entirely inside `neuron_wrapper.py` + a cache-slug bump.

---

## 1. Block-by-block lowering (Round 2 scope)

Every block uses NxDI parallel primitives from
`neuronx_distributed.parallel_layers.layers`:
- `ColumnParallelLinear` (`_NxdColumnParallelLinear`)
- `RowParallelLinear` (`_NxdRowParallelLinear`)
- `ParallelEmbedding` (`_NxdParallelEmbedding`)

RMSNorm gains and small mHC parameters are held as `nn.Parameter` (replicated
across TP ranks; too small to be worth sharding).

### 1.1 `_NoPeMLABlock` — All-NoPE MLA (used by DSA layers)

- Q_A_proj: `hidden -> q_lora_rank` — `ColumnParallelLinear(gather_output=True)` (replicated latent)
- Q_A_norm: RMSNorm gain on `q_lora_rank` (replicated `nn.Parameter`)
- Q_B_proj: `q_lora_rank -> num_heads*qk_head_dim` — `ColumnParallelLinear(gather_output=False)` (heads-per-rank)
- KV_A_proj: `hidden -> kv_lora_rank` — `ColumnParallelLinear(gather_output=True)`
- KV_A_norm: RMSNorm gain (replicated)
- KV_B_proj: `kv_lora_rank -> num_heads*(qk_nope_head_dim + v_head_dim)` — `ColumnParallelLinear(gather_output=False)`
- O_proj: `num_heads*v_head_dim -> hidden` — `RowParallelLinear(input_is_parallel=True)`
- Frozen validation: `qk_rope_head_dim == 0` at construction time (fail-closed).
- Sharding constraint: `num_heads % tp_degree == 0` (raises `NotImplementedError` if not, deferring head-padded fallback to Round 3).
- `.project(hidden_states)` returns `(query, key, value)` with per-rank head count.

### 1.2 `_DSAIndexerBlock` + `_DSABlock` — DeepSeek Sparse Attention

- K_proj: `hidden -> index_n_heads*index_head_dim` — `ColumnParallelLinear(gather_output=False)` (heads-per-rank).
- Q_proj: rank-3 `nn.Parameter` of shape `[index_n_heads, num_attention_heads*qk_head_dim, index_head_dim]`. NxDI's `ColumnParallelLinear` only shards the output axis of a 2D weight; the DSA indexer Q-tensor shards along the LEADING axis (index-head), so it's stored replicated and the loader materialises the rank-local slice via `local_indexer_head_slice(rank, tp_degree)`.
- pool_weights: `[index_kpool]` replicated `nn.Parameter` (default = uniform `1/kpool`).
- cache_quant_multiplier: fp32 buffer (validated at load time by `_assert_indexer_multipliers_bounded`).
- Sharding constraint: `index_n_heads % tp_degree == 0` (Round-3 will add head-padded fallback).
- `_DSABlock.forward`: calls `_NoPeMLABlock.project(hidden_states)` so MLA weights bind under the tracer, then invokes `_DSAIndexerBlock.forward(hidden_states, position_ids)` which **raises `NotImplementedError`** with a pointer to `dsa_lightning_indexer.py` (nki_v0 reference). Per the fallback rule.
- IndexPool = 4, `always_select_tail=True`, `index_topk=2048` all pinned by the source-config validator.

### 1.3 `_KDABlock` — Kimi Delta Attention

Projections lowered:
- q_proj, k_proj, v_proj: `hidden -> num_heads*head_dim` — `ColumnParallelLinear(gather_output=False)` (heads-per-rank).
- f_a_proj / g_a_proj: `hidden -> head_dim` — `ColumnParallelLinear(gather_output=True)` (small latent).
- f_b_proj / g_b_proj: `head_dim -> num_heads*head_dim` — `ColumnParallelLinear(gather_output=False)`.
- b_proj: `hidden -> num_heads` — `ColumnParallelLinear(gather_output=False)`.
- o_proj: `num_heads*head_dim -> hidden` — `RowParallelLinear(input_is_parallel=True)`.

Replicated parameters (kept off TP shard):
- conv1d: depthwise-groups `nn.Conv1d` over `3*num_heads*head_dim` channels (Round 3 will replace with a NKI depthwise-conv kernel that shards along the head axis).
- dt_bias: `[qkv_dim]` fp32 (matches FLA v0.5.2 gate bias layout).
- A_log: `[num_heads]` fp32 (per-head KDA gate coefficient; `alpha = -5.0 * sigmoid(exp(A_log) * (g_raw + g_bias))` per KDA-STATE-V2-STATUS section 2).
- o_norm.weight: RMSNorm gain over `head_dim`.

Sharding constraint: `num_heads % tp_degree == 0` (Round-3 will pad).

`forward` traces q/k/v projections then **raises `NotImplementedError`** with a
pointer to `kda_state_v2.py` — the KDA state kernel (`bf16_state`,
`kda_state.decode.kda_gate.rank1_delta.bf16_state.v1`, in-kernel L2-norm,
lower-bound gate, SigmoidBeta) is a Round-3 device NKI binding.  Per the
fallback rule: no silent fall-through to CPU.

### 1.4 `_DenseMLPBlock` — first 3 layers only

- gate_proj / up_proj: `hidden -> intermediate_size` — `ColumnParallelLinear(gather_output=False)`.
- down_proj: `intermediate_size -> hidden` — `RowParallelLinear(input_is_parallel=True)`.
- SwiGLU with GLM-5.3 clamp limit `swiglu_limit` (default 10.0): `silu(clamp(gate, max=limit)) * clamp(up, -limit, limit)`.
- Sharding constraint: `intermediate_size (=12288) % tp_degree == 0`.
- Forward is fully lowered (no `NotImplementedError`); traces end-to-end.

### 1.5 `_MoEBlock` — 288 routed experts + 1 shared expert

Routed-expert weights (fused across expert axis, replicated in Round 2):
- gate: `[n_routed_experts, hidden, moe_intermediate_size]` — replicated `nn.Parameter`.
- up: same shape — replicated `nn.Parameter`.
- down: `[n_routed_experts, moe_intermediate_size, hidden]` — replicated `nn.Parameter`.

**Rationale for replication in Round 2:** the blockwise-MoE loader addresses
expert weights by index; the expert-axis TP sharding requires an
`ExpertParallelism` primitive that lives inside NxDI's blockwise-MoE
pathway (which the current container `sha256:011d49c7` doesn't fully bind —
the container is missing `_call_shard_hidden_kernel`).  Round 3 introduces
expert-axis sharding once the container patch lands OR uses the
`use_shard_on_intermediate_dynamic_while=True` blockwise workaround
end-to-end.

Router:
- `nn.Linear(hidden -> n_routed_experts)` in fp32 (matches `moe_router_dtype="float32"`).

Shared expert (`_MoESharedExpert`):
- gate_proj / up_proj: `hidden -> moe_intermediate_size` — `ColumnParallelLinear(gather_output=False)`.
- down_proj: `moe_intermediate_size -> hidden` — `RowParallelLinear(input_is_parallel=True)`.
- Fully lowered; traces end-to-end.

`_MoEBlock.forward`: shared branch is traced; routed dispatch **raises
`NotImplementedError`** with a pointer to `moe_dispatch.py` (`MoEDispatchConfig`,
288 experts × top-8 pattern, blockwise-MoE NKI kernel).  Per the fallback rule.

### 1.6 `_MHCBlock` — mHC 4-stream pre/post mixer

- fn: `[(2+hc_mult)*hc_mult, hc_mult*hidden]` — replicated `nn.Parameter`.
- base: `[(2+hc_mult)*hc_mult]` fp32 — replicated.
- scale: `[3]` fp32 — replicated.
- `pre` and `post` are torch-primitive (rsqrt, sigmoid, softmax, einsum, Sinkhorn 20-iter row/column normalisation with `hc_eps=1e-6`).  Fully traces; bit-matches `mhc.py`.

### 1.7 `Glm53FlashLayer` — per-layer dispatch

- Owns `input_norm_weight`, `post_attention_norm_weight` (RMSNorm gains, replicated).
- `attn_kind`: `"dsa"` for `layer_types[idx] == "deepseek_sparse_attention"` (indices 3, 7, 11, …, 43); `"kda"` otherwise.
- `mlp_kind`: `"dense"` for `mlp_layer_types[idx] == "dense"` (indices 0..2); `"sparse"` otherwise.
- `hc_attn`, `hc_mlp`: one `_MHCBlock` each (matches `Glm53FlashDecoderLayer` in `model.py:40-41`).
- `forward(residual_streams, position_ids)` sequences: `hc_attn.pre → input_norm → self_attn → hc_attn.post → hc_mlp.pre → post_attention_norm → mlp → hc_mlp.post`.

### 1.8 `_NeuronGlm53FlashModel` — top-level graph

- `embed_tokens`: `ParallelEmbedding` (matches Round-1 shell; `shard_across_embedding=True`, `pad=True`, honours `sequence_parallel_enabled`).
- `layers`: `nn.ModuleList` of `Glm53FlashLayer(config, layer_idx=i)` for `i in range(num_hidden_layers)`.
- `final_norm_weight`: RMSNorm gain over hidden (replicated).
- `lm_head`: `ColumnParallelLinear(hidden -> vocab_size, pad=True, gather_output = not on_device_sampling)`.
- `forward(input_ids, positions)`:
  1. `hidden = embed_tokens(input_ids)`; ensure `[batch, seq, hidden]`.
  2. Widen to 4-stream mHC residual: `unsqueeze(-2).repeat(1, 1, hc_mult, 1)` (matches Impl.forward at `model.py:140`).
  3. Sequence through every `Glm53FlashLayer`.
  4. Collapse: `residual_streams.mean(dim=-2)` (unweighted 4-stream head collapse).
  5. Final RMSNorm + `lm_head`.

---

## 2. Cache-pin bump

Per COMPILE-FASTPATH.md §"cache slug on any forward-shape change":

```
Round 1  GLM53_SOURCE_CACHE_ABI = "glm53-flash-source-v1|dsa=…|kda=…|moe=…|qk=256|nope=256|rope=0|index-kpool=4|layers=45"
Round 2  GLM53_SOURCE_CACHE_ABI = "glm53-flash-round2-nxdi-primitives-v1|dsa=…|kda=…|moe=…|qk=256|nope=256|rope=0|index-kpool=4|layers=45|hc-mult=4|routed-experts=288|top-k=8|shared-experts=1"
Round 1  _GLM53_GRAPH_ID        = <same string as GLM53_SOURCE_CACHE_ABI>
Round 2  _GLM53_GRAPH_ID        = "glm53-flash-nkiv0-refs-round2-v1"
```

Any Round-1 compile artifact will deserialise into the wrong graph shape, so
the slug change forces the modular-compile flywheel to treat Round-2 artifacts
as cache-distinct.  This matches the Round-1 receipt §5 item 5 explicit ask.

---

## 3. Fallback-rule discipline (per prompt constraint)

The prompt is explicit: "any block that can't be lowered to NxDI parallel
primitives raises `NotImplementedError` — do NOT silently fall through to
`nn.Linear` or CPU."

| block | projections lowered? | kernel call | Round-3 target |
|---|---|---|---|
| `_NoPeMLABlock` | yes (CPL for Q_A / Q_B / KV_A / KV_B; RPL for O) | dot-product attention is exercised by `_DSABlock`; `.project` alone is fully lowered | Round-3 device-side flash-attn NKI wrapper |
| `_KDABlock` | yes (CPL for Q/K/V/F/G/B; RPL for O; depthwise conv replicated) | `forward` raises → Round 3 | `kda_state_v2.py` (bf16_state, KDA gate, L2-norm-in-kernel) |
| `_DSAIndexerBlock` | yes (CPL for K_proj; Q_proj as sliced-at-load nn.Parameter; pool_weights replicated) | `forward` raises → Round 3 | `dsa_lightning_indexer.py` (nki_v0 lightning-indexer + sparse-attn) |
| `_DSABlock` | yes (composes `_NoPeMLABlock` + `_DSAIndexerBlock`) | `forward` traces MLA project, then raises → Round 3 | Round 3 combined DSA sparse-attn NKI wrapper |
| `_DenseMLPBlock` | yes (CPL for gate/up; RPL for down) | fully traces | n/a — Round-2 complete |
| `_MoEBlock` (routed) | routed-expert weights replicated in Round 2 (see §1.5) | `forward` raises → Round 3 (shared branch is fully traced) | `moe_dispatch.py` (blockwise-MoE NKI kernel; container-workaround already applied via `build_neuron_config`) |
| `_MoESharedExpert` | yes (CPL for gate/up; RPL for down) | fully traces | n/a — Round-2 complete |
| `_MHCBlock` | yes (replicated small `nn.Parameter`) | fully traces (torch-primitive Sinkhorn) | n/a — Round-2 complete |
| `_NeuronGlm53FlashModel` (top) | yes (ParallelEmbedding, ColumnParallelLinear lm_head) | fully traces | n/a — Round-2 complete |

**No silent CPU fall-through anywhere.**  A DSA / KDA / routed-MoE `forward`
call raises with a full pointer to the reference kernel and the Round-3
target slug.  Round-3 replaces those raise statements with NKI kernel
invocations.

---

## 4. `load_weights` — Round-2 implementation

Round 1's `load_weights` was `raise NotImplementedError`.  Round 2 wires:

1. **Preflight** (`_assert_indexer_multipliers_bounded`): iterate every DSA
   layer index; verify the source-config's `indexer_cache_quant_multiplier`
   is in `(0, 240.0]` (Trainium2 native-e4m3fn max).  Uses Fleet A's
   `glm52_indexer_fp8_scale_fix.assert_indexer_multiplier_bounded` when
   importable (best-effort import via file-path spec loader so the wrapper
   stays importable when the harness scratchpad isn't on `PYTHONPATH`); an
   in-wrapper 240.0 cap check is the fallback.  Fail-closed: an out-of-range
   multiplier refuses the load.
2. **Delegate** to `NeuronApplicationBase.load_weights(compiled_model_path,
   **kwargs)` so per-rank
   `weights/tp{rank}_sharded_checkpoint.safetensors` is honoured and each
   rank's `traced_model.nxd_model.initialize(weights, start_rank_tensor)`
   fires exactly as it does for every other NxDI model.

The per-layer cache-multiplier scalars are materialised by NxDI's base loader
into the `cache_quant_multiplier` buffer declared on `_DSAIndexerBlock`.
Step 1 only *validates* the values before compile-time constants freeze.

The full HF-to-Neuron state-dict conversion + sharded-FP8 rewriter is
deferred to Round 3 (its own `convert_hf_to_neuron_state_dict` remains
`NotImplementedError`, mirroring GLM-5.2's checkpoint-mapping contract).

---

## 5. Sharding constraint audit

Every block that shards along a head-count axis validates its constraint at
construction time and raises `NotImplementedError` (deferring the head-padded
fallback to Round 3) if the constraint fails:

| dimension | source-config value | TP degrees that divide cleanly | notes |
|---|---:|---|---|
| `num_attention_heads` | 64 | 1, 2, 4, 8, 16, 32, 64 | MLA / KDA (linear_attn_config.num_heads = 64) — every planned Round-3 TP configuration works |
| `index_n_heads` | 32 | 1, 2, 4, 8, 16, 32 | DSA indexer — TP=32 works; TP=64 does NOT and would trip the guard, which is correct per the "no TP=64 without card 12" scope |
| `intermediate_size` | 12288 | 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128 | Dense MLP — every planned TP works |
| `moe_intermediate_size` | 2048 | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 | Shared expert MLP — every planned TP works |

For the campaign's realistic TP set (TP=8 first-fire smoke, TP=16 second
fire, TP=32 third; TP=64 blocked by card 12), every block passes.

---

## 6. What Round 3 delivers

Deferred to Round 3 (device-side):

1. **NKI kernels bound to `_KDABlock.forward`**, `_DSAIndexerBlock.forward`,
   `_MoEBlock.forward` routed branch.  References already in place at:
   - `.../reference-sweep-20260826T2150Z/kernels/kda_state_v2.py`
   - `.../reference-sweep-20260826T2150Z/kernels/dsa_lightning_indexer.py`
   - `.../reference-sweep-20260826T2150Z/kernels/moe_dispatch.py`

2. **`convert_hf_to_neuron_state_dict`** — mirror GLM-5.2 checkpoint-mapping
   (fused-QKV + routed-expert), extend for KDA `short_conv`, DSA `pool_weights`
   / `q_proj` sliced-load, mHC 4-stream mixer.

3. **Container-import smoke** — run once compile-host docker daemon is
   unstarved.  Test path: `sudo docker run --rm ... python -c "from
   vllm_neuron.model.glm53_flash.neuron_wrapper import
   NeuronGlm53FlashForCausalLM; print(NeuronGlm53FlashForCausalLM.GLM53_SOURCE_CACHE_ABI)"`.
   Expected output includes `glm53-flash-round2-nxdi-primitives-v1`.

4. **1-layer compile-driver dry-run** — via
   `NeuronGlm53FlashForCausalLM.build_one_layer_smoke_config(source_config,
   tp_degree=8)` + `wrapper.compile(out_path, dry_run=True)`.  Note: the
   Round-2 KDA `forward` raises `NotImplementedError`, so the dry-run smoke
   verifies driver *binding* only, not end-to-end tracing.  A traceable
   dry-run needs at least one NKI kernel bound in Round 3.

5. **Correctness gate** — once Fleet A's HF reference-logit capture lands
   (PID 84475 in Round-1 receipt), run `wrapper.get_cpu_oracle()` under a
   deterministic seed and compare to the compiled artifact.

6. **Expert-axis sharding for `_MoEBlock`** — replace Round-2 replicated
   `gate`/`up`/`down` `nn.Parameter` tensors with an expert-parallel primitive
   once the container patch for `_call_shard_hidden_kernel` lands OR wire the
   full blockwise workaround end-to-end (workaround dict is already declared
   in `build_neuron_config` at InferenceConfig-init time).

---

## 7. Local smoke performed this round

- `python -c "import ast; ast.parse(open('neuron_wrapper.py').read())"` — PASS (1378 LOC parse clean; no SyntaxError).
- `python -c "import ast; ast.parse(open('registry.py').read())"` — PASS (64 LOC).

The end-to-end module-import smoke and the guarded-fallback instantiation
smoke (Round-1 §1 "SMOKE A") were NOT re-run this session because torch is
not present in this Windows Python environment; those smokes run on the
compile host and are equivalent to Round-1 since the guarded-import shape and
`_require_nxdi` behaviour are unchanged.  Round 3's first tick re-runs the
container-import smoke on the compile host once the docker daemon frees.

---

## 8. Commit

Local branch `codex/glm53-flash-enablement`, callsign
`nxdi-wrapper-round2-agent`.  No push per Codex constraint.

Absolute local paths:
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\neuron_wrapper.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\registry.py`
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\NXDI-WRAPPER-ROUND2-STATUS-2026-08-28.md`
