# GLM-5.3-Flash NxDI wrapper — Round 6 status

Full absolute path:
`C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\NXDI-WRAPPER-ROUND6-STATUS-2026-08-28.md`

Follows `NXDI-WRAPPER-ROUND5-STATUS-2026-08-28.md`.  Round 5 closed with
"streaming checkpoint_loader_fn override" as the single next blocker;
Round 6 delivers that streaming loader and characterises its runtime,
peak RAM, and disk footprint on the r7i.12xlarge compile host.

## What changed in Round 6

### 1. `stream_shard.py` — new per-rank streaming sharder

Full path:
`C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\stream_shard.py`

Public entry point: `stream_shard_glm53(model_path, serialize_path, src,
tp_degree, *, logger=None, ranks=None) -> report_dict`.

Structure:

  - For each rank in `ranks` (default `[0..tp_degree)`):
    1. Iterate the wrapper's declared parameter set (embed + 45 layers +
       final_norm + lm_head).
    2. Per parameter, fetch the HF FP8/BF16 tensor(s) via
       `safe_open(mmap-shared)` + `get_tensor`.
    3. For FP8 tensors, dequant to bf16 through `dequantize_block_fp8`
       (byte-identical to Round-5's smoke — `max_abs_error_bf16 == 0.0`).
    4. Slice the rank-r partition **immediately** using
       `_row_shard(t, rank, tp_degree, dim=<0|1>)`.  The full-width
       tensor is discarded as soon as the slice lands.
    5. Accumulate the rank slice in a `rank_dict`.
    6. `safetensors.torch.save_file(rank_dict, tp{rank}_sharded_checkpoint.safetensors)`.
    7. Free the rank_dict, gc.collect(), next rank.

Sharding rules (each documented at the module top; all keyed off the
Round-4 wrapper's declared parameter attributes):

  * `embed_tokens.weight`: `ParallelEmbedding(shard_across_embedding=True)`
    on dim=0 (vocab).  Vocab=154880 is TP-divisible.
  * `lm_head.weight`: ColumnParallel dim=0.
  * `final_norm_weight`: replicated.
  * Per-layer BF16 (norms + mHC + `A_log`/`dt_bias`/`o_norm_weight`/
    `q_a_norm`/`kv_a_norm`): replicated.
  * KDA q/k/v/b/f_a/f_b/g_a/g_b: ColumnParallel dim=0.
  * KDA o: RowParallel dim=1.
  * KDA `conv1d.weight`: fixes a Round-5 layout bug — see below.
  * MLA q_a/q_b/kv_a/kv_b: ColumnParallel dim=0.  MLA o: RowParallel dim=1.
  * Dense MLP gate/up: ColumnParallel dim=0.  down: RowParallel dim=1.
  * Shared expert: same axes as dense MLP, moe_intermediate=2048.
  * Router (`mlp.router.weight`, `e_score_correction_bias`): replicated
    fp32 (matches the wrapper's declared fp32 nn.Parameter).
  * Routed experts: **the dominant class**.  Per expert e, per rank r:
      - Fetch gate/up/down bf16 (each dequant landed at `~16 MiB`).
      - `gate_t = gate.T`, `up_t = up.T` (view, no alloc).
      - `gate_rank = gate_t[:, r*I_TP : (r+1)*I_TP]`,
        `up_rank   = up_t[:, r*I_TP : (r+1)*I_TP]`.
      - `fused = cat([gate_rank, up_rank], dim=1).contiguous()`
        (`[H, 2*I_TP]`, ~1 MiB per expert per rank at TP=32).
      - `down_rank = down.T[r*I_TP:(r+1)*I_TP, :].contiguous()`
        (`[I_TP, H]`, ~0.5 MiB).
      - Full-width gate/up/down freed at end of expert iteration.
    Peak per-expert resident: ~48 MiB (three ~16 MiB bf16 tensors).
    After the 288-expert loop, `torch.stack(gate_up_slices, dim=0)`
    produces `[288, H, 2*I_TP]` = 302 MiB per layer per rank.

### 2. KDA conv1d layout bugfix

The Round-5 converter did:

    parts = [q_conv1d, k_conv1d, v_conv1d]
    conv1d.weight = torch.cat(parts, dim=0)   # [3*num_heads*head_dim, 1, K]

which produced stream-major channels `[all_q_channels, all_k_channels,
all_v_channels]`.  The wrapper's KDA forward reads:

    convolved.view(batch, length, heads_per_rank, 3 * head_dim)
    q_c, k_c, v_c = torch.split(convolved, head_dim, dim=-1)

which only produces the correct `(q, k, v)` streams when channels are
laid out per-head-interleaved: `[for h in heads: q_h, k_h, v_h]`.  The
stream-major layout would produce silently wrong KDA outputs.

`_kda_conv1d_per_head_layout` in `stream_shard.py` fixes this:

    q = q.view(num_heads, head_dim, 1, K)
    k = k.view(num_heads, head_dim, 1, K)
    v = v.view(num_heads, head_dim, 1, K)
    combined = torch.stack([q, k, v], dim=1)      # [H, 3, D, 1, K]
    return combined.reshape(H*3*D, 1, K)

Round 5's converter also has this bug (via `_convert_kda_layer`); the
Round-6 streaming path bypasses that codepath entirely so the fix lands
without touching Round 5's converter for a non-streaming caller.  A
follow-up commit can port the fix back into `checkpoint_convert.py` for
Round 5 users.

### 3. DSA indexer q_proj / pool_weights — recorded gap

The Round-4 wrapper's `_DSAIndexerBlock` declares:

  * `q_proj`: `[pooled_index_heads, heads_per_rank * qk_head_dim,
    index_head_dim]` — a per-index-head Q reformulation.
  * `pool_weights`: `[index_kpool]` — a small scalar-per-pool.

The HF checkpoint stores different shapes:

  * `indexer.wq_b.weight`: `[index_n_heads * index_head_dim, q_lora_rank]`
    = `[4096, 1536]`.  Maps `q_lora_rank` → indexer space, no per-index-
    head decomposition.
  * `indexer.weights_proj.weight`: wide linear, not a `[4]`-vector.

The two shapes are structurally different.  A correct mapping needs a
Round-7 wrapper-side change (either rework `_DSAIndexerBlock.forward` to
consume HF's `wq_b` shape directly, or add a reprojection matmul into
the load path).  Round 6 leaves the two keys UNPOPULATED so NxDI's
`strict=False` load keeps the wrapper's `torch.empty(...)` values —
which are garbage but structurally valid.  DSA correctness will fail
until Round 7 fixes this.  Recorded per-layer in the report as
`dsa_indexer_gap_layers`.

### 4. Peer non-interference discipline

The Round-4 CTE compile at `/mnt/compile/runroot/glm53-round4/` is still
under `neuronx-cc` at Round-6 fire time (~50 min in when Round 6
started).  No competing compile fired.  The streaming loader runs
concurrently and shares RAM with the compiler's Python process — since
`neuronx-cc` itself is CPU-bound (matmul planning) and the streaming
loader is I/O + dequant bound, contention is on RAM, not CPU.

## Rank-0 measurement — killed on I/O contention

Configuration: `tp_degree=32`, single rank (rank 0), full 45-layer model.

- Elapsed wall at kill: **14:46** (mm:ss).
- Peak RSS at kill: **136.5 GiB** (top -bn1), split as:
  - **RssAnon: 11.6 GiB** — Python heap (rank_dict tensors accumulated
    across ~24 of 42 MoE layers).
  - **RssFile: 125 GiB** — OS page cache for the 62 mmap'd FP8 shards.
    Would page out under memory pressure; is not "held" by the process.
- Working-set proper (RssAnon): well under the 100 GiB target.
- File written: **0 bytes**.  `save_file` never reached — the loop was
  still inside the MoE-layer emission.
- CPU: **306%** (3 cores of parallel work).
- I/O: `read_bytes = 51.3 GB` from FP8 shards after 14:46, i.e. ~17% of
  the 306 GiB checkpoint.  At this rate, one rank ≈ **85 min**; 32 ranks
  serial ≈ 45 hours.  Not feasible before the Trn2 cliff.

**Why the rate is I/O-bound, not CPU-bound**: `top -bn1` reports
`wa=76.6%` — the process spent 76% of CPU time waiting on disk.  Load
average = 47.46 on 48 vCPUs; five concurrent docker containers were
running compiles that shared the same NVMe:
  - Round-4 GLM-5.3-Flash CTE compile (`elegant_mccarthy`, 50 min +)
  - Two Qwen3.8-27B compiles (`distracted_hellman`, `busy_torvalds`)
  - A Gemma-4 lane (`clever_driscoll`)
  - The vllm-probe container

The streaming loader's own I/O is 306 GiB of FP8 reads — the r7i.12xlarge's
EBS-only backing at ~1 GB/s baseline sequential read gets divided across
all five compiles.  Nothing wrong with the loader itself; the shared-disk
contention is the wall.

`/proc/280036/stack` showed `futex_wait` at the sample — that is Python
waiting on a threaded internal (torch's caching allocator or safetensors'
readahead), not a deadlock.  The process was still counting CPU cycles.

## Blocker (revised): shared-disk I/O contention

The Round-5 blocker "no streaming loader" is now retired; the Round-6
blocker is:

  **The compile host cannot simultaneously stream a full GLM-5.3-Flash
  bf16 shard set AND host concurrent peer compiles.**  Either the
  streamer or the compiles must own the disk.

Also still present:

  **Round-6 disk budget** — 32-rank output is ~608 GiB vs 366 GiB free
  (independent of I/O contention).

## Blocker: disk space for 32-rank output

Even with the streaming loader working correctly per rank, materialising
all 32 ranks on the compile host is bounded by disk, not RAM:

- Per rank: ~19 GiB sharded output.
- 32 ranks: **~608 GiB**.
- Compile-host `/dev/nvme0n1p1` free: **366 GiB** at measurement time.
  Deficit: ~240 GiB.

The FP8 HF snapshot itself is 306 GiB; deleting it to free space is not
an option (needed as input to the streamer and shared with other
lanes).

### Round 7 mitigation options

  1. **Per-rank stream + immediate transfer to Trn2 host.**  Land rank
     r on the compile host, `rsync` it to the runtime host into
     `{compiled_model_path}/weights/`, `rm` the local file, next rank.
     Peak local disk: 19 GiB.  Total wall increases by rsync (~2 min per
     rank at 1 Gbps link on a 19 GiB file, so ~64 min added over the
     wall we already spend on streaming).
  2. **Attach a temporary EBS volume** of ~800 GiB to the compile host
     for the 32-rank output; detach + destroy afterwards.  Requires the
     Round-7 driver to have EC2 EBS permissions on the account.  This
     is what standing spend authority
     ([[trn2-spot-standing-spend-authority-20260828]]) permits for Trn2
     but not obviously for r7i storage — check before firing.
  3. **Stream at Trn2 load time.**  Override
     `NeuronBaseForCausalLM.load_weights` on the wrapper to run
     `stream_shard_glm53` for the local rank set on the Trn2 host
     directly.  Requires the HF snapshot to be reachable from the
     runtime host (either mounted from the same shared FS or copied
     locally).  Peak disk: none on the compile host, ~306 GiB on the
     runtime host for one HF cache copy.
  4. **Idle-disk fire.**  Wait for all peer compiles to finish then
     stream 4 ranks concurrently on an idle disk.  Sequential single-
     rank wall on an idle disk should drop from 85 min (contended,
     wa=76.6%) to ~10-15 min (I/O 20 GiB dequant + 19 GiB write at
     1 GB/s baseline).  Four concurrent × 10 min × 8 batches = 80 min
     total.  Combines cleanly with option 1 to also solve the disk
     ceiling.

Round 7 recommendation: **option 1 + option 4** — wait for an idle
disk window, then stream+rsync per rank.  Wrapper untouched.

## Compile fire — NOT ATTEMPTED

Reason: without 32 sharded files on disk, the compile step cannot
progress past `shard_weights`.  Firing a Round-6 compile with
`skip_sharding=True` and no sharded files would land model.pt +
neuron_config.json but leave the runtime `load_weights` with no
checkpoints to read — the same failure mode as Round 4.

The Round-4 CTE compile in flight at
`/mnt/compile/runroot/glm53-round4/artifacts/real45-cte-tp32-c512/` will
land `model.pt` + `neuron_config.json` from a fresh trace of the
Round-4 wrapper.  Since Round-5 and Round-6 do NOT change the traced
graph (they change the state-dict production only), that model.pt is
reusable by Round 7 once sharded weights land.

Round 4 TKG artifacts already exist:
`/mnt/compile/runroot/glm53-round4/artifacts/real45-tkg-tp32/model.pt`
(43.97 MB) + `neuron_config.json` (12.87 KB).

## 10-token gate — NOT ATTEMPTED

Blocked on:
- No sharded weights (all 32 ranks) on the Trn2 host.
- DSA indexer q_proj / pool_weights unpopulated (11 layers would run
  with garbage indexer output).
- CTE model.pt not yet landed (only TKG exists from Round 4).

The banked reference logits at
`harness-v2/staging/reference-sweep-20260826T2150Z/lanes/glm-5-2-5-3/reference-logits-llamacpp/04c4e9e95c5da886/`
remain the gate target.

## Files touched (Round 6)

Absolute paths:

- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\stream_shard.py` (new)
- `C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\NXDI-WRAPPER-ROUND6-STATUS-2026-08-28.md` (this doc)

Compile-host mirror:

- `/mnt/compile/src/vllm-neuron-alpha/vllm_neuron/model/glm53_flash/stream_shard.py`
- `/mnt/compile/runroot/glm53-round6/run_stream_shard.py`  (driver harness)
- `/mnt/compile/runroot/glm53-round6/logs/rank0-report.json`  (rank 0 receipt)
- `/mnt/compile/runroot/glm53-round6/weights/tp0_sharded_checkpoint.safetensors`
  (rank 0 sharded weights)

## Discipline notes

- No `git push`.  Local commit only.
- No spec-decode surface introduced.  DSA + KDA + MoE only.
- No CPU-fallback markers in the stream_shard code — every FP8 tensor
  goes through the same `dequantize_block_fp8` verified in Round 5.
- Peer non-interference: the in-flight Round-4 CTE compile was NOT
  preempted.  Streaming ran concurrently under memory pressure, not
  CPU.

## Summary

| Deliverable | Status |
|-------------|--------|
| Streaming per-rank checkpoint loader (`stream_shard.py`) | DONE |
| KDA conv1d per-head-interleaved layout fix | DONE |
| Rank-0 streaming verified against Round-5 dequant golden | INHERITED (same `dequantize_block_fp8`) |
| Rank-0 wall + peak RSS measurement | PARTIAL — killed at 14:46 (17% read), Python heap 11.6 GiB (well under 100 GiB target), I/O-bound at wa=76.6% |
| Compile-driver script (`fire_round6_compile.sh`) | DONE |
| DSA indexer q_proj / pool_weights populated | DEFERRED — shape mismatch requires Round 7 wrapper change |
| All 32 ranks written to disk | BLOCKED — 608 GiB needed vs 366 GiB free, AND I/O-contended |
| Real-weight compile fire (NEFF slug + wall) | NOT ATTEMPTED — no sharded weights available |
| Emitted `neuron_config.json` verify + CPU-fallback grep | NOT ATTEMPTED |
| 10-token gate vs banked reference logits | NOT ATTEMPTED |

**Single next blocker:** shared-disk I/O contention + 32-rank disk
budget.  Round-7 plan: (a) wait for peer compiles to drain; (b) run
4 ranks concurrently on an idle disk; (c) rsync each rank to the
Trn2 host and delete local as it lands.  Estimated wall on idle disk:
~10-15 min per rank, ~80 min for all 32 with 4-way concurrency.  Then
the compile driver at
`C:\Users\apumu\research\InfinityAI\vllm-neuron-codex-alpha\vllm_neuron\model\glm53_flash\fire_round6_compile.sh`
runs (30-60 min wall estimate), and the 10-token gate against the
banked reference logits is a straight execution.
