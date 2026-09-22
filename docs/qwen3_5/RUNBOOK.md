# Qwen3.5-9B on Trainium2 with vLLM-Neuron: runbook

Branch `qwen3_5-gated-deltanet` on vllm-neuron `release-0.24.0.1.1.0`
(`f8abae6`). Text decoder only; the vision tower and MTP head are not loaded.

No shipped vLLM-Neuron or NxD Inference release supports Qwen3.5: it is a
hybrid of 24 Gated DeltaNet (linear-attention, recurrent) layers and 8 gated
full-attention layers, and neither library implements the recurrent layer.
This branch adds it.

## Status

| Piece | File | Verified |
| --- | --- | --- |
| Gated DeltaNet math (chunked prefill, recurrent decode, causal conv, gated norm) | `vllm_neuron/model/qwen3_5/gated_deltanet.py` | CPU, against float64 reference and transformers 5.17.0 |
| Slot-indexed recurrent state | `state.py` | CPU |
| Runner contract (padded layout, slots, resets, sink row) | `runner_state.py`, `layer_step.py` | CPU, on vllm-neuron's exact padded layout |
| Gated partial-rotary attention math | `attention_math.py` | CPU, against transformers `Qwen3_5Attention` |
| Config parsing | `config.py` | CPU, against transformers' nested config |
| Model wiring (TP, SP, paged KV, NF kernels, weight loading) | `model.py`, `factory.py` | **Syntax and lint only. Needs Trn2.** |
| Runner hookup | this runbook, step 4 | **Not applied. Needs Trn2.** |
| `torch.compile` through the `vllm_neuron` backend | | **Needs Trn2.** |

CPU tests: `.venv-cpu/bin/python -m pytest tests/qwen3_5 -q` (40 pass).

## Design in four rules

1. **State lives in per-request slots, not batch rows.** The runner calls
   `input_batch.condense()`, which moves requests between rows. State keyed on
   the row would follow the row and hand a moved request another request's
   state. The runner owns a stable `request_id -> slot` map
   (`GdnRunnerState`) and passes `gdn_state_slots` each step.
2. **Padding is masked, not skipped.** The runner lays tokens out as
   `[req0_real, req0_pad, req1_real, req1_pad, ...]` and attributes padding to
   its own request. The model segments on the padded counts and masks with the
   real `actual_num_tokens`; masked positions are exact no-ops.
3. **Prefix caching is off.** Recurrent state is not part of the prefix cache.
   The factory refuses to load with it on, and `step_inputs` raises if a new
   request arrives with computed tokens.
4. **Only full-attention layers have KV specs.** `get_kv_spec()` lists the 8
   full-attention layers. The prefill/decode switch reads layer 3's metadata,
   because layer 0 is Gated DeltaNet and has none.

## State budget

About 50 MiB per slot for the whole model (24 layers x 2 MiB F32 recurrent
state + ~96 KiB conv state), independent of prompt length. At TP=4 that is
~12.5 MiB per rank per slot. Slots default to `max_num_seqs`; override with
`VLLM_NEURON_QWEN35_STATE_SLOTS`. The model logs the total at bind time.

## Bring-up on the Trn2 host

### 1. Install

```sh
git clone <this repo> && cd vllm-neuron-qwen35 && git checkout qwen3_5-gated-deltanet
pip install -e .          # into the Neuron SDK environment that has vLLM
```

### 2. Re-run the CPU tests there

With vLLM installed, `tests/qwen3_5/conftest.py` stops stubbing and the tests
import the real package:

```sh
python -m pytest tests/qwen3_5 -q
```

### 3. Check the weights load

```sh
python -c "from vllm_neuron.model.qwen3_5.config import Qwen3_5Config as C; \
  print(C.from_configs('<snapshot>/config.json'))"
```

`load_weights` raises if any parameter is left unloaded. `A_log` must stay
F32 after loading (`KEEP_STORED_DTYPE` in `model.py`).

### 4. Hook up the runner (`vllm_neuron/vllm/worker/neuron_model_runner.py`)

Line numbers are for `f8abae6`; re-anchor before editing.

**a. Own the state.** In `NeuronModelRunner.__init__`, when the model is
Qwen3.5, create `self.gdn = GdnRunnerState(num_slots)` with the same slot count
the model allocates.

**b. Lifecycle.** When a request is added to `input_batch`, call
`self.gdn.acquire(req_id)`. Next to every `self.input_batch.remove_request(req_id)`
(around lines 1065, 1936, 1967) call `self.gdn.release(req_id)`.

**c. Per-step inputs.** In `_prepare_model_input_impl`, right after
`padded_req_indices` is computed (around line 2835), build:

```python
step = self.gdn.step_inputs(
    req_ids=req_ids,                                   # current row order
    num_computed_tokens=self.input_batch.num_computed_tokens_cpu[:num_reqs],
    actual_num_tokens=actual_num_tokens,
    num_scheduled_tokens_padded=num_scheduled_tokens_padded,  # or actual when no padding_map
    num_reqs_padded=<the batch-row bucket>,
    num_tokens=input_ids.shape[0],
    device=self.device,
)
```

and add to the model kwargs:
`gdn_request_of_token`, `gdn_lengths`, `gdn_state_slots`, `gdn_state_reset`
(the tensors) and `gdn_max_len` (the int).

Do **not** use `req_indices_per_token` from `_build_async_spec_kwargs`
(line ~1767): it assumes every request has the same token count and is only
right for decode and speculative decoding.

**d. Warmup.** Every warmup kwargs dict (e.g. `_build_prefill_synthetic_inputs`,
decode warmup) must include the same five kwargs, from
`self.gdn.warmup_inputs(...)`. The runner requires kwargs to be present
during warmup so `torch.compile` traces one graph per shape.

### 5. CPU-compile smoke

vllm-neuron has a CPU compile path (`VLLM_NEURON_CPU_COMPILE=1`,
`torch.compiler.set_stance("force_eager")`). Use it to shake out tracing
problems before a full Neuron compile.

### 6. Serve

```sh
vllm serve <snapshot> --served-model-name qwen3.5-9b \
  --tensor-parallel-size 4 --max-model-len 4096 \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}'
```

TP=4 uses the chip's four logical cores (LNC=2). Gated DeltaNet shards by
head: 8 of 32 value heads and 4 of 16 key heads per rank.

### 7. Accuracy before speed

Compare greedy continuations and logits against transformers 5.17.0 on the
pinned checkpoint (`Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`)
for short and long prompts, several concurrent requests, and a prompt long
enough to be chunk-prefilled across steps. Fluent output is not evidence:
the failure modes above (wrong state slot, padding leak, BF16 `A_log`) all
produce fluent text.

### 8. Benchmark

`bench/qwen3_5/run_bench.sh`, same protocol as the H100 baselines.

## Found on hardware (2026-09-22, trn2.48xlarge, SDK 2.32.0)

The host runs Amazon Linux 2023 (glibc 2.34) and the pip wheels need glibc
2.35, so everything runs in the AWS container that matches this release:

```sh
sudo docker run -d --name qwen35 --device=/dev/neuron0 \
  -v /mnt/data/work/andre:/work -e HF_HOME=/work/hf --entrypoint sleep \
  public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.24.0.1.1.0-neuronx-py313-sdk2.32.0-ubuntu24.04 infinity
sudo docker exec qwen35 pip install --no-deps -e /work/vllm-neuron-qwen35
```

Four things had to change, in the order they surfaced:

1. **`NEURON_SKIP_EFA_AFFINITY=1`.** The container cannot read the EFA sysfs
   path; without the flag every worker dies at startup. Not our code.
2. **The factory must accept `text_neuron_config`/`vision_neuron_config`.**
   The runner passes those whenever the checkpoint has a vision tower, and
   Qwen3.5's does.
3. **The model must declare `uses_gdn_state`/`num_state_slots`**, which is
   what the runner keys on to build the slot map. Without it the `gdn_*`
   kwargs were `None` and graph extraction died inside Dynamo.
4. **head_dim 256 fits no NKI attention kernel.** Both cap at 128 (one SBUF
   partition). `segmented_attention` raises on the kernel path, so
   `_prefill_attention` calls its dynamo-traceable torch implementation
   directly. This is the largest performance gap: 8 of 32 layers run
   unfused, prefill and decode both.

## Known risks, most likely first

1. Tracing the chunked delta rule: a Python loop over chunks and a 64-step
   forward substitution per chunk unroll into a large graph.
2. TP/SP collectives around the Gated DeltaNet layer.
3. KV replication when TP exceeds 4 (KV heads).

## Performance work after it is correct

- **Attention is the first target.** An NKI kernel for head_dim 256 would
  have to split the head across two SBUF partitions and add the two partial
  score matrices before the softmax; splitting into two 128-wide kernel
  calls is not valid, because softmax does not distribute over the split.
- Fused decode attention: fold `1 + w` into the norm weights, express partial
  rotary as `cos = 1, sin = 0` past channel 64, and find a way to apply the
  gate before `o_proj`.
- Replace the within-chunk forward substitution with an NKI kernel or the
  doubling form `prod (I + N^(2^k))` (6 matmuls for chunk 64).
- Size `max_len` per request bucket instead of per step.
