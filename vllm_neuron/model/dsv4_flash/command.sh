#!/usr/bin/env bash
# DeepSeek-V4-Flash first-fire compile driver.
#
# This is intentionally a queue-ready driver only: submit after the shared
# compile host's GLM-5.3 Round-7 lane has cleared.  Checkpoint sharding is a
# separate pre-fire step (stream_shard.py) and no speculative-decode/MTP path
# is enabled.
set -euo pipefail

: "${COMPILE_CONTRACT:?set COMPILE_CONTRACT to the lane contract JSON}"
: "${COMPILE_RUN_ROOT:?set COMPILE_RUN_ROOT to the lane run root}"

IMAGE="$(jq -r '.stack.container_digest' "$COMPILE_CONTRACT")"
TP="$(jq -r '.compile.tp // 32' "$COMPILE_CONTRACT")"
SEQ="$(jq -r '.compile.sequence_buckets[0] // 4096' "$COMPILE_CONTRACT")"
CTX_BATCH="$(jq -r '.compile.ctx_batch_size // 1' "$COMPILE_CONTRACT")"
TKG_BATCH="$(jq -r '.compile.tkg_batch_size // 1' "$COMPILE_CONTRACT")"
DISABLE_ARGMAX="$(jq -r '.compile.disable_argmax_kernel // false' "$COMPILE_CONTRACT")"
DRY_RUN="$(jq -r '.compile.dry_run // false' "$COMPILE_CONTRACT")"

MODEL_DIR="${MODEL_DIR:-/mnt/compile/hf-cache/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062}"
SRC_DIR="${SRC_DIR:-/mnt/compile/src/vllm-neuron-bravo}"
OUT_DIR="$COMPILE_RUN_ROOT/artifacts/model"

test -s "$MODEL_DIR/config.json"
test -f "$SRC_DIR/vllm_neuron/model/dsv4_flash/neuron_wrapper.py"
mkdir -p "$COMPILE_RUN_ROOT"/{cache,work/tmp,work/nxd,artifacts/model,logs}

# The stream sharder writes the rank files before this driver is queued.  Do
# not let NxDI silently start an unsharded load (or trace against a missing
# rank) when a prerequisite is absent.
for rank in $(seq 0 $((TP - 1))); do
  test -s "$COMPILE_RUN_ROOT/weights/tp${rank}_sharded_checkpoint.safetensors"
done
mkdir -p "$OUT_DIR/weights"
for rank in $(seq 0 $((TP - 1))); do
  cp -al "$COMPILE_RUN_ROOT/weights/tp${rank}_sharded_checkpoint.safetensors" \
    "$OUT_DIR/weights/" 2>/dev/null \
    || cp "$COMPILE_RUN_ROOT/weights/tp${rank}_sharded_checkpoint.safetensors" \
      "$OUT_DIR/weights/"
done

sudo docker run --rm --network none --shm-size 64g --entrypoint bash \
  -e NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  -e NEURON_COMPILE_CACHE_URL=/runroot/cache \
  -e NXD_COMPILER_WORKDIR=/runroot/work/nxd \
  -e TMPDIR=/runroot/work/tmp \
  -e MODEL_PATH=/models/dsv4-flash \
  -e OUT_PATH=/runroot/artifacts/model \
  -e SRC_PATH=/src/vllm-neuron-bravo \
  -e TP="$TP" -e SEQ="$SEQ" -e CTX_BATCH="$CTX_BATCH" -e TKG_BATCH="$TKG_BATCH" \
  -e DISABLE_ARGMAX="$DISABLE_ARGMAX" -e DRY_RUN="$DRY_RUN" \
  -v "$MODEL_DIR:/models/dsv4-flash:ro" \
  -v "$SRC_DIR:/src/vllm-neuron-bravo:ro" \
  -v "$COMPILE_RUN_ROOT:/runroot" \
  "$IMAGE" -lc '
    set -euo pipefail
    export PYTHONPATH="/src/vllm-neuron-bravo:${PYTHONPATH:-}"
    python - <<"PY"
import json
import os
import time

from vllm_neuron.model.dsv4_flash import (
    DeepseekV4FlashInferenceConfig,
    DeepseekV4FlashNeuronInferenceConfig,
    NeuronDeepseekV4FlashForCausalLM,
    build_neuron_config,
)

model_path = os.environ["MODEL_PATH"]
out_path = os.environ["OUT_PATH"]
tp = int(os.environ["TP"])
seq = int(os.environ["SEQ"])
ctx_batch = int(os.environ["CTX_BATCH"])
tkg_batch = int(os.environ["TKG_BATCH"])
disable_argmax = os.environ["DISABLE_ARGMAX"].lower() == "true"
dry_run = os.environ["DRY_RUN"].lower() == "true"

source = DeepseekV4FlashInferenceConfig.from_pretrained(model_path)
neuron = build_neuron_config(
    tp_degree=tp,
    ctx_batch_size=ctx_batch,
    tkg_batch_size=tkg_batch,
    seq_len=seq,
    torch_dtype=source.torch_dtype,
    is_continuous_batching=True,
    disable_argmax_kernel=disable_argmax,
)
inference = DeepseekV4FlashNeuronInferenceConfig(
    neuron_config=neuron,
    source_config=source,
)
# Checkpoint conversion/sharding is a separate streaming step.  The NxDI base
# compile path must not attempt to reopen the 166.87-GB HF snapshot or shard
# it a second time; it traces the wrapper against the pre-sharded rank files.
inference.neuron_config.skip_sharding = True
inference.neuron_config.save_sharded_checkpoint = True
effective = {
    "tp": tp,
    "logical_nc_config": 2,
    "sequence": seq,
    "ctx_batch_size": ctx_batch,
    "tkg_batch_size": tkg_batch,
    "torch_dtype": str(source.torch_dtype),
    "disable_argmax_kernel": disable_argmax,
    "dry_run": dry_run,
    "num_hidden_layers": source.num_hidden_layers,
    "state_cache_expected": 84,
    "wrapper_tree_keys_expected": 1024,
}
with open("/runroot/logs/effective-shape.json", "w", encoding="utf-8") as fh:
    json.dump(effective, fh, indent=2, sort_keys=True)

wrapper = NeuronDeepseekV4FlashForCausalLM(inference)
started = time.time()
wrapper.compile(out_path, dry_run=dry_run)
effective["compile_seconds"] = round(time.time() - started, 3)
with open("/runroot/logs/compile-result.json", "w", encoding="utf-8") as fh:
    json.dump(effective, fh, indent=2, sort_keys=True)
PY
  '

sudo chown -R ec2-user:ec2-user "$COMPILE_RUN_ROOT"
test -s "$COMPILE_RUN_ROOT/artifacts/model/neuron_config.json"
