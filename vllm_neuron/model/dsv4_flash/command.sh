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
: "${MODEL_DIR:?set MODEL_DIR to the reviewed DeepSeek-V4 snapshot}"
: "${SRC_DIR:?set SRC_DIR to the exact validator-merged source checkout}"
: "${AUTH_EVIDENCE_ROOT:?set AUTH_EVIDENCE_ROOT to the reviewed host-only evidence directory}"

COMPILE_CONTRACT="$(realpath "$COMPILE_CONTRACT")"
COMPILE_RUN_ROOT="$(realpath "$COMPILE_RUN_ROOT")"
MODEL_DIR="$(realpath "$MODEL_DIR")"
SRC_DIR="$(realpath "$SRC_DIR")"
AUTH_EVIDENCE_ROOT="$(realpath "$AUTH_EVIDENCE_ROOT")"
RANK_SOURCE_DIR="$COMPILE_RUN_ROOT/weights"

IMAGE="$(jq -er '.stack.container_digest' "$COMPILE_CONTRACT")"
TP="$(jq -er '.compile.tp' "$COMPILE_CONTRACT")"
SEQ="$(jq -er '.compile.sequence_buckets | if length == 1 then .[0] else error("one sequence bucket required") end' "$COMPILE_CONTRACT")"
CTX_BATCH="$(jq -er '.compile.ctx_batch_size' "$COMPILE_CONTRACT")"
TKG_BATCH="$(jq -er '.compile.tkg_batch_size' "$COMPILE_CONTRACT")"
DISABLE_ARGMAX="$(jq -r '.compile.disable_argmax_kernel | if type == "boolean" then tostring else error("compile.disable_argmax_kernel must be a boolean") end' "$COMPILE_CONTRACT")"
DRY_RUN="$(jq -r '.compile.dry_run | if type == "boolean" then tostring else error("compile.dry_run must be a boolean") end' "$COMPILE_CONTRACT")"
CONTRACT_SLUG="$(jq -er '.contract_slug' "$COMPILE_CONTRACT")"

AUTH_PACKET="${AUTH_PACKET:-$SRC_DIR/vllm_neuron/model/dsv4_flash/tp32_compile_authorization.json}"
AUTH_PACKET="$(realpath "$AUTH_PACKET")"
OUT_DIR="$COMPILE_RUN_ROOT/artifacts/model"
test -s "$MODEL_DIR/config.json"
test -f "$SRC_DIR/vllm_neuron/model/dsv4_flash/neuron_wrapper.py"
test "$TP" -eq 32
"${PYTHON:-python3}" "$SRC_DIR/vllm_neuron/model/dsv4_flash/validate_compile_authorization.py" \
  --packet "$AUTH_PACKET" \
  --compile-contract "$COMPILE_CONTRACT" \
  --evidence-root "$AUTH_EVIDENCE_ROOT" \
  --model-dir "$MODEL_DIR" \
  --compile-run-root "$COMPILE_RUN_ROOT" \
  --rank-source "$RANK_SOURCE_DIR" \
  --source-dir "$SRC_DIR" \
  --require-compile-permitted
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
  -e CONTRACT_SLUG="$CONTRACT_SLUG" \
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
contract_slug = os.environ["CONTRACT_SLUG"]

source = DeepseekV4FlashInferenceConfig.from_pretrained(model_path)
neuron = build_neuron_config(
    tp_degree=tp,
    ctx_batch_size=ctx_batch,
    tkg_batch_size=tkg_batch,
    seq_len=seq,
    torch_dtype=source.torch_dtype,
    is_continuous_batching=True,
    disable_argmax_kernel=disable_argmax,
    extra={"logical_nc_config": 2},
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
    "contract_slug": contract_slug,
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
    "wrapper_tree_keys_expected": 1285,
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

# The run root is created by this driver as the invoking user, but the container
# writes into it as root.  Hand it back to whoever launched the compile -- not to
# `ec2-user`, which does not exist on research-7 (`id ec2-user` -> no such user).
# Under `set -e` that failure aborted the driver AFTER a successful compile,
# discarding the run at its very last step.
sudo chown -R "$(id -u):$(id -g)" "$COMPILE_RUN_ROOT"
test -s "$COMPILE_RUN_ROOT/artifacts/model/neuron_config.json"
jq --arg slug "$CONTRACT_SLUG" '. + {contract_slug: $slug}' \
  "$COMPILE_RUN_ROOT/artifacts/model/neuron_config.json" \
  > "$COMPILE_RUN_ROOT/artifacts/model/neuron_config.json.tmp"
mv "$COMPILE_RUN_ROOT/artifacts/model/neuron_config.json.tmp" \
  "$COMPILE_RUN_ROOT/artifacts/model/neuron_config.json"
