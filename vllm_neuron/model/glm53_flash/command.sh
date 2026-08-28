#!/usr/bin/env bash
# GLM-5.3-Flash NxDI compile-integration driver.
#
# Mirrors the qwen35-2b/9b/4b command.sh pattern on the compile host
# (`/mnt/compile/shared-images/qwen35-*-command.sh`) so the fleet-scheduler
# can enqueue GLM-5.3-Flash lanes the same way it enqueues Qwen35 lanes.
#
# Contract fields (jq-read from `$COMPILE_CONTRACT`):
#   .stack.container_digest        - full image sha256 (must be
#                                     `sha256:011d49c7...` unless the caller
#                                     has verified the MoE blockwise-mm
#                                     workaround is unnecessary)
#   .compile.tp                    - tensor-parallel degree (target: 8)
#   .compile.sequence_buckets[0]   - active sequence bucket for this lane
#   .compile.ctx_batch_size        - CTE batch size
#   .compile.tkg_batch_size        - TKG batch size
#   .compile.disable_argmax_kernel - optional (default false)
#   .compile.emit_phases           - optional "BOTH"|"CTE"|"TKG" (default BOTH)
#   .compile.one_layer_smoke       - optional bool; when true, overrides
#                                     `num_hidden_layers=1` via
#                                     `build_one_layer_smoke_config`
#   .compile.dry_run               - optional bool; forwarded to
#                                     `NeuronBaseForCausalLM.compile(dry_run=...)`
#
# Env in from the orchestrator:
#   COMPILE_CONTRACT               - path to the lane's contract JSON
#   COMPILE_RUN_ROOT               - lane run root; artifacts go under
#                                     `$COMPILE_RUN_ROOT/artifacts/model/`
#   MODEL_DIR                      - HF snapshot dir (default set below)
#   SRC_DIR                        - vllm-neuron-alpha src (default set below)
#   NXDI_SRC                       - NxDI-inside-container path (default
#                                     `/opt/aws_neuronx_venv_pytorch_2_5/lib/python3.10/site-packages/neuronx_distributed_inference`)
#
# The full 5.3-Flash BF16 upcast is 600 GiB and the FP8 weight budget is 306
# GiB — do NOT run a full compile from this driver until Round 2 lands the
# real KDA/DSA/MoE/mHC block stack.  The 1-layer smoke path is safe:
# `initialize_model_weights=False` avoids the load, `dry_run=True` avoids the
# XLA lowering, and the shell forward is bounded to `hidden_size × hidden_size`
# per layer.

set -euo pipefail

IMAGE="$(jq -r '.stack.container_digest' "$COMPILE_CONTRACT")"
TP="$(jq -r '.compile.tp' "$COMPILE_CONTRACT")"
SEQ="$(jq -r '.compile.sequence_buckets[0]' "$COMPILE_CONTRACT")"
CTX_BATCH="$(jq -r '.compile.ctx_batch_size' "$COMPILE_CONTRACT")"
TKG_BATCH="$(jq -r '.compile.tkg_batch_size' "$COMPILE_CONTRACT")"
DISABLE_ARGMAX_KERNEL="$(jq -r '.compile.disable_argmax_kernel // false' "$COMPILE_CONTRACT")"
EMIT_PHASES="$(jq -r '.compile.emit_phases // "BOTH"' "$COMPILE_CONTRACT")"
ONE_LAYER_SMOKE="$(jq -r '.compile.one_layer_smoke // false' "$COMPILE_CONTRACT")"
DRY_RUN="$(jq -r '.compile.dry_run // false' "$COMPILE_CONTRACT")"

MODEL_DIR="${MODEL_DIR:-/mnt/compile/hf-cache/models--zai-org--GLM-5.3-Flash/snapshots/04c4e9e95c5da8862dced7e5056455116f83a7e0}"
SRC_DIR="${SRC_DIR:-/mnt/compile/src/vllm-neuron-alpha}"

test -s "$MODEL_DIR/config.json"
test -f "$SRC_DIR/vllm_neuron/model/glm53_flash/neuron_wrapper.py"
mkdir -p "$COMPILE_RUN_ROOT"/{cache,work/tmp,work/nxd,artifacts/model,logs}

START="$(date +%s)"
sudo docker run --rm --network none --shm-size 64g --entrypoint bash \
  -e NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  -e NEURON_COMPILE_CACHE_URL=/runroot/cache \
  -e NXD_COMPILER_WORKDIR=/runroot/work/nxd \
  -e TMPDIR=/runroot/work/tmp \
  -e MODEL_PATH=/models/glm53-flash \
  -e OUT_PATH=/runroot/artifacts/model \
  -e SRC_PATH=/src/vllm-neuron-alpha \
  -e TP="$TP" -e SEQ="$SEQ" -e CTX_BATCH="$CTX_BATCH" -e TKG_BATCH="$TKG_BATCH" \
  -e DISABLE_ARGMAX_KERNEL="$DISABLE_ARGMAX_KERNEL" \
  -e NXDI_EMIT_PHASES="$EMIT_PHASES" \
  -e ONE_LAYER_SMOKE="$ONE_LAYER_SMOKE" \
  -e DRY_RUN="$DRY_RUN" \
  -v "$MODEL_DIR:/models/glm53-flash:ro" \
  -v "$SRC_DIR:/src/vllm-neuron-alpha:ro" \
  -v "$COMPILE_RUN_ROOT:/runroot" \
  "$IMAGE" -lc '
    set -euo pipefail
    export PYTHONPATH="/src/vllm-neuron-alpha:${PYTHONPATH:-}"
    python - <<"PY"
import gc
import json
import os
import time

from vllm_neuron.model.glm53_flash import (
    Glm53FlashInferenceConfig,
    NeuronGlm53FlashForCausalLM,
)

model_path = os.environ["MODEL_PATH"]
out_path = os.environ["OUT_PATH"]
tp = int(os.environ["TP"])
seq = int(os.environ["SEQ"])
ctx_batch = int(os.environ["CTX_BATCH"])
tkg_batch = int(os.environ["TKG_BATCH"])
disable_argmax_kernel = os.environ["DISABLE_ARGMAX_KERNEL"].lower() == "true"
one_layer_smoke = os.environ["ONE_LAYER_SMOKE"].lower() == "true"
dry_run = os.environ["DRY_RUN"].lower() == "true"

source_config = Glm53FlashInferenceConfig.from_pretrained(model_path)

if one_layer_smoke:
    inference_config = NeuronGlm53FlashForCausalLM.build_one_layer_smoke_config(
        source_config,
        tp_degree=tp,
        ctx_batch_size=ctx_batch,
        tkg_batch_size=tkg_batch,
        seq_len=seq,
    )
else:
    inference_config = NeuronGlm53FlashForCausalLM.build_inference_config(
        source_config,
        tp_degree=tp,
        ctx_batch_size=ctx_batch,
        tkg_batch_size=tkg_batch,
        seq_len=seq,
        disable_argmax_kernel=disable_argmax_kernel,
    )

shape = {
    "tp": tp,
    "sequence": seq,
    "ctx_batch_size": ctx_batch,
    "tkg_batch_size": tkg_batch,
    "disable_argmax_kernel": disable_argmax_kernel,
    "one_layer_smoke": one_layer_smoke,
    "dry_run": dry_run,
    "emit_phases": os.environ.get("NXDI_EMIT_PHASES", "BOTH"),
    "num_hidden_layers": inference_config.num_hidden_layers,
    "vocab_size": inference_config.vocab_size,
    "hidden_size": inference_config.hidden_size,
    "cache_abi": NeuronGlm53FlashForCausalLM.GLM53_SOURCE_CACHE_ABI,
}
with open("/runroot/logs/effective-shape.json", "w") as h:
    json.dump(shape, h, indent=2, sort_keys=True)

wrapper = NeuronGlm53FlashForCausalLM(model_path, inference_config)
start = time.perf_counter()
wrapper.compile(out_path, dry_run=dry_run)
shape["compile_seconds"] = time.perf_counter() - start
with open("/runroot/logs/compile-result.json", "w") as h:
    json.dump(shape, h, indent=2, sort_keys=True)

del wrapper
gc.collect()
PY
  '
sudo chown -R ec2-user:ec2-user "$COMPILE_RUN_ROOT"

# The shell driver in dry_run mode does not emit a `model.pt`; the smoke
# passes if the neuron_config JSON lands under `$OUT_PATH` (NxDI's compile
# writes it before the trace).  A non-smoke compile leaves the traced model
# at `$OUT_PATH/model.pt`; the shard weights are written by a later
# `load_weights` invocation on the running host, not here.
if [ "$DRY_RUN" != "true" ] && [ "$ONE_LAYER_SMOKE" != "true" ]; then
  test -s "$COMPILE_RUN_ROOT/artifacts/model/model.pt"
fi
test -s "$COMPILE_RUN_ROOT/artifacts/model/neuron_config.json"

END="$(date +%s)"
model_pt="$COMPILE_RUN_ROOT/artifacts/model/model.pt"
if [ -s "$model_pt" ]; then
  MODEL_PT_SHA="$(sha256sum "$model_pt" | awk '{print $1}')"
  MODEL_PT_BYTES="$(stat -c %s "$model_pt")"
else
  MODEL_PT_SHA="dry-run-no-model-pt"
  MODEL_PT_BYTES=0
fi
printf '{"wall_seconds":%s,"model_pt_sha256":"%s","model_pt_bytes":%s,"dry_run":%s,"one_layer_smoke":%s}\n' \
  "$((END-START))" \
  "$MODEL_PT_SHA" \
  "$MODEL_PT_BYTES" \
  "$DRY_RUN" \
  "$ONE_LAYER_SMOKE" \
  > "$COMPILE_RUN_ROOT/logs/terminal.json"
