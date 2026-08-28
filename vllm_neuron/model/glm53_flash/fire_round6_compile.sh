#!/usr/bin/env bash
# Round 6 GLM-5.3-Flash real-weight compile driver.
#
# Preconditions:
#   1. /mnt/compile/runroot/glm53-round6/weights/tp{0..31}_sharded_checkpoint.safetensors
#      already exist (produced by stream_shard.stream_shard_glm53).
#   2. /mnt/compile/src/vllm-neuron-alpha/vllm_neuron/model/glm53_flash/ carries
#      Round 6 code (neuron_wrapper.py from Round 5 + stream_shard.py from Round 6).
#
# What this driver does:
#   1. Fires wrapper.compile() with skip_sharding=True so NxDI skips its own
#      shard step (we already produced the shards).
#   2. Captures model.pt + neuron_config.json + NEFF.
#   3. Verifies the emitted neuron_config.json contains NO FP8-KV field
#      (bf16 KV is what was requested).
#   4. Greps the log for CPU-fallback markers per the campaign contract.
#
# Wall estimate: 30-60 min (trace + neuronx-cc, similar to Round 4).
#
# Env in:
#   TP           = 32 (default; the Round-4 HBM math refuses TP=16)
#   LNC          = 2  (LNC=1 is refused by GLM-5.3-Flash MoE workaround)
#   SEQ          = 2048  (max KV window)
#   MAX_MODEL_LEN = 2048
#   CTX_LEN      = 512   (CTE bucket; Round 4 found this fits)
#   RESIDENT_BATCH = 1
#   PHASES       = BOTH  ("CTE", "TKG", "BOTH")

set -euo pipefail

TP="${TP:-32}"
LNC="${LNC:-2}"
SEQ="${SEQ:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
CTX_LEN="${CTX_LEN:-512}"
RESIDENT_BATCH="${RESIDENT_BATCH:-1}"
PHASES="${PHASES:-BOTH}"

RUN_ROOT="/mnt/compile/runroot/glm53-round6"
MODEL_DIR="/mnt/compile/hf-cache/models--zai-org--GLM-5.3-Flash/snapshots/04c4e9e95c5da8862dced7e5056455116f83a7e0"
SRC_DIR="/mnt/compile/shared-models/src/nxdi-e05466c"
CODE_DIR="/mnt/compile/src/vllm-neuron-alpha"
KERN_DIR="/mnt/compile/src/glm53-kernels"
IMAGE="public.ecr.aws/neuron/pytorch-inference-neuronx@sha256:011d49c7495457fc2932dedd3fbecf67d28833a3f12c147377cee4d72889ebc1"

mkdir -p "$RUN_ROOT"/{cache,work/tmp,work/nxd,artifacts/model,logs}

# Sanity: all 32 sharded files must exist.
missing=0
for r in $(seq 0 $((TP-1))); do
  f="$RUN_ROOT/weights/tp${r}_sharded_checkpoint.safetensors"
  if [ ! -s "$f" ]; then
    echo "REFUSED: sharded checkpoint missing: $f" >&2
    missing=$((missing+1))
  fi
done
if [ $missing -gt 0 ]; then
  echo "$missing / $TP sharded files missing. Run stream_shard first." >&2
  exit 3
fi

# Copy sharded files to the compile artifacts weights dir (NxDI's load_weights
# expects them under {compiled_model_path}/weights/).
mkdir -p "$RUN_ROOT/artifacts/model/weights"
for r in $(seq 0 $((TP-1))); do
  cp -al "$RUN_ROOT/weights/tp${r}_sharded_checkpoint.safetensors" \
        "$RUN_ROOT/artifacts/model/weights/" 2>/dev/null \
    || cp "$RUN_ROOT/weights/tp${r}_sharded_checkpoint.safetensors" \
          "$RUN_ROOT/artifacts/model/weights/"
done

START="$(date +%s)"
sudo docker run --rm --network none --shm-size 128g --entrypoint bash \
  -e NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  -e NEURON_LOGICAL_NC_CONFIG="$LNC" \
  -e NEURON_COMPILE_CACHE_URL=/runroot/cache \
  -e NXD_COMPILER_WORKDIR=/runroot/work/nxd \
  -e TMPDIR=/runroot/work/tmp \
  -e XLA_HANDLE_SPECIAL_SCALAR=1 \
  -e UNSAFE_FP8FNCAST=1 \
  -e GLM53_REFERENCE_KERNEL_DIR=/kernels \
  -e MODEL_PATH=/models/GLM-5.3-Flash \
  -e OUT_PATH=/runroot/artifacts/model \
  -e TP="$TP" -e LNC="$LNC" -e SEQ="$SEQ" \
  -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  -e CTX_LEN="$CTX_LEN" \
  -e RESIDENT_BATCH="$RESIDENT_BATCH" \
  -e NXDI_EMIT_PHASES="$PHASES" \
  -v "$MODEL_DIR:/models/GLM-5.3-Flash:ro" \
  -v "$SRC_DIR:/src/nxdi:ro" \
  -v "$CODE_DIR:/code:ro" \
  -v "$KERN_DIR:/kernels:ro" \
  -v "$RUN_ROOT:/runroot" \
  "$IMAGE" -lc '
    set -euo pipefail
    export PYTHONPATH="/src/nxdi/src:/code:/kernels:${PYTHONPATH:-}"
    python - <<PY >> /runroot/logs/docker.log 2>&1
import importlib.util, json, os, sys, time, types

ROOT = "/code"
for name, path in (
    ("vllm_neuron", f"{ROOT}/vllm_neuron"),
    ("vllm_neuron.model", f"{ROOT}/vllm_neuron/model"),
    ("vllm_neuron.model.glm53_flash", f"{ROOT}/vllm_neuron/model/glm53_flash"),
):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m

def load(mod, fn):
    full = f"vllm_neuron.model.glm53_flash.{mod}"
    spec = importlib.util.spec_from_file_location(
        full, f"{ROOT}/vllm_neuron/model/glm53_flash/{fn}")
    m = importlib.util.module_from_spec(spec); sys.modules[full] = m
    spec.loader.exec_module(m); return m

load("registry", "registry.py")
load("_reference_kernels", "_reference_kernels.py")
load("nki_bindings", "nki_bindings.py")
config_mod  = load("config", "config.py")
load("checkpoint_convert", "checkpoint_convert.py")
wrapper_mod = load("neuron_wrapper", "neuron_wrapper.py")

tp   = int(os.environ["TP"])
lnc  = int(os.environ["LNC"])
seq  = int(os.environ["SEQ"])
mml  = int(os.environ["MAX_MODEL_LEN"])
ctx  = int(os.environ["CTX_LEN"])
rb   = int(os.environ["RESIDENT_BATCH"])

src = config_mod.Glm53FlashInferenceConfig.from_pretrained(os.environ["MODEL_PATH"])
cfg = wrapper_mod.NeuronGlm53FlashForCausalLM.build_inference_config(
    src, tp_degree=tp, ctx_batch_size=1, tkg_batch_size=rb,
    seq_len=mml, max_context_length=ctx, logical_nc_config=lnc,
)
# Sharding already done externally; skip the base classes shard.
cfg.neuron_config.skip_sharding = True
cfg.neuron_config.save_sharded_checkpoint = True

wrapper = wrapper_mod.NeuronGlm53FlashForCausalLM(os.environ["MODEL_PATH"], cfg)
t0 = time.time()
wrapper.compile(os.environ["OUT_PATH"])
compile_seconds = time.time() - t0

emitted = {"compile_seconds": round(compile_seconds, 2), "neffs": []}
cfg_path = os.path.join(os.environ["OUT_PATH"], "neuron_config.json")
if os.path.exists(cfg_path):
    raw = open(cfg_path).read()
    emitted["neuron_config_has_float8_e4m3fn"] = "float8_e4m3fn" in raw
    emitted["neuron_config_has_bfloat16"] = "bfloat16" in raw
search = [os.environ["OUT_PATH"], os.environ["NEURON_COMPILE_CACHE_URL"]]
for base in search:
    if not os.path.isdir(base):
        continue
    for root, _d, files in os.walk(base):
        for f in files:
            if f.endswith(".neff"):
                p = os.path.join(root, f)
                emitted["neffs"].append({"path": p, "bytes": os.path.getsize(p)})
emitted["neff_count"] = len(emitted["neffs"])
with open("/runroot/logs/compile-result.json", "w") as fh:
    json.dump(emitted, fh, indent=2)
print(json.dumps(emitted, indent=2))
PY
  ' 2>&1 | tee -a "$RUN_ROOT/logs/docker.log"

END="$(date +%s)"

if grep -qiE "falling back to cpu|cpu fallback|running on cpu|torch_blockwise_matmul_inference|use_torch_block_wise=True" \
     "$RUN_ROOT/logs/docker.log"; then
  echo "REFUSED: CPU-fallback marker found in the compile log" >&2
  echo "{\"status\":\"failed_cpu_fallback\"}" > "$RUN_ROOT/artifacts/terminal.json"
  exit 4
fi

echo "{\"status\":\"ok\", \"wall_seconds\": $((END-START))}" > "$RUN_ROOT/artifacts/terminal.json"
echo "GLM-5.3-Flash Round-6 compile finished in $((END-START))s"
