#!/usr/bin/env bash
# Qwen3.5-9B on Trainium2 under vLLM-Neuron: same protocol as the H100 baselines.
# 128/512/2048 input tokens, 128 output tokens, concurrency 1, 12 prompts.
#   run_bench.sh <snapshot dir> [port] [tp]
set -u
MODEL=$1
PORT=${2:-8000}
TP=${3:-4}
OUT=${OUT:-results/qwen3_5_9b_trn2}
mkdir -p "$OUT" logs

vllm serve "$MODEL" --served-model-name qwen3.5-9b --port "$PORT" \
  --tensor-parallel-size "$TP" --max-model-len 4096 \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  > logs/serve.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT

for attempt in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
  if ! kill -0 "$SERVER" 2>/dev/null; then echo "server exited; see logs/serve.log"; exit 1; fi
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "not healthy; see logs/serve.log"; exit 1; }
echo "healthy after $((attempt * 10))s"

# Discarded warm-up so the first measured scenario does not absorb compilation.
vllm bench serve --backend vllm --model "$MODEL" --served-model-name qwen3.5-9b \
  --host 127.0.0.1 --port "$PORT" --dataset-name random \
  --random-input-len 128 --random-output-len 128 --ignore-eos \
  --max-concurrency 1 --num-prompts 4 --seed 99 > logs/warmup.log 2>&1

for INPUT in 128 512 2048; do
  echo "input=$INPUT"
  vllm bench serve --backend vllm --model "$MODEL" --served-model-name qwen3.5-9b \
    --host 127.0.0.1 --port "$PORT" --dataset-name random \
    --random-input-len "$INPUT" --random-output-len 128 --ignore-eos \
    --max-concurrency 1 --num-prompts 12 --seed 1729 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95 \
    --save-result --result-dir "$OUT" --result-filename "input-$INPUT.json" \
    > "logs/bench-$INPUT.log" 2>&1
  echo "input=$INPUT exit $?"
done
