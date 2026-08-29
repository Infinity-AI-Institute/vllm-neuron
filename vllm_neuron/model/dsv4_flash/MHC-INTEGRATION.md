# DeepSeek-V4-Flash four-stream mHC integration

This change is a dependent, host-only implementation. Its ancestry contains
both open dependency heads exactly:

1. PR #13, `e76a746407dcf434ee71f130ca0bfa76d8bc334d` — lossless I64 to I32
   hash-route conversion.
2. PR #16, `9421e5c0948281790c1380e5e897d6bbf11513bb` — frozen mHC tensor and
   equation contract.
3. This integration PR.

Land in that order. The integration branch merges both exact heads before its
implementation commit; neither dependency currently contains the other.

## Integrated contract

The embedding output is expanded to contiguous `[B, S, 4, D]` streams. Every
one of the 43 layers applies the official FP32 split/Sinkhorn collapse before
attention, the official placement/transposed-combination equation after
attention, and the same pair around the hash/routed MoE branch. The final mHC
head collapses four streams before final RMSNorm and the LM head.

All 261 mHC checkpoint tensors keep their exact source names, FP32 dtype,
shapes, and values. They are replicated on every TP32 rank. The complete
symbolic rank plan is 1,285 tensors and 19,210,553,052 bytes per rank; it does
not claim that rank files have been materialized.

Source configuration JSON round-trips `hc_mult=4`,
`hc_sinkhorn_iters=20`, and `hc_eps=1e-6`. Serialized NxDI configuration maps
are reconstructed into `DeepseekV4FlashInferenceConfig` before use.

## Commands after the dependency merge order is satisfied

Materialize the exact TP32 rank files from the pinned checkpoint:

```bash
SRC_DIR=/path/to/validator-merged/vllm-neuron
MODEL_DIR=/path/to/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062
COMPILE_RUN_ROOT=/path/to/dsv4-tp32-run
PYTHONPATH="$SRC_DIR" python -m vllm_neuron.model.dsv4_flash.stream_shard \
  "$MODEL_DIR" "$COMPILE_RUN_ROOT" \
  --tp-degree 32 --max-chunk-bytes 67108864
```

After `rank-inventory.json`, all 32 rank manifests, compiler provenance, CPU
reference bank, and emitted-contract receipt satisfy
`validate_compile_authorization.py --require-compile-permitted`, the exact
compile launcher is:

```bash
COMPILE_CONTRACT=/path/to/tp32-compile-contract.json \
COMPILE_RUN_ROOT=/path/to/dsv4-tp32-run \
MODEL_DIR=/path/to/deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
SRC_DIR=/path/to/validator-merged/vllm-neuron \
AUTH_EVIDENCE_ROOT=/path/to/reviewed/authorization-evidence \
bash /path/to/validator-merged/vllm-neuron/vllm_neuron/model/dsv4_flash/command.sh
```

These commands are documented only. This change did not materialize the
production checkpoint, invoke a compiler, or make any device, runtime,
correctness, performance, or tokenomics claim.

## Remaining compiler gates

- PR #13, PR #16, and this PR must be validator-merged in the order above.
- The exact checkpoint must be rematerialized so all 32 rank files and
  manifests bind the 1,285-tensor inventories, including 261 replicated FP32
  mHC leaves.
- The compile-authorization packet still requires compiler inventory, the
  canonical CPU reference bank, and the emitted-contract receipt.
- Compiler lowering of the four-stream graph, 20-step Sinkhorn loop, FP32 mHC
  projections, BF16 placement, and 4x4 transposed combination matmul is
  untested. No compiler-success inference is made from host tests.
