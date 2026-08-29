# DeepSeek-V4-Flash host-only enablement source

This package stages the `deepseek-ai/DeepSeek-V4-Flash-0731` adapter at the
immutable revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`. It contains
configuration, checkpoint conversion, wrapper wiring, and a TP32 checkpoint
writer. Its presence is not a compile, runtime-correctness, performance, or
tokenomics claim.

## Checkpoint publication

`stream_shard.py` reuses the transactional writer merged for GLM-5.3 while
supplying DSv4-specific immutable metadata. Before loading tensor payloads it:

- verifies the snapshot-directory revision and exact config/index SHA-256;
- requires exactly 48 source shards;
- compares every SafeTensors header with its index route;
- rejects missing, extra, duplicate, orphaned, or misrouted tensors.

For each requested TP rank it performs two bounded passes. The first derives a
complete `RankInventory` and discards each converted layer. The second emits
the same tensors in chunks no larger than 64 MiB into a private SafeTensors
file. Gaps, overlaps, dtype drift, undeclared tensors, or incomplete coverage
abort and remove the partial file. A completed checkpoint and manifest are
published only after full coverage and `fsync`.

The current conversion path still needs a production 32-rank inventory and
resource receipt before compile authorization. In particular, the writer's
chunk bound does not by itself prove the peak memory of every architecture
conversion primitive.

## Compile authorization

`tp32_compile_authorization.json` freezes TP32/LNC2/B1/S4096, BF16 emitted
weights/compute/cache, greedy argmax, and no FP8 KV, speculation, MTP, or
DSpark. `validate_compile_authorization.py` keeps compilation on HOLD until all
of these machine-verifiable receipts are present:

1. validator-merged source provenance;
2. exact 48-shard header, routing, and payload identity;
3. all 32 rank inventories/manifests and checkpoint hashes;
4. digest-bound compiler/runtime package and source inventory;
5. exact-source canonical four-prompt x ten-token full-logit CPU bank;
6. an exact emitted-contract receipt.

`command.sh` invokes that validator with `--require-compile-permitted` before
creating run directories, copying rank files, or invoking Docker. The same
preflight hashes the resolved launch snapshot (config, index, tokenizer, and
all 48 shards), all 32 resolved launch rank files, and requires the executing
source checkout to be at the exact clean validator-merged HEAD/tree recorded in
the reviewed evidence. There is no TP16 fallback.

## Host-only checks

```bash
python -m pytest -q \
  test/unit/model/test_dsv4_streaming_rank_writer.py \
  test/unit/model/test_dsv4_compile_authorization.py \
  test/unit/model/test_glm53_streaming_rank_writer.py
python vllm_neuron/model/dsv4_flash/validate_compile_authorization.py
python vllm_neuron/model/dsv4_flash/validate_compile_authorization.py \
  --require-compile-permitted  # expected exit 2 until every receipt exists
```

No r7i compile slot or Trainium device is required for these checks.
