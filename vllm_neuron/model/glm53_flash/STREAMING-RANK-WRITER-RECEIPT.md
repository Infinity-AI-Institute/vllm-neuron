# GLM-5.3-Flash streaming rank writer receipt

This host-only change implements the checkpoint-output substrate requested by
`trainium-autoresearcher#45`. It builds on merge
`cbc321829695ea10503ef8ae65f17ae309489b10` and does not use the older
`Glm52MoeDsa` converter.

## Memory discipline

The writer creates the complete SafeTensors header from a declared TP-rank
inventory, reserves the output file, and writes bounded CPU tensor chunks
directly to their final byte ranges. It does not create a converted full-model
dictionary or a full-rank tensor dictionary. Peak writer payload memory is
therefore bounded by `max_chunk_bytes`, reported alongside the full logical
rank size in the generated manifest.

Each tensor tracks byte-range coverage. Overlap, out-of-range writes, missing
ranges, undeclared tensors, dtype drift, and chunks larger than the configured
bound fail before publication. The final SafeTensors file is atomically renamed
from a private partial path only after the complete inventory is covered. The
manifest records the rank-inventory SHA-256, checkpoint SHA-256, source pins,
chunk count, and observed resource bound.

## Source discipline

`IndexedTensorReader` first runs the immutable revision/config/index preflight.
It then audits all actual SafeTensors headers against the index before loading
payloads:

- every referenced shard must exist and no extra shard is accepted;
- every indexed tensor must be in exactly its routed shard;
- missing, duplicate, or orphan tensors fail;
- every FP8 tensor must have a reciprocal `_scale_inv` partner;
- every reciprocal scale must have a weight and a floating dtype;
- header auditing loads zero payload bytes.

Payload conversion remains one source group at a time. Paired block-FP8
weight/scale tensors use reciprocal multiplication through the merged converter
primitive; unscaled FP8 holdouts fail closed.

## Scope boundary

This patch proves the transactional writer and source-reader contracts with a
small complete TP-rank fixture. Architecture-specific target-name and sharding
plans are passed as an explicit `RankInventory` plus lazy `chunk_factory`; they
cannot be inferred silently. A later adapter integration must generate that
inventory from the Glm5Next module tree and then prove its exact contract hash
before any compile or runtime use.

No model weights, r7i compile slot, Trn2 device, runtime correctness,
performance, or tokenomics claim is part of this receipt.
