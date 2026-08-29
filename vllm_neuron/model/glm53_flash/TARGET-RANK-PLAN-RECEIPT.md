# GLM-5.3-Flash target-rank plan receipt

This host-only change advances `trainium-autoresearcher#45` from the
transactional streaming writer merged as
`2c2aafffa733880a427c7a3492e5a2c97f35f093` to a complete production
Glm5Next TP=32 rank contract. It makes no compile, runtime-correctness,
performance, or tokenomics claim.

## Immutable source binding

- checkpoint revision: `04c4e9e95c5da8862dced7e5056455116f83a7e0`
- `config.json` SHA-256:
  `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`
- `model.safetensors.index.json` SHA-256:
  `3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05`

The builder runs the merged immutable metadata preflight, requires the
qualified TP=32 topology, and bijectively consumes every indexed text weight
outside the separately excluded vision and layer-45 MTP trees. Missing,
duplicate, unexpected, or unmapped text weights fail closed.

## Rank contract

For rank 0 with a 64 MiB output-chunk ceiling:

- target tensors / operations: `1,262`
- logical target bytes: `19,859,704,056`
- rank inventory SHA-256:
  `0d7380ce03aeadb73d2b9dcb9a015c789a24a1a6220696717736c42cbe5d096a`
- full source-to-target plan SHA-256:
  `9c6e5c5a76d1cc11f2732a3b472ee9da92cbaa5944fd84ae0a7d7a3eb651511f`

The plan includes embedding/head shards, replicated norms and mHC state,
KDA projections and per-head-interleaved short convolution, DSA/MLA and
indexer mappings, dense MLPs, shared experts, FP32 routing state, and lazy
per-expert routed-MoE fusion. Rank identity is part of both hashes, so rank 31
cannot inherit rank 0's approval.

## Memory and publication discipline

SafeTensors payloads are read lazily by bounded source slices. Block-FP8
slices load only intersecting reciprocal-scale tiles and crop tile edges
before multiplication. Routed experts are converted one expert at a time;
the writer still writes directly to predeclared final byte ranges and retains
its overlap/gap/dtype/chunk-bound checks and atomic publication. The emitted
SafeTensors metadata and JSON manifest now bind the plan SHA-256 in addition
to the source and inventory hashes. No full-model or full-rank tensor
dictionary is constructed.

## Host-only evidence

- focused converter/writer/rank-plan tests: `27 passed`
- immutable production metadata contract test: passed (not skipped)
- positive layout coverage: row/column TP shards, KDA interleave, routed-MoE
  gate/up fusion and down projection
- adversarial coverage: cross-tile FP8 slicing, source shape drift, unexpected
  production text tensor, partial/overlap/oversize output, and provenance drift
- `ruff check`: passed
- `ruff format --check`: passed
- `py_compile`: passed
- `git diff --check`: passed

A range-only audit of all 62 production SafeTensors headers also passed. It
read 10,684,096 header bytes and zero tensor payload bytes, proved exact
header/index agreement for all 76,108 indexed tensors, and matched all 74,001
planned source shapes: 37,534 weights plus all 36,467 reciprocal scales. Audit
receipt SHA-256:
`3121e936c2bbb1012ecf3f805e2d2c2986c22d56a86d34d85ca87ba65e775d76`.

No r7i compilation slot, Trn2 device, model weights, excluded model, `main`
branch, or runtime/tokenomics path was touched.
