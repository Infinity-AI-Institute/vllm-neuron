# DeepSeek-V4-Flash main-model mHC contract

This host-only contract freezes the 261 Hyper-Connection tensors in
`deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062`:
three head tensors and six tensors for each of 43 layers. All are FP32 and
replicated on every TP32 rank. Their sorted key-set SHA-256 is
`ad07358ebd20fce2a30d3ddd889880250d5280c523eb474a226a2982833d58d6`.

The module also provides a CPU-portable implementation of the pinned official
FP32 split/Sinkhorn, pre, post, and head equations. Focused tests compare it to
an independent transcription and reject key, shape, dtype, multiplier,
iteration-count, epsilon, or mutation drift.

The integration boundary composes with the exact existing authorization packet
from source parent `2dc3d6a2a125cad006426d77a2998c5dd4b7bd13` (canonical Git-blob SHA-256
`b9cdcfdaabccfcd807a6fd5cf9cc19f03730368796f5b0d9dde86bb7c5986822`). It
freezes the digest-pinned Neuron image, TP32/LNC2/B1/S4096 topology, BF16
checkpoint/compute/cache state, and no-spec/no-MTP state. It also rejects any
future compile path that omits `/mnt/compile/OWNERSHIP.md`, cap-2,
`systemd-run --unit ... --nice=15`, network-none, or atomic `.partial` output
publication. Every execution and result claim remains false.

This does **not** mark the tensors routable yet. The current NxDI wrapper uses a
single hidden stream and ordinary residual adds; correct DeepSeek-V4-Flash uses
four hidden streams, mHC pre/post around both branches in every layer, and an
mHC collapse before the final norm/head. Compile and runtime remain forbidden
until a separately reviewed integration consumes this primitive across the
full model tree, the TP32 header audit is rerun after the I64 route conversion
lands, and every existing compile-authorization receipt passes. No correctness,
performance, or tokenomics claim is made.
