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

The integration boundary preserves the authorization lineage from source parent
`2dc3d6a2a125cad006426d77a2998c5dd4b7bd13`; the routed packet's canonical
Git-blob SHA-256 is
`1f8da802a22799cfce4d8a26e0b3676e27cd140045e51715a5e67630174f69b4`. It
freezes the digest-pinned Neuron image, TP32/LNC2/B1/S4096 topology, BF16
checkpoint/compute/cache state, and no-spec/no-MTP state. It also rejects any
future compile path that omits `/mnt/compile/OWNERSHIP.md`, cap-2,
`systemd-run --unit ... --nice=15`, network-none, or atomic `.partial` output
publication. Every execution and result claim remains false.

The dependent integration routes all 261 tensors losslessly, expands embeddings
to four hidden streams, applies mHC pre/post around attention and MoE in all 43
layers, and collapses the streams before final norm/head. The symbolic TP32 plan
is complete, but rank files are not materialized and the compile-authorization
packet remains fail-closed. No compile, runtime, correctness, performance, or
tokenomics claim is made.
