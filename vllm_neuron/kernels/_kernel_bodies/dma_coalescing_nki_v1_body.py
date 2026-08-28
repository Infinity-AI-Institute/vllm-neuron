# SPDX-License-Identifier: Apache-2.0
# NKI kernel body -- file-import version -- slug: dma_coalesced_gather.nki_v1
#
# This file exists as a PHYSICAL file (not an `exec()`-populated namespace)
# because @nki.jit's KernelRewriter.reparse_function calls
# `inspect.getsource(<decorated_fn>)` at compile time, and that call requires
# a real file on disk to walk. `exec(source_str, ns)` leaves the function's
# `__module__` as "<string>" and every NKI compile then raises
# `OSError: could not get source code`. See
# EXEC-TO-FILE-IMPORT-PATCH-2026-08-28.md for the full bug diagnosis
# discovered on the Kimi K3 Route B fire.

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import dge_mode, dma_engine, oob_mode

# Per-invocation SBUF budget headroom for coalesced destinations. Full SBUF
# is 24 MiB per NeuronCore; we hold to 2 MiB per call site to leave the
# fa_tile inner-loop working set intact (scaffold s.2.3). Baked in at
# file-authoring time to match the sibling shim's constant of the same name.
SBUF_BUDGET_BYTES_PER_CALL_SITE = 2 * 1024 * 1024


@nki.jit
def dma_coalesced_gather_nki_v1(
    source_hbm,
    indices,
    out_sbuf,
    K,
    per_transfer_size,
    num_transfers,
    engine=None,
    oob_behavior=None,
    name="dma_coalesced_gather_nki_v1",
):
    """K-way coalescing HBM->SBUF indirect gather.

    Positional-arg contract (matches v0 `dma_coalesced_gather` docstring API):
        source_hbm         : nl.NkiTensor in HBM, shape [T, B].
                             T is the source-record count (>= max(indices)+1),
                             B is per-transfer bytes.
        indices            : nl.NkiTensor, shape [N], dtype int32 (or uint32 for
                             engines that require unsigned indirect indices).
                             Values are record indices into `source_hbm`.
                             `-1` sentinels short-circuit the destination write
                             when oob_behavior == oob_mode.skip.
        out_sbuf           : nl.NkiTensor in SBUF, shape [N, B].
                             Laid out as K rows per K-group; the g-th group
                             writes rows [g*K:(g+1)*K].
        K                  : coalesce factor (compile-time baked). K=1 is a
                             passthrough; K>=2 emits ceil(N/K) descriptors of
                             K*B bytes each.
        per_transfer_size  : B, bytes per logical packet (compile-time baked).
        num_transfers      : N, total logical packet count (compile-time baked).
        engine             : dma_engine.dma (default) or dma_engine.gpsimd_dma.
        oob_behavior       : oob_mode.skip (default) preserves KV-cache -1
                             sentinel semantics from attention_tkg.
        name               : per-call symbolic name; per-group descriptors
                             append `_g{g}` for A/B provenance.

    Address-pattern shape (per K-group):
        source_hbm.ap(
            [
                [B, K],   # outer: K packets of B bytes each; stride B, count K
                [1, B],   # inner: B contiguous bytes per packet
            ],
            offset=0,
            vector_offset=indices[g*K : (g+1)*K],
            indirect_dim=0,
        )

    This mirrors the existing attention_tkg `k_prior_reshaped.ap(...)` shape
    at `harness-v2/staging/cycle630/remote-core.py:2421` (indirect_dim=0,
    vector_offset=<record-count>-wide, [[stride, count], [1, stride]]). The
    single new degree of freedom is the compile-time-baked K.
    """
    engine = engine if engine is not None else dma_engine.dma
    oob_behavior = oob_behavior if oob_behavior is not None else oob_mode.skip
    B = int(per_transfer_size)
    N = int(num_transfers)
    K_int = int(K)

    # ---- Compat: K == 1 passthrough ----
    # Emits one descriptor per packet using the SAME ap() shape shrunk to K=1.
    # Byte-identical NEFF vs the un-wrapped baseline is NOT guaranteed here
    # (the wrapper adds one extra ap() level); correctness is certified by the
    # `verify_splice --tokens 10` gate, not by NEFF-byte diff at K=1.
    if K_int == 1:
        for i in nl.affine_range(N):
            nisa.dma_copy(
                dst=out_sbuf[i:i + 1, :],
                src=source_hbm.ap(
                    [
                        [B, 1],
                        [1, B],
                    ],
                    offset=0,
                    vector_offset=indices[i:i + 1],
                    indirect_dim=0,
                ),
                oob_mode=oob_behavior,
                dma_engine=engine,
                name=name + "_i" + str(i),
            )
        return

    # ---- SBUF budget guard (compile-time constant when K is JIT-baked) ----
    coalesced_bytes = K_int * B
    assert coalesced_bytes <= SBUF_BUDGET_BYTES_PER_CALL_SITE, (
        "dma_coalesced_gather_nki_v1: K*B = " + str(coalesced_bytes)
        + " exceeds SBUF budget " + str(SBUF_BUDGET_BYTES_PER_CALL_SITE)
        + " - shrink K or split the call site."
    )

    # ---- Coalesced path (Path A, uniform K) ----
    # G = ceil(N / K). Each K-group folds K adjacent gather transfers into
    # one descriptor of K*B bytes.
    G = (N + K_int - 1) // K_int
    for g in nl.affine_range(G):
        g0 = g * K_int
        g1 = g0 + K_int
        nisa.dma_copy(
            dst=out_sbuf[g0:g1, :],
            src=source_hbm.ap(
                [
                    [B, K_int],   # outer: K packets of B bytes each
                    [1, B],       # inner: B contiguous bytes per packet
                ],
                offset=0,
                vector_offset=indices[g0:g1],
                indirect_dim=0,
            ),
            oob_mode=oob_behavior,
            dma_engine=engine,
            name=name + "_g" + str(g),
        )
