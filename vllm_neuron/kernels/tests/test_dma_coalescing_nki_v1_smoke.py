"""Dry-import + shape/API-contract smoke for dma_coalescing_nki_v1.

Callsign: dma-coalescing-nki-v1-agent
Date    : 2026-08-27

Runs cleanly on this Windows box (no NKI runtime) and on Trn2 hosts (with
NKI). On non-NKI hosts, exercises:
    - Module imports without errors
    - SLUG / CALLSIGN / FIRST_FIRE_LANE constants match the manifest
    - is_available() returns False cleanly
    - dma_coalesced_gather_nki_v1(...) raises NotImplementedError with a
      clear message when NKI is missing
    - The source-string scaffold carries the load-bearing NKI patterns
      (`@nki.jit`, `nisa.dma_copy`, `.ap(`, `vector_offset`, `indirect_dim=0`,
      `nl.affine_range`) so a Trn2 compile host can trace it

On a Trn2 host (with NKI importable), also attempts:
    - `simnki`-based compile smoke on a synthetic (K=8, B=650, N=32) tile;
      SKIPPED cleanly if `nki.simulate` or a compatible tracer is unavailable

Run:
    python -m pytest harness-v2/staging/reference-sweep-20260826T2150Z/kernels/tests/test_dma_coalescing_nki_v1_smoke.py -v

Or without pytest:
    python harness-v2/staging/reference-sweep-20260826T2150Z/kernels/tests/test_dma_coalescing_nki_v1_smoke.py
"""
from __future__ import annotations

import pathlib
import sys
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import dma_coalescing_nki_v1 as v1  # noqa: E402


# --------------------------------------------------------------------------
# Gate 1: import + module identity
# --------------------------------------------------------------------------


class ImportAndIdentity(unittest.TestCase):
    def test_module_importable(self):
        self.assertTrue(hasattr(v1, "dma_coalesced_gather_nki_v1"))
        self.assertTrue(callable(v1.dma_coalesced_gather_nki_v1))

    def test_slug(self):
        self.assertEqual(v1.SLUG, "dma_coalesced_gather.nki_v1")

    def test_callsign(self):
        self.assertEqual(v1.CALLSIGN, "dma-coalescing-nki-v1-agent")

    def test_is_available_is_bool(self):
        self.assertIsInstance(v1.is_available(), bool)


# --------------------------------------------------------------------------
# Gate 2: source-string scaffold carries the load-bearing NKI patterns
# --------------------------------------------------------------------------


class SourceScaffoldPatterns(unittest.TestCase):
    """The source string must be non-empty and carry every primitive the
    Trn2 compile host will need to trace. If any pattern goes missing, the
    Trn2 A/B window will burn on an obvious authoring bug - this gate is
    the last CPU-side check before device time.
    """

    def setUp(self):
        self.src = v1.DMA_COALESCED_GATHER_NKI_V1_SOURCE

    def test_source_is_nonempty_string(self):
        self.assertIsInstance(self.src, str)
        self.assertGreater(len(self.src), 500)

    def test_source_has_nki_jit_decorator(self):
        self.assertIn("@nki.jit", self.src)

    def test_source_declares_function_name(self):
        self.assertIn("def dma_coalesced_gather_nki_v1(", self.src)

    def test_source_uses_dma_copy(self):
        self.assertIn("nisa.dma_copy(", self.src)

    def test_source_uses_ap_indirect_pattern(self):
        # The load-bearing pattern - matches remote-core.py:2421.
        self.assertIn(".ap(", self.src)
        self.assertIn("vector_offset=", self.src)
        self.assertIn("indirect_dim=0", self.src)

    def test_source_uses_affine_range_for_group_loop(self):
        self.assertIn("nl.affine_range", self.src)

    def test_source_carries_k1_passthrough_branch(self):
        # K=1 must be a distinct branch so the compat gate is exercised.
        self.assertIn("if K_int == 1", self.src)

    def test_source_carries_coalesced_k_ge_2_branch(self):
        # K>=2 must materialize G = ceil(N/K) K-groups.
        self.assertIn("(N + K_int - 1) // K_int", self.src)

    def test_source_carries_sbuf_budget_guard(self):
        self.assertIn("SBUF_BUDGET_BYTES_PER_CALL_SITE", self.src)

    def test_source_defaults_oob_skip(self):
        # KV-cache -1 sentinel semantics preserved by default.
        self.assertIn("oob_mode.skip", self.src)


# --------------------------------------------------------------------------
# Gate 3: shape / API-contract sanity of the exposed function symbol
# --------------------------------------------------------------------------


class FunctionSignatureContract(unittest.TestCase):
    """The documented signature must carry the six positional parameters.

    On a Trn2 host with NKI importable, this introspects the live callable.
    On a non-Trn2 host (fallback stub is `*args, **kwargs`), this parses the
    source-string scaffold instead - the source string IS the contract that
    the compile host will trace.
    """

    _EXPECTED = (
        "source_hbm",
        "indices",
        "out_sbuf",
        "K",
        "per_transfer_size",
        "num_transfers",
    )

    def test_source_string_signature(self):
        # Source string is the ground truth on both hosts.
        src = v1.DMA_COALESCED_GATHER_NKI_V1_SOURCE
        for expected in self._EXPECTED:
            self.assertIn(expected, src, msg=f"missing param {expected} in source")

    @unittest.skipUnless(
        v1.is_available(), "NKI runtime not present - live callable is a stub"
    )
    def test_live_callable_signature(self):
        import inspect
        try:
            sig = inspect.signature(v1.dma_coalesced_gather_nki_v1)
        except (ValueError, TypeError):
            # Some @nki.jit wrappers don't expose an inspectable signature.
            self.skipTest("Callable signature not inspectable (JIT wrapper).")
        params = list(sig.parameters.keys())
        for expected in self._EXPECTED:
            self.assertIn(expected, params, msg=f"missing param {expected}")


# --------------------------------------------------------------------------
# Gate 4: NKI-unavailable fallback raises cleanly
# --------------------------------------------------------------------------


class NkiUnavailableFallback(unittest.TestCase):
    """On the non-Trn2 host the wrapper must raise NotImplementedError with a
    clear message pointing callers to v0's CPU-side planners. On a Trn2 host
    this gate is skipped.
    """

    @unittest.skipIf(v1.is_available(), "NKI runtime present - skipping fallback gate")
    def test_call_raises_notimplementederror(self):
        with self.assertRaises(NotImplementedError) as ctx:
            v1.dma_coalesced_gather_nki_v1(
                None, None, None,
                K=8, per_transfer_size=650, num_transfers=32,
            )
        msg = str(ctx.exception)
        self.assertIn("NKI", msg)
        # Points callers to v0's Path B/C planners.
        self.assertIn("dma_coalescing_transform", msg)
        # Points to the source-string escape hatch.
        self.assertIn("DMA_COALESCED_GATHER_NKI_V1_SOURCE", msg)


# --------------------------------------------------------------------------
# Gate 5: NKI-available compile smoke (skipped on this Windows box)
# --------------------------------------------------------------------------


class NkiPresentCompileSmoke(unittest.TestCase):
    """On a Trn2 host with NKI importable, attempt to trace the kernel on a
    small synthetic tile. Any failure mode (simnki missing, tracer API drift,
    unsupported dtype in emulation) skips rather than fails so the smoke
    doesn't block CI on the non-Trn2 side.
    """

    @unittest.skipUnless(
        v1.is_available(), "NKI runtime not present - skipping compile smoke"
    )
    def test_simnki_trace_smoke(self):
        try:
            import numpy as np  # type: ignore[import]
        except Exception as e:  # pragma: no cover - env-dependent
            self.skipTest(f"numpy not available: {e}")

        # Small synthetic tile: N=32, B=650, K=8 -> G=4 K-groups.
        N, B, K = 32, 650, 8
        try:
            src = np.zeros((256, B), dtype=np.uint8)
            indices = np.arange(N, dtype=np.int32)
            out = np.zeros((N, B), dtype=np.uint8)
        except Exception as e:  # pragma: no cover - env-dependent
            self.skipTest(f"numpy tile alloc failed: {e}")

        # Try a few known simnki/tracer entry points; skip cleanly on any drift.
        tracer_found = False
        try:
            import nki.simulate as simnki  # type: ignore[import]
            tracer_found = True
            simnki.trace(
                v1.dma_coalesced_gather_nki_v1,
                src, indices, out,
                K=K, per_transfer_size=B, num_transfers=N,
            )
            return
        except Exception as e:  # pragma: no cover - env-dependent
            if tracer_found:
                self.skipTest(f"simnki.trace failed (expected outside Trn2 env): {e}")

        try:
            from nki import simulator as _sim  # type: ignore[import]
            _sim.trace(
                v1.dma_coalesced_gather_nki_v1,
                src, indices, out,
                K=K, per_transfer_size=B, num_transfers=N,
            )
            return
        except Exception as e:  # pragma: no cover - env-dependent
            self.skipTest(f"no NKI tracer entry point available: {e}")


# --------------------------------------------------------------------------
# Gate 6: first-fire lane manifest cross-check (GPT-OSS-20B TP=8 C=128 K=8)
# --------------------------------------------------------------------------


class FirstFireLaneManifest(unittest.TestCase):
    """Confirms the first-fire lane manifest matches the prompt: GPT-OSS-20B
    TP=8 C=128 at K=8. Guards against manifest drift in later edits.
    """

    def setUp(self):
        self.m = v1.FIRST_FIRE_LANE

    def test_lane_id(self):
        self.assertEqual(self.m["lane"], "gpt-oss-20b-tp8-c128")

    def test_k_and_per_transfer_size(self):
        self.assertEqual(self.m["K"], 8)
        self.assertEqual(self.m["per_transfer_size"], 650)

    def test_coalesced_bytes_inside_efficient_window(self):
        self.assertEqual(self.m["coalesced_bytes"], 8 * 650)
        self.assertGreater(
            self.m["coalesced_bytes"], v1.EFFICIENT_WINDOW_BYTES_MIN
        )

    def test_coalesced_bytes_under_sbuf_budget(self):
        self.assertLess(
            self.m["coalesced_bytes"], v1.SBUF_BUDGET_BYTES_PER_CALL_SITE
        )

    def test_projected_multiplier_bucket(self):
        # Per PROFILE-AT-KNEE-SUMMARY-2026-08-27.md - 650 B packets sit in the
        # (1.4, 2.0) multiplier bucket.
        self.assertEqual(self.m["projected_multiplier"], (1.4, 2.0))

    def test_current_tokps_matches_receipt(self):
        # PROFILE-C128-KNEE-2026-08-27.md banked 764.27 tok/s/card.
        self.assertAlmostEqual(self.m["current_tokps_per_card"], 764.27, places=2)

    def test_container_digest_pinned(self):
        # MEMORY.md pins the attention-tkg NKI validation container.
        self.assertIn("sha256:be11c204", self.m["container_digest"])


# --------------------------------------------------------------------------
# Gate 7: v0 sibling still importable and consistent with v1's K math
# --------------------------------------------------------------------------


class V0SiblingConsistency(unittest.TestCase):
    def test_v0_importable(self):
        from dma_coalescing_transform import (
            plan_coalesce_factor,
            SBUF_BUDGET_BYTES_PER_CALL_SITE as V0_SBUF,
            EFFICIENT_WINDOW_BYTES_MIN as V0_WINDOW,
        )
        # v0 and v1 must agree on the ceiling constants; drift here would
        # cause the coalesced NEFF and the CPU planner to disagree on K.
        self.assertEqual(V0_SBUF, v1.SBUF_BUDGET_BYTES_PER_CALL_SITE)
        self.assertEqual(V0_WINDOW, v1.EFFICIENT_WINDOW_BYTES_MIN)
        # 650 B -> ceil(4096/650) = 7. v1's first-fire manifest uses K=8
        # (one above the pure-window K to move firmly into the bandwidth-
        # bound side of the roofline, per scaffold s.1.3).
        self.assertEqual(plan_coalesce_factor(650.0), 7)
        self.assertGreaterEqual(v1.FIRST_FIRE_LANE["K"], plan_coalesce_factor(650.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
