"""Golden smoke test for `dma_coalescing_transform`.

Run:
    python -m pytest harness-v2/staging/reference-sweep-20260826T2150Z/kernels/tests/

Or without pytest:
    python harness-v2/staging/reference-sweep-20260826T2150Z/kernels/tests/test_dma_coalescing_smoke.py

The test simulates a descriptor stream matching the three profiled knees
(GPT-OSS-20B TP8 C=128, TP4 C=4, Qwen3-32B TP8 C=16), invokes the transform,
and asserts:

    (1) descriptor-count compression ratio  N/M  >=  4     (target from the prompt)
    (2) coalesce_factor_k selected matches the scaffold plan
    (3) K=1 passthrough leaves the plan trivially unchanged (compat gate)
    (4) NEFF-content diff correctly rejects a byte-identical candidate (require-different)
        and accepts a byte-identical K=1 self-insert (require-different=False)
    (5) SBUF budget overflow is refused
    (6) KV-slab plan padding is bounded (<20% overhead in the intended regime)

No NKI toolchain is needed for these gates - they exercise the CPU-side battery
that certifies the mechanism BEFORE any Trn2 device time is spent.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest

# Add the parent dir (`kernels/`) to sys.path so the import path works from
# both `pytest` and `python <file>` invocations.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from dma_coalescing_transform import (  # noqa: E402
    EFFICIENT_WINDOW_BYTES_MIN,
    SBUF_BUDGET_BYTES_PER_CALL_SITE,
    CoalescingPlan,
    DescriptorSiteReport,
    KvSlabLayoutPlan,
    analyze_descriptor_stream,
    apply_kv_slab_layout,
    build_coalescing_plan,
    plan_coalesce_factor,
    plan_kv_slab_layout,
    run_neff_content_check,
)


# --------------------------------------------------------------------------
# Fixture: three-knee summary-json blocks (matching Fleet A Tier-3 receipts)
# --------------------------------------------------------------------------

_KNEE_FIXTURES = {
    "gpt_oss_tp8_c128": {
        # PROFILE-C128-KNEE-2026-08-27.md :36, :68
        "dma_hw_dyn": {"packet_count": 5_900_000, "mean_bytes_per_packet": 650.0},
        "dma_sw_dyn": {"packet_count": 1_300_000, "mean_bytes_per_packet": 3_100.0},
        "dma_static": {"packet_count": 620_000, "mean_bytes_per_packet": 8_400.0},
    },
    "gpt_oss_tp4_c4": {
        # PROFILE-C4-KNEE-2026-08-27.md :27, :52
        "dma_hw_dyn": {"packet_count": 4_200_000, "mean_bytes_per_packet": 967.0},
        "dma_sw_dyn": {"packet_count": 900_000, "mean_bytes_per_packet": 2_800.0},
        "dma_static": {"packet_count": 400_000, "mean_bytes_per_packet": 7_100.0},
    },
    "qwen3_32b_tp8_c16": {
        # PROFILE-C16-KNEE-2026-08-27.md :26, :54
        "dma_hw_dyn": {"packet_count": 1_600_000, "mean_bytes_per_packet": 94.0},
        "dma_sw_dyn": {"packet_count": 3_000_000, "mean_bytes_per_packet": 2_233.0},
        "dma_static": {"packet_count": 689_900, "mean_bytes_per_packet": 9_317.0},
    },
}


def _write_fixture(fixture_key: str, out_dir: pathlib.Path) -> list[str]:
    """Write a summary-json per rank (8 shards) mirroring Fleet A's on-disk layout."""
    paths = []
    for rank in range(8):
        p = out_dir / f"summary-rank{rank}.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(_KNEE_FIXTURES[fixture_key], fh)
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------
# Test cases
# --------------------------------------------------------------------------


class DescriptorCompressionGolden(unittest.TestCase):
    """Gate 1: descriptor-count compression >= 4 on every measured knee."""

    def _assert_compression(self, fixture_key: str, min_ratio: int):
        with tempfile.TemporaryDirectory() as td:
            paths = _write_fixture(fixture_key, pathlib.Path(td))
            reports = analyze_descriptor_stream(paths)

        # Aggregate hw_dyn compression across all 8 shards.
        hw_reports = [r for r in reports if r.descriptor_class == "hw_dyn"]
        self.assertGreater(
            len(hw_reports), 0, f"No hw_dyn reports for {fixture_key}"
        )
        anchor = hw_reports[0]
        # Descriptor-count compression ratio == coalesce_factor_k because each
        # K-group folds K descriptors into 1.
        self.assertGreaterEqual(
            anchor.coalesce_headroom_k,
            min_ratio,
            f"{fixture_key}: K={anchor.coalesce_headroom_k} below >= {min_ratio}. "
            f"mean_bytes={anchor.mean_bytes_per_packet}",
        )
        # DMA-active reduction should be strictly positive under coalescing.
        self.assertGreater(anchor.projected_dma_active_reduction_pct, 0.0)

    def test_gpt_oss_tp8_c128_k_at_least_7(self):
        # 650 B packets -> ceil(4096/650) = 7. Prompt asked >= 4.
        self._assert_compression("gpt_oss_tp8_c128", min_ratio=7)

    def test_gpt_oss_tp4_c4_k_at_least_5(self):
        # 967 B packets -> ceil(4096/967) = 5.
        self._assert_compression("gpt_oss_tp4_c4", min_ratio=5)

    def test_qwen3_32b_tp8_c16_k_at_least_44(self):
        # 94 B packets -> ceil(4096/94) = 44 (largest headroom in campaign).
        self._assert_compression("qwen3_32b_tp8_c16", min_ratio=44)


class BreakevenKPlanner(unittest.TestCase):
    """Gate 2: `plan_coalesce_factor` matches the scaffold table."""

    def test_k_1_when_already_in_window(self):
        self.assertEqual(plan_coalesce_factor(EFFICIENT_WINDOW_BYTES_MIN), 1)
        self.assertEqual(plan_coalesce_factor(EFFICIENT_WINDOW_BYTES_MIN + 1000), 1)

    def test_scaffold_table_matches(self):
        # Scaffold s.1.3 published table (target 4 KiB, not the extreme break-even).
        self.assertEqual(plan_coalesce_factor(650.0), 7)     # ceil(4096/650)
        self.assertEqual(plan_coalesce_factor(967.0), 5)     # ceil(4096/967)
        self.assertEqual(plan_coalesce_factor(94.0), 44)     # ceil(4096/94)


class PassthroughCompatGate(unittest.TestCase):
    """Gate 3: K=1 is a no-op path; the plan reports K=1 when no coalescing needed."""

    def test_static_class_k_is_1(self):
        # dma_static packets are 8+ KiB already -> K must be 1.
        report = DescriptorSiteReport(
            site_name="fake::dma_static",
            packet_count=100,
            mean_bytes_per_packet=8400.0,
            total_bytes=100 * 8400,
            descriptor_class="static",
            coalesce_headroom_k=plan_coalesce_factor(8400.0),
            projected_dma_active_reduction_pct=0.0,
            fires_first=False,
        )
        self.assertEqual(report.coalesce_headroom_k, 1)


class NeffContentDiffGate(unittest.TestCase):
    """Gate 4: NEFF-content diff catches byte-identical no-ops (GEMMA4-LESSONS A6)."""

    def _make_neff(self, td: pathlib.Path, name: str, payload: bytes) -> pathlib.Path:
        # 1024-byte header + payload = minimal valid NEFF shape for byte-diff.
        p = td / name
        with open(p, "wb") as fh:
            fh.write(b"\x00" * 1024)
            fh.write(payload)
        return p

    def test_candidate_byte_identical_to_baseline_rejects(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = pathlib.Path(td)
            baseline = self._make_neff(td_p, "baseline.neff", b"AAAA" * 100)
            candidate = self._make_neff(td_p, "candidate.neff", b"AAAA" * 100)
            ok, reason = run_neff_content_check(baseline, candidate,
                                                require_different=True)
            self.assertFalse(ok, msg=reason)
            self.assertIn("byte-identical", reason)

    def test_candidate_differs_accepts(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = pathlib.Path(td)
            baseline = self._make_neff(td_p, "baseline.neff", b"AAAA" * 100)
            candidate = self._make_neff(td_p, "candidate.neff", b"BBBB" * 100)
            ok, reason = run_neff_content_check(baseline, candidate,
                                                require_different=True)
            self.assertTrue(ok, msg=reason)

    def test_k1_self_insert_compat_gate(self):
        # K=1 wrapper MUST leave NEFF byte-identical (zero-overhead compat).
        with tempfile.TemporaryDirectory() as td:
            td_p = pathlib.Path(td)
            baseline = self._make_neff(td_p, "baseline.neff", b"AAAA" * 100)
            candidate = self._make_neff(td_p, "candidate.neff", b"AAAA" * 100)
            ok, reason = run_neff_content_check(baseline, candidate,
                                                require_different=False)
            self.assertTrue(ok, msg=reason)
            self.assertIn("K=1 compat gate PASS", reason)

    def test_k1_self_insert_differs_rejects(self):
        # If the K=1 wrapper produced a different NEFF, the compat gate FAILs.
        with tempfile.TemporaryDirectory() as td:
            td_p = pathlib.Path(td)
            baseline = self._make_neff(td_p, "baseline.neff", b"AAAA" * 100)
            candidate = self._make_neff(td_p, "candidate.neff", b"BBBB" * 100)
            ok, reason = run_neff_content_check(baseline, candidate,
                                                require_different=False)
            self.assertFalse(ok, msg=reason)
            self.assertIn("K=1 compat gate FAIL", reason)


class SbufBudgetGuard(unittest.TestCase):
    """Gate 5: `plan_kv_slab_layout` refuses overflow."""

    def test_overflow_marked_not_fit(self):
        plan = plan_kv_slab_layout(
            tokens_per_block=64,
            d_head=128,
            dtype_bytes=2,
            hbm_budget_bytes_per_shard=1_000_000,   # 1 MB shard budget - tiny
            total_blocks_per_shard=100_000,          # forces overflow
            current_bytes_per_packet=650.0,
        )
        self.assertFalse(plan.fits_hbm)
        self.assertIn("OVERFLOWS", plan.reason)
        # apply_kv_slab_layout must refuse to mutate.

        class _FakeConfig:
            pass
        cfg = _FakeConfig()
        with self.assertRaises(RuntimeError):
            apply_kv_slab_layout(cfg, plan)


class KvSlabPaddingBound(unittest.TestCase):
    """Gate 6: padding cost is bounded (<20% overhead in the intended regime)."""

    def test_gpt_oss_c128_padding_under_20pct(self):
        # Realistic GPT-OSS-20B TP8 shard: KV per block ~ 64*128*2 = 16 KiB.
        # K=7 * 64 tokens = 448 tokens per aggregated block; slab ~ 112 KiB.
        # Padding <= EFFICIENT_WINDOW_BYTES_MIN per block; 4 KiB / 112 KiB < 4%.
        plan = plan_kv_slab_layout(
            tokens_per_block=64,
            d_head=128,
            dtype_bytes=2,           # bf16
            hbm_budget_bytes_per_shard=8 * 1024 * 1024 * 1024,   # 8 GiB
            total_blocks_per_shard=1024,
            current_bytes_per_packet=650.0,
        )
        self.assertTrue(plan.fits_hbm)
        total_slab = plan.aligned_stride_bytes * 1024
        pad_frac = plan.padding_bytes_per_shard / max(1, total_slab)
        self.assertLess(pad_frac, 0.20, msg=f"padding {pad_frac:.1%} > 20%")


class BuildCoalescingPlanEndToEnd(unittest.TestCase):
    """Gate 7: top-level orchestration binds an anchor + projected multiplier."""

    def test_gpt_oss_tp8_c128_hybrid_plan(self):
        with tempfile.TemporaryDirectory() as td:
            paths = _write_fixture("gpt_oss_tp8_c128", pathlib.Path(td))
            plan = build_coalescing_plan(
                "gpt-oss-20b-tp8-c128",
                paths,
                kv_block_kwargs=dict(
                    tokens_per_block=64,
                    d_head=128,
                    dtype_bytes=2,
                    hbm_budget_bytes_per_shard=8 * 1024 * 1024 * 1024,
                    total_blocks_per_shard=1024,
                ),
            )
        self.assertIsInstance(plan, CoalescingPlan)
        self.assertEqual(plan.lane, "gpt-oss-20b-tp8-c128")
        # 650 B packets -> multiplier (1.4, 2.0).
        self.assertEqual(plan.projected_tokps_multiplier, (1.4, 2.0))
        self.assertGreaterEqual(plan.coalesce_factor_k, 7)
        # Anchor site should be the hw_dyn (largest headroom).
        anchor = plan.site_reports[0]
        self.assertEqual(anchor.descriptor_class, "hw_dyn")
        self.assertTrue(anchor.fires_first)
        # Slab plan present since kv_block_kwargs was given.
        self.assertIsNotNone(plan.slab_plan)
        self.assertIsInstance(plan.slab_plan, KvSlabLayoutPlan)
        self.assertTrue(plan.slab_plan.fits_hbm)

    def test_qwen3_32b_tp8_c16_biggest_headroom(self):
        with tempfile.TemporaryDirectory() as td:
            paths = _write_fixture("qwen3_32b_tp8_c16", pathlib.Path(td))
            plan = build_coalescing_plan(
                "qwen3-32b-tp8-c16",
                paths,
                kv_block_kwargs=dict(
                    tokens_per_block=64,
                    d_head=128,
                    dtype_bytes=2,
                    hbm_budget_bytes_per_shard=8 * 1024 * 1024 * 1024,
                    total_blocks_per_shard=1024,
                ),
            )
        # 94 B packets -> multiplier (2.4, 3.0) - the largest headroom class.
        self.assertEqual(plan.projected_tokps_multiplier, (2.4, 3.0))
        self.assertGreaterEqual(plan.coalesce_factor_k, 44)


class ApplyPlanMutatesNeuronConfig(unittest.TestCase):
    """Gate 8: `apply_kv_slab_layout` sets the three required NeuronConfig knobs."""

    def test_apply_writes_block_size_and_coalesce_factor(self):
        class _FakeConfig:
            block_size = 64
            dma_coalesce_factor = 1
            use_shard_on_intermediate_dynamic_while = None

        cfg = _FakeConfig()
        plan = plan_kv_slab_layout(
            tokens_per_block=64,
            d_head=128,
            dtype_bytes=2,
            hbm_budget_bytes_per_shard=8 * 1024 * 1024 * 1024,
            total_blocks_per_shard=1024,
            current_bytes_per_packet=650.0,
        )
        apply_kv_slab_layout(cfg, plan)
        self.assertEqual(cfg.block_size, plan.kv_block_size_target)
        self.assertEqual(cfg.dma_coalesce_factor, plan.coalesce_factor_k)
        # MoE workaround preserved (MEMORY.md).
        self.assertTrue(cfg.use_shard_on_intermediate_dynamic_while)


if __name__ == "__main__":
    unittest.main(verbosity=2)
