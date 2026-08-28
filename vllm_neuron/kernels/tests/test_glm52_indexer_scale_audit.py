# SPDX-License-Identifier: Apache-2.0
"""Tests for glm52_indexer_fp8_scale_fix.

Two levels of coverage:

1. **Synthetic** — deterministic, runs in CI. Builds tiny in-memory scale
   dictionaries that model a clean checkpoint (audit passes) and an OCP-448
   checkpoint (audit fails). Verifies the audit, the patch generator, the
   calibration merge, and the load-time assertion.

2. **Live** — env-var-gated. If ``GLM_FP8_INDEX_PATH`` points at a real
   converted checkpoint or scale manifest, the audit runs end-to-end
   against it and prints the report to test output. The test does NOT
   assert pass/fail on the live path — that would gate CI on a checkpoint
   the operator has to hand-pointer. It only asserts the audit surface is
   internally consistent.

All tests are pure-Python + pytest; no torch or safetensors required.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

# Path shim so this file works both as a package member and via `pytest
# tests/test_glm52_indexer_scale_audit.py` from the kernels directory.
import sys

_KERNELS_DIR = Path(__file__).resolve().parents[1]
if str(_KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(_KERNELS_DIR))

import glm52_indexer_fp8_scale_fix as fix  # noqa: E402


# --- Helpers -----------------------------------------------------------------


NUM_LAYERS = 78
FULL_LAYERS = tuple(i for i in range(NUM_LAYERS) if i == 0 or (i - 1) % 4 == 3)
# GLM 5.2 default schedule per LANE-STATE §2.1: 20 full + 58 shared.
# Layer 0 owns a full indexer plus every fourth layer thereafter.


def _clean_scales(
    *,
    indexer_mult: float = 100.0,
    mla_k_mult: float = 100.0,
    mla_v_mult: float = 100.0,
) -> dict[str, float]:
    """Build a synthetic manifest that mimics a well-calibrated checkpoint.

    Multipliers are equal across MLA-k, MLA-v, and indexer. That is not
    exactly the geometric truth (index_head_dim differs from head_dim, so
    activation_max differs), but it lets the audit's OCP-signature ratio
    gate see a ratio of 1.0 — well outside the 1.867 signature window.
    """

    scales: dict[str, float] = {}
    for layer_idx in range(NUM_LAYERS):
        scales[f"model.layers.{layer_idx}.self_attn.k_cache_quant_multiplier"] = mla_k_mult
        scales[f"model.layers.{layer_idx}.self_attn.v_cache_quant_multiplier"] = mla_v_mult
        if layer_idx in FULL_LAYERS:
            scales[
                f"model.layers.{layer_idx}.self_attn.indexer.cache_quant_multiplier"
            ] = indexer_mult
    return scales


def _bad_ocp_scales(
    *,
    mla_mult: float = 100.0,
) -> dict[str, float]:
    """Build a manifest where indexer multipliers still carry the OCP-448
    scaling — indexer = mla * SCALE_COMPENSATION.
    """

    scales = _clean_scales(indexer_mult=mla_mult, mla_k_mult=mla_mult, mla_v_mult=mla_mult)
    for name in list(scales):
        if name.endswith(".indexer.cache_quant_multiplier"):
            scales[name] = mla_mult * fix.SCALE_COMPENSATION
    return scales


def _above_cap_scales() -> dict[str, float]:
    """Force at least one indexer multiplier above the Trainium cap."""

    scales = _clean_scales()
    victim = f"model.layers.{FULL_LAYERS[0]}.self_attn.indexer.cache_quant_multiplier"
    scales[victim] = fix.MULTIPLIER_CAP + 0.5
    return scales


# --- Constants sanity --------------------------------------------------------


class TestConstantsMirrorUpstream:
    """Mirror ``static_fp8.py`` — if any of these drift, upstream changed."""

    def test_neuron_max_is_240(self) -> None:
        assert fix.NEURON_LEGACY_E4M3_MAX == 240.0

    def test_ocp_max_is_448(self) -> None:
        assert fix.OCP_E4M3_MAX == 448.0

    def test_downscale_is_240_over_448(self) -> None:
        assert math.isclose(fix.WEIGHT_DOWNSCALE, 240.0 / 448.0, rel_tol=1e-9)

    def test_compensation_is_448_over_240(self) -> None:
        assert math.isclose(fix.SCALE_COMPENSATION, 448.0 / 240.0, rel_tol=1e-9)

    def test_downscale_and_compensation_are_reciprocal(self) -> None:
        assert math.isclose(
            fix.WEIGHT_DOWNSCALE * fix.SCALE_COMPENSATION, 1.0, rel_tol=1e-9
        )

    def test_multiplier_cap_matches_scaffold_prescription(self) -> None:
        # Scaffold §10 item 3 says: assert `cache_quant_multiplier <= 240.0`.
        # The cap MUST equal 240.0 exactly.
        assert fix.MULTIPLIER_CAP == 240.0


# --- Audit -------------------------------------------------------------------


class TestAuditSynthetic:
    def test_clean_manifest_passes(self) -> None:
        report = fix.audit_indexer_scales(_clean_scales())
        assert report.verdict == "clean"
        assert report.indexer_layers_seen == 20
        assert report.layers_above_cap == []
        assert report.layers_with_ocp_signature == []
        assert report.max_indexer_multiplier == 100.0

    def test_ocp_signature_flips_verdict(self) -> None:
        report = fix.audit_indexer_scales(_bad_ocp_scales())
        assert report.verdict == "requantize_required"
        # Every full-indexer layer must be flagged; MLA multipliers are
        # under the cap so the absolute-cap gate stays quiet — the OCP
        # ratio gate is the entire signal.
        assert set(report.layers_with_ocp_signature) == set(FULL_LAYERS)
        assert report.layers_above_cap == []

    def test_above_cap_flips_verdict(self) -> None:
        report = fix.audit_indexer_scales(_above_cap_scales())
        assert report.verdict == "requantize_required"
        assert FULL_LAYERS[0] in report.layers_above_cap

    def test_empty_manifest_verdict_is_unknown(self) -> None:
        report = fix.audit_indexer_scales({})
        assert report.verdict == "unknown"
        assert report.indexer_layers_seen == 0

    def test_missing_mla_pair_is_recorded(self) -> None:
        # Include indexer entries but drop all MLA entries — the OCP ratio
        # gate cannot fire but the absolute-cap gate still can.
        scales = {
            f"model.layers.{FULL_LAYERS[0]}.self_attn.indexer.cache_quant_multiplier": 50.0,
        }
        report = fix.audit_indexer_scales(scales)
        assert FULL_LAYERS[0] in report.missing_mla_pairs
        assert report.verdict == "clean"

    def test_audit_report_serializes_to_json(self) -> None:
        report = fix.audit_indexer_scales(_bad_ocp_scales())
        payload = report.to_dict()
        text = json.dumps(payload, sort_keys=True)
        loaded = json.loads(text)
        assert loaded["verdict"] == "requantize_required"
        assert loaded["constants"]["NEURON_LEGACY_E4M3_MAX"] == 240.0


class TestAuditFromJsonManifest:
    def test_flat_json_map_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "cache_quant_multipliers.json"
        path.write_text(json.dumps(_bad_ocp_scales()), encoding="utf-8")
        report = fix.audit_indexer_scales(path)
        assert report.verdict == "requantize_required"

    def test_glm52_manifest_layout_is_read(self, tmp_path: Path) -> None:
        # Nested layout as emitted by checkpoint_converter.py.
        manifest = {
            "calibration_manifest": {
                "cache_quant_multipliers": {
                    "values": _bad_ocp_scales(),
                    "contract": {"status": "complete"},
                }
            }
        }
        # The reader picks it up when the file lives inside a directory.
        (tmp_path / "glm52-static-fp8-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        report = fix.audit_indexer_scales(tmp_path)
        assert report.verdict == "requantize_required"

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            fix.audit_indexer_scales(tmp_path / "does-not-exist.json")

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            fix.audit_indexer_scales(tmp_path)


# --- Patch generation --------------------------------------------------------


class TestPatchGeneration:
    def test_patch_scales_offending_layers_by_downscale(self) -> None:
        report = fix.audit_indexer_scales(_bad_ocp_scales(mla_mult=100.0))
        patch = fix.build_requantize_patch(report)
        cqm = patch["cache_quant_multipliers"]
        assert len(cqm) == 20
        for name, new_value in cqm.items():
            old_value = 100.0 * fix.SCALE_COMPENSATION
            assert math.isclose(
                new_value, old_value * fix.WEIGHT_DOWNSCALE, rel_tol=1e-9
            ), name
            # And the new value must be inside the cap.
            assert new_value <= fix.MULTIPLIER_CAP

    def test_patch_ignores_clean_layers_by_default(self) -> None:
        # Half-and-half: 10 layers on the OCP path, 10 already migrated.
        scales = _clean_scales(indexer_mult=100.0, mla_k_mult=100.0)
        for layer_idx in FULL_LAYERS[:10]:
            scales[
                f"model.layers.{layer_idx}.self_attn.indexer.cache_quant_multiplier"
            ] = 100.0 * fix.SCALE_COMPENSATION
        report = fix.audit_indexer_scales(scales)
        patch = fix.build_requantize_patch(report)
        assert len(patch["cache_quant_multipliers"]) == 10

    def test_patch_can_include_clean_layers_for_full_rewrite(self) -> None:
        report = fix.audit_indexer_scales(_bad_ocp_scales())
        patch = fix.build_requantize_patch(report, include_clean_layers=True)
        assert len(patch["cache_quant_multipliers"]) == 20  # all full layers

    def test_patch_rejects_bad_downscale(self) -> None:
        report = fix.audit_indexer_scales(_bad_ocp_scales())
        with pytest.raises(ValueError):
            fix.build_requantize_patch(report, downscale=0.0)
        with pytest.raises(ValueError):
            fix.build_requantize_patch(report, downscale=2.0)

    def test_patch_metadata_records_verdict(self) -> None:
        report = fix.audit_indexer_scales(_bad_ocp_scales())
        patch = fix.build_requantize_patch(report)
        assert patch["fix_metadata"]["source_report_verdict"] == "requantize_required"
        assert patch["fix_metadata"]["cap"] == fix.MULTIPLIER_CAP
        assert patch["fix_metadata"]["layers_touched"] == 20


class TestCalibrationMerge:
    def test_merge_touches_only_indexer_keys(self) -> None:
        base = {
            "projection_input_scales": {"model.layers.0.mlp.gate_proj.input_scale": 0.5},
            "cache_quant_multipliers": {
                "model.layers.0.self_attn.k_cache_quant_multiplier": 100.0,
                "model.layers.0.self_attn.indexer.cache_quant_multiplier": 187.0,
            },
        }
        report = fix.audit_indexer_scales(base["cache_quant_multipliers"])
        patch = fix.build_requantize_patch(report)
        merged = fix.patch_calibration_dict(base, patch)
        # projection_input_scales is untouched.
        assert merged["projection_input_scales"] == base["projection_input_scales"]
        # MLA k stays untouched.
        assert (
            merged["cache_quant_multipliers"]["model.layers.0.self_attn.k_cache_quant_multiplier"]
            == 100.0
        )
        # Indexer got rescaled.
        expected = 187.0 * fix.WEIGHT_DOWNSCALE
        assert math.isclose(
            merged["cache_quant_multipliers"][
                "model.layers.0.self_attn.indexer.cache_quant_multiplier"
            ],
            expected,
            rel_tol=1e-9,
        )

    def test_merge_of_empty_patch_returns_copy(self) -> None:
        base = {
            "projection_input_scales": {"a": 1.0},
            "cache_quant_multipliers": {"b": 2.0},
        }
        merged = fix.patch_calibration_dict(base, {})
        assert merged == base
        # And it is a copy, not the same dict.
        assert merged is not base
        assert merged["projection_input_scales"] is not base["projection_input_scales"]


# --- Load-time assertion -----------------------------------------------------


class TestAssertion:
    def test_accepts_scalar_below_cap(self) -> None:
        assert fix.assert_indexer_multiplier_bounded(100.0, layer_idx=0) == 100.0

    def test_accepts_scalar_at_cap(self) -> None:
        # Exactly-240 is representable and must pass.
        assert fix.assert_indexer_multiplier_bounded(240.0, layer_idx=0) == 240.0

    def test_rejects_scalar_above_cap(self) -> None:
        with pytest.raises(fix.IndexerMultiplierOutOfRange) as excinfo:
            fix.assert_indexer_multiplier_bounded(240.1, layer_idx=7)
        assert "layer 7" in str(excinfo.value)
        assert "240" in str(excinfo.value)

    def test_rejects_ocp_signature_value(self) -> None:
        # A checkpoint left on the OCP-448 calibration.
        with pytest.raises(fix.IndexerMultiplierOutOfRange):
            fix.assert_indexer_multiplier_bounded(447.9, layer_idx=42)

    def test_rejects_zero(self) -> None:
        with pytest.raises(fix.IndexerMultiplierOutOfRange):
            fix.assert_indexer_multiplier_bounded(0.0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(fix.IndexerMultiplierOutOfRange):
            fix.assert_indexer_multiplier_bounded(-1.0)

    def test_rejects_nan(self) -> None:
        with pytest.raises(fix.IndexerMultiplierOutOfRange):
            fix.assert_indexer_multiplier_bounded(float("nan"))

    def test_rejects_inf(self) -> None:
        with pytest.raises(fix.IndexerMultiplierOutOfRange):
            fix.assert_indexer_multiplier_bounded(float("inf"))

    def test_accepts_object_with_item(self) -> None:
        class _Scalar:
            def item(self) -> float:
                return 100.0

        assert fix.assert_indexer_multiplier_bounded(_Scalar()) == 100.0

    def test_rejects_non_scalar_object(self) -> None:
        class _NotScalar:
            pass

        with pytest.raises(fix.IndexerMultiplierOutOfRange):
            fix.assert_indexer_multiplier_bounded(_NotScalar())

    def test_custom_cap_is_respected(self) -> None:
        # A stricter cap should reject a previously-accepted value.
        with pytest.raises(fix.IndexerMultiplierOutOfRange):
            fix.assert_indexer_multiplier_bounded(150.0, cap=100.0)

    def test_install_wrapper_gates_setter(self) -> None:
        # Duck-typed class stand-in for Glm52FullIndexer.
        class _Stub:
            def __init__(self) -> None:
                self.layer_idx = 3
                self.stored: float | None = None

            def set_cache_quant_multiplier(self, multiplier: float) -> None:
                if multiplier <= 0:
                    raise ValueError("upstream check")
                self.stored = float(multiplier)

        fix.install_load_time_assertion(_Stub)
        obj = _Stub()
        obj.set_cache_quant_multiplier(100.0)
        assert obj.stored == 100.0
        with pytest.raises(fix.IndexerMultiplierOutOfRange):
            obj.set_cache_quant_multiplier(300.0)


# --- End-to-end: audit + patch + assertion round-trip ------------------------


class TestEndToEnd:
    def test_apply_patch_then_assertion_passes(self) -> None:
        """After applying the patch, the assertion must pass for every layer."""

        scales = _bad_ocp_scales(mla_mult=100.0)
        report = fix.audit_indexer_scales(scales)
        assert report.verdict == "requantize_required"

        patch = fix.build_requantize_patch(report)
        # Merge into a "current calibration file" that mirrors the on-disk
        # scale set exactly.
        current = {
            "projection_input_scales": {},
            "cache_quant_multipliers": dict(scales),
        }
        merged = fix.patch_calibration_dict(current, patch)

        # Every rewritten indexer multiplier must now be inside the cap.
        for name, value in merged["cache_quant_multipliers"].items():
            if name.endswith(".indexer.cache_quant_multiplier"):
                layer_idx = int(name.split(".")[2])
                fix.assert_indexer_multiplier_bounded(value, layer_idx=layer_idx)

        # And a re-audit must now be clean.
        new_report = fix.audit_indexer_scales(merged["cache_quant_multipliers"])
        assert new_report.verdict == "clean"


# --- Live-checkpoint audit (env-var-gated) -----------------------------------


class TestLiveCheckpointAudit:
    """Runs against the operator's real checkpoint when GLM_FP8_INDEX_PATH is set.

    The test does NOT fail on ``requantize_required``. It only prints the
    report to test output so an operator running this locally can see the
    verdict without gating CI on a checkpoint that has to be hand-pointered.
    """

    def _get_source(self) -> Path:
        env = os.environ.get("GLM_FP8_INDEX_PATH")
        if not env:
            pytest.skip("GLM_FP8_INDEX_PATH not set; skipping live checkpoint audit")
        source = Path(env).expanduser()
        if not source.exists():
            pytest.skip(f"GLM_FP8_INDEX_PATH points at nonexistent {source}")
        return source

    def test_audit_is_internally_consistent(self) -> None:
        source = self._get_source()
        report = fix.audit_indexer_scales(source)
        # Every layer flagged above-cap must also be present in layer_audits.
        indices_in_audits = {la.layer_idx for la in report.layer_audits}
        for layer_idx in report.layers_above_cap:
            assert layer_idx in indices_in_audits
        for layer_idx in report.layers_with_ocp_signature:
            assert layer_idx in indices_in_audits
        # Median and max must be from the same distribution.
        if report.indexer_layers_seen:
            values = sorted(
                la.indexer_multiplier for la in report.layer_audits
            )
            assert report.max_indexer_multiplier == values[-1]
            assert report.median_indexer_multiplier == values[len(values) // 2]

    def test_audit_report_summary_printable(self, capsys: pytest.CaptureFixture[str]) -> None:
        source = self._get_source()
        report = fix.audit_indexer_scales(source)
        # Fine to always fire this — the print goes to captured output only.
        print(fix._format_report_text(report))
        captured = capsys.readouterr()
        assert "verdict" in captured.out
