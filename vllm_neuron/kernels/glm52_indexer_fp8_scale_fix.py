# SPDX-License-Identifier: Apache-2.0
"""GLM 5.2 indexer FP8 cache_quant_multiplier scale-cap fix.

This module implements the *fastest unblock* Python-side fix for the GLM 5.2
ten-token gate failure (2026-08-06: token-0 centred cosine 0.9695 vs 0.99
bar). Root cause per lane manager tick-1 in
``harness-v2/staging/reference-sweep-20260826T2150Z/lanes/glm-5-2-5-3/
LANE-STATE-20260827T222500Z.md`` §2.1 Mode B: the indexer-side
``cache_quant_multiplier`` scalars were left on the OCP-E4M3 (qmax=448)
calibration path while the MLA-side scalars had already been re-run through
``neuron_legacy_e4m3fn_qmax240``. The Trainium2 e4m3 kernel clamps at
``NEURON_LEGACY_E4M3_MAX = 240``. Cache activations quantized with an
OCP-448 multiplier silently truncate their peak magnitudes at write time,
which distorts the indexer's top-K score for token 0 (the only causally
visible token at prefill) and misses argmax equality on the ten-token gate.

Three deliverables here, matching NKI-DSA-SPARSE-ATTENTION-SCAFFOLD-2026-08-27
§10 items 2 and 3:

1. :func:`audit_indexer_scales` — read a converted safetensors checkpoint's
   per-layer ``.indexer.cache_quant_multiplier`` scalars and flag any that
   would drive the effective clamp point above Trainium's 240 max, and any
   whose ratio to the sibling MLA ``k_cache_quant_multiplier`` matches the
   OCP-448 -> 240 signature (nominal 448 / 240 = 1.867).

2. :func:`requantize_indexer_scales` — write a JSON patch that scales every
   offending ``.indexer.cache_quant_multiplier`` down by
   ``WEIGHT_DOWNSCALE = 240 / 448`` so the on-disk scalar matches the
   Trainium2 e4m3 kernel's clamp point. The patch is meant to be fed to
   ``models/glm5-2/tools/retarget-glm52-static-fp8-scales.py`` (which is the
   canonical rewriter that owns the safetensors byte edit + manifest bump).
   Producing the patch here rather than mutating shards keeps this fix
   reversible and reviewable.

3. :func:`assert_indexer_multiplier_bounded` — a load-time assertion helper
   that ``Glm52FullIndexer.__init__`` (in vllm-neuron) can call after
   ``set_cache_quant_multiplier`` to turn silent numeric drift into a
   fail-closed load error. The intended production integration is a one-line
   ``assert_indexer_multiplier_bounded(self.cache_quant_multiplier,
   layer_idx=self.layer_idx)`` at the end of
   ``Glm52FullIndexer.set_cache_quant_multiplier``.

Design constraints
------------------
* Pure standard library + NumPy for the audit/requantize path. ``torch`` and
  ``safetensors`` are optional imports — the module must run inside the
  campaign's Windows staging harness where neither is guaranteed. When
  ``safetensors`` is present the audit reads directly from shards; when it
  is absent the audit falls back to a companion JSON scale manifest.
* Semantics of ``cache_quant_multiplier`` at write time (see
  ``vllm_neuron/model/glm52_moe_dsa/cache_ops.py:write_paged_cache`` L99–109):

  ::

      stored = (values.to(fp32) * multiplier)
                   .clamp(-clamp_max, clamp_max)
                   .to(fp8)   # clamp_max = 240 for e4m3fn

  A well-calibrated multiplier is ``qmax / activation_max``. For the
  indexer's post-``k_norm`` key values, activation_max is O(1) after the
  ``LayerNorm``, so a healthy multiplier lands in the low-hundreds range.
  A multiplier value that sits at or above 240 is a strong signature that
  the calibrator used ``qmax=448`` (OCP) rather than ``qmax=240`` (Trainium
  legacy); the fix is a straight ``* WEIGHT_DOWNSCALE`` rescale.

* The assertion cap is deliberately equal to ``NEURON_LEGACY_E4M3_MAX``. The
  scaffold §10 item 3 spells it out: ``assert multiplier <= 240.0``. The
  cap covers the well-known OCP-448 signature (multiplier ≈ 448) but does
  NOT catch a checkpoint whose activation_max was mis-estimated in the
  other direction. That is a separate calibration bug and outside the
  scope of this fix — the audit is a proxy, not a proof.

References
----------
* Root cause: ``.../lanes/glm-5-2-5-3/LANE-STATE-20260827T222500Z.md`` §2.1.
* Kernel scaffold prescribing the fix: ``.../kernels/
  NKI-DSA-SPARSE-ATTENTION-SCAFFOLD-2026-08-27.md`` §10.
* Existing constants: ``third_party/vllm-neuron/vllm_neuron/model/
  glm52_moe_dsa/static_fp8.py`` (``NEURON_LEGACY_E4M3_MAX``,
  ``OCP_E4M3_MAX``, ``WEIGHT_DOWNSCALE``).
* Canonical scale rewriter: ``models/glm5-2/tools/
  retarget-glm52-static-fp8-scales.py``.
* Load-time entry point: ``third_party/vllm-neuron/vllm_neuron/model/
  glm52_moe_dsa/indexer.py::Glm52FullIndexer.set_cache_quant_multiplier``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:  # NumPy is present in the campaign staging harness.
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a hard runtime dep here
    np = None  # type: ignore[assignment]


# --- Constants replicated from vllm-neuron's static_fp8.py --------------------
# Kept in-module (not imported) so this fix stays runnable in staging without
# a vllm-neuron install. If these drift from the upstream file, the tests
# below fail loudly (see test_constants_mirror_upstream in the tests file).

NEURON_LEGACY_E4M3_MAX = 240.0  # Trainium2 e4m3fn saturation magnitude.
OCP_E4M3_MAX = 448.0            # OCP e4m3fn saturation magnitude.
WEIGHT_DOWNSCALE = NEURON_LEGACY_E4M3_MAX / OCP_E4M3_MAX  # 0.53571...
SCALE_COMPENSATION = OCP_E4M3_MAX / NEURON_LEGACY_E4M3_MAX  # 1.86666...

# Assertion cap. Matches scaffold §10 item 3 verbatim.
MULTIPLIER_CAP = NEURON_LEGACY_E4M3_MAX

# Heuristic ratio at which the indexer multiplier is "suspiciously close" to
# the MLA multiplier's OCP signature. If indexer / MLA is within 5% of
# SCALE_COMPENSATION, the audit flags the indexer as un-migrated.
OCP_SIGNATURE_RATIO_TOLERANCE = 0.05

# Regexes matching the names used by
# ``required_cache_quant_multiplier_keys`` in
# ``glm52_moe_dsa/checkpoint_converter.py``.
_INDEXER_KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\.indexer\.cache_quant_multiplier$"
)
_MLA_K_KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\.k_cache_quant_multiplier$"
)
_MLA_V_KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\.v_cache_quant_multiplier$"
)


# --- Audit data classes ------------------------------------------------------


@dataclass
class LayerAudit:
    """Per-layer audit record."""

    layer_idx: int
    indexer_multiplier: float
    mla_k_multiplier: float | None
    mla_v_multiplier: float | None
    exceeds_cap: bool
    ocp_signature: bool  # True if indexer_multiplier / mla_k ≈ SCALE_COMPENSATION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditReport:
    """Full audit report for a converted GLM 5.2 checkpoint."""

    checkpoint_path: str
    layer_audits: list[LayerAudit] = field(default_factory=list)
    total_layers: int = 0
    indexer_layers_seen: int = 0
    layers_above_cap: list[int] = field(default_factory=list)
    layers_with_ocp_signature: list[int] = field(default_factory=list)
    max_indexer_multiplier: float = 0.0
    median_indexer_multiplier: float = 0.0
    missing_mla_pairs: list[int] = field(default_factory=list)
    verdict: str = "unknown"  # "clean" | "requantize_required" | "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "total_layers": self.total_layers,
            "indexer_layers_seen": self.indexer_layers_seen,
            "layers_above_cap": self.layers_above_cap,
            "layers_with_ocp_signature": self.layers_with_ocp_signature,
            "max_indexer_multiplier": self.max_indexer_multiplier,
            "median_indexer_multiplier": self.median_indexer_multiplier,
            "missing_mla_pairs": self.missing_mla_pairs,
            "verdict": self.verdict,
            "constants": {
                "NEURON_LEGACY_E4M3_MAX": NEURON_LEGACY_E4M3_MAX,
                "OCP_E4M3_MAX": OCP_E4M3_MAX,
                "WEIGHT_DOWNSCALE": WEIGHT_DOWNSCALE,
                "SCALE_COMPENSATION": SCALE_COMPENSATION,
                "MULTIPLIER_CAP": MULTIPLIER_CAP,
            },
            "layer_audits": [la.to_dict() for la in self.layer_audits],
        }


# --- Scale-manifest readers --------------------------------------------------


def _try_read_scales_from_safetensors(
    checkpoint_dir: Path,
) -> dict[str, float] | None:
    """Attempt to open every safetensors shard and pull the scalar multipliers.

    Returns None if ``safetensors`` isn't importable or the checkpoint has no
    ``model.safetensors.index.json``. The audit path falls through to the
    companion-JSON reader below in that case.
    """

    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ImportError:
        return None

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        return None

    wanted = {
        name
        for name in weight_map
        if name.endswith(".cache_quant_multiplier")
    }
    if not wanted:
        return None

    scales: dict[str, float] = {}
    by_shard: dict[str, list[str]] = {}
    for name in wanted:
        shard = weight_map.get(name)
        if isinstance(shard, str):
            by_shard.setdefault(shard, []).append(name)

    for shard, names in by_shard.items():
        shard_path = checkpoint_dir / shard
        if not shard_path.is_file():
            continue
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for name in names:
                try:
                    tensor = handle.get_tensor(name)
                except Exception:  # noqa: BLE001 - defensive parse
                    continue
                try:
                    value = float(tensor.reshape(-1)[0].to("cpu").float())
                except Exception:  # noqa: BLE001
                    try:
                        value = float(tensor.reshape(-1)[0])
                    except Exception:  # noqa: BLE001
                        continue
                if math.isfinite(value) and value > 0:
                    scales[name] = value
    return scales or None


def _try_read_scales_from_companion_json(
    checkpoint_dir: Path,
) -> dict[str, float] | None:
    """Fall-back reader for staging: a companion JSON with the scalar map.

    Two accepted layouts:
      1. ``glm52-static-fp8-manifest.json`` with
         ``["calibration_manifest"]["cache_quant_multipliers"]["values"]`` — the
         format emitted by ``checkpoint_converter.write_manifest``.
      2. Flat ``.scales.json`` — ``{name: value}`` map dropped alongside the
         safetensors index (also what the campaign's staging harness uses).
    """

    candidates = [
        checkpoint_dir / "glm52-static-fp8-manifest.json",
        checkpoint_dir / "model.safetensors.index.scales.json",
        checkpoint_dir / "cache_quant_multipliers.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        # Layout 1: nested manifest.
        cal = payload.get("calibration_manifest")
        if isinstance(cal, dict):
            cqm = cal.get("cache_quant_multipliers")
            if isinstance(cqm, dict):
                values = cqm.get("values")
                if isinstance(values, dict):
                    result = {
                        str(k): float(v)
                        for k, v in values.items()
                        if isinstance(v, (int, float))
                        and math.isfinite(float(v))
                        and float(v) > 0
                    }
                    if result:
                        return result
        # Layout 2: flat map (values or fully-qualified names).
        flat = {
            str(k): float(v)
            for k, v in payload.items()
            if isinstance(v, (int, float))
            and math.isfinite(float(v))
            and float(v) > 0
        }
        if flat:
            return flat
    return None


def load_scale_manifest(source: Path | str) -> dict[str, float]:
    """Load ``.cache_quant_multiplier`` scalars from a checkpoint dir OR JSON.

    ``source`` may be a directory (checkpoint root) or a JSON file path. The
    reader tries safetensors first, then several JSON layouts. Raises
    ``FileNotFoundError`` if no readable source is found — callers should catch
    and skip in test/audit contexts.
    """

    path = Path(source).expanduser()
    if path.is_dir():
        scales = _try_read_scales_from_safetensors(path)
        if scales is not None:
            return scales
        scales = _try_read_scales_from_companion_json(path)
        if scales is not None:
            return scales
        raise FileNotFoundError(
            f"no readable scale manifest in {path} (looked for "
            f"model.safetensors.index.json, "
            f"glm52-static-fp8-manifest.json, and a flat scale JSON)"
        )
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - clear error
            raise ValueError(
                f"{path} is not a valid JSON scale manifest: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"{path} does not contain a JSON object mapping name -> scale"
            )
        cal = payload.get("calibration_manifest")
        if isinstance(cal, dict):
            cqm = cal.get("cache_quant_multipliers")
            if isinstance(cqm, dict):
                values = cqm.get("values")
                if isinstance(values, dict):
                    result = {
                        str(k): float(v)
                        for k, v in values.items()
                        if isinstance(v, (int, float))
                        and math.isfinite(float(v))
                        and float(v) > 0
                    }
                    if result:
                        return result
        flat = {
            str(k): float(v)
            for k, v in payload.items()
            if isinstance(v, (int, float))
            and math.isfinite(float(v))
            and float(v) > 0
        }
        if flat:
            return flat
        raise ValueError(f"{path} has no positive scalar values")
    raise FileNotFoundError(source)


# --- Audit function ----------------------------------------------------------


def _group_by_layer(
    scales: Mapping[str, float],
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    indexer: dict[int, float] = {}
    mla_k: dict[int, float] = {}
    mla_v: dict[int, float] = {}
    for name, value in scales.items():
        match = _INDEXER_KEY_RE.match(name)
        if match:
            indexer[int(match.group(1))] = float(value)
            continue
        match = _MLA_K_KEY_RE.match(name)
        if match:
            mla_k[int(match.group(1))] = float(value)
            continue
        match = _MLA_V_KEY_RE.match(name)
        if match:
            mla_v[int(match.group(1))] = float(value)
    return indexer, mla_k, mla_v


def audit_indexer_scales(
    source: Path | str | Mapping[str, float],
    *,
    cap: float = MULTIPLIER_CAP,
    ocp_ratio_tolerance: float = OCP_SIGNATURE_RATIO_TOLERANCE,
) -> AuditReport:
    """Audit indexer cache_quant_multiplier values against the Trainium2 cap.

    Two independent gates fire, either of which flips the report verdict to
    ``requantize_required``:

    1. **Absolute-cap gate.** Any indexer multiplier ``m > cap`` (default 240).
       A multiplier at 240 is exactly at the clamp point; per the scaffold
       §10 item 3, ``<= 240.0`` is the assertion.
    2. **OCP-signature gate.** ``indexer_multiplier / mla_k_multiplier`` sits
       within ``ocp_ratio_tolerance`` of ``SCALE_COMPENSATION = 448 / 240``
       for at least one layer. This catches the case where the OCP-448
       calibration slipped through even at multiplier values slightly below
       240 (e.g. an activation_max of 1.01 gives a 442.6 multiplier —
       under the absolute cap but still on the wrong path).

    Args:
        source: Either a checkpoint directory / manifest JSON path, or an
            already-loaded ``{name: multiplier}`` mapping (useful for tests
            and callers that already have the scalars in hand).
        cap: Upper bound for the multiplier. Values >= cap fail the audit.
            Defaults to ``MULTIPLIER_CAP`` (240.0).
        ocp_ratio_tolerance: Fractional tolerance around
            ``SCALE_COMPENSATION`` for the second gate. Defaults to 5%.
    """

    if isinstance(source, Mapping):
        scales: dict[str, float] = {
            str(k): float(v) for k, v in source.items()
        }
        checkpoint_repr = "<in-memory mapping>"
    else:
        scales = load_scale_manifest(source)
        checkpoint_repr = str(Path(source).resolve())

    indexer, mla_k, mla_v = _group_by_layer(scales)
    report = AuditReport(checkpoint_path=checkpoint_repr)
    report.indexer_layers_seen = len(indexer)
    report.total_layers = max(
        (max(indexer, default=-1), max(mla_k, default=-1), max(mla_v, default=-1))
    ) + 1

    for layer_idx in sorted(indexer):
        m = indexer[layer_idx]
        mk = mla_k.get(layer_idx)
        mv = mla_v.get(layer_idx)
        exceeds_cap = m >= cap
        ocp_signature = False
        if mk is not None and mk > 0:
            ratio = m / mk
            ocp_signature = (
                abs(ratio - SCALE_COMPENSATION) / SCALE_COMPENSATION
                <= ocp_ratio_tolerance
            )
        else:
            report.missing_mla_pairs.append(layer_idx)
        audit = LayerAudit(
            layer_idx=layer_idx,
            indexer_multiplier=m,
            mla_k_multiplier=mk,
            mla_v_multiplier=mv,
            exceeds_cap=exceeds_cap,
            ocp_signature=ocp_signature,
        )
        report.layer_audits.append(audit)
        if exceeds_cap:
            report.layers_above_cap.append(layer_idx)
        if ocp_signature:
            report.layers_with_ocp_signature.append(layer_idx)

    indexer_values = sorted(indexer.values())
    if indexer_values:
        report.max_indexer_multiplier = indexer_values[-1]
        report.median_indexer_multiplier = indexer_values[len(indexer_values) // 2]

    if report.indexer_layers_seen == 0:
        report.verdict = "unknown"
    elif report.layers_above_cap or report.layers_with_ocp_signature:
        report.verdict = "requantize_required"
    else:
        report.verdict = "clean"
    return report


# --- Requantization patch generation ----------------------------------------


def build_requantize_patch(
    report: AuditReport,
    *,
    downscale: float = WEIGHT_DOWNSCALE,
    include_clean_layers: bool = False,
) -> dict[str, Any]:
    """Produce a calibration-JSON patch that
    ``retarget-glm52-static-fp8-scales.py`` can apply.

    The output has the shape the canonical rewriter expects
    (``projection_input_scales`` + ``cache_quant_multipliers``). Only the
    indexer entries are populated — projection input scales are left empty
    so the rewriter is a no-op on the MLA-side path.

    Note that ``retarget-glm52-static-fp8-scales.py`` is strict: its
    ``_positive_map`` requires every checkpoint scalar be present in the
    input. That is a full-manifest rewrite. For a targeted indexer-only fix,
    the intended flow is:

        1. Read the current calibration file that was fed into
           ``checkpoint_converter.py`` at conversion time.
        2. Apply :func:`patch_calibration_dict` to it (rescaling only the
           indexer entries).
        3. Feed the patched file to
           ``retarget-glm52-static-fp8-scales.py`` to rewrite the on-disk
           safetensors bytes.

    This function returns only the *delta* — the caller merges it into the
    existing calibration JSON. That keeps the fix reversible and lets the
    reviewer see exactly which scalars moved.
    """

    if downscale <= 0 or downscale > 1:
        raise ValueError(
            "downscale must be in (0, 1]; got "
            f"{downscale}. WEIGHT_DOWNSCALE = 240/448 is the intended value."
        )

    patch: dict[str, float] = {}
    for audit in report.layer_audits:
        touched = audit.exceeds_cap or audit.ocp_signature
        if not (touched or include_clean_layers):
            continue
        name = (
            f"model.layers.{audit.layer_idx}.self_attn."
            "indexer.cache_quant_multiplier"
        )
        patch[name] = audit.indexer_multiplier * downscale
    return {
        "projection_input_scales": {},
        "cache_quant_multipliers": patch,
        "fix_metadata": {
            "downscale": downscale,
            "cap": MULTIPLIER_CAP,
            "layers_touched": len(patch),
            "source_report_verdict": report.verdict,
        },
    }


def patch_calibration_dict(
    calibration: Mapping[str, Any],
    patch: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge :func:`build_requantize_patch` output into a full calibration.

    The merged calibration is what a caller feeds to
    ``retarget-glm52-static-fp8-scales.py``. This function does the merge
    in-Python so the tests can verify the merge is a no-op on
    ``projection_input_scales`` and rewrites only the requested indexer
    ``cache_quant_multipliers`` keys.
    """

    result: dict[str, Any] = {
        "projection_input_scales": dict(
            calibration.get("projection_input_scales") or {}
        ),
        "cache_quant_multipliers": dict(
            calibration.get("cache_quant_multipliers") or {}
        ),
    }
    for key, value in (patch.get("projection_input_scales") or {}).items():
        result["projection_input_scales"][str(key)] = float(value)
    for key, value in (patch.get("cache_quant_multipliers") or {}).items():
        result["cache_quant_multipliers"][str(key)] = float(value)
    return result


# --- Load-time assertion ----------------------------------------------------


class IndexerMultiplierOutOfRange(ValueError):
    """Raised when a per-layer indexer multiplier exceeds the Trainium cap."""


def assert_indexer_multiplier_bounded(
    multiplier: Any,
    *,
    layer_idx: int | None = None,
    cap: float = MULTIPLIER_CAP,
) -> float:
    """Fail-closed check for a single indexer ``cache_quant_multiplier``.

    Meant to be dropped into
    ``Glm52FullIndexer.set_cache_quant_multiplier`` so the load path refuses
    an un-migrated checkpoint before it ever reaches a compile or a serve.

    Accepts a Python float, a NumPy scalar, or a torch scalar tensor.
    Duck-types on ``float()`` and ``.item()`` so the function is importable
    in staging without ``torch``.

    The check is ``value > cap`` (strict). A value AT the cap (exactly 240)
    is permitted — the Trainium kernel writes at ``clamp(-240, 240)`` and
    exactly-240 activations are representable in e4m3fn.
    """

    layer_tag = "" if layer_idx is None else f" for layer {layer_idx}"

    try:
        if hasattr(multiplier, "item") and callable(multiplier.item):
            value = float(multiplier.item())
        else:
            value = float(multiplier)
    except (TypeError, ValueError) as exc:
        raise IndexerMultiplierOutOfRange(
            f"indexer cache_quant_multiplier{layer_tag} is not a scalar"
        ) from exc
    if not math.isfinite(value):
        raise IndexerMultiplierOutOfRange(
            f"indexer cache_quant_multiplier{layer_tag} is non-finite: {value!r}"
        )
    if value <= 0:
        raise IndexerMultiplierOutOfRange(
            f"indexer cache_quant_multiplier{layer_tag} must be positive; "
            f"got {value!r}"
        )
    if value > cap:
        raise IndexerMultiplierOutOfRange(
            f"indexer cache_quant_multiplier{layer_tag} = {value!r} exceeds "
            f"Trainium2 e4m3fn max = {cap!r}. Re-quantize this checkpoint's "
            "indexer scalars via neuron_legacy_e4m3fn_qmax240 (multiply by "
            f"{WEIGHT_DOWNSCALE!r}) or re-run the converter with "
            "static_fp8_weight_format='neuron_legacy_e4m3fn_qmax240'."
        )
    return value


def install_load_time_assertion(indexer_cls: Any) -> None:  # pragma: no cover
    """Wrap ``Glm52FullIndexer.set_cache_quant_multiplier`` in-place.

    Intended for tests and one-off audits. Production integration should be
    the explicit patch upstream (see the PR body next to this file).
    """

    original = indexer_cls.set_cache_quant_multiplier

    def _wrapper(self: Any, multiplier: float) -> None:
        assert_indexer_multiplier_bounded(
            multiplier,
            layer_idx=getattr(self, "layer_idx", None),
        )
        original(self, multiplier)

    indexer_cls.set_cache_quant_multiplier = _wrapper  # type: ignore[attr-defined]


# --- CLI entry point --------------------------------------------------------


def _format_report_text(report: AuditReport) -> str:
    lines = [
        f"checkpoint : {report.checkpoint_path}",
        f"verdict    : {report.verdict}",
        f"indexer layers seen  : {report.indexer_layers_seen}",
        f"layers above cap     : {report.layers_above_cap}",
        f"layers with OCP sig  : {report.layers_with_ocp_signature}",
        f"max indexer mult     : {report.max_indexer_multiplier:.4f}",
        f"median indexer mult  : {report.median_indexer_multiplier:.4f}",
    ]
    if report.missing_mla_pairs:
        lines.append(
            f"layers missing MLA-k sibling : {report.missing_mla_pairs}"
        )
    return "\n".join(lines)


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    audit_p = sub.add_parser(
        "audit",
        help="scan a checkpoint directory or JSON manifest for out-of-range "
        "indexer multipliers",
    )
    audit_p.add_argument("source", type=Path)
    audit_p.add_argument("--json", action="store_true", help="emit JSON")

    patch_p = sub.add_parser(
        "patch",
        help="emit a calibration JSON delta that rescales offending indexer "
        "multipliers by WEIGHT_DOWNSCALE",
    )
    patch_p.add_argument("source", type=Path)
    patch_p.add_argument(
        "--output",
        type=Path,
        help="write the patch to this file (default: stdout)",
    )

    args = parser.parse_args(argv)

    if args.cmd == "audit":
        report = audit_indexer_scales(args.source)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        else:
            print(_format_report_text(report))
        return 0 if report.verdict != "requantize_required" else 2

    if args.cmd == "patch":
        report = audit_indexer_scales(args.source)
        patch = build_requantize_patch(report)
        text = json.dumps(patch, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text)
        return 0

    return 1  # pragma: no cover - argparse rejects unknown subcommands


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))
