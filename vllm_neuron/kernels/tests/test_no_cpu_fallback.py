# SPDX-License-Identifier: Apache-2.0
"""Universal no-CPU-fallback assertion test module.

**Discipline:** every future compile lane on any Neuron model MUST hit
this test module.  If a compile emits ANY CPU-fallback indicator we
know how to detect, we FAIL LOUDLY here.  The point is to make it
impossible to ship a NEFF that silently fell back — the failure mode
we ate on Gemma-4-26B-A4B (MFU 0.06%).

Three coverage layers
---------------------

1. **Compile-log grep** — pure text search over the compile log.
   Detects the canonical CPU-fallback messages emitted by the Neuron
   compiler and NxDI.  Runs on ANY host, requires no Trn2.

2. **NEFF-content assertion** — for a landed compile artifact dir
   (`<slug>/logs/` and/or `<slug>/model.pt`), verify:
     - the compile log inside the artifact dir has no fallback warnings;
     - (best-effort) if `neuron-mlir-tool` or `neuron_cc.disasm` is
       available, no unresolved `stablehlo` ops that lower to host.

3. **Runtime probe** — during a served inference, verify that
   `neuron-top --sample 10` shows no host-CPU activity beyond driver
   polling.  Best-effort, skips cleanly when Trn2 is not accessible.

Usage
-----

    # Standalone compile-log check (any host):
    pytest kernels/tests/test_no_cpu_fallback.py \\
        --compile-log /path/to/compile.log

    # Piped via stdin:
    cat compile.log | pytest kernels/tests/test_no_cpu_fallback.py \\
        --compile-log -

    # Artifact-dir NEFF-content assertion:
    pytest kernels/tests/test_no_cpu_fallback.py \\
        --artifact-dir /path/to/compiled/model_dir

    # Runtime probe (best-effort, requires Trn2):
    pytest kernels/tests/test_no_cpu_fallback.py \\
        --runtime-probe

    # Full battery:
    pytest kernels/tests/test_no_cpu_fallback.py \\
        --compile-log compile.log --artifact-dir /path/to/model_dir --runtime-probe

CLI options are declared in `tests/conftest.py`.

Fail-loud grep pattern
----------------------

The grep pattern below is deliberately broad.  Rationale for each
alternative:

  * `falling back to cpu`               — Neuron compiler top-level
                                          fall-through message.
  * `torch_blockwise_matmul_inference`  — Gemma-4 §B5 hazard: MoE
                                          silently CPU-executed.
  * `op fallback`                       — NxDI generic op-fallback
                                          warning.
  * `cpu-side`                          — StablHLO->CPU handoff
                                          announcement.
  * `emitting host code`                — Neuron compiler stage log
                                          when an op lowers to host
                                          instead of the NC/GPSIMD.
  * `nki .* not (found|available)`      — the NKI subsystem itself
                                          absent → fall to CPU.
  * `partition cap exceeded`            — nc_find_index8 / other
                                          per-partition caps that
                                          demote a graph to host.
  * `unsupported operation.*fallback`   — a per-op unsupported /
                                          fallback pair.

A match anywhere in the compile log fails the test.  The grep pattern
constant `CPU_FALLBACK_GREP_PATTERNS` is exported for downstream reuse
by other harness rules.

This test module is universal — it is NOT Gemma-4-specific despite
sitting under the gemma-4 kernels tree — and every compile lane on
every model should invoke it.
"""

from __future__ import annotations

import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pytest


# ==========================================================================
# Fail-loud grep patterns (exported for downstream reuse)
# ==========================================================================

CPU_FALLBACK_GREP_PATTERNS: Tuple[str, ...] = (
    r"falling back to cpu",
    r"torch_blockwise_matmul_inference",
    r"op fallback",
    r"cpu-side",
    r"emitting host code",
    r"nki\b.*\bnot (?:found|available)",
    r"partition cap exceeded",
    r"unsupported operation.*fallback",
    r"fell back to (?:cpu|host|torch)",
    r"host code emitted",
)

# Precompiled union pattern — matches ANY of the alternatives, case-insensitive.
_CPU_FALLBACK_UNION_RE = re.compile(
    "|".join(f"(?:{p})" for p in CPU_FALLBACK_GREP_PATTERNS),
    re.IGNORECASE,
)


def find_cpu_fallback_matches(lines: Iterable[str]) -> List[Tuple[int, str, str]]:
    """Scan lines for CPU-fallback indicators.

    Returns
    -------
    List[(line_number_1_indexed, matched_substring, full_line)]
        Empty list if clean.
    """
    matches: List[Tuple[int, str, str]] = []
    for idx, line in enumerate(lines, start=1):
        m = _CPU_FALLBACK_UNION_RE.search(line)
        if m:
            matches.append((idx, m.group(0), line.rstrip("\n")))
    return matches


# ==========================================================================
# Make the sibling kernels/ dir importable so we can smoke-test
# `gemma4_no_fallback_mitigations` from here.
# ==========================================================================

_HERE = Path(__file__).resolve().parent
_KERNELS_DIR = _HERE.parent
if str(_KERNELS_DIR) not in sys.path:
    sys.path.insert(0, str(_KERNELS_DIR))


# ==========================================================================
# 1. Compile-log grep test
# ==========================================================================


class TestCompileLogGrep:
    """Fail loudly if the compile log contains any CPU-fallback indicator."""

    def test_grep_patterns_are_nonempty(self):
        """The pattern list is the discipline; a truncation is a bug."""
        assert len(CPU_FALLBACK_GREP_PATTERNS) >= 8, (
            "CPU_FALLBACK_GREP_PATTERNS truncated — restore the full pattern list."
        )
        for pat in CPU_FALLBACK_GREP_PATTERNS:
            # Every pattern must compile as-is.
            re.compile(pat, re.IGNORECASE)

    def test_grep_matches_known_positive_examples(self):
        """Sanity: the pattern actually catches the strings we care about."""
        positives = [
            "WARNING: falling back to CPU for op aten.foo",
            "op fallback: emitting host code for stablehlo.bar",
            "Using torch_blockwise_matmul_inference (CPU path)",
            "NKI not available; falling back to CPU",
            "partition cap exceeded; op nc_find_index8 demoted to host",
            "unsupported operation aten.baz; fallback to torch",
            "Fell back to host for op quux",
        ]
        for s in positives:
            assert _CPU_FALLBACK_UNION_RE.search(s), (
                f"CPU_FALLBACK_GREP_PATTERNS failed to match a known positive: {s!r}"
            )

    def test_grep_ignores_known_negative_examples(self):
        """Sanity: the pattern does NOT trigger on benign lines."""
        negatives = [
            "Compile complete: NEFF written to /tmp/model.neff",
            "TPOT=8.79 ms; throughput=114 tok/s",
            "NKI kernel dsa_sparse_attention.nki_v1 loaded",
            "loading weights: 20.85 s",
            # A message that mentions CPU is fine as long as it's not a fallback:
            "Compile-driver host CPU: 8 cores, 32 GB",
        ]
        for s in negatives:
            assert not _CPU_FALLBACK_UNION_RE.search(s), (
                f"CPU_FALLBACK_GREP_PATTERNS triggered on benign line: {s!r}"
            )

    def test_compile_log_is_clean(self, compile_log_lines):
        """FAIL if a --compile-log was provided and it contains any indicator.

        When no --compile-log is passed, this test is SKIPPED (not passed) so
        the CI operator sees explicitly that the check was not invoked.
        """
        if compile_log_lines is None:
            pytest.skip(
                "no --compile-log passed; skipping (call with "
                "`pytest ... --compile-log <path>` to activate)"
            )
        matches = find_cpu_fallback_matches(compile_log_lines)
        assert not matches, (
            f"CPU-fallback indicator(s) found in compile log:\n"
            + "\n".join(
                f"  line {ln}: matched {pat!r} in: {line}"
                for ln, pat, line in matches[:20]
            )
            + (f"\n  ... ({len(matches) - 20} more)" if len(matches) > 20 else "")
        )


# ==========================================================================
# 2. NEFF-content assertion (artifact-dir scan)
# ==========================================================================


class TestArtifactDirClean:
    """Scan every log inside an artifact dir for CPU-fallback indicators."""

    def test_artifact_dir_logs_are_clean(self, artifact_dir):
        if artifact_dir is None:
            pytest.skip(
                "no --artifact-dir passed; skipping (call with "
                "`pytest ... --artifact-dir <path>` to activate)"
            )
        log_files = list(artifact_dir.rglob("*.log")) + list(
            artifact_dir.rglob("*.txt")
        )
        if not log_files:
            pytest.skip(
                f"no *.log or *.txt files found under {artifact_dir}; nothing to scan"
            )
        problems: List[str] = []
        for log_file in log_files:
            try:
                lines = log_file.read_text(errors="replace").splitlines()
            except OSError as e:
                problems.append(f"could not read {log_file}: {e}")
                continue
            matches = find_cpu_fallback_matches(lines)
            for ln, pat, line in matches:
                problems.append(
                    f"{log_file}:{ln} matched {pat!r} in: {line}"
                )
        assert not problems, (
            "CPU-fallback indicator(s) found in artifact-dir logs:\n"
            + "\n".join(problems[:20])
            + (f"\n  ... ({len(problems) - 20} more)" if len(problems) > 20 else "")
        )

    def test_artifact_dir_has_no_stablehlo_host_lowerings(self, artifact_dir):
        """Best-effort: if `neuron-mlir-tool` (or `neuron_cc.disasm`) exists,
        run it against the NEFF and grep for unresolved `stablehlo` ops that
        would lower to host.

        Skips cleanly on any host where neither tool is available.
        """
        if artifact_dir is None:
            pytest.skip("no --artifact-dir passed")
        tool = shutil.which("neuron-mlir-tool") or shutil.which("neuron_cc.disasm")
        if tool is None:
            pytest.skip(
                "neither `neuron-mlir-tool` nor `neuron_cc.disasm` on PATH; "
                "NEFF-content lowering scan requires the Neuron compiler tools"
            )
        neffs = list(artifact_dir.rglob("*.neff")) + list(
            artifact_dir.rglob("model.pt")
        )
        if not neffs:
            pytest.skip(f"no *.neff or model.pt under {artifact_dir}; nothing to disasm")

        problems: List[str] = []
        for neff in neffs:
            try:
                out = subprocess.run(
                    [tool, "--emit-mlir", str(neff)],
                    capture_output=True, text=True, timeout=60,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                problems.append(f"{tool} on {neff}: {e}")
                continue
            combined = (out.stdout or "") + (out.stderr or "")
            # An unresolved stablehlo op → lowers to host.
            unresolved = re.findall(
                r"stablehlo\.\w+.*host_lowered", combined, re.IGNORECASE
            )
            if unresolved:
                problems.append(
                    f"{neff}: {len(unresolved)} stablehlo op(s) lower to host"
                )
        assert not problems, (
            "NEFF contains stablehlo ops that lower to host:\n"
            + "\n".join(problems)
        )


# ==========================================================================
# 3. Runtime probe test (best-effort, skips without Trn2)
# ==========================================================================


class TestRuntimeProbe:
    """Verify a served inference does not spike host CPU beyond driver polling."""

    def _neuron_top_available(self) -> bool:
        return shutil.which("neuron-top") is not None

    def _trn2_accessible(self) -> bool:
        if not self._neuron_top_available():
            return False
        try:
            out = subprocess.run(
                ["neuron-top", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            return out.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def test_runtime_probe_no_host_cpu_activity(self, runtime_probe_enabled):
        if not runtime_probe_enabled:
            pytest.skip("--runtime-probe not passed; skipping runtime probe")
        if not self._trn2_accessible():
            pytest.skip("neuron-top not available or Trn2 not accessible")

        # `neuron-top --sample 10` samples every ~1s for 10 samples.
        # We look for the "CPU" column in the output; a CPU utilization
        # consistently above the driver-poll baseline (~5%) across samples
        # indicates the compute path is running on host.
        try:
            out = subprocess.run(
                ["neuron-top", "--sample", "10"],
                capture_output=True, text=True, timeout=60,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            pytest.fail(f"neuron-top invocation failed: {e}")

        # Parse "CPU%" values; the exact column layout depends on the
        # Neuron SDK version, so we look for any float% value adjacent to
        # a "CPU" or "cpu%" header and cap the max at 20% (well above driver
        # polling, well below a compute path).
        text = (out.stdout or "") + (out.stderr or "")
        # Extract any "CPU: 12.3%" or "cpu% 12.3" style values.
        cpu_pcts = [
            float(m.group(1))
            for m in re.finditer(r"cpu[%\s:]*\s*([\d.]+)\s*%", text, re.IGNORECASE)
        ]
        if not cpu_pcts:
            pytest.skip("neuron-top output did not include a parseable CPU% column")

        max_cpu = max(cpu_pcts)
        assert max_cpu < 20.0, (
            f"neuron-top reported max CPU={max_cpu:.1f}% across 10 samples; "
            f"driver polling baseline is ~5%.  A sustained >20% CPU suggests "
            f"the compute path is running on host.  Raw samples: {cpu_pcts}"
        )


# ==========================================================================
# 4. Universal-harness smoke tests (no --compile-log required)
# ==========================================================================


class TestUniversalHarnessSmoke:
    """Prove the module is importable and the pattern list is stable.

    These are the tests that run in the default `pytest -q` sweep — they
    ensure the guard-rail file itself has not regressed.  Add-only.
    """

    def test_module_importable(self):
        """The test module + its patterns must import cleanly by file path.

        Uses `importlib.util.spec_from_file_location` so it works from any
        pytest rootdir / cwd — the tests directory is not on sys.path.
        """
        spec = importlib.util.spec_from_file_location(
            "test_no_cpu_fallback_self", str(Path(__file__).resolve())
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module.find_cpu_fallback_matches)
        assert module.CPU_FALLBACK_GREP_PATTERNS

    def test_stdin_reader_smoke(self):
        """`find_cpu_fallback_matches` works on any Iterable[str] source."""
        clean = io.StringIO(
            "Compile started\n"
            "TP degree 8\n"
            "NKI kernel loaded\n"
            "Compile complete: 12.3 s\n"
        )
        assert find_cpu_fallback_matches(clean) == []

        dirty = io.StringIO(
            "Compile started\n"
            "WARNING: falling back to CPU for op aten.foo\n"
            "Compile complete\n"
        )
        matches = find_cpu_fallback_matches(dirty)
        assert len(matches) == 1
        assert matches[0][0] == 2  # 1-indexed line number
        assert "falling back to cpu" in matches[0][1].lower()

    def test_gemma4_no_fallback_mitigations_importable(self):
        """The renamed mitigation module must still import cleanly."""
        import gemma4_no_fallback_mitigations as g
        assert callable(g.should_disable_argmax_kernel)
        assert callable(g.verify_activation_branch_coverage)
        assert callable(g.import_pr172_flash_attention)
        assert callable(g.import_pr172_kv_cache_manager)
        # Trigger #3 mitigation: at TP=8, vocab=262144, B=256 → True.
        assert g.should_disable_argmax_kernel(
            vocab_size=262_144, tp_degree=8, batch_size=256,
        ) is True
        # At B=1 (PR #172's validation), no disable needed at TP=8 for smaller vocab.
        assert g.should_disable_argmax_kernel(
            vocab_size=32_000, tp_degree=8, batch_size=1,
        ) is False
