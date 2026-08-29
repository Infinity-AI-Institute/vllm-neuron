# SPDX-License-Identifier: Apache-2.0
"""pytest conftest for the kernels/tests suite.

Registers CLI options and session-scoped fixtures consumed by
`test_no_cpu_fallback.py`.  Pytest requires `pytest_addoption` to live
in a `conftest.py` (or the plugin manager), never inside a test module
that is only collected — putting it in the test module itself causes
`getoption("--...")` to fail with "no option named ..." when other
suites are collected in the same session.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("no-cpu-fallback")
    group.addoption(
        "--compile-log",
        action="store",
        default=None,
        help=(
            "Path to a compile log to scan for CPU-fallback indicators. "
            "Use '-' to read from stdin."
        ),
    )
    group.addoption(
        "--artifact-dir",
        action="store",
        default=None,
        help=(
            "Path to a compiled artifact directory (contains model.pt + logs/); "
            "scans every log inside for CPU-fallback indicators."
        ),
    )
    group.addoption(
        "--runtime-probe",
        action="store_true",
        default=False,
        help=(
            "Run `neuron-top --sample 10` and verify no host-CPU activity "
            "beyond driver polling.  Skips if Trn2 is not accessible."
        ),
    )


@pytest.fixture(scope="session")
def compile_log_lines(request) -> Optional[List[str]]:
    path = request.config.getoption("--compile-log")
    if path is None:
        return None
    if path == "-":
        return sys.stdin.read().splitlines()
    log_path = Path(path)
    if not log_path.exists():
        pytest.fail(f"--compile-log path does not exist: {log_path}")
    return log_path.read_text(errors="replace").splitlines()


@pytest.fixture(scope="session")
def artifact_dir(request) -> Optional[Path]:
    d = request.config.getoption("--artifact-dir")
    if d is None:
        return None
    p = Path(d)
    if not p.exists():
        pytest.fail(f"--artifact-dir path does not exist: {p}")
    if not p.is_dir():
        pytest.fail(f"--artifact-dir must be a directory: {p}")
    return p


@pytest.fixture(scope="session")
def runtime_probe_enabled(request) -> bool:
    return bool(request.config.getoption("--runtime-probe"))
