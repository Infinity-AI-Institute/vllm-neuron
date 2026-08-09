# SPDX-License-Identifier: Apache-2.0
"""Best-effort machine-readable progress records for graph extraction."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from vllm_neuron import envs

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_MAX_TEXT_FIELD = 8192


def _bounded(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_TEXT_FIELD:
        return value[:_MAX_TEXT_FIELD] + "...<truncated>"
    return value


def emit_trace_milestone(
    event: str,
    *,
    parent_rank: int,
    stage: str,
    **fields: Any,
) -> None:
    """Append one JSON record with one ``os.write`` call.

    The sink is intentionally best effort: inability to write diagnostics must
    not turn a valid graph into a failed graph. A separate file per rank avoids
    cross-rank line interleaving; fork children for one rank use ``O_APPEND``.
    """
    directory = envs.VLLM_NEURON_TRACE_MILESTONE_DIR
    if directory is None:
        return

    record = {
        "schema_version": _SCHEMA_VERSION,
        "event": event,
        "stage": stage,
        "parent_rank": parent_rank,
        "pid": os.getpid(),
        "wall_time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
    }
    record.update({key: _bounded(value) for key, value in fields.items()})
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()

    try:
        root = Path(directory)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"unsafe milestone directory: {root}")
        path = root / f"rank-{parent_rank}.jsonl"
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    except Exception:
        logger.warning(
            "Unable to write trace milestone event=%s rank=%d",
            event,
            parent_rank,
            exc_info=True,
        )
