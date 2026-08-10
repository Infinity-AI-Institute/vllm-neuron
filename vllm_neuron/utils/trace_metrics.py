# SPDX-License-Identifier: Apache-2.0
"""Low-overhead, per-pipeline receipts for FX graph capture."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _peak_rss_bytes() -> int:
    """Return this process's resident-set high-water mark when available."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        return value if value > (1 << 30) else value * 1024
    except (ImportError, OSError, ValueError):
        return 0


@dataclass
class TraceMetrics:
    """Counters and timings owned by exactly one FX-to-HLO invocation."""

    fast_trace: bool
    pid: int = field(default_factory=os.getpid)
    started_perf_s: float = field(default_factory=time.perf_counter, repr=False)
    started_peak_rss_bytes: int = field(default_factory=_peak_rss_bytes)
    graph_string_renders: int = 0
    graph_code_renders: int = 0
    graph_dump_files: int = 0
    graph_dump_files_suppressed: int = 0
    failure_diagnostics: int = 0
    pass_wall_seconds: dict[str, float] = field(default_factory=dict)
    trace_wall_seconds: float = 0.0
    peak_rss_bytes: int = 0
    peak_rss_delta_bytes: int = 0
    success: bool | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        # These are deliberately dynamic rather than dataclass fields so the
        # optional observability machinery never enters the receipt schema.
        self._event_seq = 0
        self._event_path: Path | None = None
        self._heartbeat_stop: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._event_phase: str | None = None

    @staticmethod
    def _events_enabled() -> bool:
        return os.environ.get("VLLM_NEURON_TRACE_EVENTS") == "1"

    def _emit_event(self, workdir: str, event: str, phase: str, **extra: Any) -> None:
        if not self._events_enabled():
            return
        path = Path(workdir) / "trace_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._event_seq += 1
        row = {
            "schema_version": 1,
            "run_id": os.environ.get("VLLM_NEURON_TRACE_RUN_ID"),
            "event_seq": self._event_seq,
            "event": event,
            "phase": phase,
            "pid": self.pid,
            "rank": os.environ.get("RANK"),
            "workdir": str(Path(workdir)),
            "wall_time_utc": time.time(),
            "monotonic_ns": time.monotonic_ns(),
            **extra,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()

    def begin_phase(self, workdir: str, phase: str) -> None:
        """Record a phase and emit opt-in heartbeats without changing the graph."""
        if not self._events_enabled():
            return
        self._stop_heartbeat()
        self._event_path = Path(workdir) / "trace_events.jsonl"
        self._event_phase = phase
        self._emit_event(workdir, "phase_started", phase)
        try:
            interval = float(os.environ.get("VLLM_NEURON_TRACE_EVENT_INTERVAL_SECONDS", "30"))
        except ValueError:
            interval = 30.0
        interval = max(1.0, interval)
        stop = threading.Event()
        self._heartbeat_stop = stop

        def heartbeat() -> None:
            while not stop.wait(interval):
                self._emit_event(workdir, "phase_heartbeat", phase)

        thread = threading.Thread(target=heartbeat, name="neuron-trace-heartbeat", daemon=True)
        self._heartbeat_thread = thread
        thread.start()

    def _stop_heartbeat(self) -> None:
        stop = self._heartbeat_stop
        thread = self._heartbeat_thread
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._heartbeat_stop = None
        self._heartbeat_thread = None

    def finish(self, error: BaseException | None = None) -> None:
        self._stop_heartbeat()
        if self._events_enabled() and self._event_path is not None:
            self._emit_event(
                str(self._event_path.parent),
                "phase_finished",
                self._event_phase or "unknown",
                success=error is None,
                error_type=type(error).__name__ if error is not None else None,
            )
        self.trace_wall_seconds = time.perf_counter() - self.started_perf_s
        self.peak_rss_bytes = _peak_rss_bytes()
        self.peak_rss_delta_bytes = max(
            0, self.peak_rss_bytes - self.started_peak_rss_bytes
        )
        self.success = error is None
        self.error_type = type(error).__name__ if error is not None else None
        self.error_message = str(error) if error is not None else None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("started_perf_s", None)
        result["schema_version"] = 1
        result["mode"] = "fast" if self.fast_trace else "baseline"
        return result

    def write(self, workdir: str) -> str:
        """Atomically write this pipeline's receipt and return its path."""
        path = Path(workdir) / "trace_metrics.json"
        tmp_path = path.with_name(f".{path.name}.{self.pid}.tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        os.replace(tmp_path, path)
        return str(path)


def render_graph(gm, metrics: TraceMetrics | None) -> str:
    if metrics is not None:
        metrics.graph_string_renders += 1
    return str(gm.graph)


def render_code(gm, metrics: TraceMetrics | None) -> str:
    if metrics is not None:
        metrics.graph_code_renders += 1
    return gm.code
