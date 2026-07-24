"""Perf instrumentation for the flat vs scene_sparse latency A/B.

Measurement only -- no retrieval logic reads or branches on this module.
Callers (eval scripts) call reset() before a query, then read TIMINGS/COUNTS
after _build_retrieved returns. Single-threaded use only (module-level dict,
no locking).
"""
from __future__ import annotations

TIMINGS: dict[str, float] = {}
COUNTS: dict[str, int] = {}


def reset() -> None:
    TIMINGS.clear()
    COUNTS.clear()


def record_time(key: str, seconds: float) -> None:
    TIMINGS[key] = seconds


def record_count(key: str, value: int) -> None:
    COUNTS[key] = value
