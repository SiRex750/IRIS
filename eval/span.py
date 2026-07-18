"""Shared predicted-span constructor for temporal grounding metrics.

DECISIONS.md 2026-07-17 §3 (copy count corrected 2026-07-17-later §A6): the
predicted grounding span was built independently in three places as
min(ts)->max(ts) over the top-K retrieved frames. Scattered top-K frames then
enclosed almost the whole video, so IoP collapsed even when retrieval hit the
correct region. This module is the single replacement; every call site
imports predict_span() instead of building its own span.
"""
from __future__ import annotations

from typing import Any, Sequence

# Priority order for the timestamp field on a frame (dict key or attribute).
_TIMESTAMP_FIELDS = ("timestamp", "timestamp_sec")

# Priority order for the relevance-score field on a frame. last_retrieval_score
# is the query-specific PPR score set by L2Asphodel.retrieve_ppr(); pagerank_score
# is the static structural score. score/pagerank are generic fallback names for
# frame objects outside this codebase's retrieval dict/FrameRecord shapes.
_SCORE_FIELDS = ("last_retrieval_score", "score", "pagerank_score", "pagerank")


def _get_field(frame: Any, names: Sequence[str]) -> float | None:
    if isinstance(frame, dict):
        for name in names:
            if name in frame and frame[name] is not None:
                return float(frame[name])
        return None
    for name in names:
        if hasattr(frame, name):
            val = getattr(frame, name)
            if val is not None:
                return float(val)
    return None


def _frame_timestamp(frame: Any) -> float:
    ts = _get_field(frame, _TIMESTAMP_FIELDS)
    if ts is None:
        raise ValueError(
            f"frame has neither a 'timestamp' nor 'timestamp_sec' field: {frame!r}"
        )
    return ts


def predict_span(
    frames: Sequence[Any],
    mode: str = "ppr_peak",
    half_width: float | None = None,
    duration: float | None = None,
) -> tuple[float, float] | None:
    """Build the predicted temporal span from retrieved frames.

    Returns (start, end), or None for empty input.

    frames: sequence with a timestamp (.timestamp or .timestamp_sec — normalize
            INSIDE this function; callers must not adapt) and a relevance score
            (.score / .pagerank, else retrieval rank order).
    mode="minmax":   min(ts) -> max(ts). The legacy construction. THIS IS THE BUG.
                     Retained ONLY as an explicit ablation arm. Never the default.
    mode="ppr_peak": t* = timestamp of the highest-scoring frame;
                     span = [t*-half_width, t*+half_width], clipped to [0, duration].
    """
    if not frames:
        return None

    if mode == "minmax":
        timestamps = [_frame_timestamp(f) for f in frames]
        return (min(timestamps), max(timestamps))

    if mode == "ppr_peak":
        if half_width is None:
            raise ValueError(
                "predict_span(mode='ppr_peak') requires half_width. It is deliberately "
                "unset here — tuned on val and frozen in a later task. Do not invent a value."
            )

        best_frame = None
        best_score = None
        have_scores = all(_get_field(f, _SCORE_FIELDS) is not None for f in frames)
        if have_scores:
            for f in frames:
                s = _get_field(f, _SCORE_FIELDS)
                if best_score is None or s > best_score:
                    best_score = s
                    best_frame = f
        else:
            # No usable score on any frame: fall back to retrieval rank order
            # (first element = most relevant), per this function's contract.
            best_frame = frames[0]

        t_star = _frame_timestamp(best_frame)
        start = t_star - half_width
        end = t_star + half_width
        start = max(start, 0.0)
        if duration is not None:
            end = min(end, duration)
        return (start, end)

    raise ValueError(f"Unknown span mode: {mode!r}")
