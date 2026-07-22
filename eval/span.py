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

import numpy as np

# FROZEN 2026-07-18: half_width for mode="ppr_peak", duration-anchor method --
# median gold half-span on the N=64 in-sample grounded set (dev_100.jsonl ∩
# index_cache ∩ gsub_val.json). Confirmation sweep over {1.0..6.0}s showed
# mIoP flat (0.2510-0.2596, all bootstrap CIs overlapping); the anchor (2.2)
# fell inside the argmax's CI, so it stands per the pre-registered freeze
# rule (anchor wins unless statistically worse than argmax). NOT an argmax
# pick. This is the single authoritative source -- every call site below
# reads this constant rather than hardcoding its own literal. predict_span()
# itself keeps half_width as a required arg with no baked-in default; this
# constant is what callers pass in.
FROZEN_HALF_WIDTH_SECONDS: float = 2.2

# Priority order for the timestamp field on a frame (dict key or attribute).
_TIMESTAMP_FIELDS = ("timestamp", "timestamp_sec")

# Priority order for the relevance-score field on a frame. last_retrieval_score
# is the query-specific PPR score set by L2Asphodel.retrieve_ppr(); pagerank_score
# is the static structural score. score/pagerank are generic fallback names for
# frame objects outside this codebase's retrieval dict/FrameRecord shapes.
_SCORE_FIELDS = ("last_retrieval_score", "score", "pagerank_score", "pagerank")

# Field carrying the frame's CLIP embedding (set by iris.query._build_retrieved
# from FrameRecord.clip_embedding -- the SAME embedding L2Asphodel.retrieve_ppr
# scored sem_rank against, not a second embedding path).
_EMBEDDING_FIELDS = ("clip_embedding", "embedding")

# peak_source="ppr_score": t* = frame with the highest PPR/retrieval score
#   (the original ppr_peak behavior, retained verbatim as a named ablation arm).
# peak_source="clip_in_ppr_top8": t* = frame with the highest raw CLIP cosine
#   similarity to the query embedding, chosen ONLY among the frames already
#   passed in (i.e. whichever set retrieval -- PPR or otherwise -- already
#   returned). This does not change which frames are retrieved; it only
#   changes which of those already-retrieved frames the span centers on.
#   Diagnostic finding (2026-07-19, read-only PPR-peak-selection probe): PPR's
#   own top-1 wins on a query-blind codec_rank term 79% of the time on the
#   zero-IoP failure set while showing no systematic query-relevance edge over
#   the correct in-gold frame (sem_rank win-rate ~53%, coin-flip) -- i.e. PPR
#   assembles a good top-8 but then mis-picks the peak within it. This is the
#   new default; falls back to peak_source="ppr_score" automatically when no
#   usable CLIP signal is available (query_embedding=None, or no frame carries
#   a usable embedding), so callers that don't thread query_embedding through
#   keep the old behavior unchanged.
_PEAK_SOURCES = ("clip_in_ppr_top8", "ppr_score")


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


def _frame_embedding(frame: Any) -> Any | None:
    if isinstance(frame, dict):
        for name in _EMBEDDING_FIELDS:
            if name in frame and frame[name] is not None:
                return frame[name]
        return None
    for name in _EMBEDDING_FIELDS:
        if hasattr(frame, name):
            val = getattr(frame, name)
            if val is not None:
                return val
    return None


def _pick_by_clip_similarity(frames: Sequence[Any], query_embedding: Any) -> Any | None:
    """Among `frames` (the already-retrieved set -- unchanged), return the one
    with highest raw cosine similarity to query_embedding. Returns None (caller
    falls back to peak_source="ppr_score") when the query embedding is
    zero-norm or no frame carries a usable embedding -- never raises, since a
    missing embedding is not a contract violation for callers that only ever
    used the ppr_score path."""
    q = np.asarray(query_embedding, dtype=np.float64).flatten()
    q_norm = float(np.linalg.norm(q))
    if q_norm < 1e-8:
        return None

    best_frame = None
    best_sim = None
    for f in frames:
        emb = _frame_embedding(f)
        if emb is None:
            continue
        e = np.asarray(emb, dtype=np.float64).flatten()
        e_norm = float(np.linalg.norm(e))
        if e_norm < 1e-8:
            continue
        sim = float(np.dot(e, q) / (e_norm * q_norm))
        if best_sim is None or sim > best_sim:
            best_sim = sim
            best_frame = f
    return best_frame


def _pick_by_ppr_score(frames: Sequence[Any]) -> Any:
    """Original ppr_peak selection: highest last_retrieval_score/score/pagerank
    field, or retrieval rank order (first element) when no frame carries a score."""
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
        best_frame = frames[0]
    return best_frame


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
    peak_source: str = "clip_in_ppr_top8",
    query_embedding: Any | None = None,
    return_peak: bool = False,
) -> tuple[float, float] | tuple[tuple[float, float] | None, float | None] | None:
    """Build the predicted temporal span from retrieved frames.

    Returns (start, end), or None for empty input.
    When return_peak=True, returns ((start, end), peak_timestamp).

    frames: sequence with a timestamp (.timestamp or .timestamp_sec — normalize
            INSIDE this function; callers must not adapt) and a relevance score
            (.score / .pagerank, else retrieval rank order).
    mode="minmax":   min(ts) -> max(ts). The legacy construction. THIS IS THE BUG.
                     Retained ONLY as an explicit ablation arm. Never the default.
    mode="ppr_peak": t* = timestamp of the peak frame (chosen per peak_source,
                     among `frames` -- this never changes WHICH frames were
                     retrieved, only which of them the span centers on);
                     span = [t*-half_width, t*+half_width], clipped to [0, duration].

    peak_source (only meaningful for mode="ppr_peak"):
        "clip_in_ppr_top8" (default): t* = the frame among `frames` with the
            highest raw CLIP cosine similarity to query_embedding. Requires
            query_embedding; falls back to "ppr_score" automatically if
            query_embedding is None or no frame carries a usable embedding.
        "ppr_score": t* = the frame among `frames` with the highest PPR/
            retrieval score (original ppr_peak behavior). Retained as a named,
            explicit ablation arm -- pass peak_source="ppr_score" to reproduce
            pre-2026-07-19 behavior exactly.
    query_embedding: the SAME query embedding retrieval already used (e.g.
        iris.query._embed_query's return value) -- required for
        peak_source="clip_in_ppr_top8"; not a second embedding computation.
    """
    if not frames:
        return (None, None) if return_peak else None

    if mode == "minmax":
        timestamps = [_frame_timestamp(f) for f in frames]
        span = (min(timestamps), max(timestamps))
        return (span, None) if return_peak else span

    if mode == "ppr_peak":
        if half_width is None:
            raise ValueError(
                "predict_span(mode='ppr_peak') requires half_width. It is deliberately "
                "unset here — tuned on val and frozen in a later task. Do not invent a value."
            )
        if peak_source not in _PEAK_SOURCES:
            raise ValueError(f"Unknown peak_source: {peak_source!r}")

        best_frame = None
        if peak_source == "clip_in_ppr_top8" and query_embedding is not None:
            best_frame = _pick_by_clip_similarity(frames, query_embedding)
        if best_frame is None:
            # peak_source="ppr_score", or clip_in_ppr_top8 had no usable CLIP
            # signal (no query_embedding / no frame embeddings) -- fall back
            # to the original score-based peak rather than inventing a value.
            best_frame = _pick_by_ppr_score(frames)

        t_star = _frame_timestamp(best_frame)
        start = t_star - half_width
        end = t_star + half_width
        start = max(start, 0.0)
        if duration is not None:
            end = min(end, duration)
        span = (start, end)
        return (span, t_star) if return_peak else span

    raise ValueError(f"Unknown span mode: {mode!r}")

