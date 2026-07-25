"""One canonical FrameRecord/AsphodelNode -> retrieval-dict serializer.

Before this module existed, three call sites built their own frame dict:
iris.query._build_retrieved, iris.scene_retrieval._frame_to_dict /
_node_to_dict, and iris.query_reformulation._frame_record_to_retrieved_dict.
They drifted -- e.g. the query_reformulation copy silently dropped
scene_id/tier/pict_type/codec_conf/motion-geometry fields that a *retrieved*
frame dict carries, so a temporally-expanded frame did not expose the same
schema as a directly-retrieved one. This module is the single source of
truth; callers that need a dict from a FrameRecord or an AsphodelNode should
build it here instead of inlining another copy.

Two distinct ordering concepts are threaded through on purpose and must never
be conflated:
  retrieval_rank:      0-based position in relevance order (best match first).
  presentation_order:  0-based position in chronological (timestamp) order,
                        used only when handing frames to a captioner/answerer.
Sorting a frame list into presentation_order must not overwrite or discard
retrieval_rank.
"""
from __future__ import annotations

from typing import Any


def frame_record_to_dict(
    fr: Any,
    *,
    last_retrieval_score: float = 0.0,
    retrieval_contributions: dict | None = None,
    pagerank_score: float | None = None,
) -> dict[str, Any]:
    """Canonical dict for a FrameRecord that was not scored by the graph
    (e.g. a temporal-neighbor expansion pull, or the no-graph fallback sort).

    Every field the schema defines is populated from the FrameRecord when
    the FrameRecord actually carries it; only fields the graph computes
    (pagerank_score/last_retrieval_score/retrieval_contributions) may be
    passed in explicitly, defaulting to neutral values -- never silently
    zeroed when the source has a real value.
    """
    return {
        "frame_idx": fr.frame_idx,
        "timestamp": fr.timestamp,
        "scene_id": getattr(fr, "scene_id", -1),
        "tier": "L1_PEAK" if getattr(fr, "is_peak", False) else "L3_CANDIDATE",
        "pict_type": getattr(fr, "pict_type", "?"),
        "codec_conf": getattr(fr, "codec_conf", 0.5),
        "luma_diff_energy": fr.luma_diff_energy,
        "luma_entropy": fr.luma_entropy,
        "action_score": fr.action_score,
        "persistence_value": fr.persistence_value,
        "is_peak": fr.is_peak,
        "clip_embedding": fr.clip_embedding,
        "caption": fr.caption,
        "pagerank_score": pagerank_score if pagerank_score is not None else getattr(fr, "pagerank_score", 0.0),
        "last_retrieval_score": last_retrieval_score,
        "motion_magnitude": getattr(fr, "motion_magnitude", 0.0),
        "divergence": getattr(fr, "divergence", 0.0),
        "curl": getattr(fr, "curl", 0.0),
        "jacobian_frobenius": getattr(fr, "jacobian_frobenius", 0.0),
        "hessian_max_eigenvalue": getattr(fr, "hessian_max_eigenvalue", 0.0),
        "motion_entropy": getattr(fr, "motion_entropy", 0.0),
        "retrieval_contributions": dict(retrieval_contributions) if retrieval_contributions else {},
        "retrieval_rank": None,
        "presentation_order": None,
    }


def node_to_dict(node: Any, fr: Any | None) -> dict[str, Any]:
    """Canonical dict for an AsphodelNode returned by graph retrieval/PPR,
    filled out with the matching FrameRecord for fields the graph node does
    not itself carry (is_peak/caption/pict_type/scene_id/motion geometry)."""
    return {
        "frame_idx": node.frame_idx,
        "timestamp": node.timestamp,
        "scene_id": getattr(fr, "scene_id", getattr(node, "scene_id", -1)),
        "tier": getattr(node, "tier", None),
        "pict_type": getattr(fr, "pict_type", getattr(node, "pict_type", "?")),
        "codec_conf": getattr(node, "codec_conf", getattr(fr, "codec_conf", 0.5)),
        "luma_diff_energy": node.luma_diff_energy,
        "luma_entropy": getattr(fr, "luma_entropy", 0.0),
        "action_score": node.action_score,
        "persistence_value": node.persistence_value,
        "is_peak": getattr(fr, "is_peak", False),
        "clip_embedding": getattr(fr, "clip_embedding", None),
        "caption": getattr(fr, "caption", None),
        "pagerank_score": node.pagerank_score,
        "last_retrieval_score": getattr(node, "last_retrieval_score", 0.0),
        "motion_magnitude": getattr(fr, "motion_magnitude", 0.0),
        "divergence": getattr(fr, "divergence", 0.0),
        "curl": getattr(fr, "curl", 0.0),
        "jacobian_frobenius": getattr(fr, "jacobian_frobenius", 0.0),
        "hessian_max_eigenvalue": getattr(fr, "hessian_max_eigenvalue", 0.0),
        "motion_entropy": getattr(fr, "motion_entropy", 0.0),
        "retrieval_contributions": dict(getattr(node, "retrieval_contributions", {}) or {}),
        "retrieval_rank": None,
        "presentation_order": None,
    }


def assign_retrieval_rank(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stamp retrieval_rank = list position (frames must already be in
    relevance order). Returns the same list, mutated in place, for chaining."""
    for rank, frame in enumerate(frames):
        frame["retrieval_rank"] = rank
    return frames


def assign_presentation_order(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a NEW list sorted chronologically (by timestamp, then frame_idx)
    with presentation_order stamped -- does not mutate retrieval_rank, and
    does not reorder the input list, so callers that still need relevance
    order keep it."""
    ordered = sorted(frames, key=lambda f: (float(f["timestamp"]), int(f["frame_idx"])))
    out = [dict(f) for f in ordered]
    for order, frame in enumerate(out):
        frame["presentation_order"] = order
    return out
