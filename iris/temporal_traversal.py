"""Direction-aware temporal graph traversal for Structured Query V2 (section 12).

Replaces the legacy expand_temporal_neighbors' symmetric,
indexed-frame-position radius with a relation-directed traversal over
timestamps and scene structure:

  AFTER   -- candidates strictly later than the anchor, same/following scenes.
  BEFORE  -- candidates strictly earlier than the anchor, same/preceding scenes.
  DURING / CURRENT -- a small symmetric time window, same scene preferred.
  CAUSE   -- keeps the observable-effect anchor, favors PRECEDING context,
             allows limited same-time context; never excludes the anchor.
  MANNER  -- favors the event itself and close local context.

Everything is measured in seconds (FrameRecord.timestamp) and scene_id, never
in indexed-frame ordinal position, and every traversal is bounded by both a
time window (config.temporal_context_seconds) and a scene-hop budget
(config.temporal_scene_hops) plus a hard frame-count budget
(config.max_context_frames) -- this is deliberately NOT the old fixed
"radius=2 selected frames" behavior.
"""
from __future__ import annotations

from typing import Any, Callable

from iris.frame_serialization import frame_record_to_dict
from iris.query_reformulation import Relation, QueryPlanV2


def _video_duration(index: Any) -> float:
    if not index.frames:
        return 0.0
    return max(float(fr.timestamp) for fr in index.frames)


def _scene_order(index: Any) -> dict[int, int]:
    """{scene_id: position} ordered by scene_spans' start time when available
    (real valley-boundary chronological order), else by scene_id itself."""
    scene_spans = getattr(index, "scene_spans", None) or {}
    if scene_spans:
        ordered = sorted(scene_spans.keys(), key=lambda sid: scene_spans[sid][0])
    else:
        ordered = sorted({getattr(fr, "scene_id", -1) for fr in index.frames})
    return {sid: pos for pos, sid in enumerate(ordered)}


def _select_anchor(retrieved_frames: list[dict], occurrence_selector: str | None) -> dict:
    """"first time X" -> earliest strong anchor occurrence; "last time X" ->
    latest. Otherwise the existing best-ranked (retrieval_rank 0) candidate."""
    if occurrence_selector == "first":
        return min(retrieved_frames, key=lambda f: (float(f["timestamp"]), int(f["frame_idx"])))
    if occurrence_selector == "last":
        return max(retrieved_frames, key=lambda f: (float(f["timestamp"]), int(f["frame_idx"])))
    return retrieved_frames[0]


def _scene_hops(scene_pos: dict[int, int], anchor_scene: int | None, candidate_scene: int | None) -> int:
    if anchor_scene is None or candidate_scene is None:
        return 0
    if anchor_scene not in scene_pos or candidate_scene not in scene_pos:
        return 0
    return abs(scene_pos[candidate_scene] - scene_pos[anchor_scene])


def _position_prior_bonus(position_prior: tuple[float, float] | None, timestamp: float, duration: float) -> float:
    if position_prior is None or duration <= 0:
        return 0.0
    lo, hi = position_prior
    frac = timestamp / duration
    if lo <= frac <= hi:
        return 1.0
    # Smooth falloff outside the window instead of a hard cliff.
    dist = min(abs(frac - lo), abs(frac - hi))
    return max(0.0, 1.0 - dist * 4.0)


def traverse_directional(
    index: Any,
    retrieved_frames: list[dict[str, Any]],
    plan: QueryPlanV2,
    config: Any,
    *,
    target_similarity_fn: "Callable[[int], float] | None" = None,
) -> list[dict[str, Any]]:
    """Direction-aware candidate expansion + rerank around the anchor.

    retrieved_frames: the anchor-retrieval result (already in relevance
    order; retrieval_rank already stamped by the caller).
    target_similarity_fn: optional callback frame_idx -> cosine similarity
    against the plan's target/entity embedding, used only for the final
    rerank blend (section 12's "target/entity similarity when available").
    """
    if not retrieved_frames:
        return retrieved_frames

    relation = plan.relation
    if relation == Relation.NONE:
        return retrieved_frames

    time_window = float(getattr(config, "temporal_context_seconds", 4.0))
    scene_hop_budget = int(getattr(config, "temporal_scene_hops", 1))
    frame_budget = int(getattr(config, "max_context_frames", 8))

    anchor = _select_anchor(retrieved_frames, plan.occurrence_selector)
    anchor_ts = float(anchor["timestamp"])
    anchor_scene = anchor.get("scene_id")
    anchor_idx = int(anchor["frame_idx"])

    scene_pos = _scene_order(index)
    duration = _video_duration(index)

    existing_by_idx = {int(f["frame_idx"]): dict(f) for f in retrieved_frames}
    candidates: dict[int, dict[str, Any]] = {}

    for fr in index.frames:
        if fr.clip_embedding is None:
            continue
        frame_idx = int(fr.frame_idx)
        dt = float(fr.timestamp) - anchor_ts
        candidate_scene = getattr(fr, "scene_id", None)
        hops = _scene_hops(scene_pos, anchor_scene, candidate_scene)

        include = False
        if frame_idx == anchor_idx:
            include = True
        elif relation == Relation.AFTER:
            include = 0 < dt <= time_window and hops <= scene_hop_budget
        elif relation == Relation.BEFORE:
            include = -time_window <= dt < 0 and hops <= scene_hop_budget
        elif relation in (Relation.DURING, Relation.CURRENT):
            half = time_window / 2.0
            include = abs(dt) <= half and hops <= scene_hop_budget
        elif relation == Relation.CAUSE:
            # Favor preceding context; allow limited same-time context; the
            # effect (anchor) frame itself is never excluded (handled above).
            include = (-time_window <= dt <= time_window * 0.25) and hops <= scene_hop_budget
        elif relation == Relation.MANNER:
            half = time_window / 2.0
            include = abs(dt) <= half and hops == 0
        elif relation == Relation.SEQUENCE:
            include = 0 <= dt <= time_window and hops <= scene_hop_budget
        else:
            include = False

        if not include:
            continue

        if frame_idx in existing_by_idx:
            record = existing_by_idx[frame_idx]
        else:
            record = frame_record_to_dict(fr)
        candidates[frame_idx] = record

    def _direction_score(record: dict[str, Any]) -> float:
        frame_idx = int(record["frame_idx"])
        if frame_idx == anchor_idx:
            return 1e6  # anchor always ranks first, never excluded by the budget trim
        dt = abs(float(record["timestamp"]) - anchor_ts)
        recency = max(0.0, 1.0 - dt / max(time_window, 1e-6))
        candidate_scene = record.get("scene_id")
        hops = _scene_hops(scene_pos, anchor_scene, candidate_scene)
        scene_continuity = 1.0 / (1.0 + hops)
        prior_bonus = _position_prior_bonus(plan.position_prior, float(record["timestamp"]), duration)
        action_score = float(record.get("action_score", 0.0))
        codec_conf = float(record.get("codec_conf", 0.5))
        target_sim = target_similarity_fn(frame_idx) if target_similarity_fn is not None else 0.0
        score = (
            0.35 * recency
            + 0.20 * scene_continuity
            + 0.15 * prior_bonus
            + 0.10 * action_score
            + 0.05 * codec_conf
            + 0.15 * target_sim
        )
        return score

    ordered = sorted(candidates.values(), key=lambda r: (-_direction_score(r), int(r["frame_idx"])))
    trimmed = ordered[:frame_budget] if frame_budget > 0 else ordered

    for record in trimmed:
        contributions = dict(record.get("retrieval_contributions") or {})
        contributions.update({
            "temporal_relation": relation,
            "temporal_direction": plan.temporal_direction,
            "dt_from_anchor_seconds": float(record["timestamp"]) - anchor_ts,
            "scene_hops_from_anchor": _scene_hops(scene_pos, anchor_scene, record.get("scene_id")),
            "is_temporal_anchor": int(record["frame_idx"]) == anchor_idx,
            "position_prior_bonus": _position_prior_bonus(plan.position_prior, float(record["timestamp"]), duration),
        })
        record["retrieval_contributions"] = contributions

    from iris.frame_serialization import assign_retrieval_rank
    return assign_retrieval_rank(trimmed)
