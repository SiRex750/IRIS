"""Retrieval-side query reformulation helpers for video QA.

This module is deliberately deterministic.  It improves frame retrieval without
adding another model variable, which keeps NExT-QA experiments comparable while
the answer model remains fixed.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable


OPTION_LETTERS = "ABCDE"


@dataclass(frozen=True)
class QueryPlan:
    """Small, inspectable plan used by evaluators before frame retrieval."""

    original_query: str
    retrieval_queries: tuple[str, ...]
    family: str | None = None
    temporal_relation: str | None = None
    needs_temporal_expansion: bool = False
    notes: tuple[str, ...] = ()


def parse_mc_answer(text: str) -> int | None:
    """Parse only an explicit MC marker.

    This intentionally avoids regexes like ``\\b([A-E])\\b`` because those match
    ordinary English text such as "A boy..." and create a strong false-A bias.
    Returns the zero-based option index expected by NExT-QA, or None if the
    model did not follow the required format.
    """

    if not text:
        return None

    match = re.search(
        r"(?im)^\s*(?:FINAL\s+)?ANSWER\s*:\s*(?:OPTION\s*)?\(?([A-E])\)?\b",
        text,
    )
    if match:
        return OPTION_LETTERS.index(match.group(1).upper())
    return None


def format_mc_label(option_index: int | None) -> str:
    """Human-readable option label for logs."""

    if option_index is None:
        return "?"
    if 0 <= option_index < len(OPTION_LETTERS):
        return OPTION_LETTERS[option_index]
    return f"?{option_index}"


def reformulate_query(
    question: str,
    family: str | None = None,
    max_queries: int = 5,
) -> QueryPlan:
    """Produce visual-search query strings from a NExT-QA question.

    The generated strings are retrieval-only.  They should describe visible
    people, objects, actions, and temporal regions, but must not invent causes
    or answer content before evidence is retrieved.
    """

    original = _clean_spaces(question.strip())
    lower = original.lower()
    notes: list[str] = []

    temporal_relation = _detect_temporal_relation(lower)
    needs_temporal = family in {"C", "T"} or temporal_relation is not None
    if family in {"C", "T"}:
        notes.append("family_requires_temporal_context")
    if temporal_relation is not None:
        notes.append(f"temporal_relation:{temporal_relation}")

    queries: list[str] = [original]

    visual = _to_visual_description(lower)
    if visual:
        queries.append(visual)

    entity_action = _entity_action_phrase(visual or lower)
    if entity_action:
        queries.append(entity_action)

    temporal = _temporal_phrase(lower, temporal_relation)
    if temporal:
        queries.append(temporal)

    if family == "C" or lower.startswith("why "):
        cause_safe = _causal_visual_phrase(visual or lower)
        if cause_safe:
            queries.append(cause_safe)

    deduped = _dedupe_queries(queries)
    if max_queries > 0:
        deduped = deduped[:max_queries]

    return QueryPlan(
        original_query=original,
        retrieval_queries=tuple(deduped),
        family=family,
        temporal_relation=temporal_relation,
        needs_temporal_expansion=needs_temporal,
        notes=tuple(notes),
    )


def fuse_ranked_results(
    ranked_lists: Iterable[list[dict[str, Any]]],
    top_k: int,
    *,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse frame rankings from multiple reformulated queries.

    Uses reciprocal-rank fusion so a frame appearing in multiple query variants
    rises without requiring all score scales to be comparable.
    """

    if top_k <= 0:
        return []

    scores: dict[int, float] = {}
    hits: dict[int, int] = {}
    best_rank: dict[int, int] = {}
    records: dict[int, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, frame in enumerate(ranked):
            frame_idx = int(frame["frame_idx"])
            scores[frame_idx] = scores.get(frame_idx, 0.0) + 1.0 / (rrf_k + rank + 1)
            hits[frame_idx] = hits.get(frame_idx, 0) + 1
            best_rank[frame_idx] = min(best_rank.get(frame_idx, rank), rank)
            if frame_idx not in records:
                records[frame_idx] = dict(frame)

    ordered = sorted(
        records,
        key=lambda fi: (-scores[fi], best_rank[fi], fi),
    )[:top_k]

    fused: list[dict[str, Any]] = []
    for frame_idx in ordered:
        frame = dict(records[frame_idx])
        contributions = dict(frame.get("retrieval_contributions") or {})
        contributions["query_reformulation_rrf_score"] = scores[frame_idx]
        contributions["query_reformulation_hits"] = hits[frame_idx]
        contributions["query_reformulation_best_rank"] = best_rank[frame_idx]
        frame["retrieval_contributions"] = contributions
        fused.append(frame)
    return fused


def expand_temporal_neighbors(
    index: Any,
    retrieved_frames: list[dict[str, Any]],
    *,
    radius: int = 2,
    max_frames: int | None = None,
) -> list[dict[str, Any]]:
    """Add nearby indexed frames around each retrieved frame.

    The radius is measured in selected/indexed frames, not raw video frames.
    This keeps captioning cost bounded by the survivor set while giving causal
    and temporal questions before/after context.
    """

    if radius <= 0 or not retrieved_frames:
        return list(retrieved_frames[:max_frames] if max_frames else retrieved_frames)

    by_idx = {int(fr.frame_idx): fr for fr in index.frames}
    ordered = sorted(by_idx)
    positions = {frame_idx: pos for pos, frame_idx in enumerate(ordered)}

    original_records = {int(f["frame_idx"]): dict(f) for f in retrieved_frames}
    selected: dict[int, dict[str, Any]] = {}

    for seed_rank, frame in enumerate(retrieved_frames):
        seed_idx = int(frame["frame_idx"])
        if seed_idx not in positions:
            continue
        seed_pos = positions[seed_idx]
        start = max(0, seed_pos - radius)
        stop = min(len(ordered), seed_pos + radius + 1)

        for pos in range(start, stop):
            frame_idx = ordered[pos]
            if frame_idx in selected:
                continue
            if frame_idx in original_records:
                record = dict(original_records[frame_idx])
                contributions = dict(record.get("retrieval_contributions") or {})
                contributions["temporal_expansion"] = False
                contributions.setdefault("temporal_seed_rank", seed_rank)
                record["retrieval_contributions"] = contributions
            else:
                fr = by_idx[frame_idx]
                record = _frame_record_to_retrieved_dict(fr)
                record["retrieval_contributions"] = {
                    "temporal_expansion": True,
                    "temporal_source_frame_idx": seed_idx,
                    "temporal_seed_rank": seed_rank,
                    "temporal_distance_indexed_frames": abs(pos - seed_pos),
                }
            selected[frame_idx] = record

    expanded = [selected[frame_idx] for frame_idx in sorted(selected)]

    if max_frames is not None and len(expanded) > max_frames:
        expanded = _trim_temporal_context(expanded, retrieved_frames, max_frames)

    return expanded


def _frame_record_to_retrieved_dict(frame_record: Any) -> dict[str, Any]:
    return {
        "frame_idx": frame_record.frame_idx,
        "timestamp": frame_record.timestamp,
        "luma_diff_energy": frame_record.luma_diff_energy,
        "action_score": frame_record.action_score,
        "persistence_value": frame_record.persistence_value,
        "is_peak": frame_record.is_peak,
        "clip_embedding": frame_record.clip_embedding,
        "luma_entropy": frame_record.luma_entropy,
        "caption": frame_record.caption,
        "pagerank_score": frame_record.pagerank_score,
        "last_retrieval_score": 0.0,
        "retrieval_contributions": {},
    }


def _trim_temporal_context(
    expanded: list[dict[str, Any]],
    retrieved_frames: list[dict[str, Any]],
    max_frames: int,
) -> list[dict[str, Any]]:
    if max_frames <= 0:
        return []

    seed_order = {int(f["frame_idx"]): rank for rank, f in enumerate(retrieved_frames)}

    def priority(frame: dict[str, Any]) -> tuple[int, int, int]:
        frame_idx = int(frame["frame_idx"])
        if frame_idx in seed_order:
            return (0, seed_order[frame_idx], frame_idx)
        contrib = frame.get("retrieval_contributions") or {}
        return (
            1,
            int(contrib.get("temporal_distance_indexed_frames", 9999)),
            frame_idx,
        )

    kept = sorted(expanded, key=priority)[:max_frames]
    return sorted(kept, key=lambda frame: int(frame["frame_idx"]))


def _clean_spaces(text: str) -> str:
    return " ".join(text.split())


def _dedupe_queries(queries: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for query in queries:
        cleaned = _clean_spaces(query.strip(" ?."))
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _detect_temporal_relation(question: str) -> str | None:
    checks = [
        ("before", "before"),
        ("after", "after"),
        ("then", "sequence"),
        ("next", "sequence"),
        ("first", "beginning"),
        ("beginning", "beginning"),
        ("start", "beginning"),
        ("middle", "middle"),
        ("end", "end"),
        ("last", "end"),
    ]
    for needle, relation in checks:
        if re.search(rf"\b{re.escape(needle)}\b", question):
            return relation
    return None


def _to_visual_description(question: str) -> str:
    text = question
    replacements = [
        (r"\bwhy did\b", ""),
        (r"\bwhy does\b", ""),
        (r"\bwhat did\b", ""),
        (r"\bwhat does\b", ""),
        (r"\bwhat is\b", ""),
        (r"\bwhat are\b", ""),
        (r"\bwho is\b", ""),
        (r"\bwho are\b", ""),
        (r"\bhow did\b", ""),
        (r"\bwhere did\b", ""),
        (r"\bwhen did\b", ""),
        (r"\bin the video\b", ""),
        (r"\bin this video\b", ""),
        (r"\bat the beginning of the video\b", "beginning"),
        (r"\bin the beginning of the video\b", "beginning"),
        (r"\bin the middle of the video\b", "middle"),
        (r"\bat the end of the video\b", "end"),
        (r"\bin the end of the video\b", "end"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return _clean_spaces(text.strip(" ?."))


def _entity_action_phrase(text: str) -> str:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9']+", text.lower())
        if token not in _STOPWORDS
    ]
    if not tokens:
        return ""
    return " ".join(tokens[:8])


def _temporal_phrase(question: str, relation: str | None) -> str:
    if relation == "middle":
        return "middle video action"
    if relation == "beginning":
        return "beginning video action"
    if relation == "end":
        return "end video action"
    if relation == "before":
        return _clean_spaces(re.sub(r"\bbefore\b", "", question).strip(" ?."))
    if relation == "after":
        return _clean_spaces(re.sub(r"\bafter\b", "", question).strip(" ?."))
    if relation == "sequence":
        return "sequence of actions"
    return ""


def _causal_visual_phrase(text: str) -> str:
    phrase = _entity_action_phrase(text)
    if not phrase:
        return "person action context"
    return f"{phrase} surrounding context"


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "did",
    "do",
    "does",
    "for",
    "from",
    "happen",
    "happened",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "there",
    "this",
    "to",
    "video",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
