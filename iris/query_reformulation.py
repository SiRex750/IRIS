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


# ═══════════════════════════════════════════════════════════════════════════
# Structured Query Reformulation V2
#
# Everything below is additive and does not change any function above this
# banner.  reformulate_query/fuse_ranked_results/expand_temporal_neighbors
# remain the "legacy" reformulator, selectable via
# IRISConfig.query_reformulation_mode="legacy" and still covered by the
# original tests.  V2 is a structured, type-code-aware planner selected via
# query_reformulation_mode="structured_v2" (see iris.retrieval_entry).
#
# Known problems in the legacy reformulator this section fixes:
#   - every C/T question got the same handful of canned prompts regardless of
#     which relation (before/after/during/cause/manner) actually applied;
#   - "while"/"when"/"as" were never detected, so ~61/269 temporal questions
#     silently got no relation at all;
#   - the eight-token stopword-filtered phrase truncated long visual
#     descriptions and deleted words that carry real visual meaning
#     (on/to/from/with, phrasal verbs, spatial prepositions);
#   - "middle video action"/"beginning video action"/"end video action"/
#     "sequence of actions"/"person action context" are vague CLIP prompts,
#     not retrieval operations -- beginning/middle/end must be a numeric
#     prior over timestamps, not text handed to CLIP;
#   - "first time X"/"last time X" were folded into the same bucket as literal
#     beginning/end-of-video, which they are not;
#   - temporal neighbor expansion was symmetric and measured in indexed-frame
#     positions, never in seconds or direction-aware.
# ═══════════════════════════════════════════════════════════════════════════


class Relation:
    """Validated relation vocabulary for QueryPlanV2.  Plain string constants
    (not enum.Enum) so relation values serialize trivially into telemetry/
    JSON traces without a custom encoder."""

    BEFORE = "BEFORE"
    AFTER = "AFTER"
    DURING = "DURING"
    CAUSE = "CAUSE"
    MANNER = "MANNER"
    CURRENT = "CURRENT"
    SEQUENCE = "SEQUENCE"
    NONE = "NONE"

    ALL = (BEFORE, AFTER, DURING, CAUSE, MANNER, CURRENT, SEQUENCE, NONE)


@dataclass(frozen=True)
class QueryPlanV2:
    """Structured, inspectable retrieval plan produced by build_query_plan_v2.

    Backward compatible by addition: this is a separate dataclass from the
    legacy boolean QueryPlan, not a replacement of it, so any code holding a
    QueryPlan keeps working unchanged.
    """

    original_query: str
    normalized_query: str
    type_code: str | None = None
    family: str | None = None
    relation: str = Relation.NONE
    relation_source: str = "fallback"  # "type_code" | "lexical" | "fallback" | "conflict"
    temporal_direction: int | str = 0  # -1, 0, +1, or "bidirectional"
    anchor_queries: tuple[str, ...] = ()
    target_query: str | None = None
    position_prior: tuple[float, float] | None = None  # (lo, hi) as a fraction of video duration
    occurrence_selector: str | None = None  # "first" | "last" | None
    parser_confidence: float = 0.0
    needs_temporal_traversal: bool = False
    fallback_reason: str | None = None
    corrections_applied: tuple[str, ...] = ()
    aliases_applied: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    query_roles: tuple[str, ...] = ()  # parallel to anchor_queries (+ "target" if target_query is set)
    query_weights: tuple[float, ...] = ()  # parallel to query_roles


TYPE_CODE_RELATION: dict[str, str] = {
    "TN": Relation.AFTER,
    "TP": Relation.BEFORE,
    "TC": Relation.DURING,
    "CW": Relation.CAUSE,
    "CH": Relation.MANNER,
}

RELATION_DIRECTION: dict[str, int | str] = {
    Relation.AFTER: 1,
    Relation.BEFORE: -1,
    Relation.DURING: 0,
    Relation.CAUSE: -1,     # favor preceding context, effect frame not excluded
    Relation.MANNER: 0,     # event-local context
    Relation.CURRENT: 0,
    Relation.SEQUENCE: 1,
    Relation.NONE: 0,
}

# Longest phrases first so "just before"/"immediately after" win over the
# bare "before"/"after" they contain.
_TEMPORAL_PHRASE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bimmediately after\b", Relation.AFTER),
    (r"\bjust before\b", Relation.BEFORE),
    (r"\bprior to\b", Relation.BEFORE),
    (r"\bearlier than\b", Relation.BEFORE),
    (r"\bat the same time\b", Relation.DURING),
    (r"\bsimultaneously\b", Relation.DURING),
    (r"\buntil\b", Relation.BEFORE),
    (r"\bbefore\b", Relation.BEFORE),
    (r"\bafter\b", Relation.AFTER),
    (r"\bfollowing\b", Relation.AFTER),
    (r"\bonce\b", Relation.AFTER),
    (r"\blater\b", Relation.AFTER),
    (r"\bnext\b", Relation.AFTER),
    (r"\bsubsequently\b", Relation.AFTER),
    (r"\bduring\b", Relation.DURING),
    (r"\bwhile\b", Relation.DURING),
    (r"\bwhen\b", Relation.DURING),
    (r"\bas\b", Relation.DURING),
    (r"\bthen\b", Relation.SEQUENCE),
)

# Explicit video-position phrases ONLY -- bare words like "start"/"end"/
# "first"/"last" are deliberately NOT matched here (see docstring below).
_POSITION_PHRASE_PATTERNS: tuple[tuple[str, tuple[float, float]], ...] = (
    (r"\bat the beginning of the video\b", (0.0, 0.15)),
    (r"\bat the start of the video\b", (0.0, 0.15)),
    (r"\bin the beginning of the video\b", (0.0, 0.15)),
    (r"\bin the first part\b", (0.0, 0.15)),
    (r"\bin the middle of the video\b", (0.4, 0.6)),
    (r"\bnear the end\b", (0.8, 1.0)),
    (r"\bat the end of the video\b", (0.85, 1.0)),
    (r"\bin the end of the video\b", (0.85, 1.0)),
    (r"\bin the last part\b", (0.85, 1.0)),
)

_OCCURRENCE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bthe first time\b", "first"),
    (r"\bthe last time\b", "last"),
)

_QUESTION_SCAFFOLD_PATTERNS: tuple[str, ...] = (
    r"\bwhat does\b",
    r"\bwhat did\b",
    r"\bwhy did\b",
    r"\bwhy does\b",
    r"\bhow did\b",
    r"\bhow does\b",
    r"\bwho is\b",
    r"\bwho are\b",
    r"\bwhere did\b",
    r"\bwhen did\b",
    r"\bin the video\b",
    r"\bin this video\b",
)

_CONNECTOR_RELATION: dict[str, str] = {
    "before": Relation.BEFORE,
    "after": Relation.AFTER,
    "while": Relation.DURING,
    "when": Relation.DURING,
    "as": Relation.DURING,
    "during": Relation.DURING,
}

# Conservative, training-derived typo/grammar corrections. Every application
# is recorded in QueryPlanV2.corrections_applied -- never applied silently.
# Deliberately small: object names, colors, and identities are never touched.
TYPO_MAP: dict[str, str] = {
    "wipping": "wiping",
    "shaked": "shook",
    "runs pass": "runs past",
}

# Small, conservative, optional visual-action alias map (section 9). Applying
# an alias never removes the original phrase -- it only adds one alternate
# prompt (see _apply_action_alias). Empty/short by design; grow only via
# retrieval ablation on train/val_tune, never on val_confirm.
ACTION_ALIASES: dict[str, str] = {
    "pick up": "lift",
    "picks up": "lifts",
    "picking up": "lifting",
    "put down": "place",
    "puts down": "places",
    "putting down": "placing",
    "walk away": "leave",
    "walks away": "leaves",
    "walking away": "leaving",
    "turn around": "turn body",
    "turns around": "turns body",
    "turning around": "turning body",
}

# Multi-word phrases that must never be split apart by clause-aware
# normalization -- phrasal verbs, spatial relations, source/destination
# relations, interaction words. Longest-first is not required here since
# these are only used for containment checks, not regex alternation order.
_PRESERVED_PHRASES: tuple[str, ...] = (
    "pick up", "put down", "put on", "take off", "sit down", "stand up",
    "turn around", "walk away", "get up", "get off", "get on",
)


def _normalize_unicode_punctuation(text: str) -> str:
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _fix_separated_possessive(text: str) -> str:
    """'baby s mouth' -> "baby's mouth" -- only for a lone ' s ' token that is
    not itself a real word (as/is/us/gas/bus/...), so this never touches
    ordinary text."""
    _SAFE_SHORT_WORDS = {"as", "is", "us", "gas", "bus", "yes", "his"}

    def _repl(m: re.Match) -> str:
        word = m.group(1)
        if word.lower() in _SAFE_SHORT_WORDS:
            return m.group(0)
        return f"{word}'s"

    return re.sub(r"\b(\w+) s\b", _repl, text)


def normalize_query_text(question: str) -> tuple[str, list[str]]:
    """Conservative, deterministic normalization (section 8).

    Returns (normalized_text, corrections_applied). Every correction is
    recorded; nothing here touches object names, colors, or identities.
    """
    corrections: list[str] = []
    text = _normalize_unicode_punctuation(question)
    text = _clean_spaces(text)

    deduped = re.sub(r"\b(\w+)( \1\b)+", r"\1", text, flags=re.IGNORECASE)
    if deduped != text:
        corrections.append("duplicate_determiner_removed")
        text = deduped

    fixed_possessive = _fix_separated_possessive(text)
    if fixed_possessive != text:
        corrections.append("separated_possessive_fixed")
        text = fixed_possessive

    lower = text.lower()
    for typo, fix in TYPO_MAP.items():
        pattern = rf"\b{re.escape(typo)}\b"
        if re.search(pattern, lower):
            text = re.sub(pattern, fix, text, flags=re.IGNORECASE)
            corrections.append(f"typo:{typo}->{fix}")
            lower = text.lower()

    return _clean_spaces(text), corrections


def _all_temporal_relations(lower_text: str) -> list[str]:
    """All distinct relations whose phrase appears in the text, in the
    priority order they would be selected -- used both to pick the primary
    lexical relation and to detect/report multi-relation-group conflicts."""
    found: list[str] = []
    for pattern, relation in _TEMPORAL_PHRASE_PATTERNS:
        if re.search(pattern, lower_text) and relation not in found:
            found.append(relation)
    return found


def _detect_position_prior(lower_text: str) -> tuple[tuple[float, float] | None, str | None]:
    """Explicit beginning/middle/end video-position phrases ONLY.

    Deliberately does NOT match bare "start"/"end"/"first"/"last" -- e.g.
    "the bird starts shaking", "the children start to jump", and "end up"
    must not be treated as a beginning/end-of-video prior; those are ordinary
    verbs, not a video-position claim. "The first/last time X" is handled
    separately by _detect_occurrence_selector, not here.
    """
    for pattern, prior in _POSITION_PHRASE_PATTERNS:
        if re.search(pattern, lower_text):
            return prior, f"position_prior:{pattern}"
    return None, None


def _detect_occurrence_selector(lower_text: str) -> str | None:
    for pattern, selector in _OCCURRENCE_PATTERNS:
        if re.search(pattern, lower_text):
            return selector
    return None


def _strip_question_scaffolding(text: str) -> str:
    for pattern in _QUESTION_SCAFFOLD_PATTERNS:
        text = re.sub(pattern, "", text)
    return _clean_spaces(text.strip(" ?."))


_CLAUSE_PATTERN = re.compile(
    r"^(?:what does|what did|why did|why does|how did|how does)\s+"
    r"(?:the |a |an )?(?P<subject>[a-z][a-z0-9' ]*?)\s+do\b\s*"
    r"(?P<connector>before|after|while|when|as|during)\s+"
    r"(?P<clause>.+)$"
)


def _extract_anchor_target(lower_text: str) -> dict[str, Any]:
    """Deterministic clause extraction (section 5). Never invents an
    unobserved answer action, cause, or object -- every string returned here
    is built only from tokens already present in the question.
    """
    stripped = lower_text.strip(" ?.")

    match = _CLAUSE_PATTERN.match(stripped)
    if match:
        subject = _clean_spaces(match.group("subject"))
        connector = match.group("connector")
        clause = _clean_spaces(match.group("clause").strip(" ?."))
        relation = _CONNECTOR_RELATION[connector]

        if relation == Relation.DURING and clause and not clause.startswith(subject):
            # "while the lady sings" -- clause carries its own subject.
            anchor_event = clause
            target_entity = _clean_spaces(f"{subject} near {clause}")
        else:
            # "before wiping the baby's mouth" -- same subject as the anchor.
            anchor_event = _clean_spaces(f"{subject} {clause}")
            target_entity = subject

        return {
            "anchor_event": anchor_event,
            "target_entity": target_entity,
            "relation": relation,
            "confidence": 0.85,
            "fallback_reason": None,
        }

    if re.match(r"^why (did|does)\b", stripped):
        anchor_event = _strip_question_scaffolding(stripped)
        return {
            "anchor_event": anchor_event,
            "target_entity": None,
            "relation": Relation.CAUSE,
            "confidence": 0.7,
            "fallback_reason": None,
        }

    if re.match(r"^how (did|does)\b", stripped):
        anchor_event = _strip_question_scaffolding(stripped)
        return {
            "anchor_event": anchor_event,
            "target_entity": None,
            "relation": Relation.MANNER,
            "confidence": 0.7,
            "fallback_reason": None,
        }

    anchor_event = _strip_question_scaffolding(stripped)
    return {
        "anchor_event": anchor_event or stripped,
        "target_entity": None,
        "relation": Relation.NONE,
        "confidence": 0.3,
        "fallback_reason": "no_clause_pattern_matched",
    }


def _apply_action_alias(phrase: str, enabled: bool) -> tuple[str | None, list[str]]:
    if not enabled or not phrase:
        return None, []
    for original, alias in ACTION_ALIASES.items():
        if original in phrase:
            return phrase.replace(original, alias, 1), [f"{original}->{alias}"]
    return None, []


def _to_clip_prompt(phrase: str | None) -> str | None:
    if not phrase:
        return None
    phrase = _clean_spaces(phrase.strip(" ?."))
    if not phrase:
        return None
    return f"a video frame showing {phrase}"


def build_query_plan_v2(
    question: str,
    *,
    type_code: str | None = None,
    family: str | None = None,
    config: Any = None,
) -> QueryPlanV2:
    """Build a structured retrieval plan (section 3/4/5).

    Type code (NExT-QA TN/TP/TC/CW/CH) is the primary source of the temporal
    operator when available (section 4). Lexical cues refine/add position or
    occurrence information and are recorded even when the type code wins, so
    a type/lexical conflict is never silently discarded (relation_source
    becomes "conflict" and the conflict is logged in notes).
    """
    action_aliases_enabled = getattr(config, "action_aliases_enabled", True)
    typo_normalization_enabled = getattr(config, "typo_normalization_enabled", True)

    original = question.strip()
    if typo_normalization_enabled:
        normalized, corrections = normalize_query_text(original)
    else:
        normalized, corrections = _clean_spaces(_normalize_unicode_punctuation(original)), []

    lower = normalized.lower()
    notes: list[str] = []

    lexical_relations = _all_temporal_relations(lower)
    lexical_relation = lexical_relations[0] if lexical_relations else None
    type_relation = TYPE_CODE_RELATION.get((type_code or "").upper())

    if type_relation is not None:
        relation = type_relation
        relation_source = "type_code"
        if lexical_relation is not None and lexical_relation != type_relation:
            relation_source = "conflict"
            notes.append(
                f"type_lexical_conflict:type_code={type_code}:{type_relation},"
                f"lexical={lexical_relation}"
            )
    elif lexical_relation is not None:
        relation = lexical_relation
        relation_source = "lexical"
    else:
        relation = Relation.NONE
        relation_source = "fallback"

    if len(lexical_relations) > 1:
        notes.append(f"multiple_lexical_relations:{lexical_relations}")

    anchor_info = _extract_anchor_target(lower)

    if relation == Relation.NONE and anchor_info["relation"] != Relation.NONE:
        relation = anchor_info["relation"]
        relation_source = "lexical" if relation_source == "fallback" else relation_source

    direction = RELATION_DIRECTION.get(relation, 0)

    position_prior, position_note = _detect_position_prior(lower)
    if position_note:
        notes.append(position_note)
    occurrence = _detect_occurrence_selector(lower)

    target_query = _to_clip_prompt(anchor_info.get("target_entity"))
    total_budget = 3
    anchor_budget = total_budget - (1 if target_query else 0)

    anchor_primary = _to_clip_prompt(anchor_info["anchor_event"])
    anchor_queries: list[str] = [anchor_primary] if anchor_primary else []
    roles: list[str] = ["anchor_primary"] if anchor_primary else []
    weights: list[float] = [1.0] if anchor_primary else []

    aliases_applied: list[str] = []
    if len(anchor_queries) < anchor_budget:
        alias_phrase, aliases_applied = _apply_action_alias(
            anchor_info["anchor_event"], action_aliases_enabled
        )
        alias_prompt = _to_clip_prompt(alias_phrase)
        if alias_prompt and alias_prompt not in anchor_queries:
            anchor_queries.append(alias_prompt)
            roles.append("anchor_alternate")
            weights.append(0.6)

    fallback_reason = anchor_info["fallback_reason"]
    if anchor_info["confidence"] < 0.5 and len(anchor_queries) < anchor_budget:
        original_as_prompt = _clean_spaces(normalized.strip(" ?."))
        if original_as_prompt and original_as_prompt not in anchor_queries:
            anchor_queries.append(original_as_prompt)
            roles.append("fallback_original")
            weights.append(0.4)

    if not anchor_queries:
        # Nothing parsed at all -- always safe to fall back to the raw
        # question rather than embedding nothing.
        anchor_queries = [_clean_spaces(normalized.strip(" ?."))]
        roles = ["fallback_original"]
        weights = [1.0]
        fallback_reason = fallback_reason or "empty_parse"

    anchor_queries = anchor_queries[:anchor_budget] if anchor_budget > 0 else anchor_queries[:1]
    roles = roles[: len(anchor_queries)]
    weights = weights[: len(anchor_queries)]

    if target_query:
        roles = roles + ["target"]
        weights = weights + [0.5]

    needs_traversal = relation != Relation.NONE

    return QueryPlanV2(
        original_query=original,
        normalized_query=normalized,
        type_code=type_code,
        family=family,
        relation=relation,
        relation_source=relation_source,
        temporal_direction=direction,
        anchor_queries=tuple(anchor_queries),
        target_query=target_query,
        position_prior=position_prior,
        occurrence_selector=occurrence,
        parser_confidence=anchor_info["confidence"],
        needs_temporal_traversal=needs_traversal,
        fallback_reason=fallback_reason,
        corrections_applied=tuple(corrections),
        aliases_applied=tuple(aliases_applied),
        notes=tuple(notes),
        query_roles=tuple(roles),
        query_weights=tuple(weights),
    )


def all_embedding_texts(plan: QueryPlanV2) -> tuple[str, ...]:
    """All distinct prompt strings a caller must embed for this plan, in
    query_roles order -- guaranteed len <= 3 (implementation constraint)."""
    texts = list(plan.anchor_queries)
    if plan.target_query and plan.target_query not in texts:
        texts.append(plan.target_query)
    return tuple(texts[:3])
