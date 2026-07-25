# Structured Query Reformulation V2

Status: implemented, unit-tested, **not yet run against val_tune** (no NExT-QA
video cache / index cache is present in this checkout -- see "What could not
be run" below). Not approved for val_confirm.

## 1. Legacy behavior (unchanged, still available)

`iris/query_reformulation.py`'s original `reformulate_query` /
`fuse_ranked_results` / `expand_temporal_neighbors` are untouched and remain
selectable via `IRISConfig.query_reformulation_mode="legacy"`. Known
limitations that motivated V2 (measured over the 639-question held-out set
referenced in this task):

- ~3.97 full retrieval queries per question, each running its own scene
  search + PPR/graph retrieval -- no batching, no shared PPR call.
- 525/639 questions had near-duplicate variants (token Jaccard >= 0.75) voted
  as if independent in RRF.
- 61/269 temporal questions got no detected relation ("while"/"when"/"as"
  were never matched).
- 100 questions contain multiple relation groups; the first lexical match won
  with no record of the ones it discarded.
- 164 questions collapsed to one of four canned CLIP prompts ("middle video
  action", "beginning video action", "end video action", "sequence of
  actions") regardless of their actual content.
- The 8-token stopword-filtered phrase dropped visually meaningful words
  (on/to/from/with) and truncated 372 long visual descriptions.
- Temporal neighbor expansion was symmetric and measured in indexed-frame
  positions, not seconds or direction.
- The `_embed_query` tuple-return contract was violated at three call sites
  (`scripts/nextqa_single_video_eval.py`, `eval/grounding_scorer.py`,
  `eval/mc_scorer.py`, `scripts/eval_grounding_arms.py`) -- all fixed in this
  change (see "Correctness bugs fixed" in the top-level report).

## 2. V2 architecture

```
retrieve_for_question(question, index, config, type_code=None, family=None, trace=None)
        │  iris/retrieval_entry.py -- the ONE canonical entry point
        ▼
  mode = config.query_reformulation_mode  ("none" | "legacy" | "structured_v2")
        │
        ├─ none ──────► iris.query._call_embed_query + _build_retrieved (frozen baseline, byte-identical)
        ├─ legacy ────► iris.query_reformulation.reformulate_query/fuse_ranked_results (+ legacy_symmetric expansion)
        └─ structured_v2:
             1. build_query_plan_v2()               -> QueryPlanV2           (iris/query_reformulation.py)
             2. all_embedding_texts(plan)  (<=3 strings)
             3. iris.query._call_embed_queries()     -> [Q, D] float32 matrix, L2-cache   (iris/query.py)
             4. _multi_query_retrieve()              -> ONE PPR / scene-sparse call        (iris/l2_asphodel.py, iris/scene_retrieval.py)
             5. traverse_directional()  (if needed)  -> direction-aware context expansion   (iris/temporal_traversal.py)
             6. canonical frame dicts throughout                                            (iris/frame_serialization.py)
```

Every frame dict, from any of the three modes, has the same schema (see
section 13 below) -- there is exactly one serializer.

## 3. QueryPlanV2 schema

```python
@dataclass(frozen=True)
class QueryPlanV2:
    original_query: str
    normalized_query: str
    type_code: str | None
    family: str | None
    relation: str                 # Relation.{BEFORE,AFTER,DURING,CAUSE,MANNER,CURRENT,SEQUENCE,NONE}
    relation_source: str          # "type_code" | "lexical" | "fallback" | "conflict"
    temporal_direction: int|str   # -1, 0, +1, or "bidirectional"
    anchor_queries: tuple[str, ...]   # <= 3 total together with target_query
    target_query: str | None
    position_prior: tuple[float, float] | None   # (lo, hi) fraction of video duration
    occurrence_selector: str | None               # "first" | "last" | None
    parser_confidence: float
    needs_temporal_traversal: bool
    fallback_reason: str | None
    corrections_applied: tuple[str, ...]
    aliases_applied: tuple[str, ...]
    notes: tuple[str, ...]
    query_roles: tuple[str, ...]      # parallel to anchor_queries (+"target")
    query_weights: tuple[float, ...]  # parallel to query_roles
```

`Relation` is a plain string-constant class (not `enum.Enum`) so it
serializes into JSON telemetry with no custom encoder.

## 4. Type-code mapping

| NExT-QA type code | Relation | temporal_direction |
|---|---|---|
| TN | AFTER  | +1 |
| TP | BEFORE | -1 |
| TC | DURING | 0  |
| CW | CAUSE  | -1 (favor preceding context) |
| CH | MANNER | 0  (event-local context) |
| D* (descriptive) | NONE | 0 (semantic retrieval, no forced traversal) |

Type code is the primary operator when available (`relation_source ==
"type_code"`). Lexical cues still run and are recorded: if they agree,
nothing changes; if they disagree, `relation_source` becomes `"conflict"`,
the type-code relation still wins, and the lexical relation is logged in
`plan.notes` (e.g. `type_lexical_conflict:type_code=TN:AFTER,lexical=BEFORE`)
-- never silently dropped. Without a type code, the first lexical relation
(priority-ordered, longest phrase first) is used; with neither, relation is
`NONE` and `needs_temporal_traversal` is `False`.

## 5. Clause extraction rules

`_extract_anchor_target` matches the pattern
`(what does|what did|why did|why does|how did|how does) <subject> do
(before|after|while|when|as|during) <clause>` deterministically:

- BEFORE/AFTER: anchor = `"<subject> <clause>"` (clause completes the same
  subject's action, e.g. "lady wiping the baby's mouth"); target = subject.
- DURING with a clause that has its own subject (e.g. "while the lady
  sings"): anchor = clause verbatim; target = `"<subject> near <clause>"`.
- `why did/does ...` with no clause match: relation = CAUSE, anchor = the
  question with only its scaffolding stripped (the observable effect).
- `how did/does ...` with no clause match: relation = MANNER, same anchor
  construction.
- Nothing matches: `relation_source="fallback"`, `parser_confidence=0.3`,
  `fallback_reason="no_clause_pattern_matched"`, and the raw (scaffold-
  stripped) question is used as the anchor -- never an invented action.

Every token in every generated prompt is drawn from the original question;
`test_no_invented_content_anchor_is_subset_of_question_tokens` enforces this.

## 6. Temporal vocabulary

Longest-phrase-first matching (so "just before"/"immediately after" win over
bare "before"/"after"):

- BEFORE: before, prior to, earlier than, until, just before
- AFTER: after, following, once, later, next, subsequently, immediately after
- DURING: while, when, as, during, at the same time, simultaneously
- SEQUENCE: then

Explicit video-position phrases (**only** full phrases, never bare words) map
to a numeric `(lo, hi)` fraction-of-duration prior: "at the beginning/start of
the video" -> `(0.0, 0.15)`, "in the middle of the video" -> `(0.4, 0.6)`,
"at/near the end of the video" -> `(0.85, 1.0)`. Bare "start"/"end"/"first"/
"last" are deliberately never matched here -- "the bird starts shaking",
"the children start to jump", and "end up" produce `position_prior = None`.
"The first/last time X" is a separate `occurrence_selector` ("first"/"last"),
resolved against **anchor occurrences**, not video position.

## 7/8/9. Normalization and alias policy

- Question scaffolding ("what does"/"why did"/"in the video"/...) is stripped
  only after clause parsing, never via a fixed-length token cutoff.
- `_PRESERVED_PHRASES` and the clause-based extraction never split phrasal
  verbs, spatial prepositions (behind/under/toward/through), or
  source/destination relations (to/from/into) apart -- there is no bag-of-
  tokens stopword filter in the V2 path at all.
- `normalize_query_text()` (section 8): unicode punctuation, duplicate
  determiners ("the the"), a small conservative `TYPO_MAP` seeded from
  training-observed cases (wipping->wiping, shaked->shook, "runs pass"->"runs
  past"), and a narrow separated-possessive fix. Every correction is recorded
  in `corrections_applied`; disable via `config.typo_normalization_enabled`.
- `ACTION_ALIASES` (section 9) is a small, hand-reviewed table (pick up/put
  down/walk away/turn around and their inflections). Applying an alias never
  removes the original phrase -- it only *adds* one alternate prompt, and the
  applied mapping is recorded in `aliases_applied`. Disable via
  `config.action_aliases_enabled`. No WordNet, no LLM.

## 10. Batch text embedding + cache

`iris.query._embed_queries(texts, config)` tokenizes every miss in one
`clip.tokenize`/`model.encode_text` call, L2-normalizes, and returns a
`[Q, D]` float32 matrix plus telemetry (`cache_hits`, `cache_misses`,
`num_clip_batches`, per-query norms). A bounded LRU
(`config.query_embedding_cache_size`, default 512) is keyed by
`(normalized_text, clip_revision, device)` -- a revision change is always a
cache miss. `_embed_query`/`_call_embed_query` are untouched;
`_embed_queries([text])` performs the byte-identical tokenize/encode/
normalize sequence as `_embed_query(text)` (see
`test_batch_embed_matches_single_query_embed`).

## 11. Multi-query scoring

`L2Asphodel.retrieve_ppr` and `scene_retrieval.retrieve_scene_sparse` both
gained optional `query_embeddings`/`query_weights`/`multi_query_combine`
parameters. The default combine is **weighted max**: per node,
`max_j(weight_j * relu(cosine(query_j, node)))` -- never an average of the
query vectors (that would erase a specialized anchor/target prompt's
meaning). `multi_query_combine="logsumexp"` is available as a smoother
alternative behind the same config flag. Exactly one PPR solve happens
either way; multi-query only changes how the PPR seed's semantic term is
computed. At `Q=1, weight=1.0` both functions are provably identical to the
pre-existing single-query path (`test_multi_query_ppr_matches_single_query_
within_tolerance`, `test_scene_sparse_multi_query_equivalence_at_q1`).

## 12. Direction-aware traversal

`iris/temporal_traversal.py::traverse_directional` replaces the legacy
symmetric survivor-index radius. It operates on `FrameRecord.timestamp`
(seconds) and `scene_id`/`index.scene_spans` (real valley-boundary order),
bounded by `config.temporal_context_seconds` (time window) AND
`config.temporal_scene_hops` (adjacent-scene budget), with a hard
`config.max_context_frames` cap:

- AFTER: `0 < dt <= window`, scene hop `<= budget`.
- BEFORE: `-window <= dt < 0`, scene hop `<= budget`.
- DURING/CURRENT: `|dt| <= window/2`, scene hop `<= budget`.
- CAUSE: `-window <= dt <= 0.25*window` (favors preceding context, allows
  limited same-time context); the effect/anchor frame is never excluded.
- MANNER: `|dt| <= window/2`, restricted to the anchor's own scene (hop 0).

The anchor itself is chosen via `plan.occurrence_selector`: `"first"`/`"last"`
pick the earliest/latest matching candidate by timestamp; otherwise the
existing best-ranked (retrieval_rank 0) candidate is used. After candidate
selection, a single deterministic rerank score blends recency, scene
continuity, the numeric position prior, action score, codec confidence, and
(when available) target-entity cosine similarity -- logged per-candidate in
`retrieval_contributions`.

## 13. Frame schema / telemetry

`iris/frame_serialization.py` is the single serializer
(`frame_record_to_dict`, `node_to_dict`) used by `iris.query._build_retrieved`,
`iris.scene_retrieval`, and `iris.temporal_traversal` -- previously these were
three independently-drifting copies (the `query_reformulation.py` copy
silently dropped `scene_id`/`tier`/`pict_type`/`codec_conf`/motion-geometry
fields). `retrieval_rank` (relevance order) and `presentation_order`
(chronological order for the captioner) are separate fields;
`assign_presentation_order` never mutates `retrieval_rank`.

`retrieve_for_question`'s telemetry dict (populated into `trace` when
`config.query_trace_enabled=True`) includes the full plan, embedding
cache hit/miss counts, `num_ppr_calls` (always 1 in structured_v2),
`scene_boundaries_crossed`, `context_frame_count`, and per-stage timings
(`parse_sec`/`embed_sec`/`retrieve_sec`/`traverse_sec`). Telemetry collection
never changes retrieval behavior.

## 14. Fallback behavior

- Clause parser confidence `< 0.5` -> the raw (normalized) question is
  appended as a lower-priority `"fallback_original"` prompt, within the
  3-embedding budget; `fallback_reason` is always set.
- Zero anchors parsed at all -> the raw question becomes the sole anchor
  (`fallback_reason="empty_parse"`).
- `ranking_mode="legacy"` (alpha/beta/gamma/delta blend) has no multi-query
  form; `_multi_query_retrieve` falls back to the single highest-weighted
  query embedding rather than averaging vectors, and this is visible in
  telemetry as a `ranking_mode`-driven code path, not a silent behavior
  change.

## 15. Edge-device considerations

- <= 3 CLIP text embeddings per question, batched into a single tokenize/
  encode call.
- Exactly one PPR (or scene-sparse) call per question in the normal path.
- `max_context_frames` defaults to 8 (was 24 in the old debug evaluator
  default).
- No LLM calls, no heavy NLP dependencies (regex + small dicts only),
  deterministic given the same question/type_code/config.
- The bounded embedding cache (`query_embedding_cache_size`, default 512
  entries of `[embed_dim]` float32 -- ~1MB at dim=512) is the only added
  memory cost.

## 16. Evaluation arms and fairness treatment

Six arms are defined in the task (A: raw; B: legacy no expansion; C: legacy +
symmetric expansion; D: structured_v2 no traversal; E: structured_v2 +
directional traversal; F: E with batched embeddings/one PPR, which is what
this implementation always does when structured_v2 is selected -- D/E/F
collapse to "structured_v2 with `temporal_traversal_mode` = none/directional/
directional", since batched embeddings and single-PPR are not optional in
this codebase's structured_v2 path). Every arm reads the exact same
`dataset_manifest.json`-registered question text and NExT-QA type code;
`retrieve_for_question`'s `family`/`type_code` inputs come only from the
dataset row, never from answer choices.

Per `benchmark_runs/paper_setup_20260720T074844Z_1e431b7/fairness_contract.md`
rule 3 ("same query text... no per-method reformulation variance"): IRIS with
Structured Query Planner V2 must be reported as **"IRIS + Structured Query
Planner V2"**, a clearly separated ablation, not folded into the existing
"fair comparison" tables that assume verbatim query text for every method.
The fairness contract itself is unmodified by this change.

## 17. Known limitations

- The clause-extraction regex covers the "what/why/how ... do
  before/after/while/when/as/during ..." family explicitly; other phrasings
  (e.g. relative clauses, multi-clause questions) fall through to the
  `fallback_reason="no_clause_pattern_matched"` path with lower confidence
  rather than a tailored parse.
- `multi_query_combine="logsumexp"` is implemented but not separately
  ablated against `"weighted_max"` in this change (no val_tune data
  available in this environment -- see the top-level report).
- `ranking_mode="legacy"` does not get true multi-query scoring (see
  section 14) -- only `ranking_mode="ppr"` and `graph_mode="scene_sparse"`
  get the full weighted-max/logsumexp treatment.
- No val_tune ablation numbers exist yet for any arm (A-F) -- `tuning/` in
  this checkout has no NExT-QA index cache (`eval/data/nextqa/` does not
  exist here), so `scripts/nextqa_single_video_eval.py --query-mode
  structured_v2` has been smoke-tested end-to-end with synthetic
  FrameRecords + real CLIP embeddings (see the top-level report), but not
  against real video/gold spans. This document will be updated with actual
  ablation numbers once that data is available and a val_tune run is
  explicitly requested.
