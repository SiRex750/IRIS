# Appendix C item 35(b) — shared-signal investigation

Read-only investigation. Traces `codec_conf` and the action score's first channel to
their common root (`frame_features["packet_size"]`) on `siddanth/peak-source-a6-p1`,
and asks whether §6.1's grounded/correct decomposition is confounded by that sharing.
No paper edit made; this is the artifact item 35(b) asks for.

## 1. The two paths, in full

### 1a. Action score, channel 1 (`iris/action_score.py`)

Per frame, `ActionScoreModule.score_all`:

```
residual   = packet_size for every frame in the batch          # raw
motion     = motion_magnitude for every frame
luma_ent   = luma_entropy for every frame

residual_n = _normalize(residual)
motion_n   = _normalize(motion)
entropy_n  = _normalize(luma_ent)

weight_sum = luma_diff_weight + motion_weight + luma_entropy_weight   # = 0.5+0.3+0.2 = 1.0

action_score = (luma_diff_weight*residual_n + motion_weight*motion_n
                + luma_entropy_weight*entropy_n) / weight_sum
action_score = clip(action_score, 0, 1)
```

`_normalize` (static, applied independently per channel, over the **whole video's**
frame batch, not grouped by anything):

```
if n < 50:  min_value, max_value = min(values), max(values)          # exact min-max
else:       min_value, max_value = percentile(values,2), percentile(values,98)  # clipped min-max
normalized = clip((values - min_value) / (max_value - min_value), 0, 1)
```

So channel 1 of the action score is `packet_size` run through a per-video, ungrouped,
(for real clips) 2nd/98th-percentile-clipped min-max scaling, then given a **fixed,
config-declared weight of exactly 0.5** (`luma_diff_weight=0.5`, the field item 6/35(a)
traced) in a three-term weighted sum whose denominator is 1.0. `packet_size` is the
single largest of the three inputs by weight (0.5 vs 0.3 motion vs 0.2 entropy).

Consumption: `action_score` is read at admission (tiering/peak-finding, §3.1 — decides
which frames become graph nodes at all), and — see §2 below — a second time, inside
edge-weight construction, via `_motion_similarity`'s default branch.

### 1b. `codec_conf` (`iris/ingest.py` lines ~386–422, `iris/l2_asphodel.py` `retrieve_ppr`)

At ingest, once per video, after `_build_graph`:

```
codec_conf_source        = "packet_size"     # IRISConfig default; also set explicitly
                                              # in scripts/pnowa_test_run.py's BASE dict
codec_conf_pictype_norm  = True              # IRISConfig default; also explicit in BASE

raw_signal[fi] = packet_size[fi]             # codec_conf_source == "packet_size" branch

# grouped by pict_type ("I"/"P"/"B"/"?", whatever Charon-V reports):
for each pict_type group with >= 2 members:
    rp_map[group] = _rank_percentile(raw_signal restricted to that group)
# groups of size 1 get rp = 0.5 (from _rank_percentile's n==1 branch)

codec_conf_map[fi] = 0.1 + 0.9 * rp_map.get(fi, 0.5)     # codec_conf in [0.1, 0.9]
```

`_rank_percentile` (`iris/ingest.py:65`): average-tied rank divided by `(n-1)`, i.e. the
textbook percentile rank — a **monotonic, order-preserving** transform of the group's raw
`packet_size` values, with ties resolved by mean rank.

`codec_conf` is stored on the graph node and the `FrameRecord` and does **not** mix in
motion or luma entropy at this stage — it is a pure (renormalized) function of
`packet_size` alone, given the default/actual `codec_conf_source="packet_size"`.

At query time, inside `L2Asphodel.retrieve_ppr` (the PPR seed construction, `l2_asphodel.py:1094–1122`):

```
raw_sem[nid]   = max(0, cosine(node.embedding, query_embedding))    # pure CLIP similarity
sem_rank       = _rank_pct(raw_sem)                                  # same rank-percentile fn

raw_codec[nid] = node.codec_conf                                     # already computed above
codec_rank     = _rank_pct(raw_codec)                                 # rank-percentile AGAIN,
                                                                       # this time over the
                                                                       # query-time node set

seed_raw[nid] = max(0, lambda_ * sem_rank[nid] + (1 - lambda_) * codec_rank[nid])
seed[nid]     = seed_raw[nid] / sum(seed_raw.values())      # or uniform 1/n if all zero
```

`lambda_ = ppr_lambda = 0.5` (`IRISConfig` default, and the frozen value in
`scripts/pnowa_test_run.py`'s `BASE`, per its own docstring: "ppr_lambda=0.5").

**Invariant worth stating precisely**: `_rank_pct` applied to `raw_codec` (i.e. to values
that are themselves already `0.1 + 0.9*rank_percentile(packet_size | pict_type)`) is
order-preserving — it depends only on the *ordering* of `codec_conf` values among the
query-time node set, and reproduces that same ordering (it does not un-mix the
per-pict-type grouping; it re-ranks the already-grouped-and-scaled values globally). So
`codec_rank`'s ordering equals `codec_conf`'s ordering, which equals the per-pict-type
`packet_size` rank ordering. `codec_rank` is exactly and only a re-scaled ranking of
`packet_size`.

`seed` therefore blends, additively at 0.5/0.5, (a) a purely query-semantic term
(`sem_rank`, zero `packet_size` dependence — `_semantic_similarity` reads only CLIP
embeddings) with (b) a term that is **100% `packet_size`-derived** (`codec_rank`, given
`codec_conf_source="packet_size"`).

`seed` is then passed as `personalization` into `nx.pagerank(g, weight="weight",
personalization=seed, alpha=damping)` (`damping = ppr_damping = 0.5`, same source). The
top-`k` nodes by the resulting PageRank score (`last_retrieval_score`) are returned as
`retrieved_frames`.

## 2. A second, less obvious `packet_size` route: edge weights

`motion_similarity_mode` defaults to `"action_score"` (`IRISConfig.motion_similarity_mode
= "action_score"`, comment: "today's behavior"), and is **not overridden** in
`scripts/pnowa_test_run.py`. Under this default, `_motion_similarity(node_u, node_v, ...)`
returns:

```
motion = 1 - abs(node_u.action_score - node_v.action_score) / max_score_range
```

— i.e. the edge's motion term is derived from the two nodes' `action_score`, which (§1a)
is 50%-weighted `packet_size`. §6.1 runs `graph_mode="flat"`, and the flat construction
path always uses the `"fully_connected"` edge-weight formula (`block_diagonal` and
`hierarchical_sparse` are `scene_sparse`-only branches — see §4.3's mode table),
which is:

```
edge_weight(u, v) = alpha * semantic(u, v) + beta * motion(u, v)     # no temporal term
```

`alpha = 0.4`, `beta = 0.3` (`IRISConfig` defaults, not overridden in `pnowa_test_run.py`,
and independently verified against 33 cached `config_snapshot`s in item 9's resolution).

So the graph's **edge weights** — which `nx.pagerank(..., weight="weight", ...)` uses to
propagate the seed across the graph, not just the seed itself — carry a second,
nonlinear `packet_size` dependency, at the `beta=0.3` slot, mediated through
`action_score`'s own 0.5 internal weight on `packet_size`. This is not decomposable into
a clean percentage (the dependency runs through `abs(Δaction_score)/range`, not a linear
combination), so I report it as: **non-zero, present in every edge of the graph PPR ranks
over, and additional to the seed-level dependency in §1b** — not as an exact fraction.
No per-clip numeric trace was pulled to attempt a variance-based estimate; if that
number is wanted, it needs an experiment (e.g. holding `codec_conf` fixed and permuting
`packet_size`'s contribution to `action_score` alone), not a code read.

## 3. Where the two paths diverge

Both paths start from the identical per-frame scalar: `frame_features["packet_size"]`,
read once at ingest from the Charon-V codec demux. They diverge immediately after that
read, at the normalization step:

- **Action score channel 1**: per-video (ungrouped), 2nd/98th-percentile-clipped
  min-max scaling of the *raw* value, then linearly blended (not re-ranked) with two
  other channels at fixed weights 0.5/0.3/0.2.
- **`codec_conf`**: per-pict-type-grouped, average-tied rank-percentile of the *raw*
  value, rescaled to `[0.1, 0.9]`, with **no** blending with motion or entropy at ingest
  time.

These are different transforms of the same raw number (one is a (clipped) affine
rescaling of the value itself and folds in two other quantities at fixed weight; the
other is a pict-type-grouped rank transform of the value alone). They are **not**
numerically identical, and their *rankings* need not agree either — a frame that is a top
decile `packet_size` value within its own pict-type group could sit anywhere in the
percentile-clipped, ungrouped, motion/entropy-blended `action_score` distribution.

What *is* shared past this point:
- `codec_conf`'s later `codec_rank` re-ranking (§1b) does not introduce a new
  dependency; it is order-preserving on `codec_conf` itself, hence on the per-pict-type
  `packet_size` ranking.
- `action_score`'s later use in edge weights (§2) is a **second, independent
  reappearance** of the same raw `packet_size` reading (via a different normalization
  path than `codec_conf`'s), inside the same PageRank call that `codec_conf` seeds.

So by the time `nx.pagerank` runs, `packet_size` has entered the computation **twice**,
through two different normalizations, at two different structural roles (personalization
vector and edge-transition weights) — not once. Quantified where it can be:
- Seed: exactly 50% (`1 - ppr_lambda = 0.5`) of the additive blend is `codec_rank`,
  itself 100% `packet_size`-derived.
- Edges: `beta = 0.3` of each `fully_connected` edge weight is the motion term, itself
  driven in part (not cleanly quantifiable, see §2) by `packet_size` via `action_score`'s
  0.5 internal weight.
- Everything else (`lambda_ = 0.5` of the seed; `alpha = 0.4` of each edge weight) is
  `sem_rank` / CLIP cosine similarity — zero `packet_size` dependence, confirmed by
  reading `_semantic_similarity`, which touches only `.embedding` fields.

## 4. Does span selection depend on `packet_size`? (§6.1: `span_mode="ppr_peak"`, `peak_source="clip_in_ppr_top8"`)

`eval/span.py::predict_span` with `peak_source="clip_in_ppr_top8"` calls
`_pick_by_clip_similarity(frames, query_embedding)`, which picks the peak frame `t*` by
**raw CLIP cosine similarity to the query embedding among `frames`** — nothing in that
function reads `codec_conf`, `action_score`, or `packet_size`. Given a fixed candidate
set, the peak choice itself is `packet_size`-free.

But `frames` **is** `retrieved_frames` — the exact top-`k` (`k=12` in
`scripts/pnowa_test_run.py`) output of `graph.retrieve_ppr`, i.e. the object traced in
§§1b–3 above. So:

- **Span location conditional on the retrieved set: independent of `packet_size`.**
- **Set membership — which frames are even eligible to be the peak, and therefore which
  region of the video the span can possibly center on — is not independent of
  `packet_size`**, for the reasons in §3. If the gold-adjacent frame never enters the
  top-12 because the PPR ranking (seed and/or edges) disfavored it, no amount of
  CLIP-similarity peak-picking recovers it, and the predicted span cannot land near the
  gold region.

One caveat on the fallback path: `predict_span` falls back to `_pick_by_ppr_score`
(which picks by `last_retrieval_score`, i.e. the raw PPR score directly — a *third*,
even more direct `packet_size` route) when `query_embedding is None` or no candidate
frame carries a usable embedding. For cached-index frames with CLIP embeddings already
populated (the normal case for this eval), this fallback should not trigger; I did not
verify it never does across the actual 120-question run, only that the code path exists
and is not the documented default.

## 5. Does the answerer's frame selection depend on `packet_size`? Same or different route?

**Same route, not a different one.** In both `iris/query.py::query` (legacy) and
`_query_v2` (the `cerberus_mode="v2"` path used for the AnswerClaims contract), the
single call `retrieved_frames = _build_retrieved(index, query_embedding, config)` produces
the frame list that is passed, unmodified, into `wrapper_populate_cache(cache_obj,
retrieved_frames)`, whose `cache_obj.as_context_text()` becomes the literal context string
handed to the answerer (`aria.generate(...)` / `_generate_answer_claims_v2_wire(...)`).

There is no second, independently-computed frame selection for the answerer — grounding
(§4) and answering read from the identical Python list object produced by one
`retrieve_ppr` (or, under `scene_sparse`, `retrieve_scene_sparse`, which also calls
`retrieve_ppr` on an induced subgraph — same seed/edge mechanics, not a different
mechanism) call.

## 6. §6.1's decomposition: do grounding and correctness share `packet_size` as a common upstream input?

**Yes**, and precisely: both `IoP@0.5`-grounded status and answer correctness are
computed from outputs of one shared retrieval event (§5), and that event's ranking
depends on `packet_size` at (at least) two points — the PPR seed (exactly 50% weight, via
`codec_conf`) and the edge weights the seed is propagated over (via `action_score`,
non-zero but not cleanly quantified). Grounded-ness and correctness are not independent
measurements of two separably-tuned subsystems; they are two different readouts (an IoP
overlap test, and an LLM accuracy check) of what one retrieval mechanism handed the
answerer.

**What this does and does not undermine, precisely:**

- It does **not** make the coupling spurious by itself. "Grounded" is, by construction, a
  proxy for "the correct region's frames are present in the evidence the answerer
  received" — and because grounding and answering read the *same* frame list (§5), a
  grounded question mechanically means the answerer had the right evidence in its
  context window; an ungrounded one means it did not. That causal link (right evidence
  present → higher chance of a right answer) is real and is not an artifact of
  `packet_size` specifically — it would hold even if retrieval ranked purely on
  `sem_rank`. The replication of the ~15-point gap across the in-sample and held-out
  samples (§6.1, `[C4.2]`) remains genuine evidence of *that* link, and nothing here
  erases it.

- It **does** undermine a stronger, unstated reading: that a grounded question was
  grounded *because* the query semantically matched the right content, and that the
  gap therefore demonstrates the answerer engaging with query-relevant evidence
  specifically. Since up to half the seed (and a further, non-zero share of the edge
  weights) that produced the ranking is a codec artifact uncorrelated with the query
  text, a question can be "grounded" in this pipeline's sense because `packet_size`
  happened to rank the correct region's frames upward for reasons that have nothing to
  do with what was asked. When that happens, correctness still tracks grounding (the
  right frames are still in context, whatever put them there), so the **gap itself is
  not inflated or explained away by this** — but "grounded" cannot be read as "retrieved
  for the query-semantic reason NExT-GQA's motivation (§2.3) cares about (not merely
  present, but present because relevant)." That is a narrower property than what
  "grounded" as measured here actually establishes.

- Could a shared upstream signal produce the observed gap **without the answerer
  actually using the retrieved spans**? I can rule this out only partially. The
  mechanistic route this trace establishes — grounded ⇒ correct-region frames literally
  present in the LLM's prompt context ⇒ the answerer *has the opportunity* to use them —
  is well-supported by the code (§5: identical list feeds both measurements). That is
  a real channel through which grounding could cause correctness, and the more
  parsimonious explanation of the data than a confound. But I cannot rule out, from a
  code trace alone, a residual confound of the shape "some property correlated with
  `packet_size`-favorable retrieval (e.g. a codec/scene characteristic) also correlates
  with question answerability independent of retrieved content" — e.g. if certain scenes
  have both distinctive codec activity (helping `codec_conf` rank them correctly) and are
  disproportionately easy questions for an unrelated reason (world-knowledge bias,
  shorter/simpler phrasing). Checking that would require a control analysis (e.g.
  P(correct|grounded) vs P(correct|ungrounded) computed within `packet_size`- or
  codec-activity-matched strata, or an ablation with `codec_conf_source` and
  `motion_similarity_mode` forced to purely-semantic settings) that does not currently
  exist in `eval_results/`. So: the direct mechanistic explanation is supported and is
  the better-evidenced account; a subtler confound is not excluded and has not been
  tested.

**On the specific sentence, "which supports the coupling independently of where the
levels land"**: this clause is about replication of the *gap* across the in-sample and
held-out samples (i.e., robustness to a level shift between the two measurements) — not
about whether the retrieval mechanism that produces "grounded" is itself independent of
non-semantic signals. Nothing in this trace touches that claim, and it does not need to
narrow. The place that would need a caveat, if the surrounding prose leans on this
decomposition to support the §2.3 framing ("right for wrong reasons" — i.e., a
faithfulness-style claim that a grounded answer is right *because* the retrieval found
query-relevant evidence), is exactly the point above: **"grounded" here can arise from a
codec-derived, query-independent ranking signal at up to half the seed weight, so a
grounded/correct question is not always grounded for a query-semantic reason** — only
demonstrably grounded in the weaker, but still real, sense of "the right frames were in
the context the answerer saw."

## 7. §4 impact: nil, confirmed

§4.1's identity gate (`fully_connected` ↔ `block_diagonal`, `scene_sparse` graph mode)
checks that two construction code paths produce byte-identical node and edge sets given
the *same* inputs and config — it is a structural equivalence claim about the
construction step itself, not a claim about what any config field means or whether two
fields are independent signals. It holds regardless of how `packet_size` is or is not
shared between `codec_conf` and the action score, and regardless of which
`motion_similarity_mode` or `codec_conf_source` is in force, because those are inputs to
the comparison, not things the comparison is checking the independence of. Additionally,
§4.1/§4.2 run under `graph_mode="scene_sparse"` with the `fully_connected` /
`block_diagonal` edge formulas — a different code path from §6.1's `graph_mode="flat"`
run — so the two are not even sharing a runtime call. No re-scoping of §4 is needed.

## Wording §6.1 would need if the independence claim is narrowed

No claim of independence between `codec_conf` and the action score currently appears in
§6.1 or elsewhere in the body (confirmed by grep — see item 35(a)/(b) for what does and
doesn't appear); the sentence flagged as at-risk in item 35(b) was the general framing in
§2.3 ("point at the evidence... right for the wrong reasons... report grounded accuracy
rather than answer accuracy alone") as it is echoed by §6.1's coupling paragraph. If that
framing is read as claiming the grounded/correct gap demonstrates query-semantic
faithfulness specifically, §6.1 would need one added clause near the coupling paragraph
along these lines (draft language, not proposed as final):

> "'Grounded' here is a property of the retrieved-frame set, which our PPR ranking
> assembles from a 50/50 blend of query-semantic similarity and a codec-derived signal
> (`codec_conf`, itself a renormalization of the same `packet_size` quantity used in the
> action score — Appendix C item 35(b)); a question can be grounded because retrieval
> found the right region for a non-semantic (codec) reason. The reported gap therefore
> shows that having the right evidence in context predicts a correct answer, but does not
> by itself show retrieval found that evidence *because it was query-relevant*."

This is offered as the shape of the needed caveat, not as approved wording — per the
task, no edit was made to `paper/IRIS_paper_draft.md`.
