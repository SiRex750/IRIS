# Harness canonical-path parity re-verification (Part 2b)

**Run timing note:** this check was requested "before Family 2 starts," but
Family 2 (`ppr_lambda`) had already completed and been pushed to GitHub by
the time this request arrived. This check was therefore run **retroactively**
against the same harness/config that produced both Family 1 and Family 2's
results. If it had failed, that would implicate Family 2's already-published
results too, not just future work -- it did not fail (see below), but the
timing discrepancy is recorded here rather than silently treated as
"before."

## Plain-language answer

**Yes, this harness measures the real production pipeline, not an
approximation of it, for the retrieval path.** `scripts/part3_tune.py` calls
`iris.ingest.ingest()` for indexing and the exact same two internal
functions `iris.query.query()` itself calls for retrieval
(`iris.query._call_embed_query` then `iris.query._retrieve_with_l1`) -- it
does not reimplement embedding, graph construction, or PPR. This is the
opposite failure mode from the parked `t0_reproduction.py` benchmark, which
bypassed `ingest()`/`query()` and called `graph_sparse.retrieve_ppr()`
directly.

**One real exception found, not covered by the stop condition as written:**
the mIoP/IoU metric functions used by this harness (`eval/metrics.py`) are a
**separate reimplementation**, not the canonical validated implementation
`metric_registry.json` designates
(`benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts/nextgqa_metrics.py`).
See Task 6 below -- formulas were checked and are mathematically equivalent
for well-formed spans, so Family 1/2's numbers are not believed wrong, but
this is a genuine duplicate-implementation gap that should be closed, not
waved away by the formulas happening to agree.

## Task 1 -- harness location

`scripts/part3_tune.py` (repo root: `/home/ccbd/IRIS-1/scripts/part3_tune.py`).
This is the only script Family 1 and Family 2 were run through
(`python scripts/part3_tune.py retrieval_strategy` /
`python scripts/part3_tune.py ppr_lambda`).

## Task 2 -- retrieval call trace

`scripts/part3_tune.py`:
- `ensure_indexes()` (line 141) calls `iris_ingest.ingest(str(vpath), cfg)` --
  real production ingest, real `IRISIndex` objects saved via
  `iris_ingest.save_index()`.
- `evaluate_config()` (lines 166, 180-181) calls
  `from iris.query import _call_embed_query, _retrieve_with_l1` then
  `query_embedding, _ = _call_embed_query(q["question"], cfg)` followed by
  `retrieved_frames, _ = _retrieve_with_l1(index, query_embedding, cfg)`.

Compare to `iris/query.py`'s `_query_v2()` (the function `query()` dispatches
to for both `cerberus_mode="v2"` and `"none"`), lines 919-920:
```
query_embedding, embed_telemetry = _call_embed_query(question, config)
retrieved_frames, l1_telemetry = _retrieve_with_l1(index, query_embedding, config)
```
Same two functions, same order, same arguments. `_retrieve_with_l1` in turn
calls `_build_retrieved` -> `iris.scene_retrieval.retrieve_scene_sparse` ->
`iris.l2_asphodel`'s `graph.retrieve_ppr()` -- the harness never touches
`graph_sparse`/`l2_asphodel` directly and never hand-builds a seed vector,
graph, or PPR call itself. **No reconstruction found.**

## Task 3 -- config object identity

`scripts/part3_tune.py::make_config()`:
```python
cfg = IRISConfig()
cfg.cerberus_mode = "none"
for k, v in {**DEFAULTS, **overrides}.items():
    setattr(cfg, k, v)
```
This is a real `iris.iris_config.IRISConfig` instance. The same `cfg` object
is passed directly into `iris_ingest.ingest(str(vpath), cfg)` and into
`_call_embed_query(q["question"], cfg)` / `_retrieve_with_l1(index,
query_embedding, cfg)` -- read via `getattr(config, "retrieval_strategy",
...)` etc. inside `iris/ingest.py`, `iris/scene_retrieval.py`, and
`iris/l2_asphodel.py` exactly as `query()`/`ingest()` callers do. There is no
separate evaluation-only config object or parallel code path keyed off these
values.

## Task 4 -- 5-question retrieval-order comparison

Fixed question IDs (recorded before running):

| video | qid | question |
|---|---|---|
| 4882821564 | 1 | why did the boy pick up one present from the group of them and move to the sofa |
| 2435100235 | 7 | how does the man cycling try to sell the watch to the man in the trishaw |
| 2834146886 | 8 | what does the white dog do after going to the cushion |
| 8132842161 | 2 | why did the man in white hold tightly to the boy in white |
| 4260763967 | 4 | why is the boy in yellow reaching out to things on the green mat |

Config used: `retrieval_strategy="hybrid"` (Family 1's frozen value),
everything else at default (matches the exact config-hash
`cab2bac1628012a3` used throughout Family 1's `hybrid` trial and as the base
for every Family 2 trial before `ppr_lambda` override).

For each question: (a) called the harness's exact two calls directly
(`_call_embed_query` + `_retrieve_with_l1`), and (b) called
`iris.query.query()` directly on a freshly-loaded copy of the same index
with the identical config. Compared `retrieved_frame_idxs` order.

**Result: all 5/5 match exactly, same order, same frame indices.** Full
side-by-side data in `harness_parity.json`. No divergence to report.

## Task 5 -- no silent zeroing/skipping

`evaluate_config()`'s per-question loop wraps the embed+retrieve calls in a
bare `try/except Exception: continue` -- on any failure it **drops that
question from the scored set entirely** (visible in the final
`n_questions_scored` count), it does not substitute a zero/default result
and silently include it in the average. This matches the honest-failure
behavior the check is looking for.

Separately, inside the canonical production code itself (not
harness-specific): `iris/ingest.py` computes `clip_embedding` unconditionally
for every frame in `frames_to_index` at ingest time, and any frame that ends
up without a scene_id/embedding is *excluded* from graph construction
(`_compute_scene_centroids`, `iris/ingest.py:158-166`) rather than included
with a zeroed value. `iris/l2_asphodel.py:retrieve_ppr`'s
`node.embedding is None -> sem=0.0` branch exists as a defensive fallback in
production code shared identically by the harness and `query()` -- it is not
something the harness introduces, and in practice should not be reachable
given ingest's exclusion behavior above.

## Task 6 -- metric function provenance

**This is the one place this check did not come back clean.**
`metric_registry.json` designates
`benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts/nextgqa_metrics.py`
as the audited/validated official metric implementation (27/27 synthetic
tests passing per that setup task). `scripts/part3_tune.py` does not import
it -- it imports `eval/metrics.py`, which was written fresh for this tuning
harness rather than reusing the validated module.

I compared the formulas directly:
- `nextgqa_metrics.py::iop_single`/`iou_single` clip a reversed predicted
  span to zero-length and compute `intersection/pred_len` and
  `intersection/union` respectively, returning 0.0 on a zero-length
  predicted span or zero union.
- `eval/metrics.py::get_tIoU` computes the same `max(0, min(e,e')-max(s,s'))`
  intersection and the same union/pred-length denominators, with the same
  zero-guards.

For well-formed spans (`start <= end`, which is guaranteed for gold spans by
Part 2's structural validation and for predicted spans by construction in
`predicted_span_from_frames`), the two are mathematically equivalent -- I do
not believe Family 1/2's mIoP/IoP@0.5 numbers are wrong. But this is a real,
literal duplicate-implementation gap, exactly the pattern this check exists
to catch, and it should be closed (either by having `eval/metrics.py` import
from the canonical module, or by formally reconciling and registering
`eval/metrics.py` in `metric_registry.json`) before treating the tuning
metrics as fully audited-canonical rather than "independently re-derived and
spot-checked equivalent."

## Verdict

- Retrieval path: **canonical, verified, 5/5 match** -- meets the stop
  condition's literal criteria (exact match on all 5 questions, no
  reconstruction found).
- Metric implementation: **not the canonical registered module, but
  formula-equivalent** -- flagged as an open item, not a blocker under the
  stop condition as written, but should not be silently carried forward
  without a decision on whether to consolidate.

HARNESS_CANONICAL_PATH_VERIFIED
