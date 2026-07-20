# Cache Provenance Caveat: Captioner Default Change (moondream → minicpm)

**UPDATE (2026-07-20, later same day)**: A full cache reset was performed
(`smoke/cache_reset_minicpm_final_report.md`). Byte-level inspection of every cache location in
this repo, done *before* deleting anything, found **zero cached captions of any provenance
anywhere** — the risk described below was preventive, not realized. All 6 cache locations
(272 files, ~3.06 GB) were deleted regardless for a clean baseline; the 3 smoke-test video caches
were freshly rebuilt and query-verified to use MiniCPM-V 4.6 with no fallback. See that report for
full detail, including a caveat that the standard smoke harness's on-disk `.npz` files don't
themselves contain captions (query-time captions are only proven via the trace/log and an ad-hoc
post-query re-save demonstration, not the harness's own saved cache file).

**2026-07-20**: `IRISConfig.captioner_backend`'s default was changed from `"moondream"` back to
`"minicpm"` (`iris/iris_config.py`), and `configs/default_iris_config.json` now explicitly states
`captioner_backend`/`answerer_backend`/`answerer_endpoint`/`answerer_model` rather than relying on
implicit dataclass defaults. This reverses an earlier unintentional drift (`captioner_backend`
silently defaulted to `"moondream"` even though the seated, live-verified production captioner is
`minicpm-v4.6`, served by Ollama at ingest time).

## What this means for existing `index_cache/` entries

`FrameRecord.caption` is computed **once** per frame, at whichever time the frame was first
captioned (either at ingest, or lazily on the first query that retrieves it — see
`iris/query.py::_ensure_captions`), and is then **serialized into the cache**
(`iris/ingest.py::save_index`/`load_index`). Changing `captioner_backend`'s default does **not**
retroactively recaption anything already sitting in a `.npz` cache file.

**Any index cache built/captioned before this change has moondream-generated captions baked into
its `FrameRecord.caption` fields.** Any index cache built/captioned after this change (with no
explicit `captioner_backend` override) will have minicpm-v4.6-generated captions instead.

## Concrete risk

**Mixing moondream-captioned and minicpm-captioned frames within the same benchmark run or
comparison table silently confounds results.** The two captioners produce systematically different
caption style/detail/length (see `smoke/smoke_report.md` and the item-3 fix in this repo's history:
moondream tended toward short generic scene descriptions; minicpm's prompt asks for an enumerated
per-person/object/action list). A method-vs-method or before/after comparison that unknowingly
reads captions from caches spanning this default change is not comparing like-for-like evidence
quality — it is comparing (at least partly) captioner identity, not the thing actually being
measured.

## What was NOT done here (by design)

No automatic cache migration/recaptioning was performed. Per explicit instruction, this is flagged,
not auto-fixed. Before running or trusting any benchmark that reuses a pre-existing `index_cache/`
directory (including anything under `smoke/cache/`, `benchmark_runs/*/cache/`, or any other cached
`.npz` index from before 2026-07-20), verify which captioner produced its cached captions —
e.g. by inspecting caption style/length, or by checking the git commit date of the run that
produced the cache against this change's commit — before including it in a comparison. When in
doubt, force a recaption (delete the affected cache and re-ingest, or clear `FrameRecord.caption`
before querying) rather than assuming provenance.

See also: `benchmark_runs/paper_setup_20260720T074844Z_1e431b7/canonical_pipeline.md` for the
broader canonical-pipeline trace this caveat is scoped against (that snapshot predates this change
and is left as-is/immutable; this file is the authoritative note going forward).
