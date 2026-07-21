# IRIS Paper-Setup Report — 2026-07-20T07:48:44Z @ 1e431b7a94f6deb9e14cb7c69a504dfcf3d4c731

## Starting git state
Branch `fix/charon-full-decode-geometry`, 2 commits ahead of `main`, no divergence, no merge
conflicts. Large pre-existing uncommitted working tree (32 modified files) untouched by this task.
Full detail: `git_state.txt`. No commit was made by this setup task.

## Files inspected
`iris/*.py` (ingest, query, pipeline, l2_asphodel, scene_retrieval, l2_index, charon_v,
action_score, codec_validator, cached_frame, aria, cerberus_v, cerberus_layers, claim_contract,
iris_config, types), `api.py`, `configs/default_iris_config.json`, `eval/grounding_scorer.py`,
`eval/data/nextqa/{val.csv,gsub_val.json,gsub_test.json}`, `data/nextqa_exp1a/*.csv`,
`benchmark_runs/t0_nextgqa_89_cpu/*`, `benchmarks/*.py`, `CCBD.zip` listing.

## Files created
Everything under `benchmark_runs/paper_setup_20260720T074844Z_1e431b7/` (this directory). No
production file was modified.

## Canonical pipeline
`canonical_pipeline.md`. Headline finding: **no CLIP-peak-reranking / predicted-temporal-span
stage exists in `iris/`** — the assumed architecture in the task prompt does not match the code.
Also: `iris/pipeline.py`'s top-level wrapper functions are dead code from `api.py`'s perspective;
default retrieval path is `scene_retrieval.retrieve_scene_sparse` (not flat-graph PPR directly);
default `cerberus_mode` is `"legacy"`, not the fully-typed `"v2"` claim-contract path.

## Dataset readiness — BLOCKED at official scale
`dataset_manifest.json` / `setup_failures.jsonl`. Only 91 of the videos referenced by
`val.csv`/`gsub_val.json` exist locally. `gsub_val.json` and `gsub_test.json` are byte-identical
(same SHA-256) — there is no independent official test-split annotation in this repo. Per user
decision, an 89-video/255-event non-official subset is registered as an explicitly-labeled
placeholder for infrastructure dry-runs; it passed 0/255 structural violations (start<=end,
in-bounds).

## Metric readiness
`metric_registry.json`, `metric_definitions.md`, `scripts/nextgqa_metrics.py`. 27/27 synthetic
tests pass (`synthetic_metric_test_report.txt`), handwritten expected values, no real prediction
scored. Flagged discrepancy: existing `eval/grounding_scorer.py::iop()` uses a different
multi-gold-span protocol (union-before-scoring) than the official max-over-spans protocol
implemented here — needs reconciliation before any official number is reported.

## Lambda meaning — proven
`lambda_semantics.md`, `ppr_formula.md`, `lambda_endpoint_tests.txt`: 10/10 synthetic-graph tests
pass against real production code (`L2Asphodel.retrieve_ppr`). Confirmed:
`seed = lambda*sem_rank + (1-lambda)*codec_rank`; `lambda=1.0` is semantic-only, `lambda=0.0` is
codec-only; no cross-contamination between signals; deterministic node ordering.

## Peak/span configuration options — NOT_IMPLEMENTED
`configs/peak_span_modes.json`. All 3 requested peak-selection modes and all span-construction
concepts (peak timestamp, half-width, boundary clamping, predicted start/end) require new code;
only `ppr_score_legacy` (= today's default PPR top-1 output, relabeled) is usable as-is.

## Model/backend registry
Captioner: `moondream` (`vikhyatk/moondream2`) default, `minicpm-v` (Ollama) and `blip` alternates.
Answerer: `llama_server` (`granite4:micro` @ `127.0.0.1:8091/v1`) default, `ollama`/`openai`
alternates. CLIP: `openai/CLIP@d05afc4`. Environment is CPU-only (`torch 2.13.0+cpu`,
`cuda_available=False`, no `nvidia-smi` on PATH).

## Baseline / method registry
`method_registry.json`: every L1/L2/L3 baseline from the task list registered with an honest
`availability` (`IMPLEMENTED` / `PARTIAL` / `NOT_IMPLEMENTED`) rather than assumed working. Roughly
half of the requested ablations require new code (see registry for the exact list — notably all
L1-admission-strategy ablations except the action-score-weight ones, and most L3 verifier-bypass
ablations).

## Layer-factorial registry
`configs/layer_factorial/F{111,110,101,100,011,010,001,000}.json`, all 8 created and immutable.
Only **F111** (full production) and **F101** (L2 OFF only) are executable today; the other 6 are
blocked on L1-uniform-admission and/or L3-no-verifier code that doesn't exist yet.

## Latency instrumentation
`timer_registry.json` fully specifies every per-video/per-question timer and resource metric plus
the clock policy (monotonic wall clock, `perf_counter_ns` for CPU, cold/warm separation, no
zero-latency reporting, no cross-method timer reuse). Not yet inserted into `iris/` source — that
is code work, out of scope for SETUP ONLY.

## Leakage protections
`split_manifest.json` (video-level, hash-seeded, 71 val_tune / 18 val_confirm videos on the
placeholder subset, `official_test` intentionally left unpopulated) + `scripts/split_guard.py`,
whose 4 rejection/acceptance behaviors were verified by direct execution (see
`logs/00-02_dry_run_log.txt`).

## Fairness protections
`fairness_contract.md`, 23 explicit rules covering dataset/split/query/metric/K/pool/failure/
timeout/seed/model-revision/hardware/thread/precision/cold-warm/timing/normalization/bootstrap/
no-fallback/no-post-test-tuning/no-leakage/no-oracle/quote-vs-reproduce/negative-results-retained.

## Paper comparison registry
`paper_comparison_registry.csv`, 7 rows (NExT-GQA, VideoChat-TPO, TOGA, VideoMind, MUPA,
Chain-of-Glimpse, ReMoRa), each with an arXiv id found via web search where available. Most
per-table metric fields are marked `UNVERIFIED` (a search summary is not a page/table citation);
row 2 (VideoChat-TPO)'s identity itself could not be confirmed and is flagged for manual check
before any citation. Local primary-source PDFs matched for MUPA and ReMoRa in `CCBD.zip` — see
`paper_sources/README.md`.

## Remaining blockers (why `PRE_BENCHMARK_SETUP_READY.json` is NOT created)

1. Official NExT-GQA dataset not verifiable locally (Section 3 stop condition met).
2. Peak-selection / span-construction stage does not exist in code (Sections 7, 9, 10 partial).
3. L1 uniform-budget-matched admission does not exist in code (blocks 4/8 factorial cells + a
   fair-baseline method entry).
4. L3 no-verifier bypass does not exist in code (blocks 4/8 factorial cells + a fair-baseline
   method entry).
5. Latency instrumentation is specified but not inserted into `iris/` source.
6. `paper_comparison_registry.csv` numeric fields need primary-source page/table verification
   before any figure is quoted in a paper.
7. `eval/grounding_scorer.py::iop()` vs `scripts/nextgqa_metrics.py::iop()` multi-gold-span
   protocol discrepancy needs reconciliation.

## Setup gate table (task Section 16)

| # | Gate | Status |
|---|---|---|
| 1 | No merge conflicts | PASS |
| 2 | Canonical production path documented | PASS |
| 3 | Dataset manifest complete | **FAIL at official scale** (placeholder-only PASS) |
| 4 | Validation/test split leakage guard active | PASS |
| 5 | Metric definitions implemented and synthetic-tested | PASS |
| 6 | Lambda semantics proven | PASS |
| 7 | Peak/span modes registered | PASS (registered as NOT_IMPLEMENTED, honestly) |
| 8 | Method registry complete | PASS (registered, ~half flagged NOT_IMPLEMENTED) |
| 9 | Eight layer-factorial configs complete | PASS as files (2/8 executable) |
| 10 | Component-ablation configs complete | PARTIAL (blocked entries flagged, not configured) |
| 11 | Sweep configs complete | PASS (2/5 dimensions blocked: peak_rule, span_half_width) |
| 12 | Timer instrumentation complete | PARTIAL (specified, not inserted into source) |
| 13 | Output schemas complete | PASS |
| 14 | Paper registry complete | PARTIAL (rows exist, numeric fields UNVERIFIED) |
| 15 | Fairness contract complete | PASS |
| 16 | All future commands prepared | PASS (00-02 dry-run executed, 03-20 templated) |
| 17 | No real benchmark result was produced | PASS |

**Overall: NOT READY for official-scale benchmarking.** Infrastructure is in place and testable
(dry-run/synthetic checks all pass), but the dataset, peak/span code, two of the eight
layer-factorial cells' prerequisite code, and latency instrumentation are open work.

## Exact next command to run the first smoke test

Once the above blockers are resolved (at minimum: official dataset acquired and verified per
`dataset_manifest.json`'s checklist), the first real command is:

```
python commands/03_run_production_smoke.py \
  --config configs/layer_factorial/F111.json \
  --split val_tune \
  --out-dir benchmark_runs/smoke_<UTC_TIMESTAMP>
```

This has **not** been run. It is a template only.

## Confirmation

No real benchmark testing was started. No video was decoded for evaluation purposes. No model
(captioner, answerer, verifier) was loaded or queried. No accuracy/grounding/latency number was
computed against a real dataset question. `real_benchmark_started: false` (see `provenance.json`).
