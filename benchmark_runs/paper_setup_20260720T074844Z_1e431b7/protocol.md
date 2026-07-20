# Benchmark Protocol Summary

1. Freeze pipeline: `canonical_pipeline.md` documents the exact production call graph
   (`iris.ingest.ingest()` -> `iris.query.query()`), defaults (`graph_mode=scene_sparse`,
   `ranking_mode=ppr`, `cerberus_mode=legacy`), and dead/legacy paths not to be confused with it.
2. Dataset: NExT-GQA official split required; currently unverifiable locally, see
   `dataset_manifest.json`. A non-official 89-video subset is registered as a dry-run placeholder
   only, never to be reported as an official result.
3. Splits: video-level, hash-seeded, `split_manifest.json`, enforced by `scripts/split_guard.py`.
4. Metrics: `metric_registry.json` / `scripts/nextgqa_metrics.py`, 27 synthetic tests passed, no
   real prediction scored.
5. Lambda: proven formula `lambda*semantic + (1-lambda)*codec` in `lambda_semantics.md`, 10
   synthetic-graph endpoint tests passed on real production code.
6. Peak/span: registered as NOT_IMPLEMENTED — no code path exists yet, see
   `configs/peak_span_modes.json`.
7. Methods: `method_registry.json`, honestly marked IMPLEMENTED / PARTIAL / NOT_IMPLEMENTED per
   entry against the real canonical pipeline.
8. Layer factorial: 8 immutable config files in `configs/layer_factorial/`; only F111 and F101 are
   executable today (L1 OFF and L3 OFF require new code).
9. Sweeps: `configs/sweep/` (peak_rule, lambda, damping, candidate_k, span_half_width) plus
   `configs/sweep/sweep_plan.md` sequential order with leakage guards; not run.
10. Latency: `timer_registry.json` defines every timer; not yet inserted into `iris/` source.
11. Output schemas: `schemas/` + `artifact_schema.json`, all header-only / empty templates.
12. Paper comparisons: `paper_comparison_registry.csv`, 7 rows, tiers A/B/C, many fields flagged
    `UNVERIFIED` pending manual table extraction — see `paper_sources/README.md`.
13. Fairness: `fairness_contract.md`, 23 explicit rules.
14. Commands: `commands/00-20`, only 00-02 executed in dry-run/self-test form; 03-20 are templates.

No PRE_BENCHMARK_SETUP_READY.json is issued from this snapshot — multiple gates remain open. See
`setup_report.md` for the itemized gate table and exact next command.
