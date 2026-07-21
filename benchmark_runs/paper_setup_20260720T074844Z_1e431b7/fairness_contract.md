# Fairness Contract

Applies to every comparison table produced from this setup's infrastructure. A method violating
any rule below must be excluded from the "fair comparison" tables, not silently adjusted.

1. **Same dataset files.** Every method in a comparison row reads the same `dataset_manifest.json`-
   registered files (video root, question CSV, grounding annotation) and the same SHA-256 hashes.
2. **Same split.** No comparison mixes `val_tune`/`val_confirm`/`official_test` rows.
3. **Same query text.** Verbatim question text from the dataset CSV, no per-method reformulation
   variance, per `configs/layer_factorial/*.json::shared_fairness_fields.query_text_source`.
4. **Same metric implementation.** `scripts/nextgqa_metrics.py`, one implementation, one commit.
5. **Same K for budget-matched methods.** `l2_retrieve_top_k` (or the L1 retention-budget
   equivalent) must be numerically identical across methods being compared as "budget-matched."
6. **Same candidate pool when labeled same-pool.** E.g. `raw_clip_same_pool`/`clip_in_ppr_topk`
   must operate over the exact PPR-emitted candidate set (see `configs/peak_span_modes.json`
   requirement_notes) — verified by asserting equal candidate-ID sets, not merely equal size.
7. **Same failure denominator.** A timeout/exception counts as a wrong answer + IoP=0 + IoU=0 in
   the denominator, never dropped from `n`.
8. **Same timeout.** `answerer_timeout` / per-question timeout fixed across compared methods
   (120s in `configs/layer_factorial/*.json::shared_fairness_fields.timeout_seconds`).
9. **Same random seeds.** `L1_random_budget_matched_30seeds` and any other seeded baseline must
   record and reuse the exact 30-seed list across every method compared against it.
10. **Same model revision.** CLIP (`openai/CLIP@d05afc4`), answerer (`granite4:micro`), captioner
    (`moondream`/`vikhyatk/moondream2`) revisions pinned identically — see
    `configs/layer_factorial/*.json::shared_fairness_fields.model_revisions`.
11. **Same hardware for local latency comparisons.** This environment is CPU-only
    (`torch 2.13.0+cpu`, `cuda_available=False`) — any comparison against a GPU-run baseline must be
    labeled as cross-hardware and excluded from a "same-hardware latency" table.
12. **Same CPU thread count** across compared local runs (not yet pinned by this setup — record the
    actual thread count used at run time in `environment.json` for each run directory).
13. **Same GPU precision** — not applicable in this CPU-only environment; record explicitly as N/A
    rather than omitting the field.
14. **Same cold/warm state.** Never compare a warm run's latency to a cold run's latency (see
    `timer_registry.json::clock_policy.cold_warm_separation`).
15. **Same timing boundaries.** Timer start/stop points must match `timer_registry.json`'s
    per-video/per-question timer definitions exactly across compared methods.
16. **Same answer normalization.** `configs/layer_factorial/*.json::shared_fairness_fields
    .answer_normalization`.
17. **Same bootstrap samples for paired comparisons.** 10,000 replicates, `artifact_schema.json`.
18. **No silent fallback.** Every method in `method_registry.json` marked `NOT_IMPLEMENTED` must
    hard-fail if invoked, not substitute a different code path (see `split_guard.py` pattern of
    explicit rejection as the template for method-level guards to be added alongside real
    implementations).
19. **No post-test tuning.** Once `commands/12_freeze_test_configuration` runs, no sweep config
    may be re-opened before `commands/13_run_official_nexgqa_test` runs.
20. **No test-data leakage.** Enforced by `scripts/split_guard.py` (see `split_manifest.json`).
21. **No oracle method in fair comparison tables.** `L3_gold_span_oracle` in `method_registry.json`
    is explicitly tagged `"category": "ORACLE"` and must be filtered out of any table claiming fair
    superiority.
22. **Published results clearly separated from local reproductions.** `paper_comparison_registry.csv`'s
    `quote_or_reproduce` column marks every external row `QUOTE`/`ARCHITECTURAL_CONTEXT_ONLY` — no
    external paper's number is to be presented as if reproduced locally unless a `REPRODUCE` row is
    added after actually running that paper's released code, which this setup task does not do.
23. **Negative and null results must be retained.** `L3_shuffled_caption_negative_control`,
    `L3_query_blind_negative_control` results, once run, are kept in the results directory
    regardless of outcome — no run directory is deleted for producing an unfavorable result.
