# Sequential Validation Sweep Plan (NOT EXECUTED)

Order, exactly as specified by the task, each step freezing before the next begins:

1. Tune peak rule on `val_tune`. **BLOCKED**: only `ppr_score_legacy` is implemented (the other two
   peak rules require new code — see `configs/peak_span_modes.json`). Cannot meaningfully "tune"
   across 1 available option; step is a placeholder until `raw_clip_same_pool` /
   `clip_in_ppr_topk` exist.
2. Freeze peak rule.
3. Tune `ppr_lambda` in {0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00} on `val_tune`. Implemented, ready
   to run once commands are executed (not part of this setup task).
4. Freeze `ppr_lambda`.
5. Tune `ppr_damping` in {0.50, 0.65, 0.80, 0.85, 0.90} on `val_tune`. Implemented.
6. Freeze `ppr_damping`.
7. Tune `l2_retrieve_top_k` in {4, 8, 12, 16} on `val_tune`. Implemented.
8. Freeze `l2_retrieve_top_k`.
9. Tune span half-width in {1.0, 1.5, 2.2, 3.0, 4.0} seconds. **BLOCKED**: no span-construction
   stage exists (see `configs/peak_span_modes.json`) — nothing to tune yet.
10. Freeze span half-width.
11. Evaluate once on `val_confirm`.
12. Freeze the final test configuration.
13. Execute the official test once. **BLOCKED**: official test annotations not verifiable locally
    (see `dataset_manifest.json`, `setup_failures.jsonl`).

## Guards against joint test-set optimization

- `scripts/split_guard.py::guard_tuning_command` rejects any command in steps 1-10 that targets
  `split="official_test"` or whose config path contains "test"/"official".
- `scripts/split_guard.py::guard_official_test_command` rejects step 13 unless `split="official_test"`
  exactly, and additionally refuses to run while the `official_test` partition is unpopulated
  (0 videos) — both refusal paths are exercised in `scripts/split_guard.py`'s own self-test.
- Steps 1-10 read only `val_tune`; step 11 reads `val_confirm` exactly once; no step may read both
  in the same invocation (would require a two-split flag not exposed by any command template).
