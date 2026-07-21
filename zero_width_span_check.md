# Family 4 pre-check: zero-width predicted span occurrence

## Implementation in use

`scripts/part3_tune.py`'s `best_over_gold_spans()` calls `nextgqa_metrics.iop()`/`iou()` loaded from `/home/ccbd/IRIS-1/benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts/nextgqa_metrics.py` (the consolidated canonical module, per the Part 2c commit `5ceff0b`) -- **not** `eval/metrics.py`'s independent reimplementation, which was retired from this harness's scoring path.

The Part 2c zero-width special-case fix (`if pred_s == pred_e: return 1.0 if gold_s <= pred_s <= gold_e else 0.0`) **is present** in `iop_single()`: `True`.

## Empirical occurrence check across Family 4's grid

| l2_retrieve_top_k | n_scored | n_zero_width_predicted_spans |
|---|---|---|
| 4 | 2685 | 0 |
| 8 | 2685 | 0 |
| 12 | 2685 | 0 |
| 16 | 2685 | 0 |

**Zero-width predicted spans never occurred at any tested l2_retrieve_top_k value (4, 8, 12, 16).** Consistent with Part 2c's finding on Families 1-3 (0/29,535) -- even the smallest tested top_k (4) still retrieves enough frames with distinct timestamps that a single-unique-timestamp collapse essentially never happens on this dataset. No fix was needed for this family's data (it was already applied to `nextgqa_metrics.py` as of Part 2c/the `5ceff0b` push, and remains correctly wired in); this is stated plainly per the task's instruction to report even a clean 'never occurred' result rather than skip the check.
