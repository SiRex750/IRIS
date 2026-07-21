# Part 2c -- Empirical metric-implementation parity check

## Plain-language summary

**Real data: clean, 0 mismatches out of 29,535 question-scorings** (11
trials -- Family 1's 4 `retrieval_strategy` values + Family 2's 7
`ppr_lambda` values -- x 2,685 val_tune questions each). `eval/metrics.py`
(what the tuning harness actually uses) and
`benchmark_runs/paper_setup_.../scripts/nextgqa_metrics.py` (the module
`metric_registry.json` designates as canonical) produce byte-identical
IoP, IoU, and all four threshold flags on every single real question in
both families. Family 1 and Family 2's published rankings and winners are
unaffected -- confirmed two ways: (a) zero per-question mismatches, and (b)
recomputed per-trial mIoP/mIoU from both implementations independently
match the already-published `tuning/all_trials.csv` numbers to 5 decimal
places for all 11 trials (see table below).

**Synthetic edge cases: 1 out of 9 mismatch, confirmed and real, not
negligible.** The zero-duration-predicted-span-inside-gold case diverges:
`eval/metrics.py` returns `(IoP=1.0, IoU=0.0)`, `nextgqa_metrics.py`
returns `(IoP=0.0, IoU=0.0)`. This is not a bug in the sense of "wrong
arithmetic" -- it's a genuine specification difference. The official
`doc-doc/NExT-GQA` scorer (`code/TempGQA/eval_ground.py::get_tIoU`)
special-cases a zero-width predicted span: if it falls inside the gold
span, it counts as perfect precision (`IoP=1`) with `IoU=0` (a point has no
measurable overlap fraction of a union). `eval/metrics.py` was written to
replicate that special case directly from the official source.
`nextgqa_metrics.py` has no such special case -- it clips any non-positive-
length predicted span straight to `IoP=0`, regardless of location. **This
means the module `metric_registry.json` calls "canonical/validated" is
itself less faithful to the true official formula than the module it was
meant to validate against, for this one edge case.**

I verified this divergence never actually manifested in the real data: **0
out of 29,535 predicted spans across all 11 trials had zero duration**
(checked directly against `metric_parity_diff.csv`). `l2_retrieve_top_k>=5`
combined with `hybrid` frame-indexing makes a single-unique-timestamp
retrieval essentially never happen in practice on this dataset. So while
the divergence is real and confirmed, it had zero effect on any number
already published.

## Does Family 1/2's published ranking still hold?

**Yes, unchanged.** Recomputed per-trial mIoP/mIoU from both
implementations independently, compared against `tuning/all_trials.csv`:

| trial | published mIoP | eval_metrics recompute | nextgqa_metrics recompute | published mIoU | eval_metrics mIoU | nextgqa_metrics mIoU |
|---|---|---|---|---|---|---|
| retrieval_strategy=peak_only | 0.26394 | 0.26394 | 0.26394 | 0.21358 | 0.21358 | 0.21358 |
| retrieval_strategy=top_k_action | 0.25324 | 0.25324 | 0.25324 | 0.19592 | 0.19592 | 0.19592 |
| retrieval_strategy=peak_neighbors | 0.26483 | 0.26483 | 0.26483 | 0.21100 | 0.21100 | 0.21100 |
| retrieval_strategy=hybrid | 0.27748 | 0.27748 | 0.27748 | 0.20471 | 0.20471 | 0.20471 |
| ppr_lambda=0.00 | 0.27070 | 0.27070 | 0.27070 | 0.17854 | 0.17854 | 0.17854 |
| ppr_lambda=0.10 | 0.27245 | 0.27245 | 0.27245 | 0.18344 | 0.18344 | 0.18344 |
| ppr_lambda=0.25 | 0.27769 | 0.27769 | 0.27769 | 0.19374 | 0.19374 | 0.19374 |
| ppr_lambda=0.50 | 0.27748 | 0.27748 | 0.27748 | 0.20471 | 0.20471 | 0.20471 |
| ppr_lambda=0.75 | 0.27944 | 0.27944 | 0.27944 | 0.20858 | 0.20858 | 0.20858 |
| ppr_lambda=0.90 | 0.27944 | 0.27944 | 0.27944 | 0.20893 | 0.20893 | 0.20893 |
| ppr_lambda=1.00 | 0.27958 | 0.27958 | 0.27958 | 0.20749 | 0.20749 | 0.20749 |

Identical across all three columns for every trial, both metrics. Family
1's `hybrid` win and Family 2's `lambda=0.25` win both stand exactly as
reported.

## Consolidation -- deliberately NOT performed yet

The task asked to point `scripts/part3_tune.py` at `nextgqa_metrics.py`
"only after step 6 passes clean." Step 6 did not pass fully clean --
1/9 synthetic cases mismatched. I am not doing the swap as specified,
and specifically **not** because switching to `nextgqa_metrics.py` as-is
would silently adopt a less-faithful-to-the-official-spec formula for a
real (if currently unexercised) edge case. Blindly consolidating onto the
"designated canonical" file here would make the codebase *less* correct,
not more.

**Recommendation instead of blind consolidation:** fix
`nextgqa_metrics.py`'s `iop_single`/`iou_single` to add the official
zero-width-span special case (matching `eval_ground.py` and
`eval/metrics.py`'s existing behavior), re-run this same synthetic suite to
confirm 9/9 match, *then* point `part3_tune.py` at the now-corrected
canonical module and update `metric_registry.json`. This is a one-function
fix, not attempted here without explicit sign-off since it means editing a
file the registry calls "already validated" -- that status claim itself
would need updating.

**Other callers of `eval/metrics.py`:** none found. Only
`scripts/part3_tune.py` and this check's own scripts import it (an earlier
broad grep hit `benchmarks/t0_reproduction.py` and
`benchmarks/exp1a_metrics.py` on the substring `eval_metrics`, but those
are unrelated local variable names in those files, not imports of this
module -- verified with a targeted import-statement grep). So retiring
`eval/metrics.py` later, once consolidation is actually done, would be
safe with no other dependents to update.

## Outputs

- `metric_parity_diff.csv` -- all 29,535 real-data question x trial scorings, both implementations, match flag (0 mismatches).
- `metric_parity_synthetic.csv` -- 9 hand-verified edge cases (8 requested + 1 added to isolate the zero-duration-outside-gold case from the inside-gold divergence), both implementations, match flag (1 mismatch, documented above).
- This report.

## Verdict

- **Stop condition (ranking-changing mismatch): NOT triggered.** No
  real-data mismatch exists, and the confirmed synthetic mismatch never
  occurred in real data -- Family 1/2's winners and full rankings are
  unaffected. `BLOCKED_METRIC_PARITY` is not warranted.
- **Full "everything matches, consolidation done" success token: not
  printed**, because it wouldn't be true -- there is a real, confirmed,
  unfixed 1/9 synthetic divergence, and consolidation was deliberately
  withheld pending a decision on fixing `nextgqa_metrics.py` first.
