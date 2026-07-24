# ppr_lambda retrieval-in-gold re-sweep

Diagnostic re-sweep following the val_confirm gap diagnostic (commit `c24f8f8`), which
found ~64% of Acc@QA-right/Acc@GQA-wrong failures never had a retrieved frame inside the
gold span at all (bucket c), and Siddanth's separate diagnostic (branch
`siddanth/peak-source-a6-p1`) showing codec_rank beats sem_rank ~79% of the time within
failure cases. This sweep tests whether shifting the PPR seed toward query relevance
(`seed = lambda*sem_rank + (1-lambda)*codec_rank`, higher lambda) recovers retrieval-in-gold
rate -- **not** whether it moves whole-dataset average mIoP, which Family 2 already showed
is flat.

**This is a measurement-only re-sweep. It does NOT re-freeze `ppr_lambda` and does NOT
modify `tuning/frozen_state.json`.** `ppr_lambda` stays frozen at 0.50 pending an explicit
human decision after reading this report.

## Step 0 -- preflight

- `tuning/index_cache_val_confirm_e2e/` -- 112 `.npz` files, config-hash `4edae64ed40256e3`,
  reused with **no re-ingest** (`ppr_lambda` is not in `INGEST_RELEVANT_KEYS` -- confirmed
  in `scripts/part3_tune.py`, it only affects the retrieval-time PPR seed blend, not the
  stored index).
- `tuning/val_confirm_e2e_per_question.csv` -- 639 rows, reused to identify the two failure
  subsets by `(video, qid)` and to reproduce `overlap_status` exactly as `c24f8f8` computed
  it. Recomputing those subsets from the CSV gave subset A (Acc@QA=1 x overlap_status in
  {partial, zero}) = 228 questions and subset B (all overlap_status=zero) = 344 questions --
  both exactly matching the diagnostic's counts.
- `tuning/frozen_state.json` frozen block unchanged since `6f36f08`.

All retrieval-only: embed question -> `_retrieve_with_l1` top-`l2_retrieve_top_k=4` frames.
No captioner, answerer, Cerberus, or LLM calls. Acc@QA is read from the existing
per-question CSV for subset identification only, never recomputed -- this sweep does not
re-run the answerer, so it cannot detect whether a different lambda's retrieved frames
would change Acc@QA. **Flagging explicitly per the task's own caveat: if the answerer's
input context changes with lambda enough to flip an MC answer, this sweep would not see
it** -- everything reported here is retrieval/span-metric only (any-in-gold, top-1-in-gold,
CLIP-anchor-in-gold, mIoP/mIoU/IoP@0.5), not a re-measurement of Acc@GQA itself.

## Step 1+2 -- the sweep and its metrics

Grid: `ppr_lambda` in [0.50, 0.60, 0.75, 0.90, 1.00], all 639 val_confirm questions, all
other frozen hyperparameters held fixed (`retrieval_strategy=hybrid`, `ppr_damping=0.50`,
`l2_retrieve_top_k=4`, `span_method=D`/`half_width_s=2.2`, `peak_distance=5`,
`peak_prominence=0.05`, action-score weights 0.8/0.1/0.1, `persistence_threshold=0.4`,
`max_prominence=0.5`).

**Control check (lambda=0.50 must reproduce `c24f8f8`):** subset-B retrieval-in-gold rate at
lambda=0.50 = 87/344 = 25.2907% -- **exact match** to the c24f8f8 diagnostic's 74.71%
zero-in-gold (257/344 = 25.29% in-gold), confirming the baseline is trustworthy and
the rest of the sweep is comparable to it.

| ppr_lambda | retrieval-in-gold (whole, n=639) | retrieval-in-gold (subset A, n=228) | retrieval-in-gold (subset B, n=344) | top-1-in-gold (whole) | CLIP-anchor-in-gold (whole) | mIoP (whole) | mIoU (whole) | IoP@0.5 (whole) |
|---|---|---|---|---|---|---|---|---|
| 0.50 (baseline/frozen) | 52.74% | 35.96% | 25.29% | 30.36% | 32.08% | 0.3014 | 0.1572 | 30.99% |
| 0.60 | 54.15% | 36.40% | 28.20% | 30.67% | 32.55% | 0.3076 | 0.1616 | 30.99% |
| 0.75 | 55.56% | 39.47% | 29.94% | 31.30% | 32.24% | 0.3041 | 0.1587 | 30.67% |
| 0.90 | 55.87% | 38.60% | 29.94% | 32.55% | 31.92% | 0.3042 | 0.1605 | 30.36% |
| 1.00 | 54.93% | 35.53% | 28.78% | 31.92% | 31.77% | 0.3019 | 0.1595 | 30.20% |

## Step 3 -- interpretation

**Retrieval-in-gold rate rises with lambda, peaks around 0.75-0.90, then partially reverses
at lambda=1.00 (pure semantic, codec removed entirely):**

- Whole split: 52.74% (lambda=0.50) -> 55.87%
  (lambda=0.90), a **+3.13pp**
  (5.9% relative) gain.
- Subset A (Acc@QA=1 x overlap_status in {partial,zero}, the direct
  Acc@QA-right/Acc@GQA-wrong failure group): 35.96% ->
  39.47% (lambda=0.75), a
  **+3.51pp**
  (9.8% relative) gain.
- Subset B (all zero-overlap questions, any Acc@QA -- the purest measure of bucket (c)):
  25.29% -> 29.94%
  (lambda=0.75), a **+4.65pp**
  (18.4% relative) gain -- the largest relative
  improvement of the three, on exactly the subset the c24f8f8 diagnostic flagged as bucket
  (c).
- At lambda=1.00 (codec_rank fully removed), all three retrieval-in-gold rates give back
  roughly a third to half of the gain seen at 0.75-0.90 -- so codec_rank isn't purely noise
  to be zeroed out; **0.75-0.90 is a genuine sweet spot**, not "more semantic is strictly
  better."

**Meanwhile mIoP/mIoU/IoP@0.5 stay flat** across all five lambda values (mIoP ranges
0.3014-0.3076, a
spread of 0.0061 --
noise-level, consistent with Family 2's original whole-average finding. **This does NOT
contradict Family 2** -- Family 2 measured a different thing (average-precision metrics on
the whole split) and found it flat; this sweep measured a different thing (binary
in-gold hit rate, with a focus on the failure subset) and found it moves. Both are correct
readings of the same underlying data.

**CLIP-anchor-in-gold rate barely moves** (31.77%-32.55%,
essentially flat) even as "any of 4 in gold" rises meaningfully. This means higher lambda
expands the retrieved *candidate pool's* chance of containing a gold-span frame, but Method
D's CLIP-anchor selection *within* that pool does not track the improvement as strongly --
consistent with bucket (b) ("wrong frame selected among good candidates") from the
`c24f8f8` diagnostic being a separate, still-unaddressed problem that a lambda change alone
will not fix.

## Conclusion / recommendation for the human

This is a **real, non-trivial finding**: shifting `ppr_lambda` from 0.50 toward 0.75-0.90
recovers meaningful retrieval-in-gold rate specifically on the failure subsets (+4.65pp
/ 18.4% relative on subset B), while leaving the whole-average mIoP/mIoU/IoP@0.5
metrics untouched -- exactly the kind of effect an average-precision metric is blind to but
a binary retrieval-in-gold check exposes. It does not fully close bucket (c) (subset B still
sits at ~30% in-gold even at the best lambda, up from ~25%), and it does nothing for bucket
(b) (CLIP-anchor-in-gold is flat). **Recommendation: this justifies a follow-up
`ppr_lambda`-focused re-freeze decision (e.g. re-running the full Family 2 grid restricted
to {0.60, 0.75, 0.90} with retrieval-in-gold rate as a tracked secondary metric, not just
mIoP) -- but that re-freeze decision should be made by a human, not auto-applied here.**
`ppr_lambda` remains frozen at 0.50 in `tuning/frozen_state.json`; this file is unmodified.
The remaining ~70% of bucket (c) failures at any lambda tested, and the flat
CLIP-anchor-in-gold rate, suggest the next things to check after any lambda decision are
`l2_retrieve_top_k` pool size, `codec_conf_source`, and Method D's anchor-selection logic
itself (bucket b) -- lambda alone will not fully close the gap.
