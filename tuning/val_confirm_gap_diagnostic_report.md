# val_confirm Acc@GQA gap diagnostic

Pure analysis + retrieval-only re-check on top of `val_confirm_e2e_per_question.csv`
(commit `6f36f08`, 639 questions, config-hash `4edae64ed40256e3`). No captioner/answerer
calls, no re-ingest -- retrieval-only re-check reused the already-ingested indexes in
`tuning/index_cache_val_confirm_e2e/` (112 videos).

Baseline from that run: Acc@QA=0.5462, Acc@GQA=0.1894 (mIoP/IoP@0.5 unaffected by this
diagnostic -- read-only).

## Step 0 -- preflight

All three prerequisites confirmed present and consistent before starting:
- `tuning/val_confirm_e2e_per_question.csv` -- 639 data rows.
- `tuning/index_cache_val_confirm_e2e/` -- 112 `.npz` files, all under config-hash
  `4edae64ed40256e3`.
- `tuning/frozen_state.json` frozen config unchanged (no family run since commit
  `6f36f08` touches `frozen_state.json`'s `frozen` block).

## Step 1 -- overlap_status x Acc@QA cross-tab (from CSV alone)

| overlap_status | Acc@QA=True | Acc@QA=False | row total |
|---|---|---|---|
| cleared (IoP>=0.5) | 121 (18.94%) | 77 (12.05%) | 198 |
| partial (0<IoP<0.5) | 52 (8.14%) | 45 (7.04%) | 97 |
| zero (IoP==0) | 176 (27.54%) | 168 (26.29%) | 344 |

Consistency check: `cleared x Acc@QA=True` = 121/639 = 18.94%,
which exactly matches the reported Acc@GQA=0.1894 from commit `6f36f08` (Acc@GQA is by
definition `acc_qa AND iop>=0.5`, i.e. exactly this cell).

### Key group: Acc@QA=1 x overlap_status in {partial, zero}

**228 / 639 questions (35.68%)** get the multiple-choice answer right
but fail to ground it well enough for Acc@GQA -- of which 52 are "partial" overlap
and 176 are "zero" overlap (no intersection at all).

`gap_seconds` distribution for this group (0 seconds if IoP>0, i.e. overlap already exists
but is below the 0.5 threshold; otherwise the minimum edge-to-edge distance to the nearest
gold span):

- min = 0.000s
- median = 5.092s
- mean = 11.529s
- max = 76.213s

| gap bucket | count | % of key group |
|---|---|---|
| 0 (touching, IoP<0.5) | 52 | 22.81% |
| 0-2.2s | 29 | 12.72% |
| 2.2-5s | 30 | 13.16% |
| 5-15s | 52 | 22.81% |
| >15s | 65 | 28.51% |

The median gap (5.09s) is **more than double** `half_width_s=2.2`.
Only 29/228 (12.72%) fall in the
"near miss, gap < half_width" bucket where a wider window alone would plausibly close the
gap; the majority (117/228 =
51.32%) sit 5+ seconds away.

## Step 2 -- retrieval-only re-check (reused index cache, no LLM calls)

Re-ran retrieval only (embed question -> `_retrieve_with_l1` top-`l2_retrieve_top_k=4`
frames) for every question in the key group, loading each video's already-ingested index
from `tuning/index_cache_val_confirm_e2e/` under the frozen config. For each question,
checked whether **any** of the 4 retrieved frame timestamps falls inside **any** gold span.

**Subset A (Acc@QA=1 x overlap_status in {partial, zero}, n=228):**
- retrieved-in-gold = YES: 82 (35.96%)
- retrieved-in-gold = NO: 146 (64.04%)

**Subset B (all overlap_status=zero rows, any Acc@QA, n=344) -- reported separately since
pure retrieval failure matters regardless of whether the MC answer was right:**
- retrieved-in-gold = YES: 87 (25.29%)
- retrieved-in-gold = NO: 257 (74.71%)

## Step 3 -- synthesis into the three failure buckets

Crossing Step 1's gap distance against Step 2's retrieval-hit flag, over the 228-question
key group (Acc@QA=1 x overlap_status in {partial, zero}):

| bucket | definition | count | % |
|---|---|---|---|
| (a) widen-window | retrieved-in-gold=YES, gap<2.2s | 44 | 19.30% |
| (b) wrong-frame-anchored | retrieved-in-gold=YES, gap>=2.2s | 38 | 16.67% |
| (c) retrieval/ranking failure | retrieved-in-gold=NO | 146 | 64.04% |

**Dominant bucket: (c), at 64.04% of the failure group.**

## Conclusion

Bucket (c) -- retrieval never found a frame inside the gold span at all -- accounts for
roughly **two-thirds** of the Acc@QA-right/Acc@GQA-wrong gap, and remains the largest
single cause even restricted to `overlap_status=zero` alone regardless of Acc@QA
(74.71% of all zero-overlap questions had zero retrieved frames in
gold). Bucket (a), the "span is centered close but too narrow" case that would motivate
widening `half_width_s`, is only 19.30% of the failure group --
a real but minority effect. Bucket (b), where the CLIP anchor picked a worse frame out of
an otherwise-good retrieved pool, is smaller still (16.67%).

**This does not support "widen the window" or "fix span centering" as the primary next
fix.** The data points at retrieval/ranking itself as the dominant failure mode -- roughly
two-thirds of the time, no amount of span post-processing on the retrieved pool can recover
the answer, because the right neighborhood was never retrieved. This is consistent with
the codec_rank-dominance pattern flagged separately on branch
`siddanth/peak-source-a6-p1`; the two diagnostics point at the same underlying issue from
different angles. Any width/centering fix (bucket a+b = 35.96%
of the failure group) is a secondary, smaller-impact improvement on top of a retrieval fix,
not a substitute for one.
