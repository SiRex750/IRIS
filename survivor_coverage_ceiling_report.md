# PART A -- L1 Survivor-Coverage Ceiling (diagnostic only)

**Not an official NExT-GQA metric** -- listed in `metric_registry.json` as
`Survivor_coverage_ceiling`, `is_official: false`. Read-only: this run did
not touch, modify, or depend on `tuning/all_trials.csv`,
`tuning/frozen_state.json`, or anything Family 3 (PART B) writes -- it
reused the same cached indexes Family 1/2 already built under
`retrieval_strategy="hybrid"` (config-hash `cab2bac1628012a3`), no
re-ingest, no GPU, no PPR, no captioning/answering.

## Setup

- Dataset: val_tune, 450 videos / **2,685 questions**.
- L1 config: `retrieval_strategy="hybrid"` (Family 1's frozen result) --
  the same admission behavior the tuning families are building on, not a
  different/default setting.
- "Survived" = at least one of that video's L1-admitted frame timestamps
  falls inside the question's gold span (max-overlap gold span used when
  multiple exist, same rule as the official metric).

## Headline number

**Ceiling = 2,681 / 2,685 = 99.85%.**

## Breakdown by question type

| family | rate | n |
|---|---|---|
| causal (CW/CH) | 99.81% | 1,564 |
| temporal (TN/TC/TP) | 99.91% | 1,121 |

Essentially flat across question type -- no meaningful causal/temporal gap.

## Breakdown by gold-span length

(Threshold: short ≤2.5s, long >2.5s -- this project's existing convention,
`smoke/smoke_report.md`.)

| bucket | rate | n |
|---|---|---|
| short (≤2.5s) | 99.18% | 488 |
| long (>2.5s) | 100.00% | 2,197 |

All 4 failures fall in the short-span bucket (short spans are 18% of the
question set but 100% of the failures).

## Failure cluster (task 5)

All 4 failing questions, in full:

| video_id | qid | gold_span_length_s | total_frames | survivor_count | retention_pct |
|---|---|---|---|---|---|
| 7164729910 | 8 | 0.4 | 656 | 88 | 13.415% |
| 7001078933 | 2 | 0.6 | 633 | 86 | 13.586% |
| 7164729910 | 5 | 0.4 | 656 | 88 | 13.415% |
| 4838145161 | 7 | 0.4 | 1921 | 257 | 13.378% |

**This does NOT cluster around aggressively-pruned videos.** Failure-case
median retention is 13.415%, success-case median retention is 13.404% --
essentially identical, and both sit right at the typical retention level
for this dataset. What the 4 failures actually share is **gold span length
(0.4-0.6s, the shortest in the entire failing set)** -- these are windows
so brief that even a well-retained, representatively-sampled frame set can
simply miss landing a frame inside them by chance, independent of how much
of the video was kept. This is a sampling-density-vs-window-width problem,
not an over-pruning problem.

## What this means for Families 1-2's ~0.28 mIoP plateau

**The ceiling is 99.85%, i.e. L1 admission is not the bottleneck.** With
`retrieval_strategy="hybrid"` (Family 1's frozen choice), essentially every
question's correct answer window already has at least one admitted frame
sitting inside it before Layer 2 ever ranks anything. The ~0.28 mIoP
plateau Families 1-2 have shown is **not explained by frames failing to
survive L1** -- fixing L1 admission further could realistically move at
most ~0.15 percentage points of ceiling (2,681→2,685 possible), nowhere
near the gap between 99.85% survivor coverage and the ~28% mIoP the
pipeline is actually achieving. The real gap lives downstream: **Layer 2's
PPR ranking (which of the ~13% retained frames gets selected into the
final top-k) and/or Layer 3, not Layer 1 admission.** No amount of further
`ppr_damping`/`l2_retrieve_top_k`/peak-sensitivity tuning on the admission
side can close a gap that measurement shows isn't there -- the tuning
families in progress (2 onward) are correctly targeting the actual
bottleneck (ranking quality within an already-adequate candidate pool),
not a phantom admission problem.

## Outputs

- `survivor_coverage_ceiling.csv` -- per-question record (video_id, qid,
  question_type, gold_span_length_s, span_bucket, survived, total_frames,
  survivor_count, retention_pct).
- This report.

SURVIVOR_COVERAGE_CEILING_MEASURED
