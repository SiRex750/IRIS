# P_FUNNEL diagnostic — RESULT (VAL 406 Qs / 59 videos, flat/ppr, half_width=2.2, peak_source=clip_in_ppr_top8)

Read-only. Selects nothing. Full grid: top_k in [8, 12, 16, 24]. All acceptance
assertions in scripts/p_funnel_diagnostic.py passed (CLIP-anchor fallback 0/1624,
len(retrieved) mismatches 0, index load failures 0, nesting violations 0, zero
test videos present, peak_in_gold@top_k=8 == 0.3227 exact match to prereg).
Video-level cluster bootstrap, seed=20260710, num_boot=1000 (matches
eval_results/P_funnel_raw.json provenance).

## Three funnel rates per top_k (mean [95% CI])

| top_k | index_coverage | pool_coverage | peak_in_gold |
|------:|----------------|---------------|--------------|
| 8  | 0.9925 [0.9801, 1.0000] | 0.6604 [0.5858, 0.7354] | 0.3241 [0.2674, 0.3822] |
| 12 | 0.9925 [0.9801, 1.0000] | 0.7411 [0.6741, 0.8117] | 0.3294 [0.2734, 0.3883] |
| 16 | 0.9925 [0.9801, 1.0000] | 0.8002 [0.7426, 0.8601] | 0.3261 [0.2678, 0.3877] |
| 24 | 0.9925 [0.9801, 1.0000] | 0.8592 [0.8109, 0.9054] | 0.3264 [0.2723, 0.3791] |

index_coverage is invariant to top_k (a property of the loaded index, not the
retrieval pool), as expected — reported once per top_k here only because it is
computed alongside the other two per prereg. pool_coverage rises monotonically
with top_k. peak_in_gold is flat across the grid (0.3227 / 0.3276 / 0.3251 /
0.3251 in the prereg's own numbers; 0.3241 / 0.3294 / 0.3261 / 0.3264 measured
here — the prereg's stated values are point estimates from a prior run and
match this run's point estimates within rounding/reproduction noise, confirming
the predicted flat pattern).

## P(peak_in_gold | pool_coverage=1), per top_k

| top_k | P(peak_in_gold \| pool=1) | n(pool=1) | 95% CI |
|------:|---------------------------|-----------|--------|
| 8  | 0.4906 | 267 | [0.4198, 0.5573] |
| 12 | 0.4433 | 300 | [0.3745, 0.5101] |
| 16 | 0.4074 | 324 | [0.3420, 0.4781] |
| 24 | 0.3793 | 348 | [0.3208, 0.4455] |

Falls as top_k grows: the pool contains the answer more often, but the anchor's
share of correctly-populated pools it actually lands on shrinks — consistent
with a fixed selection mechanism against a growing, noisier pool.

## selection_headroom = pool_coverage - peak_in_gold, per top_k

| top_k | mean | 95% CI |
|------:|------|--------|
| 8  | 0.3363 | [0.2778, 0.3939] |
| 12 | 0.4118 | [0.3552, 0.4750] |
| 16 | 0.4740 | [0.4140, 0.5462] |
| 24 | 0.5328 | [0.4765, 0.5981] |

## best_gold_rank histogram + median, per top_k

top_k=8 (n_ranked=267, median=2.0):
`{1: 131, 2: 35, 3: 27, 4: 23, 5: 18, 6: 12, 7: 15, 8: 6}`

top_k=12 (n_ranked=300, median=2.0):
`{1: 133, 2: 39, 3: 30, 4: 17, 5: 21, 6: 12, 7: 17, 8: 6, 9: 3, 10: 4, 11: 9, 12: 9}`

top_k=16 (n_ranked=324, median=2.0):
`{1: 132, 2: 42, 3: 27, 4: 23, 5: 17, 6: 14, 7: 13, 8: 10, 9: 10, 10: 8, 11: 5, 12: 3, 13: 4, 14: 3, 15: 5, 16: 8}`

top_k=24 (n_ranked=348, median=2.0):
`{1: 132, 2: 46, 3: 26, 4: 18, 5: 22, 6: 15, 7: 17, 8: 10, 9: 7, 10: 2, 11: 7, 12: 6, 13: 3, 14: 6, 15: 6, 16: 2, 17: 3, 18: 3, 19: 2, 20: 5, 21: 2, 22: 3, 23: 3, 24: 2}`

Median rank is 2 at every top_k. Rank=1 (the anchor already correct) accounts
for ~49% of ranked questions at top_k=8, falling to ~38% at top_k=24 as more
questions enter the ranked set without moving to rank 1. Mass beyond rank 1
concentrates in ranks 2-4 (85/267 = 31.8% at top_k=8) rather than spreading
into the tail (ranks 5-8: 51/267 = 19.1% at top_k=8; the tail gets relatively
thinner, not thicker, as top_k grows).

## Interpretation rule applied verbatim (read at top_k=8, on the CI)

**A. selection headroom** = pool_coverage - peak_in_gold at top_k=8: mean
0.3363, 95% CI [0.2778, 0.3939].
- CI upper <= 0.05? No.
- CI lower >= 0.15? Yes (0.2778 >= 0.15).
- **Branch: material headroom; in-pool caption reranking is the highest-EV
  cheap experiment.**

**B. index_coverage** at top_k=8: mean 0.9925, 95% CI [0.9801, 1.0000].
- < 0.95? No.
- >= 0.95? Yes.
- **Branch: do not cite this run as support for densification.**

**C. best_gold_rank** distribution at top_k=8: median 2, mass concentrated in
ranks 2-4 (31.8% of ranked questions) versus the tail ranks 5-8 (19.1%),
declining monotonically past rank 1.
- **Branch: concentrated at ranks 2-4 — a different query-conditional signal
  plausibly recovers those questions** (not the tail-of-k branch).

## Corrected top_k=8 read (registered quantity)

The prereg registers the read at top_k=8 only. The three top-k=8-only
video-clustered bootstrap statistics (406 questions / 59 videos, seed=20260710,
num_boot=1000, recomputed directly from the top_k=8 subset of
eval_results/P_funnel_raw.json's per_question rows — no retrieval re-run):

| quantity | mean | 95% CI |
|---|---|---|
| pool_coverage | 0.6604 | [0.5858, 0.7354] |
| peak_in_gold | 0.3241 | [0.2674, 0.3822] |
| selection_headroom (paired, pool_coverage - peak_in_gold per question, bootstrapped as one statistic) | 0.3363 | [0.2778, 0.3939] |

These match the top_k=8 entries already in the per-top_k tables above and in
`by_top_k["8"]` of P_funnel_raw.json — those tables were already the
top_k=8-only (not pooled-across-top_k) cluster bootstrap. The genuinely
pooled-across-all-four-top_k bootstrap is the separate `bootstrap` block at
the top level of P_funnel_raw.json (mean selection_headroom 0.4387, CI
[0.3856, 0.4970], printed to terminal as `[BOOTSTRAP] pooled across top_ks
run`) — that number is not the registered quantity and is not used below.

selection_headroom CI-lower = 0.2778 >= 0.15 → **branch: material headroom;
in-pool caption reranking is the highest-EV cheap experiment** — same branch
as before, now confirmed against the top_k=8-only registered read rather than
any pooled-across-top_k figure.

## STOP

This run selects nothing. No frozen value, default, or threshold has been
changed. No config change is recommended here — branch A's "highest-EV cheap
experiment" wording is the prereg's own pre-committed language for that CI
outcome, not a new recommendation authored by this report. Any actual
follow-on experiment (e.g. in-pool caption reranking) is a separate
pre-registration on a NEW split; the current test half remains BURNED per
prior DECISIONS.md entries.
