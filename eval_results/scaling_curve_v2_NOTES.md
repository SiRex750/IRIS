# Scaling curve v2 — what this run does and does not claim

**SUPERSEDED by `eval_results/scaling_curve_v2_textquery.md`**: synthetic
queries measured a scene-centroid shortcut path real text queries never take
(`shortcut_pct` 4-24% vs 0%). Retained for provenance.

Human-readable summary of `eval_results/scaling_curve_v2.{md,json}`. Full
detail lives in the linked report; this is the quick-reference record.

## Headline fitted exponents (UCF-only fit)

Fit is `log(median_total_retrieval_s) = k·log(N) + c`, least-squares on UCF
points only (VIRAT excluded from the fit — different dataset, marked
separately, see below).

- **flat**: k = 2.086, fit on the 4 UCF points that completed within the
  3600s cap (N = 91, 212, 1,102, 2,963). Consistent with O(N²) PPR over a
  fully-connected graph.
- **scene_sparse**: k = 0.663, fit on all 6 UCF points (N = 91 through
  13,506, all completed). Sub-quadratic, sub-linear.

## Fair-construction verification

Tested on Abuse042 (N=2,963): flat built from frames already loaded by the
scene_sparse ingest (this run's default) vs. flat built from an independent,
from-scratch ingest. Same survivor count, same edge count, 5.6% latency
difference — inside the run's normal noise band.

**Verdict: MATCH.** The shared-frames flat construction used throughout this
curve is not biased in flat's favor; no methodology change needed.

## The frontier is wall-time, not memory

Under a uniform 3600s / 29GB watchdog applied identically to every flat
measurement (including VIRAT, unlike v1), flat never hits the memory cap at
any N tested up to N=13,506. Arrest047 (N=6,559) and Arson019 (N=13,506)
both time out at 3600s with peak RSS still climbing (17.0GB and 26.9GB
respectively) — short of the 29GB cap. Abuse042 (N=2,963), which v1 reported
as failed under its old 480s budget, completes in 793s under the uniform
cap. The apparent "flat fails past N≈3k" result in v1 was a budget artifact,
not a hardware infeasibility — corrected here.

## VIRAT: a marked cross-dataset corroborator, not part of the fit

VIRAT (N=4,892) is plotted separately and excluded from both fitted
exponents because it's a different video with different codec history and
scene granularity. It's reported because its flat latency (7.9916s) lands
within 0.05% of what the UCF-only k=2.086 fit predicts at N=4,892 —
one data point of agreement across datasets, not a replication.

## Standing caveats

1. **Synthetic queries, not text.** All 50 queries/clip are seeded samples
   of survivor CLIP embeddings (`query_embed_s` is null throughout) — this
   curve measures retrieval graph mechanics, not end-to-end text-query
   latency.
2. **scene_sparse's own ingest cost is separate and is O(N²).** This curve
   is about per-query retrieval latency after ingest completes, not ingest
   throughput. A block-diagonal ingest optimization is future work, not
   attempted here.
3. **Mixed-dataset curve; UCF and VIRAT are not pooled.** See VIRAT note
   above — kept separate deliberately.

## Superseded: v1

`eval_results/scaling_curve.{json,md}` is kept in the repo, labelled as the
superseded, budget-confounded run (480s watchdog for UCF arms, VIRAT flat
point reused uncapped) so the correction from v1 to v2 is on the record
rather than silently overwritten.
