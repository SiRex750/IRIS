# Part 3e -- Joint l2_retrieve_top_k x ppr_lambda x Span-Method Comparison

## 1. Setup

- Grid: `l2_retrieve_top_k` in {4, 5, 8, 12, 16} (Family 4 / part3c's grid)
  x `ppr_lambda` in {0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00} (Family 2's
  grid) x span method {A, B, C, D} -- **140 cells**, full `val_tune` set
  (2685 questions, 450 videos) on every cell, no sampling.
- Frozen upstream config: `retrieval_strategy="hybrid"`, `ppr_damping=0.50`
  (read live from `tuning/frozen_state.json`).
- Ingestion: `l2_retrieve_top_k` **is** in `INGEST_RELEVANT_KEYS`
  (changes the retrieved pool), `ppr_lambda` is not. This run therefore
  ingests once per K value (5 ingests) and reuses each K's cached index
  across all 7 lambda values, exactly mirroring part3d's lambda-reuse
  structure one level up. All 5 K-value caches were already fully
  populated in `tuning/index_cache_scenespans/` from part3c/part3d --
  **zero fresh ingests were needed for this entire 140-cell run**.

### The Method C bug, fixed first

Part 3d's report flagged Method C's 94.38% fallback rate as unreliable.
Two fixes landed before this sweep ran (both with regression tests,
confirmed to fail pre-fix / pass post-fix):

1. `iris/query.py`'s `_build_retrieved` PPR branch read
   `getattr(node, "scene_id", None)` off the graph node instead of the
   frame object -- but this branch only executes when
   `graph_mode="flat"`, which nothing in this project's default config
   actually uses.
2. **The real cause**: `iris/scene_retrieval.py`'s `_node_to_dict`
   (built by `retrieve_scene_sparse`'s DESCEND branch -- the branch this
   project's default `graph_mode="scene_sparse"` config actually takes on
   nearly every query) didn't include a `scene_id` key **at all**, so
   `predicted_span_from_frames_scene`'s `.get("scene_id")` always
   returned `None` and fell back to Method A's span on almost every
   question.

Post-fix, **Method C's fallback rate is 0.00% on all 140 cells** in this
sweep (verified explicitly -- every single row's
`method_c_fallback_rate` is `0.0`), and a K=4/lambda=0.50 sanity re-run
before launching the full grid showed Method C's numbers jump from
mIoP=0.2784/IoP@0.5=0.2242 (pre-fix, ~= Method A via fallback) to
mIoP=0.3038/IoP@0.5=0.3028 (post-fix, now close to Method D) -- the fix
materially changes Method C's standing, not just its fallback-rate
bookkeeping.

## 2. Best lambda per (K, span method), with bootstrap CIs

Percentile bootstrap (1000 resamples, n=2685) on each (K, method)'s top-3
lambda candidates by raw mIoP (`tuning/lambda_k_bootstrap_ci.json`,
`scripts/part3e_bootstrap_ci.py`). **Every single within-K lambda
comparison across all 20 (K, method) groups shows CI overlap** between
the raw winner and its runner-up(s) -- lambda is flat within every K,
exactly as found in the pure-lambda sweep (part3d). Summary (best lambda
per K per method; full comparisons in the JSON):

| K | Method A best | Method B best | Method C best | Method D best |
|---|---|---|---|---|
| 4 | lambda=0.90, mIoP=0.2831 | lambda=0.50, mIoP=0.3128 | lambda=0.50, mIoP=0.3038 | lambda=0.50, mIoP=0.3021 |
| 5 | lambda=1.00, mIoP=0.2796 | lambda=0.50, mIoP=0.3150 | lambda=0.50, mIoP=0.3027 | lambda=0.50, mIoP=0.3015 |
| 8 | lambda=0.25, mIoP=0.2703 | lambda=0.50, mIoP=0.3159 | lambda=0.25, mIoP=0.3060 | lambda=0.25, mIoP=0.3049 |
| 12 | lambda=0.10, mIoP=0.2652 | lambda=0.75, mIoP=0.3115 | lambda=0.10, mIoP=0.3059 | lambda=0.10, mIoP=0.3049 |
| 16 | lambda=0.00, mIoP=0.2622 | lambda=1.00, mIoP=0.3112 | lambda=0.10, mIoP=0.3045 | lambda=0.10, mIoP=0.3040 |

At every K, every method's own top-2/3 lambda values sit within ~0.005-0.015
mIoP of each other, well inside each candidate's ~0.015-wide 95% CI
half-width -- none of these per-K "best lambda" picks is statistically
distinguishable from its own runner-up.

## 3. Does K=4 still hold up?

**Yes, statistically -- no K beats K=4 with significance for any
method.** Cross-K comparison of each method's single best (K, lambda)
cell in the entire grid against K=4's own best cell:

| method | best (K, lambda) overall | mIoP | 95% CI | K=4 best (lambda) | mIoP | 95% CI | overlap |
|---|---|---|---|---|---|---|---|
| A | (4, 0.90) | 0.2831 | [0.2703, 0.2953] | (4, 0.90) -- same cell | -- | -- | -- |
| B | (8, 0.50) | 0.3159 | [0.3005, 0.3321] | (4, 0.50) | 0.3128 | [0.2963, 0.3291] | **True** |
| C | (8, 0.25) | 0.3060 | [0.2912, 0.3217] | (4, 0.50) | 0.3038 | [0.2884, 0.3192] | **True** |
| D | (12, 0.10) | 0.3049 | [0.2909, 0.3204] | (4, 0.50) | 0.3021 | [0.2869, 0.3169] | **True** |

Every method's overall raw-mIoP peak (which happens to land at K=8 or
K=12 for B/C/D) is within noise of that same method's K=4 cell -- **the
~0.002-0.003 mIoP gaps between K=4 and the raw-best K are an order of
magnitude smaller than the ~0.015-0.016-wide CI half-widths**. There is
no statistical case for moving K off 4 based on mIoP alone.

That said, section 4 shows K materially changes a *different* property
(Method B's zero-width degeneracy) that mIoP alone doesn't capture --
see the recommendation in section 6 for how that changes the picture.

## 4. Method B / D zero-width incidence across K and lambda

Method D: **0.00% zero-width incidence and 0.00% CLIP-anchor-fallback
rate on every one of the 140 cells** -- completely stable, exactly as in
the K=4-only part3d run. Its fixed `half_width_s=2.2` half-window
structurally prevents collapse regardless of K or lambda.

Method B: zero-width incidence falls sharply as K grows (more candidate
frames per cluster means less frequent single-frame clusters), and rises
mildly with lambda within a K:

| K \ lambda | 0.00 | 0.10 | 0.25 | 0.50 | 0.75 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|
| 4 | 22.8% | 25.0% | 27.6% | 29.8% | 30.5% | 30.7% | 31.2% |
| 5 | 19.4% | 20.4% | 21.0% | 22.0% | 22.2% | 21.8% | 22.5% |
| 8 | 12.8% | 12.5% | 12.4% | 10.0% | 9.0% | 8.8% | 9.2% |
| 12 | 7.0% | 7.3% | 6.5% | 4.3% | 3.0% | 3.0% | 3.3% |
| 16 | 4.1% | 3.9% | 3.5% | 1.6% | 1.1% | 1.3% | 1.4% |

At the currently-frozen K=4, nearly a third of Method B's predictions
are degenerate zero-width spans at high lambda -- this was the basis for
part3d's recommendation to prefer Method D over B despite B's higher raw
mIoP. **At K=16, that same rate drops to ~1-4%**, close to Method D's 0%
and no longer a dominant driver of B's numbers.

## 5. mIoU alongside mIoP -- does the "wins on precision, loses on
   coverage" pattern persist at higher K?

At lambda=0.50 (near-optimal for both B and D at every K):

| K | B mIoP | B mIoU | B zero-width | D mIoP | D mIoU | D zero-width |
|---|---|---|---|---|---|---|
| 4 | 0.3128 | 0.0720 | 29.8% | 0.3021 | 0.1639 | 0.0% |
| 5 | 0.3150 | 0.0828 | 22.0% | 0.3015 | 0.1634 | 0.0% |
| 8 | 0.3160 | 0.1147 | 10.0% | 0.3020 | 0.1642 | 0.0% |
| 12 | 0.3095 | 0.1407 | 4.3% | 0.3015 | 0.1634 | 0.0% |
| 16 | 0.3061 | 0.1545 | 1.6% | 0.2998 | 0.1629 | 0.0% |

Two things move in step: as K rises, Method B's zero-width rate falls
and its mIoU rises correspondingly (0.072 -> 0.155, more than doubling),
closing most (not all) of the gap to Method D's near-constant mIoU
(~0.163-0.164 at every K). **The mechanism identified in part3d --
zero-width spans mechanically inflating IoP against a metric that can't
penalize them, while IoU correctly does -- is confirmed directly**: as
the artifact shrinks with K, so does the mIoU gap. Even at K=16 (B's
healthiest cell), B's mIoU (0.1545) still trails D's (0.1629) by ~5%,
while B's mIoP lead over D shrinks to essentially nothing (0.3061 vs
0.2998, a ~2% gap, itself within bootstrap noise per section 2/3).
Method D is the more consistent, K-invariant answer at every K tested;
Method B only becomes competitive on mIoU once K is pushed well above
the currently-frozen value.

## 6. Recommendation

**Freeze `ppr_lambda=0.50` and `l2_retrieve_top_k=4` -- no change from
the current frozen values -- with span method D (peak-anchored) as the
next default.**

Reasoning:

1. **K: keep 4.** No K value beats K=4 with statistical significance for
   any method (section 3) -- every method's cross-K best-cell comparison
   against its own K=4 cell shows full CI overlap. Moving K purely for
   mIoP has no support in this data.
2. **Lambda: keep 0.50.** Confirmed again under the corrected Method C
   and across the full K grid (section 2): no lambda value is
   distinguishable from its neighbors under any (K, method) combination.
3. **Span method: D, not B, and this now generalizes across K.** At
   K=4, Method B's raw mIoP win is substantially inflated by zero-width
   spans (29.8% of its predictions, section 4) -- an IoP-metric artifact
   against a gold-span distribution with 4.4s median width, not genuine
   localization (confirmed in part3d). Pushing K up to 16 shrinks that
   artifact (to ~1.6%) and closes most of the mIoU gap, but even at its
   best K, Method B's mIoU still trails Method D's by ~5%, while its
   mIoP lead shrinks to statistical noise (section 5). **Since K=4 is
   staying frozen (point 1), and Method B needs K well above 4 to even
   partially escape its zero-width problem, Method D is the more
   trustworthy choice at the actual K value this system will run at.**
   Method D is also the only method whose numbers (mIoP, mIoU,
   zero-width rate, fallback rate) are essentially invariant to both K
   and lambda across the entire 140-cell grid -- the most robust,
   least-parameter-sensitive of the four.
4. **Method C, now fixed, is a legitimate second-place candidate but not
   the pick.** Post-fix its mIoP (0.30-0.31 range) sits between B and D
   at every K, with 0% fallback everywhere, but its mIoU (~0.10-0.11) is
   noticeably below Method D's (~0.16) at every K -- it inherits some of
   B's precision-without-coverage character since it also anchors on a
   single frame's local neighborhood (the scene boundary) rather than a
   fixed window.

**Not recommended, and why:** don't freeze Method B even at a higher K --
the current frozen K=4 stays, and B's advantage doesn't survive at K=4.
Don't move K to chase Method B's mIoU improvement -- the K=4-vs-higher-K
gap is statistically indistinguishable for every method's own best cell,
so there's no justified mIoP gain to offset the added complexity/cost of
a higher K, and D already delivers B's best-case mIoU at K=4 with zero
extra cost.

LAMBDA_K_SPAN_METHOD_COMPARISON_COMPLETE
