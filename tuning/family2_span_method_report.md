# Part 3d -- Joint ppr_lambda x Span-Method Comparison

## 1. Setup

- Grid: `ppr_lambda` in {0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00} (Family
  2's original grid) x span method in {A, B, C, D}, full `val_tune` set
  (2685 questions, 450 videos), **no sampling, no `--limit`**.
- Frozen upstream config, held fixed for the entire sweep:
  `retrieval_strategy="hybrid"`, `ppr_damping=0.50`, `l2_retrieve_top_k=4`
  (all read live from `tuning/frozen_state.json` at run time, not
  hardcoded). Only `ppr_lambda` and the span-construction method vary.
- `ppr_lambda` is confirmed **not** in `part3_tune.INGEST_RELEVANT_KEYS`,
  so all 450 video indexes were ingested **once** (reused the existing
  `tuning/index_cache_scenespans/` cache from the prior scene-spans work,
  0/450 needed fresh ingest) and shared across all 7 lambda values --
  no 7x-redundant decode pass.
- Sanity gate (Step 3): ran lambda=0.50 alone across all four methods
  first. Method A: mIoP=0.2790, IoP@0.5=0.2246 vs the historical Family-2
  record at lambda=0.5 (mIoP=0.27748, IoP@0.5=0.21527) -- within ~0.5-4%
  relative, consistent with the same config. Gate passed; full 7-lambda
  run launched.
- Script fixes made before running (both committed separately,
  `305f954`): (1) CSV rows are now flushed to
  `tuning/lambda_span_method_comparison.csv` after each lambda's four
  methods finish, not only at the end, so a crash mid-sweep can't lose
  completed cells; (2) added `--lambdas` to restrict the grid, used for
  the sanity pass.
- Correctness gate: `tests/test_span_methods.py` (Method D tests already
  present, covering the CLIP-anchor case, no-query-embedding fallback,
  and single-retrieved-frame clamping) plus every other test file
  touching `eval/metrics.py` / `iris/scene_retrieval.py`
  (`test_audit_fixes.py`, `test_scene_sparse_descend.py`) -- **61/61
  passed**, no fixes needed.

## 2. Full grid results (28 rows)

| lambda | method | mIoP | IoP@0.3 | IoP@0.5 | mIoU | IoU@0.3 | IoU@0.5 | n |
|---|---|---|---|---|---|---|---|---|
| 0.00 | A | 0.2714 | 0.3177 | 0.2317 | 0.1617 | 0.2216 | 0.0991 | 2685 |
| 0.00 | B | 0.2904 | 0.3102 | 0.2875 | 0.0631 | 0.0801 | 0.0332 | 2685 |
| 0.00 | C | 0.2708 | 0.3214 | 0.2313 | 0.1642 | 0.2272 | 0.1013 | 2685 |
| 0.00 | D | 0.2878 | 0.3572 | 0.2875 | 0.1560 | 0.2332 | 0.1237 | 2685 |
| 0.10 | A | 0.2734 | 0.3222 | 0.2302 | 0.1681 | 0.2294 | 0.1035 | 2685 |
| 0.10 | B | 0.2955 | 0.3173 | 0.2920 | 0.0643 | 0.0834 | 0.0324 | 2685 |
| 0.10 | C | 0.2727 | 0.3259 | 0.2298 | 0.1706 | 0.2350 | 0.1058 | 2685 |
| 0.10 | D | 0.2886 | 0.3561 | 0.2894 | 0.1556 | 0.2328 | 0.1203 | 2685 |
| 0.25 | A | 0.2788 | 0.3240 | 0.2358 | 0.1779 | 0.2369 | 0.1121 | 2685 |
| 0.25 | B | 0.3042 | 0.3270 | 0.3035 | 0.0682 | 0.0890 | 0.0380 | 2685 |
| 0.25 | C | 0.2782 | 0.3278 | 0.2354 | 0.1804 | 0.2425 | 0.1143 | 2685 |
| 0.25 | D | 0.2951 | 0.3654 | 0.2942 | 0.1600 | 0.2380 | 0.1240 | 2685 |
| **0.50** | A | 0.2790 | 0.3345 | 0.2246 | 0.1928 | 0.2615 | 0.1244 | 2685 |
| **0.50** | **B** | **0.3128** | 0.3386 | **0.3132** | 0.0720 | 0.0942 | 0.0447 | 2685 |
| **0.50** | C | 0.2784 | 0.3382 | 0.2242 | 0.1954 | 0.2670 | 0.1266 | 2685 |
| **0.50** | **D** | 0.3021 | 0.3728 | 0.3028 | 0.1639 | 0.2440 | 0.1278 | 2685 |
| 0.75 | A | 0.2812 | 0.3345 | 0.2186 | 0.1965 | 0.2600 | 0.1203 | 2685 |
| 0.75 | B | 0.3076 | 0.3326 | 0.3091 | 0.0702 | 0.0909 | 0.0428 | 2685 |
| 0.75 | C | 0.2806 | 0.3382 | 0.2183 | 0.1991 | 0.2656 | 0.1225 | 2685 |
| 0.75 | D | 0.2996 | 0.3702 | 0.2980 | 0.1628 | 0.2425 | 0.1281 | 2685 |
| 0.90 | A | 0.2831 | 0.3378 | 0.2197 | 0.1979 | 0.2648 | 0.1196 | 2685 |
| 0.90 | B | 0.3050 | 0.3281 | 0.3065 | 0.0702 | 0.0920 | 0.0425 | 2685 |
| 0.90 | C | 0.2825 | 0.3415 | 0.2194 | 0.2004 | 0.2704 | 0.1218 | 2685 |
| 0.90 | D | 0.2993 | 0.3698 | 0.2987 | 0.1619 | 0.2417 | 0.1248 | 2685 |
| 1.00 | A | 0.2825 | 0.3367 | 0.2186 | 0.1955 | 0.2618 | 0.1166 | 2685 |
| 1.00 | B | 0.3069 | 0.3300 | 0.3091 | 0.0707 | 0.0927 | 0.0436 | 2685 |
| 1.00 | C | 0.2819 | 0.3404 | 0.2183 | 0.1980 | 0.2674 | 0.1188 | 2685 |
| 1.00 | D | 0.2994 | 0.3683 | 0.2998 | 0.1617 | 0.2406 | 0.1248 | 2685 |

(Bold = current frozen lambda / the two methods that lead at every lambda.)

## 3. Best lambda per span method, with bootstrap CIs

Percentile bootstrap (1000 resamples over the n=2685 questions) run on
each method's top-3 lambda candidates by raw mIoP
(`tuning/lambda_bootstrap_ci.json`, `scripts/part3d_bootstrap_ci.py`):

| method | best lambda (raw) | mIoP | 95% CI | runner-up | mIoP | 95% CI | distinguishable? |
|---|---|---|---|---|---|---|---|
| A | 0.90 | 0.2831 | [0.2712, 0.2960] | 1.00 | 0.2825 | [0.2694, 0.2957] | **No** -- CIs overlap almost completely |
| B | 0.50 | 0.3128 | [0.2962, 0.3299] | 0.75 | 0.3076 | [0.2913, 0.3241] | **No** |
| C | 0.90 | 0.2825 | [0.2693, 0.2947] | 1.00 | 0.2819 | [0.2693, 0.2947] | **No** -- CIs are nearly identical |
| D | 0.50 | 0.3021 | [0.2879, 0.3167] | 0.75 | 0.2996 | [0.2848, 0.3136] | **No** |

**Every method's raw lambda winner is statistically indistinguishable
from its runner-up(s)** -- the ~0.005-0.010 mIoP gaps that separate the
top 2-3 lambda values per method are well inside each candidate's own
~0.012-0.015-wide 95% CI half-width. Lambda is effectively flat across
the tested grid once you're anywhere in the [0.5, 1.0] neighborhood, for
every span method. This is consistent with the original Family 2 sweep
(section `selection_detail_ppr_lambda` in `frozen_state.json`), which
already showed a similarly shallow mIoP curve across the same grid.

Conveniently, **lambda=0.50 -- the already-frozen value -- is the raw
top-3 for every method except A** (top-1 for B and D, not in A/C's top-3
where 0.90/1.00 lead marginally, but A/C's own top-3 members aren't
distinguishable from 0.50 either given the CI widths above). There is no
statistical case for moving lambda off its current frozen value.

## 4. Does 0.25 still win under any method?

**No.** 0.25 does not appear in the top-3 lambda candidates for any of
the four span methods (top-3 lists above are drawn from {0.5, 0.75, 0.9,
1.0} exclusively). Looking at the full grid in section 2, 0.25's mIoP is
the 4th-or-5th-best value out of 7 for every method -- solidly mid-pack,
never a contender.

This confirms the hypothesis in `part3d_lambda_span_method_comparison.py`'s
docstring: the original Family 2 selection of `ppr_lambda=0.25`
(`selection_detail_ppr_lambda` in `frozen_state.json`) was an artifact of
scoring with Method A's min-max span under the *pre-anchor-fix* CLIP
similarity code, not a real property of low lambda. With the anchor bug
fixed and three additional span methods added, low lambda has no
advantage anywhere in this grid -- mIoP rises roughly monotonically from
lambda=0.0 to lambda=0.75-0.90 for every method, then flattens.

## 5. Method C's fallback rate, and Method B/D's zero-width incidence

| lambda | Method C fallback rate | Method B zero-width rate | Method D zero-width rate | Method D CLIP-anchor fallback rate |
|---|---|---|---|---|
| 0.00 | 94.38% | 22.79% | 0.00% | 0.00% |
| 0.10 | 94.38% | 24.95% | 0.00% | 0.00% |
| 0.25 | 94.38% | 27.64% | 0.00% | 0.00% |
| 0.50 | 94.38% | 29.83% | 0.00% | 0.00% |
| 0.75 | 94.38% | 30.50% | 0.00% | 0.00% |
| 0.90 | 94.38% | 30.65% | 0.00% | 0.00% |
| 1.00 | 94.38% | 31.25% | 0.00% | 0.00% |

**Two findings here materially change the recommendation:**

**(a) Method C's fallback rate is suspiciously high (94.38%, constant
across lambda) and its numbers should not be trusted in this run.** In
part3c (K-grid comparison, no CLIP query-embedding threaded through),
Method C's fallback rate was 5.62%. The jump to 94.38% traces to a real
bug, not a genuine property of Method C: `iris/query.py`'s `_build_retrieved`
(the default L2/hybrid PPR path -- taken every time here since
`use_l1=False` by default) populates each retrieved frame's `scene_id` via
`getattr(node, "scene_id", None)` where `node` is the PPR graph node
object, which never carries a `scene_id` attribute -- so this is `None`
for essentially every retrieved frame, regardless of lambda or CLIP
anchoring. (The L1-cache retrieval path a few hundred lines later in the
same file correctly reads `getattr(fr, "scene_id", -1)` off the frame
object -- the bug is specific to the non-L1 path used by this run's
config.) With `scene_id` almost always `None`, Method C falls back to
Method A's min-max span on nearly every question, which is exactly why
Method A and Method C's mIoP/IoU numbers in section 2 are near-identical
at every single lambda (e.g. lambda=0.50: A mIoP=0.2790 vs C
mIoP=0.2784). **Method C's true localization quality is not evaluated by
this run at all -- it needs `iris/query.py`'s `scene_id` bug fixed and a
re-run before it can be fairly compared.** This is out of scope for this
task's script/test changes (not in `eval/metrics.py` or
`iris/scene_retrieval.py`, not something Step 2's ingest-efficiency check
touches), so it's flagged here as follow-up rather than fixed inline.

**(b) Method B's zero-width-span incidence climbs steadily with lambda,
from 22.8% at lambda=0.0 to 31.2% at lambda=1.0 -- nearly a third of its
predictions at high lambda are single-instant spans.** Gold spans in
`val_tune` have a median width of 4.4s (p10=1.8s, p90=14.3s) --
essentially none are point-events, so a zero-width prediction landing
inside a gold window is a metric-gaming artifact of the IoP formula
(a zero-width span nested inside any interval scores IoP=1.0 by
construction) rather than genuine localization. **This directly explains
Method B's mIoU collapse in section 2** (mIoU=0.06-0.07, roughly a third
of Method A/D's mIoU at the same lambda) -- Method B is winning on mIoP
partly by exploiting this metric artifact, not by covering the gold
window better. Method D has **zero** zero-width incidence at every
lambda (its fixed `half_width_s=2.2` half-window structurally prevents
collapse) and a CLIP-anchor fallback rate of 0.00% everywhere (the query
embedding was always available and non-degenerate in this run) -- it
gets a meaningfully higher mIoU than B (0.16 vs 0.07 at lambda=0.5) while
trailing B's mIoP by only ~0.01.

## 6. mIoP vs mIoU: the "wins on precision, loses on coverage" pattern

Restating section 2's mIoU column alongside mIoP makes the pattern
explicit (values at lambda=0.50, the frozen/leading value):

| method | mIoP | mIoU | mIoU as % of mIoP |
|---|---|---|---|
| A | 0.2790 | 0.1928 | 69% |
| B | 0.3128 | 0.0720 | **23%** |
| C | 0.2784 | 0.1954 | 70% (unreliable -- see section 5a) |
| D | 0.3021 | 0.1639 | 54% |

Method B leads on mIoP by the widest margin of any method pair in this
comparison, but its mIoU is by far the lowest of the four -- less than a
quarter of its own mIoP, and roughly a third of Method A/C's mIoU. This
is the pattern flagged in the task brief, now traced to a concrete
mechanism (section 5b): Method B increasingly collapses to zero-width
spans as lambda grows, which the IoP metric can't penalize but IoU
correctly does. Method D sits between A and B -- higher mIoP than A,
substantially higher mIoU than B, with none of B's degenerate-span
mechanism.

## 7. Recommendation

**Freeze `ppr_lambda=0.50` (no change from the current frozen value) with
span method D (peak-anchored, `predicted_span_from_frames_peak`) as the
next default -- not Method B, despite Method B's higher raw mIoP.**

Reasoning:

1. **Lambda: keep 0.50.** No lambda value is statistically distinguishable
   from its neighbors under any method (section 3) -- moving off the
   current frozen value has no support in this data, and 0.25 (the old
   Family 2 pick) is decisively out of contention now that the anchor
   confound is gone (section 4).
2. **Span method: D over B.** Method B's ~0.01 mIoP lead over D at every
   lambda is real in the raw numbers, but roughly a third of B's
   predictions at the frozen lambda are zero-width single-instant spans
   against a gold-span distribution with a 4.4s median width -- that's a
   known IoP-metric artifact, not genuine localization improvement
   (section 5b, 6). Method D achieves nearly the same mIoP with zero
   zero-width incidence, a 0% CLIP-anchor fallback rate, and more than
   double Method B's mIoU. It's the more trustworthy number even though
   it isn't the raw mIoP leader.
3. **Method C is not a candidate this round.** Its numbers in this run
   are an artifact of a 94.38% fallback-to-Method-A rate caused by a
   `scene_id`-attribution bug in `iris/query.py`'s default (non-L1)
   retrieval path (section 5a). It should be re-evaluated only after that
   bug is fixed and the sweep re-run -- it may or may not beat D once its
   actual scene-lookup mechanism is exercised instead of falling back
   ~19 times out of 20.

**Follow-up work, not blocking this freeze decision:** fix the `scene_id`
attribution bug in `iris/query.py` (`_build_retrieved`'s PPR branch reads
`node.scene_id` instead of `fr.scene_id`) and re-run this comparison to
get a fair read on Method C; consider whether Method B's zero-width
spans are ever legitimate (very short gold events) worth special-casing
rather than treating uniformly as an artifact, before ruling it out
long-term.

LAMBDA_SPAN_METHOD_COMPARISON_COMPLETE
