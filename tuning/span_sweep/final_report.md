# Adaptive span-construction sweep (Methods D/E/F) -- final report

**Generated:** 2026-07-28, on `worker-1` (`/home/ccbd/IRIS-1`), branch `feat/prerun-fixes`

## 1. Does adaptive width beat fixed width?

**No.** No cell -- across all 50 (Method D: 8, Method E: 30, Method F: 12)
-- clears the pre-registered acceptance criteria against the frozen
baseline `D@half_width_s=2.2`. Exactly zero E or F cells even improve
**both** mIoP and mIoU simultaneously (criterion 1) against the frozen
baseline on point estimate alone. **Recommendation: keep Method D at the
frozen `half_width_s=2.2`.**

A secondary, less meaningful comparison is required by this task's own
spec (compare against "the best Method D cell" by the stated
mIoP-primary/IoP@0.5-tiebreak selection rule, not necessarily the frozen
one) -- see section 4 for why that literal "best D" cell (`half_width_s=0.5`)
is itself a degenerate corner solution, not a meaningful baseline. Against
that corner, 10 Method-E cells improve both metrics on point estimate, but
**all 10 fail the bootstrap significance criterion** (95% CI on the paired
mIoP difference does not exclude zero for any of them). No cell passes all
four acceptance criteria against either anchor.

## 2. Reproduction gate and S1's isolated effect

**PASSED**, exact match (see `reproduction_gate.json`):

| metric | expected | actual |
|---|---|---|
| n | 2685 | 2685 |
| Gold@4 | 0.5303538175046555 | 0.5303538175046555 |
| mIoP | 0.29781971 | 0.297819714053369 |
| mIoU | 0.16086580 | 0.1608658073672957 |
| IoP@0.5 | 0.30093 | 0.30093109869646184 |

**S1's isolated effect on val_tune** (Method D @ hw=2.2, `duration_s` passed
vs omitted): mIoP +0.006959, mIoU +0.003823, IoP@0.5 +0.005214, IoU@0.5
+0.006331. The `duration_s` clamp fired (changed the span's `hi` bound) on
221/2685 spans. This is the first measurement of S1's effect on val_tune;
previously it was known only on val_confirm (+0.0048 mIoP / +0.0047 IoP@0.5
there, clamp firing on 46/639 spans -- consistent in direction and
comparable magnitude with this val_tune result).

**Duration source**: `eval/data/nextqa/gsub_val.json`'s per-video
`duration` field, confirmed available for all 450 val_tune videos before
the sweep began.

## 3. Full 50-cell grid

| cell | method | params | mIoP | mIoU | IoP@0.3 | IoP@0.5 | IoU@0.3 | IoU@0.5 | zero_w | clip_fb | degen_fb | mean_w | median_w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | D | {"half_width_s": 0.5} | 0.3123 | 0.0564 | 0.3326 | 0.3155 | 0.0462 | 0.0123 | 0.000 | 0.000 | - | 0.98 | 1.00 |
| 2 | D | {"half_width_s": 0.8} | 0.3109 | 0.0831 | 0.3423 | 0.3155 | 0.0991 | 0.0305 | 0.000 | 0.000 | - | 1.56 | 1.60 |
| 3 | D | {"half_width_s": 1.1} | 0.3097 | 0.1059 | 0.3512 | 0.3136 | 0.1445 | 0.0529 | 0.000 | 0.000 | - | 2.14 | 2.20 |
| 4 | D | {"half_width_s": 1.5} | 0.3079 | 0.1313 | 0.3628 | 0.3128 | 0.1929 | 0.0872 | 0.000 | 0.000 | - | 2.89 | 3.00 |
| 5 | D | {"half_width_s": 2.2} (**frozen**) | 0.3045 | 0.1646 | 0.3754 | 0.3061 | 0.2469 | 0.1259 | 0.000 | 0.000 | - | 4.18 | 4.40 |
| 6 | D | {"half_width_s": 3.0} | 0.2994 | 0.1881 | 0.3918 | 0.2890 | 0.2849 | 0.1479 | 0.000 | 0.000 | - | 5.62 | 6.00 |
| 7 | D | {"half_width_s": 4.0} | 0.2920 | 0.2039 | 0.4004 | 0.2652 | 0.3192 | 0.1505 | 0.000 | 0.000 | - | 7.36 | 8.00 |
| 8 | D | {"half_width_s": 6.0} | 0.2773 | 0.2165 | 0.3818 | 0.2182 | 0.3274 | 0.1426 | 0.000 | 0.000 | - | 10.64 | 12.00 |
| 9 | E | {"tau": 0.3, "w_min": 0.6} | 0.2841 | 0.2129 | 0.3743 | 0.2447 | 0.3140 | 0.1546 | 0.000 | 0.000 | 0.0 | 11.34 | 10.50 |
| 10 | E | {"tau": 0.3, "w_min": 1.0} | 0.2841 | 0.2132 | 0.3743 | 0.2447 | 0.3147 | 0.1546 | 0.000 | 0.000 | 0.0 | 11.34 | 10.50 |
| 11 | E | {"tau": 0.3, "w_min": 1.4} | 0.2839 | 0.2134 | 0.3743 | 0.2447 | 0.3143 | 0.1546 | 0.000 | 0.000 | 0.0 | 11.34 | 10.50 |
| 12 | E | {"tau": 0.3, "w_min": 1.8} | 0.2837 | 0.2133 | 0.3743 | 0.2447 | 0.3147 | 0.1538 | 0.000 | 0.000 | 0.0 | 11.34 | 10.50 |
| 13 | E | {"tau": 0.3, "w_min": 2.2} | 0.2835 | 0.2136 | 0.3750 | 0.2447 | 0.3151 | 0.1546 | 0.000 | 0.000 | 0.0 | 11.35 | 10.50 |
| 14 | E | {"tau": 0.4, "w_min": 0.6} | 0.2909 | 0.1992 | 0.3780 | 0.2592 | 0.2920 | 0.1464 | 0.000 | 0.000 | 0.0 | 9.39 | 8.00 |
| 15 | E | {"tau": 0.4, "w_min": 1.0} | 0.2909 | 0.1999 | 0.3780 | 0.2592 | 0.2935 | 0.1467 | 0.000 | 0.000 | 0.0 | 9.40 | 8.00 |
| 16 | E | {"tau": 0.4, "w_min": 1.4} | 0.2906 | 0.2004 | 0.3780 | 0.2588 | 0.2939 | 0.1467 | 0.000 | 0.000 | 0.0 | 9.41 | 8.00 |
| 17 | E | {"tau": 0.4, "w_min": 1.8} | 0.2894 | 0.2004 | 0.3780 | 0.2574 | 0.2942 | 0.1467 | 0.000 | 0.000 | 0.0 | 9.42 | 8.00 |
| 18 | E | {"tau": 0.4, "w_min": 2.2} | 0.2890 | 0.2013 | 0.3780 | 0.2562 | 0.2957 | 0.1493 | 0.000 | 0.000 | 0.0 | 9.44 | 8.00 |
| 19 | E | {"tau": 0.5, "w_min": 0.6} | 0.3005 | 0.1782 | 0.3702 | 0.2827 | 0.2596 | 0.1300 | 0.000 | 0.000 | 0.0 | 7.09 | 5.31 |
| 20 | E | {"tau": 0.5, "w_min": 1.0} | 0.3004 | 0.1795 | 0.3713 | 0.2831 | 0.2611 | 0.1304 | 0.000 | 0.000 | 0.0 | 7.11 | 5.31 |
| 21 | E | {"tau": 0.5, "w_min": 1.4} | 0.2995 | 0.1808 | 0.3702 | 0.2823 | 0.2633 | 0.1307 | 0.000 | 0.000 | 0.0 | 7.14 | 5.31 |
| 22 | E | {"tau": 0.5, "w_min": 1.8} | 0.2987 | 0.1820 | 0.3713 | 0.2812 | 0.2667 | 0.1318 | 0.000 | 0.000 | 0.0 | 7.18 | 5.31 |
| 23 | E | {"tau": 0.5, "w_min": 2.2} | 0.2981 | 0.1839 | 0.3721 | 0.2801 | 0.2704 | 0.1345 | 0.000 | 0.000 | 0.0 | 7.23 | 5.31 |
| 24 | E | {"tau": 0.6, "w_min": 0.6} | 0.3071 | 0.1484 | 0.3620 | 0.3032 | 0.2108 | 0.1114 | 0.000 | 0.000 | 0.0 | 4.81 | 3.24 |
| 25 | E | {"tau": 0.6, "w_min": 1.0} | 0.3068 | 0.1513 | 0.3631 | 0.3035 | 0.2149 | 0.1136 | 0.000 | 0.000 | 0.0 | 4.86 | 3.24 |
| 26 | E | {"tau": 0.6, "w_min": 1.4} | 0.3057 | 0.1548 | 0.3635 | 0.3032 | 0.2223 | 0.1158 | 0.000 | 0.000 | 0.0 | 4.92 | 3.24 |
| 27 | E | {"tau": 0.6, "w_min": 1.8} | 0.3058 | 0.1590 | 0.3683 | 0.3043 | 0.2313 | 0.1203 | 0.000 | 0.000 | 0.0 | 5.01 | 3.24 |
| 28 | E | {"tau": 0.6, "w_min": 2.2} | 0.3059 | 0.1644 | 0.3717 | 0.3050 | 0.2410 | 0.1274 | 0.000 | 0.000 | 0.0 | 5.13 | 3.24 |
| 29 | E | {"tau": 0.7, "w_min": 0.6} | 0.3154 | 0.1122 | 0.3520 | 0.3158 | 0.1493 | 0.0723 | 0.000 | 0.000 | 0.0 | 2.90 | 1.80 |
| 30 | E | {"tau": 0.7, "w_min": 1.0} | 0.3155 | 0.1186 | 0.3561 | 0.3177 | 0.1590 | 0.0767 | 0.000 | 0.000 | 0.0 | 3.00 | 1.80 |
| 31 | E | {"tau": 0.7, "w_min": 1.4} | 0.3143 | 0.1255 | 0.3590 | 0.3184 | 0.1728 | 0.0819 | 0.000 | 0.000 | 0.0 | 3.14 | 1.80 |
| 32 | E | {"tau": 0.7, "w_min": 1.8} | 0.3141 | 0.1333 | 0.3639 | 0.3184 | 0.1896 | 0.0886 | 0.000 | 0.000 | 0.0 | 3.31 | 1.80 |
| 33 | E | {"tau": 0.7, "w_min": 2.2} | 0.3134 | 0.1413 | 0.3672 | 0.3196 | 0.2048 | 0.0998 | 0.000 | 0.000 | 0.0 | 3.51 | 2.20 |
| 34 | E | {"tau": 0.8, "w_min": 0.6} (**best E**) | 0.3197 | 0.0735 | 0.3426 | 0.3207 | 0.0804 | 0.0346 | 0.000 | 0.000 | 0.0 | 1.50 | 0.75 |
| 35 | E | {"tau": 0.8, "w_min": 1.0} | 0.3188 | 0.0850 | 0.3467 | 0.3233 | 0.0987 | 0.0410 | 0.000 | 0.000 | 0.0 | 1.70 | 1.00 |
| 36 | E | {"tau": 0.8, "w_min": 1.4} | 0.3171 | 0.0971 | 0.3497 | 0.3233 | 0.1225 | 0.0492 | 0.000 | 0.000 | 0.0 | 1.94 | 1.40 |
| 37 | E | {"tau": 0.8, "w_min": 1.8} | 0.3154 | 0.1093 | 0.3557 | 0.3225 | 0.1475 | 0.0611 | 0.000 | 0.000 | 0.0 | 2.22 | 1.80 |
| 38 | E | {"tau": 0.8, "w_min": 2.2} | 0.3143 | 0.1212 | 0.3609 | 0.3233 | 0.1695 | 0.0756 | 0.000 | 0.000 | 0.0 | 2.52 | 2.20 |
| 39 | F | {"alpha": 1.0, "w_min": 0.6} (**best F**) | 0.3018 | 0.1384 | 0.3616 | 0.2905 | 0.1892 | 0.0831 | 0.000 | 0.000 | - | 6.10 | 4.42 |
| 40 | F | {"alpha": 1.0, "w_min": 1.4} | 0.3012 | 0.1458 | 0.3639 | 0.2905 | 0.2034 | 0.0886 | 0.000 | 0.000 | - | 6.18 | 4.42 |
| 41 | F | {"alpha": 1.0, "w_min": 2.2} | 0.3005 | 0.1551 | 0.3680 | 0.2894 | 0.2246 | 0.0983 | 0.000 | 0.000 | - | 6.34 | 4.42 |
| 42 | F | {"alpha": 1.5, "w_min": 0.6} | 0.2964 | 0.1600 | 0.3691 | 0.2685 | 0.2335 | 0.0901 | 0.000 | 0.000 | - | 8.02 | 6.45 |
| 43 | F | {"alpha": 1.5, "w_min": 1.4} | 0.2957 | 0.1650 | 0.3698 | 0.2685 | 0.2439 | 0.0939 | 0.000 | 0.000 | - | 8.07 | 6.45 |
| 44 | F | {"alpha": 1.5, "w_min": 2.2} | 0.2951 | 0.1714 | 0.3721 | 0.2674 | 0.2592 | 0.1020 | 0.000 | 0.000 | - | 8.18 | 6.45 |
| 45 | F | {"alpha": 2.0, "w_min": 0.6} | 0.2935 | 0.1749 | 0.3698 | 0.2585 | 0.2547 | 0.1047 | 0.000 | 0.000 | - | 9.26 | 7.97 |
| 46 | F | {"alpha": 2.0, "w_min": 1.4} | 0.2930 | 0.1785 | 0.3706 | 0.2585 | 0.2629 | 0.1073 | 0.000 | 0.000 | - | 9.30 | 7.97 |
| 47 | F | {"alpha": 2.0, "w_min": 2.2} | 0.2922 | 0.1832 | 0.3717 | 0.2574 | 0.2741 | 0.1143 | 0.000 | 0.000 | - | 9.37 | 7.97 |
| 48 | F | {"alpha": 3.0, "w_min": 0.6} | 0.2885 | 0.1915 | 0.3698 | 0.2492 | 0.2849 | 0.1233 | 0.000 | 0.000 | - | 10.75 | 10.00 |
| 49 | F | {"alpha": 3.0, "w_min": 1.4} | 0.2884 | 0.1935 | 0.3702 | 0.2492 | 0.2894 | 0.1248 | 0.000 | 0.000 | - | 10.77 | 10.00 |
| 50 | F | {"alpha": 3.0, "w_min": 2.2} | 0.2876 | 0.1964 | 0.3709 | 0.2480 | 0.2957 | 0.1304 | 0.000 | 0.000 | - | 10.81 | 10.00 |

**Zero-width rate is 0.000 in every single cell** -- Method F's whole
purpose (fixing Method B's 29.8% zero-width failure at K=4) is validated
structurally, and Method E's `w_min` floor achieves the same. `clip_anchor_fallback_rate`
and `degenerate_fallback_rate` are 0 everywhere -- every question in
val_tune has a usable query embedding and a non-degenerate CLIP profile.
Full per-question detail for the four designated cells (frozen D@2.2, best
D by mIoP, best E, best F) is in `per_question/*.csv`.

Note the clear, monotone D-curve: mIoP falls and mIoU rises as
`half_width_s` grows. This is the mechanical IoP-vs-IoU trade-off the task
warned about -- **not evidence about localization quality**, just what
happens to a ratio-of-intersection-to-width metric as width changes with
retrieval held fixed.

## 4. Best cell per method vs frozen D@2.2

| | mIoP | mIoU | IoP@0.5 | notes |
|---|---|---|---|---|
| **Frozen D@2.2** (cell 5) | 0.30454 | 0.16462 | 0.30615 | current deployed baseline |
| **Best D by mIoP** (cell 1, hw=0.5) | 0.31231 | **0.05642** | 0.31546 | degenerate corner: narrowest width in the grid, mIoU collapses to a third of the frozen cell's. This is the literal winner under "primary=mIoP, tiebreak=IoP@0.5" -- flagged here as not a meaningful baseline, not adopted as the comparison anchor for the headline finding. |
| **Best E by mIoP** (cell 34, tau=0.8/w_min=0.6) | 0.31974 | 0.07350 | 0.32067 | also a narrow-width regime (median width 0.75s); same mIoU collapse as best-D, worse than frozen D on mIoU by more than half |
| **Best F by mIoP** (cell 39, alpha=1.0/w_min=0.6) | 0.30182 | 0.13835 | 0.29050 | closest of the three "best" picks to frozen D's mIoU, still below it |

**Every "best by mIoP" cell across all three methods has LOWER mIoU than
the frozen D@2.2 baseline.** This is the exact artifact criterion 1 exists
to catch: selecting by mIoP alone systematically favors narrower
predictions, which mechanically inflates IoP-based metrics while
collapsing IoU. None of these "best" cells should be read as "the sweep
recommends this."

## 5. Gold-duration-bucket breakdown: frozen D vs best E (mechanism check)

| bucket | n | D@2.2 mIoP | E(best) mIoP | Δ mIoP | D@2.2 mIoU | E(best) mIoU | Δ mIoU |
|---|---|---|---|---|---|---|---|
| <2s | 278 | 0.0881 | 0.1210 | **+0.0330** | 0.0859 | 0.0690 | **-0.0169** |
| 2-5s | 1125 | 0.2056 | 0.2246 | +0.0190 | 0.1741 | 0.0774 | **-0.0968** |
| 5-10s | 731 | 0.3395 | 0.3649 | +0.0254 | 0.2006 | 0.0834 | **-0.1171** |
| >=10s | 551 | 0.5694 | 0.5543 | -0.0151 | 0.1372 | 0.0548 | **-0.0825** |

**This is the mechanism, and it runs the wrong direction.** The task's own
hypothesis was that adaptive width should help the `<2s` bucket
substantially (a fixed 4.4s window structurally cannot fit inside a <2s
gold span) and should not hurt `>=10s`. Instead: the `<2s` bucket's mIoP
improves (as expected, since 0.75s median width fits inside short gold
spans much better than 4.4s) but its **mIoU gets worse**, and mIoU
deteriorates sharply in every other bucket, worst in the `5-10s` bucket
(-0.117). The adaptive method isn't localizing better -- it's predicting
narrower spans across the board (median width 0.75s vs D's fixed 4.4s),
which is exactly the short-prediction IoP artifact (Chrono,
arXiv:2406.18113) the acceptance criteria were built to catch. This
confirms, mechanistically, why criterion 1 failed against the frozen
anchor: the mIoP gain is not coming from better localization of short
events, it's coming from predicting short spans for everything.

## 6. Zero-width and fallback rates

Every cell in the grid: zero-width rate 0.000, CLIP-anchor fallback rate
0.000, degenerate-profile fallback rate 0.000 (E only), weight fallback
rate not separately logged as non-zero in any F cell. No recommended cell
exists (see section 1), so there is nothing to report a rate "for a
recommended cell" -- these are reported for completeness across the whole
grid instead (section 3's table).

## 7. Acceptance criteria -- full result

Registered before looking at results (task section 8):

1. **mIoP improves AND mIoU improves** (point estimate): 0/49 candidates
   pass against frozen D@2.2. 10/49 pass against the degenerate
   best-D-by-mIoP corner (cells 29-38, all Method E, tau in {0.7, 0.8}).
2. **Bootstrap 95% CI on paired mIoP diff excludes zero** (2000 resamples,
   450 videos, seed 20260728): **0/10** of the criterion-1 survivors pass.
   Every one of the 10 candidates' mIoP CI straddles zero (e.g. best case,
   cell 34: diff +0.00743, CI [-0.00346, +0.01781]).
3. **Bootstrap 95% CI on paired mIoU diff excludes zero**: 10/10 pass (the
   mIoU gain over the degenerate hw=0.5 anchor is large and clearly
   significant -- but criterion 1 already failed against the meaningful
   frozen anchor, and criterion 2 already failed here, so this is moot).
4. **Zero-width <= 1%, all fallback rates <= 5%**: all 10 candidates pass
   this criterion in isolation (0% everywhere).

**No candidate passes all four.** Full detail in `bootstrap_ci.json`.

## 8. Fixed-a-priori sensitivity checks (at best-E-by-mIoP, cell 34 -- reported for transparency, not as validation of a recommended cell)

| knob | default | mIoP (default) | mIoU (default) | variant | mIoP (variant) | mIoU (variant) |
|---|---|---|---|---|---|---|
| smooth_s | 0.5 | 0.31974 | 0.07350 | 0.0 | 0.31752 | 0.06655 |
| gap | 1 | 0.31974 | 0.07350 | 0 | 0.32141 | 0.06621 |
| anchor_source | topk | 0.31974 | 0.07350 | survivors | 0.31152 | 0.07210 |

All three swept variants move mIoP and mIoU by less than 0.01 in either
direction -- none of the fixed-a-priori choices is a hidden lever large
enough to change the section-1 conclusion.

## 9. Honest read

**Null result, and a clean one.** Retrieval quality (Gold@4=0.5304) is
identical in every cell by construction -- span construction cannot
improve or hurt what's retrieved, only how it's packaged into a
[start, end] interval. Across 50 genuinely different width rules (8 fixed,
30 profile-thresholded, 12 weighted-spread), none produces a span that is
simultaneously more precise (mIoP) and better-localized (mIoU) than the
current frozen fixed-width rule, once "improves both metrics" is checked
against a meaningful anchor and for bootstrap significance rather than a
bare point estimate against whatever `argmax(mIoP)` happens to land on.

The gold-duration-bucket breakdown (section 5) explains why: the
narrow-width regime that maximizes mIoP is a global shrink, not a
targeted response to short gold spans. It helps the `<2s` bucket's mIoP
exactly as much as it hurts every other bucket's mIoU. There is no
adaptive-localization signal here distinguishable from "predict a shorter
window, IoP goes up mechanically, IoU goes down mechanically."

This joins the project's established pattern (per the task's own framing):
`ppr_lambda`, `ppr_damping`, `action_score_weights`, `persistence_gate`,
and the peak-detection parameters have all previously landed on flat
results. Span-construction width joins that list. **Keep `span_method=D`,
`half_width_s=2.2`** -- unchanged, per this task's constraint that
`tuning/frozen_state.json` is never written here regardless of outcome.

## 10. What did not run, and why

- **val_confirm was never loaded, read, or scored.** Enforced by
  `assert_no_confirm_videos`, called on every invocation of
  `scripts/span_sweep.py` / `scripts/span_sweep_final.py` at three points
  (video-id level, question level, and post-retrieval record level) --
  never fired, 0 confirm videos found among the 450 tune videos or 2685
  questions at any point.
- **The NExT-GQA test split was never touched.**
- **No captioner, answerer, or Cerberus call; no ingest.** `ensure_indexes`
  was replaced with a read-only `check_index_cache` that never ingests --
  confirmed 450/450 videos present under config-hash `4edae64ed40256e3`
  before any retrieval began (would have hard-stopped and reported rather
  than silently ingesting if any were missing).
- **`tuning/frozen_state.json` was never written.** This is a measurement
  task; the freeze decision is the reader's, not this script's.
- **No cell was dropped, sampled, or interpolated.** All 50 cells ran on
  all 2685 questions; `n_scored=2685` in every row of `grid_results.csv`.
- **The originally-planned per-question diff for a reproduction-gate
  failure never applied**, since the gate passed. No committed val_tune
  per-question baseline file exists to diff against in any case (only
  aggregate numbers were given in the task spec) -- noted here so a future
  gate failure isn't blocked on a diff artifact that doesn't exist.
- **A GPU-embedding run-to-run jitter was discovered and worked around**,
  not modified away: `iris/query.py`'s CLIP text embedding runs on CUDA
  when available, causing ~1e-4-magnitude floating-point differences in
  retrieval results between separate process invocations. Not a bug in
  this task's code, not touched (production `iris/` code is out of
  scope) -- worked around by computing every number in this report from a
  single retrieval pass (`scripts/span_sweep_final.py`), never stitching
  values across two different processes' runs. See
  `tuning/span_sweep/environment.json`'s `gpu_note` for detail.
- **Two commits landed on this branch from a concurrent session** while
  this task's background computation was running (`a929453`, `eeaeacc` --
  an Item-D audit and an official-test-split runner). Neither touches
  `eval/metrics.py`, `eval/`, `tuning/span_sweep/`, or
  `split_manifest.json`'s video-id lists; confirmed via `git diff` before
  committing this task's work on top of them. See `environment.json`'s
  `concurrent_work_note`.
