# half_width duration-anchor confirmation sweep — tuning report (UNRATIFIED)

**Status:** untracked report, not committed. No value has been written to `iris_config.py`
or `eval/span.py::predict_span` defaults. This is the record for chat ratification only.

**State at run time:** `git status --porcelain` clean, HEAD = `b31166dbb709ca0390701943a29783fbaf569236`.

**Method:** answerer-free, retrieval-only. `eval/grounding_scorer.py` primitives
(`frames_in_window`, `iop`, `load_indexes`) + `eval/span.py::predict_span(mode="ppr_peak")`.
No `aria`, no captions, no LLM call anywhere in this run. N=64 (dev_100.jsonl ∩ index_cache ∩
gsub_val.json grounded), in-sample, no held-out split (compute-limited, accepted per prior turn).
Retrieval config = the "Proposed System" arm from `scripts/pillar2_grounded_qa.py`
(`cerberus_mode="v2"`, `ranking_mode="ppr"`, `ppr_lambda=0.5`, `top_k=8`, Variant B L1 weights).
Bootstrap: video-level cluster resampling, 1000 resamples, seed=20260710 (matches
`bootstrap_paired_differences` convention already in the codebase).

Anchor **w\*=2.2s** = median gold half-span on the 64-question set (decided in chat prior to
this run — this is a confirmation sweep, not an argmax search).

## STEP 1 — Confirmation sweep

Grid brackets the anchor and spans p25→p75 of the half-span distribution (1.35→3.65s) plus wider points.

| w (s) | mIoP | IoP@0.5 | 95% CI (mIoP) |
|---|---|---|---|
| 1.0 | 0.2582 | 0.2656 | [0.1569, 0.3649] |
| 1.5 | 0.2574 | 0.2656 | [0.1600, 0.3574] |
| 2.0 | 0.2596 | 0.2656 | [0.1647, 0.3547] |
| **2.2 (anchor)** | **0.2586** | **0.2812** | **[0.1647, 0.3520]** |
| 2.5 | 0.2570 | 0.2812 | [0.1645, 0.3476] |
| 3.0 | 0.2557 | 0.2656 | [0.1652, 0.3438] |
| 4.0 | 0.2569 | 0.2500 | [0.1751, 0.3380] |
| 6.0 | 0.2510 | 0.1719 | [0.1829, 0.3239] |

## STEP 2 — Curve shape + freeze rule

- **Shape: FLAT, not single-peaked.** mIoP ranges only 0.2510–0.2596 across the entire grid
  (a 0.0086 spread) — 4 sign changes in the slope over 7 intervals, i.e. noisy/flat, not a clean
  single peak. This is well inside noise given bootstrap CI half-widths of ~0.09–0.10.
- **argmax = w=2.0** (mIoP=0.2596, 95% CI [0.1647, 0.3547]).
- **Pre-registered check:** is anchor (2.2) mIoP within argmax's CI? **Yes** — 0.2586 ∈
  [0.1647, 0.3547].
- **Freeze case: ANCHOR STANDS.** Per the rule set before this run, the anchor is not
  statistically distinguishable from the argmax, so w=2.2 is the candidate under ratification —
  **not** the argmax (2.0), even though they're numerically close here.

## STEP 3 — Two-population breakdown at w=2.2

| Population | mIoP | IoP@0.5 |
|---|---|---|
| ALL (n=64) | 0.2586 | 0.2812 |
| SINGLE-interval (n=57) | 0.2704 | 0.2982 |
| MULTI-interval (n=7) | 0.1623 | 0.1429 |

The multi-interval subset is clearly worse (structural cap confirmed — a single symmetric span
around one peak cannot cover disjoint gold intervals). The single-interval, ceiling-corrected
number (mIoP=0.2704) is the more representative "grounding quality" figure, and it's still well
below the roadmap's expected range.

## STEP 4 — Span-fix lift (minmax vs ppr_peak @ w=2.2) — MAJOR FINDING

| mode | mIoP | 95% CI |
|---|---|---|
| minmax (old bug) | 0.2744 | [0.2060, 0.3478] |
| ppr_peak (w=2.2) | 0.2586 | [0.1647, 0.3520] |
| **delta (ppr_peak − minmax)** | **−0.0158** | **[−0.0928, +0.0666]** |

**Roadmap v8 pre-registered expectation: mIoP moves from ~0.31 (minmax) toward high-30s
(ppr_peak). This does NOT hold.**

- Observed minmax (0.2744) is already below the roadmap's stated ~0.31 baseline.
- Observed ppr_peak (0.2586) is not only short of "high-30s," it is *numerically lower* than
  minmax on this set — though the delta's CI ([−0.0928, +0.0666]) comfortably contains zero, so
  the direction is not statistically distinguishable from "no change" either. There is no
  evidence of the expected lift, and no evidence of the fix making things worse — just no
  demonstrated lift at all on this N=64 sample.
- Reported plainly, not adjusted: neither the grid nor the config was changed to chase the
  expected number.

## STEP 5 — IoP histogram at w=2.2 (P1 diagnostic)

```
IoP [0.0,0.1)    n=39  #######################################
IoP [0.1,0.2)    n= 2  ##
IoP [0.2,0.3)    n= 3  ###
IoP [0.3,0.4)    n= 2  ##
IoP [0.4,0.5)    n= 0
IoP [0.5,0.6)    n= 1  #
IoP [0.6,0.7)    n= 6  ######
IoP [0.7,0.8)    n= 1  #
IoP [0.8,0.9)    n= 2  ##
IoP [0.9,1.0)    n= 1  #
IoP 1.00         n= 7  #######
(n=64 total; n at exactly 0.0 = 38)
```

**Mass sits near 0, not just under 0.5.** 39/64 questions (61%) land in [0.0, 0.1), and 38 of
those are *exactly* 0.0 — the predicted span and gold span don't overlap at all. Only a small
tail (≈17/64) clusters at IoP≥0.6, including 7 exact 1.0s.

**This answers the P1 diagnostic directly: the dominant failure mode is RETRIEVAL, not
localization.** If localization/span-width were the bottleneck, mass would sit clustered just
under 0.5 (correct region, span too wide/narrow to clear the threshold) — that's not what's
observed. Instead the distribution is bimodal-ish: either retrieval found the right neighborhood
(IoP clusters 0.6–1.0) or it missed entirely (IoP=0, majority case). Widening or narrowing
half_width cannot fix a retrieval miss — which is also *why* the sweep in Step 1 is flat: half_width
only has leverage over the ~17/64 questions where retrieval already landed near the gold window;
for the other ~39/64 it's inert by construction.

## Bottom line for ratification

1. **Freeze candidate: w=2.2s** (anchor stands per pre-registered rule — not the argmax).
2. **Major finding:** the roadmap v8 expected lift (minmax ~0.31 → ppr_peak high-30s) is **not
   observed**. Both numbers are lower than expected, and the span-mode delta is statistically
   indistinguishable from zero on N=64.
3. **Major finding:** grounding failures are **retrieval-dominated**, not localization-dominated —
   61% of questions have zero temporal overlap between predicted and gold spans, and half_width
   has no lever on those. Tuning half_width further will not move the headline number much; the
   retrieval stage is the bottleneck.
4. Nothing was written to config. Freeze into `iris_config`/`predict_span` defaults remains a
   separate, explicitly authorized step.
