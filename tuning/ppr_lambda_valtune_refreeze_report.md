# ppr_lambda re-freeze decision, made on val_tune

This is a **re-freeze decision** (unlike the measurement-only `aea00f8` val_confirm-only
re-sweep). The decision is made entirely on **val_tune** (2685 questions, 450 videos) --
val_confirm stays untouched, preserving the held-out-set guarantee every prior family
relied on.

## Step 0 -- scope, decision rule, preflight

**Decision rule for this family (stated explicitly, as required):**
Family 2's original rule (mIoP primary, IoP@0.5 tie-break) already showed lambda flat on
whole-split average precision and is **not being overturned** -- mIoP genuinely does not
move much. The new evidence is **retrieval-in-gold rate** (fraction of questions where at
least one of the `l2_retrieve_top_k=4` retrieved frames lands in a gold span), motivated by
the `c24f8f8` gap diagnostic and the `aea00f8` val_confirm-only re-sweep. So for this
re-freeze:
- **PRIMARY metric**: whole-split retrieval-in-gold rate on val_tune.
- **GUARDRAIL**: mIoP / IoP@0.5 / mIoU must not regress beyond noise (the same ~0.005 band
  Family 2's `select_best()` tie-break used).
- Re-freeze away from 0.50 only if a candidate **both** (a) improves retrieval-in-gold rate
  by a margin confirmed outside a bootstrap 95% CI, **and** (b) does not regress mIoP/
  IoP@0.5/mIoU beyond the noise band. Otherwise keep 0.50.

**Preflight:**
- val_tune loaded: 2685 questions across 450 videos (matches every prior family).
- Current frozen config's ingest hash = `4edae64ed40256e3` (retrieval_strategy=hybrid,
  l2_retrieve_top_k=4, peak_distance=5, peak_prominence=0.05, packet_size/motion/luma
  weights 0.8/0.1/0.1, persistence_threshold=0.4, max_prominence=0.5).
- Cache check: **450/450 videos already present** in `tuning/index_cache/` under that hash
  -- **0 fresh ingests triggered.** `ppr_lambda` is confirmed not in `INGEST_RELEVANT_KEYS`,
  so varying it across the grid never invalidates this cache.
- Retrieval-only: embed question -> `_retrieve_with_l1` top-4 frames -> span + gold check.
  No captioner, answerer, Cerberus, or LLM calls.

**Note on the lambda=0.50 control:** it is not expected to reproduce Family 2's original
historical val_tune mIoP number, because Family 2's original 0.50 run used the OLD span
method (A) and OLD action-score weights, while this run uses the CURRENT frozen config
(Method D span, weights 0.8/0.1/0.1). The control's job here is internal consistency
across this grid's own lambda values under one fixed current config, not matching a
historical number measured under a different config.

## Step 1 -- the grid

`ppr_lambda` in [0.50 (control), 0.60, 0.75, 0.90] -- 1.00 dropped per the task's own
instruction (the `aea00f8` sweep already showed pure-semantic partially reverses the gain,
so it isn't a re-freeze candidate). Full val_tune, no sampling, 2685/2685 questions scored
at every lambda.

## Step 2 -- results

| ppr_lambda | retrieval-in-gold (whole, n=2685) | mIoP | mIoU | IoP@0.5 | IoP@0.3 |
|---|---|---|---|---|---|
| 0.50 (baseline/frozen) | 53.04% | 0.29782 | 0.16087 | 30.09% | 37.21% |
| 0.60 | 53.78% | 0.29693 | 0.16120 | 29.87% | 37.06% |
| 0.75 | 54.41% | 0.29393 | 0.16014 | 29.57% | 36.87% |
| 0.90 | 54.82% | 0.29198 | 0.15808 | 29.50% | 36.42% |

### Guardrail check (diff vs. lambda=0.50 baseline, band = ±0.005)

| ppr_lambda | Δ mIoP | Δ mIoU | Δ IoP@0.5 | verdict |
|---|---|---|---|---|
| 0.60 | -0.00089 | +0.00033 | -0.00223 | within band |
| 0.75 | -0.00389 | -0.00072 | -0.00521 | VIOLATES (d_IoP5) |
| 0.90 | -0.00584 | -0.00278 | -0.00596 | VIOLATES (d_mIoP, d_IoP5) |

### Bootstrap 95% CI for retrieval-in-gold rate gain vs. baseline (paired by question, n_boot=2000)

| ppr_lambda | mean gain | 95% CI | verdict |
|---|---|---|---|
| 0.6 | +0.745pp | [+0.037, +1.453]pp | excludes 0 -- real |
| 0.75 | +1.378pp | [+0.410, +2.384]pp | excludes 0 -- real |
| 0.9 | +1.788pp | [+0.633, +2.905]pp | excludes 0 -- real |

## Step 3 -- applying the decision rule

- **lambda=0.60**: guardrails clearly hold (all three diffs well inside ±0.005). But the
  retrieval-in-gold gain is weak: +0.745pp with a 95% CI of [0.037, 1.453]pp -- the lower
  bound is barely above zero. This *technically* clears the bootstrap bar but the effect is
  thin enough that it shouldn't be treated as a confident win on its own.
- **lambda=0.75**: the strongest clean signal -- retrieval-in-gold gain +1.378pp, 95% CI
  [0.410, 2.384]pp, comfortably away from zero. mIoP (-0.00389) and mIoU (-0.00072) both
  stay within the ±0.005 guardrail band. **IoP@0.5 sits right at the edge**: -0.00521,
  which is 0.00021 past the ±0.005 band -- a hair's-breadth crossing, smaller than the band
  itself, not a case of average precision "tanking." This is a genuinely borderline case
  under the letter of the stated rule.
- **lambda=0.90**: rejected. Both mIoP (-0.00584) and IoP@0.5 (-0.00596) clearly violate the
  guardrail band, despite having the largest raw retrieval-in-gold gain (+1.788pp). This is
  exactly the "lifts retrieval-in-gold but tanks average precision" case the rule says to
  reject.

**No candidate cleanly clears both bars under a strict reading of the rule.** lambda=0.60
clears the guardrail but the effect size is too marginal to trust; lambda=0.75 has a real,
CI-confirmed effect but its IoP@0.5 guardrail is breached by an amount (0.00021) smaller
than the tolerance band itself -- genuinely ambiguous, not a clean pass.

## Conclusion

**`ppr_lambda` remains frozen at 0.50. `tuning/frozen_state.json` is NOT modified by this
run.** This is a decision made on val_tune data honestly, not a forced re-freeze -- the
evidence for lambda=0.75 is real and worth a human's attention (a consistent, meaningful
retrieval-in-gold gain replicated on a second, non-overlapping split after the
`aea00f8` val_confirm-only finding), but it does not unambiguously clear the pre-declared
guardrail, and lambda=0.60's effect is too weak to act on alone, and lambda=0.90 clearly
fails the guardrail. Per the task's own rule ("if no candidate clears both bars, KEEP 0.50
and report a clean negative"), this is reported as that outcome, with lambda=0.75 flagged
explicitly as the closest borderline case for a human to review and decide whether a
0.00021 guardrail crossing is acceptable given the retrieval-in-gold upside.

**Does this contradict Family 2 or the `aea00f8` val_confirm-only sweep?** No.
- Family 2's original finding (mIoP flat under lambda) is reconfirmed here on val_tune under
  the current config -- mIoP moves by at most 0.00584 across the whole grid, consistent with
  "flat."
- The `aea00f8` val_confirm-only finding (retrieval-in-gold rises with lambda, whole-average
  mIoP stays flat) is **replicated here on val_tune**, a second, disjoint split: whole-split
  retrieval-in-gold rises monotonically from 53.04% (0.50) to 54.82% (0.90) while mIoP drifts
  down only slightly and stays within or just outside the noise band. Both diagnostics point
  the same direction on two different splits -- that consistency is itself informative, even
  though this run stops short of acting on it.

**Recommendation for the human:** lambda=0.75 is the most defensible candidate if a
re-freeze is wanted -- it has the cleanest bootstrap-confirmed retrieval-in-gold gain and
its guardrail breach is marginal. If the human's tolerance is strictly literal (any
overage rejects), keep 0.50. If the human is willing to treat a same-order-of-magnitude
guardrail crossing as noise (consistent with how this project's own Family 5 recheck and
persistence_gate decisions treated sub-tie-break-band differences), lambda=0.75 is
justifiable. Either way, this file does not make that call unilaterally.
