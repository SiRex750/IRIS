# P_SELCEILING result

Implements eval_results/P_selceiling_prereg.md (pre-registered at commit
a5d1cdd) via scripts/p_selceiling_diagnostic.py. Read-only. VAL only, top_k=8
only. Selected nothing, changed no default. All fail-loud assertions passed.

## Provenance
```
git_commit:   a5d1cdd68565e5545dea91ab4fcde36e6a354437
git_dirty:    true (uncommitted P_selceiling_diagnostic.py / raw / result.md
              at run time -- this run's own outputs)
config_hash:  abbcf1b1bdaf84c8fda9b281f4700c968cfbb2295537fbb3a128c4767cf2f1b0
timestamp_utc: 2026-07-25T09:29:19.338925+00:00
n_questions:  406
n_videos:     59
top_k:        8
num_boot:     1000
seed:         20260710
span_mode:    ppr_peak / half_width=2.2 / peak_source=clip_in_ppr_top8
```

## Assertions (all fail-loud, all PASSED)
1. Recoverable set reproduces the funnel's own per-question pool_coverage /
   peak_in_gold flags at top_k=8 (cross-checked row-by-row against
   eval_results/P_funnel_raw.json): 0 mismatches.
   recoverable_set size = 136 = pool_coverage(267) - peak_in_gold(131). PASSED.
2. Caption availability counted, nothing skipped: 3248 pool frames scored,
   0 skipped, 1544 (47.54%) carried a cached caption. PASSED.
3. Zero TEST videos present (existing loader assertion + explicit re-check):
   59 val videos, 27 test videos on file, zero overlap. PASSED.
4. Every recoverable-set row has pool_coverage=1 AND peak_in_gold=0 by
   construction: PASSED.

## Recoverable set
n=136 (33.50% of all 406 VAL questions at top_k=8).

## Reads (video-clustered bootstrap, 1000 resamples, seed=20260710)

| arm | recovered_fraction (recoverable set) | CI | tie_rate |
|---|---|---|---|
| ARM 1 -- lexical (GATE) | 0.2085 | [0.1360, 0.2783] | 0.6544 |
| ARM 2 -- CLIP-text (context) | 0.2358 | [0.1631, 0.3148] | 0.1029 |

Agreement rate (ARM 1 pick == ARM 2 pick, recoverable set): 0.3162

Whole-pool context (all 406 questions, not just recoverable set):

| arm | mean | CI |
|---|---|---|
| ARM 1 -- lexical | 0.2793 | [0.2266, 0.3357] |
| ARM 2 -- CLIP-text | 0.2833 | [0.2230, 0.3450] |

Caption availability: 1544 / 3248 pool frames (47.54%) carried a cached
caption; the remaining 52.46% were scored as non-recoveries (lexical score
0.0, CLIP-text score -inf), never skipped, per the pre-registered treatment.

## Gate (pre-registered, read off ARM 1 lexical ONLY)
ARM 1 recovered_fraction CI-lower = 0.1360.
```
<= 0.10  -> NOT worth a fresh-split experiment
>= 0.30  -> worth building/validating on a NEW split
otherwise -> INCONCLUSIVE
```
**Branch: INCONCLUSIVE.** 0.1360 falls strictly between 0.10 and 0.30.
Report as such; build nothing. ARM 2 (CLIP-text, 0.2358 CI
[0.1631, 0.3148]) is context only and does not move this gate.

## STOP condition
Selects nothing. Changes no default, threshold, or frozen value. No
reranker built or seated.
