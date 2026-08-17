# T5 -- codec shot-bucketing vs uniform sampling at matched budget (n=19)

> n=19 is the full set of annotated anomalous videos available. Each video is 5.3pp of any aggregate. Pooled AUC is not computed -- the 150 Normal videos have no local .mp4, and pooled AUC is confounded regardless (90.7% of negative pairs come from other videos).

**Forward-pass counting mode:** `COUNTED` (nn.Module forward passes during the whole run: **0**). Shot spans come from packet demux only -- no pixel decode, no model.

**Pre-registered before the run:** primary arm `F_shot1_action`, primary contrast vs `C_uniform` at b = 10.5%, primary outcomes dM2/dM3, effect of interest 1.0pp, three-branch read. M1 reported but not primary.

## Shot spans (zero-decode, from the .mp4 alone)

| video | N frames | I-frames | shots | mean shot len | k @10.5% | shots as % of k | demux+span s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Abuse028 | 1412 | 6 | 20 | 70.6 | 148 | 13.5 | 0.03 |
| Abuse030 | 1544 | 7 | 55 | 28.1 | 162 | 34.0 | 0.03 |
| Arrest001 | 2374 | 10 | 52 | 45.7 | 249 | 20.9 | 0.03 |
| Arrest007 | 3144 | 16 | 76 | 41.4 | 330 | 23.0 | 0.04 |
| Arrest024 | 3629 | 15 | 74 | 49.0 | 381 | 19.4 | 0.04 |
| Arrest030 | 8642 | 36 | 252 | 34.3 | 907 | 27.8 | 0.09 |
| Arrest039 | 15835 | 64 | 431 | 36.7 | 1662 | 25.9 | 0.21 |
| Arson007 | 6252 | 27 | 100 | 62.5 | 656 | 15.2 | 0.07 |
| Arson009 | 743 | 6 | 27 | 27.5 | 78 | 34.6 | 0.01 |
| Arson010 | 3159 | 18 | 84 | 37.6 | 331 | 25.4 | 0.03 |
| Arson011 | 1266 | 11 | 31 | 40.8 | 132 | 23.5 | 0.02 |
| Arson016 | 1795 | 8 | 58 | 30.9 | 188 | 30.9 | 0.02 |
| Arson018 | 842 | 4 | 22 | 38.3 | 88 | 25.0 | 0.01 |
| Arson022 | 8640 | 35 | 253 | 34.2 | 907 | 27.9 | 0.12 |
| Arson035 | 1437 | 7 | 28 | 51.3 | 150 | 18.7 | 0.03 |
| Arson041 | 3754 | 18 | 54 | 69.5 | 394 | 13.7 | 0.05 |
| Assault006 | 8096 | 34 | 251 | 32.3 | 850 | 29.5 | 0.08 |
| Assault010 | 16177 | 65 | 430 | 37.6 | 1698 | 25.3 | 0.17 |
| Assault011 | 2288 | 10 | 62 | 36.9 | 240 | 25.8 | 0.04 |

Total demux+span wall time for all 19: **1.1s**. Every video has shots S well below the 10.5% budget k, so the coverage floor (1 or 2 frames per shot) spends only a fraction of the budget and the rest is apportioned -- see the `note` column of results.csv.

## Budget 5.0%

| arm | mean n admitted | M1 window hit | M2 gold recall | M3 precision |
|---|---:|---:|---:|---:|
| C uniform | 239 | 1.0000 | 0.0495 | 0.2901 |
| D random (5 seeds) | 239 | 1.0000 | 0.0499 | 0.2948 |
| E action-score top-k (R0 anchor) | 239 | 0.8421 | 0.0520 | 0.3112 |
| F shot 1/shot, highest-action  [PRIMARY] | 239 | 0.9474 | 0.0501 | 0.3048 |
| G shot 2/shot, highest-action | 239 | 0.9474 | 0.0511 | 0.3080 |
| H shot 1/shot, midpoint (pure geometry) | 239 | 1.0000 | 0.0508 | 0.3014 |
| I shot 2/shot, midpoint (pure geometry) | 239 | 1.0000 | 0.0516 | 0.3015 |

## Budget 10.0%

| arm | mean n admitted | M1 window hit | M2 gold recall | M3 precision |
|---|---:|---:|---:|---:|
| C uniform | 479 | 1.0000 | 0.0993 | 0.2915 |
| D random (5 seeds) | 479 | 1.0000 | 0.1006 | 0.2958 |
| E action-score top-k (R0 anchor) | 479 | 0.8947 | 0.1109 | 0.3287 |
| F shot 1/shot, highest-action  [PRIMARY] | 479 | 0.9474 | 0.1005 | 0.3096 |
| G shot 2/shot, highest-action | 479 | 0.9474 | 0.1007 | 0.3071 |
| H shot 1/shot, midpoint (pure geometry) | 479 | 1.0000 | 0.0999 | 0.2934 |
| I shot 2/shot, midpoint (pure geometry) | 479 | 1.0000 | 0.0987 | 0.2925 |

## Budget 10.5%  <-- PRIMARY

| arm | mean n admitted | M1 window hit | M2 gold recall | M3 precision |
|---|---:|---:|---:|---:|
| C uniform | 503 | 1.0000 | 0.1046 | 0.2914 |
| D random (5 seeds) | 503 | 1.0000 | 0.1052 | 0.2954 |
| E action-score top-k (R0 anchor) | 503 | 0.8947 | 0.1177 | 0.3304 |
| F shot 1/shot, highest-action  [PRIMARY] | 503 | 0.9474 | 0.1062 | 0.3081 |
| G shot 2/shot, highest-action | 503 | 0.9474 | 0.1051 | 0.3063 |
| H shot 1/shot, midpoint (pure geometry) | 503 | 1.0000 | 0.1046 | 0.2928 |
| I shot 2/shot, midpoint (pure geometry) | 503 | 1.0000 | 0.1042 | 0.2942 |

## Budget 20.0%

| arm | mean n admitted | M1 window hit | M2 gold recall | M3 precision |
|---|---:|---:|---:|---:|
| C uniform | 958 | 1.0000 | 0.1994 | 0.2916 |
| D random (5 seeds) | 958 | 1.0000 | 0.1989 | 0.2934 |
| E action-score top-k (R0 anchor) | 958 | 1.0000 | 0.2259 | 0.3289 |
| F shot 1/shot, highest-action  [PRIMARY] | 958 | 0.9474 | 0.2036 | 0.3100 |
| G shot 2/shot, highest-action | 958 | 0.9474 | 0.2016 | 0.3075 |
| H shot 1/shot, midpoint (pure geometry) | 958 | 1.0000 | 0.2005 | 0.2926 |
| I shot 2/shot, midpoint (pure geometry) | 958 | 1.0000 | 0.1997 | 0.2940 |

## Budget 30.0%

| arm | mean n admitted | M1 window hit | M2 gold recall | M3 precision |
|---|---:|---:|---:|---:|
| C uniform | 1437 | 1.0000 | 0.2989 | 0.2914 |
| D random (5 seeds) | 1437 | 1.0000 | 0.3011 | 0.2936 |
| E action-score top-k (R0 anchor) | 1437 | 1.0000 | 0.3525 | 0.3316 |
| F shot 1/shot, highest-action  [PRIMARY] | 1437 | 0.9474 | 0.3080 | 0.3104 |
| G shot 2/shot, highest-action | 1437 | 0.9474 | 0.3068 | 0.3096 |
| H shot 1/shot, midpoint (pure geometry) | 1437 | 1.0000 | 0.3003 | 0.2923 |
| I shot 2/shot, midpoint (pure geometry) | 1437 | 1.0000 | 0.2999 | 0.2933 |

## PRIMARY read -- F_shot1_action minus C_uniform at 10.5%

| video | gold % | shots | F M2 | C M2 | dM2 pp | F M3 | C M3 | dM3 pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Abuse028 | 5.4 | 20 | 0.0000 | 0.1053 | -10.526 | 0.0000 | 0.0541 | -5.405 |
| Abuse030 | 5.6 | 55 | 0.1279 | 0.1047 | +2.326 | 0.0679 | 0.0556 | +1.235 |
| Arrest001 | 12.7 | 52 | 0.0532 | 0.1063 | -5.316 | 0.0643 | 0.1285 | -6.426 |
| Arrest007 | 20.1 | 76 | 0.1141 | 0.1046 | +0.951 | 0.2182 | 0.2000 | +1.818 |
| Arrest024 | 57.9 | 74 | 0.1314 | 0.1047 | +2.665 | 0.7244 | 0.5774 | +14.698 |
| Arrest030 | 19.3 | 252 | 0.1068 | 0.1044 | +0.240 | 0.1963 | 0.1918 | +0.441 |
| Arrest039 | 19.7 | 431 | 0.0904 | 0.1051 | -1.474 | 0.1697 | 0.1974 | -2.768 |
| Arson007 | 55.2 | 100 | 0.0959 | 0.1049 | -0.898 | 0.5046 | 0.5518 | -4.726 |
| Arson009 | 12.9 | 27 | 0.0729 | 0.1042 | -3.125 | 0.0897 | 0.1282 | -3.846 |
| Arson010 | 11.0 | 84 | 0.1127 | 0.1040 | +0.867 | 0.1178 | 0.1088 | +0.906 |
| Arson011 | 67.7 | 31 | 0.1167 | 0.1039 | +1.284 | 0.7576 | 0.6742 | +8.333 |
| Arson016 | 44.3 | 58 | 0.1006 | 0.1044 | -0.377 | 0.4255 | 0.4415 | -1.596 |
| Arson018 | 39.3 | 22 | 0.1178 | 0.1057 | +1.208 | 0.4432 | 0.3977 | +4.545 |
| Arson022 | 5.8 | 253 | 0.1257 | 0.1038 | +2.196 | 0.0695 | 0.0573 | +1.213 |
| Arson035 | 20.9 | 28 | 0.1960 | 0.1030 | +9.302 | 0.3933 | 0.2067 | +18.667 |
| Arson041 | 39.6 | 54 | 0.1117 | 0.1050 | +0.673 | 0.4213 | 0.3959 | +2.538 |
| Assault006 | 85.4 | 251 | 0.1019 | 0.1049 | -0.304 | 0.8282 | 0.8529 | -2.471 |
| Assault010 | 6.3 | 430 | 0.1243 | 0.1047 | +1.957 | 0.0748 | 0.0630 | +1.178 |
| Assault011 | 25.6 | 62 | 0.1177 | 0.1041 | +1.365 | 0.2875 | 0.2542 | +3.333 |

| metric | mean d (pp) | sd (pp) | F>C | F<C | tie | 95% bootstrap CI (pp) | CI half-width | branch |
|---|---:|---:|---:|---:|---:|---|---:|---|
| dM2 | +0.159 | 3.847 | 12 | 7 | 0 | [-1.615, +1.766] | 1.691 | **UNDERPOWERED** |
| dM3 | +1.667 | 6.464 | 12 | 7 | 0 | [-0.941, +4.646] | 2.793 | **UNDERPOWERED** |

- **dM2: UNDERPOWERED** -- CI spans 0 but its half-width >= 1.0pp -- this corpus CANNOT distinguish a real effect from null. Not a tie.
- **dM3: UNDERPOWERED** -- CI spans 0 but its half-width >= 1.0pp -- this corpus CANNOT distinguish a real effect from null. Not a tie.

Bootstrap: 10,000 percentile resamples over the 19 videos, seed 20260805 (imported from r0_gate_full).

## Secondary contrasts (not pre-registered as primary; read with care)

| contrast | budget % | metric | mean d (pp) | 95% CI (pp) | half-width | branch |
|---|---:|---|---:|---|---:|---|
| E_action_topk - C_uniform | 10.5 | dM2 | +1.309 | [-2.745, +5.628] | 4.186 | UNDERPOWERED |
| E_action_topk - C_uniform | 10.5 | dM3 | +3.897 | [-4.403, +13.203] | 8.803 | UNDERPOWERED |
| G_shot2_action - C_uniform | 10.5 | dM2 | +0.047 | [-1.771, +1.734] | 1.753 | UNDERPOWERED |
| G_shot2_action - C_uniform | 10.5 | dM3 | +1.488 | [-1.326, +4.627] | 2.977 | UNDERPOWERED |
| H_shot1_mid - C_uniform | 10.5 | dM2 | +0.001 | [-0.416, +0.421] | 0.419 | TIES |
| H_shot1_mid - C_uniform | 10.5 | dM3 | +0.141 | [-0.608, +0.972] | 0.790 | TIES |
| I_shot2_mid - C_uniform | 10.5 | dM2 | -0.044 | [-0.703, +0.655] | 0.679 | TIES |
| I_shot2_mid - C_uniform | 10.5 | dM3 | +0.278 | [-1.090, +1.808] | 1.449 | UNDERPOWERED |
| F_shot1_action - E_action_topk | 10.5 | dM2 | -1.150 | [-5.826, +3.346] | 4.586 | UNDERPOWERED |
| F_shot1_action - E_action_topk | 10.5 | dM3 | -2.230 | [-12.247, +7.250] | 9.749 | UNDERPOWERED |
| F_shot1_action - D_random | 10.5 | dM2 | +0.099 | [-1.385, +1.522] | 1.454 | UNDERPOWERED |
| F_shot1_action - D_random | 10.5 | dM3 | +1.267 | [-1.264, +4.153] | 2.708 | UNDERPOWERED |

The `F - E_action_topk` rows are the scientifically interesting pair: E is the existing R0 codec selector at the same budget, so this isolates what the SHOT BUCKETING adds on top of the action score it already uses.

## F_shot1_action - C_uniform across the budget sweep

| budget % | dM2 pp | dM2 CI | dM2 branch | dM3 pp | dM3 CI | dM3 branch |
|---:|---:|---|---|---:|---|---|
| 5.0 | +0.053 | [-0.767, +0.889] | TIES | +1.475 | [-1.351, +4.796] | UNDERPOWERED |
| 10.0 | +0.126 | [-1.495, +1.607] | UNDERPOWERED | +1.804 | [-0.835, +4.739] | UNDERPOWERED |
| 10.5 | +0.159 | [-1.615, +1.766] | UNDERPOWERED | +1.667 | [-0.941, +4.646] | UNDERPOWERED |
| 20.0 | +0.419 | [-2.911, +3.514] | UNDERPOWERED | +1.838 | [-0.738, +4.824] | UNDERPOWERED |
| 30.0 | +0.913 | [-4.087, +5.589] | UNDERPOWERED | +1.894 | [-0.716, +4.918] | UNDERPOWERED |

## POST-HOC diagnostic -- temporal dispersion (NOT pre-registered)

Found while checking why Abuse028 admitted zero gold frames. `action_score_propagated` is hold-forward-propagated from the retained tier, so it is a STEP function: one distinct value per retained frame, ~9.5 frames per plateau. Picking 'the highest-action frames' therefore degenerates into picking whole plateaus from their earliest frame (the tie-break is ascending frame_idx, imported from `r0.top_k_by_score`), so the budget is spent on ADJACENT, near-duplicate frames. Blocks = maximal runs of consecutive selected frames = the number of distinct temporal locations actually sampled.

| arm | mean run length | mean blocks | budget k | distinct locations as % of budget |
|---|---:|---:|---:|---:|
| C uniform | 1.00 | 503 | 503 | 100.0 |
| E action-score top-k (R0 anchor) | 13.19 | 42 | 503 | 8.4 |
| F shot 1/shot, highest-action  [PRIMARY] | 3.68 | 143 | 503 | 28.4 |
| G shot 2/shot, highest-action | 3.72 | 142 | 503 | 28.2 |
| H shot 1/shot, midpoint (pure geometry) | 1.00 | 503 | 503 | 100.0 |
| I shot 2/shot, midpoint (pure geometry) | 1.00 | 503 | 503 | 100.0 |

This is the mechanism behind the whole table. Uniform samples k distinct instants; the action top-k arm (E) spends the same k frames on ~8% as many distinct instants, which is why it posts the largest mean dM3 AND the widest CI. Shot bucketing's per-shot coverage floor forces dispersion and recovers most of that spread, which is why F sits between E and uniform. The pure-geometry arms (H/I) use no score, never clump, and consequently track uniform almost exactly.

**Abuse028, the worst cell (F dM2 = -10.5pp):** its 76-frame gold window [165,240] lies entirely inside one 237-frame shot. That shot received 10 of the 148 budgeted frames, all placed on higher-action plateaus at 106-121 and 250-255 -- straddling the window without entering it. Plain action top-k (arm E) also scores 0/148 gold frames there, so this is the ACTION SCORE missing the anomaly, not the bucketing: mean score inside the gold window is 0.4643 vs 0.4261 outside, i.e. almost no signal. Uniform gets 8/148 by construction.

## M1 -- reported, NOT a primary outcome

At 10.5%: uniform M1 = 1.0000, F_shot1_action M1 = 0.9474. 17/19 videos have exactly one gold window (max 2), so M1 is near-binary per video and saturates. It is excluded from the verdict by pre-registration.

## T5 VERDICT

**UNDERPOWERED** on the pre-registered primary contrast (F_shot1_action - C_uniform, b = 10.5%): dM2 = +0.159pp [-1.615, +1.766] (UNDERPOWERED), dM3 = +1.667pp [-0.941, +4.646] (UNDERPOWERED).

n=19 is the full set of annotated anomalous videos available. Each video is 5.3pp of any aggregate. Pooled AUC is not computed -- the 150 Normal videos have no local .mp4, and pooled AUC is confounded regardless (90.7% of negative pairs come from other videos).
