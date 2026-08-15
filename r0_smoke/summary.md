# R0 gate smoke test -- codec triage vs uniform sampling (n=8)

> n=8 is a smoke test, not a result. Each video is 12.5% of any aggregate here. This validates the harness and gives an early direction only. The real R0 run is n=19 (all annotated anomalous videos already on disk).

**Forward-pass counting mode:** `COUNTED` (nn.Module forward passes observed during ingest: **0**)

## Inputs

| video | N frames | gold frames | gold % | gold windows | prod retention % |
|---|---:|---:|---:|---:|---:|
| Abuse028 | 1412 | 76 | 5.4 | 1 | 10.48 |
| Abuse030 | 1544 | 86 | 5.6 | 1 | 10.49 |
| Arrest001 | 2374 | 301 | 12.7 | 1 | 10.45 |
| Arrest007 | 3144 | 631 | 20.1 | 1 | 10.59 |
| Arson007 | 6252 | 3451 | 55.2 | 1 | 10.43 |
| Arson009 | 743 | 96 | 12.9 | 1 | 11.04 |
| Assault006 | 8096 | 6911 | 85.4 | 1 | 10.42 |
| Assault010 | 16177 | 1022 | 6.3 | 2 | 10.41 |

## Arm A -- production (`is_retained_tier`), natural retention

| arm | retention % | M1 window hit | M2 gold recall | M3 precision |
|---|---:|---:|---:|---:|
| A production | 10.54 | 1.0000 | 0.1054 | 0.2560 |
| B recomputed | 10.54 | 0.8750 | 0.1180 | 0.2737 |  <!-- n=8 -->

Arm B ran on 8/8 videos.

## Matched-budget sweep -- Arms C (uniform) and D (random, 5 seeds)

| budget % | arm | M1 window hit | M2 gold recall | M3 precision |
|---:|---|---:|---:|---:|
| 5.0 | C uniform | 1.0000 | 0.0500 | 0.2553 |
| 5.0 | D random (mean+/-sd across videos) | 1.0000 +/- 0.0000 | 0.0500 +/- 0.0044 | 0.2568 +/- 0.2764 |
| 10.0 | C uniform | 1.0000 | 0.0986 | 0.2540 |
| 10.0 | D random (mean+/-sd across videos) | 1.0000 +/- 0.0000 | 0.0990 +/- 0.0113 | 0.2548 +/- 0.2764 |
| 10.5 | C uniform | 1.0000 | 0.1049 | 0.2543 |
| 10.5 | D random (mean+/-sd across videos) | 1.0000 +/- 0.0000 | 0.1034 +/- 0.0119 | 0.2546 +/- 0.2769 |
| 20.0 | C uniform | 1.0000 | 0.1991 | 0.2541 |
| 20.0 | D random (mean+/-sd across videos) | 1.0000 +/- 0.0000 | 0.1964 +/- 0.0138 | 0.2544 +/- 0.2749 |
| 30.0 | C uniform | 1.0000 | 0.2980 | 0.2538 |
| 30.0 | D random (mean+/-sd across videos) | 1.0000 +/- 0.0000 | 0.2997 +/- 0.0156 | 0.2547 +/- 0.2738 |

## Head-to-head at matched budget (10.5%, Arm A's natural retention)

| arm | M1 | M2 | M3 |
|---|---:|---:|---:|
| A production (~10.54%) | 1.0000 | 0.1054 | 0.2560 |
| B recomputed | 0.8750 | 0.1180 | 0.2737 |
| C uniform 10.5% | 1.0000 | 0.1049 | 0.2543 |
| D random 10.5% | 1.0000 | 0.1034 | 0.2546 |

**Delta A - C (pp) at 10.5%:** M1 +0.00, M2 +0.05, M3 +0.18

## Divergence check -- production (A) vs recomputed (B)

| video | A retention % | B retention % | dRet pp | dM1 pp | dM2 pp | dM3 pp | flag |
|---|---:|---:|---:|---:|---:|---:|---|
| Abuse028 | 10.48 | 10.48 | -0.00 | -100.00 | -7.89 | -4.05 | DIVERGENT >5pp |
| Abuse030 | 10.49 | 10.49 | +0.00 | +0.00 | -9.30 | -4.94 | DIVERGENT >5pp |
| Arrest001 | 10.45 | 10.45 | +0.00 | +0.00 | -2.66 | -3.23 | ok |
| Arrest007 | 10.59 | 10.59 | +0.00 | +0.00 | -8.40 | -15.92 | DIVERGENT >5pp |
| Arson007 | 10.43 | 10.43 | -0.00 | +0.00 | +5.71 | +30.21 | DIVERGENT >5pp |
| Arson009 | 11.04 | 11.04 | +0.00 | +0.00 | +21.88 | +25.61 | DIVERGENT >5pp |
| Assault006 | 10.42 | 10.42 | +0.00 | +0.00 | -2.65 | -21.68 | DIVERGENT >5pp |
| Assault010 | 10.41 | 10.41 | +0.00 | +0.00 | +13.41 | +8.14 | DIVERGENT >5pp |

## Pre-registered verdict

| IRIS arm | dM1 pp | dM2 pp | dM3 pp | half-budget win | Scen. A | Scen. B | Scen. C |
|---|---:|---:|---:|---|---|---|---|
| A (production) | +0.00 | +0.05 | +0.18 | False | False | True | False |
| B (recomputed) | -12.50 | +1.31 | +1.94 | False | False | False | False |

(All deltas are versus Arm C uniform at b = 10.5%.)

- Lowest Arm C budget in the sweep reaching M1 >= 0.99: 5.0%
- Scenario A fires if EITHER arm gains >5pp on M2, or reaches 99% M1 at <=half that budget.

### VERDICT: **Scenario B**

M1 is tied at ceiling and A (production) wins on M2/M3. Margins are stated above -- read them before treating this as a win.

## Kill criterion

**TRIGGERED.** Arm C (uniform) reaches M1 = 1.0000 >= 0.99 at b = 10.5%. M1 (window hit rate) cannot be a headline metric regardless of which scenario applies -- uniform sampling already saturates it.

## Caveat

n=8 is a smoke test, not a result. Each video is 12.5% of any aggregate here. This validates the harness and gives an early direction only. The real R0 run is n=19 (all annotated anomalous videos already on disk).
