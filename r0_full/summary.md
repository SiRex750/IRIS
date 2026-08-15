# R0 gate -- full run: codec triage vs uniform sampling (n=19)

> n=19 is the full set of annotated anomalous videos available. Each video is 5.3pp of any aggregate. Pooled AUC is not computed -- the 150 Normal videos have no local .mp4, and pooled AUC is confounded regardless (90.7% of negative pairs come from other videos).

**Forward-pass counting mode:** `COUNTED` (nn.Module forward passes observed during ingest: **0**)

## Inputs

| video | N frames | gold frames | gold % | gold windows | prod retention % |
|---|---:|---:|---:|---:|---:|
| Abuse028 | 1412 | 76 | 5.4 | 1 | 10.48 |
| Abuse030 | 1544 | 86 | 5.6 | 1 | 10.49 |
| Arrest001 | 2374 | 301 | 12.7 | 1 | 10.45 |
| Arrest007 | 3144 | 631 | 20.1 | 1 | 10.59 |
| Arrest024 | 3629 | 2101 | 57.9 | 1 | 10.42 |
| Arrest030 | 8642 | 1666 | 19.3 | 1 | 10.45 |
| Arrest039 | 15835 | 3121 | 19.7 | 1 | 10.41 |
| Arson007 | 6252 | 3451 | 55.2 | 1 | 10.43 |
| Arson009 | 743 | 96 | 12.9 | 1 | 11.04 |
| Arson010 | 3159 | 346 | 11.0 | 1 | 10.64 |
| Arson011 | 1266 | 857 | 67.7 | 2 | 11.14 |
| Arson016 | 1795 | 795 | 44.3 | 1 | 10.47 |
| Arson018 | 842 | 331 | 39.3 | 1 | 10.57 |
| Arson022 | 8640 | 501 | 5.8 | 1 | 10.42 |
| Arson035 | 1437 | 301 | 20.9 | 1 | 10.58 |
| Arson041 | 3754 | 1486 | 39.6 | 1 | 10.50 |
| Assault006 | 8096 | 6911 | 85.4 | 1 | 10.42 |
| Assault010 | 16177 | 1022 | 6.3 | 2 | 10.41 |
| Assault011 | 2288 | 586 | 25.6 | 1 | 10.45 |

## Arm A -- production (`is_retained_tier`), natural retention

| arm | retention % | M1 window hit | M2 gold recall | M3 precision |
|---|---:|---:|---:|---:|
| A production | 10.54 | 1.0000 | 0.1072 | 0.2992 |

Arm B is **disabled** in this run (`--arm-b` off). It contributes to no table and no verdict.

## Matched-budget sweep -- Arms C (uniform) and D (random, 5 seeds)

| budget % | arm | M1 window hit | M2 gold recall | M3 precision |
|---:|---|---:|---:|---:|
| 5.0 | C uniform | 1.0000 | 0.0495 | 0.2901 |
| 5.0 | D random (mean+/-sd across videos) | 1.0000 +/- 0.0000 | 0.0499 +/- 0.0034 | 0.2948 +/- 0.2315 |
| 10.0 | C uniform | 1.0000 | 0.0993 | 0.2915 |
| 10.0 | D random (mean+/-sd across videos) | 1.0000 +/- 0.0000 | 0.1006 +/- 0.0082 | 0.2958 +/- 0.2324 |
| 10.5 | C uniform | 1.0000 | 0.1046 | 0.2914 |
| 10.5 | D random (mean+/-sd across videos) | 1.0000 +/- 0.0000 | 0.1052 +/- 0.0087 | 0.2954 +/- 0.2334 |
| 20.0 | C uniform | 1.0000 | 0.1994 | 0.2916 |
| 20.0 | D random (mean+/-sd across videos) | 1.0000 +/- 0.0000 | 0.1989 +/- 0.0107 | 0.2934 +/- 0.2307 |
| 30.0 | C uniform | 1.0000 | 0.2989 | 0.2914 |
| 30.0 | D random (mean+/-sd across videos) | 1.0000 +/- 0.0000 | 0.3011 +/- 0.0105 | 0.2936 +/- 0.2301 |

## Head-to-head at matched budget (10.5%, Arm A's natural retention)

| arm | M1 | M2 | M3 |
|---|---:|---:|---:|
| A production (~10.54%) | 1.0000 | 0.1072 | 0.2992 |
| C uniform 10.5% | 1.0000 | 0.1046 | 0.2914 |
| D random 10.5% | 1.0000 | 0.1052 | 0.2954 |

**Delta A - C (pp) at 10.5%:** M1 +0.00, M2 +0.26, M3 +0.78

## Chance baseline -- how far is Arm A from a no-information selector?

A selector carrying no label information yields M2 = its budget fraction and M3 = the gold base rate, by construction. Columns 6 and 9 are the distance from that null.

| video | gold % | A retention % | A M2 | retention/100 | A M2 - retention | A M3 | gold frac | A M3 - gold frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Abuse028 | 5.4 | 10.48 | 0.0789 | 0.1048 | -0.0259 | 0.0405 | 0.0538 | -0.0133 |
| Abuse030 | 5.6 | 10.49 | 0.1047 | 0.1049 | -0.0003 | 0.0556 | 0.0557 | -0.0001 |
| Arrest001 | 12.7 | 10.45 | 0.1229 | 0.1045 | +0.0185 | 0.1492 | 0.1268 | +0.0224 |
| Arrest007 | 20.1 | 10.59 | 0.0983 | 0.1059 | -0.0077 | 0.1862 | 0.2007 | -0.0145 |
| Arrest024 | 57.9 | 10.42 | 0.1047 | 0.1042 | +0.0006 | 0.5820 | 0.5789 | +0.0031 |
| Arrest030 | 19.3 | 10.45 | 0.1014 | 0.1045 | -0.0030 | 0.1872 | 0.1928 | -0.0056 |
| Arrest039 | 19.7 | 10.41 | 0.1045 | 0.1041 | +0.0004 | 0.1978 | 0.1971 | +0.0007 |
| Arson007 | 55.2 | 10.43 | 0.1046 | 0.1043 | +0.0003 | 0.5537 | 0.5520 | +0.0017 |
| Arson009 | 12.9 | 11.04 | 0.1250 | 0.1104 | +0.0146 | 0.1463 | 0.1292 | +0.0171 |
| Arson010 | 11.0 | 10.64 | 0.1127 | 0.1064 | +0.0064 | 0.1161 | 0.1095 | +0.0065 |
| Arson011 | 67.7 | 11.14 | 0.1179 | 0.1114 | +0.0065 | 0.7163 | 0.6769 | +0.0394 |
| Arson016 | 44.3 | 10.47 | 0.1057 | 0.1047 | +0.0009 | 0.4468 | 0.4429 | +0.0039 |
| Arson018 | 39.3 | 10.57 | 0.1269 | 0.1057 | +0.0212 | 0.4719 | 0.3931 | +0.0788 |
| Arson022 | 5.8 | 10.42 | 0.1058 | 0.1042 | +0.0016 | 0.0589 | 0.0580 | +0.0009 |
| Arson035 | 20.9 | 10.58 | 0.1030 | 0.1058 | -0.0028 | 0.2039 | 0.2095 | -0.0055 |
| Arson041 | 39.6 | 10.50 | 0.1063 | 0.1050 | +0.0014 | 0.4010 | 0.3958 | +0.0052 |
| Assault006 | 85.4 | 10.42 | 0.1042 | 0.1042 | -0.0001 | 0.8531 | 0.8536 | -0.0006 |
| Assault010 | 6.3 | 10.41 | 0.1047 | 0.1041 | +0.0006 | 0.0635 | 0.0632 | +0.0004 |
| Assault011 | 25.6 | 10.45 | 0.1041 | 0.1045 | -0.0004 | 0.2552 | 0.2561 | -0.0009 |

**Mean (A M2 - retention) = +0.0017** (sd 0.0099, min -0.0259, max +0.0212)
**Mean (A M3 - gold frac) = +0.0073** (sd 0.0212, min -0.0145, max +0.0788)

## Paired per-video comparison -- Arm A vs Arm C at 10.5%

| video | gold % | A M2 | C M2 | dM2 pp | A M3 | C M3 | dM3 pp |
|---|---:|---:|---:|---:|---:|---:|---:|
| Abuse028 | 5.4 | 0.0789 | 0.1053 | -2.632 | 0.0405 | 0.0541 | -1.351 |
| Abuse030 | 5.6 | 0.1047 | 0.1047 | +0.000 | 0.0556 | 0.0556 | +0.000 |
| Arrest001 | 12.7 | 0.1229 | 0.1063 | +1.661 | 0.1492 | 0.1285 | +2.068 |
| Arrest007 | 20.1 | 0.0983 | 0.1046 | -0.634 | 0.1862 | 0.2000 | -1.381 |
| Arrest024 | 57.9 | 0.1047 | 0.1047 | +0.000 | 0.5820 | 0.5774 | +0.458 |
| Arrest030 | 19.3 | 0.1014 | 0.1044 | -0.300 | 0.1872 | 0.1918 | -0.469 |
| Arrest039 | 19.7 | 0.1045 | 0.1051 | -0.064 | 0.1978 | 0.1974 | +0.046 |
| Arson007 | 55.2 | 0.1046 | 0.1049 | -0.029 | 0.5537 | 0.5518 | +0.185 |
| Arson009 | 12.9 | 0.1250 | 0.1042 | +2.083 | 0.1463 | 0.1282 | +1.814 |
| Arson010 | 11.0 | 0.1127 | 0.1040 | +0.867 | 0.1161 | 0.1088 | +0.731 |
| Arson011 | 67.7 | 0.1179 | 0.1039 | +1.400 | 0.7163 | 0.6742 | +4.207 |
| Arson016 | 44.3 | 0.1057 | 0.1044 | +0.126 | 0.4468 | 0.4415 | +0.532 |
| Arson018 | 39.3 | 0.1269 | 0.1057 | +2.115 | 0.4719 | 0.3977 | +7.418 |
| Arson022 | 5.8 | 0.1058 | 0.1038 | +0.200 | 0.0589 | 0.0573 | +0.156 |
| Arson035 | 20.9 | 0.1030 | 0.1030 | +0.000 | 0.2039 | 0.2067 | -0.272 |
| Arson041 | 39.6 | 0.1063 | 0.1050 | +0.135 | 0.4010 | 0.3959 | +0.508 |
| Assault006 | 85.4 | 0.1042 | 0.1049 | -0.072 | 0.8531 | 0.8529 | +0.014 |
| Assault010 | 6.3 | 0.1047 | 0.1047 | +0.000 | 0.0635 | 0.0630 | +0.052 |
| Assault011 | 25.6 | 0.1041 | 0.1041 | +0.000 | 0.2552 | 0.2542 | +0.106 |

| metric | mean d (pp) | sd (pp) | A > C | A < C | exact tie | 95% bootstrap CI (pp) |
|---|---:|---:|---:|---:|---:|---|
| dM2 | +0.256 | 1.068 | 8 | 6 | 5 | [-0.236, +0.710] |
| dM3 | +0.780 | 2.029 | 14 | 4 | 1 | [-0.005, +1.753] |

Bootstrap: 10,000 percentile resamples over the 19 videos, seed 20260805.

**The 95% CI for dM2 spans zero** ([-0.236, +0.710] pp): this run does not establish that Arm A differs from uniform sampling on M2.
**The 95% CI for dM3 spans zero** ([-0.005, +1.753] pp): this run does not establish that Arm A differs from uniform sampling on M3.

## Divergence check -- production (A) vs recomputed (B)

Not applicable: Arm B is disabled in this run.

## Verdict

| IRIS arm | dM1 pp | dM2 pp | dM3 pp | half-budget win |
|---|---:|---:|---:|---|
| A (production) | +0.00 | +0.26 | +0.78 | False |

(All deltas are aggregate Arm A minus Arm C uniform at b = 10.5%. Arm B is excluded from the verdict by design.)

- Lowest Arm C budget in the sweep reaching M1 >= 0.99: 5.0%

### V1 -- as pre-registered (verbatim rule)

**Scenario B** -- M1 is tied at ceiling and Arm A wins on M2/M3. The margin is not part of this rule.

### V2 -- margin-qualified

V2 adds a minimum-margin requirement that was NOT in the original pre-registration. It was added on 5 Aug 2026 after the n=8 run showed the original Scenario B condition firing on a 0.05pp margin. Both verdicts are reported so the change is auditable. V1 is the rule as written.

**No scenario matched** -- Scenario B fired under V1 but fails the added margin test: per-video mean dM2 = +0.256 pp (CI [-0.236, +0.710]), dM3 = +0.780 pp (CI [-0.005, +1.753]). Neither exceeds 1.0pp with a CI excluding zero.

## Kill criterion

Lowest swept budget at which Arm C reaches M1 >= 0.99: **5.0%** (Arm C M1 at 5%/10%/10.5%/20%/30% = 1.0000, 1.0000, 1.0000, 1.0000, 1.0000).

**TRIGGERED.** Arm C (uniform) reaches M1 = 1.0000 >= 0.99 at b = 10.5%. M1 (window hit rate) cannot be a headline metric regardless of which scenario applies -- uniform sampling already saturates it.

## Caveat

n=19 is the full set of annotated anomalous videos available. Each video is 5.3pp of any aggregate. Pooled AUC is not computed -- the 150 Normal videos have no local .mp4, and pooled AUC is confounded regardless (90.7% of negative pairs come from other videos).
