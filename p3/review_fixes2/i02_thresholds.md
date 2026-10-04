# Item 2: thresholds 0.5 and 2 px (supplementary; verdicts stay at tau = 1)

## Sintel, FINAL pass, stale/valid (median over 23 sequences)

| comparison | tau | median low end / bf 0 | median high end / B | median diff | sequences with rise > 0.001 |
|---|---|---|---|---|---|
| x264 CRF 45 vs 12 | 0.5 | 0.0008 | 0.1149 | 0.1111 | 23/23 |
| x264 CRF 45 vs 12 | 1.0 | 0.0015 | 0.1313 | 0.1263 | 23/23 |
| x264 CRF 45 vs 12 | 2.0 | 0.0090 | 0.1521 | 0.1364 | 23/23 |
| NVENC QP 45 vs 18 | 0.5 | 0.0033 | 0.2229 | 0.2159 | 23/23 |
| NVENC QP 45 vs 18 | 1.0 | 0.0042 | 0.2289 | 0.2176 | 23/23 |
| NVENC QP 45 vs 18 | 2.0 | 0.0121 | 0.2661 | 0.2249 | 23/23 |
| x264 QP 24 B vs bf 0 | 0.5 | 0.0050 | 0.0438 | 0.0374 | 23/23 |
| x264 QP 24 B vs bf 0 | 1.0 | 0.0055 | 0.0445 | 0.0386 | 23/23 |
| x264 QP 24 B vs bf 0 | 2.0 | 0.0176 | 0.0604 | 0.0335 | 23/23 |
| NVENC QP 28 B=P vs bf 0 | 0.5 | 0.0318 | 0.0682 | 0.0219 | 21/23 |
| NVENC QP 28 B=P vs bf 0 | 1.0 | 0.0328 | 0.0697 | 0.0222 | 22/23 |
| NVENC QP 28 B=P vs bf 0 | 2.0 | 0.0363 | 0.0740 | 0.0207 | 21/23 |

## CCTV confirmatory test, STALE_MOV (median over the 27 eligible clips)

| comparison | tau | median low end / bf 0 | median high end / B | median diff | clips with rise > 0.01 |
|---|---|---|---|---|---|
| C1 x264 CRF 45 - CRF 12 | 0.5 | 0.0311 | 0.2212 | 0.1510 | 27/27 |
| C1 x264 CRF 45 - CRF 12 | 1.0 | 0.0547 | 0.2487 | 0.1504 | 27/27 |
| C1 x264 CRF 45 - CRF 12 | 2.0 | 0.1588 | 0.3784 | 0.1470 | 27/27 |
| C2 NVENC QP 45 - QP 18 | 0.5 | 0.0279 | 0.0851 | 0.0422 | 27/27 |
| C2 NVENC QP 45 - QP 18 | 1.0 | 0.0487 | 0.1005 | 0.0441 | 27/27 |
| C2 NVENC QP 45 - QP 18 | 2.0 | 0.1443 | 0.2438 | 0.0631 | 25/27 |
| C3 x264 QP 24 B - bf 0 | 0.5 | 0.0304 | 0.1441 | 0.1122 | 27/27 |
| C3 x264 QP 24 B - bf 0 | 1.0 | 0.0535 | 0.1924 | 0.1196 | 27/27 |
| C3 x264 QP 24 B - bf 0 | 2.0 | 0.1539 | 0.3323 | 0.1188 | 27/27 |
| C4 NVENC QP 28 B=P - bf 0 | 0.5 | 0.0270 | 0.1435 | 0.1233 | 27/27 |
| C4 NVENC QP 28 B=P - bf 0 | 1.0 | 0.0429 | 0.1701 | 0.1281 | 27/27 |
| C4 NVENC QP 28 B=P - bf 0 | 2.0 | 0.1553 | 0.2989 | 0.1370 | 27/27 |

tau = 1 rows reproduce the pre-registered numbers (p3/sweep/summary.md, p3/virat_confirm/verdicts.csv).
