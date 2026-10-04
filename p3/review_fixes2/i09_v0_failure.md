# Item 9: the V0 failure in the confirmatory test (supplementary)

- Failing clip(s): **VIRAT_S_010005_02_000177_000203** (V0 median |MV - RAFT| 1.840 px vs the 1.5 px limit; the other 26 eligible clips: median 0.558, max 1.174 px). V0 as a whole: 26/27 >= 21 -> PASS.
- Why it stays: V0 is a test-level gate (>= 75% of eligible clips), not a per-clip exclusion rule; the pre-registered analysed set is fixed by eligibility (>= 2,000 moving cell-frames) alone, and removing a clip after seeing V0 would be a post-hoc exclusion. Its RAFT-based numbers are flagged, and the 26-clip recount below shows the verdicts do not depend on it.

## V0 cells of the failing clip vs the other eligible clips (recomputed; V0 reproduced to 2e-3 px)

|  | V0 cells | median err px | err > 3 px | median |RAFT| px | median |MV| px | |RAFT| > 8 px | FB-consistent share | median err, FB-consistent | median err, FB-inconsistent |
|---|---|---|---|---|---|---|---|---|---|
| VIRAT_S_010005_02_000177_000203 | 5568 | 1.840 | 0.335 | 5.657 | 6.844 | 0.188 | 0.481 | 0.828 | 3.350 |
| others (median of 26) | 4565 | 0.558 | 0.040 | 2.673 | 2.778 | 0.000 | 0.802 | 0.473 | 1.241 |

Reading: the clip's V0 cells move faster than in any other eligible clip (median |RAFT| above, a large share above 8 px), and about half of them fail RAFT's own forward-backward check (item 8b). On its FB-consistent V0 cells the median |MV - RAFT| is below the 1.5 px limit; the excess error sits in the FB-inconsistent cells, i.e. where RAFT itself is unreliable. Descriptive diagnosis, not a re-test of V0.

## C1-C4 with and without the failing clip (tau = 1; sensitivity, not a verdict)

| prediction | 27 clips | without it | median diff 27 | median diff 26 | its own diff |
|---|---|---|---|---|---|
| C1 x264 CRF 45 - CRF 12 | 27/27 (need 22) | 26/26 (need 21) | 0.1504 | 0.1670 | 0.0983 |
| C2 NVENC QP 45 - QP 18 | 27/27 (need 21) | 26/26 (need 20) | 0.0441 | 0.0448 | 0.0197 |
| C3 x264 QP 24 B - bf 0 | 27/27 (need 22) | 26/26 (need 21) | 0.1196 | 0.1189 | 0.1551 |
| C4 NVENC QP 28 B=P - bf 0 | 27/27 (need 21) | 26/26 (need 20) | 0.1281 | 0.1265 | 0.1494 |
