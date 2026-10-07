# p3/review_fixes3: follow-ups to the third internal review of the Paper 3 draft

**Reporting and exploratory analysis on saved data only. Nothing here is a new verdict.** No encoding, MV extraction or RAFT was run; the scripts only read p3/sweep/results.csv, p3/PREREGISTRATION.md, p3/sweep/independent_verdicts.txt, p3/review_fixes/sintel_stale_mov.csv and p3/virat_confirm (results.csv, eligibility_v0.csv). No existing results file is modified. Every number in this folder is printed by a script; this README is written by `make_readme.py`.

| item | script | output |
|---|---|---|
| R1. The two "8.7%" values; the "22/23" count | `r1_eight_point_seven.py` | `r1_eight_point_seven.md/.csv` |
| R2. The MPEG-4 P1 row | `r2_mpeg4_p1.py` | `r2_mpeg4_p1.md/.csv` |
| R3. Fold changes | `r3_fold_changes.py` | `r3_fold_changes.md/.csv` |
| R4. Bootstrap 95% CIs (exploratory, not pre-registered) | `r4_bootstrap_ci.py` | `r4_bootstrap_ci.md/.csv` |
| R5. tau = 0.5 / 2 appendix | `r5_tau_appendix.py` | `r5_tau_appendix.md/.csv` |
| README + draft-number check | `make_readme.py` | `README.md` |

Run from this folder: `python r1_eight_point_seven.py`, ..., `python r5_tau_appendix.py`, then `python make_readme.py`. `_common.py` holds the shared readers.

## Draft numbers checked

Each draft number is recomputed in the draft's precision (Sintel, FINAL pass, tau = 1, median over 23 sequences unless stated).

| draft | taken to be | recomputed | result |
|---|---|---|---|
| 0.34% | x264 at ~41 dB (x264_crf23) median stale/valid | 0.34% | match |
| 3.3% | NVENC at ~41 dB (nvenc_qp28) median stale/valid | 3.3% | match |
| 16% | MPEG-4 at ~41 dB (mpeg4_q4) median stale/valid | 16% | match |
| 0.55→4.4% | x264 QP 24 bf 0 -> B (pb1, B_naive) medians | 0.55→4.4% | match |
| 3.3→7.0% | NVENC QP 28 bf 0 -> B=P (bqeq, B_naive) medians | 3.3→7.0% | match |
| 3.3→8.7% | NVENC QP 28 bf 0 -> bf 2 default B QP (B_naive) medians | 3.3→8.7% | match |
| 22/23 | P2 NVENC QP 28 B=P - bf 0 > 0.001 (pre-registered count) | 22/23 | match |

All draft numbers match. Note on "22/23": it is the > 0.001 count; with a > 0.01 rule the B=P arm is 16/23, the default-B-QP arm 19/23 (22/23 at > 0.001, 22/23 at > 0), and Sintel STALE_MOV B=P 21/23 (22/23 at > 0).

## Key numbers

- **R1:** stale/valid median NVENC QP 28 bf 0 = 0.0328, bf 2 default B QP (B_naive) = 0.0867, B=P = 0.0697; Sintel STALE_MOV B=P pair = 0.0403 -> 0.0874. The two "8.7%" are 0.0867 (stale/valid, default B QP) and 0.0874 (STALE_MOV, B=P): different numbers. Default-B-QP minus bf 0: 19/23 > 0.01, 22/23 > 0.001, 22/23 > 0; "22/23" confirmed as the pre-registered B=P count (> 0.001).
- **R2:** MPEG-4 P1 was pre-registered as reported-only, not a kill-rule row: stale/valid(mpeg4_q31) - stale/valid(mpeg4_q2) > 0.001, threshold >= 16/23, count 19/23, PASS (matches independent_verdicts.txt: yes).
- **R3:** x264 CRF 12 -> 45: 0.0015 -> 0.1313 (88.9x of medians, 37.3x per-seq); NVENC QP 18 -> 45: 0.0042 -> 0.2289 (54.0x of medians, 46.5x per-seq); x264 QP 24: bf 0 -> B (pb1, B_naive): 0.0055 -> 0.0445 (8.1x of medians, 9.2x per-seq); NVENC QP 28: bf 0 -> B=P (bqeq, B_naive): 0.0328 -> 0.0697 (2.1x of medians, 2.1x per-seq); matched ~41 dB: x264 x264_crf23 -> NVENC nvenc_qp28: 0.0034 -> 0.0328 (9.5x of medians, 5.2x per-seq); matched ~41 dB: x264 x264_crf23 -> MPEG-4 mpeg4_q4: 0.0034 -> 0.1611 (46.9x of medians, 18.7x per-seq); matched ~41 dB: NVENC nvenc_qp28 -> MPEG-4 mpeg4_q4: 0.0328 -> 0.1611 (4.9x of medians, 3.6x per-seq).
- **R4 (exploratory):** Sintel median paired diff [95% CI]: P1 x264 CRF 45 - CRF 12 0.1263 [0.0302, 0.2503]; P1 NVENC QP 45 - QP 18 0.2176 [0.0511, 0.3441]; P1 MPEG-4 q31 - q2 (reported, not kill) 0.0250 [0.0152, 0.0472]; P2 x264 QP 24 B - bf 0 0.0386 [0.0269, 0.0453]; P2 NVENC QP 28 B=P - bf 0 0.0222 [0.0125, 0.0318]. CCTV STALE_MOV: C1 0.1504 scene [0.1125, 0.2557] / clip [0.1220, 0.2200]; C2 0.0441 scene [0.0330, 0.0860] / clip [0.0307, 0.0860]; C3 0.1196 scene [0.1075, 0.1716] / clip [0.1075, 0.1608]; C4 0.1281 scene [0.1200, 0.1742] / clip [0.1182, 0.1622]. 19/19 intervals where the zero test applies exclude 0. The P1 Spearman interval [1.0000, 1.0000] is degenerate: per-sequence rho = 1 in 19/23 sequences (minimum 0.964), so report it as that count, not as a CI.
- **R5:** Sintel x264 CRF 12 -> 45 tau 0.5: 0.0008->0.1149, rise>0.01 21/23; Sintel x264 CRF 12 -> 45 tau 2: 0.0090->0.1521, rise>0.01 23/23; Sintel NVENC QP 18 -> 45 tau 0.5: 0.0033->0.2229, rise>0.01 22/23; Sintel NVENC QP 18 -> 45 tau 2: 0.0121->0.2661, rise>0.01 23/23; Sintel x264 QP 24: bf 0 -> B (pb1, B_naive) tau 0.5: 0.0050->0.0438, rise>0.01 22/23; Sintel x264 QP 24: bf 0 -> B (pb1, B_naive) tau 2: 0.0176->0.0604, rise>0.01 21/23; Sintel NVENC QP 28: bf 0 -> B=P (bqeq, B_naive) tau 0.5: 0.0318->0.0682, rise>0.01 17/23; Sintel NVENC QP 28: bf 0 -> B=P (bqeq, B_naive) tau 2: 0.0363->0.0740, rise>0.01 17/23; Sintel matched ~41 dB: x264 x264_crf23 -> NVENC nvenc_qp28 tau 0.5: 0.0021->0.0318, rise>0.01 13/23; Sintel matched ~41 dB: x264 x264_crf23 -> NVENC nvenc_qp28 tau 2: 0.0131->0.0363, rise>0.01 13/23; Sintel matched ~41 dB: x264 x264_crf23 -> MPEG-4 mpeg4_q4 tau 0.5: 0.0021->0.1602, rise>0.01 20/23; Sintel matched ~41 dB: x264 x264_crf23 -> MPEG-4 mpeg4_q4 tau 2: 0.0131->0.1667, rise>0.01 20/23; Sintel matched ~41 dB: NVENC nvenc_qp28 -> MPEG-4 mpeg4_q4 tau 0.5: 0.0318->0.1602, rise>0.01 20/23; Sintel matched ~41 dB: NVENC nvenc_qp28 -> MPEG-4 mpeg4_q4 tau 2: 0.0363->0.1667, rise>0.01 20/23; CCTV C1 tau 0.5: 0.0311->0.2212, rise>0.01 27/27; CCTV C1 tau 2: 0.1588->0.3784, rise>0.01 27/27; CCTV C2 tau 0.5: 0.0279->0.0851, rise>0.01 27/27; CCTV C2 tau 2: 0.1443->0.2438, rise>0.01 25/27; CCTV C3 tau 0.5: 0.0304->0.1441, rise>0.01 27/27; CCTV C3 tau 2: 0.1539->0.3323, rise>0.01 27/27; CCTV C4 tau 0.5: 0.0270->0.1435, rise>0.01 27/27; CCTV C4 tau 2: 0.1553->0.2989, rise>0.01 27/27. Matched-quality codec contrasts: not available on CCTV.
