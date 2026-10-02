# Sintel sweep: original vs rerun with single-threaded MV extraction

Original: p3/sweep (bb8122c; frame-threaded MV export). Rerun: p3/sweep_rerun (same arms, sequences, settings and code except lib.mvs.extract_mvs single-threaded; p3/sweep/DEVIATIONS.md 1). Pre-registration 0bf4cac. Written by p3/sweep_rerun/compare.py.

- Rerun: 2026-10-02T01:30:14+0530 → 2026-10-02T03:32:09+0530, HEAD `7eb3f8aa93eadad7724b1d62199f84cfabf32f82`, code sha256 `0b6947b01bcfcd78…` (original `06054cafaf79df6b…`), runs 1081, failures 0, check mismatches 1 (original: 1081, 0, 1).
- Code files that differ (sha256): p3/lib/mvs.py

- Rerun summary.md written by summarize_sweep.py: ok
- Rerun independent_verdicts.txt written by p3/sweep/independent_verdicts.py.

## 1. Verdicts, original vs rerun (τ = 1, FINAL pass)

### summarize_sweep.py

| prediction | original value | original | rerun value | rerun | verdict change | value |
|---|---|---|---|---|---|---|
| P1 x264 CRF 45 vs 12 | 23/23 with diff > 0.001 | PASS | 23/23 with diff > 0.001 | PASS |  | same |
| P1 NVENC QP 45 vs 18 | 23/23 with diff > 0.001 | PASS | 23/23 with diff > 0.001 | PASS |  | same |
| P1 mpeg4 q 31 vs 2 (reported, not a kill rule) | 19/23 with diff > 0.001 | PASS | 19/23 with diff > 0.001 | PASS |  | same |
| P1 Spearman(CRF, stale/valid), 7 x264 CRF levels | 23/23 with rho >= 0.8 | PASS | 23/23 with rho >= 0.8 | PASS |  | same |
| P2 x264 QP 24: B naive minus bf0 d1 | 23/23 with diff > 0.001 | PASS | 23/23 with diff > 0.001 | PASS |  | same |
| P2 NVENC QP 28 B=P: B naive minus bf0 d1 | 22/23 with diff > 0.001 | PASS | 22/23 with diff > 0.001 | PASS |  | same |
| P2b abs(B scaled - B naive), x264 QP 24 pair | median 0.0009 | PASS | median 0.0009 | PASS |  | same |
| P2b abs(B scaled - B naive), NVENC QP 28 B=P pair | median 0.0005 | PASS | median 0.0005 | PASS |  | same |
| P3 ref 2 vs CRF 23 medium | median abs diff 0.00073 | PASS | median abs diff 0.00073 | PASS |  | same |
| P3 ref 3 vs CRF 23 medium | median abs diff 0.00045 | PASS | median abs diff 0.00045 | PASS |  | same |
| P3 keyint 30 vs CRF 23 medium | median abs diff 0.00012 | PASS | median abs diff 0.00012 | PASS |  | same |
| P3 umh 32 vs umh 16 | median abs diff 0.00029 | PASS | median abs diff 0.00029 | PASS |  | same |
| P3 umh 64 vs umh 16 | median abs diff 0.00023 | PASS | median abs diff 0.00023 | PASS |  | same |
| P2c median x264 B effect > median abs change of every P3 arm | median B effect 0.03859 | PASS | median B effect 0.03859 | PASS |  | same |

### independent_verdicts.py (chat auditor)

| prediction | original value | original | rerun value | rerun | verdict change | value |
|---|---|---|---|---|---|---|
| P1 x264 CRF45-CRF12 | 23/23 | PASS | 23/23 | PASS |  | same |
| P1 NVENC QP45-QP18 | 23/23 | PASS | 23/23 | PASS |  | same |
| P1 mpeg4 q31-q2 (not kill) | 19/23 | PASS | 19/23 | PASS |  | same |
| P1 Spearman(CRF, stale) >= 0.8 | 23/23 | PASS | 23/23 | PASS |  | same |
| P2 x264 QP24 B - bf0 | 23/23 | PASS | 23/23 | PASS |  | same |
| P2 NVENC QP28 B=P - bf0 | 22/23 | PASS | 22/23 | PASS |  | same |
| P2b x264 median \|scaled-naive\| | 0.0009 | PASS | 0.0009 | PASS |  | same |
| P2b NVENC median \|scaled-naive\| | 0.0005 | PASS | 0.0005 | PASS |  | same |
| P3 ref 2 | 0.00073 | PASS | 0.00073 | PASS |  | same |
| P3 ref 3 | 0.00045 | PASS | 0.00045 | PASS |  | same |
| P3 keyint 30 | 0.00012 | PASS | 0.00012 | PASS |  | same |
| P3 umh 32 | 0.00029 | PASS | 0.00029 | PASS |  | same |
| P3 umh 64 | 0.00023 | PASS | 0.00023 | PASS |  | same |
| P2c median x264 B effect > every P3 median | 0.03859 vs max 0.00073 | PASS | 0.03859 vs max 0.00073 | PASS |  | same |

**Any verdict changed: NO** (summarize_sweep: 14 rows; independent: 14 rows).

## 2. Largest per-sequence change (FINAL, τ = 1)

Per prediction: the largest |rerun − original| stale/valid over every arm/group the prediction uses and every sequence; and the largest change in the per-sequence quantity it counts.

| prediction | max abs Δ stale/valid | where | max abs Δ counted quantity | sequence | sequences changed |
|---|---|---|---|---|---|
| P1 x264 CRF 45 vs 12 | 0 | – | 0 | – | 0/23 |
| P1 NVENC QP 45 vs 18 | 0 | – | 0 | – | 0/23 |
| P1 mpeg4 q 31 vs 2 (reported, not a kill rule) | 0 | – | 0 | – | 0/23 |
| P1 Spearman(CRF, stale/valid), 7 x264 CRF levels | 0 | – | 0 | – | 0/23 |
| P2 x264 QP 24: B naive minus bf0 d1 | 0 | – | 0 | – | 0/23 |
| P2 NVENC QP 28 B=P: B naive minus bf0 d1 | 0 | – | 0 | – | 0/23 |
| P2b abs(B scaled - B naive), x264 QP 24 pair | 0 | – | 0 | – | 0/23 |
| P2b abs(B scaled - B naive), NVENC QP 28 B=P pair | 0 | – | 0 | – | 0/23 |
| P3 ref 2 vs CRF 23 medium | 0 | – | 0 | – | 0/23 |
| P3 ref 3 vs CRF 23 medium | 0 | – | 0 | – | 0/23 |
| P3 keyint 30 vs CRF 23 medium | 0 | – | 0 | – | 0/23 |
| P3 umh 32 vs umh 16 | 0 | – | 0 | – | 0/23 |
| P3 umh 64 vs umh 16 | 0 | – | 0 | – | 0/23 |
| P2c median x264 B effect > median abs change of every P3 arm | 0 | – | 0 | – | 0/23 |

## 3. results.csv, row by row (every column, exact)

- Rows: original 4845, rerun 4845; only in one run: 0.
- Rows with any differing value: 741/4845; columns involved: block_excluded_area_share (345), epe_gt3_area_pct (345), epe_mean (345), epe_median (330), false_static_cells (333), false_static_of_valid (333), false_static_of_zero (207), flip_ownref (345), flip_ownref_cells (345), flip_x264ref (345), flip_x264ref_cells (345), mean_p_qp (495), missed_cells (173), missed_of_valid (173), mv_coverage_pct (336), n_vectors (345), precision (179), recall (173), reuse_cells (345), stale_cells (330), stale_cells_16_32 (171), stale_cells_2_8 (283), stale_cells_32_inf (126), stale_cells_8_16 (260), stale_of_reuse (229), stale_of_valid (330), stale_of_valid_16_32 (171), stale_of_valid_2_8 (283), stale_of_valid_32_inf (126), stale_of_valid_8_16 (260), tp_cells (173), zero_cells (345), zero_mv_share (345).

| pass | encode_id | group | rows (seq × τ) |
|---|---|---|---|
| clean | nvenc_qp28_bf2_bqeq | d1 | 54 |
| clean | nvenc_qp28_bf2_bqeq | dgt1_naive | 3 |
| clean | nvenc_qp28_bf2_bqeq | dgt1_scaled | 3 |
| clean | x264_qp24_bf2_pb1 | d1 | 60 |
| clean | x264_qp24_bf2_pb1 | dgt1_naive | 6 |
| clean | x264_qp24_bf2_pb1 | dgt1_scaled | 6 |
| final | nvenc_qp28_bf2 | B_naive | 51 |
| final | nvenc_qp28_bf2 | B_scaled | 51 |
| final | nvenc_qp28_bf2 | d1 | 51 |
| final | nvenc_qp28_bf2 | dgt1_naive | 51 |
| final | nvenc_qp28_bf2 | dgt1_scaled | 51 |
| final | nvenc_qp28_bf2_bqeq | d1 | 54 |
| final | x264_crf23_bf2 | B_naive | 48 |
| final | x264_crf23_bf2 | B_scaled | 48 |
| final | x264_crf23_bf2 | d1 | 48 |
| final | x264_crf23_bf2 | dgt1_naive | 48 |
| final | x264_crf23_bf2 | dgt1_scaled | 48 |
| final | x264_qp24_bf2_pb1 | d1 | 60 |

## 4. Encodes and block records

- Encodes byte-identical (sha256): 1081/1081; different: 0; missing in rerun: 0; only in rerun: 0.
- Block record files compared: 2162 (block + frame-type files present in both runs); differing: 224 (blocks_comparison.csv).
- Expected from p3/sweep/DEVIATIONS.md 1: only the last frame of B-frame arms; up to 112 block files (the threaded export can vary between runs, so the 26 matching B files may also differ).

| arm | files differing | differing frames (distinct) |
|---|---|---|
| nvenc_qp28_bf2 | 34 | 19 49 |
| nvenc_qp28_bf2_bqeq | 74 | 19 39 49 |
| x264_crf23_bf2 | 32 | 19 49 |
| x264_qp24_bf2_pb1 | 84 | 19 32 39 49 |

Runtime of compare.py: 89s.
