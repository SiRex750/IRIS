# Paper 3 — Pre-registration: Sintel sweep
Date: 2026-09-29. Written after pilots v1/v2 (exploratory) and the texture check; before any full-sweep data.
Code frozen at commit: d8a1d385e5265d025c7b6a39a0bd9b53219ef99c. Pilot evidence: commits b7392cb (v1), d8a1d38 (v2).

## Question
Do codec-MV reuse gates (reuse a cell if |MV| < tau) make more unsafe reuse decisions when encoder settings change, and which settings matter?

## Data
All 23 MPI-Sintel training sequences, all frames. FINAL pass = primary. CLEAN pass = secondary (quality series and B-frame pairs only).
No sequence is dropped after seeing results.

## Arms (all: 24 fps, yuv420p, keyint 250, scenecut 0, ref 1, bframes 0, threads 1 / one slice per frame, unless stated)
- x264 medium CRF 12, 18, 23, 28, 33, 38, 45
- x264 CRF 23 variants: ref 2; ref 3; keyint 30; preset ultrafast; preset veryslow; me umh merange 16 / 32 / 64; bframes 2 (b-adapt 0, b-pyramid none)
- x264 constant QP 24: bframes 0; bframes 2 (b-adapt 0, b-pyramid none, pbratio 1.0) — decoded B QP = P QP
- NVENC constqp p4 refs 1: QP 18, 23, 28, 33, 38, 45; QP 28 bf 2 (b_ref_mode disabled) default B QP; QP 28 bf 2 with b_qfactor 1, b_qoffset 0 (B QP = P QP)
- mpeg4 q 2, 4, 8, 16, 31 (threads 1)
Clean pass: the x264 CRF series, NVENC QP series, and the two matched-QP B-frame pairs only.

## Metrics (as in pilot v2 code; tau = 1 px is primary, 0.5 and 2 reported)
- Unit: 4x4 cell on frame t; GT from gt_grid (codec-independent); a cell is valid if gt_cover >= 0.5.
- Gate: reuse if |MV| < tau; a cell with no vector (intra) = recompute.
- PRIMARY: stale = valid cells reused with |GT| > 2 px, divided by valid cells ("stale/valid"). Final pass.
- Group for P1/P3: P-frame vectors, d = 1. Group for P2: B-frame vectors (naive) vs d = 1 P-frame vectors of the bf 0 arm at the same QP.
- Secondary, descriptive only (no predictions): missed reuse (GT-static cells, |GT| < 0.5, that the gate recomputes, divided by valid; clean pass), EPE, flip rate, precision/recall, preset and encoder comparisons.

## Predictions (per sequence; counts out of 23)
P1 Quantiser. stale/valid at worst quality minus at best quality > 0.001:
  x264 CRF 45 vs 12 in >= 18/23; NVENC QP 45 vs 18 in >= 18/23. (mpeg4 q 31 vs 2 reported, predicted >= 16/23, not part of the kill rule.)
  Also: Spearman(quality level, stale/valid) over the 7 x264 CRF levels >= 0.8 in >= 16/23.
P2 B-frames at matched QP. stale/valid(B-frame vectors) minus stale/valid(bf 0, d=1) > 0.001:
  x264 QP 24 pair in >= 18/23; NVENC QP 28 B=P pair in >= 16/23.
  P2b: median over sequences of |B_scaled - B_naive| stale/valid < 0.005 (effect is not reference distance).
  P2c: median B effect (x264 pair) > median |change| of every P3 arm.
P3 Small factors. For each of ref 2, ref 3, keyint 30 (vs CRF 23 medium) and umh 32, umh 64 (vs umh 16):
  median over sequences of |change in stale/valid| < 0.5 x median |change| from CRF 23 to CRF 33.

## Kill rules and what they mean
- P1 fails for BOTH x264 and NVENC: drop the claim that the quantiser changes gate safety; paper becomes a robustness result.
- P1 fails for one encoder only: claim is encoder-specific; stated as such.
- P2 fails (either encoder): drop the B-frame claim for that encoder; report as a null.
- P3 fails for an arm: that factor is reported as mattering; drop "only quantiser and B-frames matter".

## Rules
- Thresholds, arms, metrics and sequences are fixed by this file. No tuning after seeing results.
- If a bug is found: fix it, record it in p3/sweep/DEVIATIONS.md, rerun the WHOLE sweep, report both runs.
- Per-sequence results are primary; no pooled means in headline tables.
- VIRAT gets its own pre-registration after this sweep is audited.
