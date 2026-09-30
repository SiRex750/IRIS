# Paper 3 — Pre-registration: VIRAT analysis
Date: 2026-09-30. Written after data collection (runner commit e9e0b81, 1242/1242 records OK, RAFT on 54/54 clips) and before any metric was computed. Sintel pre-registration: 0bf4cac; Sintel results: bb8122c. Code at this commit: e9e0b81a017f833d8fda28ee8d9a9494ee0d1d47.

## Question
On real CCTV footage, do codec-MV reuse gates make more unsafe or different decisions when encoder settings change, as they did on Sintel?

## Data
54 VIRAT clips (CCTV 01), frames 0-299, 22 encoded arms + the camera's own encode (NATIVE), as collected. No clip is dropped after seeing results. VIRAT_S_000002 (1080p, B-frames) is included and also reported without.
Scenes: clips sharing the VIRAT_S_XXXXYY prefix come from the same camera scene; all results are also reported per scene.

## Definitions (same as the Sintel sweep unless stated; tau = 1 px primary, 0.5 and 2 reported)
- Cell: 4x4 on frame t. Gate: reuse if |MV| < tau; no vector (intra) = recompute.
- Frames: P-frames with d = 1 (bf 0 arms); for B arms, B-frame vectors (naive). I-frames excluded.
- Pseudo-GT: RAFT flow forward-splatted onto frame t's grid (gt_grid style); a cell is valid if its RAFT cover >= 0.5. RAFT is an estimate, not ground truth; it is called "RAFT-estimated" everywhere.
- STALE (RAFT) = valid cells reused with |RAFT| > 2 px, divided by valid cells.
- FLIP = share of in-frame cells whose gate decision differs from x264 CRF 12 on the same frame and cell (no RAFT needed).
- ELIGIBLE clip (for STALE predictions): at least 0.1% of its valid cell-frames have |RAFT| > 2 px. Computed from RAFT and the source only, before any gate metric. If fewer than 20 clips are eligible, V1-stale and V2 become descriptive only.

## V0 — RAFT validity gate (checked first)
On x264 CRF 12, cells with |RAFT| > 2 px and source texture in the top quartile (mean luma gradient, quartiles over all clips): median |MV - RAFT| < 1.5 px in >= 75% of eligible clips.
If V0 fails: all RAFT-based results (V1-stale, V2) are reported as unreliable and carry no verdict; FLIP results stand.

## Predictions (counts over eligible clips for STALE, over all 54 for FLIP)
V1-stale Quantiser. STALE(worst) - STALE(best) > 0.0005: x264 CRF 45 vs 12 in >= 75% of eligible clips; NVENC QP 45 vs 18 in >= 75%.
V1-flip Quantiser. FLIP(x264 CRF 45) > FLIP(x264 CRF 18) in >= 75% of clips, and Spearman(CRF, FLIP) over CRF 18..45 >= 0.8 in >= 70% of clips. NVENC: FLIP(QP 45) > FLIP(QP 18) in >= 75%.
V2 B-frames at matched QP. STALE(B vectors) - STALE(bf 0, d = 1) > 0.0005: x264 QP 24 pair in >= 75% of eligible clips; NVENC QP 28 B=P pair in >= 65%.
Descriptive only (no verdicts): mpeg4 series, NATIVE encodes, source group (h264 ~1 Mbps vs mpeg4 ~8.4 Mbps), missed reuse, tau 0.5 and 2, a pixel-change proxy (reused cells whose source mean abs luma change > 8 grey levels).

## Kill rules
- V1 fails for BOTH x264 and NVENC (stale and flip): the Sintel quantiser effect does not carry over to CCTV; the paper says so.
- V1 fails for one encoder: stated as encoder-specific.
- V2 fails for an encoder: the B-frame claim is Sintel-only for that encoder.
- V0 fails: VIRAT supports only the FLIP claims.

## Rules
Thresholds, arms, frames and clips are fixed by this file. No tuning after seeing results. A bug means: fix, record in p3/virat/DEVIATIONS.md, rerun the whole analysis, report both. Per-clip results are primary; per-scene tables are also required; no pooled means in headline tables.
