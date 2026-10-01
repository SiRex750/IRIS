# Paper 3 — Pre-registration: confirmatory CCTV test
Date: 2026-10-02. Motivated by the exploratory VIRAT finding (p3/virat/independent_check.txt): wrong reuse measured as a share of MOVING content followed the Sintel pattern in 10/10 eligible clips. This test checks that on data never analysed.
## Data
VIRAT CCTV 01 clips, frames 300-599 (or to the end of the clip). Clips with fewer than 150 frames after frame 299 are excluded; VIRAT_S_000002 is excluded (1080p; RAFT impractical). Same knobs and RAFT as runner e9e0b81, single-threaded MV extraction. Arms: x264 CRF 12, 45; x264 QP 24 bf 0 and bf 2 (b-adapt 0, no pyramid, pbratio 1.0); NVENC QP 18, 23, 45; NVENC QP 28 bf 0 and bf 2 B=P.
## Metric
STALE_MOV = valid cells reused (|MV| < 1 px) with |RAFT| > 2 px, divided by valid cells with |RAFT| > 2 px (P-frames d=1 for bf 0 arms; B-frame vectors naive for B arms). Eligible clip: at least 2,000 moving cell-frames. V0 RAFT check as in 45f3e98. If fewer than 10 clips are eligible, all predictions are descriptive only.
## Predictions (over eligible clips)
C1 x264: STALE_MOV(CRF 45) - STALE_MOV(CRF 12) > 0.01 in >= 80%.
C2 NVENC: STALE_MOV(QP 45) - STALE_MOV(QP 18) > 0.01 in >= 75%.
C3 x264 B-frames: STALE_MOV(QP 24 B) - STALE_MOV(QP 24 bf 0) > 0.01 in >= 80%.
C4 NVENC B-frames: STALE_MOV(QP 28 B=P) - STALE_MOV(QP 28 bf 0) > 0.01 in >= 75%.
C5 NVENC flip vs its own QP 18 (all clips): FLIP(QP 45) > FLIP(QP 23) in >= 75%.
## Kill rules
C1 and C2 both fail: the quality effect does not carry to CCTV. C3 or C4 fails: the B-frame claim is Sintel-only for that encoder. C5 fails: NVENC decisions do not track quality on CCTV.
## Rules
Fixed by this file. No tuning after results. Bugs: fix, record in DEVIATIONS.md, rerun all, report both. Per-clip and per-scene results primary.
