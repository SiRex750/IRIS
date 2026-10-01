# Paper 3 — Sintel sweep: deviations from the pre-registration (0bf4cac)

## 1. Frame-threaded MV export on the last frame of B-frame streams (found after the sweep, during the VIRAT analysis)

Found: 2026-09-30, by the VIRAT analysis self-test (p3/virat/analyze_virat.py `selftest`, which recomputes Sintel sweep
counters for alley_1 from the sweep's saved encodes and compares them with results.csv). See p3/virat/DEVIATIONS.md 1.

Cause: `lib.mvs.extract_mvs` decodes with `thread_type="AUTO"` (FFmpeg frame threading), and the sweep
(run_pilot_v2.run_one) used that default. For a B-frame stream, the motion vectors exported for the LAST frame can
change from run to run. With `thread_type=None` (single-threaded) the export is reproducible.

Scope in the Sintel sweep (checked 2026-10-02 the same way as p3/virat/check_mv_threading.py): every sweep encode
(encodes/final, encodes/clean; 1081 arm files) was re-extracted with `thread_type=None` and compared, frame by frame,
with the block records the sweep stored (blocks/<pass>/<arm>__<seq>.parquet: x, y, w, h, dir, mv_x, mv_y) and with the
stored frame types (`__frames.parquet`).
- Frame types: identical in all 1081 files.
- bf 0 arms (943 files: all x264 bf 0, NVENC bf 0 and mpeg4 arms): 0 files differ.
- B-frame arms (138 files): 112 differ, each at exactly one frame, the last frame of the sequence, always a P-frame.

| pass  | arm                  | files differing |
|-------|----------------------|-----------------|
| final | x264_qp24_bf2_pb1    | 20/23 |
| final | x264_crf23_bf2       | 16/23 |
| final | nvenc_qp28_bf2       | 17/23 |
| final | nvenc_qp28_bf2_bqeq  | 18/23 |
| clean | x264_qp24_bf2_pb1    | 22/23 |
| clean | nvenc_qp28_bf2_bqeq  | 19/23 |
| total |                      | 112/138 |

Group of the differing frame (from `__frames.parquet`):
- `d1` in 109 files (all 71 FINAL-pass files and 38 CLEAN-pass files): the stream ends `...B B P P`.
- `dgt1` in 3 CLEAN-pass files, where the last P-frame's previous reference is 2 or 3 frames back:
  x264_qp24_bf2_pb1 on ambush_4 (frame 32, ends `B P B P`) and on market_6 (frame 39, ends `P B B P`);
  nvenc_qp28_bf2_bqeq on market_6 (frame 39).

The 26 B-frame files that match are not shown to be immune: the threaded export can vary between runs, and they may
simply have matched this time.

Effect on the pre-registered predictions: none. No prediction uses the `d1` or `dgt1` group of a B-frame arm.
P1 and P3 use the `d1` / `d1(assumed)` groups of bf 0 arms; P2 uses the `B_naive` group of the B-frame arms against the
`d1` group of the bf 0 arm at the same QP; P2b uses `B_scaled` and `B_naive`; P2c combines P2 and P3. The flip
references (x264_crf12, nvenc_qp18, mpeg4_q2) are bf 0 arms, so flip counts of other arms are unaffected. Every
other frame, including every B-frame, matched exactly. The independent check
(independent_verdicts.py) reads the same groups. The Sintel verdicts are therefore unchanged.

What can change: in results.csv, the `d1` rows of the B-frame arms (descriptive only; summary.md already lists them as
not used by any prediction), the `dgt1_naive` / `dgt1_scaled` rows of the 3 CLEAN-pass files above, and the stored
block records of those last frames.

Rerun: the pre-registration's rule ("fix it, record it in p3/sweep/DEVIATIONS.md, rerun the WHOLE sweep, report both
runs") has not been followed for this bug. The sweep was not rerun, and results.csv is as collected.

From now on, all MV extraction in Paper 3 is single-threaded (`thread_type=None`).
