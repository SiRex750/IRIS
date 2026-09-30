# Paper 3 — VIRAT: deviations from the pre-registration (45f3e98)

## 1. Frame-threaded MV export on the last frame of B-frame streams (found before any VIRAT metric was computed)

Found: 2026-09-30, by the analysis self-test (analyze_virat.py `selftest`, which recomputes Sintel sweep counters from
the sweep's saved encodes and compares them with p3/sweep/results.csv). The `d1` group of the two B-frame arms did
not match, and differed between two runs of the self-test.

Cause: `lib.mvs.extract_mvs` decodes with `thread_type="AUTO"` (FFmpeg frame threading). For a stream whose display
order ends `...B B P P`, the motion vectors exported for the LAST frame change from run to run. With
`thread_type=None` (single-threaded) the export is reproducible, and every other frame is identical to the threaded
export. P-only streams (bf 0 arms, NATIVE) were not affected in any check.

Scope in the VIRAT data (check_mv_threading.py → mv_threading_check.csv; all 1242 arm files re-extracted
single-threaded and compared with the stored arrays): 9 files differ, each only at frame 299 (a P-frame with d = 1):
x264_qp24_bf2_pb1 on VIRAT_S_000002, VIRAT_S_000205_02, VIRAT_S_010000_01, VIRAT_S_010000_06, VIRAT_S_010002_05,
VIRAT_S_010004_03; nvenc_qp28_bf2 on VIRAT_S_010000_01; nvenc_qp28_bf2_bqeq on VIRAT_S_010000_01, VIRAT_S_010002_07.
Frame 299 of a B-frame arm is in that arm's `d1` group. No pre-registered verdict uses that group (V2 compares the
`B` group of the B-frame arms with the `d1` group of the bf 0 arms), so no verdict can change; FLIP/STALE rows of the
`d1` group of B-frame arms in results.csv can.

Fix: `check_mv_threading.py --fix` writes the single-threaded arrays for those 9 files to data/<clip>/mv_fix/<arm>.npz;
the collected files are left unchanged. The V0 re-extraction of x264 CRF 12 vectors uses `thread_type=None`.
Both analyses are run and reported, as the pre-registration requires:
- primary: `analyze_virat.py --mv-fix` → results.csv, verdicts*.csv, summary.md
- as collected: `analyze_virat.py` → results_as_collected.csv, verdicts*_as_collected.csv, summary_as_collected.md
summary.md states whether the two verdict tables are identical.

Sintel: the same export was used in the Sintel sweep (bb8122c). The self-test shows the Sintel `d1` group of the B-frame
arms on alley_1 is affected the same way; the Sintel predictions (P2 uses the `B` group) do not use that group. Not
changed here; to be recorded in a Sintel deviations note.

## 2. Implementation choices the pre-registration did not fix (stated here, decided before any metric was computed)

- Texture quartile population (V0): all cells of source frames 1..299 of all 54 clips, each cell-frame counted once;
  75th percentile from a 0.001-wide histogram. The same threshold is used for the table without VIRAT_S_000002.
- V0 cells per clip: x264 CRF 12, `d1` frames, RAFT-valid cells with |RAFT| > 2 px, a vector present, texture >= the
  threshold; median |MV - RAFT| over those cells of the clip. A clip with no such cell does not count as passing.
- "Reused cells whose source mean abs luma change > 8": per-cell mean |Y_t - Y_(t-1)| recomputed exactly from the
  decoded source (the stored uint8 is rounded); reported as a share of in-frame cells and of reused cells.
- Kill rules: "V1 fails for an encoder (stale and flip)" is read both ways (all components fail / any component
  fails) and both are reported. A V1-stale without a verdict (V0 failed, or < 20 eligible) is not counted as a failure.
- "75% of N clips" means at least ceil(0.75 N) clips.
- Spearman(CRF, FLIP) is NaN (fails) if any FLIP value is missing or all six are equal.
