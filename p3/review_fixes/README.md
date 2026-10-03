# p3/review_fixes: follow-ups to the internal review of the Paper 3 draft

**Everything here is reporting and exploratory analysis on saved data. Nothing here is a new verdict.** No
pre-registered verdict changes, and no existing results file is modified. The scripts only read p3/sweep,
p3/sweep_rerun, p3/virat, p3/virat_confirm and the Sintel ground truth. No new encodes or MV re-extraction were
needed: the saved bf 0 block records are identical to the single-threaded rerun.

| item | script | output |
|---|---|---|
| A. Per-scene tables, both CCTV tests | `a_per_scene.py` | `per_scene.md`, `per_scene.csv` |
| B. Sintel STALE_MOV (stale share of moving cells) | `b_sintel_stale_mov.py` | `sintel_stale_mov.md`, `sintel_stale_mov.csv` |
| C. x264 ref 2 / ref 3 reference usage inferred from vectors | `c_ref_usage.py` | `ref_usage.md`, `ref_usage.csv` |
| D. Code availability of CodecSight, MVTrack, CMC (web search only) | – | `code_availability.md` |

`_common.py` loads the Sintel cell GT the same way as the sweep. `_cache/` (GT and texture arrays) is not committed.
Run each script from this folder: `python a_per_scene.py`, and so on.
