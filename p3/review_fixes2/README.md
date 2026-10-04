# p3/review_fixes2: follow-ups to the second internal review of the Paper 3 draft

**Everything here is supplementary or robustness analysis. Nothing here is a new verdict.** The pre-registered
verdicts (p3/sweep/summary.md, p3/sweep_rerun/COMPARISON.md, p3/virat_confirm/verdicts.csv) are unchanged, and no
existing results file is modified: the scripts only read p3/sweep, p3/sweep_rerun, p3/virat, p3/virat_confirm,
p3/review_fixes/_common.py (Sintel GT), the Sintel ground truth and the VIRAT source clips.

The only new compute is item 8(b): RAFT backward flow (t -> t-1) on the confirmatory test's 27 eligible clips,
same model / weights / frames as the forward pass (`i08b_raft_backward_compute.py`, run detached; log in
`i08b_raft_backward_compute.log`; flow in `_raft_bwd/`, not committed).

| item | script | output |
|---|---|---|
| 1. Configuration / sequence / encode count | `i01_config_count.py` | `i01_config_count.md` |
| 2. tau = 0.5 and 2 px, quality ends and matched-QP B pairs (Sintel + CCTV) | `i02_thresholds.py` | `i02_thresholds.md/.csv` |
| 3. Missed reuse per quality level (Sintel clean; CCTV descriptive) | `i03_missed_reuse.py` | `i03_missed_reuse.md/.csv` |
| 4. Sintel predictions on 19 sequences | `i04_19seq_verdicts.py` | `i04_19seq_verdicts.md/.csv` |
| 5. Excluded-cell share (Sintel GT cover, CCTV RAFT cover) | `i05_excluded_cells.py` | `i05_excluded_cells.md/.csv` |
| 6. B-frame cells with no past-pointing vector | `i06_bframe_no_past.py` | `i06_bframe_no_past.md/.csv` |
| 7. CCTV encode frame rate | `i07_cctv_fps.py` | `i07_cctv_fps.md/.csv` |
| 8a. C1-C4 on high-texture cells | `i08a_high_texture.py` | `i08a_high_texture.md`, `*_summary.csv` |
| 8b. C1-C4 on forward-backward consistent cells | `i08b_raft_backward_compute.py`, then `i08b_fb_consistency.py` | `i08b_fb_consistency.md`, `*_summary.csv`, `*_cells.csv` |
| 9. The V0 failure in the confirmatory test | `i09_v0_failure.py` (after 8a, 8b) | `i09_v0_failure.md/.csv` |
| 10. ANVIL author list (web lookup) | – | `i10_anvil_authors.md` |

Run each script from this folder (`python i01_config_count.py`, ...). `_common.py` holds the shared readers.
`_cache/` (texture and FB-consistency masks) and `_raft_bwd/` are not committed.
