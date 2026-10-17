# CodecSight-matched 2 fps arm: deviations and observations
Pre-registration: `PREREGISTRATION.md` (cc2eaf9).

## 1. x264 CRF arms at 2 fps are not QP-equivalent to the native-rate CRF arms (observation, 2026-10-14)
Written before any metric of this arm. This is an observation; no definition changes.
- In the pilot (`pilot/manifest.json`, clip VIRAT_S_000207_01_000094_000156, 80 frames at 2 fps), x264 CRF gave a
  mean P-frame QP of 4.8 at CRF 12 (PSNR-Y 56.1 dB, 5,111 kbps) and 37.9 at CRF 45.
- At native rate the same CRF settings give P-frame QPs near the CRF value. The difference arises because x264's CRF
  rate control depends on the frame rate.
- The arm definitions are unchanged, and F1 still compares x264 CRF 45 with CRF 12. But the 2 fps CRF arms are not
  QP-equivalent to the native-rate CRF arms of the confirmatory test and the CodecSight simulation.
- The constant-QP arms (x264 QP 24 bf 0 and bf 2; every NVENC arm) are unaffected: their decoded QPs match the set
  values (checked per encode).

## 2. Collection stopped and resumed; manifest merged (process, 2026-10-14)
Written before any check or metric of this arm. No definition, data value or code changes.
- The full run (`collect.py`, code at b819381) was stopped by request at 14:41, after all 243 encodes and RAFT for 12
  of 27 clips. The clip in progress had no output file (outputs are written to a temporary file and renamed).
- The run was restarted with the same command and code. It skipped existing encodes and RAFT files and ran RAFT for
  the other 15 clips.
- A restart rewrites `manifest.json` with "skipped_exists" records for existing outputs. The encode records
  (checks, I-frame positions) and the first 12 RAFT records were kept from the first run's manifest, backed up as
  `manifest.encode_phase.json`. They were merged into `manifest.json`: per encode and per clip, the record with
  status ok. Both runs' start and end times are kept.

## 3. analyze.py crashed once before writing any result (bug, 2026-10-14)
- On its first run, analyze.py stopped at the common-truth summary with an AttributeError. It used `ctd.clip`, which
  is pandas' DataFrame.clip method, not the column. It was fixed to `ctd["clip"]` and the whole analysis was rerun.
- The crash came after per_unit.csv was written and before verdicts.md, verdicts.csv or any summary was written.
  So there is no earlier result to report. The rerun recomputed per_unit.csv from scratch.
- checks.py had already run, and R-chain had already failed, before analyze.py was first started.

## 4. R-chain diagnostic on synthetic motion (post-results diagnostic, 2026-10-17)
Written after all results; R-chain FAIL stands as the pre-registered outcome and this changes no gate, threshold or
result. `diag_chain.py` -> `diag_chain.txt`. It uses collect.py's `load_raft` and the same flow and chain composition
code as `collect.raft_clip`, on one real frame (VIRAT_S_000206_07_001501_001600, source frame 600, 1280x720),
12 composed 1 px steps (truth 12 px right).
- (a) Global integer shift, edge padding, interior pixels: median D_x 11.99 px, F_x 12.05 px, median r 0.018.
  Pass rule fixed before running (|median - 12| < 0.5 px for D_x and F_x): PASS, so no chain or RAFT bug was found.
- (b) 24x48 px patch moving 1 px/frame over the static frame, patch pixels: median D_x 12.10 px, F_x 12.22 px,
  median r 0.044; static background median |D| 0.01 px, |F| 0.10 px.
- On clean synthetic motion, global or small object, the chain is well below the 0.2 r threshold. So the FAIL on
  real clips is not explained by the composition code itself. F picks up a small y drift (+0.17 to +0.24 px over 12 steps).
