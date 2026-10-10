# CodecSight policy simulation: deviations and bugs
Pre-registration: p3/codecsight_sim/PREREGISTRATION.md, commit 9f5b5a3 (unchanged). Entries are in the order found.
None of them changes a definition, threshold, arm or configuration of the pre-registration.

## 1. Validation 2a: wrong expected value in one synthetic test (test code, not analysis code)
- Found: 2026-10-07, first run of `test_policy.py`, before any real-data metric. Result: 22/23 passed.
- The failing check expected CCTV derived tokens (80 x 45 px) to overlap "12 or 13" cell rows.
  - 45 px = 11.25 cells, and a token's start offset inside a 4-px cell is 0, 1, 2 or 3 px. So every token spans
    floor(45i/4) to floor((45i + 44.99)/4), which is 12 rows, and every token has 12 x 20 = 240 cells.
  - `simulate.token_matrix` returned 240 for every token, which is correct.
- Fix: the expected value is now {240}. No analysis code changed. Re-run: 23/23 passed.

## 2. Validation 2b: CCTV reference lookup returned nothing (validation code)
- Found: first run of `simulate.py --repro`. Both CCTV rows showed saved value NaN and failed.
- Cause: `cr.clip == clip` compared pandas' `DataFrame.clip` method, not the `clip` column, so no saved row was
  selected. The computed side was not affected.
- Fix: `cr["clip"] == clip`. No other `.clip` attribute access exists in the code (checked with grep). Validation
  2b re-run in full.

## Rerun
Validation 2a and 2b were re-run in full after these fixes. No real-data metric of the simulation had been
computed before them, so there is no earlier run to report beside the final one.

## 3. Code changes after validation 2a/2b (no definition changed)
- 2026-10-07, before the full run. `simulate.py` main loop now runs the configurations in three phases, in the
  pre-registered order: primary, then tau 1, then the descriptive ones (tau 0.25, 16/32/64 px, 2 FPS a and b).
  Policy, truth, gate and geometry code are unchanged.
  - While making this edit, two string literals were broken (a newline inside "\n"). The file did not parse, so
    the code never ran and produced no output. Fixed before any run.
- `analyze.py` and `independent_verdicts.py` were added. The bootstrap code was checked on the main paper's saved
  values: it reproduces all 13 intervals of `p3/review_fixes3/r4_bootstrap_ci.csv` exactly (same seed and draws).
- Validation 2a (23/23) and 2b (12/12, ALL MATCH) were re-run with the final code (sha256 c536bf62…). The results
  are identical to the earlier runs; only the hash line changed. The full run used the same code hash.

## 4. Commit and code-hash notes (no code or result changed)
- On branch `sonu/p3-clean` the pre-registration commit is `b523bca`. Its content is identical to `9f5b5a3`, the
  commit cited in the outputs; only the commit hash differs.
- The code hash (sha256 c536bf62…) is computed over the files' LF bytes. On a CRLF checkout (e.g. Windows with
  `core.autocrlf=true`), convert the files to LF before hashing, or the hash will not match.
