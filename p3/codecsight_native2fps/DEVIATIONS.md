# Deviations: native-rate encode, 2 fps sampling (reading B)

## 1. analyze.py crash in the gate stage (bug fix, no value seen)
- Run 1 (code cbd6be6, 2026-10-18 01:41): every clip's gate inputs were computed, then `run_gates` raised
  `TypeError: 'method' object is not iterable` at `for c in gt.clip`: `gt.clip` is pandas' `DataFrame.clip` method,
  not the `clip` column. The crash happened before any V0 median, false-move rate, pass flag or count was computed
  or written; gates.csv / gates.md / gates.json were not created and no policy code ran. The console log
  (analyze.console.log of run 1, kept as analyze.run1.console.log) holds only per-clip run times.
- Fix: `gt["clip"]` in the two list comprehensions. Nothing else changed. Run 2 reruns everything (§9).

## 2. Run 2 stopped by hand during the policy stage (no deviation in method)
- Run 2 (code d9251bf, 2026-10-18 ~01:47): stages 1-3 finished and wrote gates.csv / gates.md / gates.json /
  drift_by_lag.csv (counted N = 25; V0 PASS 24/25; drift gate FAIL 0/25). The run was then stopped on request
  (~01:56) during the policy stage, before any clip finished it; no policy output was written.
- Run 3 (same code) reruns everything from the start; its gate files replace run 2's. Run 2's gate results are
  kept in this file above for comparison with run 3.
- Run 3 (02:08-02:33) finished; its gates.json is identical to run 2's (gates.run2.json). All results are from run 3.
