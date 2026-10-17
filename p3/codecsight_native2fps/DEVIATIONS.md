# Deviations: native-rate encode, 2 fps sampling (reading B)

## 1. analyze.py crash in the gate stage (bug fix, no value seen)
- Run 1 (code cbd6be6, 2026-10-18 01:41): every clip's gate inputs were computed, then `run_gates` raised
  `TypeError: 'method' object is not iterable` at `for c in gt.clip`: `gt.clip` is pandas' `DataFrame.clip` method,
  not the `clip` column. The crash happened before any V0 median, false-move rate, pass flag or count was computed
  or written; gates.csv / gates.md / gates.json were not created and no policy code ran. The console log
  (analyze.console.log of run 1, kept as analyze.run1.console.log) holds only per-clip run times.
- Fix: `gt["clip"]` in the two list comprehensions. Nothing else changed. Run 2 reruns everything (§9).
