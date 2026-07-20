# 04_run_val_tune_peak

**Purpose:** Sweep configs/sweep/peak_rule/*.json on val_tune.

**CLI signature:** `python commands/04_run_val_tune_peak.py --config --split=val_tune --out-dir [--out-dir <path>]`
(refuses reuse of an existing --out-dir; prints git SHA + config hash; tees to a timestamped log)

**Status:** NOT EXECUTED; also currently only 1/3 peak rules implemented
