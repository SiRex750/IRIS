# 14_run_efficiency_benchmark

**Purpose:** Populate latency/resource CSVs using timer_registry.json instrumentation.

**CLI signature:** `python commands/14_run_efficiency_benchmark.py --config --split --out-dir [--out-dir <path>]`
(refuses reuse of an existing --out-dir; prints git SHA + config hash; tees to a timestamped log)

**Status:** NOT EXECUTED; instrumentation not yet inserted into iris/ source
