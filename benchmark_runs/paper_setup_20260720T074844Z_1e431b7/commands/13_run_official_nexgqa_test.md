# 13_run_official_nexgqa_test

**Purpose:** Execute the official NExT-GQA test split once.

**CLI signature:** `python commands/13_run_official_nexgqa_test.py --config --split=official_test --out-dir [--out-dir <path>]`
(refuses reuse of an existing --out-dir; prints git SHA + config hash; tees to a timestamped log)

**Status:** NOT EXECUTED and currently BLOCKED by split_guard.py (official_test partition unpopulated)
