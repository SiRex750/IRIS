# 03_run_production_smoke

**Purpose:** One-video, one-question smoke test through ingest()+query() on the canonical spine.

**CLI signature:** `python commands/03_run_production_smoke.py --config --split --out-dir [--out-dir <path>]`
(refuses reuse of an existing --out-dir; prints git SHA + config hash; tees to a timestamped log)

**Status:** NOT EXECUTED (would decode video/run models -- forbidden in this task)
