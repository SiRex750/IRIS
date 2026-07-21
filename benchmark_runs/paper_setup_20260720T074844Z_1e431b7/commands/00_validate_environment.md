# 00_validate_environment

**Purpose:** Check python/torch/faiss/networkx import, GPU presence, ffmpeg/av availability, pip freeze snapshot.

**CLI signature:** `python commands/00_validate_environment.py N/A (no split) [--out-dir <path>]`
(refuses reuse of an existing --out-dir; prints git SHA + config hash; tees to a timestamped log)

**Status:** EXECUTED_DURING_SETUP (dry-run only: environment.json captured)
