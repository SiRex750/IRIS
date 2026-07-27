# 13_run_official_nexgqa_test

**Purpose:** Execute the official NExT-GQA test split once.

**CLI signature:** `python commands/13_run_official_nexgqa_test.py --config --split=official_test --out-dir [--out-dir <path>]`
(refuses reuse of an existing --out-dir; prints git SHA + config hash; tees to a timestamped log)

**Status:** NOT EXECUTED and currently BLOCKED by split_guard.py (official_test partition unpopulated)

---

## Correction (2026-07-27, prerun-fix pack D2)

The "official test split not available" premise behind this blocker is **stale**.
The root `dataset_manifest.json` (generated 2026-07-21, one day after this
document) records the official NExT-GQA test split as fully acquired and
structurally validated: 990/990 videos, 5553/5553 rows, `val_test_video_overlap:
0`. That acquisition is a fact about the pipeline's build process, not about any
one checkout.

This does **not** mean the blocker is resolved on every box. Re-running
`split_guard.guard_official_test_command("official_test")` against this
directory's own `split_manifest.json` (a smaller, separate, pre-acquisition
snapshot with `official_test.video_count == 0`) on 2026-07-27 still returns
`REJECTED / exit 3`, unchanged from the original status above — see
`tuning/prerun_fixes/D2_split_guard_status.md` for the exact command and output.

Populating this directory's `official_test` partition from the real 990-video
test split was **not done** as part of this fix pack: the box this check ran on
does not have a verifiable local copy of `test.csv`/`gsub_test.json` matching
`dataset_manifest.json`'s recorded hashes (`eval/data/nextqa/test.csv` is absent
entirely; the local `gsub_test.json`/`gsub_val.json` are a stale, byte-identical
pair from an earlier snapshot — see `tuning/prerun_fixes/D1_dataset_verification.json`).
Populating the partition from data that cannot itself be verified would just move
the same uncertainty one level down, so this was left as an open item rather than
silently "fixed." The correct fix is to run split_guard population on the box (or
from the artifact store) that actually holds the hash-verified 990-video test
files, then re-run the command above.
