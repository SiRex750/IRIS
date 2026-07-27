# D2 -- split_guard status (gpubox re-run)

**Box:** `worker-1`, Linux 6.8.0-134-generic, repo at `/home/ccbd/IRIS-1`
**Generated:** 2026-07-27

**Supersedes:** `tuning/prerun_fixes/D2_split_guard_status.md` (ran on a Windows
checkout without the real dataset; `split_guard` correctly rejected there for a
different, also-genuine reason -- that box's `official_test` partition was
unpopulated *and* its underlying dataset files were unverifiable. This re-run
happens on the box that D1 (`D1_dataset_verification_gpubox.json`) confirmed
holds the real, hash-verified dataset.)

## 1. Inspect `split_manifest.json` for `official_test`

Two files named `split_manifest.json` exist in this repo and they are **not**
the same schema:

- **Root** `split_manifest.json` (protected, unchanged by this task): schema is
  `{tune_videos, confirm_videos, ...}`, no `partitions`/`official_test` concept
  at all. This is the val_tune/val_confirm split used elsewhere.
- **`benchmark_runs/paper_setup_20260720T074844Z_1e431b7/split_manifest.json`**
  (the file `split_guard.py`'s `MANIFEST_PATH` actually resolves to, since it's
  `<script_dir>/../split_manifest.json`): schema is `{partitions: {val_tune,
  val_confirm, official_test}}`. This is the file `split_guard.py` reads.

Before this task, `official_test` in the benchmark_runs copy read:

```json
{
  "video_count": 0,
  "video_ids": [],
  "question_count": 0,
  "question_ids": [],
  "status": "UNPOPULATED -- official test annotations not available locally, see setup_failures.jsonl"
}
```

## 2. Execute `guard_official_test_command("official_test")` -- before population

```
$ cd benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts
$ python3 -c "
import split_guard
try:
    split_guard.guard_official_test_command('official_test')
    print('RESULT: PASSED (no exception), no exit called')
except SystemExit as e:
    print('RESULT: SystemExit code=', e.code)
"
```

Output:

```
REJECTED: official_test partition is unpopulated (0 videos) -- official NExT-GQA test
annotations were not verifiable locally at setup time. See setup_failures.jsonl.
The official test command cannot run until this is resolved.
RESULT: SystemExit code= 3
```

Exact match to the guard's designed rejection path (`ot["video_count"] == 0`,
`sys.exit(3)`).

## 3. Population

D1 (`D1_dataset_verification_gpubox.json`) confirmed on this box that
`eval/data/nextqa/test.csv` and `gsub_test.json` hash-match
`dataset_manifest.json`'s `files_written` entries, so populating is a mechanical
step, not a judgement call.

Populated `official_test.video_ids` with every key in `gsub_test.json` whose
`location` is non-empty (990 videos, matching D1 step 3), and
`official_test.question_ids` with `"{video_id}::{location_key}"` for every
question in those videos' `location` dicts (5553 questions, matching D1 step
3). File written in-place at
`benchmark_runs/paper_setup_20260720T074844Z_1e431b7/split_manifest.json`.

Resulting counts:

```
video_count: 990
question_count: 5553
```

Both match D1's independently-derived `n_videos_with_gold=990`,
`n_questions=5553` for the test split exactly.

## 4. Re-run `guard_official_test_command("official_test")` -- after population

```
$ python3 -c "
import split_guard
try:
    split_guard.guard_official_test_command('official_test')
    print('RESULT: PASSED (no exception), no exit called')
except SystemExit as e:
    print('RESULT: SystemExit code=', e.code)
"
RESULT: PASSED (no exception), no exit called
```

**split_guard now passes.**

## 5. Documentation updated

Appended dated correction blocks (preserving all existing text) to:

- `benchmark_runs/paper_setup_20260720T074844Z_1e431b7/commands/13_run_official_nexgqa_test.md`
- `benchmark_runs/paper_setup_20260720T074844Z_1e431b7/setup_report.md`

Both now state: split_guard passes on `worker-1` as of 2026-07-27, but **the
official test command has still not been executed** -- populating the
partition removes the guard blocker, it does not authorize or constitute a
run.

## 6. Protected-file scope note

Per task constraints, root `split_manifest.json` is the one exception among
protected files and only its `official_test` key was authorized to change.
That root file, however, has no `official_test` key at all (see section 1) --
the file `split_guard.py` actually reads and that was actually populated is
the `benchmark_runs/.../split_manifest.json` copy. Root `split_manifest.json`
was left completely untouched (hash unchanged, verified in
`final_report_gpubox.md`). Only the `official_test` key of the benchmark_runs
copy changed; diff shown in `final_report_gpubox.md`.
