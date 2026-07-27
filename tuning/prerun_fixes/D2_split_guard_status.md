# D2 — split_guard status (2026-07-27)

## 1. What `split_guard.py` checks

`benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts/split_guard.py` is a
standalone leakage guard (not wired into `iris/`) that reads
`benchmark_runs/paper_setup_20260720T074844Z_1e431b7/split_manifest.json` (its own
`partitions` dict with `val_tune` / `val_confirm` / `official_test` entries) and
exposes three checks: `guard_tuning_command` rejects any tuning invocation whose
split name or config path looks like it targets the test set, `guard_question_id`
rejects a question id that belongs to `official_test` being scored under a tuning
split, and `guard_official_test_command` rejects the official-test command unless
the split name is exactly `"official_test"` **and** that partition's
`video_count > 0`. Concretely, "official_test partition unpopulated" means the
`partitions.official_test` entry in this directory's own `split_manifest.json` has
`video_count: 0, video_ids: [], question_count: 0, question_ids: []` — an explicit
placeholder, not a missing key.

## 2. Does `official_test` exist and is it populated?

Checked `benchmark_runs/paper_setup_20260720T074844Z_1e431b7/split_manifest.json`
directly:

```
val_tune:      video_count=71, question_count=204
val_confirm:   video_count=18, question_count=51
official_test: video_count=0,  question_count=0
               status: "UNPOPULATED -- official test annotations not available
                        locally, see setup_failures.jsonl"
```

The key exists. It is unpopulated, exactly as the stale docs claim.

(Note: this is a **different, smaller, earlier** manifest than the root
`split_manifest.json` used by the rest of the repo — root has 567 val videos /
454 tune + 113 confirm and no `official_test` key at all. `split_guard.py` reads
only the `benchmark_runs/.../split_manifest.json` copy, never the protected root
one, so nothing protected was touched by this check.)

## 3. Executed split_guard against official_test — exact output

Command run:
```
cd benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts
python3 -c "
import split_guard
try:
    split_guard.guard_official_test_command('official_test')
    print('PASSED - no exception')
except SystemExit as e:
    print('SystemExit code:', e.code)
"
```
Output:
```
REJECTED: official_test partition is unpopulated (0 videos) -- official NExT-GQA test
annotations were not verifiable locally at setup time. See setup_failures.jsonl.
The official test command cannot run until this is resolved.
SystemExit code: 3
```

**split_guard does NOT pass.** It refuses with exit code 3, unchanged from the
document's original claim.

## 4. Why it refuses, minimal fix, and why the fix was NOT applied

The refusal is purely because `partitions.official_test.video_count == 0` in
`benchmark_runs/.../split_manifest.json`. The minimal fix would be a pure data
population step: fill `video_ids`/`question_ids`/`video_count`/`question_count`
in that partition from the real `test.csv`/`gsub_test.json`, matching the same
schema as the populated `val_tune`/`val_confirm` entries. No hyperparameter or
split-boundary code changes are implied by that step in principle.

**Not implemented**, because on this box the source data needed to populate it
correctly is not itself verifiable (per D1,
`tuning/prerun_fixes/D1_dataset_verification.json`):

- `eval/data/nextqa/test.csv` does not exist on this checkout at all.
- `eval/data/nextqa/gsub_test.json` present here is byte-identical to
  `gsub_val.json` and does not match `dataset_manifest.json`'s recorded
  `files_written` hash for `gsub_test.json` in content-derived terms (same bytes,
  but the video/question counts recomputed from it — 999 videos / 2997 questions —
  don't match the manifest's own recorded test counts of 990/5553either, meaning
  this local file is some other/stale snapshot, not proven to be the exact
  990-video/5553-question set the manifest describes).

Populating `official_test` from these files would require *judgment calls* about
which of the 999 present-but-uncertain video ids actually belong in the official
990, which is exactly the kind of split-boundary decision the task instructions
say to stop on rather than make unilaterally. So per instructions: **stopped and
reporting**, not implemented.

## 5. Stale documents corrected

Both documents were updated with an appended, dated correction block (original
text preserved, nothing deleted):

- `benchmark_runs/paper_setup_20260720T074844Z_1e431b7/commands/13_run_official_nexgqa_test.md`
- `benchmark_runs/paper_setup_20260720T074844Z_1e431b7/setup_report.md`

Each correction states: the blanket "test split unavailable" premise is stale
relative to the root `dataset_manifest.json` (2026-07-21, 990/990 videos,
5553/5553 rows verified), but this specific checkout's local files still do not
reflect that acquisition, and split_guard re-run today still rejects for the same
structural reason. See each file's "Correction (2026-07-27...)" section.

## Bottom line for D2

- `official_test` partition: exists, confirmed unpopulated (0 videos).
- split_guard: executed, still REJECTS (exit 3) — **does not pass**.
- Fix: identified (pure manifest population from test.csv/gsub_test.json) but
  **not applied**, because this box lacks a verifiable source for that
  population step. This is a blocker, reported as such, not silently resolved.
