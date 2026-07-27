# D1 / D2 / A2 re-run on the GPU box -- final report

**Generated:** 2026-07-27

## 1. GO / NO-GO

**Conditional GO for the official 990-video test run from this box**: the
dataset (D1) is verified and correct, and the split_guard blocker (D2) is
cleared -- but the run cannot start yet because the `llama-server` binary
matching the recorded val_confirm_e2e spec (A2) exists only in an ephemeral
scratch location and must be copied to a stable path and re-verified first.

## 2. Box

- **Hostname:** `worker-1`
- **`uname -a`:** `Linux worker-1 6.8.0-134-generic #134~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Jun 30 14:05:04 UTC x86_64 x86_64 x86_64 GNU/Linux`
- **Repo path:** `/home/ccbd/IRIS-1`
- **Branch:** `feat/prerun-fixes`

Precondition gate (checked before any other work): all 4 passed --
`eval/data/nextqa/test.csv` exists; `gsub_val.json`/`gsub_test.json` are not
byte-identical; `tuning/index_cache_val_confirm_e2e/` holds 112 `.npz` files
under config-hash `4edae64ed40256e3`; at least one `llama-server` binary
exists on this box.

## 3. D1 -- dataset verification

Full detail: `tuning/prerun_fixes/D1_dataset_verification_gpubox.json`.
Supersedes `tuning/prerun_fixes/D1_dataset_verification.json` (ran on a
Windows checkout without the real data; also contained a counting bug --
it counted `len(gsub[video])`, i.e. the 3 dict keys `duration`/`location`/`fps`
per video, not `len(gsub[video]['location'])`, the actual question count).

**Hashes** (all four files match `dataset_manifest.json`'s `files_written`
entries; none match `official_source_file_sha256`, which is expected --
`files_written` is post schema-fix, `official_source` is pre-fix):

| file | local sha256 | matches files_written |
|---|---|---|
| val.csv | `ee0ef210ae341bec53fdeb723ae54958307ed2e49988d2d3763d3c22abf4201a` | yes |
| test.csv | `a8d67c5a648c9da8fb4f6fa9939ad1e31fb43f9a92fdab48ec38a1cd622e23db` | yes |
| gsub_val.json | `527f402851836224aa03c9d4b0016b858a678d21596a2348eb2bdb774c74970a` | yes |
| gsub_test.json | `0c16c31699238a0f2b8ace45aeed7f193149ce2f011d6ed4f1ef7170ef5d41e9` | yes (also matches official_source) |

`gsub_val.json != gsub_test.json` by hash: **confirmed** (distinct hashes above).

**Corrected entry / with-gold / question counts** (method: `n_video_entries =
len(gsub)`; `n_videos_w_gold = sum(1 for v in gsub.values() if
v.get("location"))`; `n_questions = sum(len(v["location"]) for v in
gsub.values() if v.get("location"))`):

| split | video entries | videos with gold | questions | expected (dataset_manifest.json) | match |
|---|---|---|---|---|---|
| val | 570 | 567 | 3358 | 567 videos / 3358 questions | yes |
| test | 999 | 990 | 5553 | 990 videos / 5553 questions | yes |

(test's 999 entries / 990 with non-empty `location` is the expected shape,
not a defect, per task instructions.)

CSV cross-check: `val.csv` = 3358 rows / 567 unique videos; `test.csv` = 5553
rows / 990 unique videos -- exact match to the gold counts above.

`split_manifest.json` (root) `tune_videos` (454) + `confirm_videos` (113) =
567, overlapping **567/567** (full overlap) with val gsub's 570 video
entries -- confirms these are genuinely the val files.

Direct `set(gsub_val.keys()) & set(gsub_test.keys())` intersection: **0**,
matching the manifest's recorded `val_test_video_overlap: 0`.

`.mp4` inventory (`eval/data/nextqa/NExTVideo_flat/`, 1552 files total):
test 990/990 present; val 562/567 present (matches manifest's
`videos_successfully_downloaded: 562`).

## 4. D2 -- split_guard

Full detail: `tuning/prerun_fixes/D2_split_guard_status_gpubox.md`.

`split_guard.py`'s `MANIFEST_PATH` resolves to
`benchmark_runs/paper_setup_20260720T074844Z_1e431b7/split_manifest.json`
(distinct from, and a different schema than, root `split_manifest.json`).
Before this task, `guard_official_test_command("official_test")` returned
`SystemExit(3)` ("official_test partition is unpopulated").

**Change made**: populated `official_test.video_ids` (990 videos, every key
in `gsub_test.json` with non-empty `location`) and `official_test.
question_ids` (5553 entries, format `"{video_id}::{location_key}"`) in
`benchmark_runs/paper_setup_20260720T074844Z_1e431b7/split_manifest.json`
only. This was a mechanical population, not a judgement call, since D1
independently confirmed the source files are hash-verified.

**split_guard now passes**: `guard_official_test_command("official_test")`
re-executed after population returns with no exception (previously
`SystemExit(3)`).

Dated correction blocks were appended (all prior text preserved) to:
- `benchmark_runs/paper_setup_20260720T074844Z_1e431b7/commands/13_run_official_nexgqa_test.md`
- `benchmark_runs/paper_setup_20260720T074844Z_1e431b7/setup_report.md`

Both state clearly: split_guard passes on `worker-1` now, but the official
test command has **not** been executed -- this authorises the run to happen,
it does not authorise or constitute running it.

## 5. A2 -- binary inventory

Full detail: `tuning/prerun_fixes/A2_binary_inventory_gpubox.json`.
Supersedes `tuning/prerun_fixes/A2_binary_inventory.json` (Windows box, zero
hits, uninformative).

`find / -iname 'llama-server*' -type f` found:

| path | sha256 | version | notes |
|---|---|---|---|
| `/home/ccbd/.claude/jobs/de60ff71/tmp/build/llama.cpp/build/bin/llama-server` | `250d2eb35a6ddfa1df060599f8bc3169944ef60a77429b2cc395e062859e0a06` | `1a064ab` | matches recorded spec exactly; **ephemeral location** |
| `/usr/local/lib/ollama/llama-server` | `234b05b2138264f8fb263c3205e85f4c290e8afe5067e280a4f6f90cdac5696b` | `b4d6c7d8f` | Ollama's bundled runner, stable path, used by the blind-ablation run, NOT this build |

(`llama-server-simulator.py` also matched the glob but is a Python script,
not a binary -- excluded.)

The recorded val_confirm_e2e spec is llama.cpp tag **b10099**, commit
**1a064ab**, `GGML_CUDA=ON`, `sm_89`, CUDA 12.6. The source checkout at
`/home/ccbd/.claude/jobs/de60ff71/tmp/build/llama.cpp` has `git rev-parse
HEAD` = `1a064ab0921238c1daa397d6f4a900ef33884de2` (tag `b10099`,
confirmed via `git describe --tags` and `git merge-base --is-ancestor`), and
its `CMakeCache.txt` confirms `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=89`,
CUDA 12.6.20 toolkit -- an exact match to the recorded spec.

`/usr/local/lib/ollama/llama-server`'s hash (`234b05b2...`) differs from the
from-source build's hash (`250d2eb3...`) -- confirmed different binaries,
different llama.cpp versions, different compilers (GCC 13.3.1 vs 12.3.0).

**Binary that should be pinned**: the from-source `b10099`/`1a064ab` build,
**not** Ollama's bundled runner (the val_confirm_e2e report explicitly chose
from-source over Ollama because Ollama's backend silently drops
`cache_prompt=false`, needed for the seed-based determinism fix).

**Reference SHA-256 to pin**: `250d2eb35a6ddfa1df060599f8bc3169944ef60a77429b2cc395e062859e0a06`

**Not yet pinned**: this exact binary currently lives only at
`/home/ccbd/.claude/jobs/de60ff71/tmp/build/llama.cpp/build/bin/llama-server`,
inside another background job's scratch directory, which is deleted when
that job is cleaned up. It must be copied to a stable path (e.g.
`/usr/local/bin/llama-server-b10099`) and its SHA-256 re-verified against
`250d2eb3...06` at that stable path before the official run starts. Copying
a binary and re-verifying it is a filesystem operation beyond the scope of
this verification-only task, so it was not done here.

## 6. What did not run, and why

- **The official test split itself was not run.** Not authorised by this
  task (explicitly out of scope) and not possible yet regardless -- D2 only
  clears the split_guard gate, it does not launch anything.
- **No model was loaded, no captioner/answerer invoked, no ingestion, no
  evaluation, no tuning.** Per task constraints.
- **The seven code fixes on this branch were not touched.** Per task
  constraints -- they are reviewed and correct.
- **The `b10099`/`1a064ab` llama-server binary was not copied to a stable
  path or pinned in any config file.** It was located and hash-verified
  (A2), but moving/installing it is outside "verification only" and its
  current location will not survive job cleanup -- flagged as the remaining
  blocker before the official run can execute (see section 5).
- **Protected files**: all confirmed byte-identical before/after (see
  `$CLAUDE_JOB_DIR/tmp/protected_before.txt` vs `protected_after.txt`,
  diff was empty). Root `split_manifest.json` is untouched; only
  `benchmark_runs/paper_setup_20260720T074844Z_1e431b7/split_manifest.json`'s
  `partitions.official_test` key changed (diff shown in
  `tuning/prerun_fixes/D2_split_guard_status_gpubox.md` section 6;
  `val_tune`/`val_confirm` keys in that same file are unchanged).
