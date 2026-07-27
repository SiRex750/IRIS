# Official 990-video NExT-GQA test-split pre-flight

**The official 990-video test run CANNOT proceed — there is no executable
official-test runner in this repository.**

Everything the run *consumes* is in place (media, disk, split manifest, pinned
server). What is missing is the thing that would run it: all 22 entries in
`benchmark_runs/paper_setup_20260720T074844Z_1e431b7/commands/` are `.md`
specification documents, and the `13_run_official_nexgqa_test.py` that
`13_run_official_nexgqa_test.md` advertises does not exist. Detail in §7.

Read-only audit. No eval, ingest, captioner, answerer, or LLM call was made. No
video was ingested. The test split was not run.

- `git rev-parse HEAD` at start: `505c2f70bfd310eefd662aea327526cbd3ef9428`
- branch: `feat/prerun-fixes` (unchanged)
- `git status --porcelain` at start:
  ```
   M eval/metrics.py
   M external_data/qvhighlights/manifests/external_train.jsonl
  ?? scratch/bakeoff_harness_dataset.json
  ?? scripts/answerer_bakeoff_harness.py
  ?? scripts/span_sweep.py
  ?? scripts/span_sweep_bootstrap.py
  ?? tests/test_span_methods_ef.py
  ?? tuning/span_sweep/
  ```

---

## 1. MEDIA — PASS, nothing missing

**Video root:** `/home/ccbd/IRIS-1/eval/data/nextqa/NExTVideo_flat`

That is the same root the val videos load from: `scripts/part3_tune.py:56`
```python
VIDEO_DIR = REPO / "eval" / "data" / "nextqa" / "NExTVideo_flat"
```
used by both `load_val_tune_questions()` (`part3_tune.py:209`) and
`load_val_confirm_questions()` (`scripts/val_confirm_e2e_eval.py:177`), which
test the exact same predicate (`vpath.exists()` on `{vid}.mp4`).

Parsing `eval/data/nextqa/test.csv`: 5553 rows, **990 unique video ids**.

| | count |
|---|---|
| present (`{root}/{vid}.mp4` exists) | **990** |
| missing | **0** |

First 20 missing ids: *(none — the missing list is empty)*

Cross-check against gold annotations: `eval/data/nextqa/gsub_test.json` holds 999
video entries, 990 of them gold-bearing, 5553 total `location` entries. Videos in
`test.csv` but without gold: 0. Gold-bearing videos absent from `test.csv`: 0.
The 990/5553 figures reconcile exactly across CSV and gsub.

The directory holds 1552 `.mp4` files total (val + test + others).

## 2. INDEX CACHE — zero test videos cached; all 990 need ingest

**Frozen ingest-config hash: `4edae64ed40256e3`**, computed live, not assumed:

```
python: cfg = part3_tune.make_config(json.load(open('tuning/frozen_state.json'))['frozen'])
        part3_tune.ingest_config_hash(cfg)  ->  4edae64ed40256e3
```
(`INGEST_RELEVANT_KEYS` = `retrieval_strategy, l2_retrieve_top_k, peak_distance,
peak_prominence, peak_order, packet_size_weight, motion_weight,
luma_entropy_weight, persistence_threshold, max_prominence` —
`scripts/part3_tune.py:193-195`.)

Entries matching `*__4edae64ed40256e3.npz`:

| cache dir | entries under frozen hash |
|---|---|
| `tuning/index_cache` (canonical, `part3_tune.py:57`) | 450 |
| `tuning/index_cache_val_confirm_e2e` | 112 |
| `tuning/determinism_gate/index_cache` | 112 |
| `eval/data/nextqa/index_cache` | 0 |
| **distinct videos** | **562** |

All 562 are validation videos.

| | count |
|---|---|
| test videos already cached under the frozen hash | **0** |
| test videos to ingest | **990** |

(For context, `tuning/index_cache` holds 10 800 files = 450 videos × 24 distinct
config hashes from the tuning sweeps. Only the `4edae64ed40256e3` slice is
relevant to the official run.)

## 3. DISK — sufficient headroom for the index, but the root filesystem is at 98%

`df -h` — the index cache, the video root, and `$HOME` are all on one filesystem:

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  938G  865G   26G  98% /
```

`du -sh`: video root `6.1G`, `tuning/index_cache` `9.3G` (all 24 hashes).

On-disk index size per video, measured over the 450 existing frozen-hash `.npz`
files:

| statistic | value |
|---|---|
| mean | 1.0035 MB |
| median | 0.830 MB |
| p90 | 1.928 MB |

**Projection:** 990 × 1.0035 MB = **0.993 GB**.

Against 26 GB free that is **3.8% of free space — well under the 80% flag.**
The projection is if anything conservative: test videos are slightly shorter than
the val videos the mean was measured on (mean frame_count 1137.5 vs 1200.9 for
the ingested val_tune subset, a 0.947× ratio).

**Caution, not a blocker:** the root filesystem is 98% full with 26 GB absolute
headroom, and it is the *only* filesystem — index cache, videos, logs, and
per-question CSVs all land on it. The ~1 GB index is comfortable; there is no
margin for anything unexpected on top of it.

## 4. TIME BUDGET

### Ingest

No ingest log in the tree prints a per-video wall-clock figure
(`grep -n "ingest done in\|ingest_wall" tuning/*.log tuning/logs/*.log logs/*.log`
returns nothing; the logs only print `[ingest] N/M done` progress with no
timestamps). Throughput was therefore derived from `.npz` mtimes — the file
write is the last step of each ingest, so the mtime spread over a single ingest
pass *is* the wall-clock for that pass. Three independent passes:

| pass | n videos | wall span | s/video (8 workers) | inter-completion p90 |
|---|---|---|---|---|
| `tuning/index_cache_val_confirm_e2e` | 112 | 254.9 s | 2.30 | 4.62 s |
| `tuning/determinism_gate/index_cache` | 112 | 262.8 s | 2.37 | 5.18 s |
| `tuning/index_cache` @ `4edae64ed40256e3` | 450 | 980.3 s | 2.18 | 4.50 s |

All three passes used `ThreadPoolExecutor(max_workers=8)`
(`part3_tune.py:223,251`; `val_confirm_e2e_eval.py:190,216`).

**Ingest projection for 990 videos: 2158–2346 s = 0.60–0.65 h.**

Note on the requested "mean and p90 wall-clock ingest seconds per video": the
logs do not record per-video latency, only completion order under 8-way
parallelism, so the p90 column above is the p90 *inter-completion gap*, not a
per-video duration. Per-video latency is roughly 8× that (~17–19 s) but that
number is **UNDETERMINED** from the available logs — the aggregate wall-clock
figure is the one that is actually measured, and it is what the projection uses.

### Answer stage

The relevant precedent is `tuning/val_confirm_e2e_per_question.csv` /
`tuning/logs/val_confirm_e2e_run.log` — 639 questions, **live captioning**, same
frozen config, same box:

```
VAL_CONFIRM_E2E_METRICS_JSON={"n_scored": 639, ...,
  "median_retrieval_span_ms": 5.017663002945483,
  "median_caption_answer_ms": 2217.5388670002576,
  "p95_caption_answer_ms": 3728.7926179997157,
  "total_wall_s": 1370.9179673500184}
```

Recomputed from the per-question CSV: mean 2097.2 ms, median 2217.5 ms,
p90 3441.7 ms. Retrieval+span is 5 ms median — negligible.

**Answer-stage projection for 5553 questions:**

| basis | per-question | 5553 questions |
|---|---|---|
| observed end-to-end wall (1370.9 s / 639) | 2.145 s | 3.31 h |
| mean caption+answer | 2.097 s | 3.24 h |
| median caption+answer | 2.218 s | 3.42 h |
| p90 caption+answer (pessimal) | 3.442 s | 5.31 h |

**Do not** use the `tuning/determinism_gate/arm_pass*_per_question.csv` figure
(median 174 ms) for this projection — those passes ran with `--caption-load`
against a frozen `captions_dump.json`, so 174 ms is answerer-only latency with
the captioner removed. It applies only if the official run also replays frozen
captions.

### Total

| stage | range |
|---|---|
| ingest (990 videos) | 0.60 – 0.65 h |
| inference (5553 questions) | 3.2 – 5.3 h |
| **total** | **3.8 – 6.0 h** |

Central estimate ≈ **3.9–4.1 h**; the 6.0 h upper bound assumes the test split
sits at the val split's p90 caption+answer latency throughout, which is
deliberately pessimistic.

## 5. MANIFEST CONFLICT — there is no conflict; the two files have different schemas

Both paths, both `official_test` contents:

**A. `/home/ccbd/IRIS-1/split_manifest.json`**
sha256 `7864ba3692234320fc453573d6fa8be5e12779f222fd99a841c63a745e98f12a`

Top-level keys: `seed, n_videos_total, n_videos_tune, n_videos_confirm,
tune_videos, confirm_videos, sha256_of_split_arrays`.

**`official_test` key: ABSENT.** Not "unpopulated" — the key does not exist, and
neither does a `partitions` object. Values: `seed=20260721`,
`n_videos_total=567`, `n_videos_tune=454`, `n_videos_confirm=113`.

**B. `/home/ccbd/IRIS-1/benchmark_runs/paper_setup_20260720T074844Z_1e431b7/split_manifest.json`**
sha256 `1cc8c5f03b5827d3f7152bb049a16663ee37898616402ebb7f87bbba6455f19a`

Top-level keys: `generated_by, seed, algorithm, source_note, val_tune_fraction,
partitions`. `partitions.official_test`:

```
video_count    : 990
video_ids      : list len 990  ['10109006686', '10128261054', '10149430394', ...]
question_count : 5553
question_ids   : list len 5553 ['10109006686::0', '10109006686::2', '10109006686::3', ...]
status         : "POPULATED 2026-07-27 on worker-1 from
                  eval/data/nextqa/{test.csv,gsub_test.json}, verified against
                  dataset_manifest.json by D1_dataset_verification_gpubox.json
                  (files_written hash match). Mechanical population, not a
                  re-derivation of the split."
```

**Which one the test-split path reads — from the loader code:**

`benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts/split_guard.py:13`
```python
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "..", "split_manifest.json")
```

The path is resolved relative to `split_guard.py`'s own directory, so it resolves
to **B**, the `benchmark_runs` copy — never the repo root. `guard_official_test_command`
then does `manifest["partitions"]["official_test"]["video_count"]`
(`split_guard.py:64-65`), which on **A** would raise `KeyError: 'partitions'`
before reaching any `official_test` lookup.

Conversely, every in-repo consumer of **A** reads `tune_videos`/`confirm_videos`
and nothing else: `part3_tune.py:199`, `val_confirm_e2e_eval.py:165`,
`span_sweep.py:327`, `span_sweep_bootstrap.py:89`, `parta_survivor_ceiling.py:31`.
**A** is written by `scripts/part2_dataset_setup.py:336`; **B** by
`benchmark_runs/.../scripts/make_split_manifest.py`.

**Canonical: both, for disjoint purposes.** `benchmark_runs/.../split_manifest.json`
is canonical for the `official_test` partition, because it is the only file that
has that key and the only one `split_guard.py` can read. The repo-root
`split_manifest.json` is canonical for the val_tune/val_confirm partition, because
it is the only file the tuning harnesses read and the only one with
`tune_videos`/`confirm_videos`. They are not two copies of one artifact and one
does not shadow the other. Neither file was modified.

## 6. SPLIT ARITHMETIC — the 5 videos and 34 questions are fully accounted for

The premise needs one correction: the gap is not between the manifest and the
partitions, it is between the **manifest** and what the **loaders** keep.
`split_manifest.json` records `n_videos_tune=454` and `n_videos_confirm=113`
(= 567), not 450 and 112. The 450/112 figures are post-filter usable counts.

Reproducing both loaders' filters exactly — `vpath.exists()` then
`gsub[vid]["location"][qid]` truthiness (`part3_tune.py:207-214`,
`val_confirm_e2e_eval.py:175-182`):

| partition | manifest videos | usable videos | usable questions |
|---|---|---|---|
| val_tune | 454 | 450 | 2685 |
| val_confirm | 113 | 112 | 639 |
| **total** | **567** | **562** | **3324** |

`val.csv` rows over the 567 manifest videos: **3358**. `gsub_val.json` `location`
entries over the same 567: **3358**. So D1's 567/3358 is the manifest-level count
and 562/3324 is the loader-level count; the gap is **5 videos / 34 questions**.

**The 5 videos, with the reason each was dropped:**

| video id | partition | val.csv rows | mp4 present | in gsub_val | gsub location entries | dropped by |
|---|---|---|---|---|---|---|
| `4367056464` | val_tune | 9 | **No** | Yes | 9 | `if not vpath.exists(): continue` |
| `4372494989` | val_tune | 8 | **No** | Yes | 8 | `if not vpath.exists(): continue` |
| `6011836129` | val_tune | 3 | **No** | Yes | 3 | `if not vpath.exists(): continue` |
| `6018490041` | val_tune | 7 | **No** | Yes | 7 | `if not vpath.exists(): continue` |
| `5996148663` | val_confirm | 7 | **No** | Yes | 7 | `if not vpath.exists(): continue` |

9 + 8 + 3 + 7 + 7 = **34 questions**. Exact match, no residual.

**Reason: missing media, and only missing media.** All five have full gold
annotations in `gsub_val.json` and full rows in `val.csv`; none is filtered on
question type, gold absence, or anything else. Zero questions were dropped on
videos that otherwise survived — the loss is entirely at video granularity.
`ls /home/ccbd/IRIS-1/eval/data/nextqa/NExTVideo_flat/{id}*` returns nothing for
all five, so it is not an extension or naming variant.

This is a val-split-only issue and has **no bearing on the test run** — §1
confirms all 990 test videos are present.

## 7. LAUNCHER — written, verified, no server left running

**`/home/ccbd/.local/iris/bin/serve-granite-pinned.sh`** (mode `0755`).

It sets `LD_LIBRARY_PATH=/home/ccbd/.local/iris/bin` (required — the pinned
binary has no rpath back to that directory; see
`tuning/prerun_fixes/pinned_binary.json`, `ldd_dependencies.runtime_requirement`)
and `exec`s `/home/ccbd/.local/iris/bin/llama-server-b10099` with
`-ngl 999 -c 8192 --parallel 1 --host 127.0.0.1 --port 8091 --alias granite4:micro`
and `-m /home/ccbd/.ollama/models/blobs/sha256-97c417dcc0534b0737c74016fb2af083cb17c3b51eaac621192d23961b7024eb`.

Guards, in order:
1. binary exists and is executable, else exit 1;
2. **`sha256sum` of the binary must equal
   `250d2eb35a6ddfa1df060599f8bc3169944ef60a77429b2cc395e062859e0a06`, else exit 2**
   — this runs before *any* mode, including `--version`;
3. GGUF file exists, else exit 3;
4. GGUF pin check, else exit 4. The GGUF lives in Ollama's content-addressed blob
   store so its filename *is* its sha256; the script verifies the basename equals
   `sha256-97c417dc...` rather than re-hashing 2.1 GB on every launch. Set
   `VERIFY_GGUF=1` to force a full re-hash (exit 5 on mismatch).

Any argument is forwarded verbatim to the binary and the serving flags are
skipped, so `--version` cannot accidentally start a server.

**Verification run (the only execution performed):**
```
$ ss -ltnp | grep 8091            -> port 8091 free
$ /home/ccbd/.local/iris/bin/serve-granite-pinned.sh --version
version: 1 (1a064ab)
built with GNU 12.3.0 for Linux x86_64
exit=0
$ ss -ltnp | grep 8091            -> port 8091 free
$ pgrep -af llama-server-b10099   -> no llama-server-b10099 running
```
Exit 0 means the SHA-256 gate executed and passed. Confirmed independently:
`sha256sum /home/ccbd/.local/iris/bin/llama-server-b10099` →
`250d2eb35a6ddfa1df060599f8bc3169944ef60a77429b2cc395e062859e0a06`. The version
string matches `pinned_binary.json`'s recorded `version_output_from_destination`
(tag `b10099`, commit `1a064ab`). **No server is running.**

### The blocker found here

`benchmark_runs/paper_setup_20260720T074844Z_1e431b7/commands/13_run_official_nexgqa_test.md`
advertises:

> **CLI signature:** `python commands/13_run_official_nexgqa_test.py --config --split=official_test --out-dir [...]`

`find benchmark_runs/.../commands -type f | sed 's/.*\.//' | sort | uniq -c` →
**`22 md`**. There are no `.py` files in `commands/` at all. The advertised
runner does not exist.

Nor is there a substitute: `grep -rn "official_test" --include=*.py .` (excluding
`.git` and worktrees) matches only `make_split_manifest.py` and `split_guard.py`
— manifest construction and the leakage guard. No evaluation harness references
the split. `grep -rn '"--split"' --include=*.py scripts/ eval/` matches only
`scripts/nextqa_single_video_eval.py:297`, a single-video tool.

The nearest working harness is `scripts/val_confirm_e2e_eval.py`, and it is
hard-wired to the validation split — `split["confirm_videos"]`, `val.csv`,
`gsub_val.json` (`val_confirm_e2e_eval.py:165-172`). It has no test-split code
path.

`split_guard.py`'s own docstring confirms the wiring was never done:
> "Not wired into iris/ (setup-only artifact). Intended usage from future command scripts"

---

## GO / NO-GO

### **NO-GO.**

**Blocker: no executable official-test runner exists.** All 22 files in
`benchmark_runs/paper_setup_20260720T074844Z_1e431b7/commands/` are `.md`;
`13_run_official_nexgqa_test.py` — the entry point its own `.md` documents — is
absent, no other script accepts `--split=official_test`, and `split_guard.py` is
imported by nothing. There is no code path from "run the test split" to an
actual evaluation. This is a build blocker, not a data blocker.

Everything else is clear:

| check | status |
|---|---|
| 1. Media — 990/990 test mp4s present | **PASS** |
| 2. Index cache — 0 cached, 990 to ingest | PASS (expected; no stale-cache hazard) |
| 3. Disk — 0.99 GB projected vs 26 GB free (3.8%) | **PASS** (caution: root fs at 98%) |
| 4. Time — 3.8–6.0 h total | PASS |
| 5. Manifest — `official_test` populated 990/5553, correctly located for `split_guard.py` | **PASS** |
| 6. Split arithmetic — 5 videos / 34 questions, all missing val media, val-only | **PASS**, fully explained |
| 7. Launcher — written, SHA-gated, `--version` verified, nothing left running | **PASS** |

**To convert this to GO,** one thing must be built: an evaluation entry point
that loads the test split (`test.csv` + `gsub_test.json`), calls
`guard_official_test_command("official_test")` against
`benchmark_runs/.../split_manifest.json`, and runs the frozen config end to end —
i.e. `val_confirm_e2e_eval.py`'s loop against the test-split loader. That is a
code change and is out of scope for this read-only pre-flight; it was not made.

### Total projected wall-clock

**3.8 – 6.0 hours** (ingest 0.60–0.65 h + inference 3.2–5.3 h), central estimate
**≈ 4 hours**, once a runner exists. Excludes model load, report generation, and
any bootstrap-CI pass.

---

## Protected-file integrity

Hashed before the audit and again after. **All byte-identical.**

| file | sha256 (before == after) |
|---|---|
| `split_manifest.json` | `7864ba3692234320fc453573d6fa8be5e12779f222fd99a841c63a745e98f12a` |
| `dataset_manifest.json` | `ae90d6490c89498455dcb09beb5724bb43611fed9d0365cc8af507014a48f853` |
| `benchmark_runs/paper_setup_20260720T074844Z_1e431b7/split_manifest.json` | `1cc8c5f03b5827d3f7152bb049a16663ee37898616402ebb7f87bbba6455f19a` |
| `tuning/frozen_state.json` | `612a93ba88af53eff17d766885dd3bcf8f65e502e4822b60c1a5c04db4c0b1b7` |
| `eval/data/nextqa/test.csv` | `a8d67c5a648c9da8fb4f6fa9939ad1e31fb43f9a92fdab48ec38a1cd622e23db` |
| `eval/data/nextqa/val.csv` | `ee0ef210ae341bec53fdeb723ae54958307ed2e49988d2d3763d3c22abf4201a` |
| `eval/data/nextqa/gsub_test.json` | `0c16c31699238a0f2b8ace45aeed7f193149ce2f011d6ed4f1ef7170ef5d41e9` |
| `eval/data/nextqa/gsub_val.json` | `527f402851836224aa03c9d4b0016b858a678d21596a2348eb2bdb774c74970a` |
| `tuning/blind_ablation/**` (tree digest) | `2d8fcf20278e93a17e1f86c775bc203442f822f4cbfc107465efa6555094d83f` |

There is no `frozen_state.json` at the repo root; the file is at
`tuning/frozen_state.json` and is hashed above.

Files written by this task: `tuning/prerun_fixes/item_D_audit.md`,
`tuning/prerun_fixes/test_split_preflight.md`,
`/home/ccbd/.local/iris/bin/serve-granite-pinned.sh` (outside the repo). Nothing
else.
