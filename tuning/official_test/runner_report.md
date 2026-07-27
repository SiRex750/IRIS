# Official test-split runner — build and validation report

## 1. Is the 990-video run ready to launch?

**Yes — the runner is built, wired, and validated end to end. One caveat the
human must accept before launching: there is currently no valid val_confirm
baseline to compare the test numbers against, for two independently-verified
reasons that both predate this work (see §3).** The 990-video run was NOT
launched, per the task spec.

Run state at start: `git rev-parse HEAD` = `a92945332743d2454753723496ada781bab40b70`,
branch `feat/prerun-fixes`. The tree was **not** clean, contrary to the task's
precondition — `git status --porcelain` showed:

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

These are somebody's in-progress span-sweep work (`scripts/span_sweep.py`
imports Methods E/F out of the modified `eval/metrics.py`). Stashing or
reverting them to satisfy the precondition would have destroyed that work, so
the state was recorded and built around instead. Verified non-interference:
`git diff --stat eval/metrics.py` is **317 insertions, 0 deletions** — pure
additions of new functions; `predicted_span_from_frames_peak`, the only
`eval/metrics.py` symbol this runner uses, is untouched.

## 2. The exact command to launch the 990-video run

Two commands. The server must be up first.

```bash
# 1. Serve the pinned answerer (SHA-gated; refuses to start if the binary drifted)
nohup setsid /home/ccbd/.local/iris/bin/serve-granite-pinned.sh \
  > ~/llama_server_pinned.log 2>&1 < /dev/null & disown

# 2. Launch the official test split (inside tmux — this runs for hours)
cd /home/ccbd/IRIS-1
python3 scripts/val_confirm_e2e_eval.py --split=official_test --resume \
  2>&1 | tee tuning/official_test/official_run.log
```

Notes on that command:

- **`--resume` is required, not optional, on this box.** 25 test videos are
  already ingested into `tuning/index_cache_official_test/` from the validation
  runs below. Without `--resume` the non-empty-cache guard aborts the run.
  The live output CSV has been cleared, so `--resume` will report
  `0 question(s) already present` and score all 5553 from scratch — it only
  reuses the 25 cached indexes.
- If it dies for any reason, **re-run the identical command**. It picks up from
  the last flushed question.
- Serving is via `serve-granite-pinned.sh` only. The runner aborts if
  `answerer_endpoint` points at Ollama's 11434; there is no fallback path.

Outputs land in `tuning/official_test/`:
`official_test_e2e_per_question.csv`, `official_test_e2e_metrics.json`,
`official_test_e2e_environment.json`.

## 3. val_confirm reproduction check

**Result: the `--split` parameterisation is a proven no-op. 0 differences
attributable to this change. 46 of 639 rows do differ from the committed CSV,
and every one is attributable to a commit that landed after that CSV was
recorded.**

### Why not a literal byte-for-byte diff of the whole CSV

The task asked for byte-for-byte. That bar is unsatisfiable by construction,
independently of this change, for two reasons — both verified, not assumed:

1. Two columns, `retrieval_span_ms` and `caption_answer_ms`, are wall-clock
   measurements. No two runs of any pipeline produce identical values.
2. The captioner changed after the baseline was recorded (see below), so every
   caption-derived column would differ regardless.

So the check was aimed at the columns the captioner and answerer **cannot**
influence: retrieval output, span construction, and gold. If `--split` is a true
no-op, these must match exactly. Script and full output:
`tuning/official_test/smoke_10/verify_val_confirm_reproduction.py`,
`.../val_confirm_reproduction.json`.

### What it returned

| check | result |
|---|---|
| questions loaded by the generalised loader | 639 (committed CSV: 639) |
| question identity **and order** match | **True** |
| rows compared | 639 |
| `pred_span_start` differences | **0** |
| `gold_spans` / `gold_answer_idx` / `question` / `type` differences | **0** |
| `pred_span_end` differences | 46 |
| `iop` / `iou` differences | 21 (all downstream of the 46) |

### Attribution of the 46

Every one of the 46 is `pred_span_end` **reduced** relative to the committed
value (46 of 46 narrowed, 0 widened), with `pred_span_start` never moving. That
is the exact signature of a duration clamp. Commit `457355d` (2026-07-27) states
it in its own message:

> **S1 fixes missing `duration_s` at every `predicted_span_from_frames_peak`
> call site so spans clamp to end-of-video.**

The committed baseline CSV was produced by commit `6f36f08` on **2026-07-24
12:20** — three days before S1 landed. So the baseline's spans ran past
end-of-video and current code correctly clamps them.

Independently confirmed this is not mine:
`git diff HEAD -- scripts/val_confirm_e2e_eval.py` filtered to the
`retrieve_for_question` / `predicted_span_from_frames_peak` /
`_call_embed_query` / `half_width_s=` / `duration_s=` lines returns **empty** —
the retrieval and span call site is byte-identical to what it was before this
change.

### Second, separate staleness: the captioner silently changed

Not asked for, found while validating, and it matters more than the S1 clamp.

`iris/aria.py:130-134` picks the captioner by probing which ollama tag happens
to exist, preferring `minicpm-v4.6` over `minicpm-v`. It is a probe, not a pin:

```python
if "minicpm-v4.6:latest" in models or "minicpm-v4.6" in models:
    model_name = "minicpm-v4.6"
elif "minicpm-v:latest" in models or "minicpm-v" in models:
    model_name = "minicpm-v"
```

Timeline, from ollama manifest mtimes and the run logs:

| when | event |
|---|---|
| 2026-07-14 21:59 | `minicpm-v` pulled |
| 2026-07-24 12:00 | val_confirm baseline runs — log records `"captioner_model": "minicpm-v"` |
| 2026-07-24 **17:27** | `minicpm-v4.6` pulled — 5.5 h *after* the baseline |
| 2026-07-27 (today) | every run resolves to `minicpm-v4.6` |

`iris/iris_config.py:110` and `iris/aria.py:142-145` both state that the
**seated production captioner is `minicpm-v4.6`**. So the direction of the
problem is the opposite of what it first looks like: today's runs are correct,
and the recorded val_confirm baseline accidentally ran on the older checkpoint
because v4.6 had not been pulled yet.

**Consequence for the launch decision:** the recorded val_confirm numbers
(`Acc@GQA_unverified = 0.1894`, `mIoP = 0.3014`) are not a valid comparator for
whatever the test split returns — they differ in the captioner *and* in the S1
span clamp. Publishing a test number against that baseline would be comparing
across two uncontrolled changes.

**Recommendation, not done here:** re-run val_confirm once under current HEAD
before or alongside the official run (~20-25 min; all 112 indexes are already
cached, so it is answer-stage only). That produces a like-for-like baseline.
It was not done here because it would overwrite a recorded held-out artifact,
which is a call for a human, not a validation step.

## 4. Smoke results and refreshed 990 projection

`--split=official_test --limit-videos 10`. Full log, CSV, metrics, and
environment in `tuning/official_test/smoke_10/`.

**These accuracy numbers are a plumbing check, not a result.** 67 questions off
10 videos is far too small to mean anything, and they are quoted only to show
the scorer produced sane values rather than zeros or nulls.

| | |
|---|---|
| videos ingested | 10 / 10 |
| questions scored | 67 |
| parse failures (`pred_answer_idx` empty/-1/None) | **0** |
| empty raw answers | **0** |
| retrieval failures / ingest failures / tracebacks | **0** |
| answer label spread | A 7, B 18, C 18, D 11, E 13 (not collapsed to one letter) |
| minicpm truncation rate | 0.0 (134 calls, 0 truncated) |
| wall clock | 95.2 s |
| mean ingest s/video (8 workers, wall) | 1.90 |
| median caption+answer | 1383.6 ms |
| p95 caption+answer | 2893.8 ms |
| Acc@QA / Acc@GQA_unverified | 0.5224 / 0.2537 |
| mIoP / mIoU | 0.4612 / 0.2282 |
| IoP@0.5 / IoU@0.5 | 0.5075 / 0.1493 |

**Bonus determinism result.** The smoke was run twice — once before and once
after the internal-seam fixes in §5 — as two fully independent invocations
(fresh process, fresh captioning, fresh answering). Diffing the two 67-row
per-question CSVs on every column except the two wall-clock timing columns:
**0 differences.** Identical captions, identical answers, identical spans. That
is a real, if small, end-to-end determinism check on the pinned serving path,
obtained for free.

Preflight assertions all passed, recorded verbatim in the log:

```
[serving] /v1/models OK -- alias 'granite4:micro' present (advertised: ['granite4:micro'])
[serving] outgoing sampler payload verified on the wire: {'temperature': 0.0, 'top_k': 1, 'top_p': 1.0, 'seed': 42, 'cache_prompt': False}
[guard] split_guard manifest: /home/ccbd/IRIS-1/benchmark_runs/paper_setup_20260720T074844Z_1e431b7/split_manifest.json
[guard] guard_official_test_command('official_test') returned cleanly
[guard] leakage check OK -- 0 of 990 test videos appear in val_tune or val_confirm (567 tuning ids checked)
```

The sampler check reads the serialised HTTP body off `httpx.Client.send`, not
the config or the source — `captured_body_keys` in `environment.json` confirms
`cache_prompt` really is on the wire.

### Refreshed 990-video projection

| stage | basis | range |
|---|---|---|
| ingest, 990 videos | 1.90 s/video observed here; 2.18-2.37 s/video from three prior val passes | **0.52 - 0.66 h** |
| inference, 5553 questions | 1.42 s/q (10-video run) to 1.81 s/q (25-video run, includes index loads) | **2.19 - 2.79 h** |
| inference, pessimal | p95 ≈ 3.0 s/q sustained | 4.63 h |
| **total** | | **2.7 - 3.5 h central, up to ~5.3 h worst case** |

This is **faster** than the 3.8-6.0 h projected in
`tuning/prerun_fixes/test_split_preflight.md`. The reason is the captioner
change in §3: `minicpm-v4.6` is a 1.6 GB checkpoint against `minicpm-v`'s 5.5 GB.
The speedup is a side effect of the same uncontrolled change, not an
optimisation.

Disk: 990 indexes projected at ~1 GB against 27.5 GB free at ingest time. The
runner aborts before starting under 5 GB and aborts cleanly mid-run under 2 GB,
re-checking every 50 videos.

## 5. Resume test — what it skipped on restart

Done with a real `kill -9`, not a simulated one.

1. Started `--split=official_test --limit-videos 25 --resume` on an empty output
   CSV. 25 videos ingested, scoring began.
2. `kill -9` during scoring at 45 flushed rows.
3. Post-kill state: CSV parsed cleanly, **45 complete rows, 19 distinct videos**,
   25 indexes on disk. No truncated or malformed row — each row is flushed and
   `fsync`ed before the next question starts.
4. Restarted the identical command. It reported:

```
[resume] 45 question(s) already present in official_test_e2e_per_question.csv -- these will be skipped
[ingest] fresh_ingests=0 cache_hits=25 (resumed from cache)
[repeat 1/1] scored=127 Acc@QA=0.5512 Acc@GQA=0.2598
[resume] aggregates recomputed over the full CSV: 172 questions (45 resumed + 127 scored now)
```

**Skipped on restart: 45 questions and 25 video ingests. Scored: the remaining
127. Final total 172 — no double-counting, no gaps.** Aggregate metrics are
recomputed over the full on-disk CSV, so a resumed run reports over the whole
split rather than only the portion it personally scored.

### A defect found and fixed during this test

The first kill attempt produced an **empty** CSV despite the run being 50
questions deep. The original script was two-phase — retrieve+caption *every*
question into memory, then answer them all — so nothing reached disk until
captioning was fully done. On the 990-video run that means a crash at hour 3
would lose all three hours, making `--resume` decorative.

Restructured to interleave: retrieve → caption → answer → score → append →
flush → `fsync`, one question at a time. This changes no value —
`context_text` was already finalised per question inside the loop before the
answerer ran, and the answerer is stateless per request (`temperature=0`,
`seed=42`, `cache_prompt=false`) — it only moves the answerer call earlier in
wall-clock time. `--repeat > 1` keeps the batched shape, which it needs in order
to re-answer an identical record set.

### A regression I introduced and then fixed

Worth recording because the first version of this work shipped it. Running the
wider suite caught **4 failures in `tests/test_c0_h2_caption_replay_and_repeat.py`**.
Verified they were mine, not pre-existing: `git checkout -- scripts/val_confirm_e2e_eval.py`
made all 5 pass again, restoring the change re-broke them.

The cause was that the refactor silently bypassed four internal seams the
existing suite monkeypatches to redirect a run into `tmp_path`:

| seam the suite patches | how the refactor broke it | fix |
|---|---|---|
| `TUNING_DIR`, `INDEX_CACHE_DIR`, `PER_QUESTION_CSV` | paths were baked into a `SPLIT_SPECS` dict at import time, so patching the globals did nothing — tests would have written into the real repo | resolve through `out_dir_for()` / `index_cache_dir_for()`, which read the globals at call time; `output_paths_for_mode` returns the module globals for the val_confirm baseline pair |
| `load_val_confirm_questions` | `main` called `load_split_questions` instead | `main` calls the historical alias when `split == "val_confirm"` (the alias delegates, so it is the same code) |
| `ensure_indexes_e2e` (patched with a 2-arg lambda) | `main` passed a `cache_dir=` kwarg | called with the historical two positional args for val_confirm |
| `smoke_test_backend` | `main` called the new `/v1/models` + wire-capture checks instead, which do real HTTP | the hard serving assertions now run for `official_test` only — they are the official-run protocol, and adding a hard network precondition to val_confirm would itself violate "behaves byte-identically" |

Three regression tests were added to pin these seams shut
(`test_directory_resolution_honours_patched_module_globals`,
`test_val_confirm_uses_historical_internal_seams`, and the rewritten
`test_index_caches_are_disjoint_per_split`).

Full suite after the fix: **555 passed, 5 skipped, 1 xfailed, 1 failed**. The
one failure is `tests/test_query_reformulation_v2.py::test_batch_embed_shape_and_l2_normalization`,
which is **pre-existing and unrelated** — verified by stashing this change and
watching it fail identically at HEAD.

## 6. What did not run, and why

- **The 990-video official run.** Not launched, per the task spec: "Do not
  launch the test run — a human launches that after reading your report."
- **A full empirical val_confirm re-run producing a fresh baseline CSV.** Not
  done — it would overwrite `tuning/val_confirm_e2e_per_question.csv`, a
  recorded held-out artifact. Recommended in §3 as a human decision. The
  retrieval/span/gold reproduction check that *was* run is the part that
  isolates this change, which is what the task asked to establish.
- **Byte-for-byte diff of the complete val_confirm CSV.** Unsatisfiable by
  construction — see §3.
- **Mid-run disk-abort path exercised against a real low-disk condition.**
  The thresholds, the every-50-videos re-check, the future cancellation, and
  the `SystemExit(4)` clean-exit path are implemented and unit-covered for
  their constants, but the box has 27.5 GB free and was not filled to trigger
  them for real. **UNDETERMINED** whether the abort behaves correctly against a
  genuine `ENOSPC` mid-ingest.
- **Whether interleaving changes answerer throughput at 5553-question scale.**
  The argument that it cannot change *values* is given in §5 and is solid. Its
  effect on wall-clock at full scale is extrapolated from a 172-question run,
  not measured. **UNDETERMINED.**

## 7. Tests

`tests/test_split_parameterisation.py` — **25 passed**, offline, no model call,
no server. Covers: the split spec table; that `official_test` ids come from the
benchmark_runs manifest and `val_confirm` ids from the repo root; split
disjointness; that the default is `val_confirm`; that the historical two-arg
`output_paths_for_mode` call still resolves to the historical filenames; that
`official_test` cannot write to a val_confirm path; per-split cache isolation;
that directory resolution honours patched module globals; that `main` still
calls the historical internal seams; the `load_val_confirm_questions` alias;
`ensure_indexes_e2e` signature backward-compatibility; that the generalised
loader reproduces the committed val_confirm question set **in order** and with
identical gold; 990/5553 for the test split; the sampler and disk constants; and
the resume ledger including a truncated-final-line case.

Wider suite: **555 passed, 5 skipped, 1 xfailed, 1 pre-existing unrelated
failure** (see §5).

## Protected-file integrity

Hashed before and after. **All byte-identical.**

| file | sha256 |
|---|---|
| `split_manifest.json` | `7864ba3692234320fc453573d6fa8be5e12779f222fd99a841c63a745e98f12a` |
| `dataset_manifest.json` | `ae90d6490c89498455dcb09beb5724bb43611fed9d0365cc8af507014a48f853` |
| `benchmark_runs/.../split_manifest.json` | `1cc8c5f03b5827d3f7152bb049a16663ee37898616402ebb7f87bbba6455f19a` |
| `tuning/frozen_state.json` | `612a93ba88af53eff17d766885dd3bcf8f65e502e4822b60c1a5c04db4c0b1b7` |
| `eval/data/nextqa/test.csv` | `a8d67c5a648c9da8fb4f6fa9939ad1e31fb43f9a92fdab48ec38a1cd622e23db` |
| `eval/data/nextqa/val.csv` | `ee0ef210ae341bec53fdeb723ae54958307ed2e49988d2d3763d3c22abf4201a` |
| `eval/data/nextqa/gsub_test.json` | `0c16c31699238a0f2b8ace45aeed7f193149ce2f011d6ed4f1ef7170ef5d41e9` |
| `eval/data/nextqa/gsub_val.json` | `527f402851836224aa03c9d4b0016b858a678d21596a2348eb2bdb774c74970a` |
| `tuning/blind_ablation/**` (tree digest) | `2d8fcf20278e93a17e1f86c775bc203442f822f4cbfc107465efa6555094d83f` |
| `tuning/determinism_gate/**` (tree digest) | `cfcfa20a9415952dafb52e974d4ca172a792661e088036e24c49f0aa7955a939` |

No hyperparameter was tuned. `tuning/frozen_state.json` was read live and never
written. The span method is unchanged (Method D, `half_width_s=2.2`). Nothing
was downloaded or ingested outside the test split.

## Machine state left behind

- **`llama-server-b10099` is running** on 127.0.0.1:8091 (pid at time of
  writing 1272415), serving `granite4:micro` from the pinned binary. Left up
  deliberately so the launch command in §2 works immediately. Stop it with
  `pkill -f llama-server-b10099` if that is not wanted.
- `tuning/index_cache_official_test/` holds **25 ingested test videos**. Valid
  under the frozen config hash `4edae64ed40256e3`; the run will reuse them.
- The live output path `tuning/official_test/official_test_e2e_per_question.csv`
  is **cleared** — validation output was moved into `smoke_10/` so it cannot be
  mistaken for, or resumed into, the real run.
