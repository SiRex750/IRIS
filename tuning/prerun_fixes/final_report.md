# IRIS pre-run fix pack — final report

Branch: `feat/prerun-fixes`. Executed CPU-only, on the Windows checkout at
`C:\Users\swara\IRIS`. No captioner, no answerer, no ingest, no evaluation
pass was run. No NExT-GQA test split video or annotation was touched.

## 1. Go / no-go for the official test run

**NO-GO on this box, for a data reason — not a code reason.** Every code fix
in this pack (S1, X1, A1, A2, C0, H2, C2's wiring) is complete, tested, and
merge-ready. The blocker is that this checkout's local
`eval/data/nextqa/` directory does not hold the real 990-video/5553-question
test split the pipeline actually built on 2026-07-21 (per
`dataset_manifest.json`) — see §2. The official run should proceed on (or
this checkout should be re-synced from) whatever box/artifact store actually
holds the files matching `dataset_manifest.json`'s `files_written` hashes,
then re-run D1 there before starting.

## 2. D1 — dataset verification results

Full detail: `D1_dataset_verification.json`. Summary:

| file | local SHA-256 | matches official_source? | matches files_written? |
|---|---|---|---|
| `val.csv` | `c1b93f52...` | no | no |
| `test.csv` | **MISSING** | n/a | n/a |
| `gsub_val.json` | `0c16c316...` | no | no |
| `gsub_test.json` | `0c16c316...` | **yes** | **yes** |

- `gsub_val.json` and `gsub_test.json` are **byte-identical** (same SHA-256)
  — the task's own stop condition ("If they are equal, stop and report — the
  data is unusable") is triggered. Reported and continued with the rest of
  the pack per the top-level instruction to report blockers and proceed.
- Recomputed directly from the local files (not trusted from any manifest):
  999 videos / 2997 questions in both `gsub_val.json` and `gsub_test.json` —
  matching neither the manifest's expected val (567/3358) nor test
  (990/5553) counts. This is some other/older NExT-GQA snapshot, not the
  pipeline's own output.
- `split_manifest.json`'s 567 tune+confirm video ids have **zero overlap**
  with the local `val.csv`/`gsub_val.json` video ids. Per the task's own
  criterion, zero overlap means the local val files are not val.
- Direct recomputed video-id intersection of local `gsub_val.json` vs
  `gsub_test.json`: **999** (full overlap, because the files are identical) —
  contradicting the manifest's recorded `val_test_video_overlap: 0`, because
  the manifest describes the real dataset build, not what's physically on
  this checkout today.
- Only **91 of the expected 990** test-split `.mp4` files are present.
- This box also does not have `tuning/index_cache_val_confirm_e2e/` at all
  (confirmed before starting D1) — the task's own stated precondition for
  running D1/D2 ("must execute on the box that holds
  `tuning/index_cache_val_confirm_e2e/`") is not met by this checkout.

## 3. D2 — split_guard status

Full detail: `D2_split_guard_status.md`. `benchmark_runs/paper_setup_.../split_manifest.json`'s
`official_test` partition exists but is confirmed unpopulated (`video_count:
0`). Re-executing `split_guard.guard_official_test_command("official_test")`
today reproduces the exact original rejection: `SystemExit(3)`,
"official_test partition is unpopulated." **split_guard does not pass.**

The minimal fix (populate that partition from `test.csv`/`gsub_test.json`)
was **identified but not implemented** — per D1, this box has no verifiable
copy of those files (`test.csv` absent, local `gsub_test.json` is a stale/
mismatched snapshot), so populating from what's here would require judgment
calls about which of the 999 present-but-uncertain video ids belong in the
real 990, which the task instructions say to stop on rather than decide
unilaterally. Both stale documents referencing this blocker
(`commands/13_run_official_nexgqa_test.md`, `setup_report.md`) were updated
with dated correction blocks (original text preserved) noting that the 2026-07-21
acquisition made the blanket "unavailable" claim stale, while confirming the
structural blocker itself is still real on this specific checkout.

## 4. C2 — caption audit

Full detail: `C2_caption_audit.md` + `C2_caption_stats.json`. From the 2556
captions in `tuning/blind_ablation/captions_dump.json` (the recorded 639-question
blind-ablation run, arm A, question-blind):

- **Empty-caption rate: 0%** (0/2556). **`[CAPTION_FAILED]` rate: 0%** (0/2556).
  Prior Acc@QA/Acc@GQA numbers for that specific run were **not** computed
  with any blank evidence.
- **Truncation-risk: none observed.** Max caption length ≈142 tokens, well
  under the 400-token first-attempt `num_predict` ceiling.
- 99.8% (534/535) of frames revisited by more than one question in the same
  video reused byte-identical cached captions — the per-video cache is
  behaving correctly.
- Qualitatively, sampled captions are generic scene descriptions, not
  question-aware — plausibly relevant to the standing 61.1%-vs-51.7% grounded/
  ungrounded accuracy gap noted in project memory, though this audit alone
  cannot prove causality.
- This is a retrospective read of one already-recorded 639-question run, not
  a guarantee for the 990-video official run. `get_minicpm_truncation_stats()`
  is now wired into `scripts/val_confirm_e2e_eval.py` (see §5) so the rate is
  visible going forward without needing another manual audit.

## 5. Code fixes

| Item | File(s) | What changed | Test |
|---|---|---|---|
| **S1** | `scripts/val_confirm_e2e_eval.py`, `scripts/blind_ablation_eval.py`, `scripts/query_reformulation_v2_ablation.py`, `scripts/part3_tune.py` | Every call site of `predicted_span_from_frames_peak` now passes `duration_s=q["duration"]` (or forwards it through `default_predicted_span`), so spans clamp to end-of-video instead of extending past it. | `tests/test_s1_duration_s_call_sites.py` (4 tests; confirmed to fail pre-fix, pass post-fix) |
| **X1** | `iris/l1_elysium.py::as_context_text` | Frame blocks now sorted by `(timestamp_sec, frame_idx)` ascending before joining, instead of dict-insertion (retrieval-rank) order. `set_facts` untouched. | `tests/test_x1_chronological_context.py` (3 tests; confirmed fail pre-fix, pass post-fix) |
| **A1** | `iris/aria.py` — `LlamaServerBackend` (openai-client kwargs path, schema-format `requests.post` payload, native `/completion` fallback payload) and `LlamaBackend` (default chat path, `response_format` path, native `/api/chat` options) | Added `top_k=1, top_p=1.0` (or the Ollama-native equivalent in `options`) alongside the existing `temperature=0.0`, on every request-construction path in both classes. | `tests/test_a1_greedy_decoding.py` (6 tests; confirmed fail pre-fix with `KeyError`, pass post-fix). Two pre-existing exact-match assertions in `tests/test_aria.py` were updated to reflect the new `extra_body` content (this is the deliberate A1 behavior change, not scope creep). |
| **A2** | `scripts/answerer_provenance.py` (new), wired into `scripts/val_confirm_e2e_eval.py::main()` | New provenance-capture helper: serving process cmdline (via `/proc/<pid>/cmdline`), server binary SHA-256 + `--version`, GGUF path/hash, `GET /v1/models` response, and the exact sampler params sent. Written to `tuning/val_confirm_e2e_environment<suffix>.json` right after the answerer smoke test. | `tests/test_a2_answerer_provenance.py` (8 tests, all passing) |
| **C0** | `scripts/caption_dump_io.py` (new), wired into `scripts/val_confirm_e2e_eval.py` | `--caption-dump <path>` writes every generated caption in the `{video:{qid:{frame_idx:caption}}}` schema; `--caption-load <path>` populates captions from such a dump and skips captioning entirely, raising `KeyError` loudly on any miss. Neither flag changes default behaviour. `iris/` production code untouched. | `tests/test_c0_h2_caption_replay_and_repeat.py` tests (a)/(b)/(c) — default-unchanged, dump-then-load round-trip, and loud-miss-on-load (confirmed fail pre-fix with `AttributeError`, pass post-fix) |
| **H2** | `scripts/val_confirm_e2e_eval.py` | `--repeat N` (default 1) runs the answer stage N times over frozen retrieval + captions (requires `--caption-load`, enforced with a clear `SystemExit` if omitted). Writes one per-question CSV per repeat plus `val_confirm_e2e_repeat_summary<suffix>.json` with mean/min/max/flip-counts for Acc@QA and Acc@GQA. Grounding metrics (mIoP/mIoU/IoP@0.5/IoU@0.5) reported once, not per repeat. | Same test file as C0 (`test_repeat_requires_caption_load`, `test_repeat_reports_mean_min_max_and_flip_counts_and_grounding_once`) |
| **C2 wiring** | `scripts/val_confirm_e2e_eval.py` | `aria.get_minicpm_truncation_stats()` (existed, was never called by any eval script) is now called once per run and included in the final `VAL_CONFIRM_E2E_METRICS_JSON` output under `minicpm_truncation_stats`. | `tests/test_c2_truncation_stats_wiring.py` (confirmed fail pre-fix with `AttributeError`, pass post-fix) |

All 47 tests across the seven new/touched test files pass
(`./.venv/Scripts/python.exe -m pytest tests/test_s1_duration_s_call_sites.py
tests/test_x1_chronological_context.py tests/test_a1_greedy_decoding.py
tests/test_aria.py tests/test_a2_answerer_provenance.py
tests/test_c0_h2_caption_replay_and_repeat.py
tests/test_c2_truncation_stats_wiring.py -q` → 47 passed). Full repo suite:
`510 passed, 5 skipped, 1 xfailed, 5 failed` — the 5 failures are pre-existing
in `tests/test_qvhighlights_setup.py` (a Windows tmp-path escape-guard issue
in `scripts/setup_qvhighlights.py`, unrelated to any of these 9 items and not
touched by this pack).

## 6. Binary inventory

Full detail: `A2_binary_inventory.json`. `find / -iname 'llama-server*' -type
f` returned **zero hits** on this box. The `llama.cpp` source checkout present
in the repo has never been compiled here (no build directory, no `.exe`
files) and its current HEAD (`b9985-1-g91c631b21`) does not even contain
commit `1a064ab` in its history. **The from-source `b10099`/`1a064ab` build
that produced the recorded val_confirm_e2e headline numbers does not exist on
this box.** This is itself a finding, not a gap in the search: the recorded
run's exact serving environment is confirmed gone from this checkout. The
newly-added `scripts/answerer_provenance.py` (A2) is the mechanism to prevent
this from recurring silently on future runs.

## 7. Not done, and why

- **D2's minimal population fix** — not applied; requires data this box
  doesn't verifiably have (see §3).
- **A1 GPU determinism verification** — explicitly deferred per Part D
  instructions; specified in full in `D_deferred_gpu_gate.md` for the GPU box
  to execute. Not attempted on CPU, since a CPU pass would be falsely
  reassuring.
- **Any run of the captioner, answerer, ingest, or evaluation pass** — out of
  scope per the task ("This entire task runs CPU-only... No captioner run, no
  answerer run, no ingest, no evaluation pass").
- **Any hyperparameter tuning or sweep** — out of scope, none attempted.
- **`tuning/blind_ablation/**`** — read-only throughout (hashed, used as C2's
  audit source), never modified; confirmed byte-identical in
  `protected_hashes_before.txt` vs `protected_hashes_after.txt`.
