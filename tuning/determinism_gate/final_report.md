# A1 determinism gate -- final report

**Generated:** 2026-07-27, on `worker-1` (`/home/ccbd/IRIS-1`)

## 1. PASS / FAIL

**PASS.** All 639/639 raw answer strings are byte-identical across all three
repeat passes, using the pinned `b10099`/`1a064ab` binary served with
`--parallel 1` and the sampler params pinned by the A1 fix in `iris/aria.py`.

## 2. Pinned binary

- **Source path:** `/home/ccbd/.claude/jobs/de60ff71/tmp/build/llama.cpp/build/bin/llama-server` (another job's ephemeral scratch dir, verified still present and hash-matching immediately before the copy)
- **Destination path:** `/home/ccbd/.local/iris/bin/llama-server-b10099` (`/opt/iris/bin` was not writable by this user -- uid 1001, no passwordless sudo -- so a stable, job-independent path outside the repo was used instead: `/home/ccbd/.local/iris/bin`)
- **Source SHA-256:** `250d2eb35a6ddfa1df060599f8bc3169944ef60a77429b2cc395e062859e0a06`
- **Destination SHA-256:** `250d2eb35a6ddfa1df060599f8bc3169944ef60a77429b2cc395e062859e0a06` (match)
- **`--version` from destination:** `version: 1 (1a064ab)` / `built with GNU 12.3.0 for Linux x86_64`
- **Dependencies:** 8 shared libraries (`libllama-server-impl.so`, `libllama-common.so.0`, `libmtmd.so.0`, `libllama.so.0`, `libggml.so.0`, `libggml-base.so.0`, `libggml-cpu.so.0`, `libggml-cuda.so.0`) resolved from the doomed job-scratch tree per `ldd`; all copied alongside the binary. The binary's baked `RUNPATH` still points at the job-scratch path (`readelf -d`), but the dynamic linker checks `LD_LIBRARY_PATH` before `RUNPATH`, so `LD_LIBRARY_PATH=/home/ccbd/.local/iris/bin` correctly overrides it -- verified via `ldd` (all 8 libs resolve to the destination dir) and a working `--version` invocation from the destination.
- **Full record:** `tuning/prerun_fixes/pinned_binary.json`

## 3. Byte-identity and flip counts

From `tuning/determinism_gate/byte_identity_analysis.json` (derived from
`raw_answers_log.jsonl`, 1918 logged `aria.generate` calls = 1 smoke-test call
+ 3x639 answer-stage calls, in strict call order):

| comparison | identical raw strings |
|---|---|
| pass1 vs pass2 | 639/639 |
| pass1 vs pass3 | 639/639 |
| pass2 vs pass3 | 639/639 |
| all 3 identical | 639/639 |

**Parsed-answer-index flip count: 0/639** (independently confirmed both by
the byte-identity analysis and by `repeat_summary.json`'s
`n_flipped_questions: 0` for both `Acc@QA` and `Acc@GQA_unverified`).

No mismatches occurred, so the "if it fails" diagnostic protocol (fresh-process
2-pass re-run) was not needed and was not run.

## 4. Acc@QA per pass, and grounding metrics (once)

| pass | Acc@QA | Acc@GQA (unverified) |
|---|---|---|
| 1 | 0.539906 | 0.186228 |
| 2 | 0.539906 | 0.186228 |
| 3 | 0.539906 | 0.186228 |
| **mean** | **0.539906** | **0.186228** |
| **min** | **0.539906** | **0.186228** |
| **max** | **0.539906** | **0.186228** |

Grounding metrics (deterministic, reported once): `mIoP=0.306294`,
`mIoU=0.160891`, `IoP@0.5=0.314554`, `IoU@0.5=0.118936`.

**These numbers are not, and are not intended to be, a reproduction of the
recorded `Acc@QA=349/639=0.5462` from the original headline run.** That
number is permanently unreproducible (captions never persisted, binary path
never recorded for that run). This run used a fresh, independently-verified
ingest and the frozen `captions_dump.json`, and got a self-consistent
0.539906 across 3 passes -- close to but not identical to 349/639, which is
expected and was not tuned toward.

## 5. `answerer_provenance.py` -- did it capture the pinned hash correctly?

**Yes.** `tuning/determinism_gate/environment.json` records:

```json
"binary_path": "/home/ccbd/.local/iris/bin/llama-server-b10099",
"binary_sha256": "250d2eb35a6ddfa1df060599f8bc3169944ef60a77429b2cc395e062859e0a06",
"binary_version_output": "version: 1 (1a064ab)\nbuilt with GNU 12.3.0 for Linux x86_64",
"cmdline": ".../llama-server-b10099 -m ... --parallel 1",
"sampler_params_sent": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 42, "cache_prompt": false}
```

`binary_sha256` matches the pinned reference hash exactly, `cmdline` confirms
`--parallel 1` was in effect, and `sampler_params_sent` matches the A1-pinned
values exactly. This run doubles as an end-to-end pass of the A2 provenance
tooling itself.

## 6. Recommendation: is the official 990-video run cleared to proceed?

**Yes, with one action item.** The determinism property the official run
depends on -- byte-identical answers under repeat, at temperature 0, with
`cache_prompt=false` and `--parallel 1` -- is now demonstrated (639/639), on
this box, with the exact pinned binary and sampler params, and the pin is
recorded at a stable, job-independent path (`tuning/prerun_fixes/pinned_binary.json`).
Action item before the official run: launch its `llama-server` from
`/home/ccbd/.local/iris/bin/llama-server-b10099` with
`LD_LIBRARY_PATH=/home/ccbd/.local/iris/bin` and `--parallel 1` set explicitly
in whatever launch script/systemd unit is used -- do not reuse
`/usr/local/lib/ollama/llama-server` (wrong build, drops `cache_prompt=false`)
and do not omit `--parallel 1` (removes the isolation this gate tested under).

## 7. What did not run, and why

- **The NExT-GQA test split was not run.** Out of scope -- this gate clears
  it, it does not run it.
- **The captioner was not run.** Captions loaded frozen from
  `tuning/blind_ablation/captions_dump.json` via `--caption-load`;
  `tuning/blind_ablation/**` was not modified (hash-verified unchanged, see
  below).
- **No hyperparameter was tuned; sampler params were not changed** from the
  A1-pinned values (verified from the actual `/v1/chat/completions` payload
  and from `answerer_provenance.py`'s captured `sampler_params_sent`, not
  just from reading `iris/aria.py`'s source).
- **The fresh-process 2-pass diagnostic was not run** -- only required "if it
  fails," and it passed on the first (raw-logged) run.
- **The first (non-instrumented) run of this gate was discarded and
  re-run.** It completed successfully (identical aggregate Acc@QA/Acc@GQA,
  0 parsed-answer flips) but did not capture raw answer strings, which the
  byte-identity requirement needs; re-running with `aria.generate`
  instrumentation added (pure logging wrapper, no logic change, no edits to
  any tracked file) cost about the same ~6 minutes as the first run and
  produced the byte-identical result reported above. Both runs used a
  dedicated fresh `tuning/determinism_gate/index_cache/` (gitignored,
  `*.npz`, not committed) so as not to touch or depend on
  `tuning/index_cache_val_confirm_e2e/`, which remains untouched from the
  prior D1 verification.
- **Protected files**: confirmed byte-identical before/after (see
  `$CLAUDE_JOB_DIR/tmp/protected_before2.txt` vs `protected_after2.txt` in
  the commit process) -- `tuning/frozen_state.json`, `dataset_manifest.json`,
  `split_manifest.json`, `tuning/val_confirm_e2e_*`,
  `tuning/blind_ablation/**`, all four `eval/data/nextqa/` annotation files.
