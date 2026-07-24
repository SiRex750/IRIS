# END-TO-END val_confirm run — full pipeline, verification layer off

First run in this tuning phase to produce the project's actual target metric
(Acc@GQA), not a retrieval-only proxy. Every prior number (mIoP/mIoU/IoP@0.5)
came from `scripts/part3_tune.py`, which never calls the captioner, answerer,
or Cerberus — it calls `iris.query._retrieve_with_l1` directly. This run used
the real pipeline (captioner + answerer via `iris.aria.generate()`, the same
call path `scripts/nextqa_single_video_eval.py` exercises for one video),
adapted to the full `val_confirm` split, all 12 currently-frozen
hyperparameters applied, and the Cerberus verification layer explicitly
disabled (`cerberus_mode="none"`) — a deliberate simplification for this run,
not an oversight.

Run script: `scripts/val_confirm_e2e_eval.py`. Full log: `tuning/logs/val_confirm_e2e_run.log`.

## Answerer backend: real llama-server, not Ollama

`iris_config.py`'s default `answerer_backend="llama_server"` expects a real
llama-server process at `127.0.0.1:8091` — nothing was listening there at the
start of this task. Rather than falling back to Ollama's `LlamaBackend` (which
`aria.py` itself warns silently drops the `cache_prompt=false` request
parameter required for the seed-based determinism fix), a real llama-server
was built and installed:

- **Binary**: built from source, `llama.cpp` release tag `b10099` (commit
  `1a064ab`), with `GGML_CUDA=ON` targeting `sm_89` (RTX 4090) via the
  CUDA 12.6 toolkit at `/usr/local/cuda-12.6` (the system default `nvcc` at
  `/usr/bin/nvcc` was a stale CUDA 11.5 install that doesn't support Ada and
  had to be bypassed explicitly via `-DCMAKE_CUDA_COMPILER`). No matching
  Linux CUDA prebuilt exists in llama.cpp's release assets (only
  Vulkan/ROCm/SYCL/CPU for Ubuntu x64), and the machine's Vulkan ICD list has
  no NVIDIA entry, so a prebuilt binary would not have reached the GPU —
  building from source was necessary, not a fallback of convenience.
- **Weights**: the `granite4:micro` GGUF blob Ollama already had pulled
  (`~/.ollama/models/blobs/sha256-97c417dcc0534b0737c74016fb2af083cb17c3b51eaac621192d23961b7024eb`,
  2.1GB, `Q4_K_M`, verified valid `GGUF` magic bytes) — reused directly as
  `llama-server -m`, avoiding a redundant multi-GB download of the same
  weights.
- **Launch**: `llama-server -m <blob> --host 127.0.0.1 --port 8091 --alias granite4:micro -ngl 999 -c 8192`.

**Verification (all 3 required checks passed):**
1. `curl http://127.0.0.1:8091/v1/models` → HTTP 200, lists `granite4:micro`
   (3.4B params, `Q4_K_M`, `n_ctx_train=131072`).
2. Sent the identical prompt twice with `cache_prompt: false`. Both responses'
   `timings.cache_n` / `usage.prompt_tokens_details.cached_tokens` = **0** on
   both calls — proof the KV cache was NOT reused for the second identical
   request, i.e. `cache_prompt: false` was genuinely honored, not silently
   dropped (an Ollama endpoint receiving this same flag would not surface a
   `cache_n` field distinguishing this at all).
3. Ran a real question through `iris.aria.generate()` via `IRISConfig()`'s
   default `answerer_backend="llama_server"` → got back a real, parseable
   response: `"ANSWER: B\nREASON: The sky on a clear day is typically blue."`
   → `parse_mc_answer` → index 1 ("B"). Confirmed end-to-end.

Ollama itself was left running throughout (it still serves the captioner,
minicpm-v, at ingest time, per `configs/default_iris_config.json`) — this was
a purely additive second process, port 8091, answerer role only.

## Dataset

`val_confirm` split (`split_manifest.json`, seed 20260721) — the held-out half
of the 567-video split (454 `val_tune` / 113 `val_confirm`) never touched by
any hyperparameter family tuned so far, specifically reserved to catch
overfitting to `val_tune`.

- Nominal `confirm_videos`: 113
- Usable after existence/gold-span filtering (same rule `part3_tune.py`'s
  `load_val_tune_questions()` applies): **112 videos, 639 questions**
  (1 video dropped — same class of filtering that took val_tune's nominal 454
  down to 450, applied here to `val_confirm`'s smaller nominal 113).

## Fresh-ingest confirmation

Dedicated cache dir `tuning/index_cache_val_confirm_e2e/`, confirmed empty
before the run started. Ingest log: **112/112 videos required fresh ingest,
0 cache hits** — this run reused nothing computed for `val_tune` or any prior
hyperparameter family.

## Config

All 12 frozen hyperparameters read live from `tuning/frozen_state.json`'s
`"frozen"` block at run time (not hardcoded):

| field | value |
|---|---|
| retrieval_strategy | hybrid |
| ppr_lambda | 0.50 |
| ppr_damping | 0.50 |
| l2_retrieve_top_k | 4 |
| span_method | D (half_width_s=2.2) |
| peak_distance | 5 |
| peak_prominence | 0.05 |
| packet_size_weight | 0.8 |
| motion_weight | 0.1 |
| luma_entropy_weight | 0.1 |
| persistence_threshold | 0.4 |
| max_prominence | 0.5 |

Plus explicitly set (frozen but not tuned by any Part 3 family):
`ranking_mode="ppr"`, `codec_conf_source="packet_size"`,
`codec_conf_pictype_norm=True`, `cerberus_mode="none"`.

## Results

n_scored = 639 / 639 (100% of usable questions scored; zero retrieval failures).

| metric | value |
|---|---|
| **Acc@GQA (cerberus_mode=none, unverified)** | **0.1894** (121/639) |
| Acc@QA | 0.5462 (349/639) |
| mIoP | 0.3014 |
| mIoU | 0.1572 |
| IoP@0.3 | 0.3834 |
| IoP@0.5 | 0.3099 |
| IoU@0.3 | 0.2379 |
| IoU@0.5 | 0.1127 |
| median retrieval+span construction latency | 5.02 ms |
| p95 retrieval+span construction latency | 6.15 ms |
| median captioning+answering latency | 2217.5 ms |
| p95 captioning+answering latency | 3728.8 ms |
| total wall time | 1370.9 s (≈22.8 min) |

**Answer/grounding decomposition**: Acc@QA (0.5462) is much higher than
Acc@GQA (0.1894) — most of the gap is the answerer getting questions right
without the predicted span reaching IoP≥0.5 against gold, not the answerer
being wrong outright. Every question scored, so this is not survivorship bias
from dropped questions.

## Comparison against val_tune's frozen-config grounding numbers

The `action_score_weights` winning combo (the same weights frozen here) on
`val_tune`: mIoP=0.29782, IoP@0.5=0.30093 (retrieval-only proxy, from
`tuning/frozen_state.json`'s `selection_detail_action_score_weights`).

`val_confirm_e2e` (real end-to-end, held-out split): mIoP=0.3014,
IoP@0.5=0.3099 — both slightly *higher* than the val_tune numbers, not lower.
**No sign of overfitting to val_tune** — grounding quality generalizes to the
held-out split within noise, if anything landing marginally better.

## Spot check

All 639/639 raw answers non-empty and successfully parsed by
`parse_mc_answer` (0 unparseable). All 639/639 predicted spans used the real
CLIP-similarity anchor (`used_clip_anchor=True`, 0 fallbacks to
`retrieved_frames[0]`). Two example raw answers, read directly from
`tuning/val_confirm_e2e_per_question.csv`:

- video `9873067604`, qid 4: predicted "C" (correct, gold "C"), IoP=0.0 (span
  missed gold window despite correct answer — an Acc@QA-but-not-Acc@GQA case).
- video `7508439506`, qid 1: predicted "A" (correct, gold "A"), IoP=0.0
  (same pattern).

Per-question results (predicted/gold answer, predicted/gold span, IoP, IoU,
Acc@QA, Acc@GQA) for all 639 questions: `tuning/val_confirm_e2e_per_question.csv`.

## Known limitations

1. **Verification is off by explicit choice, not oversight.** Cerberus's NLI
   truth-gate is skipped (`cerberus_mode="none"`). Acc@GQA here reflects the
   raw answerer's output with no safety net — any hallucinated or unsupported
   claim the verifier would normally catch stays uncorrected. Labeled
   explicitly as **"Acc@GQA (cerberus_mode=none, unverified)"** everywhere
   above, never bare "Acc@GQA."
2. **`val_confirm` is not the official NExT-GQA test set.** Treat this run's
   Acc@GQA as a checkpoint, not a paper-reportable final number.
3. **`ranking_mode="ppr"` and `codec_conf_source="packet_size"`** are carried
   forward as-is — neither has been compared against its alternative
   (`"legacy"` / `"action_score"`) on this metric; still an open, disclosed
   assumption.
4. **Determinism**: this run used a real `llama-server` backend with
   `cache_prompt=false` genuinely honored (confirmed above) — the Ollama
   determinism concern that originally blocked this run does **not** apply to
   these numbers.
