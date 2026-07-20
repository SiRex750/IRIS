# Day-2 Production Smoke Test Report

Executed 2026-07-20, ~14:34-14:48 (local log timestamps), git SHA `1e431b7a94f6deb9e14cb7c69a504dfcf3d4c731`,
branch `fix/charon-full-decode-geometry`. Harness: `smoke_scripts/run_smoke.py`. This is a REAL run:
3 videos were decoded, real CLIP/moondream/granite4:micro/DeBERTa-NLI models were loaded and queried,
24 real `iris.query.query()` calls were made. No ground truth was used anywhere in ingestion,
retrieval, captioning, or answering — gold spans were loaded only into `smoke/selected_ids.json` for
post-hoc reporting and were never passed into any pipeline call.

## Selected IDs (recorded before execution — `smoke/selected_ids.json`)

3 videos: `6936757706`, `3079724515`, `8900428927` (all from the `data/nextqa_exp1a` placeholder
subset; see dataset caveat below). 12 questions: 6 causal (CW/CH), 6 temporal (TN/TC); gold-span
total length ranges 1.8s-6.4s (3 "short" ≤2.5s, 9 "long" >2.5s — this dataset's videos are 10-29s
long, so spans don't reach the tens-of-seconds scale; short/long is relative to this candidate pool).

**Dataset caveat (carried from `benchmark_runs/paper_setup_20260720T074844Z_1e431b7/dataset_manifest.json`):**
these are genuine NExT-QA validation-split rows with genuine human-annotated gold spans from
`eval/data/nextqa/val.csv` / `gsub_val.json`, but this repo could not independently verify
`gsub_val.json` as the authentic official 567-video NExT-GQA split (it has 999 keys and is
byte-identical to `gsub_test.json`). This smoke test proves pipeline *mechanics*, not an official
NExT-GQA accuracy number.

## LAYER 1 (`smoke/layer1_outputs.csv`, per-frame detail in `smoke/per_question_trace.jsonl`)

| video | duration_s | total_frames | frames_decoded_pass2 | survivor_count | retention_% | l1_runtime_s | l1_rss_delta_MB |
|---|---|---|---|---|---|---|---|
| 3079724515 | 29 | 874 | 119 | 119 | 13.62 | 327.90 | 912.62 |
| 6936757706 | 20 | 600 | 80 | 80 | 13.33 | 27.31 | 62.89 |
| 8900428927 | 10 | 308 | 33 | 33 | 10.71 | 11.06 | -34.81 |

**Measurement caveat:** `3079724515` was ingested first (alphabetical video order) and its
327.9s/912MB includes one-time cold model loading (CLIP, spacy `en_core_web_sm`, first faiss/torch
import) that is NOT isolated from pure decode+feature-extraction time — this harness did not
implement the cold/warm model-load separation that `timer_registry.json` (from the paper-setup
snapshot) specifies. The other two videos' L1 runtimes (27.3s, 11.1s) are the more representative
per-video decode cost with models already warm. This is a real gap to fix before official timing.

`survivor_count == frames_decoded_pass2` for all 3 videos — confirms `retrieval_strategy="hybrid"`
(the config default) indexes every frame that survived Pass-1 candidate thresholding and got
Pass-2 decoded, with no further down-selection before L2. Matches `canonical_pipeline.md`.

Per-frame `selected_frames` (frame_idx, timestamp, action_score, persistence_value, codec_conf,
is_peak, pict_type) for all 232 survivor frames across the 3 videos are in
`smoke/per_question_trace.jsonl` (layer2.l1_telemetry is per-question; full survivor lists are
implicitly the same per-video set consumed by every question against that video — see
`layer1_outputs.csv` companion detail in the trace file's `layer2.top_k_ordered` entries, which
are drawn from this survivor pool).

## LAYER 2 (`smoke/layer2_outputs.csv`)

Node/edge counts match the L1 survivor counts exactly (80/119/33 nodes for the 3 videos — the L2
graph is built 1:1 over L1 survivors, no separate L2-side admission). Edge-type breakdown shows
`hierarchical_sparse` mode's temporal / semantic_salient / hierarchy_peak_salient / motion_neighbor
edges and their overlaps, consistent with `canonical_pipeline.md`.

**Retrieval branch mix (from `scene_retrieval` debug lines in `smoke/stdout.log`):** 36/48 calls
(12 questions × 2 runs × ... wait, 12×2=24 `query()` calls, each internally calling
`_build_retrieved` twice — once in the L2-instrumentation call, once inside the real `query()`
call — so 48 total `retrieve_scene_sparse` invocations) took the **DESCEND** branch (margin ≤
0.015, real PPR run over an induced subgraph); 12/48 took the **SHORTCUT** branch (margin > tau,
exact top-k, no PPR). On SHORTCUT calls, `top1_ppr_score` is correctly `None` in
`layer2_outputs.csv` (all 3 of `8900428927`'s first three questions) — this is the expected,
code-confirmed behavior from `configs/peak_span_modes.json`/`scene_retrieval.py`: SHORTCUT returns
`retrieval_contributions: {}`, no `sem_rank`/`codec_rank`/PPR score is computed for those
questions, and this report does **not** fabricate a value for them.

`selected_peak` / `predicted_temporal_span` fields in `per_question_trace.jsonl` are explicitly
labeled `*_NOT_A_REAL_PIPELINE_STAGE` — see `canonical_pipeline.md`: there is no CLIP-peak-reranking
or predicted-span stage in production code. What's reported is the rank-1 retrieved frame and the
min/max envelope of retrieved timestamps (the `scattered_minmax_span` legacy quantity), never
presented as a genuine model output.

**Peak-in-gold:** intentionally NOT computed inline by the harness or exposed to the pipeline —
per the correctness gate ("peak-in-gold only after prediction is frozen and passed to evaluator"),
computing it here would risk the appearance of gold-span information touching the retrieval call.
It can be computed post-hoc, outside the pipeline, from `per_question_trace.jsonl`'s
`retrieved_frame_idxs`/timestamps against `selected_ids.json`'s `gold_spans` — not done in this
report to keep that separation explicit and auditable.

## LAYER 3 (`smoke/layer3_outputs.csv`)

12/12 questions produced a captioning+answering+verification cycle with no exceptions. 1/12
(`6936757706`/qid 11) produced a non-abstained "verified-ish" answer with real evidence citations;
the other 11 abstained (`"Insufficient verified evidence to answer this question."`) after
CerberusV rejected/marked-unverifiable the answerer's claims against the caption-only fact set —
this is the CerberusV legacy gate working as designed (strict; no claim survives unless directly
entailed by a caption), not a pipeline failure. Latencies vary widely (9.7s-452.6s total) mostly
driven by `caption_latency_s` (0s-336s) — CPU moondream captioning cost scales with
`frames_captioned` (0-574 in this run; `frames_captioned=574` for `8900428927`/qid 2 is notably
high and worth a follow-up look at why `_ensure_captions` decoded so many frames for a 33-survivor-node
video — likely GOP-seek decoding more frames than nodes because of pict_type dispersion, not a bug
confirmed in this pass).

## CORRECTNESS GATES

| # | Gate | Result | Evidence |
|---|---|---|---|
| 1 | One canonical ingest path | PASS | Every video went through `iris.ingest.ingest()` only; see harness docstring/code |
| 2 | One canonical query path | PASS | Every question went through `iris.query.query()` only for the actual answer; the only other call is a read-only instrumentation call to the identical internal functions (`_call_embed_query`, `_retrieve_with_l1`) |
| 3 | No benchmark-only alternative retrieval logic | PASS | No retrieval logic is reimplemented in `run_smoke.py`; L2 telemetry is read from the real call's return value and node annotations |
| 4 | No ground-truth data enters retrieval, captioning or answering | PASS | `selected_ids.json`'s `gold_spans`/`gold_answer_idx` are read only for post-hoc trace metadata (video_id, qid, family, span category) written alongside results; never passed to `ingest()`, `query()`, or any embedding/captioning/answering call — confirmed by code inspection of `run_smoke.py::run_one_question` |
| 5 | Same result after cache save/reload | PASS (3/3 videos) | `smoke/cache_parity.json`: structural graph fingerprint AND per-frame embedding hash identical for fresh vs. reloaded index, all 3 videos |
| 6 | Same result across two deterministic repeated runs | **PARTIAL: 11/12 (91.7%)** | `smoke/determinism.json`. Retrieval (`retrieved_frame_idxs`) was identical in all 12/12 pairs. The 1 failure (`6936757706`/qid 7) diverged only in the answerer's free-text generation (raw LLM sampling via Ollama, no fixed seed/temperature=0 set anywhere in the config or `aria.py`'s `LlamaBackend`), which cascaded into different claim-splitting and a different abstention outcome. **Root cause identified, not hidden: this is a real non-determinism in Layer 3's LLM call, not L1 or L2.** |
| 7 | No production graph mutation during query | PASS (24/24 calls) | `smoke/graph_nonmutation.json`: structural fingerprint (node/edge/weight/edge_type set, excluding the documented per-query `retrieval_contributions`/`last_retrieval_score` node annotations) identical before/after every `query()` call |
| 8 | Every result contains traceable frame IDs and timestamps | PASS | Every `layer2.top_k_ordered` entry and `layer3.retrieved_frame_idxs` in `per_question_trace.jsonl` carries `frame_idx` + `timestamp_s` |
| 9 | No missing method is replaced by another method | PASS, with one documented deviation | Answerer backend: config default `llama_server`@8091 was not running in this environment; harness explicitly set `answerer_backend="llama"`@11434 (Ollama, already running) with the **same model** `granite4:micro` — recorded in `run_smoke.py`'s docstring and here, not silently substituted, and no method was silently swapped for a *different* method |
| 10 | All failures are recorded | PASS | `smoke/failures.json`: 0 ingest errors, 0 question errors (all 24 `query()` calls completed without exception); the 1 determinism gate failure is recorded in `determinism.json`, not suppressed |

## Overall

9.5/10 gates fully pass; gate 6 (determinism) passes at 91.7% (11/12) with a clearly identified,
non-hidden root cause in Layer 3 LLM sampling. This is a genuine, actionable finding, not a smoke
test failure to explain away: **the answerer call has no fixed seed/temperature=0, so repeated
identical questions against an identical, unmutated index can produce different final answers.**
Recommended before any accuracy number is trusted: pin `temperature=0` (or an explicit seed) on the
`LlamaBackend`/`LlamaServerBackend` chat-completion call in `iris/aria.py`.

No other correctness gate failed. Ingest -> save -> reload -> query round-trips are bit-identical.
The production graph is never structurally mutated by a query. No ground truth touched any pipeline
call. Only one canonical ingest path and one canonical query path were used throughout.

## Confirmation

This was a REAL run: real video decode, real CLIP/moondream/granite4:micro/DeBERTa-NLI inference,
real captions, real answers, real verification decisions. Not a metric-evaluation run — no
Acc@QA/Acc@GQA/IoP/IoU was computed here (that would require the official dataset per
`benchmark_runs/paper_setup_20260720T074844Z_1e431b7/dataset_manifest.json`, which remains
unresolved). This report proves pipeline *mechanics and correctness*, not benchmark accuracy.
