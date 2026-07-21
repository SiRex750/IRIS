# IRIS Canonical Production Pipeline (code-grounded trace)

Captured 2026-07-20 against `fix/charon-full-decode-geometry` @ `1e431b7`. Method: read source directly and
traced the real call graph, not file-name assumptions.

## 0. Real call graph

`api.py` (`POST /api/process`) -> `iris.ingest.ingest(video_path)` (build once, cached by SHA-256 in
`_INDEX_CACHE`) -> `iris.query.query(question, index)` (per-question, no video re-read, no graph rebuild).

`iris.pipeline.run_pipeline()` is a thin harness that itself just calls `iris.ingest.ingest()` then
`iris.query.query()`. It does **not** call its own module-level `wrapper_*` functions defined earlier in
`pipeline.py` (`wrapper_init_l1_cache`, `wrapper_populate_cache`, `wrapper_l2_retrieve`,
`wrapper_cerberus_gate`, `get_clip_model`, `get_frame_clip_embedding`, ...) — **those are dead code** from
`api.py`/`query.py`'s perspective. They exist only because `benchmarks/paper_benchmark.py` and
`benchmarks/exp1a_budget_matched.py` still import `wrapper_l2_retrieve` directly. `iris/query.py`
re-declares its own near-identical copies of the L1/Cerberus wrapper logic rather than importing from
`pipeline.py`.

Default config (`IRISConfig` in `iris/iris_config.py`; `configs/default_iris_config.json` only overrides
`salient_thresh`, `candidate_thresh`, `alpha`, `beta`, `peak_order`) has `graph_mode="scene_sparse"` and
`ranking_mode="ppr"`. So the **actual default L2 entry point is `iris/scene_retrieval.py:retrieve_scene_sparse`**,
not a direct call to `L2Asphodel.retrieve`/`retrieve_ppr` — those are the primitives `scene_retrieval.py`
calls internally.

## MAJOR CORRECTION vs. the architecture assumed by the benchmark task prompt

**No CLIP-peak-reranking / predicted-temporal-span stage exists anywhere in `iris/`.** Grepped for
`span|peak_rerank|rerank|predicted_span|half_width|boundary_clamp` — no matches outside `charon_v`'s
"PEAK" *admission tier* terminology (which is a Layer-1 concept, not a query-time reranking stage) and
unrelated retrieval-strategy code. There is no CLIP-based reranking of PPR candidates, no predicted-span
object, and no boundary-clamping logic. The pipeline goes:

```
video -> L1 admission (Charon-V + ActionScoreModule) -> survivor frames
      -> L2 retrieval (scene_retrieval.retrieve_scene_sparse -> PPR/exact) -> retrieved frames
      -> lazy per-frame captioning (query._ensure_captions -> ARIA captioner)
      -> ARIA answerer -> Cerberus-V verification (legacy) or claim-contract v2 -> final answer
```

There is **no separate temporal-span-prediction output** — IRIS answers from a *set* of retrieved frames,
not a predicted `[start, end]` interval. This has direct consequences for Section 7/9 of this setup (see
`lambda_semantics.md` / method registry / layer-factorial notes): "peak selection" and "span construction"
as specified in the task prompt do not correspond to any existing code path and are registered as
**NOT_IMPLEMENTED / requires new code**, not configuration of an existing stage.

## LAYER 1 — Charon-V (`iris/charon_v.py`) + Action Score (`iris/action_score.py`)

| Responsibility | File / Function | Inputs | Outputs | Config | Cache | Failure modes | Production? |
|---|---|---|---|---|---|---|---|
| Codec validation | `iris/codec_validator.py::validate_video` | video path, `level` | `ValidationResult` | `codec_validation_level` | none | `ValueError` on reject in `ingest()`; import failure silently swallowed in both `api.py` and `ingest.py` | Yes |
| Zero-decode packet-size curve | `charon_v._demux_packet_curve` | video path | `(all_frame_energies, iframe_indices, energies, pts_to_packet)` | n/a | n/a | `ValueError` if no video stream | Yes |
| Adaptive per-scene thresholding | `charon_v.parse_video` (Pass 1) | packet curve, thresholds | per-scene `(salient, candidate)` percentile cuts | `threshold_mode`, `salient_thresh`, `candidate_thresh` | n/a | falls back to global percentile if scene empty | Yes |
| Peak detection (legacy tier) | `charon_v.detect_peaks` | packet curve, thresholds, `order` | PEAK frame_idx set | `peak_order` | n/a | — | Computed but explicitly commented legacy/unused for scoring — real admission signal is continuous `action_score` |
| Decode + feature extraction (Pass 2) | `charon_v.parse_video` main loop | video path, thresholds, `full_decode` | `output_frames` (tier, luma_diff_energy, packet_size, motion_magnitude, motion_vectors, geometry), `stats` | `candidate_thresh`, `salient_thresh`, `adaptive`, `peak_order`, `threshold_mode` | none (in-memory) | frame/packet count mismatch raises unless PTS None/dup | Yes |
| Motion vector geometry | `charon_v.compute_motion_geometry` | raw MVs, frame w/h | divergence, curl, Jacobian Frobenius norm, Hessian max eigenvalue, motion entropy | n/a | n/a | all-zero dict on degenerate input; sqrt clamp for Hessian | Yes |
| Luma diff / entropy | inline in `parse_video` | Y-plane ndarray | `luma_diff_energy` (vs previous retained frame), `luma_entropy` | n/a | n/a | entropy try/except -> 0.0 | Yes |
| Scene-boundary segmentation | `charon_v.compute_valley_scene_boundaries` | packet curve, iframe_indices, fps | scene spans (hard cap 300 frames) | n/a (hardcoded) | n/a | — | Yes, feeds `scene_id` for `graph_mode="scene_sparse"` |
| Continuous action score | `action_score.ActionScoreModule.score_all` | packet_size, motion_magnitude, luma_entropy | `action_score` [0,1], `is_peak`, `persistence_value` | `packet_size_weight`(0.5), `motion_weight`(0.3), `luma_entropy_weight`(0.2), `peak_distance`, `peak_prominence`, `persistence_threshold` | n/a | `ValueError` on negative/zero weight sum | Yes |
| Peak persistence (video-wide NMS) | `ingest._build_index_from_records` step 2 | `is_peak` set, `nms_window` | de-duplicated peaks | `nms_window` (default 10) | n/a | — | Yes |
| Frame admission / retention budget | `ingest._build_index_from_records` step 4 | `output_frames`, `retrieval_strategy` | `frames_to_index` | `retrieval_strategy` ("hybrid" default = all output frames) | n/a | config-driven, not query-driven (explicit) | Yes |
| Codec-confidence signal | `ingest._build_index_from_records` step 6.5 | packet_size/action_score, pict_type | `codec_conf` in [0.1,1.0] | `codec_conf_source`, `codec_conf_pictype_norm` | stored on node/record | — | Yes; used only by PPR seed, not admission |
| Cache create/load | `ingest.save_index`/`load_index` | `IRISIndex` | `.npz` (JSON manifest + `emb_{idx}` arrays) | n/a | `.npz` on disk; `api.py` uses in-memory `_INDEX_CACHE` instead | deterministic rebuild via `_build_graph` | Yes (`.npz` path used by benchmarks; `api.py` uses in-memory cache) |

## LAYER 2 — `iris/l2_asphodel.py` + `iris/scene_retrieval.py` + `iris/l2_index.py`

| Responsibility | File / Function | Config | Production? |
|---|---|---|---|
| Frame->node conversion | `L2Asphodel.add_frame_nodes_bulk` | n/a | Yes |
| Node tier assignment | `L2Asphodel._assign_tier` | `l2_salient_action_thresh`, `candidate_thresh` | Yes |
| Edge construction dispatch | `L2Asphodel._update_all_edge_weights` | `graph_edge_mode` (default `hierarchical_sparse`), `graph_temporal_window`, `graph_semantic_top_k`, `graph_motion_top_k`, `graph_semantic_threshold` | Yes |
| Temporal edges | `_add_temporal_edges` | window=`graph_temporal_window` | Yes (in `hierarchical_sparse`) |
| Hierarchy edges | `_add_hierarchy_edges` | tiers | Yes, falls back to all nodes if no salient/peak in scene |
| Semantic edges | `_add_salient_semantic_edges` | `graph_semantic_top_k`, `graph_semantic_threshold` (FAISS `IndexFlatIP` if K>=200) | Yes |
| Motion-neighbor edges | `_add_motion_neighbor_edges` | `graph_motion_top_k` (cKDTree if N>=200) over 6-D motion vector | Yes |
| Cross-scene edges | `L2Asphodel.add_cross_scene_edges` | `scene_crossscene_mode` (default `rep_only`), `scene_crossscene_threshold_pctile` | Yes, but written only into a temporary subgraph copy inside `scene_retrieval`'s DESCEND branch — never merged into the persistent production graph |
| Query embedding | `iris/_clip.py` via `iris.query._embed_query` | `clip_revision` | Yes, CUDA->CPU fallback with telemetry |
| **Personalization vector ("lambda")** | `L2Asphodel.retrieve_ppr` (lines ~1271-1281) | `ppr_lambda` (default 0.5), `ppr_damping` (default 0.5) | See `lambda_semantics.md` — exact formula extracted, not assumed |
| Semantic/codec rank | `l2_asphodel._rank_pct` | n/a | rank-percentile transform, not raw cosine/codec_conf |
| PPR | `L2Asphodel.retrieve_ppr` | `ranking_mode="ppr"` (default) | Yes — actual default algorithm; `nx.pagerank(personalization=seed, alpha=damping)`, stable id tie-break; falls back to unpersonalized teleport only on `nx.NetworkXError`/`ZeroDivisionError` |
| Legacy/direct retrieval | `L2Asphodel.retrieve` (alpha*semantic + beta*motion_sim + gamma*persistence + delta*pagerank) | `alpha,beta,gamma,delta` sum to 1.0 (validated), `ranking_mode="legacy"` | Alive, not default |
| Scene-sparse coarse prune + margin gate | `scene_retrieval.retrieve_scene_sparse` | `scene_shortlist_width`, `scene_shortcut_margin` (0.015), `scene_neighbor_window` (30), `scene_crossscene_*` | **Yes — actual production L2 entry point** given default `graph_mode="scene_sparse"` |
| L2 tiered ANN index (HNSW/PQ) | `iris/l2_index.py::L2TieredIndex` | `l2_hnsw_m`, `l2_hnsw_ef_search`, `l2_pq_*`, `l2_embed_dim` | Built at every `enrich_nodes_bulk` call, but only *consulted* by the non-default `ranking_mode="legacy"` path — dead weight (cost paid, unused) under default `ranking_mode="ppr"` |
| CLIP peak reranking / predicted span / span half-width / boundary clamping | — | — | **DOES NOT EXIST** — see correction above |

## LAYER 3 — `iris/l1_elysium.py`, `iris/aria.py`, `iris/cerberus_v.py`, `iris/cerberus_layers.py`, `iris/query.py`

| Responsibility | File / Function | Config | Production? |
|---|---|---|---|
| Frame retrieval for captioning (lazy) | `query.py::_ensure_captions` | n/a | Yes; seeks nearest keyframe, decodes forward within GOP only |
| Captioner backend | `aria.py`: `MiniCPMCaptioner` (Ollama), `MoondreamCaptioner` (HF), `BLIPCaptioner` (HF) | `captioner_backend` (dataclass default `"moondream"`; default JSON config does not set it) | Yes; MiniCPM discards truncated (`done_reason=="length"`) captions; all exceptions -> `CaptionResult(success=False, caption="[CAPTION_FAILED]")` |
| Caption ordering | `L1ElysiumCache.as_context_text` | n/a | admission (insertion) order, joined with `"\n\n---\n\n"` |
| L1 context cache | `iris/l1_elysium.py::L1ElysiumCache` | `l1_capacity`(64), `l1_w_*` weights (sum to 1.0), `l1_hessian_saturation_scale` | Used as actual retrieval mechanism only if `config.use_l1=True` (default `False`); otherwise used purely as the context-text formatter for ARIA in the legacy path |
| Answerer backend | `aria.py`: `LlamaServerBackend` (default, `granite4:micro` @ `127.0.0.1:8091/v1`), `LlamaBackend` (Ollama), `OpenAIBackend` | `answerer_backend`(`llama_server`), `answerer_endpoint`, `answerer_model`, `answerer_schema_format`(True), `answerer_max_tokens`, `answerer_timeout` | Yes; `OpenAIBackend` raises without `OPENAI_API_KEY`; `LlamaServerBackend` falls back OpenAI-compat -> raw `/completion` |
| Answer parsing / claim contract | `iris/claim_contract.py::AnswerClaims`, `query.py::_generate_answer_claims_v2[_wire]` | `cerberus_mode` (default `"legacy"`), `answerer_schema_format` | v2 path fully wired but **not default**; exactly one core claim enforced by `__post_init__`; one corrective retry on parse failure, then `compliance_failed=True`/`badge="unverified"` (no fallback to legacy) |
| Evidence citation (v2) | `iris/cerberus_layers.py::verify_answer` | n/a | v2-only; claim citing a frame outside `retrieved_frames` -> `"unverifiable"`, never fetches a new frame |
| Verification (legacy, DEFAULT) | `iris/cerberus_v.py::CerberusV.verify` + `query.py::wrapper_cerberus_gate` | `cerberus_high_thresh`(0.70), `cerberus_low_thresh`(0.35), `disable_nli` | Yes — default path; any exception -> all claims `unverifiable`, gate closed; `_ner_overlap` mode never verifies (only appends to `unverifiable`, by design per P1-20) |
| Confidence scoring (NLI) | `CerberusV._full_nli` / `cerberus_layers.score_nli_pair` (`cross-encoder/nli-deberta-v3-base`) | n/a | Forced to CPU on Windows to avoid a known CUDA access-violation segfault; negation-aware threshold flip; GPE/LOC geographic precision check |
| Abstention (legacy) | `query.py::query()` | n/a | `final_answer` = joined verified claims, else `"Insufficient verified evidence to answer this question."` |
| Abstention (v2) | `cerberus_layers._compute_badge` | n/a | badge in `{verified, partial, flagged, unverified}` per explicit rule table |

## Benchmark harnesses vs. production spine

- `benchmarks/t0_retrieval_integrity.py` — imports `iris.ingest.{ingest, load_index, save_index, _build_graph}` and `iris.query` directly. **Faithful to the real production call graph.**
- `benchmarks/paper_benchmark.py`, `benchmarks/exp1a_budget_matched.py` — use real L1/L2 primitive modules (`charon_v`, `action_score`, `L2Asphodel`) but drive retrieval through the **dead `iris.pipeline.wrapper_l2_retrieve` shim**, not `ingest.py`/`query.py`/`scene_retrieval.py`. Numbers from these scripts measure a legacy retrieval-wiring path, not what `api.py` serves today.
- `benchmarks/t0_reproduction.py` — reimplements its own top-to-bottom orchestration loop (own CLIP/BLIP loading) rather than calling `ingest()`/`query()` — a legacy/simplified reproduction harness, not a thin production wrapper.
- `benchmarks/exp0_compression_accuracy.py` — Layer 1 only (`charon_v`/`action_score`), no L2/L3.

**Rule for this benchmark setup going forward: only a harness that calls `iris.ingest.ingest()` then
`iris.query.query()` (i.e. the `t0_retrieval_integrity.py` pattern) qualifies as exercising the canonical
production path.** Command templates in `commands/` are written against this pattern.

## Open items this file creates for later sections

1. Peak-selection modes (`ppr_score_legacy`, `raw_clip_same_pool`, `clip_in_ppr_topk`) and span construction
   (Section 7 of the task) reference a stage that does not exist in code today. These are registered as
   **NOT_IMPLEMENTED** configurations requiring new code, not existing knobs — see `method_registry.json`
   and `configs/peak_span_modes.json`.
2. `cerberus_mode="v2"` (typed claims + verifier) vs `"legacy"` (default) is a real, already-implemented
   switch and is used as the basis for the L3-ON/L3-OFF factorial cells and the "no verifier"/"raw answerer"
   method-registry baselines.
3. `graph_mode="scene_sparse"` (default) vs `graph_mode="flat"` and `ranking_mode="ppr"` (default) vs
   `ranking_mode="legacy"` are real switches and are used as the basis for L2-ON/L2-OFF.
