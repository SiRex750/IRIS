# IRIS Source-Verified Bug Audit

Date: 2026-06-27

Repo inspected from zip: `/Users/apotdar/Downloads/IRIS (2).zip`

Extracted source tree: `/Users/apotdar/Documents/Codex/2026-06-27/v/work/IRIS_zip_repo/IRIS`

## Verification Notes

- I extracted the repo zip and inspected the backend, frontend, tests, configs, benchmark scripts, and logs.
- `python3 -m compileall -q .` passed after redirecting bytecode writes with `PYTHONPYCACHEPREFIX`.
- I could not run the pytest suite because both the system Python and bundled Python lack `pytest`; direct runtime imports also hit missing project deps such as `scipy`.
- The earlier docs-only audit is still available at `outputs/IRIS_bug_audit.md`. This file is the source-verified version.

## Priority Fix Order

1. Fix the fail-open Cerberus fallback in `pipeline.py`.
2. Fix PIL-image captioning failure between `pipeline.py` and `aria.py`.
3. Fix `action_score.py` hardcoded `max_prominence`.
4. Fix `_normalize()` rare-event percentile collapse.
5. Fix first-frame `residual_energy = 1.0`.
6. Add codec validation before Charon-V parses any file.
7. Remove the rabbit/meadow NLI fact filter and Big Buck Bunny zero-shot vocabulary bias.
8. Fix L1 keep-score normalization and recency.
9. Make L2 retrieval use graph structure/PageRank or remove the ceremonial graph work.
10. Fix API/frontend response-shape drift.
11. Replace paper/eval claims based on one cartoon clip with real annotated CCTV evaluation.

## Critical Correctness Bugs

### 1. Cerberus fails open and marks claims verified when verification crashes

- Source: `pipeline.py:469-475`
- Issue: `wrapper_cerberus_gate()` catches any exception from `CerberusV.verify()` and then sets `verified_claims = list(claims)`, `is_verified = True`, and `is_mocked = True`.
- Impact: any model loading failure, missing dependency, malformed cache object, CUDA/CPU issue, or NLI exception silently converts unverified claims into verified claims. This is the worst hallucination-control bug in the repo.
- Fix: fail closed. Put claims into `unverifiable_claims`, set `verified_claims = []`, set `is_verified = False`, and expose the error in telemetry. Only use a mock verifier in explicit test mode.

### 2. OpenAI vision captioning fails on the normal PIL-cache path

- Source: `pipeline.py:145-154`, `aria.py:167-195`
- Issue: `get_semantic_and_clip_caption()` passes `pil_img` into `aria.generate_caption_for_frame()`. But `generate_caption_for_frame()` assumes a PyAV frame and calls `frame.to_image()`. A PIL image does not have `to_image()`.
- Impact: the primary OpenAI caption path fails for the intended fast path where Charon-V cached PIL images. The code then falls back to BLIP or `[CAPTION_FAILED]`, causing poor captions and weak Cerberus evidence.
- Fix: update `aria.generate_caption_for_frame()` to accept either a PIL `Image.Image` or a PyAV frame. If the input already is a PIL image, use it directly instead of calling `.to_image()`.

### 3. `max_prominence` is hardcoded to `0.5`

- Source: `action_score.py:24`, `action_score.py:100-105`, `iris_config.py:43`, `pipeline.py:520-528`
- Issue: persistence is computed as `prominence / self.config.max_prominence`, and the default divisor is fixed at `0.5`.
- Impact: peak persistence is scaled by a global constant instead of the actual video. Real footage with a different prominence scale gets distorted peaks.
- Fix: derive the divisor from the current `find_peaks()` prominences:

```python
max_prominence = float(np.max(prominences)) if len(prominences) else 1.0
if max_prominence < 1e-8:
    max_prominence = 1.0
```

Then clamp each persistence value to `[0.0, 1.0]`.

### 4. Tests lock in the hardcoded prominence behavior

- Source: `tests/test_action_score.py:58-103`, especially `tests/test_action_score.py:65`
- Issue: tests explicitly describe and assert "Option B" behavior where `max_prominence=0.5` is used as the divisor.
- Impact: fixing the real peak bug will require changing tests that currently protect the broken behavior.
- Fix: replace those assertions with per-video max-prominence expectations and add a regression test where the strongest actual prominence is not `0.5`.

### 5. Percentile normalization can zero out rare events

- Source: `action_score.py:127-145`
- Issue: for arrays with at least 50 frames, `_normalize()` uses 2nd and 98th percentiles. If the percentile range collapses, it returns `np.zeros_like(values)`.
- Impact: a short action burst in a long static clip can disappear completely.
- Fix: if percentile range collapses, fall back to global min/max. If global min/max also collapses, return a neutral constant such as `0.5`, not all zeros.

### 6. First frame is treated as maximally active

- Source: `charon_v.py:164-171`, `charon_v.py:263-271`
- Issue: the first frame has no previous luma plane, but both decode passes set `residual_energy = 1.0`.
- Impact: frame 0 becomes a false maximum and contaminates thresholding, peak detection, tiering, and action scoring.
- Fix: set the first-frame energy to `0.0`, or exclude the first frame from threshold and peak calculations.

### 7. Charon-V decodes the video twice

- Source: `charon_v.py:147-183`, `charon_v.py:231-399`
- Issue: Pass 1 decodes all frames for luma energy. Pass 2 opens the container again, decodes all frames again, extracts motion vectors, computes luma differences again, and builds output records.
- Impact: duplicated decode and duplicated pixel arithmetic. This is still live in the zip.
- Fix: either collect all required data in one decode pass, or use the stronger Tier 0 design: demux coded packet sizes first, select candidate frames, then decode only survivors.

### 8. The project claims H.264 residual energy but computes decoded pixel difference

- Source: `README.md:3-12`, `charon_v.py:168-169`, `charon_v.py:268-269`, `frame_motion_descriptor.py:25-26`
- Issue: `residual_energy` is a luma frame-difference proxy, not H.264 transform residual coefficient energy.
- Impact: the central "codec-native residual" claim is scientifically overstated.
- Fix: rename the field to `luma_diff_energy` or implement a true compressed-domain signal. For a practical codec-native path, use packet coded size from demuxed packets as the Tier 0 saliency signal.

### 9. No codec validation exists before motion-vector extraction

- Source: `charon_v.py:139-148`, `charon_v.py:231-242`, `api.py:190-195`
- Issue: the parser accepts any video path and the API accepts `.mp4`, `.mov`, `.avi`, `.mkv`, and `.webm`, but the motion-vector logic assumes H.264 with exportable MVs.
- Impact: H.265, VP9, AV1, Xvid/MPEG-4 AVI, B-frame-heavy encodes, and bad GOP layouts can silently produce empty or misleading motion vectors.
- Fix: add `codec_validator.py` and validate codec, container, GOP, B-frames, and MV availability before parsing. Reject or re-encode incompatible files.

### 10. Motion-vector extraction silently degrades to empty data

- Source: `charon_v.py:313-331`
- Issue: MV extraction catches `AttributeError` and `TypeError` and silently sets `motion_vectors = []`.
- Impact: unsupported codecs and PyAV side-data failures look like low-motion videos rather than pipeline failures.
- Fix: track MV availability in stats, warn or raise when no MVs are found across sampled non-I frames, and route incompatible inputs through the codec validator.

## Layer 1 / Charon-V Bugs

### 11. `compute_motion_geometry()` loops over motion vectors in Python

- Source: `charon_v.py:23-33`
- Issue: every MV is processed in a Python loop with per-vector indexing and scalar updates.
- Impact: avoidable latency on MV-heavy frames.
- Fix: convert MVs to a NumPy array and use vectorized binning, for example `np.add.at` over grid indices.

### 12. Motion magnitude is computed with another Python loop

- Source: `charon_v.py:377-378`
- Issue: `mvs_mags = [np.sqrt(mv[4]**2 + mv[5]**2) for mv in motion_vectors]` repeats per-vector Python work for raw records.
- Impact: extra Layer 1 work just to satisfy the action-score interface.
- Fix: compute MV magnitudes once in vectorized form and share the result with both geometry and raw-record output.

### 13. `residual_energy` naming leaks into all downstream layers

- Source: `action_score.py:31-35`, `pipeline.py:350-356`, `l2_asphodel.py:28-33`, `api.py:103-106`, frontend `IrisApp.jsx:537-538`
- Issue: downstream code and UI present decoded luma-diff as residual energy.
- Impact: every layer reinforces the inaccurate claim.
- Fix: rename across the API boundary, graph node schema, frontend labels, and docs. Keep backward-compatible aliases only temporarily.

### 14. I-frame scene thresholds are not enough to make the signal codec-native

- Source: `charon_v.py:176-179`, `charon_v.py:185-229`
- Issue: I-frame indices are used for adaptive scene thresholds, but the saliency signal still comes from full decoded luma differences.
- Impact: the pipeline is not a zero-decode compressed-domain gate.
- Fix: use packet sizes and keyframe flags from demux for the initial saliency pass.

### 15. `motion_entropy` description is misleading

- Source: `frame_motion_descriptor.py:33`, `l2_asphodel.py:32`, `charon_v.py:74-82`
- Issue: the descriptor calls entropy "chaos / unpredictability of motion"; L2 calls it entropy of motion vectors/energy. In raw records, `entropy` is actually luma histogram entropy (`charon_v.py:380-382`).
- Impact: code readers cannot tell which entropy is being weighted.
- Fix: split names: `motion_vector_entropy` for MV-grid entropy and `luma_entropy` for image histogram entropy.

## Layer 2 / Action Score Bugs

### 16. Default action-score weights are backwards

- Source: `action_score.py:17-19`, `iris_config.py:37-39`, `pipeline.py:520-523`
- Issue: defaults put 0.5 on residual and 0.3 on motion, leaving only 0.2 for entropy.
- Impact: the score is dominated by signals the team notes say underperform, while the strongest signal is underweighted.
- Fix: change defaults toward luma entropy, for example residual `0.2`, motion `0.2`, luma entropy `0.6`, then tune on dev clips.

### 17. `entropy` input is actually luma histogram entropy

- Source: `charon_v.py:380-382`, `action_score.py:64-71`
- Issue: `action_score.py` consumes `entropy`, but the source is the decoded Y-plane histogram, not motion complexity.
- Impact: the formula works on a mislabeled feature.
- Fix: rename the input to `luma_entropy` and update tests, configs, docs, and UI labels.

### 18. `score_all()` is full-video batch only

- Source: `action_score.py:50-125`
- Issue: it expects all frame features at once and computes global normalization and peaks over the full sequence.
- Impact: it cannot support true streaming and creates memory pressure on long videos.
- Fix: add chunked or online scoring, or explicitly make this an offline-only scorer.

### 19. Peak detection is duplicated between Charon-V and ActionScore

- Source: `charon_v.py:93-137`, `action_score.py:90-123`, `pipeline.py:518-559`
- Issue: Charon-V produces legacy tiers and peaks, then action scoring recomputes peak flags and overwrites frame `is_peak`.
- Impact: two peak definitions can disagree, and debugging selected frames becomes harder.
- Fix: make action-score peaks the only downstream peak source, or clearly separate Charon gating peaks from ActionScore retrieval peaks.

### 20. NMS suppressions are not exposed even though the frontend expects them

- Source: `pipeline.py:533-547`, frontend `IrisApp.jsx:777-779`, `IrisApp.jsx:852-858`
- Issue: pipeline sets `is_peak = False` for suppressed peaks but does not record `nms_suppressed` or `nms_parent`.
- Impact: frontend's "Suppressed Local Peaks" panel always has no real data.
- Fix: store suppression metadata in the debug payload.

## L1 Cache Bugs

### 21. L1 keep-score uses raw unnormalized entropy and hessian

- Source: `frame_motion_descriptor.py:18-19`, `l1_elysium.py:130-140`, `cached_frame.py:88-95`
- Issue: the descriptor promises normalization happens inside L1, but L1 passes raw values directly into `CachedFrame.keep_score()`.
- Impact: `motion_entropy` and `hessian_max_eigenvalue` can dominate the weighted score.
- Fix: normalize these fields in `_keep_score()` before calling `frame.keep_score()`, or add normalized override parameters.

### 22. Recency off-by-one makes the newest frame less than fully recent

- Source: `l1_elysium.py:64-66`, `cached_frame.py:84-86`
- Issue: `admitted_at` is assigned before `_admission_counter` increments, but recency divides by `total_admitted`.
- Impact: the first admitted frame gets recency 0 when it should be 1, and the newest frame in a small cache never reaches 1.
- Fix: compute recency against `max(total_admitted - 1, 1)` or assign `admitted_at` after incrementing with matching math.

### 23. L2 PageRank does not reach L1 cache entries

- Source: `pipeline.py:410-421`, `pipeline.py:233-265`, `cached_frame.py:53-54`
- Issue: retrieved dicts include `pagerank_score`, but `wrapper_populate_cache()` never assigns it to `CachedFrame.pagerank`.
- Impact: L1 `w_pagerank` is dead in the real pipeline.
- Fix: set `cached_frame.pagerank = frame.get("pagerank_score", 0.0)` or call `cache_obj.update_pagerank()` after population.

### 24. L1 query similarity is dead in the real pipeline

- Source: `l1_elysium.py:68-109`, `pipeline.py:606-615`
- Issue: pipeline populates the cache and immediately calls `as_context_text()`; it never calls `cache_obj.query()`.
- Impact: `query_similarity` stays 0 and `w_query` never contributes to the real context selection.
- Fix: after populating L1, run `cache_obj.query(query_embedding, top_k=...)` or remove query similarity from active scoring.

### 25. Eviction does not run in the default pipeline

- Source: `iris_config.py:46`, `iris_config.py:54`, `pipeline.py:297`, `pipeline.py:606-609`
- Issue: `l2_retrieve_top_k` defaults to 5 while L1 capacity defaults to 64, so the pipeline admits far fewer frames than capacity.
- Impact: the seven-signal eviction policy is almost never exercised in real runs.
- Fix: either lower L1 capacity for experiments, admit more evidence, or stop claiming active eviction behavior for the current pipeline.

### 26. Cache query implementation is linear Python iteration

- Source: `l1_elysium.py:76-107`
- Issue: similarity is computed one frame at a time.
- Impact: fine for tiny caches, but not scalable.
- Fix: stack embeddings into a matrix and compute all cosine similarities in one vectorized operation.

### 27. Rabbit/meadow filter drops legitimate facts

- Source: `l1_elysium.py:230-232`
- Issue: `_frame_to_nli_fact()` rejects captions containing `"rabbit"` or `"meadow"`.
- Impact: real-world nature footage loses evidence, and Big Buck Bunny hacks leak into general logic.
- Fix: remove content-specific filters. Only reject empty captions and explicit `[CAPTION_FAILED]`.

## L2 Graph Bugs

### 28. Graph is rebuilt for every pipeline retrieval

- Source: `pipeline.py:288-329`
- Issue: `wrapper_l2_retrieve()` creates `graph = L2Asphodel(config=config)` for each call.
- Impact: the graph has no persistent lifecycle and cannot amortize indexing over multiple queries for the same video.
- Fix: build and persist a graph per video at ingestion time, then query that stored graph.

### 29. Graph is flat, not hierarchical

- Source: `l2_asphodel.py:20-48`, `l2_asphodel.py:272-327`
- Issue: all frames are inserted as the same node type. There is no L1 peak root, L2 salient child, L3 candidate leaf hierarchy.
- Impact: the claimed HADES-style coarse-to-fine graph traversal does not exist.
- Fix: add tiered node types and parent-child edges: peak roots, salient children, candidate leaves.

### 30. Cross-encoder semantic edges are missing

- Source: `l2_asphodel.py:92-148`
- Issue: edge weights are based on action-score similarity and optional CLIP embedding cosine. There is no ingestion-time cross-encoder edge scoring between captions.
- Impact: graph traversal cannot use the mentor-described semantic parent/neighbor weights.
- Fix: run a small cross-encoder after captioning to assign parent-child and sibling edge weights.

### 31. Dense graph construction is O(N^2)

- Source: `l2_asphodel.py:103-148`
- Issue: `_update_all_edge_weights()` fully connects every pair of nodes.
- Impact: long videos scale poorly.
- Fix: connect only temporal windows, hierarchy edges, and semantic nearest neighbors.

### 32. Public single-node insert/enrich paths are O(N^3)-prone

- Source: `l2_asphodel.py:189-270`
- Issue: `add_frame_node()` and `enrich_node()` recompute all edges and PageRank every call.
- Impact: the pipeline currently uses bulk methods, but tests and public API still encourage the slow path.
- Fix: make these methods mark the graph dirty and defer recompute until retrieval, or route callers to bulk methods.

### 33. Retrieval ignores graph edges and PageRank

- Source: `l2_asphodel.py:350-413`
- Issue: `retrieve()` scores each node using semantic similarity, action score, and persistence. It does not inspect edges, neighbors, graph traversal, or `pagerank_score`.
- Impact: the expensive graph is mostly ceremonial. Cold-start retrieval is just action-score sorting.
- Fix: use persistent graph traversal and include edge/PageRank terms only if they improve measured retrieval quality.

### 34. `query_action_score` parameter is unused

- Source: `l2_asphodel.py:350-395`
- Issue: `retrieve()` accepts `query_action_score` but never uses it.
- Impact: the function signature and docs imply motion-coherence query matching, but implementation ignores the query action target.
- Fix: either remove the parameter or use it to compute a motion score such as `1 - abs(node.action_score - query_action_score) / range`.

### 35. `beta` means different things in edge building and retrieval

- Source: `l2_asphodel.py:78-86`, `l2_asphodel.py:143`, `l2_asphodel.py:392-395`
- Issue: in edge weights, `beta` weights motion coherence. In retrieval, `beta` weights raw `node.action_score`.
- Impact: configuration semantics are inconsistent.
- Fix: split into explicit `edge_motion_weight`, `retrieval_action_weight`, and `retrieval_persistence_weight`.

### 36. Negative cosine similarities can create negative graph weights

- Source: `l2_asphodel.py:136-143`
- Issue: cosine similarity can be negative. The code combines it directly into NetworkX edge weights.
- Impact: PageRank with negative weights is mathematically invalid or unstable.
- Fix: clamp cosine to `[0, 1]`, shift from `[-1, 1]` to `[0, 1]`, or discard negative semantic edges.

### 37. Retrieval debug prints spam stdout

- Source: `l2_asphodel.py:397-401`
- Issue: every node prints retrieval-score contributions on every query.
- Impact: noisy logs and avoidable latency.
- Fix: remove prints or route through a disabled logger/debug flag.

### 38. `refined_motion_tensor` is a fake placeholder

- Source: `l2_asphodel.py:33`, `l2_asphodel.py:45`, `l2_asphodel.py:220-221`, `l2_asphodel.py:304-306`, `pipeline.py:356`, `pipeline.py:385`, `profiler.py:151`
- Issue: docs describe a bfloat16 RMR tensor, but pipeline and graph fill it with `np.zeros(1, dtype=np.float32)`.
- Impact: graph nodes advertise a feature that is not implemented.
- Fix: remove it or make it `None` with a clear "RMR not implemented" note.

### 39. CSR export is not used by the pipeline

- Source: `l2_asphodel.py:415-495`, repo search only finds tests and demo usage.
- Issue: `export_to_csr()` is implemented but not consumed by retrieval, API, or pipeline.
- Impact: more code surface without runtime value.
- Fix: wire it into an actual vector-search/export path or move it out of the hot graph class.

## Pipeline / ARIA / Cerberus Bugs

### 40. Big Buck Bunny vocabulary is hardcoded into zero-shot captions

- Source: `pipeline.py:98-108`
- Issue: CLIP zero-shot vocabulary includes rabbit/meadow/cartoon labels specific to the test clip.
- Impact: real videos are biased by test-video labels, and the downstream rabbit/meadow filter exists to suppress this artifact.
- Fix: replace with domain-neutral labels or load vocabulary from config/dataset domain.

### 41. `filtered_nli` verifies low-confidence claims without checking them

- Source: `cerberus_v.py:160-167`
- Issue: in filtered mode, low-confidence claims are appended to `verified`.
- Impact: hedged or short claims can bypass NLI and still count as verified.
- Fix: put low-confidence claims through NER-only verification or mark them unverifiable.

### 42. Final answer falls back to raw unverified answer when no claims verify

- Source: `pipeline.py:628-630`
- Issue: comment says final answer is "verified claims only", but code uses `raw_answer` if `verified_claims` is empty.
- Impact: if all claims are rejected or unverifiable, the user still sees the original unverified answer as final.
- Fix: if there are no verified claims, return an explicit insufficient-evidence answer and expose raw answer only in debug fields.

### 43. spaCy model download happens at runtime

- Source: `cerberus_v.py:34-44`
- Issue: if `en_core_web_sm` is missing, code calls `spacy.cli.download()`.
- Impact: production/offline runs can hang or fail on network; tests become environment-dependent.
- Fix: declare the model as an installation requirement or use a deterministic fallback.

### 44. Pipeline logs are written relative to the process CWD

- Source: `pipeline.py:661-725`
- Issue: `logs/aria_debug` and `logs/pipeline` are relative paths.
- Impact: API runs launched from another directory write logs into unexpected places.
- Fix: anchor logs to `Path(__file__).parent / "logs"` or a configured output directory.

### 45. Logs containing prompts/responses are shipped in the zip

- Source: files under `logs/aria_debug/` and `logs/pipeline/`
- Issue: runtime logs with prompts, captions, and responses are included in the repo zip.
- Impact: privacy and reproducibility risk.
- Fix: remove logs from shared artifacts and add `logs/` to `.gitignore`.

### 46. `.env` is present in the zip

- Source: `.gitignore:9` ignores `.env`, but `/IRIS/.env` exists in the zip.
- Issue: environment files should not be distributed with repo zips.
- Impact: possible credential leakage.
- Fix: remove `.env` from shared artifacts and rotate any real key that may have been exposed.

## API Bugs

### 47. Upload endpoint reads the entire video into memory

- Source: `api.py:203-209`
- Issue: `content = await file.read()` loads the whole upload before writing to disk.
- Impact: large videos can exhaust memory.
- Fix: stream chunks to the temp file and enforce upload size limits.

### 48. API accepts formats the backend cannot safely process

- Source: `api.py:190-195`
- Issue: API accepts `.mov`, `.avi`, `.mkv`, and `.webm`, while Charon-V assumes H.264 motion-vector availability.
- Impact: users can submit videos that produce empty/corrupt codec features.
- Fix: accept uploads, but run codec validation and re-encode or reject before pipeline execution.

### 49. API graph payload fabricates edges instead of returning L2 graph edges

- Source: `api.py:117-151`
- Issue: `_build_graph_data()` creates temporal neighbor edges and high-coherence cross-links from retrieved frames. It does not serialize the real `L2Asphodel.graph`.
- Impact: frontend "L2 Asphodel graph" visualization is not the actual graph.
- Fix: return graph edges from `L2Asphodel` or label this as a frontend-only visualization.

### 50. API always shows PageRank as zero in graph nodes

- Source: `api.py:93-115`, `pipeline.py:745-779`
- Issue: `_build_graph_data()` reads `pagerank_score` from `debug_info["action_scores"]`, but those records do not contain PageRank. The retrieved frame dicts do contain `pagerank_score`.
- Impact: graph visualization loses actual PageRank values.
- Fix: read `pagerank_score` directly from each retrieved frame.

### 51. API error responses expose raw exception messages

- Source: `api.py:227-233`
- Issue: the 500 response includes `Pipeline execution failed: {exc}`.
- Impact: internal paths/model errors can leak to the UI.
- Fix: return a generic error to users and put detailed traces only in server logs.

## Frontend Bugs

### 52. Frontend expects `debug_info.all_frames`, but backend never sends it

- Source: frontend `IrisApp.jsx:777`, `IrisApp.jsx:1002`, `IrisApp.jsx:1184`; backend `pipeline.py:745-779`
- Issue: frontend timelines, frame inspector, extracted frames grid, NMS panel, and thumbnails depend on `debug_info.all_frames`. Pipeline debug info has `action_scores` and `retrieved_frames`, not `all_frames`.
- Impact: large parts of the UI are empty or partially broken after a successful API call.
- Fix: either add `all_frames` to pipeline debug output or rewrite the UI to use available fields.

### 53. Frontend expects thumbnails and labels that backend never sends

- Source: `IrisApp.jsx:510-520`, `IrisApp.jsx:653-655`, `IrisApp.jsx:736-745`; backend retrieved frame dict at `pipeline.py:410-423`
- Issue: frontend uses `thumbnail`, `label`, `frame_type`, `motion_magnitude`, `reasons`, and NMS metadata, but pipeline does not return these in debug all-frame data.
- Impact: "No Image", undefined labels, wrong filters, and empty triage explanations.
- Fix: define a stable frontend DTO and populate it from backend.

### 54. Frontend labels non-skip frames as total video frames

- Source: `IrisApp.jsx:1126`, backend `pipeline.py:645`
- Issue: UI label says "Total Video Frames", but `frames_processed` is `len(output_frames)`, meaning non-SKIP frames.
- Impact: misleading statistics.
- Fix: show both `stats["total"]` and non-SKIP count, or rename the label.

### 55. Cerberus debug mode is not passed to frontend

- Source: `IrisApp.jsx:945`, backend `pipeline.py:763-770`
- Issue: UI displays `debugInfo.cerberus_result.mode`, but backend does not include `mode`.
- Impact: UI defaults to `full_nli`, which can be false.
- Fix: return Cerberus mode from `wrapper_cerberus_gate()` and include it in `debug_info`.

### 56. Knowledge graph animation keeps repainting forever

- Source: `IrisApp.jsx:98-143`
- Issue: after `MAX_FRAMES`, the simulation stops updating physics but still schedules `requestAnimationFrame(simulate)` forever and redraws continuously.
- Impact: unnecessary browser CPU usage.
- Fix: stop the animation after layout settles, and redraw only on hover/resize/data change.

## Evaluation / Benchmark Bugs

### 57. Evaluation does not measure frame-selection accuracy against ground truth

- Source: `eval_suite.py:72-110`, `benchmark_iris_vs_baseline.py:8-12`, `benchmark_results/benchmark_report.json:27-30`
- Issue: eval reports frames processed, skip ratio, latency, and claim counts. It does not compute precision, recall, F1, or event localization against annotated frame intervals.
- Impact: the current numbers cannot prove better event selection.
- Fix: use frame-indexed annotations and report P/R/F1 at matched frame budgets against uniform and random baselines.

### 58. "Baseline" ablation still runs much of the IRIS pipeline

- Source: `eval_suite.py:30-67`, `eval_suite.py:72-78`
- Issue: baseline config disables NLI and sets `candidate_thresh=0.0`, but still calls `run_pipeline()`, which runs Charon-V, action scoring, L2 retrieval, L1, and ARIA.
- Impact: baseline is not a true non-IRIS baseline.
- Fix: implement separate baseline runners: uniform sampling, random sampling, and no-graph/no-NLI variants.

### 59. Benchmark is based on a single cartoon clip

- Source: `benchmark_results/benchmark_report.json:30`, tests use `mov_bbb.mp4`
- Issue: the included benchmark caveat admits it is one Big Buck Bunny clip.
- Impact: results do not support CCTV/video-retrieval claims.
- Fix: build a VIRAT-style CCTV dev/test set with manifest and annotations.

### 60. Benchmark compares frame count, not answer quality

- Source: `benchmark_iris_vs_baseline.py:8-12`, `benchmark_iris_vs_baseline.py:175-184`
- Issue: the benchmark explicitly does not compare accuracy or answer quality.
- Impact: it is useful for cost framing, not for retrieval correctness claims.
- Fix: add accuracy metrics and error analysis on ground-truth queries/events.

### 61. Default config JSON is stale

- Source: `configs/default_iris_config.json:1-7`, `iris_config.py:16-65`
- Issue: JSON config only contains a few early fields and uses `beta: 0.6`, while `IRISConfig` now has many more fields and defaults `beta` to `0.3`.
- Impact: config files and runtime defaults can drift silently because unknown/missing keys are ignored.
- Fix: regenerate default config from `IRISConfig`, include all fields, and add a config-version check.

## Scripts / Repo Hygiene Bugs

### 62. `extract_frames.py` has import-time side effects and deletes the test video

- Source: `extract_frames.py:20-24`, `extract_frames.py:83-89`
- Issue: downloading, parsing, writing JSON, and deleting `mov_bbb.mp4` happen at module top level.
- Impact: importing this file runs network and file deletion side effects.
- Fix: move execution into `main()` under `if __name__ == "__main__":`, and do not delete a repo-local video by default.

### 63. Requirements depend on a live GitHub install for CLIP

- Source: `requirements.txt:12`
- Issue: `clip @ git+https://github.com/openai/CLIP.git` requires network and GitHub access during install.
- Impact: offline or restricted installs fail.
- Fix: vendor a pinned wheel, document this as an optional extra, or switch to a package available from the configured package index.

### 64. Tests and scripts still download public videos

- Source: `test_charon_v.py:7-15`, `test_charon_v.py:45-49`, `test_track_b.py:65-80`, `benchmark_iris_vs_baseline.py:40-47`
- Issue: several paths attempt to download `mov_bbb.mp4` if local files are absent.
- Impact: tests are network-dependent unless the local clip is present.
- Fix: keep a tiny deterministic fixture or skip network tests by default.

## Previously Reported Items That Are Not Live As Written

### A. Pipeline per-frame graph insert is not live in this zip

- Source: `pipeline.py:337-396`
- Status: not live in current pipeline.
- Reason: `wrapper_l2_retrieve()` collects `feature_records` and `score_records`, then calls `graph.add_frame_nodes_bulk()` and `graph.enrich_nodes_bulk()`.
- Remaining issue: `add_frame_node()` and `enrich_node()` still expose slow recompute behavior.

### B. SKIP-frame PIL conversion is not live in `charon_v.py`

- Source: `charon_v.py:341-360`
- Status: not live as originally stated.
- Reason: `frame.to_image()` is called only inside `if tier != "SKIP"`.
- Remaining issue: double decode still exists.

### C. Cerberus has a CUDA path in this zip

- Source: `cerberus_v.py:46-56`, `cerberus_v.py:277-279`
- Status: older "no CUDA path" claim is not live as written.
- Reason: model moves to CUDA if `torch.cuda.is_available()` and inputs are moved to `model.device`.
- Remaining issue: no MPS path, runtime model downloads, broad fail-open wrapper.

### D. API/frontend `App.jsx` Vite starter is unused

- Source: `iris-frontend/src/main.jsx:1-10`, `iris-frontend/src/App.jsx:1-122`
- Status: not a runtime bug.
- Reason: `main.jsx` renders `IrisApp`, not the default `App`.
- Cleanup: remove unused starter files/assets if desired.

