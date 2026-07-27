# VIRAT N=4,892 latency A/B: flat vs scene_sparse — RESULT

Run date: 2026-07-27 (artifact mtime of `virat_latency_result.json` / stdout log in the
source scratchpad — `2026-07-27 00:13` local time). This document was written on promotion
of pre-existing artifacts; the harness was not re-run.

Source clip: `VIRAT_S_040001_01_000448_001101.mp4`, single video, single scene structure
(528 scenes). Index cache: `eval/data/virat/index_cache/VIRAT_S_040001_01_000448_001101.npz`,
already present in the repo tree prior to this promotion.

## Headline numbers

- N = 4,892 survivor frames (`n_survivors`)
- 50 seeded synthetic queries (`n_queries`), `query_seed = 20260726`
- **Median total retrieval latency**: flat **7.9448 s** vs scene_sparse **0.0072 s**
  (`median_total_retrieval_s.flat = 7.944804999999178`,
  `median_total_retrieval_s.scene_sparse = 0.0071983500092756`)
- scene_sparse branch fire rate: **descend 50 / shortcut 0** out of 50 queries
  (`scene_sparse_branch_fire_rate`: `descend: 50`, `shortcut: 0`, `shortcut_pct: 0.0`)
  — `median_total_retrieval_s.scene_sparse_shortcut` is `null` because the shortcut branch
  never fired; the reported scene_sparse median is entirely descend-branch.
- One-time setup costs, excluded from per-query timing:
  `scene_sparse_load_s = 69.27`, `flat_build_s = 80.14`

## Component breakdown (median seconds per query)

| component | flat | scene_sparse |
|---|---|---|
| query_embed_s | null (n/a, see caveats) | null (n/a, see caveats) |
| total_retrieval_s | 7.944804999999178 | 0.0071983500092756 |
| flat_ppr_s | 7.944533850008156 | null |
| scene_centroid_rank_s | null | 0.0007050499989418313 |
| subgraph_induction_s | null | 0.0029290499951457605 |
| cross_scene_edges_s | null | 0.0011674499983200803 |
| scene_ppr_s | null | 0.001905950004584156 |

PPR graph size (median nodes/edges per query):
- flat: `ppr_nodes = 4892.0`, `ppr_edges = 11963386.0`
- scene_sparse: `ppr_nodes = 214.5`, `ppr_edges = 1264.0`

## Cache-untouched assertions (from the run's own fingerprinting)

- `nextqa_cache_untouched: true` (fingerprint before/after both
  `3a25c9038eed0d6ea708d4c7aa0517c303b084ec7ee1e7d7d8fbef043d097f2c`)
- `virat_cache_untouched: true` (fingerprint before/after-setup/after all
  `b4ae9c096bfb87c58e305859f7e1c8f324fa6f2b0d0bffe24a61ae3ed889ab86`)
- `query_count_matches_across_modes: true` (50 == 50)

## Ingest / smoke-test context (companion artifacts)

From `virat_smoke_N4892_scenesparse.json` (scene_sparse ingest):
- `num_survivors: 4892`, `num_scenes: 528`, `unassigned_scene_id_count: 0`
- `ingest_wall_sec: 773.8998126983643`, `save_wall_sec: 0.4075031280517578`
- `scene_sparse_node_count: 4892`, `scene_sparse_edge_count: 23571`,
  `block_diagonal_exact: true`
- `skipped_frames_ratio: 0.7499872233863137`, `storage_reduction_factor: 3.999795584627964`

From `virat_smoke_N4892_flat.json` (in-memory flat graph build):
- `num_survivors: 4892`, `flat_node_count: 4892`, `flat_edge_count: 11963386`,
  `edge_count_matches_theoretical: true`, `flat_build_wall_sec: 78.82261204719543`

## Known caveats (from the harness docstring — not hidden)

1. **Queries are seeded synthetic CLIP-embedding samples, not text queries.**
   `query_embed_s` is `null` by design across both arms: the 50 queries were drawn by
   sampling and L2-normalizing 50 survivor CLIP embeddings (`np.random.default_rng(20260726)`),
   not produced via `_embed_query` text encoding. This measures retrieval **graph mechanics**
   (PPR cost, subgraph induction, scene routing) at N=4,892, not real end-to-end
   text-query latency. No query-embedding cost is included in either arm's total.

2. **The flat graph was built in-memory from the same loaded scene_sparse frames**,
   via `iris_ingest._build_graph(idx_scene_sparse.frames, flat_cfg_dict)`, not from a
   separate flat ingest run. This avoids a second ~13-minute ingest pass but means the
   flat arm did not go through its own independent ingest path. Flagged here as
   something to confirm does not advantage either mode before treating this as a clean
   apples-to-apples comparison — it has not been independently confirmed.

3. **Single video, single scene structure (528 scenes).** This is one data point, not a
   scaling curve. It extends the earlier VAL-scale result (`P_latency_result.md`, videos
   up to 599 frames) to CCTV length (4,892 frames) but is still only one clip.

4. **Codec: mpeg4, not H.264/HEVC.** Confirmed via ingest log
   (`[IRIS codec warn] ...: codec 'mpeg4' is not h264/hevc; motion-vector export may be
   unavailable`, `scenesparse_stdout.log` line 1). Static-camera VIRAT source, as stated
   in the task context for this run.

## Source artifacts (this promotion)

- `scripts/virat_latency_ab.py` — A/B harness (flat vs scene_sparse), reuses
  `scripts/latency_ab.py`'s timing constants/helpers.
- `eval_results/virat_latency_N4892_raw.json` — full result JSON incl. per-query records.
- `eval_results/virat_latency_N4892_stdout.log` — harness stdout.
- `scripts/virat_smoke_flat.py`, `scripts/virat_smoke_scenesparse.py` — smoke-test scripts
  used to build/verify the ingest and in-memory flat graph prior to the A/B run.
- `eval_results/virat_smoke_N4892_flat.json`, `eval_results/virat_smoke_N4892_scenesparse.json`
  — smoke-test result JSONs.
- `eval/data/virat/index_cache/VIRAT_S_040001_01_000448_001101.npz` — pre-existing index
  cache (4,892 `emb_*` keys, verified to match `n_survivors` on promotion).
