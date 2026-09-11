# Tiered (`hierarchical_sparse`) degree distribution — Arson042 (Appendix C item 37 follow-up)

One clip, built from ingest (no captioner/answerer/llama-server). `Arson042` was chosen from the `fully_connected` corpus in `eval/data/ucf/index_cache/` because N=613 sits in the 400-1500 range asked for and its existing cache already has real scene structure (110 scenes, max scene size 34, mean scene size 5.57) rather than being dominated by singleton scenes.

Re-ingested by loading the cached frame records + CLIP embeddings straight out of `eval\data\ucf\index_cache\Arson042.npz` (no video re-decode) and rebuilding the graph via the same `iris.ingest._build_graph` call sequence (`add_frame_nodes_bulk(defer_recompute=True)` then `enrich_nodes_bulk`), with `config_snapshot` copied verbatim from the existing cache except `graph_edge_mode` changed from `"fully_connected"` to `"hierarchical_sparse"`. Confirmed unchanged from the source cache: `graph_mode="scene_sparse"`, `alpha=0.4`, `beta=0.3`, `luma_diff_weight=0.5`, `motion_weight=0.3`, `luma_entropy_weight=0.2`, `graph_temporal_window=1`, `graph_semantic_top_k=4`, `graph_motion_top_k=2`, `graph_semantic_threshold=0.5`, `salient_thresh=0.35`, `candidate_thresh=0.08`. scene_id assignment reused as-is from the cached frames (same 110-scene partition the fully_connected cache used), so the two caches are comparable on identical scene boundaries. New cache written to `eval\data\ucf\index_cache\Arson042_hierarchical_sparse.npz`; `eval\data\ucf\index_cache\Arson042.npz` untouched.

## Edge counts

- `fully_connected` (existing cache): 3855 edges

- `hierarchical_sparse` (this run): 810 edges


By edge type (hierarchical_sparse):

| edge_type | count |
|---|---:|
| semantic_salient | 309 |
| hierarchy_peak_salient | 260 |
| temporal | 199 |
| motion_neighbor | 42 |

## Per-node total (undirected) degree, by edge type

| edge_type | n_edges | min | median | mean | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| **all types combined** | 810 | 0 | 2 | 2.64 | 5 | 10 | 17 |
| semantic_salient | 309 | 1 | 2 | 2.40 | 5 | 8 | 14 |
| hierarchy_peak_salient | 260 | 1 | 1.0 | 1.26 | 2 | 3 | 4 |
| temporal | 199 | 1 | 1 | 1.22 | 2 | 2 | 2 |
| motion_neighbor | 42 | 1 | 1.0 | 1.62 | 2 | 7 | 7 |

(Per-type stats above are computed only over nodes that have at least one edge of that type -- zero-degree-for-that-type nodes excluded, since most nodes are not `L1_PEAK`/`L2_SALIENT` and so structurally cannot receive every edge family.)

## Directed in/out degree (selector = source, per code semantics)

`semantic_salient` and `motion_neighbor` cap **out**-degree per source (top-4, top-2); `hierarchy_*` caps out-degree per child at 1 (one nearest parent); `temporal` is symmetric by construction (window=1, each node links its immediate successor). None of these cap **in**-degree in code. Direction below is recovered from the actual call order that won each edge in the undirected graph (see script), not inferred.

| edge_type | out-degree min/median/mean/p90/p99/max | in-degree min/median/mean/p90/p99/max |
|---|---|---|
| **all types combined** | 1/1/1.77/3/5/7 | 1/1.0/1.87/3/7/12 |
| semantic_salient | 1/1.0/1.66/3/4/4 | 1/1/1.81/3/7/11 |
| hierarchy_peak_salient | 1/2.0/1.71/3/3/4 | 1/1.0/1.00/1/1/1 |
| temporal | 1/1/1.00/1/1/1 | 1/1/1.00/1/1/1 |
| motion_neighbor | 1/1/1.35/2/2/2 | 1/1.0/1.50/2/6/6 |

## Does in-degree concentrate? Highest-in-degree nodes

Top 10 nodes by total in-degree (all edge types), with edge-type composition of their in-edges and their scene's size:

| frame_idx | in-degree | composition | scene_id | scene_size |
|---:|---:|---|---:|---:|
| 5616 | 12 | semantic_salient:11, temporal:1 | 110 | 34 |
| 5349 | 10 | semantic_salient:7, motion_neighbor:2, hierarchy_peak_salient:1 | 109 | 25 |
| 5518 | 10 | semantic_salient:8, temporal:1, hierarchy_peak_salient:1 | 110 | 34 |
| 5516 | 7 | semantic_salient:5, temporal:1, hierarchy_peak_salient:1 | 110 | 34 |
| 5599 | 7 | motion_neighbor:6, temporal:1 | 110 | 34 |
| 5721 | 7 | semantic_salient:6, hierarchy_peak_salient:1 | 110 | 34 |
| 1227 | 6 | semantic_salient:5, hierarchy_peak_salient:1 | 18 | 32 |
| 5348 | 6 | semantic_salient:5, hierarchy_peak_salient:1 | 109 | 25 |
| 5354 | 6 | semantic_salient:5, hierarchy_peak_salient:1 | 109 | 25 |
| 5355 | 6 | semantic_salient:3, motion_neighbor:2, hierarchy_peak_salient:1 | 109 | 25 |

Max in-degree observed: **12** (vs max out-degree **7**, which the top-4/top-2 per-source caps plus the single-parent hierarchy cap keep low by construction). In-degree exceeds the largest single per-source cap (semantic_salient top_k=4) by 8, so it cannot be explained by any one node's own top-k selection -- multiple independent sources are picking the same target. Concentration is real on this clip, though bounded (well under the fully_connected scene-size ceiling, see below).

## Comparison against this clip's `fully_connected` cache

| | edges | min | median | mean | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| fully_connected | 3855 | 0 | 8 | 12.58 | 31 | 33 | 33 |
| hierarchical_sparse | 810 | 0 | 2 | 2.64 | 5 | 10 | 17 |

Scene-size bound: max scene size on this clip is **34** (scene 110). `fully_connected` max degree (33) is exactly `max_scene_size - 1` for that scene, by construction (§4.1: cross-scene edges pruned, so a scene is a complete subgraph). `hierarchical_sparse` max degree (17) is also ≤ max_scene_size-1 (33) on this clip, so the same numeric ceiling holds here too, but the *mechanism* is different: fully_connected's bound comes from cross-scene pruning of an otherwise-complete per-scene subgraph (every node already has degree = scene_size-1 before pruning even applies); hierarchical_sparse's bound is never approached by construction -- temporal (window=1), hierarchy (≤1 parent), semantic_salient (≤4 out) and motion_neighbor (≤2 out) each independently produce far fewer edges per node than scene_size-1 typically allows, with cross-scene pruning only removing the (rare) excess. The two paths hit the same numeric ceiling on this clip for different structural reasons, and hierarchical_sparse's actual max is well under it.

## §3.5 extension

**Confirmed on this one clip (`Arson042`, N=613), not extended further.** hierarchical_sparse's per-node degree (0/2/2.64 min/median/mean, max 17) is well below fully_connected's (0/8/12.58, max 33) on the same scene partition, as the per-source top-k caps predict. In-degree does concentrate beyond any single per-source cap (max in-degree 12 vs max out-degree 7, largest per-source cap 4), but stays well inside the same scene-size ceiling fully_connected also respects on this clip -- concentration, where present, is real but bounded, not an unbounded blow-up. **This is one clip.** N=613 is a single point in the 24-13,506 range the fully_connected corpus covers; nothing here establishes whether in-degree concentration grows, shrinks, or stays flat as scene size or N grows, whether it behaves the same on a clip with a much larger max scene size, or whether the tiered path's known under-connection failure mode (§3.5's prune-after-select deficit) shows up elsewhere in this same clip's degree-0-for-that-type nodes. Generalizing this clip's numbers to "the" hierarchical_sparse degree distribution would repeat the same error item 8 already flagged for the zero-artifact case, just with N=1 instead of N=0.
