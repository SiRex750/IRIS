# Block-diagonal ingest build — landed

Implements `eval_results/blockdiag_build_plan.md`. New `graph_edge_mode="block_diagonal"`
in `iris/l2_asphodel.py` (`_update_all_edge_weights`) + `iris/iris_config.py`
validation. `fully_connected` and `hierarchical_sparse` are unmodified — the
default `graph_edge_mode` config value is unchanged (`"hierarchical_sparse"`);
`block_diagonal` is opt-in only.

## What it does

For `graph_edge_mode="block_diagonal"` + scene_sparse (`node_groups` present):
computes `max_score_range` globally over all N nodes exactly as before (the
one global dependency identified in the plan), then builds ONLY intra-scene
edges directly per scene, using the identical `alpha*semantic + beta*motion`
weight formula the `"fully_connected"` branch uses. Never materializes the
`N(N-1)/2` dense intermediate; no cross-scene prune pass needed (cross-scene
pairs are never visited).

## Bit-identity proof (three levels)

1. **Graph construction** (`eval_results/blockdiag_identity_gate_result.md`):
   VIRAT N=4,892 / 528 scenes. 23,571 edges compared old (`fully_connected`+
   prune) vs new (`block_diagonal`) — node set, edge-pair set, and every
   `weight`/`semantic_weight`/`motion_weight`/`temporal_weight`/`edge_type`
   field bit-identical (tolerance 0.0). PageRank score bit-identical on all
   4,892 nodes. Personalized PageRank ranking order identical across 5
   seeded queries.

2. **Grounding, scene_sparse, real pipeline** (`eval_results/blockdiag_grounding_gate_result.md`):
   526 NExT-GQA VAL questions / 86 videos, full production path (scene
   routing shortcut/descend, induced subgraphs, `add_cross_scene_edges`,
   span construction). `fully_connected` vs `block_diagonal`: **0 mismatches**
   on retrieved order, peak timestamp, span, IoP, and peak_in_gold, for every
   question. `peak_in_gold` and `mIoP` identical to full float precision
   (0.31749049429657794 / 0.31402667038753285, both arms).

3. **Frozen flat VAL cell reconfirmed unaffected**: `peak_in_gold` rerun at
   top_k=8/half_width=2.2 on graph_mode=flat gives 0.32266... → rounds to the
   committed **0.3227** exactly. Flat never builds `node_groups`, so
   `block_diagonal` is structurally unreachable from that config — this cell
   could not have been touched by the change, confirmed by rerun rather than
   assumed.

No committed grounding number was ever produced with
`graph_mode=scene_sparse + graph_edge_mode=fully_connected` (existing
scene_sparse arms use the default `hierarchical_sparse` formula, which
`block_diagonal` doesn't target) — so (2) is a purpose-built equivalence
proof for the pairing this fix actually concerns, not a rerun of a
pre-existing number. Combined with (3), no committed number changes.

Cache integrity: `eval/data/nextqa/index_cache` and `index_cache_ssparse`
sha256 unchanged before/after the grounding gate run — no re-ingest.
Existing test suite (`test_ppr_production.py`, `test_index_io.py`,
`test_l2_asphodel.py`, 19 tests) passes unchanged.

## Measured savings (VIRAT N=4,892, isolated build cost — `eval_results/blockdiag_gate_and_savings_summary.md`)

| | OLD (`fully_connected`+prune) | NEW (`block_diagonal`) | ratio |
|---|---|---|---|
| build wall time | 69.47 s | 0.316 s | ~220x faster |
| peak RSS during build | 6.36 GB | 0.95 GB | ~6.7x less |
| edges kept (both) | 23,571 | 23,571 | identical |

Old visits ~11.96M pairs to keep 23,571 (99.8% wasted); new visits exactly
23,571 pairs — zero wasted pair-visits, matching the plan's O(Σnᵢ²) vs O(N²)
prediction.

## Status

Landed as an additional opt-in mode. The config default (`"hierarchical_sparse"`)
is unchanged — switching scene_sparse ingest to use `block_diagonal` by
default is a separate decision, not made here.
