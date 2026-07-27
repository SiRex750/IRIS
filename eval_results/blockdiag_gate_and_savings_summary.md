# Block-diagonal ingest build — gate result + measured savings (VIRAT N=4,892)

Implements `eval_results/blockdiag_build_plan.md`. New `graph_edge_mode="block_diagonal"`
added to `L2Asphodel._update_all_edge_weights` (`iris/l2_asphodel.py`) and
`IRISConfig.graph_edge_mode` validation (`iris/iris_config.py`). Existing
`"fully_connected"` and `"hierarchical_sparse"` code paths are unmodified —
verified via the existing test suite (`tests/test_ppr_production.py`,
`tests/test_index_io.py`, `tests/test_l2_asphodel.py`, 19/19 pass) and via the
gate below, which builds the OLD path with its original code path untouched.

## Gate result: GATE_PASS

Full detail: `eval_results/blockdiag_identity_gate_result.{json,md}`
(script: `scripts/blockdiag_identity_gate.py`).

- N survivors: 4,892, scenes: 528
- Theoretical block-diagonal edge count `sum(n_i*(n_i-1)//2)`: 23,571
- Graph A (OLD `fully_connected` + cross-scene prune) edge count: **23,571**
- Graph B (NEW `block_diagonal`) edge count: **23,571**
- Node set: identical. Edge-pair set: identical (0 only-in-A, 0 only-in-B).
- 23,571 common edges compared on `weight`, `semantic_weight`,
  `motion_weight`, `temporal_weight`, `edge_type` — **0 mismatches**, exact
  float equality (tolerance 0.0).
- PageRank score: bit-identical on all 4,892 nodes.
- Personalized PageRank ranking order: identical top-20 order across 5
  seeded personalization vectors, and exact-score match on all nodes for
  every seed.

No mismatch found. Per the task's gate instructions, this clears the way to
report savings below — no grounding run has been executed, and nothing has
been committed.

## Measured savings: isolated build cost, OLD vs NEW (VIRAT N=4,892, 528 scenes)

Measured via `scripts/blockdiag_build_probe.py`, each mode run in its own
process (so the two builds never share a heap), isolating just the
`iris_ingest._build_graph(...)` call (cached-frame load excluded from the
timed/measured window, matching the `virat_smoke_flat.py` convention).

| | OLD (`fully_connected` + prune) | NEW (`block_diagonal`) | ratio |
|---|---|---|---|
| build wall time | 69.47 s | 0.316 s | **~220x faster** |
| peak RSS during build | 6.36 GB | 0.95 GB | **~6.7x less memory** |
| edge count (both) | 23,571 | 23,571 | identical |

Raw JSON: `eval_results/blockdiag_probe_old.json`, `eval_results/blockdiag_probe_new.json`.

Note on absolute numbers: this isolates only the `_build_graph` call from
already-loaded/cached frames+embeddings (not a full video ingest), so the
absolute peak-RSS figures here are lower than the 26GB reported for the full
`ingest()` pipeline in `virat_smoke_scenesparse.py` (which also holds decoded
video frames, CLIP embeddings, and other pipeline state in memory
concurrently). The ~220x/~6.7x ratios isolate the specific O(N²) → O(Σn_i²)
effect this fix targets; they are not a claim about the full-pipeline 26GB
number, which was never re-measured here (no re-ingest was run, per the
task's constraint).

## Complexity observed

Old: materializes all `N(N-1)/2 = 4892*4891/2 ≈ 11.96M` pairs, discards all
but 23,571 (99.8% of visited pairs are cross-scene and thrown away).
New: visits only `sum(n_i*(n_i-1)//2) = 23,571` pairs directly — i.e. the
number of pairs visited now equals the number of edges kept, exactly, with
zero wasted pair-visits. This matches the plan's O(Σn_i²) vs O(N²) prediction.

## Status: gate passed, STOPPING here for review

Per task instructions: the grounding acceptance run (`peak_in_gold == 0.3227`
exact-match confirmation) is the NEXT step and is explicitly NOT run yet.
Nothing has been committed. Changed files, uncommitted:

- `iris/l2_asphodel.py` — new `block_diagonal` branch in
  `_update_all_edge_weights`; `fully_connected`/`hierarchical_sparse`
  branches untouched (only the cross-scene prune step is now skipped when
  `graph_edge_mode == "block_diagonal"`, since block_diagonal never visits
  cross-scene pairs to begin with).
- `iris/iris_config.py` — `graph_edge_mode` now accepts `"block_diagonal"`.
- `scripts/blockdiag_identity_gate.py` — gate test (new).
- `scripts/blockdiag_build_probe.py` — isolated memory/time probe (new).
- `eval_results/blockdiag_identity_gate_result.{json,md}`,
  `eval_results/blockdiag_probe_{old,new}.json`,
  `eval_results/blockdiag_gate_and_savings_summary.md` — this run's artifacts.
