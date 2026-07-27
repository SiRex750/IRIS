# Block-diagonal ingest build — investigation + plan (read-only; NO code changed)

Scope: `_update_all_edge_weights` / `_add_weighted_edge` in `iris/l2_asphodel.py`
(the code the task calls `_build_graph`/`_update_all_edge_weights` — there is no
separate dense-build helper; `iris/ingest.py::_build_graph` is a thin wrapper
that calls `L2Asphodel.add_frame_nodes_bulk` / `enrich_nodes_bulk`, which each
call `_update_all_edge_weights` once). Nothing in this repo was modified to
produce this report.

## KEY RISK — flagged up front (task step 2)

**One global scalar leaks into every intra-scene edge's weight.**
`max_score_range = max(scores) - min(scores)` (`l2_asphodel.py:194-197`) is
computed over **all N nodes' `action_score`**, not per scene, and is passed
into every `_add_weighted_edge` call as `_motion_similarity`'s normalizer
(`l2_asphodel.py:284-288,306`). So an intra-scene edge's weight is *not* a
pure function of the two frames + scene membership in total isolation — it
also depends on the global min/max action_score across the whole video.

This is **not a blocker**: `max_score_range` is an O(N) single pass (already
computed once today, `l2_asphodel.py:192-197`), cheap to keep computing
globally before the block-diagonal loop starts. It just means the
block-diagonal build must NOT compute this scalar per-scene (that would
silently change every weight) — it must compute it once, globally, exactly as
today, and thread it into the per-scene inner loops unchanged. Named
explicitly per the task's request: **this is the one dependency that must be
preserved by construction, not by luck.** Everything else that feeds the
weight (`_semantic_similarity`, `_motion_similarity`'s per-pair cosine,
`_temporal_proximity`) is a pure pairwise function of the two `AsphodelNode`
objects — no other graph-level aggregate is involved.

Verdict: **clean refactor**, conditional on preserving the one global scalar.

## 1. Where the dense graph is materialized and pruned

`_update_all_edge_weights`, `iris/l2_asphodel.py`:

```
199	        if self.graph_edge_mode == "fully_connected":
200	            for i in range(num_nodes):
201	                for j in range(i + 1, num_nodes):
202	                    self._add_weighted_edge(sorted_ids[i], sorted_ids[j], "fully_connected", max_score_range)
203	        else:
204	            self._add_temporal_edges(sorted_ids, max_score_range)
205	            self._add_hierarchy_edges(sorted_ids, max_score_range)
206	            self._add_salient_semantic_edges(sorted_ids, max_score_range)
207	            self._add_motion_neighbor_edges(sorted_ids, max_score_range)
208	
209	        if node_groups is not None:
210	            node_to_group = {}
211	            for gid, group in enumerate(node_groups):
212	                for nid in group:
213	                    node_to_group[nid] = gid
214	            edges_to_remove = []
215	            for u, v in self.graph.edges:
216	                if node_to_group.get(u) != node_to_group.get(v):
217	                    edges_to_remove.append((u, v))
218	            self.graph.remove_edges_from(edges_to_remove)
```

Mechanism confirmed: **literally** builds all `N(N-1)/2` pairs via the
nested `for i / for j in range(i+1, ...)` loop (lines 200-202), then does a
second full pass over `self.graph.edges` to delete any pair whose
`node_groups` membership differs (lines 214-217). This is exactly the
"materialize dense, then prune" pattern described in the task.

**Confirmed this is the actual measured path, not a hypothetical.**
`scripts/virat_smoke_scenesparse.py:30-39` — the script that produced the
committed VIRAT N=4,892 ingest numbers — explicitly sets
`graph_edge_mode="fully_connected"` together with `graph_mode="scene_sparse"`.
(The `IRISConfig` default for `graph_edge_mode` is `"hierarchical_sparse"`,
per `iris/iris_config.py:54`, but the scene_sparse ingest arm overrides it to
`"fully_connected"`, which is why it takes the dense-then-prune branch at all.)
`scripts/_scaling_curve_v2_worker.py:207-220` does the same override for its
in-memory flat comparison arm.

Empirical confirmation the prune is exact and total: `eval_results/virat_latency_N4892_result.md`
records, from `virat_smoke_N4892_scenesparse.json`, `scene_sparse_edge_count: 23571`
and **`block_diagonal_exact: true`** — the persisted graph's edge count equals
`sum_C_scene_size_choose_2` (`sum(n_i*(n_i-1)//2 for n_i in survivors_per_scene)`)
exactly, i.e. zero cross-scene edges survive into the stored/ingest-time graph.

## 2. Weight dependency (see KEY RISK above)

- `_semantic_similarity(node_u, node_v)` — pure pairwise cosine of two
  embeddings (`l2_asphodel.py:239-249`). No graph-level dependency.
- `_motion_similarity(node_u, node_v, max_score_range)` — pairwise, but takes
  `max_score_range` as an argument, and that argument is the one global
  scalar described above (`l2_asphodel.py:251-288`).
- `_temporal_proximity(node_u, node_v)` — pure pairwise function of the two
  timestamps (`l2_asphodel.py:290-292`).
- `_edge_weight(semantic, motion, temporal, edge_type)` for
  `edge_type == "fully_connected"` returns `alpha*semantic + beta*motion`
  only (`l2_asphodel.py:305-306`) — `temporal` is computed and stored as the
  `temporal_weight` attribute on every edge (`_add_weighted_edge`,
  `l2_asphodel.py:390-398,403-411`) but is **not** part of the scalar
  `weight` for this edge type. The identity test (§6) must still check
  `temporal_weight` bit-for-bit since it's a stored, exported field
  (`export_graph_data`, `l2_asphodel.py:1145-1155`), even though it doesn't
  feed `weight` under this edge type.

No other global/graph-level quantity (no normalization over all edges, no
degree-dependent term, no rank-percentile) enters `_edge_weight` for the
`"fully_connected"` branch. `codec_conf` / `packet_size` / `persistence_value`
/ `gamma` never appear in `_edge_weight` at all (see §4).

## 3. Scene assignment and the cross-scene rule

`_refresh_scene_ids` (`l2_asphodel.py:220-237`): sorts nodes by
`(timestamp, frame_idx)`, walks in order, and increments `scene_id` each time
it encounters a node whose `pict_type` starts with `"I"` (case-insensitive),
except the very first node. This runs identically today and would run
identically under the block-diagonal build — it's untouched by this fix and
must stay untouched (still the single source of truth for scene membership).

Block-diagonal = "edge iff same `scene_id`" **is confirmed sufficient with NO
additional cross-scene rule needed at ingest time.** The cross-scene edge
rule lives in `add_cross_scene_edges` (`l2_asphodel.py:490-658`, modes
`"all"` / `"threshold"` / `"rep_only"`), but per its own docstring
(`l2_asphodel.py:500-506`): *"populates *graph* — normally the induced
subgraph copy... The production `self.graph` is **never** written to: all
edges go exclusively into the supplied *graph* argument."* This is called
only from `scene_retrieval.py`'s DESCEND path, on ephemeral per-query
subgraph copies — never during ingest, never persisted. The empirical
`block_diagonal_exact: true` finding in §1 is the direct proof: the stored
ingest-time graph already contains zero cross-scene edges, because the
`node_groups` prune (lines 209-218) removes every one of them (`node_groups`
covers 100% of the graph_edge_mode="fully_connected" pairs, since it's built
from the same scene_id partition).

**Correction to the task's framing**: the fix does *not* need to "add
cross-scene edges by the existing rule" at ingest time — that rule doesn't
fire at ingest time today, at all. The block-diagonal ingest build only needs
to reproduce the intra-scene blocks.

## 4. Live edge-weight components under the committed config

Committed config for scene_sparse ingest (`scripts/virat_smoke_scenesparse.py:30-39`):
`ranking_mode="ppr"`, `codec_conf_source="packet_size"`, `graph_edge_mode="fully_connected"`.

Under `graph_edge_mode="fully_connected"` (the mode actually used), every
edge is `edge_type="fully_connected"` — the `_add_temporal_edges` /
`_add_hierarchy_edges` / `_add_salient_semantic_edges` /
`_add_motion_neighbor_edges` family (lines 413-486, which is what fires
under `"hierarchical_sparse"`) never runs. So "which weights fire" reduces to:

- **Live**: `semantic` (`alpha` term) and `motion` (`beta` term) inside
  `_edge_weight`'s `"fully_connected"` branch (`l2_asphodel.py:305-306`).
  `temporal_weight` is computed/stored but structurally excluded from
  `weight` for this edge type (see §2) — still must match bit-for-bit as a
  stored attribute.
- **Dead for edge construction, regardless of `ranking_mode`**: `gamma`,
  `persistence_value`, `codec_conf`, `packet_size`. These only appear in
  `retrieve()`'s legacy scoring formula (`l2_asphodel.py:920-923,966`) and in
  `_seed_from_codec`-style PPR personalization blending
  (`l2_asphodel.py:1007-1075`), never in `_edge_weight` or
  `_update_pagerank`'s personalization dict (which uses `action_score`,
  `l2_asphodel.py:681-684`, not `persistence_value`). Confirmed by grep —
  `gamma`/`persistence_value`/`codec_conf`/`packet_size` do not appear
  anywhere between lines 155-411 of `l2_asphodel.py` (the edge-construction
  block). The earlier project note that "`persistence_value`'s gamma term
  only fires under legacy" is consistent with this — under `ppr` it doesn't
  fire either, because it never touches edges at all, only `retrieve()`.

So the fix only needs to preserve `alpha*semantic + beta*motion`
(`fully_connected` weight) plus the stored `semantic_weight` / `motion_weight`
/ `temporal_weight` triple, per intra-scene pair.

## 5. Proposed block-diagonal construction (design, no code written)

Keep everything in `_update_all_edge_weights` identical up through line 197
(node sort, `scores`/`max_score_range` computed globally over **all** N
nodes — this must stay a full-N pass, not per-scene, per the KEY RISK). Then,
when `node_groups is not None` (i.e. scene_sparse) **and**
`graph_edge_mode == "fully_connected"`:

1. Partition `sorted_ids` into per-scene sublists, **preserving the existing
   global sort order** (`(timestamp, frame_idx)`) within each sublist — i.e.
   filter `sorted_ids` by `node_to_group[nid] == gid` for each group, do not
   re-sort. This matters because `_add_weighted_edge`'s only order-dependent
   behavior is "insert once, no update path fires" (true either way, since
   each pair is visited exactly once under both old and new construction),
   but keeping insertion order identical is the cheapest way to make the
   secondary risk in §6 (edge/PageRank insertion-order sensitivity)
   moot rather than merely "probably fine."
2. For each scene's sublist (length `n_i`), run the same nested
   `for i in range(n_i): for j in range(i+1, n_i):` loop, calling
   `self._add_weighted_edge(scene_ids[i], scene_ids[j], "fully_connected", max_score_range)`
   — byte-identical call to what line 202 does today, just restricted to
   pairs within one scene.
3. Do **not** call the `node_groups` prune block (lines 209-218) in this
   path — there is nothing to prune, since cross-scene pairs are never
   visited in the first place.
4. Cross-scene edges: unchanged — still added transiently by
   `add_cross_scene_edges` at query time (§3), never at ingest.
5. Fall back to the existing dense-then-prune path unchanged whenever
   `graph_edge_mode != "fully_connected"` or `node_groups is None` (flat
   mode) — this fix is scoped to the one branch that is actually O(N²) in
   practice for scene_sparse.

Expected complexity: `O(Σ n_i²)` over scenes (plus the O(N) global
`max_score_range` pass) vs current `O(N²)`. For the VIRAT N=4,892 / 528-scene
case, `Σ n_i²` is bounded by `2 * sum_C_scene_size_choose_2 + N =
2*23,571 + 4,892 ≈ 52,034`, vs `N² ≈ 23,930,864` today — roughly a 460x
reduction in pair-visits, consistent with the reported 26GB peak being driven
by materializing ~11.96M edges (`edges_flat_theoretical_N_choose_2` /
`flat_edge_count` in the flat-mode comparison, `virat_smoke_N4892_flat.json`)
before discarding all but 23,571 of them.

## 6. Edge-identity test (specification)

**Fixture**: the existing cached VIRAT N=4,892 index
(`eval/data/virat/index_cache/VIRAT_S_040001_01_000448_001101.npz`) as the
primary case (matches the committed latency numbers), plus one small
NExT-GQA clip as a cheap second case with a different scene-count profile.

**Procedure**: load the same frame records once (`iris_ingest.load_index` or
the equivalent frame list), then build TWO `L2Asphodel` graphs from the
identical node list + identical config (`alpha`, `beta`,
`graph_edge_mode="fully_connected"`, `graph_mode="scene_sparse"`):

- **Graph A (current)**: today's `_update_all_edge_weights` — dense
  `N(N-1)/2` build, then `node_groups` prune.
- **Graph B (new)**: the block-diagonal path from §5.

**"Identical" is defined precisely as, in this order, ALL of:**

1. `set(A.graph.nodes) == set(B.graph.nodes)` (sanity check, should be
   trivially true — node insertion is untouched by this fix).
2. `{frozenset({u, v}) for u, v in A.graph.edges} == {frozenset({u, v}) for u, v in B.graph.edges}`
   — exact set equality of node-pairs. Zero missing, zero extra, on both
   sides (assert both `A - B == {}` and `B - A == {}` separately so a
   direction-specific regression is diagnosable, not just "sets differ").
3. For every edge in that common set: `A.graph[u][v]["weight"] == B.graph[u][v]["weight"]`
   and the same for `semantic_weight`, `motion_weight`, `temporal_weight`,
   `edge_type`, at **exact float equality (tolerance 0, not `np.isclose`)**.
   Justification: both paths call the exact same `_add_weighted_edge` →
   `_semantic_similarity` / `_motion_similarity` / `_temporal_proximity` /
   `_edge_weight` functions with the same operands in the same order per
   pair — this is a change in *which pairs get visited*, not in the
   arithmetic used for any visited pair, so exact bit-identity is the
   correct bar, not an approximation tolerance. Any float mismatch here
   means the refactor accidentally changed an operand (e.g. computed
   `max_score_range` per-scene instead of globally) and must fail the test,
   not be waved through with a tolerance.
4. `A.graph.number_of_edges() == B.graph.number_of_edges() == sum(n_i*(n_i-1)//2 for n_i in scene_sizes)`.
5. **Secondary check (PageRank order-sensitivity)**: after building both
   graphs, run `_update_pagerank()` on each and assert
   `A.graph.nodes[nid]["node_data"].pagerank_score == B.graph.nodes[nid]["node_data"].pagerank_score`
   for every node, exact float equality. This guards against the KEY RISK's
   quieter cousin: `nx.pagerank` builds its adjacency representation from
   the graph, and if edge *insertion order* (not just the edge set) ever
   influenced floating-point summation order internally, the block-diagonal
   build's different insertion order could produce a bit-different but
   "equally valid" PageRank result — which would silently propagate into
   every downstream committed number. Preserving the global sort order in
   §5 step 1 is the mitigation; this check is how you'd catch it if the
   mitigation were incomplete.

Any single failure above is a **STOP** — do not proceed to §7, investigate
which operand diverged first.

## 7. Acceptance guard (go/no-go)

After the edge-identity test in §6 passes on both fixtures, re-run the
NExT-GQA held-out val cell end-to-end (same harness/config that produced the
committed number) and assert:

```
peak_in_gold == 0.3227   # exact float equality, not "close to"
```

**Go**: byte-identical `peak_in_gold`, plus §6 passing on both fixtures.
Ship as "cheaper build, same graph."

**No-go**: any deviation, however small. Do not rationalize a small
deviation as "noise" — the whole premise of this fix is that nothing
downstream should be able to tell the difference. If `peak_in_gold` moves at
all, treat §6's secondary PageRank check as the first thing to re-examine
(insertion-order sensitivity), followed by re-checking whether
`max_score_range` was accidentally scoped per-scene instead of globally
(the KEY RISK from the top of this document).

## Explicitly out of scope / not attempted

- No code in `iris/l2_asphodel.py`, `iris/ingest.py`, or any other file was
  modified to produce this report.
- No re-ingest was run.
- The `"hierarchical_sparse"` branch (`_add_temporal_edges` /
  `_add_hierarchy_edges` / `_add_salient_semantic_edges` /
  `_add_motion_neighbor_edges`) is untouched by this plan — it is not the
  measured path for the committed scene_sparse ingest numbers (§1) and has
  its own multi-edge-type "keep max weight" merge logic
  (`_add_weighted_edge`, lines 399-402) that would need separate analysis
  before any similar optimization, since (unlike `"fully_connected"`) a pair
  can receive more than one call across the four families.
