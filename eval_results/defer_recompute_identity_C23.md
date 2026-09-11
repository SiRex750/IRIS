# defer_recompute identity re-verification (C23)

**Outcome: PASS.** The `defer_recompute=True` change in `iris/ingest.py::_build_graph`
(`add_frame_nodes_bulk(..., defer_recompute=True)`) is behavior-preserving.
Independently re-derived and empirically re-tested at zero tolerance across three
`graph_edge_mode`s and both the `ingest` and `load_index` paths — no non-identity found.

## 1. What `eval_results/build_dedup.md`'s safety-read trace actually checked

`build_dedup.md` Step 1 ("Safety read (before touching code)") is **pure code inspection**,
not execution — it says so explicitly ("before touching code"):

- **(a)** Traced that `add_frame_nodes_bulk` only calls `self.graph.add_node(...)` before
  its own recompute, and that `enrich_nodes_bulk` only touches `node_data.triples`/`embedding`
  — concluding it "never reads `self.graph.edges`, `pagerank_score`, or `scene_id`."
- **(b)** Traced that `_refresh_scene_ids()` derives `scene_id` from `(timestamp, frame_idx,
  pict_type)` only, independent of embeddings, so a stale first refresh doesn't matter — it
  gets unconditionally rerun by `enrich_nodes_bulk`'s own `_update_all_edge_weights()` call.
- **(c)** Enumerated other add→enrich call sites (`pipeline.py::wrapper_l2_retrieve`, several
  test/benchmark scripts) and noted `defer_recompute` defaults to `False` and is passed `True`
  only from `_build_graph`, so they're untouched by construction.

Everything **after** Step 1 (Steps 2–4: the identity gate, the `defer_recompute_equivalence.py`
script, the grounding gate, the build-cost probe) is **execution-based** — actual scripts run
against the cached VIRAT clip, not just read. So the doc's claim rests on inspection for the
"is there a hazard at all" question and on execution for the "does the change actually produce
an identical graph" question. Both are present; neither is missing.

One gap in the doc's own execution coverage: `defer_recompute_equivalence.py` (the direct
True-vs-False proof) only exercises `graph_edge_mode="hierarchical_sparse"` (the production
default). It does not directly re-run the True-vs-False comparison under `block_diagonal` or
`fully_connected` — those two modes are only cross-checked against *each other* (both already
running with `defer_recompute=True` baked into `_build_graph`) via the identity/grounding gates.
Closed below (§3).

## 2. Independent code trace (this verification)

Read `iris/l2_asphodel.py` and `iris/ingest.py` directly, independent of the doc's narrative:

- **`_build_graph`** (`iris/ingest.py:150-153`): `add_frame_nodes_bulk(..., defer_recompute=True)`
  is immediately followed by `enrich_nodes_bulk(...)` — no code runs between them. `node_groups`
  (used by both calls) is computed *before* either call, from `records[].scene_id`, not from the
  graph — unaffected.
- **`add_frame_nodes_bulk`** (`iris/l2_asphodel.py:835-921`): with `defer_recompute=True`, returns
  immediately after the node-insertion loop (`self.graph.add_node(...)` only) — never calls
  `_update_all_edge_weights()`/`_update_pagerank()`. No edges, no scene_id, no pagerank_score are
  touched on the deferred path (nodes are constructed with `embedding=None`, not edges).
- **`enrich_nodes_bulk`** (`iris/l2_asphodel.py:923-942`): sets `node_data.embedding` for each
  node, then unconditionally calls `_update_all_edge_weights()` and `_update_pagerank()` once.
- **`_update_all_edge_weights`** (`iris/l2_asphodel.py:155-242`): starts with
  `self.graph.remove_edges_from(list(self.graph.edges))` then **unconditionally** calls
  `self._refresh_scene_ids()` before doing anything else (even in the `num_nodes <= 1` early
  return). So `enrich_nodes_bulk` always rebuilds edges and `scene_id` from scratch, regardless
  of whether `add_frame_nodes_bulk`'s own (now-skipped) pass ran first.
- **Other call sites**: `pipeline.py::wrapper_l2_retrieve`'s `batch_add_frame_nodes`/
  `batch_enrich_nodes` shim (`iris/pipeline.py:302-312, 440-441`) does not pass
  `defer_recompute`, so it uses the default `False` — untouched by this change, confirming
  `build_dedup.md`'s (c).

**Conclusion: nothing reads `graph.edges`, `node_data.pagerank_score`, or `node_data.scene_id`
between the two calls, on either the ingest or the (see below) load_index path.** Matches the
doc's claim.

## 3. `load_index` path

`load_index` (`iris/ingest.py:611-679`) calls `_build_graph(index.frames, ...)` — same
`defer_recompute=True` call — but then, when the manifest carries `graph_edges`:

```python
index._graph.graph.remove_edges_from(list(index._graph.graph.edges))
for edge in manifest["graph_edges"]:
    ...
    index._graph.graph.add_edge(source, target, weight=..., edge_type=..., ...)
index._graph._refresh_scene_ids()
index._graph._update_pagerank()
```

This **unconditionally discards every edge `_build_graph` produced** (deferred or not) and
rebuilds the graph's edges from the serialized manifest, then reruns `_refresh_scene_ids()` and
`_update_pagerank()` over that manifest-derived edge set. `_build_graph`'s own edge/PageRank
output — whichever setting produced it — is never read on this path; only the node set (and
node feature/embedding data) survives from `_build_graph`. **`defer_recompute` changes nothing
observable on the `load_index` path** — the edges/PageRank/scene_id it might have skipped
computing are overwritten regardless, and the ones it does compute (via `enrich_nodes_bulk`,
still called from `_build_graph`) are also overwritten. Confirms the task's hypothesis.

## 4. Empirical re-verification (independent scripts, this session)

### 4a. Re-ran `scripts/defer_recompute_equivalence.py` fresh (hierarchical_sparse / flat)

N=4,892 nodes / 6,284 edges, both arms. `outcome: BIT_IDENTICAL` — node set, edge set, all 5
edge fields, scene_id, pagerank_score all identical. (Reproduces the doc's Step 3(b) result
exactly; raw output re-saved to `eval_results/defer_recompute_equivalence.json`.)

### 4b. New script: True-vs-False identity under `block_diagonal` and `fully_connected` (closes the gap in §1)

Wrote and ran an independent script (mirrors `_build_graph`'s construction but exposes
`defer_recompute` explicitly, same as the doc's own script) against the same cached clip
(`eval/data/virat/index_cache/VIRAT_S_040001_01_000448_001101.npz`, N=4,892/528 scenes),
comparing `defer_recompute=False` vs `True` at zero tolerance in **three** configs:

| config | n_edges (old=False) | n_edges (new=True) | edge set match | field mismatches | scene_id mismatches | pagerank mismatches |
|---|---|---|---|---|---|---|
| `scene_sparse` / `block_diagonal` | 23,571 | 23,571 | ✅ | 0 | 0 | 0 |
| `scene_sparse` / `fully_connected` | 23,571 | 23,571 | ✅ | 0 | 0 | 0 |
| `flat` / `hierarchical_sparse` | 20,540 | 20,540 | ✅ | 0 | 0 | 0 |

All three: node sets identical (4,892/4,892), edge sets identical, all 5 edge fields
(`weight`, `semantic_weight`, `motion_weight`, `temporal_weight`, `edge_type`) identical on
every common edge, every `scene_id` identical, every `pagerank_score` identical. **Zero
non-identities in any config, in either direction of the flag.**

(The `flat`/`hierarchical_sparse` edge count of 20,540 vs. 23,571 is expected and unrelated to
`defer_recompute` — it's the pre-existing difference between the production edge topology and
the `scene_sparse` fully-connected/block-diagonal comparison arms; `build_cost_final_probe.py`'s
guard only exercises the latter two.)

### 4c. Regression suite

`pytest tests/test_ppr_production.py tests/test_ppr_retrieve.py tests/test_l2_asphodel.py tests/test_index_io.py`
→ **22 passed, 1 xfailed** (re-ran independently, matches doc).

## 5. Item 4 — the 23,571-edge guard in `scripts/build_cost_final_probe.py`

That script always calls `iris_ingest._build_graph` directly, which (in the current working
tree) hardcodes `defer_recompute=True` — so run as-is it only ever exercises the `True` setting.
To answer "does it pass under both settings," §4b's `block_diagonal` and `fully_connected` rows
above are exactly its two arms (`old`=`fully_connected`, `new`=`block_diagonal`), built through
the same code path with `defer_recompute` explicit in both directions:

- `defer_recompute=False` (original behavior): 23,571 edges, both arms. **Guard would pass.**
- `defer_recompute=True` (current/pending change): 23,571 edges, both arms. **Guard passes**
  (this is what running the script as-is already confirms, and matches
  `eval_results/blockdiag_identity_gate_result.json`'s `sum_c_scene_size_choose_2: 23571` /
  `edge_count_a/b: 23571`).

**Confirmed: the guard passes under both settings.**

## Verdict

**PASS — no blocker.** Across code inspection (ingest path + load_index path), the doc's own
execution-based gates, and this session's independent re-execution (including the one config
combination — `block_diagonal`/`fully_connected` under explicit True-vs-False — that
`build_dedup.md`'s own direct equivalence script didn't cover), the change is bit-identical at
zero tolerance in every edge field, every node's `scene_id`, and every node's `pagerank_score`,
under both `ingest()` and `load_index()`. Safe to commit as behavior-preserving.
