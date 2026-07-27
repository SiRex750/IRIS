# Item D audit — is `peak_in_gold` flat by construction?

**Verdict: `peak_in_gold` is computed over ALL Charon survivors (it reads element
`[0]` of a list that PPR has already ranked globally and that top-K only
truncates), therefore its flatness across K is true by construction.**

Read-only audit. No file was modified, no eval/ingest/captioner/answerer/LLM call
was made.

- `git rev-parse HEAD` at start: `505c2f70bfd310eefd662aea327526cbd3ef9428`
- branch: `feat/prerun-fixes` (unchanged)
- `git status --porcelain` at start:
  ```
   M eval/metrics.py
   M external_data/qvhighlights/manifests/external_train.jsonl
  ?? scratch/bakeoff_harness_dataset.json
  ?? scripts/answerer_bakeoff_harness.py
  ?? scripts/span_sweep.py
  ?? scripts/span_sweep_bootstrap.py
  ?? tests/test_span_methods_ef.py
  ?? tuning/span_sweep/
  ```

---

## 0. Scope note on the table this audit was pointed at

The task described a `by_top_k` funnel table with pool coverage
`~0.53 / 0.658 / 0.798 / 0.857` at `K = 4 / 8 / 16 / 24` and `peak_in_gold ~0.325`
flat. **No such table, and no code emitting a `by_top_k` table over
`K = 4/8/16/24`, exists anywhere in this checkout.** Concrete checks run:

- `grep -rIln "peak_in_gold\|by_top_k" .` → only `scripts/part3c_span_method_comparison.py`,
  `tuning/span_method_peak_in_gold.csv`, `tests/test_scene_sparse_descend.py`
  (an unrelated `test_result_bounded_by_top_k`), plus the same two files inside
  `.claude/worktrees/sparse-temporal-reranker/`.
- `grep -rn "K_GRID\|TOP_K_GRID\|K_VALUES" --include=*.py .` → every K grid in the
  repo is `[4, 5, 8, 12, 16]` (`part3c_span_method_comparison.py:43`,
  `part3e_lambda_k_span_method_comparison.py:62`, `part3e_bootstrap_ci.py:43`).
  No grid contains 24.
- The `K=4` pool-coverage figure *is* reproducible: `gold_at_4 = 0.5303538175046555`
  in `tuning/span_sweep/sweep_summary.json` and `tuning/span_sweep/reproduction_gate.json`.
  The `K = 8/16/24` columns and the `0.325` peak figure appear in no committed artifact.

**UNDETERMINED:** the provenance of the exact `0.658 / 0.798 / 0.857 / 0.325`
numbers. They were not produced by any script in this repository, so I cannot
quote their source.

What this audit therefore does is answer the substantive question against the
**only** `peak_in_gold` computation that exists in the repo, whose committed
output is likewise exactly flat across its own K grid.

---

## 1. Location of the `peak_in_gold` computation

`scripts/part3c_span_method_comparison.py`

- peak selection + gold membership test: **lines 158–160**
- accumulation: lines 171–172
- per-question emission: lines 188–191
- per-K rate print: line 210
- CSV writer (`tuning/span_method_peak_in_gold.csv`): lines 229–233

This is the sole definition in the tree.

## 2. The peak-selection lines, quoted verbatim

`scripts/part3c_span_method_comparison.py:128-160`:

```python
        for q in questions:
            vid = q["video"]
            if vid not in index_paths:
                continue
            if vid not in index_cache:
                index_cache[vid] = iris_ingest.load_index(index_paths[vid])
            index = index_cache[vid]
            try:
                qe, _ = _call_embed_query(q["question"], cfg)
                frames, _ = _retrieve_with_l1(index, qe, cfg)
            except Exception:
                continue
            if not frames:
                continue

            gold_spans = q["gold_spans"]
            timestamps = [f["timestamp"] for f in frames]
            ...
            n_clust = n_clusters_for(frames, GAP_THRESHOLD_S)
            peak_top_ts = frames[0]["timestamp"]
            peak_in_gold = any(g[0] <= peak_top_ts <= g[1] for g in gold_spans)
```

The peak is `frames[0]["timestamp"]` — **rank 1, unconditionally**. There is no
argmax, no scan, no comparison across the retrieved set.

## 3. Trace of the candidate-frame variable back to its origin

**Does K appear in the peak-selection path? Yes — and it is provably inert.**

`k` is the loop variable over `K_GRID` (line 111) and it does reach the retrieval
call, via the config:

`scripts/part3c_span_method_comparison.py:111-112`
```python
    for k in K_GRID:
        cfg = pt.make_config({**FROZEN, "l2_retrieve_top_k": k})
```
→ line 137 `frames, _ = _retrieve_with_l1(index, qe, cfg)`

`_retrieve_with_l1` (`iris/query.py:745`) reads `use_l1` (`iris/query.py:758`);
`use_l1` is not in `tuning/frozen_state.json`, so the default `False` applies and
line 760 delegates straight to `_build_retrieved`:

`iris/query.py:759-760`
```python
    if not use_l1:
        return _build_retrieved(index, query_embedding, config, trace=trace), telemetry
```

`_build_retrieved` (`iris/query.py:295`) reads `l2_retrieve_top_k` at line 319 and
passes it as `top_k` into PPR retrieval:

`iris/query.py:319-336`
```python
    l2_retrieve_top_k = getattr(config, "l2_retrieve_top_k", 5)
    ...
        if ranking_mode == "ppr":
            lambda_ = getattr(config, "ppr_lambda", 0.5)
            damping  = getattr(config, "ppr_damping", 0.5)
            retrieved_nodes = graph.retrieve_ppr(
                query_embedding,
                top_k=l2_retrieve_top_k,
                damping=damping,
                lambda_=lambda_,
            )
```

**The K-invariance is created here.** `AsphodelGraph.retrieve_ppr`
(`iris/l2_asphodel.py:1176`) solves PageRank over the *entire* graph — every
Charon survivor — and only then sorts and slices:

`iris/l2_asphodel.py:1362-1403`
```python
        try:
            pr = nx.pagerank(
                g,
                weight="weight",
                personalization=seed,
                alpha=damping,
            )
        ...
        # Stable tie-break by node ID so repeated calls with identical inputs
        # always produce the same ordering (SCENE-003 determinism requirement).
        sorted_ids = sorted(pr, key=lambda k: (-pr[k], k))
        return [g.nodes[k]["node_data"] for k in sorted_ids[:top_k]]
```

`pr` is over all nodes; `top_k` appears only in the final `[:top_k]` slice.
Truncating the tail of a descending-sorted list cannot change element `[0]`.
`_build_retrieved` then preserves that order (`iris/query.py:346-349`, an
in-order `for node in retrieved_nodes` append), so:

> `frames[0]` is the global argmax of PPR score over all survivors, identical for
> every `K ≥ 1`.

So the answer to the audit question is: **the peak whose gold-membership is
tested is the argmax over ALL Charon survivors, not over the retrieved top-K.**
K is present in the path but cannot reach the selected element.

### Empirical confirmation from the committed artifact

`tuning/span_method_peak_in_gold.csv` (2685 questions × 5 K values):

| K | n | peak_in_gold rate |
|---|---|---|
| 4  | 2685 | 0.294227 |
| 5  | 2685 | 0.294227 |
| 8  | 2685 | 0.294227 |
| 12 | 2685 | 0.294227 |
| 16 | 2685 | 0.294227 |

Not approximately flat — bit-identical. Stronger still, grouping the CSV by
`(video, qid)` and comparing `top_frame_timestamp` across the five K values:
**0 of 2685 questions have a differing top-frame timestamp.** The anchor is the
same frame at every K, for every question.

(The committed rate is `0.294227`, not the `0.325` quoted in the task. `0.325`
does not appear as a `peak_in_gold` rate in any committed artifact — see §0.)

## 4. Other peak/anchor-in-gold computations in the repo

| Site | Anchor rule | K-dependent? | Agrees with §3? |
|---|---|---|---|
| `scripts/part3c_span_method_comparison.py:159` | `frames[0]` | No | — (this is the site under audit) |
| `scripts/query_reformulation_v2_ablation.py:191` `anchor_in_gold_at_1` | `frames[0]` | No | **Yes** — same rank-1 rule, same invariance |
| `scripts/query_reformulation_v2_ablation.py:190` `gold_at_4` | `any()` over `frames[:GOLD_AT_K]` | Yes | Different metric: this is *pool coverage*, which genuinely grows with K |
| `scripts/span_sweep.py:141-145` | `_pick_peak_by_clip(retrieved_frames, qe)`, falling back to `retrieved_frames[0]` | **Yes** | **Disagrees** — see below |
| `eval/metrics.py:160` `predicted_span_from_frames_peak` (Method D) | `_pick_peak_by_clip(...)` else `retrieved_frames[0]` | **Yes** | **Disagrees** — see below |
| `tuning/ppr_lambda_resweep_per_question.csv` columns `any_in_gold/top1_in_gold/anchor_in_gold` | UNDETERMINED — the generating script is not in the tree (`grep -rn "any_in_gold\|top1_in_gold" --include=*.py .` returns nothing) | UNDETERMINED | UNDETERMINED |
| `tuning/embedding_backbone_per_question.csv` columns `approx_retrieved_in_gold/top1_in_gold` | UNDETERMINED — same, no generating script in tree | UNDETERMINED | UNDETERMINED |

The disagreement is real and worth stating plainly: `_pick_peak_by_clip`
(`eval/metrics.py:56-79`) *does* scan the retrieved pool —

```python
    best, best_sim = None, -1.0
    for fr in retrieved_frames:
        emb = fr.get("clip_embedding")
        ...
        sim = float(np.dot(q, v / vn))
        if sim > best_sim:
            best, best_sim = fr, sim
    return best
```

— so the **Method D span anchor** is a true argmax over the top-K and *would*
move with K. The `peak_in_gold` **diagnostic** in `part3c` does not use it; it
uses `frames[0]`. The two "peaks" in the same script are not the same object.
That mismatch is the mechanism behind the flat column.

## 5. Verdict

**`peak_in_gold` is computed over all survivors, therefore its flatness across K
is true by construction.**

No fix was applied and no code was refactored, per the audit-only instruction.

### Protected-file integrity

Hashed before and after the audit; byte-identical (see
`tuning/prerun_fixes/test_split_preflight.md` §Protected-file integrity for the
full table).
