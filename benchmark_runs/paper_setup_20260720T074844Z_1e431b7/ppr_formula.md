# PPR Formula (production, `iris/l2_asphodel.py::L2Asphodel.retrieve_ppr`)

```
raw_sem[nid]   = max(0, cosine(query_embedding, node.embedding))
sem_rank       = rank_percentile(raw_sem)                       # in [0, 1]
raw_codec[nid] = node.codec_conf                                # pre-computed at ingest
codec_rank     = rank_percentile(raw_codec)                     # in [0, 1]

seed_raw[nid]  = max(0, lambda_ * sem_rank[nid] + (1 - lambda_) * codec_rank[nid])
seed[nid]      = seed_raw[nid] / sum(seed_raw.values())         # falls back to uniform teleport
                                                                  # if sum == 0

scores = networkx.pagerank(graph, alpha=damping, personalization=seed)
top_k_nodes = sorted(scores, key=score desc, tie-break=node_id asc)[:top_k]
```

- `damping` = `ppr_damping` config field (PageRank alpha, default 0.5).
- `lambda_` = `ppr_lambda` config field (default 0.5). See `lambda_semantics.md` for endpoint proof.
- Exceptions caught: `nx.NetworkXError`, `ZeroDivisionError` -> unpersonalized PageRank fallback
  (`teleport_fallback=True`, recorded per-node in `retrieval_contributions`).
- `query_embedding` with near-zero norm -> `retrieve_ppr` returns `[]` immediately (SCENE-002),
  no fallback to unguided PageRank, to avoid degenerate uniform `sem_rank`.
- Called by default via `iris/scene_retrieval.py::retrieve_scene_sparse` (DESCEND branch, over a
  temporary induced subgraph that may include cross-scene edges) since `graph_mode="scene_sparse"`
  is the `IRISConfig` default; flat-graph PPR is the fallback when `graph_mode="flat"`.
