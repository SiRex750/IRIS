"""Diagnostic: compare retrieve() ordering vs PageRank ordering.
Do NOT modify. Read-only probe for step 6.1."""
from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import iris.ingest as iris_ingest
import iris.query as iris_query

QUESTIONS = [
    "what is moving in the scene",
    "describe the main action",
    "what happens at the end",
]


def main() -> None:
    video_env = os.environ.get("IRIS_TEST_VIDEO", "mov_bbb.mp4")
    video_path = Path(video_env) if Path(video_env).is_absolute() else REPO_ROOT / video_env
    if not video_path.exists():
        print(f"ERROR: video not found at {video_path}")
        sys.exit(1)

    print(f"Ingesting: {video_path}")
    index = iris_ingest.ingest(str(video_path))
    config = iris_query._config_from_index(index, None)
    l2_retrieve_top_k = getattr(config, "l2_retrieve_top_k", 5)
    top_k = 8  # probe uses top-8 regardless of config

    graph = index._graph

    # Build global PageRank ordering once (descending)
    all_nodes = [graph.graph.nodes[n]["node_data"] for n in graph.graph.nodes]
    pr_sorted = sorted(all_nodes, key=lambda nd: nd.pagerank_score, reverse=True)
    pr_rank = {nd.frame_idx: rank + 1 for rank, nd in enumerate(pr_sorted)}
    n_nodes = len(all_nodes)

    match_count = 0

    print("\n===PHASE6_PROBE_BEGIN===")

    for q in QUESTIONS:
        emb = iris_query._embed_query(q, config)
        emb_norm = float(np.linalg.norm(emb))
        emb_type = "real_vector" if emb_norm > 1e-6 else "zero_fallback"
        print(f"\nQUESTION: {q!r}")
        print(f"  embedding_norm={emb_norm:.6f}  ({emb_type})")

        # RETRIEVE ordering — exact signature from query.py:176-180
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            retrieved_nodes = graph.retrieve(
                emb,
                query_action_score=index.index_action_score,
                top_k=top_k,
            )

        retrieve_frame_idxs = [nd.frame_idx for nd in retrieved_nodes]

        # PAGERANK ordering — top-8 by pagerank_score
        pr_top8 = pr_sorted[:top_k]
        pr_frame_idxs = [nd.frame_idx for nd in pr_top8]

        order_identical = retrieve_frame_idxs == pr_frame_idxs
        overlap = len(set(retrieve_frame_idxs) & set(pr_frame_idxs)) / top_k

        if order_identical:
            match_count += 1

        print(f"  RETRIEVE_TOPK:")
        for nd in retrieved_nodes:
            pr_r = pr_rank.get(nd.frame_idx, -1)
            print(f"    (frame={nd.frame_idx}, last_retrieval_score={nd.last_retrieval_score:.6f}, "
                  f"pagerank_score={nd.pagerank_score:.6f}, pagerank_rank={pr_r})")

        print(f"  PAGERANK_TOPK:")
        for nd in pr_top8:
            print(f"    (frame={nd.frame_idx}, pagerank_score={nd.pagerank_score:.6f})")

        print(f"  order_identical={order_identical}")
        print(f"  setoverlap@{top_k}={overlap:.4f}")

    print(f"\nSUMMARY: RETRIEVE order == PAGERANK order for {match_count}/3 questions, "
          f"N_nodes_in_graph = {n_nodes}")
    print("===PHASE6_PROBE_END===")


if __name__ == "__main__":
    main()
