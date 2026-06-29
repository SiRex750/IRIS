"""Diagnostic: compare legacy retrieve() vs retrieve_ppr() spread on a real ingest.
Read-only probe for step 6.1 / 6.2 analysis. Do NOT modify."""
from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

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
    graph = index._graph

    all_nodes = [graph.graph.nodes[n]["node_data"] for n in graph.graph.nodes]
    n_nodes = len(all_nodes)
    uniform_baseline = 1.0 / n_nodes if n_nodes > 0 else 0.0

    print(f"\nN_nodes_in_graph = {n_nodes},  uniform_baseline = {uniform_baseline:.6f}")

    print("\n===PPR_SPREAD_BEGIN===")

    for q in QUESTIONS:
        emb = iris_query._embed_query(q, config)
        emb_norm = float(np.linalg.norm(emb))

        print(f"\nQUESTION: {q!r}")
        print(f"  embedding_norm={emb_norm:.6f}")

        # ── Legacy retrieve() top-8 ────────────────────────────────────────
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            legacy_nodes = graph.retrieve(
                emb,
                query_action_score=index.index_action_score,
                top_k=8,
            )
        legacy_frame_idxs = [nd.frame_idx for nd in legacy_nodes]

        # ── PPR retrieve_ppr() — all nodes to get distribution ─────────────
        ppr_all_nodes = graph.retrieve_ppr(emb, top_k=n_nodes, damping=0.85)
        teleport_fallback = (
            ppr_all_nodes[0].retrieval_contributions.get("teleport_fallback", False)
            if ppr_all_nodes else True
        )
        ppr_scores = np.array([nd.last_retrieval_score for nd in ppr_all_nodes], dtype=np.float64)
        ppr_top8 = ppr_all_nodes[:8]
        ppr_frame_idxs = [nd.frame_idx for nd in ppr_top8]

        rank1_score = float(ppr_scores[0]) if len(ppr_scores) >= 1 else 0.0
        rank8_score = float(ppr_scores[7]) if len(ppr_scores) >= 8 else float(ppr_scores[-1])

        print(f"  teleport_fallback={teleport_fallback}")
        print(f"  PPR score distribution (all {n_nodes} nodes):")
        print(f"    min  = {float(ppr_scores.min()):.8f}")
        print(f"    max  = {float(ppr_scores.max()):.8f}")
        print(f"    mean = {float(ppr_scores.mean()):.8f}")
        print(f"    std  = {float(ppr_scores.std()):.8f}")
        print(f"    rank1_score  = {rank1_score:.8f}")
        print(f"    rank8_score  = {rank8_score:.8f}")
        print(f"    rank1-rank8_gap = {rank1_score - rank8_score:.8f}")
        print(f"    uniform_baseline = {uniform_baseline:.8f}")

        print(f"  PPR_TOPK8:")
        for nd in ppr_top8:
            print(f"    (frame={nd.frame_idx}, ppr_score={nd.last_retrieval_score:.8f})")

        overlap = len(set(legacy_frame_idxs) & set(ppr_frame_idxs)) / 8
        print(f"  LEGACY_TOP8 frames: {legacy_frame_idxs}")
        print(f"  PPR_TOP8   frames:  {ppr_frame_idxs}")
        print(f"  setoverlap@8(legacy, ppr) = {overlap:.4f}")

    print("\n===PPR_SPREAD_END===")


if __name__ == "__main__":
    main()
