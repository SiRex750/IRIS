"""Diagnostic 2: seed discriminability vs graph density.
Read-only — does not modify any production module or graph.
Fixed params: window_radius=2, damping=0.85."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import networkx as nx
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
WINDOW_RADIUS = 2
DAMPING = 0.85
TOP_K = 8


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_seed(query_emb: np.ndarray | None, node_ids: list, graph: object) -> tuple[dict, bool]:
    """Exact replica of retrieve_ppr's personalization logic."""
    raw: dict = {}
    for nid in node_ids:
        node = graph.graph.nodes[nid]["node_data"]
        sem = 0.0
        if query_emb is not None and node.embedding is not None:
            sem = cosine(query_emb, node.embedding)
        raw[nid] = max(0.0, sem)
    total = sum(raw.values())
    if total > 0.0:
        return {k: v / total for k, v in raw.items()}, False
    return {k: 1.0 / len(node_ids) for k in node_ids}, True


def ppr_stats(scores: list[float], label: str) -> None:
    arr = np.array(scores, dtype=np.float64)
    sorted_desc = np.sort(arr)[::-1]
    r1 = float(sorted_desc[0]) if len(sorted_desc) >= 1 else 0.0
    r8 = float(sorted_desc[7]) if len(sorted_desc) >= 8 else float(sorted_desc[-1])
    print(f"  {label}: min={arr.min():.8f}  max={arr.max():.8f}  "
          f"mean={arr.mean():.8f}  std={arr.std():.8f}  "
          f"rank1-rank8_gap={r1 - r8:.8f}")


def top8_by_pr(pr: dict, label: str) -> list:
    ranked = sorted(pr, key=lambda k: pr[k], reverse=True)[:TOP_K]
    print(f"  {label}_TOP8: {ranked}")
    return ranked


def main() -> None:
    video_env = os.environ.get("IRIS_TEST_VIDEO", "mov_bbb.mp4")
    video_path = Path(video_env) if Path(video_env).is_absolute() else REPO_ROOT / video_env
    if not video_path.exists():
        print(f"ERROR: video not found at {video_path}")
        sys.exit(1)

    print(f"Ingesting: {video_path}")
    index = iris_ingest.ingest(str(video_path))
    config = iris_query._config_from_index(index, None)
    prod_graph = index._graph

    node_ids = list(prod_graph.graph.nodes)
    n_nodes = len(node_ids)
    uniform_baseline = 1.0 / n_nodes
    print(f"N={n_nodes}  uniform_baseline={uniform_baseline:.8f}")
    print(f"Fixed params: window_radius={WINDOW_RADIUS}, damping={DAMPING}")

    # ── Build G_complete (copy of production graph) ───────────────────────
    G_complete = nx.Graph()
    for nid in node_ids:
        G_complete.add_node(nid)
    for u, v, data in prod_graph.graph.edges(data=True):
        G_complete.add_edge(u, v, weight=data["weight"])

    # ── Build G_temporal (±window_radius temporal edges only) ─────────────
    sorted_ids = sorted(node_ids)   # frame_idx order
    G_temporal = nx.Graph()
    for nid in sorted_ids:
        G_temporal.add_node(nid)
    temporal_edge_count = 0
    for pos, u in enumerate(sorted_ids):
        for offset in range(1, WINDOW_RADIUS + 1):
            if pos + offset >= len(sorted_ids):
                break
            v = sorted_ids[pos + offset]
            if prod_graph.graph.has_edge(u, v):
                w = prod_graph.graph[u][v]["weight"]
            else:
                w = 0.0
            G_temporal.add_edge(u, v, weight=w)
            temporal_edge_count += 1

    is_connected = nx.is_connected(G_temporal)
    print(f"G_complete edges: {G_complete.number_of_edges()}")
    print(f"G_temporal edges: {temporal_edge_count}  connected={is_connected}")

    print("\n===DIAG2_BEGIN===")

    ppr_top8_temporal_by_q: list[list] = []

    for q in QUESTIONS:
        emb = iris_query._embed_query(q, config)
        seed, fallback = build_seed(emb, node_ids, prod_graph)

        seed_values = list(seed.values())
        seed_arr = np.array(seed_values, dtype=np.float64)
        seed_sorted_ids = sorted(seed, key=lambda k: seed[k], reverse=True)[:TOP_K]
        seed_top8_vals = [(k, seed[k]) for k in seed_sorted_ids]

        seed_sorted_desc = np.sort(seed_arr)[::-1]
        s_r1 = float(seed_sorted_desc[0]) if len(seed_sorted_desc) >= 1 else 0.0
        s_r8 = float(seed_sorted_desc[7]) if len(seed_sorted_desc) >= 8 else float(seed_sorted_desc[-1])

        print(f"\nQUESTION: {q!r}  (teleport_fallback={fallback})")
        print(f"  SEED distribution (N={n_nodes}):")
        print(f"    min={seed_arr.min():.8f}  max={seed_arr.max():.8f}  "
              f"mean={seed_arr.mean():.8f}  std={seed_arr.std():.8f}  "
              f"rank1-rank8_gap={s_r1 - s_r8:.8f}")
        print(f"  SEED_TOP8: {seed_top8_vals}")

        # G_complete PPR
        pr_complete = nx.pagerank(G_complete, weight="weight", personalization=seed, alpha=DAMPING)
        top8_complete = top8_by_pr(pr_complete, "G_complete_PPR")
        ppr_stats(list(pr_complete.values()), "G_complete_PPR_dist")
        overlap_complete = len(set(seed_sorted_ids) & set(top8_complete)) / TOP_K
        print(f"  setoverlap@8(SEED_TOP8, G_complete_PPR_TOP8) = {overlap_complete:.4f}")

        # G_temporal PPR
        pr_temporal = nx.pagerank(G_temporal, weight="weight", personalization=seed, alpha=DAMPING)
        top8_temporal = top8_by_pr(pr_temporal, "G_temporal_PPR")
        ppr_stats(list(pr_temporal.values()), "G_temporal_PPR_dist")
        overlap_temporal = len(set(seed_sorted_ids) & set(top8_temporal)) / TOP_K
        print(f"  setoverlap@8(SEED_TOP8, G_temporal_PPR_TOP8) = {overlap_temporal:.4f}")

        ppr_top8_temporal_by_q.append(top8_temporal)

    print("\n  G_temporal PPR_TOP8 across all 3 questions:")
    for i, (q, top8) in enumerate(zip(QUESTIONS, ppr_top8_temporal_by_q)):
        print(f"    Q{i+1}: {top8}")
    same_all = all(ppr_top8_temporal_by_q[0] == t for t in ppr_top8_temporal_by_q)
    print(f"  G_temporal top-8 identical across all 3 questions: {same_all}")

    print("\n===DIAG2_END===")


if __name__ == "__main__":
    main()
