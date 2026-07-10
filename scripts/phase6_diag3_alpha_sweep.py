"""Diagnostic 3: alpha sweep over (G_complete, G_temporal) x (5 alpha values) x (5 questions).
Read-only — no production files touched.
Fixed params: window_radius=2, alpha in [0.15, 0.30, 0.50, 0.70, 0.85]."""
from __future__ import annotations

import itertools
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
    "is there water or a stream",
    "show a wide landscape shot",
]
ALPHAS = [0.15, 0.30, 0.50, 0.70, 0.85]
WINDOW_RADIUS = 2
TOP_K = 8


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_seed(query_emb: np.ndarray | None, node_ids: list, graph) -> dict:
    raw: dict = {}
    for nid in node_ids:
        node = graph.graph.nodes[nid]["node_data"]
        sem = 0.0
        if query_emb is not None and node.embedding is not None:
            sem = cosine(query_emb, node.embedding)
        raw[nid] = max(0.0, sem)
    total = sum(raw.values())
    if total > 0.0:
        return {k: v / total for k, v in raw.items()}
    return {k: 1.0 / len(node_ids) for k in node_ids}


def top8_by_dict(d: dict) -> list:
    return sorted(d, key=lambda k: d[k], reverse=True)[:TOP_K]


def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    u = sa | sb
    return len(sa & sb) / len(u) if u else 1.0


def mean_pairwise_jaccard(top8_lists: list[list]) -> float:
    pairs = list(itertools.combinations(range(len(top8_lists)), 2))
    if not pairs:
        return 1.0
    return float(np.mean([jaccard(top8_lists[i], top8_lists[j]) for i, j in pairs]))


def mean_overlap(ppr_top8_lists: list[list], ref_lists: list[list]) -> float:
    vals = [len(set(p) & set(r)) / TOP_K for p, r in zip(ppr_top8_lists, ref_lists)]
    return float(np.mean(vals))


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
    uniform = 1.0 / n_nodes
    print(f"N={n_nodes}  uniform_baseline={uniform:.8f}")
    print(f"Fixed params: window_radius={WINDOW_RADIUS}  alphas={ALPHAS}")

    # ── Build G_complete ──────────────────────────────────────────────────
    G_complete = nx.Graph()
    for nid in node_ids:
        G_complete.add_node(nid)
    for u, v, data in prod_graph.graph.edges(data=True):
        G_complete.add_edge(u, v, weight=data["weight"])

    # ── Build G_temporal ──────────────────────────────────────────────────
    sorted_ids = sorted(node_ids)
    G_temporal = nx.Graph()
    for nid in sorted_ids:
        G_temporal.add_node(nid)
    for pos, u in enumerate(sorted_ids):
        for offset in range(1, WINDOW_RADIUS + 1):
            if pos + offset >= len(sorted_ids):
                break
            v = sorted_ids[pos + offset]
            w = prod_graph.graph[u][v]["weight"] if prod_graph.graph.has_edge(u, v) else 0.0
            G_temporal.add_edge(u, v, weight=w)

    print(f"G_complete edges: {G_complete.number_of_edges()}")
    print(f"G_temporal edges: {G_temporal.number_of_edges()}  "
          f"connected={nx.is_connected(G_temporal)}")

    # ── Per-question seeds and pure-seed top-8 ────────────────────────────
    seeds: list[dict] = []
    pure_seed_top8s: list[list] = []
    for q in QUESTIONS:
        emb = iris_query._embed_query(q, config)
        seed = build_seed(emb, node_ids, prod_graph)
        seeds.append(seed)
        pure_seed_top8s.append(top8_by_dict(seed))

    # Seed's own cross-question Jaccard (responsiveness ceiling)
    seed_xq_jaccard = mean_pairwise_jaccard(pure_seed_top8s)

    # ── Full grid computation ─────────────────────────────────────────────
    topologies = [("G_complete", G_complete), ("G_temporal", G_temporal)]
    # results[topo_name][alpha] = list of 5 ppr_top8s (one per question)
    results: dict[str, dict[float, list[list]]] = {
        topo: {a: [] for a in ALPHAS} for topo, _ in topologies
    }
    for topo_name, G in topologies:
        for alpha in ALPHAS:
            for qi, q in enumerate(QUESTIONS):
                seed = seeds[qi]
                pr = nx.pagerank(G, weight="weight", personalization=seed, alpha=alpha)
                results[topo_name][alpha].append(top8_by_dict(pr))

    # ── Output ────────────────────────────────────────────────────────────
    print("\n===DIAG3_BEGIN===")
    print(f"\nSEED cross-question Jaccard (responsiveness ceiling): {seed_xq_jaccard:.4f}")
    print(f"PURE_SEED top-8 per question:")
    for qi, q in enumerate(QUESTIONS):
        print(f"  Q{qi+1} {q!r}: {pure_seed_top8s[qi]}")

    dump_alphas = {0.15, 0.50}

    for topo_name, G in topologies:
        print(f"\n{'='*60}")
        print(f"TOPOLOGY: {topo_name}  (edges={G.number_of_edges()})")
        for alpha in ALPHAS:
            top8_lists = results[topo_name][alpha]  # list of 5 lists
            xq_j = mean_pairwise_jaccard(top8_lists)
            ov_seed = mean_overlap(top8_lists, pure_seed_top8s)
            ov_pureseed = mean_overlap(top8_lists, pure_seed_top8s)  # same ref; reported separately per spec

            print(f"\n  a={alpha:.2f}")
            print(f"    xQ_jaccard     = {xq_j:.4f}  (lower = more query-responsive)")
            print(f"    overlap_seed   = {ov_seed:.4f}  (higher = PPR tracks seed)")
            print(f"    overlap_pureseed = {ov_pureseed:.4f}  (~1.0 = graph adds nothing; ~0 = ignores seed)")

            if alpha in dump_alphas:
                print(f"    Top-8 sets per question:")
                for qi, q in enumerate(QUESTIONS):
                    print(f"      Q{qi+1} {q!r}:")
                    print(f"        PPR_TOP8:  {top8_lists[qi]}")
                    print(f"        SEED_TOP8: {pure_seed_top8s[qi]}")
                    print(f"        overlap:   {len(set(top8_lists[qi]) & set(pure_seed_top8s[qi])) / TOP_K:.4f}")

    # ── Compact summary grid ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY GRID  (rows = topology x alpha; lower xQ_jaccard = more query-responsive)")
    print(f"  seed_xQ_jaccard (ceiling) = {seed_xq_jaccard:.4f}")
    header = f"  {'topology+alpha':<28}  {'xQ_jaccard':>10}  {'overlap_seed':>12}  {'overlap_pureseed':>16}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for topo_name, _ in topologies:
        for alpha in ALPHAS:
            top8_lists = results[topo_name][alpha]
            xq_j = mean_pairwise_jaccard(top8_lists)
            ov_s = mean_overlap(top8_lists, pure_seed_top8s)
            label = f"{topo_name}  a={alpha:.2f}"
            print(f"  {label:<28}  {xq_j:>10.4f}  {ov_s:>12.4f}  {ov_s:>16.4f}")

    print("\n===DIAG3_END===")


if __name__ == "__main__":
    main()
