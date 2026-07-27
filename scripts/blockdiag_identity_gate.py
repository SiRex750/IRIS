"""Edge-identity gate: OLD (fully_connected + prune) vs NEW (block_diagonal).

Loads the cached VIRAT N=4,892 scene_sparse index (frames + scene_ids only,
no re-decode), builds two L2Asphodel graphs from the identical frame list:

  - Graph A (OLD): graph_edge_mode="fully_connected" -- materializes the full
    N(N-1)/2 pairs, then prunes cross-scene edges (today's scene_sparse
    ingest path, per scripts/virat_smoke_scenesparse.py).
  - Graph B (NEW): graph_edge_mode="block_diagonal" -- builds only intra-
    scene pairs directly, per eval_results/blockdiag_build_plan.md.

Asserts, per eval_results/blockdiag_build_plan.md section 6:
  1. identical node set
  2. identical node-PAIR set (no missing/extra edges, either direction)
  3. bit-identical (tolerance 0.0) weight / semantic_weight / motion_weight /
     temporal_weight / edge_type on every common edge
  4. edge_count matches sum(n_i*(n_i-1)//2) over scenes
  5. bit-identical PageRank score per node (secondary, insertion-order guard)
  6. identical PPR ranking order for a handful of seeded personalization
     vectors (secondary, insertion-order guard, task step 4)

This script does NOT change graph construction behavior for existing modes
-- it only exercises the new "block_diagonal" mode added to
_update_all_edge_weights. It writes nothing to any index cache.

Memory/timing of the two build paths in isolation (separate subprocesses,
to avoid conflating the two builds' allocations in a single process) is
measured by scripts/blockdiag_build_probe.py and is NOT part of this
script's job -- this script is the correctness gate only.
"""
from __future__ import annotations

import json
import sys
import copy
from pathlib import Path

REPO = Path(r"C:\Users\Siddanth Anil\IRIS")
sys.path.insert(0, str(REPO))

import numpy as np
import networkx as nx

import iris.ingest as iris_ingest

CACHE_PATH = REPO / "eval" / "data" / "virat" / "index_cache" / "VIRAT_S_040001_01_000448_001101"
OUT_JSON = REPO / "eval_results" / "blockdiag_identity_gate_result.json"
OUT_MD = REPO / "eval_results" / "blockdiag_identity_gate_result.md"

N_PPR_SEEDS = 5
PPR_TOP_K = 20  # compare rank order of the top-K nodes under each seed


def edge_key(u, v):
    return (u, v) if u <= v else (v, u)


def compare_graphs(graph_a: nx.Graph, graph_b: nx.Graph) -> dict:
    nodes_a = set(graph_a.nodes)
    nodes_b = set(graph_b.nodes)
    node_set_match = (nodes_a == nodes_b)

    edges_a = {edge_key(u, v) for u, v in graph_a.edges}
    edges_b = {edge_key(u, v) for u, v in graph_b.edges}
    only_in_a = edges_a - edges_b
    only_in_b = edges_b - edges_a
    edge_set_match = (len(only_in_a) == 0 and len(only_in_b) == 0)

    common = edges_a & edges_b
    mismatches = []
    fields = ["weight", "semantic_weight", "motion_weight", "temporal_weight", "edge_type"]
    for (u, v) in common:
        da = graph_a[u][v]
        db = graph_b[u][v]
        for f in fields:
            va = da.get(f)
            vb = db.get(f)
            if f == "edge_type":
                same = (va == vb)
            else:
                # Exact float equality -- tolerance 0.0, per plan section 6.
                same = (float(va) == float(vb))
            if not same:
                mismatches.append({
                    "u": int(u), "v": int(v), "field": f,
                    "value_a": va, "value_b": vb,
                })

    return {
        "node_count_a": len(nodes_a),
        "node_count_b": len(nodes_b),
        "node_set_match": node_set_match,
        "edge_count_a": len(edges_a),
        "edge_count_b": len(edges_b),
        "edges_only_in_a": len(only_in_a),
        "edges_only_in_b": len(only_in_b),
        "edges_only_in_a_sample": sorted(list(only_in_a))[:20],
        "edges_only_in_b_sample": sorted(list(only_in_b))[:20],
        "edge_set_match": edge_set_match,
        "common_edge_count": len(common),
        "field_mismatch_count": len(mismatches),
        "field_mismatches_sample": mismatches[:50],
    }


def compare_pagerank(graph_a: nx.Graph, graph_b: nx.Graph) -> dict:
    mismatches = []
    for nid in graph_a.nodes:
        pa = graph_a.nodes[nid]["node_data"].pagerank_score
        pb = graph_b.nodes[nid]["node_data"].pagerank_score
        if float(pa) != float(pb):
            mismatches.append({"node": int(nid), "pagerank_a": pa, "pagerank_b": pb})
    return {
        "pagerank_bit_identical": (len(mismatches) == 0),
        "pagerank_mismatch_count": len(mismatches),
        "pagerank_mismatches_sample": mismatches[:20],
    }


def compare_ppr_ranking(graph_a: nx.Graph, graph_b: nx.Graph, seed: int) -> dict:
    """Personalized PageRank with a random personalization vector (stand-in
    for a query-conditioned seed), compared for ranking-order identity."""
    rng = np.random.default_rng(seed)
    node_ids = sorted(graph_a.nodes)
    weights = rng.random(len(node_ids))
    weights = weights / weights.sum()
    personalization = {nid: float(w) for nid, w in zip(node_ids, weights)}

    pr_a = nx.pagerank(graph_a, weight="weight", personalization=personalization, alpha=0.85)
    pr_b = nx.pagerank(graph_b, weight="weight", personalization=personalization, alpha=0.85)

    order_a = sorted(node_ids, key=lambda n: (-pr_a[n], n))[:PPR_TOP_K]
    order_b = sorted(node_ids, key=lambda n: (-pr_b[n], n))[:PPR_TOP_K]

    exact_score_match = all(float(pr_a[n]) == float(pr_b[n]) for n in node_ids)

    return {
        "seed": seed,
        "top_k_order_match": (order_a == order_b),
        "top_k_order_a": order_a,
        "top_k_order_b": order_b,
        "exact_score_match_all_nodes": exact_score_match,
    }


def main() -> None:
    cache_file = Path(str(CACHE_PATH) + ".npz")
    if not cache_file.exists():
        print(f"FATAL: cache not found at {cache_file}", file=sys.stderr)
        sys.exit(1)

    idx = iris_ingest.load_index(CACHE_PATH)
    frames = idx.frames
    n_survivors = len(frames)
    print(f"Loaded {n_survivors} cached frames from {cache_file.name}")

    old_cfg = dict(idx.config_snapshot)
    old_cfg["graph_mode"] = "scene_sparse"
    old_cfg["graph_edge_mode"] = "fully_connected"

    new_cfg = copy.deepcopy(old_cfg)
    new_cfg["graph_edge_mode"] = "block_diagonal"

    print("Building Graph A (OLD: fully_connected + cross-scene prune)...")
    graph_a = iris_ingest._build_graph(frames, old_cfg)

    print("Building Graph B (NEW: block_diagonal, direct intra-scene build)...")
    graph_b = iris_ingest._build_graph(frames, new_cfg)

    print("Comparing edge sets / weights (tolerance 0.0)...")
    edge_report = compare_graphs(graph_a.graph, graph_b.graph)

    print("Comparing PageRank scores (bit-identical)...")
    pagerank_report = compare_pagerank(graph_a.graph, graph_b.graph)

    print("Comparing Personalized PageRank ranking order across seeded queries...")
    ppr_reports = [compare_ppr_ranking(graph_a.graph, graph_b.graph, seed=20260727 + i)
                   for i in range(N_PPR_SEEDS)]

    # Scene-size cross-check against the theoretical block-diagonal edge count.
    scene_ids = [fr.scene_id for fr in frames]
    per_scene_counts: dict = {}
    for s in scene_ids:
        if s < 0:
            continue
        per_scene_counts[s] = per_scene_counts.get(s, 0) + 1
    sum_c = sum(n * (n - 1) // 2 for n in per_scene_counts.values())

    gate_pass = (
        edge_report["node_set_match"]
        and edge_report["edge_set_match"]
        and edge_report["field_mismatch_count"] == 0
        and edge_report["edge_count_a"] == sum_c
        and edge_report["edge_count_b"] == sum_c
        and pagerank_report["pagerank_bit_identical"]
        and all(r["top_k_order_match"] and r["exact_score_match_all_nodes"] for r in ppr_reports)
    )

    result = {
        "outcome": "GATE_PASS" if gate_pass else "GATE_FAIL",
        "n_survivors": n_survivors,
        "num_scenes": len(per_scene_counts),
        "sum_c_scene_size_choose_2": sum_c,
        "edge_report": edge_report,
        "pagerank_report": pagerank_report,
        "ppr_ranking_reports": ppr_reports,
    }

    OUT_JSON.write_text(json.dumps(result, indent=2))

    md_lines = [
        "# Block-diagonal edge-identity gate result",
        "",
        f"**Outcome: {result['outcome']}**",
        "",
        f"- N survivors: {n_survivors}, scenes: {len(per_scene_counts)}",
        f"- sum(n_i*(n_i-1)//2) over scenes (theoretical block-diagonal edge count): {sum_c}",
        f"- Graph A (OLD fully_connected+prune) edge count: {edge_report['edge_count_a']}",
        f"- Graph B (NEW block_diagonal) edge count: {edge_report['edge_count_b']}",
        f"- Node set match: {edge_report['node_set_match']}",
        f"- Edge-pair set match: {edge_report['edge_set_match']} "
        f"(only-in-A: {edge_report['edges_only_in_a']}, only-in-B: {edge_report['edges_only_in_b']})",
        f"- Common edges compared: {edge_report['common_edge_count']}",
        f"- Field mismatches (weight/semantic/motion/temporal/edge_type), tolerance 0.0: "
        f"{edge_report['field_mismatch_count']}",
        f"- PageRank bit-identical across all nodes: {pagerank_report['pagerank_bit_identical']} "
        f"(mismatches: {pagerank_report['pagerank_mismatch_count']})",
        f"- PPR ranking order identical across {N_PPR_SEEDS} seeded personalization vectors: "
        f"{all(r['top_k_order_match'] for r in ppr_reports)}",
        f"- PPR exact score match (all nodes, all seeds): "
        f"{all(r['exact_score_match_all_nodes'] for r in ppr_reports)}",
        "",
    ]
    if not gate_pass:
        md_lines.append("## Mismatches (sample)")
        md_lines.append("")
        for m in edge_report["field_mismatches_sample"]:
            md_lines.append(f"- edge ({m['u']}, {m['v']}) field `{m['field']}`: "
                             f"A={m['value_a']!r} B={m['value_b']!r}")
        if edge_report["edges_only_in_a_sample"]:
            md_lines.append(f"- Edges only in A (sample): {edge_report['edges_only_in_a_sample']}")
        if edge_report["edges_only_in_b_sample"]:
            md_lines.append(f"- Edges only in B (sample): {edge_report['edges_only_in_b_sample']}")
        for r in ppr_reports:
            if not r["top_k_order_match"]:
                md_lines.append(f"- PPR seed {r['seed']}: top-{PPR_TOP_K} order mismatch")
                md_lines.append(f"  A: {r['top_k_order_a']}")
                md_lines.append(f"  B: {r['top_k_order_b']}")

    OUT_MD.write_text("\n".join(md_lines) + "\n")

    print(json.dumps(result, indent=2)[:4000])
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"\n=== GATE RESULT: {result['outcome']} ===")

    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
