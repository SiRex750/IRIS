"""Diagnostic 6: re-run diag5 key cell with TRUE packet_size codec_conf.

Read-only after ingest — no iris/ files touched.

codec_conf normalization: rank_percentile(packet_size) over node set, scaled to [0.1, 1.0]:
  rank_pct_i in [0,1]  →  codec_conf_i = 0.1 + 0.9 * rank_pct_i
  (same rank_percentile function as diag5; kills dynamic-range mismatch)

Key cell reported: G_temporal and G_complete, lambda in {1.0, 0.5}, alpha in {0.15, 0.50},
same 5 questions as diag4/5.  Output between ===DIAG6_BEGIN===/===DIAG6_END===.
"""
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
ALPHAS = [0.15, 0.50]
KEY_LAMBDAS = [1.0, 0.5]
WINDOW_RADIUS = 2
TOP_K = 8


# ── helpers (verbatim from diag4/5) ────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize_seed(raw: dict) -> tuple[dict, bool]:
    total = sum(raw.values())
    if total > 0.0:
        return {k: v / total for k, v in raw.items()}, False
    n = len(raw)
    return {k: 1.0 / n for k in raw}, True


def top8(d: dict) -> list:
    return sorted(d, key=lambda k: d[k], reverse=True)[:TOP_K]


def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    u = sa | sb
    return len(sa & sb) / len(u) if u else 1.0


def mean_pairwise_jaccard(lists: list[list]) -> float:
    pairs = list(itertools.combinations(range(len(lists)), 2))
    if not pairs:
        return 1.0
    return float(np.mean([jaccard(lists[i], lists[j]) for i, j in pairs]))


def mean_overlap(ppr_lists: list[list], ref_lists: list[list]) -> float:
    return float(np.mean([len(set(p) & set(r)) / TOP_K for p, r in zip(ppr_lists, ref_lists)]))


def rank_percentile(values: dict) -> dict:
    """Average-rank ties; rank_i / (N-1) in [0,1]. Same as diag5."""
    keys = list(values.keys())
    n = len(keys)
    if n == 1:
        return {keys[0]: 0.5}
    vals = np.array([values[k] for k in keys], dtype=np.float64)
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and vals[order[j + 1]] == vals[order[j]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_pct = ranks / (n - 1)
    return {keys[idx]: float(rank_pct[idx]) for idx in range(n)}


# ── per-video analysis ─────────────────────────────────────────────────────

def run_video(video_path: Path) -> None:
    print(f"\n{'#'*70}")
    print(f"VIDEO: {video_path.name}")
    sys.stdout.flush()

    print("Running ingest...")
    sys.stdout.flush()
    index = iris_ingest.ingest(str(video_path))
    config = iris_query._config_from_index(index, None)
    prod_graph = index._graph

    node_ids = list(prod_graph.graph.nodes)
    n_nodes = len(node_ids)
    print(f"N_nodes = {n_nodes}")

    # ── true packet_size from node ──────────────────────────────────────────
    raw_ps = {
        nid: prod_graph.graph.nodes[nid]["node_data"].packet_size
        for nid in node_ids
    }
    ps_arr = np.array(list(raw_ps.values()))
    print(f"packet_size distribution: min={ps_arr.min():.1f}  max={ps_arr.max():.1f}  "
          f"mean={ps_arr.mean():.1f}  std={ps_arr.std():.1f}")

    # ── pict_type distribution ──────────────────────────────────────────────
    pict_counts: dict[str, int] = {}
    for fr in index.frames:
        pict_counts[fr.pict_type] = pict_counts.get(fr.pict_type, 0) + 1
    print(f"pict_type counts: {dict(sorted(pict_counts.items()))}")

    # ── codec_conf from true packet_size rank percentile ───────────────────
    ps_rank = rank_percentile(raw_ps)
    codec_conf = {nid: 0.1 + 0.9 * ps_rank[nid] for nid in node_ids}
    cc_arr = np.array(list(codec_conf.values()))
    print(f"codec_conf(rank-ps) distribution: min={cc_arr.min():.4f}  max={cc_arr.max():.4f}  "
          f"mean={cc_arr.mean():.4f}  std={cc_arr.std():.4f}")

    # ── I-frame confound check ──────────────────────────────────────────────
    # Build frame_idx -> pict_type map from FrameRecords
    pict_by_idx = {fr.frame_idx: fr.pict_type for fr in index.frames}
    # Base rate of I-frames in the node set
    n_i = sum(1 for nid in node_ids if pict_by_idx.get(nid, "?") == "I")
    base_rate_i = n_i / n_nodes if n_nodes > 0 else 0.0
    # Top-8 by codec_conf
    top8_codec = sorted(node_ids, key=lambda nid: codec_conf[nid], reverse=True)[:8]
    n_i_top8 = sum(1 for nid in top8_codec if pict_by_idx.get(nid, "?") == "I")
    frac_i_top8 = n_i_top8 / 8 if top8_codec else 0.0
    print(f"I-frame confound: base_rate={base_rate_i:.3f}  "
          f"frac_I_in_top8_codec={frac_i_top8:.3f}  "
          f"(top8 node ids: {top8_codec}  pict_types: {[pict_by_idx.get(nid,'?') for nid in top8_codec]})")

    # ── build throwaway graphs ──────────────────────────────────────────────
    G_complete = nx.Graph()
    for nid in node_ids:
        G_complete.add_node(nid)
    for u, v, data in prod_graph.graph.edges(data=True):
        G_complete.add_edge(u, v, weight=data["weight"])

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

    topologies = [("G_complete", G_complete), ("G_temporal", G_temporal)]

    # ── per-question seeds ──────────────────────────────────────────────────
    codec_rank = rank_percentile(raw_ps)  # same as ps_rank

    seeds_norm: dict[float, list[dict]] = {lam: [] for lam in KEY_LAMBDAS}
    pure_tops: dict[float, list[list]] = {lam: [] for lam in KEY_LAMBDAS}

    for q in QUESTIONS:
        emb = iris_query._embed_query(q, config)
        raw_sem = {
            nid: max(0.0, cosine(emb, prod_graph.graph.nodes[nid]["node_data"].embedding)
                     if prod_graph.graph.nodes[nid]["node_data"].embedding is not None else 0.0)
            for nid in node_ids
        }
        sem_rank = rank_percentile(raw_sem)

        for lam in KEY_LAMBDAS:
            blended = {nid: lam * sem_rank[nid] + (1.0 - lam) * codec_rank[nid] for nid in node_ids}
            norm_b, _ = normalize_seed(blended)
            seeds_norm[lam].append(norm_b)
            pure_tops[lam].append(top8(norm_b))

    # seed-level xQ_jaccard
    print(f"\nSeed xQ_jaccard (true packet_size rank, no graph):")
    for lam in KEY_LAMBDAS:
        xqj = mean_pairwise_jaccard(pure_tops[lam])
        print(f"  lambda={lam:.1f}: seed_xQ_jaccard = {xqj:.4f}")

    # ── PPR grid ────────────────────────────────────────────────────────────
    ppr_tops: dict = {
        tname: {lam: {a: [] for a in ALPHAS} for lam in KEY_LAMBDAS}
        for tname, _ in topologies
    }
    for tname, G in topologies:
        for lam in KEY_LAMBDAS:
            for alpha in ALPHAS:
                for qi in range(len(QUESTIONS)):
                    seed = seeds_norm[lam][qi]
                    pr = nx.pagerank(G, weight="weight", personalization=seed, alpha=alpha)
                    ppr_tops[tname][lam][alpha].append(top8(pr))

    sem_pure = pure_tops[1.0]

    # ── output ──────────────────────────────────────────────────────────────
    print(f"\n===DIAG6_BEGIN===")
    print(f"VIDEO: {video_path.name}  N={n_nodes}")
    print(f"codec_conf_source: true packet_size (rank_percentile -> [0.1, 1.0])")

    for tname, G in topologies:
        print(f"\n{'='*60}")
        print(f"TOPOLOGY: {tname}")
        for lam in KEY_LAMBDAS:
            for alpha in ALPHAS:
                top8_lists = ppr_tops[tname][lam][alpha]
                xqj    = mean_pairwise_jaccard(top8_lists)
                ov_sem = mean_overlap(top8_lists, sem_pure)
                print(f"\n  lambda={lam:.1f}  a={alpha:.2f}")
                print(f"    xQ_jaccard         = {xqj:.4f}")
                print(f"    overlap_S_sem_pure = {ov_sem:.4f}")

    # key cell summary
    print(f"\n{'='*60}")
    print(f"SUMMARY GRID -- {video_path.name}  N={n_nodes}")
    print(f"  seed_xQ_jaccard: "
          f"lambda=1.0: {mean_pairwise_jaccard(pure_tops[1.0]):.4f}  "
          f"lambda=0.5: {mean_pairwise_jaccard(pure_tops[0.5]):.4f}")
    hdr = f"  {'row':<40}  {'xQ_jacc':>8}  {'ov_sem':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tname, _ in topologies:
        for lam in KEY_LAMBDAS:
            for alpha in ALPHAS:
                top8_lists = ppr_tops[tname][lam][alpha]
                xqj    = mean_pairwise_jaccard(top8_lists)
                ov_sem = mean_overlap(top8_lists, sem_pure)
                label = f"{tname}  lam={lam:.1f}  a={alpha:.2f}"
                print(f"  {label:<40}  {xqj:>8.4f}  {ov_sem:>8.4f}")

    print(f"\nKEY CELL (lambda=0.5) -- {video_path.name}")
    for tname, _ in topologies:
        for alpha in ALPHAS:
            top8_lists = ppr_tops[tname][0.5][alpha]
            xqj    = mean_pairwise_jaccard(top8_lists)
            ov_sem = mean_overlap(top8_lists, sem_pure)
            print(f"  {tname}  a={alpha:.2f}:  xQ_jaccard={xqj:.4f}  overlap_S_sem_pure={ov_sem:.4f}")

    print(f"\n===DIAG6_END===")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    vid1_env = os.environ.get("IRIS_TEST_VIDEO", "mov_bbb.mp4")
    vid1 = Path(vid1_env) if Path(vid1_env).is_absolute() else REPO_ROOT / vid1_env
    vid2 = REPO_ROOT / "videoplayback.mp4"

    # videoplayback first (larger, more nodes, more signal)
    videos = []
    for v in [vid2, vid1]:
        if v.exists():
            videos.append(v)
        else:
            print(f"SKIP: {v} not found")

    if not videos:
        print("No videos found. Exiting.")
        sys.exit(1)

    print(f"Fixed params: window_radius={WINDOW_RADIUS}  alphas={ALPHAS}  "
          f"lambdas={KEY_LAMBDAS}  top_k={TOP_K}")
    print(f"codec_conf_source: true packet_size via AsphodelNode.packet_size")
    print(f"normalization: rank_percentile(packet_size) -> 0.1 + 0.9 * rank_pct")

    for vpath in videos:
        try:
            run_video(vpath)
        except MemoryError as e:
            print(f"\nSKIPPED {vpath.name}: OOM -- {e}")
        except Exception as e:
            import traceback
            print(f"\nSKIPPED {vpath.name}: {type(e).__name__}: {e}")
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
