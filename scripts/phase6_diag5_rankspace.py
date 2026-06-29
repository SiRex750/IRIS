"""Diagnostic 5: rank-space additive combination of semantic and codec-confidence signals.
Read-only — no production files touched.

CAVEAT: codec_conf is derived from action_score (dominant channel is packet_size per
action_score.py). True packet_size carriage is a 6.2 production change; this is the proxy test.

Diag4 finding: sem * codec_conf collapses to ~codec_conf because codec_conf's dynamic range
(std 0.2-0.33) swamps CLIP cosine spread (std <2%). Rank-space addition tests whether
normalizing both signals to rank percentiles before combining restores query-responsiveness
while still letting codec reshape the ranking.

Fixed params: window_radius=2, alphas=[0.15, 0.50], lambdas=[1.0, 0.7, 0.5, 0.3, 0.0], top_k=8.
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
LAMBDAS = [1.0, 0.7, 0.5, 0.3, 0.0]
WINDOW_RADIUS = 2
TOP_K = 8


# ── helpers (verbatim from diag4) ──────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize_seed(raw: dict) -> tuple[dict, bool]:
    """ReLU (already applied before call) + sum-normalize. Uniform fallback if all-zero."""
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


# ── rank-percentile helper ─────────────────────────────────────────────────

def rank_percentile(values: dict) -> dict:
    """Convert dict of float values to rank percentiles in [0, 1].
    Uses average rank for ties. rank_i / (N-1) so max=1.0, min=0.0.
    N=1 edge case: all 0.5.
    """
    keys = list(values.keys())
    n = len(keys)
    if n == 1:
        return {keys[0]: 0.5}
    vals = np.array([values[k] for k in keys], dtype=np.float64)
    # scipy-style average rank via argsort trick
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and vals[order[j + 1]] == vals[order[j]]:
            j += 1
        avg_rank = (i + j) / 2.0  # 0-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    # normalize to [0, 1]
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

    # ── codec_conf ─────────────────────────────────────────────────────────
    action_scores = {
        nid: prod_graph.graph.nodes[nid]["node_data"].action_score
        for nid in node_ids
    }
    as_vals = list(action_scores.values())
    as_min, as_max = min(as_vals), max(as_vals)
    as_range = as_max - as_min
    codec_conf: dict = {}
    for nid in node_ids:
        if as_range == 0.0:
            as_norm = 1.0
        else:
            as_norm = (action_scores[nid] - as_min) / as_range
        codec_conf[nid] = 0.1 + 0.9 * as_norm

    cc_arr = np.array(list(codec_conf.values()))
    print(f"codec_conf distribution: min={cc_arr.min():.4f}  max={cc_arr.max():.4f}  "
          f"mean={cc_arr.mean():.4f}  std={cc_arr.std():.4f}")

    # codec rank percentiles (fixed per video, query-independent)
    codec_rank = rank_percentile(codec_conf)

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

    # ── per-question rank-space seeds ──────────────────────────────────────
    # seeds_norm[lam][qi]  = normalized seed dict
    # pure_tops[lam][qi]   = top-8 from seed alone (no graph)
    # fallbacks[lam][qi]   = bool
    seeds_norm: dict[float, list[dict]] = {lam: [] for lam in LAMBDAS}
    pure_tops: dict[float, list[list]] = {lam: [] for lam in LAMBDAS}
    fallbacks: dict[float, list[bool]] = {lam: [] for lam in LAMBDAS}

    for q in QUESTIONS:
        emb = iris_query._embed_query(q, config)

        # raw semantic cosine per node
        raw_sem = {
            nid: max(0.0, cosine(emb, prod_graph.graph.nodes[nid]["node_data"].embedding)
                     if prod_graph.graph.nodes[nid]["node_data"].embedding is not None else 0.0)
            for nid in node_ids
        }

        # sem rank percentiles for this query
        sem_rank = rank_percentile(raw_sem)

        for lam in LAMBDAS:
            blended = {
                nid: lam * sem_rank[nid] + (1.0 - lam) * codec_rank[nid]
                for nid in node_ids
            }
            # ReLU already satisfied (both ranks >= 0), then normalize
            norm_b, fb = normalize_seed(blended)
            seeds_norm[lam].append(norm_b)
            pure_tops[lam].append(top8(norm_b))
            fallbacks[lam].append(fb)

    # seed ceiling metrics for key lambdas
    print(f"\nSeed xQ_jaccard (pure-seed, no graph):")
    for lam in [1.0, 0.5, 0.0]:
        xqj = mean_pairwise_jaccard(pure_tops[lam])
        fb_count = sum(fallbacks[lam])
        print(f"  lambda={lam:.1f}: seed_xQ_jaccard = {xqj:.4f}  (fallbacks={fb_count}/5)")

    # ── PPR grid ────────────────────────────────────────────────────────────
    # ppr_tops[tname][lam][alpha][qi] = top-8 list
    ppr_tops: dict = {
        tname: {lam: {a: [] for a in ALPHAS} for lam in LAMBDAS}
        for tname, _ in topologies
    }

    for tname, G in topologies:
        for lam in LAMBDAS:
            for alpha in ALPHAS:
                for qi in range(len(QUESTIONS)):
                    seed = seeds_norm[lam][qi]
                    pr = nx.pagerank(G, weight="weight", personalization=seed, alpha=alpha)
                    ppr_tops[tname][lam][alpha].append(top8(pr))

    # ── detailed dump: (G_temporal, lam=0.5, alpha=0.15) ───────────────────
    dump_tname, dump_lam, dump_alpha = "G_temporal", 0.5, 0.15
    print(f"\nDUMP ({dump_tname}, lambda={dump_lam}, a={dump_alpha}):")
    for qi, q in enumerate(QUESTIONS):
        ppr_t = ppr_tops[dump_tname][dump_lam][dump_alpha][qi]
        seed_t = pure_tops[dump_lam][qi]
        sem_t  = pure_tops[1.0][qi]
        cod_t  = pure_tops[0.0][qi]
        print(f"  Q{qi+1} {q!r}")
        print(f"    PPR_TOP8:        {ppr_t}")
        print(f"    SEED_TOP8(0.5):  {seed_t}")
        print(f"    SEED_TOP8(sem):  {sem_t}")
        print(f"    SEED_TOP8(cod):  {cod_t}")
        ov_sem = len(set(ppr_t) & set(sem_t)) / TOP_K
        ov_cod = len(set(ppr_t) & set(cod_t)) / TOP_K
        print(f"    overlap_sem={ov_sem:.4f}  overlap_cod={ov_cod:.4f}")

    # ── main output ─────────────────────────────────────────────────────────
    sem_pure  = pure_tops[1.0]   # pure-semantic-rank reference
    codec_pure = pure_tops[0.0]  # pure-codec-rank reference

    print(f"\n===DIAG5_BEGIN===")
    print(f"VIDEO: {video_path.name}  N={n_nodes}")

    for tname, G in topologies:
        print(f"\n{'='*60}")
        print(f"TOPOLOGY: {tname}")
        for lam in LAMBDAS:
            for alpha in ALPHAS:
                top8_lists = ppr_tops[tname][lam][alpha]
                xqj      = mean_pairwise_jaccard(top8_lists)
                ov_sem   = mean_overlap(top8_lists, sem_pure)
                ov_codec = mean_overlap(top8_lists, codec_pure)

                print(f"\n  lambda={lam:.1f}  a={alpha:.2f}")
                print(f"    xQ_jaccard         = {xqj:.4f}")
                print(f"    overlap_S_sem_pure = {ov_sem:.4f}")
                print(f"    overlap_codec_pure = {ov_codec:.4f}")

    # ── summary grid ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY GRID -- {video_path.name}  N={n_nodes}")
    print(f"  seed_xQ_jaccard: "
          f"lambda=1.0: {mean_pairwise_jaccard(pure_tops[1.0]):.4f}  "
          f"lambda=0.5: {mean_pairwise_jaccard(pure_tops[0.5]):.4f}  "
          f"lambda=0.0: {mean_pairwise_jaccard(pure_tops[0.0]):.4f}")
    hdr = f"  {'row':<40}  {'xQ_jacc':>8}  {'ov_sem':>8}  {'ov_codec':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tname, _ in topologies:
        for lam in LAMBDAS:
            for alpha in ALPHAS:
                top8_lists = ppr_tops[tname][lam][alpha]
                xqj      = mean_pairwise_jaccard(top8_lists)
                ov_sem   = mean_overlap(top8_lists, sem_pure)
                ov_codec = mean_overlap(top8_lists, codec_pure)
                label = f"{tname}  lam={lam:.1f}  a={alpha:.2f}"
                print(f"  {label:<40}  {xqj:>8.4f}  {ov_sem:>8.4f}  {ov_codec:>9.4f}")

    # ── diagnostic cell: lambda=0.5 both alpha both topo ───────────────────
    print(f"\nKEY CELL (lambda=0.5) -- {video_path.name}")
    for tname, _ in topologies:
        for alpha in ALPHAS:
            top8_lists = ppr_tops[tname][0.5][alpha]
            xqj    = mean_pairwise_jaccard(top8_lists)
            ov_sem = mean_overlap(top8_lists, sem_pure)
            print(f"  {tname}  a={alpha:.2f}:  xQ_jaccard={xqj:.4f}  overlap_S_sem_pure={ov_sem:.4f}")

    print(f"\n===DIAG5_END===")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    vid1_env = os.environ.get("IRIS_TEST_VIDEO", "mov_bbb.mp4")
    vid1 = Path(vid1_env) if Path(vid1_env).is_absolute() else REPO_ROOT / vid1_env
    vid2 = REPO_ROOT / "videoplayback.mp4"

    videos = []
    for v in [vid1, vid2]:
        if v.exists():
            videos.append(v)
        else:
            print(f"SKIP: {v} not found")

    if not videos:
        print("No videos found. Exiting.")
        sys.exit(1)

    print(f"Fixed params: window_radius={WINDOW_RADIUS}  alphas={ALPHAS}  "
          f"lambdas={LAMBDAS}  top_k={TOP_K}")

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
