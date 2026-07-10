"""Diagnostic 4: codec-discounted PPR seed vs semantic-only vs codec-only.
Read-only — no production files touched.

CAVEAT: packet_size is not carried onto nodes post-ingest. codec_conf is
derived from action_score, whose dominant channel is packet_size per
action_score.py. True packet_size carriage is a 6.2 production change;
this is the proxy test.

Fixed params: window_radius=2, alphas=[0.15, 0.30, 0.50].
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
ALPHAS = [0.15, 0.30, 0.50]
WINDOW_RADIUS = 2
TOP_K = 8
SEED_NAMES = ["S_sem", "S_codec", "S_codeconly"]


# ── helpers ────────────────────────────────────────────────────────────────

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


# ── per-video analysis ─────────────────────────────────────────────────────

def run_video(video_path: Path) -> None:
    import sys
    print(f"\n{'#'*70}")
    print(f"VIDEO: {video_path.name}")
    sys.stdout.flush()

    print(f"Running ingest...")
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
    # seeds_norm[seed_name][qi] = normalized seed dict
    # pure_tops[seed_name][qi]  = top-8 from seed alone (no graph)
    # fallbacks[seed_name][qi]  = bool
    seeds_norm: dict[str, list[dict]] = {s: [] for s in SEED_NAMES}
    pure_tops: dict[str, list[list]] = {s: [] for s in SEED_NAMES}
    fallbacks: dict[str, list[bool]] = {s: [] for s in SEED_NAMES}

    for q in QUESTIONS:
        emb = iris_query._embed_query(q, config)

        # S_sem
        raw_sem = {nid: max(0.0, cosine(emb, prod_graph.graph.nodes[nid]["node_data"].embedding)
                            if prod_graph.graph.nodes[nid]["node_data"].embedding is not None else 0.0)
                   for nid in node_ids}
        norm_sem, fb_sem = normalize_seed(raw_sem)
        seeds_norm["S_sem"].append(norm_sem)
        pure_tops["S_sem"].append(top8(norm_sem))
        fallbacks["S_sem"].append(fb_sem)

        # S_codec = S_sem * codec_conf (element-wise, before normalization)
        raw_codec = {nid: raw_sem[nid] * codec_conf[nid] for nid in node_ids}
        norm_codec, fb_codec = normalize_seed(raw_codec)
        seeds_norm["S_codec"].append(norm_codec)
        pure_tops["S_codec"].append(top8(norm_codec))
        fallbacks["S_codec"].append(fb_codec)

        # S_codeconly = codec_conf alone (query-independent)
        raw_codeconly = {nid: codec_conf[nid] for nid in node_ids}
        norm_codeconly, fb_codeconly = normalize_seed(raw_codeconly)
        seeds_norm["S_codeconly"].append(norm_codeconly)
        pure_tops["S_codeconly"].append(top8(norm_codeconly))
        fallbacks["S_codeconly"].append(fb_codeconly)

    # seed ceiling metrics (once per video)
    for sname in SEED_NAMES:
        xqj = mean_pairwise_jaccard(pure_tops[sname])
        fb_count = sum(fallbacks[sname])
        print(f"  seed_xQ_jaccard({sname}) = {xqj:.4f}  (fallbacks={fb_count}/5)")

    # ── PPR grid ────────────────────────────────────────────────────────────
    # ppr_tops[topo][sname][alpha][qi] = top-8 list
    ppr_tops: dict = {
        tname: {sname: {a: [] for a in ALPHAS} for sname in SEED_NAMES}
        for tname, _ in topologies
    }

    for tname, G in topologies:
        for sname in SEED_NAMES:
            for alpha in ALPHAS:
                for qi in range(len(QUESTIONS)):
                    seed = seeds_norm[sname][qi]
                    pr = nx.pagerank(G, weight="weight", personalization=seed, alpha=alpha)
                    ppr_tops[tname][sname][alpha].append(top8(pr))

    # ── output ──────────────────────────────────────────────────────────────
    dump_cases = {("G_temporal", "S_codec", 0.15), ("G_temporal", "S_codec", 0.50)}

    sem_pure = pure_tops["S_sem"]  # reference for overlap_S_sem_pureseed

    print("\n===DIAG4_BEGIN===")
    print(f"VIDEO: {video_path.name}  N={n_nodes}")

    for tname, G in topologies:
        print(f"\n{'='*60}")
        print(f"TOPOLOGY: {tname}")
        for sname in SEED_NAMES:
            for alpha in ALPHAS:
                top8_lists = ppr_tops[tname][sname][alpha]
                xqj = mean_pairwise_jaccard(top8_lists)
                ov_pure = mean_overlap(top8_lists, pure_tops[sname])
                ov_sem = mean_overlap(top8_lists, sem_pure)

                print(f"\n  {sname}  a={alpha:.2f}")
                print(f"    xQ_jaccard        = {xqj:.4f}")
                print(f"    overlap_pureseed   = {ov_pure:.4f}")
                print(f"    overlap_S_sem_pure = {ov_sem:.4f}")

                if (tname, sname, alpha) in dump_cases:
                    print(f"    Per-question top-8:")
                    for qi, q in enumerate(QUESTIONS):
                        print(f"      Q{qi+1} {q!r}")
                        print(f"        PPR_TOP8:  {top8_lists[qi]}")
                        print(f"        SEED_TOP8: {pure_tops[sname][qi]}")
                        ov_q = len(set(top8_lists[qi]) & set(pure_tops[sname][qi])) / TOP_K
                        print(f"        overlap:   {ov_q:.4f}")

    # ── summary grid ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY GRID — {video_path.name}  N={n_nodes}")
    print(f"  seed_xQ_jaccard  S_sem={mean_pairwise_jaccard(pure_tops['S_sem']):.4f}"
          f"  S_codec={mean_pairwise_jaccard(pure_tops['S_codec']):.4f}"
          f"  S_codeconly={mean_pairwise_jaccard(pure_tops['S_codeconly']):.4f}")
    hdr = f"  {'row':<36}  {'xQ_jacc':>8}  {'ov_pure':>8}  {'ov_S_sem':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tname, _ in topologies:
        for sname in SEED_NAMES:
            for alpha in ALPHAS:
                top8_lists = ppr_tops[tname][sname][alpha]
                xqj = mean_pairwise_jaccard(top8_lists)
                ov_pure = mean_overlap(top8_lists, pure_tops[sname])
                ov_sem = mean_overlap(top8_lists, sem_pure)
                label = f"{tname}  {sname}  a={alpha:.2f}"
                print(f"  {label:<36}  {xqj:>8.4f}  {ov_pure:>8.4f}  {ov_sem:>9.4f}")

    print("\n===DIAG4_END===")


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

    print(f"Fixed params: window_radius={WINDOW_RADIUS}  alphas={ALPHAS}  top_k={TOP_K}")

    for vpath in videos:
        try:
            run_video(vpath)
        except MemoryError as e:
            print(f"\nSKIPPED {vpath.name}: OOM — {e}")
        except Exception as e:
            print(f"\nSKIPPED {vpath.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
