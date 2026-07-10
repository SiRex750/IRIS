"""Diagnostic 7: per-pict_type normalization to remove I-frame confound.

Read-only after ingest. No iris/ files touched.

Three codec_conf variants (each -> 0.1 + 0.9 * rank_pct):
  C_raw    : rank-percentile of raw packet_size (confounded baseline, diag6 sanity check)
  C_bytype : rank-percentile of packet_size within each pict_type group separately;
             groups with <2 members get neutral mid-rank 0.5
  C_proxy  : rank-percentile of action_score (diag5 anchor, partially de-confounded)

Lambda fixed at 0.5, alpha in {0.15, 0.50}, G_complete + G_temporal, same 5 questions.
Primary: videoplayback.mp4 (N=220, I-frame confound proven in diag6).
"""
from __future__ import annotations

import itertools
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
LAMBDA = 0.5
WINDOW_RADIUS = 2
TOP_K = 8
VIDEOS = ["videoplayback.mp4"]


# ── helpers (verbatim from diag5/6) ────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize_seed(raw: dict) -> dict:
    total = sum(raw.values())
    if total > 0.0:
        return {k: v / total for k, v in raw.items()}
    n = len(raw)
    return {k: 1.0 / n for k in raw}


def top_k(d: dict) -> list:
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


def mean_overlap(a_lists: list[list], b_lists: list[list]) -> float:
    return float(np.mean([len(set(a) & set(b)) / TOP_K for a, b in zip(a_lists, b_lists)]))


def rank_percentile(values: dict) -> dict:
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
        for idx in range(i, j + 1):
            ranks[order[idx]] = avg_rank
        i = j + 1
    rank_pct = ranks / (n - 1)
    return {keys[idx]: float(rank_pct[idx]) for idx in range(n)}


def rank_percentile_bytype(ps_map: dict, pict_map: dict) -> tuple[dict, dict]:
    """
    Compute rank_percentile of packet_size within each pict_type group.
    Groups with <2 members get mid-rank 0.5.
    Returns (rank_pct_by_node, notes_by_type).
    """
    groups: dict[str, list] = {}
    for nid, ps in ps_map.items():
        pt = pict_map.get(nid, "?")
        groups.setdefault(pt, []).append(nid)

    result: dict = {}
    notes: dict = {}
    for pt, nids in groups.items():
        if len(nids) < 2:
            for nid in nids:
                result[nid] = 0.5
            notes[pt] = f"<2 members ({len(nids)}), assigned mid-rank 0.5"
        else:
            sub = {nid: ps_map[nid] for nid in nids}
            rp = rank_percentile(sub)
            result.update(rp)
            notes[pt] = f"{len(nids)} members"
    return result, notes


def codec_conf_from_rank(rank_pct: dict) -> dict:
    return {nid: 0.1 + 0.9 * rank_pct[nid] for nid in rank_pct}


# ── main per-video logic ────────────────────────────────────────────────────

def run_video(video_path: Path) -> None:
    print(f"\n{'#'*70}")
    print(f"VIDEO: {video_path.name}")
    sys.stdout.flush()

    print("Ingesting...")
    sys.stdout.flush()
    index = iris_ingest.ingest(str(video_path))
    config = iris_query._config_from_index(index, None)
    prod_graph = index._graph

    node_ids = list(prod_graph.graph.nodes)
    n_nodes = len(node_ids)
    print(f"N_nodes = {n_nodes}")

    # ── raw packet_size + pict_type from node + FrameRecord ────────────────
    raw_ps: dict = {}
    raw_as: dict = {}
    for nid in node_ids:
        nd = prod_graph.graph.nodes[nid]["node_data"]
        raw_ps[nid] = float(nd.packet_size)
        raw_as[nid] = float(nd.action_score)

    pict_by_idx = {fr.frame_idx: fr.pict_type for fr in index.frames}

    # base rate
    n_i = sum(1 for nid in node_ids if pict_by_idx.get(nid, "?") == "I")
    base_rate = n_i / n_nodes if n_nodes > 0 else 0.0

    type_counts = {}
    for nid in node_ids:
        pt = pict_by_idx.get(nid, "?")
        type_counts[pt] = type_counts.get(pt, 0) + 1
    print(f"pict_type counts in node set: {dict(sorted(type_counts.items()))}")
    print(f"base_rate_I = {base_rate:.4f}")

    # ── build three codec_conf variants ────────────────────────────────────
    rp_raw = rank_percentile(raw_ps)
    cc_raw = codec_conf_from_rank(rp_raw)

    rp_bytype, bytype_notes = rank_percentile_bytype(raw_ps, pict_by_idx)
    cc_bytype = codec_conf_from_rank(rp_bytype)

    rp_proxy = rank_percentile(raw_as)
    cc_proxy = codec_conf_from_rank(rp_proxy)

    print(f"\nC_bytype group notes:")
    for pt, note in sorted(bytype_notes.items()):
        print(f"  pict_type={pt!r}: {note}")

    # ── confound metric per variant ─────────────────────────────────────────
    def frac_i_top8(cc: dict) -> tuple[float, list, list]:
        top8_nodes = sorted(node_ids, key=lambda nid: cc[nid], reverse=True)[:TOP_K]
        types = [pict_by_idx.get(nid, "?") for nid in top8_nodes]
        frac = sum(1 for t in types if t == "I") / TOP_K
        return frac, top8_nodes, types

    fi_raw, top8_raw_nodes, top8_raw_types = frac_i_top8(cc_raw)
    fi_bytype, top8_bytype_nodes, top8_bytype_types = frac_i_top8(cc_bytype)
    fi_proxy, top8_proxy_nodes, top8_proxy_types = frac_i_top8(cc_proxy)

    print(f"\nConfound check (frac_I_in_top8 vs base_rate={base_rate:.3f}):")
    print(f"  C_raw:    frac_I_top8={fi_raw:.3f}  top8={top8_raw_nodes}  types={top8_raw_types}")
    print(f"  C_bytype: frac_I_top8={fi_bytype:.3f}  top8={top8_bytype_nodes}  types={top8_bytype_types}")
    print(f"  C_proxy:  frac_I_top8={fi_proxy:.3f}  top8={top8_proxy_nodes}  types={top8_proxy_types}")

    # ── build graphs ────────────────────────────────────────────────────────
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

    topologies = [("G_complete", G_complete), ("G_temporal", G_temporal)]

    # ── per-question embeddings + semantic ranks ────────────────────────────
    sem_ranks = []
    for q in QUESTIONS:
        emb = iris_query._embed_query(q, config)
        raw_sem = {
            nid: max(0.0, cosine(emb, prod_graph.graph.nodes[nid]["node_data"].embedding)
                     if prod_graph.graph.nodes[nid]["node_data"].embedding is not None else 0.0)
            for nid in node_ids
        }
        sem_ranks.append(rank_percentile(raw_sem))

    # ── per-variant seeds + seed_xQ_jaccard ────────────────────────────────
    variants = [
        ("C_raw",    cc_raw,    rp_raw),
        ("C_bytype", cc_bytype, rp_bytype),
        ("C_proxy",  cc_proxy,  rp_proxy),
    ]

    seeds_by_var: dict = {}
    seed_tops_by_var: dict = {}
    seed_xqj_by_var: dict = {}

    for vname, cc, codec_rank in variants:
        seeds = []
        seed_tops = []
        for sr in sem_ranks:
            blended = {nid: LAMBDA * sr[nid] + (1.0 - LAMBDA) * codec_rank[nid] for nid in node_ids}
            norm = normalize_seed(blended)
            seeds.append(norm)
            seed_tops.append(top_k(norm))
        seeds_by_var[vname] = seeds
        seed_tops_by_var[vname] = seed_tops
        seed_xqj_by_var[vname] = mean_pairwise_jaccard(seed_tops)

    # pure-semantic tops (lambda=1.0, for overlap_S_sem_pure)
    sem_pure_tops = []
    for sr in sem_ranks:
        norm = normalize_seed(sr)
        sem_pure_tops.append(top_k(norm))

    print(f"\nSeed xQ_jaccard at lambda={LAMBDA}:")
    for vname, _, _ in variants:
        print(f"  {vname}: {seed_xqj_by_var[vname]:.4f}")

    # ── PPR grid ────────────────────────────────────────────────────────────
    # ppr_tops[vname][tname][alpha] = list of 5 top-k lists
    ppr_tops: dict = {}
    for vname, _, _ in variants:
        ppr_tops[vname] = {}
        for tname, G in topologies:
            ppr_tops[vname][tname] = {}
            for alpha in ALPHAS:
                tops = []
                for qi in range(len(QUESTIONS)):
                    seed = seeds_by_var[vname][qi]
                    pr = nx.pagerank(G, weight="weight", personalization=seed, alpha=alpha)
                    tops.append(top_k(pr))
                ppr_tops[vname][tname][alpha] = tops

    # ── output ──────────────────────────────────────────────────────────────
    print(f"\n===DIAG7_BEGIN===")
    print(f"VIDEO: {video_path.name}  N={n_nodes}  base_rate_I={base_rate:.4f}")
    print(f"lambda={LAMBDA}  alphas={ALPHAS}  top_k={TOP_K}  window_radius={WINDOW_RADIUS}")
    print(f"\nSeed xQ_jaccard at lambda={LAMBDA}:")
    for vname, _, _ in variants:
        print(f"  {vname}: {seed_xqj_by_var[vname]:.4f}")

    print(f"\nC_bytype group notes:")
    for pt, note in sorted(bytype_notes.items()):
        print(f"  pict_type={pt!r}: {note}")

    # header
    col_w = [12, 12, 6, 8, 10, 10, 12]
    hdr = (f"  {'variant':<10}  {'topology':<12}  {'a':>4}  "
           f"{'frac_I':>7}  {'base_I':>6}  {'xQ_jacc':>8}  {'ov_sem':>8}  {'ov_Craw':>8}")
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    for vname, cc, codec_rank in variants:
        fi_v, _, _ = frac_i_top8(cc)
        for tname, G in topologies:
            for alpha in ALPHAS:
                tops = ppr_tops[vname][tname][alpha]
                tops_raw = ppr_tops["C_raw"][tname][alpha]
                xqj    = mean_pairwise_jaccard(tops)
                ov_sem = mean_overlap(tops, sem_pure_tops)
                ov_raw = mean_overlap(tops, tops_raw)
                print(f"  {vname:<10}  {tname:<12}  {alpha:>4.2f}  "
                      f"{fi_v:>7.3f}  {base_rate:>6.3f}  {xqj:>8.4f}  {ov_sem:>8.4f}  {ov_raw:>8.4f}")

    # ── per-question dump for C_bytype, G_temporal, alpha=0.15 ─────────────
    print(f"\n{'='*60}")
    print(f"PER-QUESTION DETAIL: C_bytype  G_temporal  alpha=0.15")
    bytype_gtemp_015 = ppr_tops["C_bytype"]["G_temporal"][0.15]
    for qi, q in enumerate(QUESTIONS):
        top8_list = bytype_gtemp_015[qi]
        types = [pict_by_idx.get(nid, "?") for nid in top8_list]
        print(f"  Q{qi+1}: {q!r}")
        print(f"       top8={top8_list}")
        print(f"       types={types}")

    print(f"\n===DIAG7_END===")


def main() -> None:
    for vname in VIDEOS:
        vpath = REPO_ROOT / vname
        if not vpath.exists():
            print(f"SKIP: {vpath} not found")
            continue
        try:
            run_video(vpath)
        except MemoryError as e:
            print(f"\nSKIPPED {vname}: OOM -- {e}")
        except Exception as e:
            import traceback
            print(f"\nSKIPPED {vname}: {type(e).__name__}: {e}")
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
