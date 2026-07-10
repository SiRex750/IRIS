"""Verify 6.2b: drive the production ppr path on videoplayback.mp4.

Proves the production wiring computes the same mechanism as diag7 C_bytype.
Expected ballpark (diag7 reference): frac_I≈0.25, ov_sem≈0.075, xQ≈0.30–0.34.
If results diverge materially, STOP and report — no tuning.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import iris.ingest as iris_ingest
import iris.query as iris_query
from iris.iris_config import IRISConfig

QUESTIONS = [
    "what is moving in the scene",
    "describe the main action",
    "what happens at the end",
    "is there water or a stream",
    "show a wide landscape shot",
]
TOP_K = 8


def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    u = sa | sb
    return len(sa & sb) / len(u) if u else 1.0


def mean_pairwise_jaccard(lists: list) -> float:
    pairs = list(itertools.combinations(range(len(lists)), 2))
    if not pairs:
        return 1.0
    return float(np.mean([jaccard(lists[i], lists[j]) for i, j in pairs]))


def mean_overlap(a_lists: list, b_lists: list) -> float:
    return float(np.mean([len(set(a) & set(b)) / TOP_K for a, b in zip(a_lists, b_lists)]))


def main() -> None:
    vpath = REPO_ROOT / "videoplayback.mp4"
    if not vpath.exists():
        print(f"ERROR: {vpath} not found")
        sys.exit(1)

    cfg_ppr = IRISConfig(
        ranking_mode="ppr",
        codec_conf_source="packet_size",
        codec_conf_pictype_norm=True,
        ppr_lambda=0.5,
        ppr_damping=0.5,
        l2_retrieve_top_k=TOP_K,
    )
    cfg_sem = IRISConfig(
        ranking_mode="ppr",
        codec_conf_source="packet_size",
        codec_conf_pictype_norm=True,
        ppr_lambda=1.0,
        ppr_damping=0.5,
        l2_retrieve_top_k=TOP_K,
    )

    print("Ingesting videoplayback.mp4 with production ppr config...")
    sys.stdout.flush()
    index = iris_ingest.ingest(str(vpath), config=cfg_ppr)

    pict_by_idx = {fr.frame_idx: fr.pict_type for fr in index.frames}
    n_nodes = len(index.frames)
    n_i = sum(1 for fr in index.frames if fr.pict_type == "I")
    base_rate = n_i / n_nodes if n_nodes > 0 else 0.0

    # Spot-check codec_conf on 5 frames (confirms step 6.5 ran)
    sample = []
    for fr in list(sorted(index.frames, key=lambda x: x.frame_idx))[:5]:
        fi = fr.frame_idx
        node_cc = index._graph.graph.nodes[fi]["node_data"].codec_conf if fi in index._graph.graph.nodes else -1.0
        sample.append((fi, fr.pict_type, fr.packet_size, fr.codec_conf, node_cc))

    print(f"N_nodes={n_nodes}  base_rate_I={base_rate:.4f}")
    print(f"Sample (frame_idx, pict_type, packet_size, fr.codec_conf, node.codec_conf):")
    for row in sample:
        print(f"  {row}")

    # Production PPR retrieval for 5 questions
    ppr_tops: list[list] = []
    for q in QUESTIONS:
        emb = iris_query._embed_query(q, cfg_ppr)
        retrieved = iris_query._build_retrieved(index, emb, cfg_ppr)
        ppr_tops.append([f["frame_idx"] for f in retrieved])

    # Semantic-pure tops (λ=1.0) for overlap_S_sem_pure
    sem_tops: list[list] = []
    for q in QUESTIONS:
        emb = iris_query._embed_query(q, cfg_sem)
        retrieved = iris_query._build_retrieved(index, emb, cfg_sem)
        sem_tops.append([f["frame_idx"] for f in retrieved])

    xqj    = mean_pairwise_jaccard(ppr_tops)
    ov_sem = mean_overlap(ppr_tops, sem_tops)

    frac_i_per_q = []
    for tops in ppr_tops:
        types = [pict_by_idx.get(fi, "?") for fi in tops]
        frac_i_per_q.append(sum(1 for t in types if t == "I") / len(tops) if tops else 0.0)
    frac_i_mean = float(np.mean(frac_i_per_q))

    print(f"\n===VERIFY62B_BEGIN===")
    print(f"VIDEO: videoplayback.mp4  N={n_nodes}  base_rate_I={base_rate:.4f}")
    print(f"config: ranking_mode=ppr  codec_conf_pictype_norm=True  ppr_lambda=0.5  ppr_damping=0.5  top_k={TOP_K}")
    print(f"\nxQ_jaccard (across 5 questions): {xqj:.4f}")
    print(f"overlap_S_sem_pure (ppr vs lambda=1.0): {ov_sem:.4f}")
    print(f"frac_I_top{TOP_K} (mean across questions): {frac_i_mean:.4f}  base_rate_I={base_rate:.4f}")
    print(f"\nPer-question top-{TOP_K}:")
    for qi, q in enumerate(QUESTIONS):
        tops = ppr_tops[qi]
        types = [pict_by_idx.get(fi, "?") for fi in tops]
        frac_i = sum(1 for t in types if t == "I") / len(tops) if tops else 0.0
        print(f"  Q{qi + 1}: {q!r}")
        print(f"       top{TOP_K}={tops}")
        print(f"       types={types}  frac_I={frac_i:.3f}")
    print(f"\n===VERIFY62B_END===")


if __name__ == "__main__":
    main()
