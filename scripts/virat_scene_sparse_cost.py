"""VIRAT real-clip built-graph cost number -- the load-bearing cost claim.

Measurement only, no mechanism change. Builds the scene_sparse index on the
real 3,077-survivor VIRAT clip (production defaults: rep_only, tau=0.015) and
reports the AUTHORITATIVE built-graph edge count -- not the zero-decode
proxy -- plus real-scale descent-time pool/edge cost over sample queries.

This replaces the 299x proxy in the cost claim with a number measured off
the actual built graph.

VERIFY: python scripts/virat_scene_sparse_cost.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest
import iris.scene_retrieval as scene_retrieval
from iris.iris_config import IRISConfig
from iris.query import _embed_query

VIDEO = REPO / "eval" / "data" / "virat" / "videos" / "VIRAT_S_000102.mp4"

# Production defaults: ranking_mode=ppr, graph_mode=scene_sparse, and the
# IRISConfig DEFAULTS for scene_crossscene_mode ("rep_only") and
# scene_shortcut_margin (0.015, tau) -- deliberately not overridden here so
# this exercises exactly what a real deployment would build/query.
CFG = IRISConfig(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=8,
    graph_mode="scene_sparse",
)

# CCTV-relevant sample queries (VIRAT is surveillance footage -- vehicles,
# people, loading/unloading -- not the generic videoplayback phrasing).
SAMPLE_QUERIES = [
    "a person walking",
    "a vehicle moving",
    "someone loading a car",
    "a person entering a building",
    "an object being carried",
]


def main() -> None:
    if not VIDEO.exists():
        print(f"FATAL: {VIDEO} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Building scene_sparse index on {VIDEO.name} "
          f"(production defaults: scene_crossscene_mode={CFG.scene_crossscene_mode}, "
          f"scene_shortcut_margin={CFG.scene_shortcut_margin}) ...")
    print("Eager build -- this is the real clip the flat-infeasibility argument rests on, "
          "expect a long tail.")
    sys.stdout.flush()

    t0 = time.time()
    idx = iris_ingest.ingest(str(VIDEO), config=CFG)
    build_wall_sec = time.time() - t0

    print()
    print(f"BUILD WALL TIME: {build_wall_sec:.1f}s ({build_wall_sec/3600:.3f}h)")
    print()

    # ── authoritative built-graph numbers ────────────────────────────────────
    num_survivors = len(idx.frames)
    scene_ids = [fr.scene_id for fr in idx.frames]
    unassigned = sum(1 for s in scene_ids if s < 0)
    num_scenes = len(set(s for s in scene_ids if s >= 0))

    per_scene_counts: dict = {}
    for s in scene_ids:
        if s < 0:
            continue
        per_scene_counts[s] = per_scene_counts.get(s, 0) + 1
    survivors_per_scene = list(per_scene_counts.values())

    built_intra = idx._graph.graph.number_of_edges()
    sum_c = sum(s * (s - 1) // 2 for s in survivors_per_scene)
    edges_flat = num_survivors * (num_survivors - 1) // 2
    ratio = edges_flat / built_intra if built_intra else float("inf")

    print("=== AUTHORITATIVE BUILT-GRAPH NUMBERS (VIRAT_S_000102.mp4) ===")
    print(f"num_survivors            = {num_survivors}")
    print(f"unassigned scene_id      = {unassigned}  (should be 0)")
    print(f"num_scenes               = {num_scenes}")
    print(f"built_intra_scene_edges  = {built_intra}   (_graph.number_of_edges(), block-diagonal, stored)")
    print(f"sum_C = sum(C(scene_size,2)) = {sum_c}")
    assert built_intra == sum_c, (
        f"EXACTNESS CHECK FAILED: built_intra ({built_intra}) != sum_C ({sum_c}) "
        "-- block-diagonal is not exact on this real clip"
    )
    print("built_intra == sum_C  (block-diagonal is exact) -- ASSERTION HELD")
    print(f"edges_flat = N*(N-1)//2  = {edges_flat}")
    print(f"ratio edges_flat / built_intra = {ratio:.1f}x")
    print()

    # ── descent-time real-scale cost over sample queries ─────────────────────
    print("=== DESCENT-TIME COST (rep_only, tau={}) over {} sample queries ===".format(
        CFG.scene_shortcut_margin, len(SAMPLE_QUERIES)
    ))
    scene_retrieval.SCENE_DIAG_RECORDS.clear()
    from dataclasses import replace as _replace
    diag_cfg = _replace(CFG, scene_diag=True)

    for q in SAMPLE_QUERIES:
        emb = _embed_query(q, diag_cfg)
        scene_retrieval.retrieve_scene_sparse(idx, emb, diag_cfg)

    records = scene_retrieval.SCENE_DIAG_RECORDS
    n_shortcut = sum(1 for r in records if r["branch"] == "shortcut")
    n_descend = sum(1 for r in records if r["branch"] == "descend")
    print(f"branch distribution: shortcut={n_shortcut}/{len(records)}  descend={n_descend}/{len(records)}")

    descend_records = [r for r in records if r["branch"] == "descend"]
    if descend_records:
        import statistics
        mean_post_pull = statistics.mean(r["post_pull_pool"] for r in descend_records)
        mean_cross_edges = statistics.mean(r["cross_scene_edges"] for r in descend_records)
        mean_base_pool = statistics.mean(r["base_pool"] for r in descend_records)
        print(f"mean base_pool (descend cases, n={len(descend_records)})       = {mean_base_pool:.1f}")
        print(f"mean post_pull_pool (descend cases)                            = {mean_post_pull:.1f}")
        print(f"mean rep_only cross_scene_edges added (descend cases)          = {mean_cross_edges:.1f}")
    else:
        print("No descend cases among sample queries (all shortcut) -- no descent-graph cost to report.")

    print()
    print("=== PER-QUERY DETAIL ===")
    for q, r in zip(SAMPLE_QUERIES, records):
        print(f"  {q!r:<32} branch={r['branch']:<9} shortlisted={r['shortlisted']}/{r['num_scenes']} "
              f"base_pool={r['base_pool']} post_pull_pool={r['post_pull_pool']} "
              f"cross_scene_edges={r['cross_scene_edges']}")


if __name__ == "__main__":
    main()
