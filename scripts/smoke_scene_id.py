"""Smoke test for zero-decode scene_id assignment (Phase 6, task 2b-i) and the
scene_sparse graph structure + centroid dispersion (Phase 6, task 2b-ii).
Not part of the phase6 measurement suite -- throwaway verification script.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import iris.ingest as iris_ingest
from iris.iris_config import IRISConfig
from iris.query import _embed_query
from iris.scene_retrieval import LinearScanScorer, _cosine_batch, retrieve_scene_sparse

# REPORT-NOT-TUNE: stated default. Do NOT adjust to hit a target bimodal count.
BIMODAL_IMPROVEMENT = 0.05  # min (within-cluster cos - cos-to-centroid) gain to call a scene bimodal


def bucket_histogram(values, edges):
    counts = [0] * (len(edges) + 1)
    for v in values:
        placed = False
        for i, e in enumerate(edges):
            if v <= e:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    labels = [f"<={e}" for e in edges] + [f">{edges[-1]}"]
    return list(zip(labels, counts))


def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def scene_dispersion(embs: np.ndarray, centroid: np.ndarray):
    """Returns (mean_cos_to_centroid, bimodal: bool | None). bimodal is None
    for scenes with < 6 survivors (tightness-histogram-only, per spec)."""
    cos_to_centroid = [_cosine(e, centroid) for e in embs]
    mean_cos_to_centroid = float(np.mean(cos_to_centroid))

    if len(embs) < 6:
        return mean_cos_to_centroid, None

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(embs)
    labels = km.labels_
    within_cos = []
    for k in (0, 1):
        members = embs[labels == k]
        if len(members) == 0:
            continue
        sub_centroid = members.mean(axis=0)
        within_cos.extend(_cosine(m, sub_centroid) for m in members)
    mean_within_cluster_cos = float(np.mean(within_cos)) if within_cos else mean_cos_to_centroid
    bimodal = (mean_within_cluster_cos - mean_cos_to_centroid) > BIMODAL_IMPROVEMENT
    return mean_cos_to_centroid, bimodal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_mode", default="flat", choices=["flat", "scene_sparse"])
    parser.add_argument("--query", default=None, help="run retrieve_scene_sparse for this query text")
    args = parser.parse_args()

    vpath = REPO_ROOT / "videoplayback.mp4"
    if not vpath.exists():
        print(f"ERROR: {vpath} not found")
        sys.exit(1)

    cfg = IRISConfig(
        ranking_mode="ppr", codec_conf_source="packet_size",
        codec_conf_pictype_norm=True, ppr_lambda=0.5, ppr_damping=0.5,
        l2_retrieve_top_k=8, graph_mode=args.graph_mode,
    )

    print(f"Ingesting videoplayback.mp4 with production ppr config (graph_mode={args.graph_mode})...")
    sys.stdout.flush()
    idx = iris_ingest.ingest(str(vpath), config=cfg)

    unassigned = [fr.frame_idx for fr in idx.frames if fr.scene_id < 0]
    assert not unassigned, f"unassigned scene_id for frame_idx: {unassigned}"

    scene_ids = [fr.scene_id for fr in idx.frames]
    num_scenes = len(set(scene_ids))

    per_scene_counts: dict = {}
    for sid in scene_ids:
        per_scene_counts[sid] = per_scene_counts.get(sid, 0) + 1
    survivors_per_scene = list(per_scene_counts.values())

    surv_edges = [1, 2, 4, 8, 16, 32, 64]
    hist = bucket_histogram(survivors_per_scene, surv_edges)

    print("ALL scene_id ASSIGNED (none unassigned)")
    print(f"num_survivors = {len(idx.frames)}")
    print(f"num_scenes (videoplayback.mp4) = {num_scenes}")
    print("survivors-per-scene histogram:")
    for label, count in hist:
        print(f"  {label}: {count}")

    if args.graph_mode != "scene_sparse":
        return

    # --- scene_sparse structure checks ---
    n_surv = len(idx.frames)
    built_edges = idx._graph.graph.number_of_edges()
    sum_c = sum(s * (s - 1) // 2 for s in survivors_per_scene)
    edges_flat = n_surv * (n_surv - 1) // 2

    print()
    print("=== SCENE_SPARSE GRAPH STRUCTURE ===")
    print(f"built_edges = {built_edges}")
    print(f"sum_C = sum(scene_size*(scene_size-1)//2) = {sum_c}")
    assert built_edges == sum_c, f"built_edges ({built_edges}) != sum_C ({sum_c}) -- block-diagonal not exact"
    print("built_edges == sum_C  (block-diagonal is exact)")
    print(f"edges_flat = {edges_flat}")
    print(f"ratio edges_flat / built_edges = {edges_flat / built_edges if built_edges else float('inf')}")

    # --- embedding-dispersion stat (per-scene, decides single-vs-sub-centroid next turn) ---
    frames_by_scene: dict = {}
    for fr in idx.frames:
        frames_by_scene.setdefault(fr.scene_id, []).append(fr)

    tightness_edges = [-0.5, 0.0, 0.5, 0.7, 0.85, 0.95]
    mean_cos_values = []
    bimodal_count = 0
    eligible_count = 0
    for sid, frs in frames_by_scene.items():
        if len(frs) < 2:
            continue
        embs = np.array([fr.clip_embedding for fr in frs], dtype=np.float64)
        centroid = idx._scene_centroids[sid]
        mean_cos, bimodal = scene_dispersion(embs, centroid)
        mean_cos_values.append(mean_cos)
        if bimodal is not None:
            eligible_count += 1
            if bimodal:
                bimodal_count += 1

    tightness_hist = bucket_histogram(mean_cos_values, tightness_edges)

    print()
    print("=== EMBEDDING-DISPERSION (structural counterpart to packet multi-burst) ===")
    print("NOTE: PRINT only, not acted on -- decides single-vs-sub-centroid next turn.")
    print("scene tightness histogram (mean_cos_to_centroid):")
    for label, count in tightness_hist:
        print(f"  {label}: {count}")
    print(f"scenes eligible for k=2 (>=6 survivors) = {eligible_count}")
    print(f"bimodal_scene_count = {bimodal_count}")
    print(f"bimodal_fraction (of eligible) = {(bimodal_count / eligible_count) if eligible_count else 0.0}")
    print(f"BIMODAL_IMPROVEMENT = {BIMODAL_IMPROVEMENT}")

    if not args.query:
        return

    print()
    print("=== QUERY MODE (scene_sparse retrieval, task 2c-i baseline) ===")
    query_emb = _embed_query(args.query, cfg)

    result1 = retrieve_scene_sparse(idx, query_emb, cfg)
    result2 = retrieve_scene_sparse(idx, query_emb, cfg)

    assert len(result1) == cfg.l2_retrieve_top_k, (
        f"expected {cfg.l2_retrieve_top_k} frames, got {len(result1)}"
    )

    idxs1 = [f["frame_idx"] for f in result1]
    idxs2 = [f["frame_idx"] for f in result2]
    assert idxs1 == idxs2, f"non-deterministic retrieval: {idxs1} != {idxs2}"

    frame_scene = {fr.frame_idx: fr.scene_id for fr in idx.frames}
    print(f"query = {args.query!r}")
    print(f"production default (tau={cfg.scene_shortcut_margin}) returned top_k = {len(result1)}, deterministic across 2 calls")
    for f in result1:
        print(f"  frame_idx={f['frame_idx']}  scene_id={frame_scene[f['frame_idx']]}  score={f['last_retrieval_score']:.4f}")

    # ── task 2c-ii: force both branches via scene_shortcut_margin ──────────
    def _run_and_capture(cfg_variant):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = retrieve_scene_sparse(idx, query_emb, cfg_variant)
        log = buf.getvalue()
        sys.stdout.write(log)
        return result, log

    print()
    print("=== FORCED DESCEND (scene_shortcut_margin=1.0, impossible margin) ===")
    cfg_descend = _replace(cfg, scene_shortcut_margin=1.0)
    descend1, log1 = _run_and_capture(cfg_descend)
    descend2, log2 = _run_and_capture(cfg_descend)

    assert "branch=descend" in log1, f"expected descend branch, got log: {log1!r}"
    assert len(descend1) == cfg.l2_retrieve_top_k, (
        f"expected {cfg.l2_retrieve_top_k} frames, got {len(descend1)}"
    )
    idxs_d1 = [f["frame_idx"] for f in descend1]
    idxs_d2 = [f["frame_idx"] for f in descend2]
    assert idxs_d1 == idxs_d2, f"non-deterministic descend retrieval: {idxs_d1} != {idxs_d2}"
    print(f"DESCEND branch OK: top_k={len(descend1)} frame_idxs={idxs_d1} deterministic across 2 calls")

    print()
    print("=== FORCED SHORTCUT (scene_shortcut_margin=-1.0, margin >= 0 always beats it) ===")
    # margin = anchor_sim - runner_up_sim is always >= 0 by construction (anchor is the
    # max), so an exact 0.0 tau can tie with an exact-zero margin (float tie between
    # near-duplicate frames, as happens for this query). A negative tau is test-only
    # and guarantees margin > tau unconditionally.
    cfg_shortcut = _replace(cfg, scene_shortcut_margin=-1.0)
    shortcut1, log3 = _run_and_capture(cfg_shortcut)

    assert "branch=shortcut" in log3, f"expected shortcut branch, got log: {log3!r}"

    # Pure 2c-i max-sim reference (coarse shortlist + exact max-sim, no PPR)
    # recomputed independently to confirm the shortcut branch reproduces it exactly.
    scene_ranking = LinearScanScorer().score(query_emb, idx._scene_centroids)
    shortlist_width = cfg.scene_shortlist_width or max(4, math.ceil(math.sqrt(num_scenes)))
    shortlist_width = min(shortlist_width, num_scenes)
    shortlisted_scene_ids = {sid for sid, _ in scene_ranking[:shortlist_width]}
    ref_survivors = [
        fr for fr in idx.frames
        if fr.scene_id in shortlisted_scene_ids and fr.clip_embedding is not None
    ]
    ref_matrix = np.stack([np.asarray(fr.clip_embedding, dtype=np.float32) for fr in ref_survivors])
    ref_sims = _cosine_batch(query_emb, ref_matrix)
    ref_order = sorted(range(len(ref_survivors)), key=lambda i: (-float(ref_sims[i]), ref_survivors[i].frame_idx))
    ref_top_idxs = [ref_survivors[i].frame_idx for i in ref_order[:cfg.l2_retrieve_top_k]]

    shortcut_idxs = [f["frame_idx"] for f in shortcut1]
    assert shortcut_idxs == ref_top_idxs, (
        f"shortcut branch does not match pure 2c-i max-sim: {shortcut_idxs} != {ref_top_idxs}"
    )
    print(f"SHORTCUT branch OK: top_k={len(shortcut1)} matches pure 2c-i max-sim exactly: {shortcut_idxs}")


if __name__ == "__main__":
    main()
