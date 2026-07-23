"""Phase 6, scene-sparse retrieval.

2c-i: coarse-prune (per-scene centroid shortlist) + exact max-sim retrieval.
2c-ii: margin gate over 2c-i's result -- a comfortable margin short-circuits
(no PPR); a close margin descends into a union-subgraph PPR over the
shortlisted pool PLUS an anchor-temporal-neighbor pull, with cross-scene
edges added on the fly (same audited weight formula as the graph build).

No PPR descent happens unless the margin gate fires. Pure function of stored
fields (embeddings, scene_id, centroids, action_score) -- deterministic,
build == reload, whichever branch a given query takes.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def _cosine_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """One query vs N rows -> N cosine similarities. 0.0 for zero-norm rows."""
    q = np.asarray(query, dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    qn = np.linalg.norm(q)
    if qn == 0.0:
        return np.zeros(m.shape[0], dtype=np.float32)
    mn = np.linalg.norm(m, axis=1)
    denom = mn * qn
    sims = np.divide(
        m @ q, denom,
        out=np.zeros(m.shape[0], dtype=np.float32),
        where=denom != 0.0,
    )
    return sims


class SceneScorer(ABC):
    """The ANN seam: rank scenes by centroid similarity to a query embedding.

    LinearScanScorer is the only implementation today. An HNSW/IVF-backed
    scorer plugs in later behind this same interface without touching callers.
    """

    @abstractmethod
    def score(self, query_emb: np.ndarray, centroids: dict[int, np.ndarray]) -> list[tuple[int, float]]:
        """Return [(scene_id, score), ...] ranked descending by score."""
        raise NotImplementedError


class LinearScanScorer(SceneScorer):
    """One cosine per scene centroid. Sub-ms at this N (tens-hundreds of scenes)."""

    def score(self, query_emb: np.ndarray, centroids: dict[int, np.ndarray]) -> list[tuple[int, float]]:
        if not centroids:
            return []
        scene_ids = sorted(centroids.keys())
        matrix = np.stack([np.asarray(centroids[sid], dtype=np.float32) for sid in scene_ids])
        sims = _cosine_batch(query_emb, matrix)
        return sorted(zip(scene_ids, sims.tolist()), key=lambda p: (-p[1], p[0]))


def _frame_to_dict(fr: Any, sim: float) -> dict:
    """2c-i shape: dict built straight from a FrameRecord + its max-sim score."""
    return {
        "frame_idx": fr.frame_idx,
        "timestamp": fr.timestamp,
        "luma_diff_energy": fr.luma_diff_energy,
        "action_score": fr.action_score,
        "persistence_value": fr.persistence_value,
        "is_peak": fr.is_peak,
        "clip_embedding": fr.clip_embedding,
        "luma_entropy": fr.luma_entropy,
        "caption": fr.caption,
        "pagerank_score": 0.0,
        "last_retrieval_score": sim,
        "retrieval_contributions": {},
        "scene_id": getattr(fr, "scene_id", None),
    }


# 2c-iii: measurement-only accumulator for scene_diag=True runs. Cleared by
# the harness between videos/questions if it wants isolated per-video stats.
SCENE_DIAG_RECORDS: list[dict] = []


def _divergence_metrics(prod_result: list[dict], pure_result: list[dict]) -> dict:
    """Divergence between the production top_k (whatever branch fired) and the
    pure 2c-i max-sim top_k (PPR skipped). jaccard/top1_changed/rank
    displacement are computed on frame_idx only -- order-sensitive, stable."""
    idxs_a = [f["frame_idx"] for f in prod_result]
    idxs_b = [f["frame_idx"] for f in pure_result]
    set_a, set_b = set(idxs_a), set(idxs_b)
    union = set_a | set_b
    jaccard = (len(set_a & set_b) / len(union)) if union else 1.0
    top1_changed = bool(idxs_a[:1] != idxs_b[:1])

    rank_a = {fi: i for i, fi in enumerate(idxs_a)}
    rank_b = {fi: i for i, fi in enumerate(idxs_b)}
    shared = set_a & set_b
    mean_rank_displacement = (
        sum(abs(rank_a[fi] - rank_b[fi]) for fi in shared) / len(shared)
        if shared else None
    )
    return {
        "jaccard": jaccard,
        "top1_changed": top1_changed,
        "mean_rank_displacement": mean_rank_displacement,
    }


def _node_to_dict(node: Any, frame_map: dict) -> dict:
    """2c-ii shape: dict built from an AsphodelNode (post-PPR) + its matching
    FrameRecord for the fields PPR doesn't carry (is_peak/caption/etc).

    scene_id comes from fr (the FrameRecord), NOT node -- AsphodelNode.scene_id
    is a monotonic I-frame-segmentation id that L2Asphodel._refresh_scene_ids
    unconditionally overwrites on every graph build (including the induced
    sub_graph copy retrieve_scene_sparse's DESCEND branch runs PPR over), so
    it never matches the real valley-boundary scene_id this dict's caller
    (eval.metrics.predicted_span_from_frames_scene) looks up in
    index.scene_spans."""
    fr = frame_map.get(node.frame_idx)
    return {
        "frame_idx": node.frame_idx,
        "timestamp": node.timestamp,
        "luma_diff_energy": node.luma_diff_energy,
        "action_score": node.action_score,
        "persistence_value": node.persistence_value,
        "is_peak": getattr(fr, "is_peak", False),
        "clip_embedding": getattr(fr, "clip_embedding", None),
        "luma_entropy": getattr(fr, "luma_entropy", 0.0),
        "caption": getattr(fr, "caption", None),
        "pagerank_score": node.pagerank_score,
        "last_retrieval_score": getattr(node, "last_retrieval_score", 0.0),
        "retrieval_contributions": getattr(node, "retrieval_contributions", {}),
        "scene_id": getattr(fr, "scene_id", None),
    }


def retrieve_scene_sparse(
    index: Any,
    query_embedding: np.ndarray,
    config: Any,
    scorer: SceneScorer | None = None,
    trace: dict | None = None,
) -> list[dict]:
    """Coarse-prune to a scene shortlist via centroid similarity (2c-i), then
    gate on the max-sim margin between the best scene and the runner-up scene:

      margin > tau  -> SHORTCUT: return the 2c-i exact max-sim top_k as-is.
      margin <= tau -> DESCEND: pull anchor-temporal neighbors into the pool,
                       add cross-scene edges over the pool, and run PPR over
                       the induced union subgraph.

    Deterministic: pure function of stored fields (embeddings, scene_id,
    centroids, action_score), no RNG, stable tie-break by frame_idx.

    trace: optional dict, populated in-place with the exact telemetry values
    already computed for the `[scene_retrieval] ...` log line (scene ids,
    scores, margin, tau, branch, base_pool, post_pull_pool,
    cross_scene_edges_added) when not None. Read-only instrumentation hook
    for iris.debug_trace -- does not affect which branch is taken, what is
    returned, or any decision made by this function. None (default) costs
    nothing extra beyond the `is not None` checks already guarding it.
    """
    # SCENE-002: Reject invalid query embeddings (all-zero or near-zero norm) before shortlist creation.
    q_norm = np.linalg.norm(query_embedding)
    if q_norm < 1e-8:
        raise ValueError("Invalid query embedding (all-zero or near-zero norm).")

    centroids = getattr(index, "_scene_centroids", None)
    if not centroids:
        raise ValueError(
            "graph_mode='scene_sparse' requires index._scene_centroids to be "
            "populated (an old flat index was reloaded in scene_sparse mode); "
            "reingest with graph_mode='scene_sparse' or use graph_mode='flat'"
        )

    scorer = scorer or LinearScanScorer()
    num_scenes = len(centroids)
    # REPORT-NOT-TUNE. Generous net by design -- the centroid is a shortlister,
    # never a filter.
    shortlist_width = getattr(config, "scene_shortlist_width", 0) or max(4, math.ceil(math.sqrt(num_scenes)))
    shortlist_width = min(shortlist_width, num_scenes)

    scene_ranking = scorer.score(query_embedding, centroids)
    shortlisted_scene_ids = {sid for sid, _ in scene_ranking[:shortlist_width]}

    survivors = [
        fr for fr in index.frames
        if fr.scene_id in shortlisted_scene_ids and fr.clip_embedding is not None
    ]
    l2_retrieve_top_k = getattr(config, "l2_retrieve_top_k", 5)

    if not survivors:
        # Fall back to all scenes
        shortlisted_scene_ids = set(centroids.keys())
        survivors = [
            fr for fr in index.frames
            if fr.scene_id in shortlisted_scene_ids and fr.clip_embedding is not None
        ]

    if not survivors:
        print(
            f"[scene_retrieval] shortlisted {len(shortlisted_scene_ids)}/{num_scenes} scenes, "
            f"survivor pool = 0 frames, branch=shortcut (empty pool)"
        )
        if trace is not None:
            trace.update({
                "shortlisted_scene_ids": sorted(shortlisted_scene_ids),
                "num_scenes": num_scenes,
                "scene_scores": dict(scene_ranking),
                "branch": "shortcut",
                "reason": "empty_pool",
                "base_pool": 0,
                "post_pull_pool": 0,
                "margin": None,
                "tau": None,
                "cross_scene_edges_added": 0,
                "retrieved_frame_idxs": [],
                "retrieved_timestamps": [],
            })
        return []

    pool_matrix = np.stack([np.asarray(fr.clip_embedding, dtype=np.float32) for fr in survivors])
    sims = _cosine_batch(query_embedding, pool_matrix)

    # SCENE-001: Adaptive fallback for recall safety.
    # If the maximum similarity in the shortlisted pool is below 0.20,
    # we fall back to all scenes.
    if float(np.max(sims)) < 0.20 and len(shortlisted_scene_ids) < num_scenes:
        shortlisted_scene_ids = set(centroids.keys())
        survivors = [
            fr for fr in index.frames
            if fr.scene_id in shortlisted_scene_ids and fr.clip_embedding is not None
        ]
        if survivors:
            pool_matrix = np.stack([np.asarray(fr.clip_embedding, dtype=np.float32) for fr in survivors])
            sims = _cosine_batch(query_embedding, pool_matrix)

    order = sorted(range(len(survivors)), key=lambda i: (-float(sims[i]), survivors[i].frame_idx))
    exact_top = [_frame_to_dict(survivors[i], float(sims[i])) for i in order[:l2_retrieve_top_k]]

    # ── MARGIN GATE ──────────────────────────────────────────────────────
    # anchor = top max-sim frame (its scene = scene A). runner_up = the best
    # max-sim frame in the best scene != A. Stable tie-break: higher sim wins;
    # equal sim -> lower frame_idx / lower scene_id wins.
    best_per_scene: dict[int, tuple[float, int]] = {}
    for fr, sim in zip(survivors, sims.tolist()):
        sim = float(sim)
        cur = best_per_scene.get(fr.scene_id)
        if cur is None or sim > cur[0] or (sim == cur[0] and fr.frame_idx < cur[1]):
            best_per_scene[fr.scene_id] = (sim, fr.frame_idx)

    ranked_scenes = sorted(best_per_scene.items(), key=lambda kv: (-kv[1][0], kv[0]))
    anchor_scene_id, (anchor_sim, anchor_frame_idx) = ranked_scenes[0]
    if len(ranked_scenes) > 1:
        _, (runner_up_sim, _) = ranked_scenes[1]
        margin = anchor_sim - runner_up_sim
    else:
        margin = float("inf")  # only one shortlisted scene -- nothing to descend across

    tau = getattr(config, "scene_shortcut_margin", 0.05)

    if margin > tau:
        if getattr(config, "scene_diag", False):
            metrics = _divergence_metrics(exact_top, exact_top)  # shortcut == pure, trivially
            record = {
                "branch": "shortcut", "margin": margin, "tau": tau,
                "num_scenes": num_scenes, "shortlisted": len(shortlisted_scene_ids),
                "base_pool": len(survivors), "post_pull_pool": len(survivors),
                "cross_scene_edges": 0,
                **metrics,
            }
            SCENE_DIAG_RECORDS.append(record)
            print(f"[scene_diag] {record}")
        print(
            f"[scene_retrieval] shortlisted {len(shortlisted_scene_ids)}/{num_scenes} scenes, "
            f"base_pool={len(survivors)} margin={margin:.4f} tau={tau} branch=shortcut"
        )
        if trace is not None:
            trace.update({
                "shortlisted_scene_ids": sorted(shortlisted_scene_ids),
                "num_scenes": num_scenes,
                "scene_scores": dict(scene_ranking),
                "branch": "shortcut",
                "margin": margin,
                "tau": tau,
                "base_pool": len(survivors),
                "post_pull_pool": len(survivors),
                "cross_scene_edges_added": 0,
                "retrieved_frame_idxs": [f["frame_idx"] for f in exact_top],
                "retrieved_timestamps": [f["timestamp"] for f in exact_top],
                "retrieved_scores": [f["last_retrieval_score"] for f in exact_top],
            })
        return exact_top

    # ── DESCEND ──────────────────────────────────────────────────────────
    window = getattr(config, "scene_neighbor_window", 30)
    base_pool_ids = {fr.frame_idx for fr in survivors}
    neighbor_frames = [
        fr for fr in index.frames
        if fr.clip_embedding is not None
        and fr.frame_idx not in base_pool_ids
        and abs(fr.frame_idx - anchor_frame_idx) <= window
    ]
    pool_frames = survivors + neighbor_frames
    pool_ids = [fr.frame_idx for fr in pool_frames]
    scene_of = {fr.frame_idx: fr.scene_id for fr in pool_frames}

    # Per-scene anchor (highest sim-to-query frame in that scene, within the
    # pool) -- needed for crossscene_mode="rep_only" ("linked only via reps").
    pool_matrix_full = np.stack([np.asarray(fr.clip_embedding, dtype=np.float32) for fr in pool_frames])
    pool_sims_full = _cosine_batch(query_embedding, pool_matrix_full)
    scene_anchor_frame: dict[int, int] = {}
    best_sim_per_scene: dict[int, float] = {}
    for fr, sim in zip(pool_frames, pool_sims_full.tolist()):
        sid = fr.scene_id
        cur = best_sim_per_scene.get(sid)
        if cur is None or sim > cur or (sim == cur and fr.frame_idx < scene_anchor_frame[sid]):
            best_sim_per_scene[sid] = sim
            scene_anchor_frame[sid] = fr.frame_idx

    graph = index._graph
    crossscene_mode = getattr(config, "scene_crossscene_mode", "all")
    crossscene_pctile = getattr(config, "scene_crossscene_threshold_pctile", 75.0)

    # Non-mutating: copy the induced subgraph (existing intra-scene edges come
    # along), add cross-scene edges to the COPY only. Keeps the shared
    # production graph clean across queries and lets sparsity sweeps (2c-iv)
    # compare modes on the same pool without leaking edges between them.
    sub_graph = graph.graph.subgraph(pool_ids).copy()
    cross_edges_added = graph.add_cross_scene_edges(
        pool_ids, scene_of,
        mode=crossscene_mode,
        threshold_percentile=crossscene_pctile,
        scene_anchors=scene_anchor_frame,
        graph=sub_graph,
    )

    damping = getattr(config, "ppr_damping", 0.5)
    lambda_ = getattr(config, "ppr_lambda", 0.5)
    ppr_nodes = graph.retrieve_ppr(
        query_embedding,
        top_k=l2_retrieve_top_k,
        damping=damping,
        lambda_=lambda_,
        graph_override=sub_graph,
    )

    frame_map = {fr.frame_idx: fr for fr in pool_frames}
    result = [_node_to_dict(node, frame_map) for node in ppr_nodes]

    if getattr(config, "scene_diag", False):
        metrics = _divergence_metrics(result, exact_top)
        pool_pairs = len(pool_frames) * (len(pool_frames) - 1) / 2
        record = {
            "branch": "descend", "margin": margin, "tau": tau,
            "num_scenes": num_scenes, "shortlisted": len(shortlisted_scene_ids),
            "base_pool": len(survivors), "post_pull_pool": len(pool_frames),
            "cross_scene_edges": cross_edges_added,
            "pool_density": (cross_edges_added / pool_pairs) if pool_pairs else 0.0,
            "crossscene_mode": crossscene_mode, "crossscene_pctile": crossscene_pctile,
            "result_frame_idxs": [f["frame_idx"] for f in result],
            **metrics,
        }
        SCENE_DIAG_RECORDS.append(record)
        print(f"[scene_diag] {record}")

    print(
        f"[scene_retrieval] shortlisted {len(shortlisted_scene_ids)}/{num_scenes} scenes, "
        f"margin={margin:.4f} tau={tau} branch=descend base_pool={len(survivors)} "
        f"post_pull_pool={len(pool_frames)} cross_scene_edges_added={cross_edges_added}"
    )

    if trace is not None:
        trace.update({
            "shortlisted_scene_ids": sorted(shortlisted_scene_ids),
            "num_scenes": num_scenes,
            "scene_scores": dict(scene_ranking),
            "branch": "descend",
            "margin": margin,
            "tau": tau,
            "base_pool": len(survivors),
            "post_pull_pool": len(pool_frames),
            "cross_scene_edges_added": cross_edges_added,
            "retrieved_frame_idxs": [f["frame_idx"] for f in result],
            "retrieved_timestamps": [f["timestamp"] for f in result],
            "retrieved_scores": [f["last_retrieval_score"] for f in result],
        })

    return result
