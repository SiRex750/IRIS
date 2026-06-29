"""Tests for retrieve_ppr() and the ranking_mode="ppr" branch in _build_retrieved."""
from __future__ import annotations

import numpy as np
import pytest

from iris.l2_asphodel import L2Asphodel
from iris.types import FrameRecord, IRISIndex


# ── helpers ────────────────────────────────────────────────────────────────

def _make_graph(n: int = 4) -> tuple[L2Asphodel, list[np.ndarray]]:
    """Build an enriched L2Asphodel with n orthogonal unit-vector embeddings."""
    graph = L2Asphodel(config={"alpha": 0.4, "beta": 0.6, "gamma": 0.0})
    embeddings = []
    for i in range(n):
        e = np.zeros(n, dtype=np.float32)
        e[i] = 1.0
        embeddings.append(e)

    feature_records = [
        {
            "frame_idx": i,
            "timestamp": float(i),
            "luma_diff_energy": 0.1 * (i + 1),
            "motion_magnitude": float(i + 1),
            "luma_entropy": 1.0,
            "refined_motion_tensor": np.zeros(1, dtype=np.float32),
        }
        for i in range(n)
    ]
    action_score_records = [
        {"action_score": 0.1 * (i + 1), "persistence_value": 0.2}
        for i in range(n)
    ]
    graph.add_frame_nodes_bulk(feature_records, action_score_records)
    enrichment_map = {i: embeddings[i] for i in range(n)}
    graph.enrich_nodes_bulk(enrichment_map)
    return graph, embeddings


def _make_index(graph: L2Asphodel, n: int) -> IRISIndex:
    frames = [
        FrameRecord(
            frame_idx=i,
            timestamp=float(i),
            luma_diff_energy=0.1,
            luma_entropy=0.0,
            motion_magnitude=0.0,
            action_score=0.1 * (i + 1),
            persistence_value=0.2,
            is_peak=False,
            clip_embedding=np.zeros(n, dtype=np.float32),
            pagerank_score=0.0,
        )
        for i in range(n)
    ]
    idx = IRISIndex(
        video_path="test.mp4",
        frames=frames,
        index_action_score=0.5,
        stats={"total": n, "skipped": 0},
        frames_processed=n,
        peak_count=0,
        skipped_frames_ratio=0.0,
        storage_reduction_factor=1.0,
        config_snapshot={},
    )
    idx._graph = graph
    return idx


# ── (a) enriched graph, aligned query → top_k returned, teleport_fallback False,
#       PPR scores are valid (positive, sum ≈ 1) ──────────────────────────────

def test_ppr_aligned_query_returns_topk_and_no_fallback():
    graph, embeddings = _make_graph(n=4)

    results = graph.retrieve_ppr(embeddings[0], top_k=3, damping=0.85)
    assert len(results) == 3
    for nd in results:
        assert nd.retrieval_contributions["teleport_fallback"] is False
        assert nd.last_retrieval_score > 0.0

    # All-node PPR scores must sum to ≈ 1 (PageRank invariant).
    all_results = graph.retrieve_ppr(embeddings[0], top_k=4)
    total = sum(nd.last_retrieval_score for nd in all_results)
    assert abs(total - 1.0) < 1e-4, f"PPR scores must sum to ~1, got {total}"


# ── (b) query_embedding=None → teleport_fallback True, valid top-k returned ──

def test_ppr_none_query_fallback_uniform():
    graph, _ = _make_graph(n=4)
    results = graph.retrieve_ppr(None, top_k=3, damping=0.85)

    assert len(results) == 3
    for nd in results:
        assert nd.retrieval_contributions["teleport_fallback"] is True


# ── (c) determinism — identical inputs → identical ordering ──────────────────

def test_ppr_deterministic():
    graph, embeddings = _make_graph(n=5)
    query = embeddings[2].copy()
    r1 = [nd.frame_idx for nd in graph.retrieve_ppr(query, top_k=4)]
    r2 = [nd.frame_idx for nd in graph.retrieve_ppr(query, top_k=4)]
    assert r1 == r2


# ── (d) _build_retrieved with ranking_mode="ppr" returns l2_retrieve_top_k dicts ──

def test_build_retrieved_ppr_mode():
    import iris.query as q

    n = 4
    top_k = 3
    graph, embeddings = _make_graph(n=n)
    idx = _make_index(graph, n=n)

    from iris.iris_config import IRISConfig
    config = IRISConfig(ranking_mode="ppr", l2_retrieve_top_k=top_k)

    query_emb = embeddings[0].copy()
    frames = q._build_retrieved(idx, query_emb, config)

    assert len(frames) == top_k
    for f in frames:
        assert "frame_idx" in f
        assert "action_score" in f
        assert "pagerank_score" in f
        assert "last_retrieval_score" in f
