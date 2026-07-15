"""Tests for the scene_sparse DESCEND retrieval branch.

Covers:
  - L2Asphodel._add_weighted_edge_to_graph (Part 1 helper)
  - L2Asphodel.add_cross_scene_edges: all three modes + validation (Part 2)
  - L2Asphodel.retrieve_ppr with graph_override (Part 3)
  - End-to-end descend path via retrieve_scene_sparse (Part 4)
  - Production graph non-mutation guarantee throughout

All tests are deterministic (no RNG).  No video file required.
Run with:  pytest tests/test_scene_sparse_descend.py -v
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from iris.l2_asphodel import L2Asphodel
from iris.scene_retrieval import retrieve_scene_sparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _orth_embs(n, dim=0):
    d = max(n, dim)
    out = []
    for i in range(n):
        e = np.zeros(d, dtype=np.float32)
        e[i] = 1.0
        out.append(e)
    return out


def _build_graph(scene_assignments, embeddings=None):
    n = len(scene_assignments)
    seen = set()
    feats = []
    for i, sid in enumerate(scene_assignments):
        pict = "I" if sid not in seen else "P"
        seen.add(sid)
        feats.append({
            "frame_idx": i, "timestamp": float(i),
            "luma_diff_energy": 0.1*(i+1), "motion_magnitude": float(i+1),
            "luma_entropy": 1.0, "refined_motion_tensor": np.zeros(1, dtype=np.float32),
            "codec_conf": 0.5+0.1*(i%5), "pict_type": pict,
        })
    acts = [{"action_score": 0.1*(i+1), "persistence_value": 0.2} for i in range(n)]
    g = L2Asphodel()
    g.add_frame_nodes_bulk(feats, acts)
    if embeddings is not None:
        g.enrich_nodes_bulk({i: embeddings[i] for i in range(n)})
    return g


def _sub(graph, pool_ids):
    return graph.graph.subgraph(pool_ids).copy()


def _two_scene(n_per=3, use_embs=True):
    n = n_per * 2
    sa = [0]*n_per + [1]*n_per
    embs = _orth_embs(n) if use_embs else None
    g = _build_graph(sa, embs)
    pool = list(range(n))
    sub = _sub(g, pool)
    sof = {i: sa[i] for i in pool}
    anch = {0: 0, 1: n_per}
    return g, sub, pool, sof, anch


# ---------------------------------------------------------------------------
# Part 1: _add_weighted_edge_to_graph
# ---------------------------------------------------------------------------


class TestAddWeightedEdgeToGraph:

    def test_new_edge_returns_new(self):
        g = _build_graph([0, 1])
        s = _sub(g, [0, 1])
        if s.has_edge(0, 1): s.remove_edge(0, 1)
        assert g._add_weighted_edge_to_graph(s, 0, 1, "cross_scene_all", 1.0) == "new"
        assert s.has_edge(0, 1)

    def test_weaker_existing_returns_unchanged(self):
        g = _build_graph([0, 1])
        s = _sub(g, [0, 1])
        s.add_edge(0, 1, weight=999.0, edge_type="x",
                   semantic_weight=0.0, motion_weight=0.0, temporal_weight=0.0)
        assert g._add_weighted_edge_to_graph(s, 0, 1, "cross_scene_all", 1.0) == "unchanged"
        assert s[0][1]["weight"] == 999.0

    def test_self_loop_returns_unchanged(self):
        g = _build_graph([0])
        s = _sub(g, [0])
        assert g._add_weighted_edge_to_graph(s, 0, 0, "cross_scene_all", 0.0) == "unchanged"

    def test_missing_node_raises_key_error(self):
        g = _build_graph([0, 1])
        s = _sub(g, [0])
        with pytest.raises(KeyError, match="not present in the target graph"):
            g._add_weighted_edge_to_graph(s, 0, 1, "cross_scene_all", 0.0)

    def test_does_not_touch_production_graph(self):
        g = _build_graph([0, 1])
        before = set(g.graph.edges())
        s = g.graph.subgraph([0, 1]).copy()
        if s.has_edge(0, 1): s.remove_edge(0, 1)
        g._add_weighted_edge_to_graph(s, 0, 1, "cross_scene_all", 1.0)
        assert set(g.graph.edges()) == before


# ---------------------------------------------------------------------------
# Part 2: add_cross_scene_edges - mode=all
# ---------------------------------------------------------------------------


class TestCrossSceneAll:

    def test_count_equals_cross_scene_pairs(self):
        n_per = 3
        g, sub, pool, sof, anch = _two_scene(n_per)
        sub.remove_edges_from(list(sub.edges()))
        new = g.add_cross_scene_edges(pool, sof, mode="all",
            threshold_percentile=50.0, scene_anchors=anch, graph=sub)
        assert new == n_per * n_per

    def test_edge_type(self):
        g, sub, pool, sof, anch = _two_scene(2)
        sub.remove_edges_from(list(sub.edges()))
        g.add_cross_scene_edges(pool, sof, mode="all",
            threshold_percentile=0.0, scene_anchors=anch, graph=sub)
        for u, v in sub.edges():
            assert sub[u][v]["edge_type"] == "cross_scene_all"

    def test_no_same_scene_edges(self):
        g, sub, pool, sof, anch = _two_scene(3)
        sub.remove_edges_from(list(sub.edges()))
        g.add_cross_scene_edges(pool, sof, mode="all",
            threshold_percentile=0.0, scene_anchors=anch, graph=sub)
        for u, v in sub.edges():
            assert sof[u] != sof[v]

    def test_prod_graph_unchanged(self):
        g, sub, pool, sof, anch = _two_scene(3)
        before = set(g.graph.edges())
        g.add_cross_scene_edges(pool, sof, mode="all",
            threshold_percentile=50.0, scene_anchors=anch, graph=sub)
        assert set(g.graph.edges()) == before


# ---------------------------------------------------------------------------
# Part 2: add_cross_scene_edges - mode=rep_only
# ---------------------------------------------------------------------------


class TestCrossSceneRepOnly:

    def test_two_scenes_exactly_one_edge(self):
        g, sub, pool, sof, anch = _two_scene(4)
        sub.remove_edges_from(list(sub.edges()))
        new = g.add_cross_scene_edges(pool, sof, mode="rep_only",
            threshold_percentile=0.0, scene_anchors=anch, graph=sub)
        assert new == 1

    def test_edge_type(self):
        g, sub, pool, sof, anch = _two_scene(3)
        sub.remove_edges_from(list(sub.edges()))
        g.add_cross_scene_edges(pool, sof, mode="rep_only",
            threshold_percentile=0.0, scene_anchors=anch, graph=sub)
        for u, v in sub.edges():
            assert sub[u][v]["edge_type"] == "cross_scene_rep"

    def test_not_all_pairs(self):
        g, sub, pool, sof, anch = _two_scene(5)
        sub.remove_edges_from(list(sub.edges()))
        new = g.add_cross_scene_edges(pool, sof, mode="rep_only",
            threshold_percentile=0.0, scene_anchors=anch, graph=sub)
        assert new == 1  # S*(S-1)/2 = 1, not 5*5=25

    def test_prod_graph_unchanged(self):
        g, sub, pool, sof, anch = _two_scene(3)
        before = set(g.graph.edges())
        g.add_cross_scene_edges(pool, sof, mode="rep_only",
            threshold_percentile=0.0, scene_anchors=anch, graph=sub)
        assert set(g.graph.edges()) == before


# ---------------------------------------------------------------------------
# Part 2: add_cross_scene_edges - mode=threshold
# ---------------------------------------------------------------------------


class TestCrossSceneThreshold:

    def test_orthogonal_produces_zero_edges(self):
        n_per = 3
        sa = [0]*n_per + [1]*n_per
        embs = _orth_embs(n_per*2)
        g = _build_graph(sa, embs)
        pool = list(range(n_per*2))
        sub = _sub(g, pool)
        sof = {i: sa[i] for i in pool}
        anch = {0: 0, 1: n_per}
        sub.remove_edges_from(list(sub.edges()))
        new = g.add_cross_scene_edges(pool, sof, mode="threshold",
            threshold_percentile=75.0, scene_anchors=anch, graph=sub)
        assert new == 0

    def test_similar_embs_at_pctile_zero(self):
        n_per = 2
        sa = [0]*n_per + [1]*n_per
        raw = [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.99, 0.14, 0.0, 0.0], dtype=np.float32),
            np.array([0.98, 0.0, 0.2, 0.0], dtype=np.float32),
            np.array([0.97, 0.0, 0.0, 0.24], dtype=np.float32),
        ]
        embs = [e / np.linalg.norm(e) for e in raw]
        g = _build_graph(sa, embs)
        pool = list(range(n_per*2))
        sub = _sub(g, pool)
        sof = {i: sa[i] for i in pool}
        anch = {0: 0, 1: 2}
        sub.remove_edges_from(list(sub.edges()))
        new = g.add_cross_scene_edges(pool, sof, mode="threshold",
            threshold_percentile=0.0, scene_anchors=anch, graph=sub)
        assert new > 0
        for u, v in sub.edges():
            assert sub[u][v]["edge_type"] == "cross_scene_threshold"

    def test_prod_graph_unchanged(self):
        g, sub, pool, sof, anch = _two_scene(3)
        before = set(g.graph.edges())
        g.add_cross_scene_edges(pool, sof, mode="threshold",
            threshold_percentile=50.0, scene_anchors=anch, graph=sub)
        assert set(g.graph.edges()) == before


# ---------------------------------------------------------------------------
# Part 2: add_cross_scene_edges - validation
# ---------------------------------------------------------------------------


class TestCrossSceneValidation:

    def test_bad_mode(self):
        g, sub, pool, sof, anch = _two_scene(2)
        with pytest.raises(ValueError, match="mode must be one of"):
            g.add_cross_scene_edges(pool, sof, mode="bad",
                threshold_percentile=50.0, scene_anchors=anch, graph=sub)

    def test_pctile_101(self):
        g, sub, pool, sof, anch = _two_scene(2)
        with pytest.raises(ValueError, match="threshold_percentile must be in"):
            g.add_cross_scene_edges(pool, sof, mode="threshold",
                threshold_percentile=101.0, scene_anchors=anch, graph=sub)

    def test_pctile_neg1(self):
        g, sub, pool, sof, anch = _two_scene(2)
        with pytest.raises(ValueError, match="threshold_percentile must be in"):
            g.add_cross_scene_edges(pool, sof, mode="threshold",
                threshold_percentile=-1.0, scene_anchors=anch, graph=sub)

    def test_pctile_boundaries_ok(self):
        g, sub, pool, sof, anch = _two_scene(2)
        for p in (0.0, 100.0):
            sc = _sub(g, pool)
            g.add_cross_scene_edges(pool, sof, mode="threshold",
                threshold_percentile=p, scene_anchors=anch, graph=sc)

    def test_missing_pool_node(self):
        g, sub, pool, sof, anch = _two_scene(2)
        bad_pool = pool + [9999]
        bad_sof = dict(sof); bad_sof[9999] = 0
        with pytest.raises(KeyError, match="not a node in the supplied graph"):
            g.add_cross_scene_edges(bad_pool, bad_sof, mode="all",
                threshold_percentile=50.0, scene_anchors=anch, graph=sub)

    def test_missing_scene_of_entry(self):
        g, sub, pool, sof, anch = _two_scene(2)
        bad = {k: v for k, v in sof.items() if k != pool[-1]}
        with pytest.raises(KeyError, match="has no entry in scene_of"):
            g.add_cross_scene_edges(pool, bad, mode="all",
                threshold_percentile=50.0, scene_anchors=anch, graph=sub)

    def test_anchor_wrong_scene(self):
        g, sub, pool, sof, anch = _two_scene(3)
        bad_anch = {0: pool[0], 1: pool[0]}  # frame 0 is scene 0, not scene 1
        with pytest.raises(ValueError, match="scene_anchors declares frame"):
            g.add_cross_scene_edges(pool, sof, mode="rep_only",
                threshold_percentile=0.0, scene_anchors=bad_anch, graph=sub)


# ---------------------------------------------------------------------------
# Part 3: retrieve_ppr with graph_override
# ---------------------------------------------------------------------------


class TestRetrievePPROverride:

    def _g(self, n_per=4):
        n = n_per * 2
        sa = [0]*n_per + [1]*n_per
        embs = _orth_embs(n)
        return _build_graph(sa, embs), embs, n

    def test_none_override_same_as_default(self):
        g, embs, n = self._g()
        q = embs[0].copy()
        r1 = [nd.frame_idx for nd in g.retrieve_ppr(q, top_k=n)]
        r2 = [nd.frame_idx for nd in g.retrieve_ppr(q, top_k=n, graph_override=None)]
        assert r1 == r2

    def test_override_results_in_pool(self):
        g, embs, n = self._g()
        pool = [0, 1]
        s = _sub(g, pool)
        res = g.retrieve_ppr(embs[0].copy(), top_k=10, graph_override=s)
        assert {nd.frame_idx for nd in res}.issubset(set(pool))
        assert {nd.frame_idx for nd in res}.isdisjoint(set(range(2, n)))

    def test_production_graph_unchanged(self):
        g, embs, n = self._g(3)
        s = _sub(g, [0, n//2])
        before = set(g.graph.edges())
        g.retrieve_ppr(embs[0].copy(), top_k=2, graph_override=s)
        assert set(g.graph.edges()) == before

    def test_missing_node_data_raises(self):
        g, embs, _ = self._g(3)
        bad = nx.Graph(); bad.add_node(999)
        with pytest.raises(KeyError, match="node_data"):
            g.retrieve_ppr(embs[0].copy(), top_k=1, graph_override=bad)

    def test_zero_query_returns_empty(self):
        g, embs, _ = self._g(3)
        s = _sub(g, [0, 1])
        assert g.retrieve_ppr(np.zeros_like(embs[0]), top_k=5, graph_override=s) == []

    def test_ppr_sums_to_one(self):
        g, embs, n = self._g(4)
        pool = list(range(n))
        s = _sub(g, pool)
        sof = {i: (0 if i < n//2 else 1) for i in pool}
        anch = {0: 0, 1: n//2}
        g.add_cross_scene_edges(pool, sof, mode="all",
            threshold_percentile=0.0, scene_anchors=anch, graph=s)
        nodes = g.retrieve_ppr(embs[0].copy(), top_k=n, graph_override=s)
        total = sum(nd.last_retrieval_score for nd in nodes)
        assert abs(total - 1.0) < 1e-4, f"PPR must sum to ~1, got {total}"

    def test_determinism(self):
        g, embs, n = self._g(4)
        s = _sub(g, list(range(n)))
        q = embs[1].copy()
        r1 = [nd.frame_idx for nd in g.retrieve_ppr(q, top_k=n, graph_override=s)]
        r2 = [nd.frame_idx for nd in g.retrieve_ppr(q, top_k=n, graph_override=s)]
        assert r1 == r2


# ---------------------------------------------------------------------------
# Part 4: End-to-end DESCEND via retrieve_scene_sparse
# ---------------------------------------------------------------------------


class _Cfg:
    scene_shortlist_width = 0
    scene_shortcut_margin = 1.0
    scene_neighbor_window = 30
    scene_crossscene_mode = "rep_only"
    scene_crossscene_threshold_pctile = 75.0
    ppr_damping = 0.5
    ppr_lambda = 0.5
    l2_retrieve_top_k = 3
    scene_diag = False


class _FR:
    def __init__(self, frame_idx, scene_id, clip_embedding, action_score=0.5,
                 timestamp=0.0, luma_diff_energy=0.0, persistence_value=0.2,
                 is_peak=False, luma_entropy=0.0, caption=None):
        self.frame_idx = frame_idx
        self.scene_id = scene_id
        self.clip_embedding = clip_embedding
        self.action_score = action_score
        self.timestamp = timestamp
        self.luma_diff_energy = luma_diff_energy
        self.persistence_value = persistence_value
        self.is_peak = is_peak
        self.luma_entropy = luma_entropy
        self.caption = caption


class _Idx:
    def __init__(self, frames, graph, centroids):
        self.frames = frames
        self._graph = graph
        self._scene_centroids = centroids


def _descend_scenario(n_per=4):
    # Two scenes with very similar centroids: margin < tau forces DESCEND.
    n = n_per * 2
    sa = [0]*n_per + [1]*n_per
    dim = 8
    e0 = [np.array([1.0]+[0.0]*(dim-1), dtype=np.float32)] * n_per
    e1 = [np.array([0.9, 0.436]+[0.0]*(dim-2), dtype=np.float32)] * n_per
    all_e = [e / np.linalg.norm(e) for e in e0 + e1]
    graph = _build_graph(sa, all_e)
    frames = [_FR(i, sa[i], all_e[i], 0.1*(i+1), float(i)) for i in range(n)]
    centroids = {}
    for sid in (0, 1):
        se = np.stack([f.clip_embedding for f in frames if f.scene_id == sid])
        centroids[sid] = np.mean(se, axis=0).astype(np.float32)
    return _Idx(frames, graph, centroids), graph


class TestEndToEndDescend:
    _Q = np.array([1.0]+[0.0]*7, dtype=np.float32)

    def test_completes_without_error(self):
        idx, _ = _descend_scenario()
        assert isinstance(retrieve_scene_sparse(idx, self._Q, _Cfg()), list)

    def test_result_bounded_by_top_k(self):
        idx, _ = _descend_scenario()
        cfg = _Cfg()
        assert len(retrieve_scene_sparse(idx, self._Q, cfg)) <= cfg.l2_retrieve_top_k

    def test_production_graph_not_mutated(self):
        idx, graph = _descend_scenario()
        before = frozenset(graph.graph.edges())
        retrieve_scene_sparse(idx, self._Q, _Cfg())
        assert frozenset(graph.graph.edges()) == before

    def test_result_has_required_keys(self):
        idx, _ = _descend_scenario()
        result = retrieve_scene_sparse(idx, self._Q, _Cfg())
        required = {"frame_idx", "timestamp", "action_score",
                    "last_retrieval_score", "retrieval_contributions"}
        for fd in result:
            assert not (required - set(fd.keys()))

    def test_all_modes_without_error(self):
        for mode in ("all", "threshold", "rep_only"):
            idx, _ = _descend_scenario(n_per=3)
            cfg = _Cfg()
            cfg.scene_crossscene_mode = mode
            assert isinstance(retrieve_scene_sparse(idx, self._Q, cfg), list)
