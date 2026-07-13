"""
Unit tests for L1 Elysium context cache.

Tests are grouped into four areas:
    1. Admission basics
    2. Eviction correctness
    3. Query and similarity ranking
    4. PageRank injection

Owner: Track A
"""
from __future__ import annotations

import numpy as np
import pytest

from iris.cached_frame import CachedFrame
from iris.frame_motion_descriptor import FrameMotionDescriptor
from iris.iris_config import IRISConfig
from iris.l1_elysium import L1ElysiumCache


# ── Helpers ────────────────────────────────────────────────────────────────

def make_frame(
    frame_idx: int,
    action_score: float = 0.5,
    persistence_value: float = 0.5,
    motion_entropy: float = 0.0,
    hessian: float = 0.0,
    embedding: np.ndarray | None = None,
) -> CachedFrame:
    """Build a CachedFrame with sensible defaults for testing."""
    motion = FrameMotionDescriptor(
        frame_idx=frame_idx,
        timestamp_sec=float(frame_idx),
        motion_entropy=motion_entropy,
        hessian_max_eigenvalue=hessian,
    )
    return CachedFrame(
        frame_idx=frame_idx,
        timestamp_sec=float(frame_idx),
        action_score=action_score,
        persistence_value=persistence_value,
        is_peak=action_score >= 0.5,
        motion=motion,
        embedding=embedding,
    )


# ── 1. Admission basics ────────────────────────────────────────────────────

def test_admit_single_frame():
    cache = L1ElysiumCache()
    cache.admit(make_frame(0))
    assert len(cache) == 1
    assert 0 in cache


def test_admit_sets_admitted_at():
    cache = L1ElysiumCache()
    cache.admit(make_frame(0))
    cache.admit(make_frame(1))
    assert cache._frames[0].admitted_at == 0
    assert cache._frames[1].admitted_at == 1


def test_admit_increments_counter():
    cache = L1ElysiumCache()
    cache.admit(make_frame(0))
    cache.admit(make_frame(1))
    assert cache._admission_counter == 2


def test_readmit_same_frame_idx_no_growth():
    """Re-admitting a frame with the same index replaces it, not duplicates it."""
    cache = L1ElysiumCache()
    cache.admit(make_frame(7))
    cache.admit(make_frame(7))
    assert len(cache) == 1
    assert cache._admission_counter == 1   # counter only bumped once


def test_is_full_flag():
    cfg = IRISConfig(l1_capacity=2)
    cache = L1ElysiumCache(config=cfg)
    cache.admit(make_frame(0))
    assert not cache.is_full
    cache.admit(make_frame(1))
    assert cache.is_full


# ── 2. Eviction correctness ────────────────────────────────────────────────

def test_eviction_removes_lowest_action_score():
    """
    With capacity=3 and all other signals equal,
    the frame with the lowest action_score is evicted first.
    """
    cfg = IRISConfig(l1_capacity=3)
    cache = L1ElysiumCache(config=cfg)

    cache.admit(make_frame(0, action_score=0.1))
    cache.admit(make_frame(1, action_score=0.5))
    cache.admit(make_frame(2, action_score=0.9))

    # Admitting a 4th triggers eviction of the lowest-score frame
    cache.admit(make_frame(3, action_score=0.8))

    assert 0 not in cache, "frame 0 (lowest action_score) should have been evicted"
    assert 3 in cache


def test_eviction_respects_keep_score_not_just_insertion_order():
    """
    A high action_score frame admitted early should survive
    over a low action_score frame admitted later.
    """
    cfg = IRISConfig(l1_capacity=2)
    cache = L1ElysiumCache(config=cfg)

    # Frame 0 admitted first but has very high score
    cache.admit(make_frame(0, action_score=0.95))
    # Frame 1 admitted second but has very low score
    cache.admit(make_frame(1, action_score=0.05))

    # Admit frame 2 — triggers eviction
    cache.admit(make_frame(2, action_score=0.7))

    # Frame 0 (high score) must survive even though it's older
    assert 0 in cache, "high-score frame should not be evicted over a low-score frame"
    assert 1 not in cache, "low-score frame should be evicted first"


# ── 3. Query and similarity ranking ───────────────────────────────────────

def test_query_empty_cache():
    cache = L1ElysiumCache()
    result = cache.query(np.array([1.0, 0.0], dtype=np.float32))
    assert result == []


def test_query_none_embedding_gets_zero_similarity():
    cache = L1ElysiumCache()
    cache.admit(make_frame(0))   # embedding=None by default
    results = cache.query(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
    assert results[0].query_similarity == 0.0


def test_query_ranking_similarity_dominates_when_gap_is_large():
    """
    When query_similarity differs by a large margin, the more similar
    frame ranks first despite being older (lower recency).

    This tests the multi-signal balance: a 0.80 similarity gap
    (0.20 weight) = 0.16 advantage, which beats any recency difference
    (0.05 weight × at most 1.0 = 0.05 max).
    """
    cache = L1ElysiumCache()

    # Frame 0 (admitted first — lower recency): nearly identical to query
    f0_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    cache.admit(make_frame(0, action_score=0.5, embedding=f0_emb))

    # Frame 1 (admitted second — higher recency): very different from query
    f1_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cache.admit(make_frame(1, action_score=0.5, embedding=f1_emb))

    query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = cache.query(query_emb, top_k=2)

    assert results[0].frame_idx == 0, (
        "frame 0 (sim=1.0) should rank above frame 1 (sim=0.0) "
        "even though frame 0 is older"
    )


def test_query_updates_similarity_on_frames():
    """After query(), frame.query_similarity should reflect the result."""
    cache = L1ElysiumCache()
    emb = np.array([1.0, 0.0], dtype=np.float32)
    cache.admit(make_frame(0, embedding=emb))

    query_emb = np.array([1.0, 0.0], dtype=np.float32)
    cache.query(query_emb)

    assert cache._frames[0].query_similarity == pytest.approx(1.0, abs=1e-5)


def test_query_top_k_limits_results():
    cache = L1ElysiumCache()
    for i in range(5):
        cache.admit(make_frame(i))

    results = cache.query(np.array([1.0, 0.0], dtype=np.float32), top_k=3)
    assert len(results) == 3


# ── 4. PageRank injection ──────────────────────────────────────────────────

def test_update_pagerank_sets_values():
    cache = L1ElysiumCache()
    cache.admit(make_frame(0))
    cache.admit(make_frame(1))

    cache.update_pagerank({0: 0.8, 1: 0.3})

    assert cache._frames[0].pagerank == pytest.approx(0.8)
    assert cache._frames[1].pagerank == pytest.approx(0.3)


def test_update_pagerank_ignores_missing_frames():
    """Frame indices not in L1 should not crash — silently ignored."""
    cache = L1ElysiumCache()
    cache.admit(make_frame(0))

    # frame_idx 99 is not in L1 — should not raise
    cache.update_pagerank({0: 0.5, 99: 0.9})

    assert cache._frames[0].pagerank == pytest.approx(0.5)


def test_pagerank_affects_keep_score():
    """A frame with high pagerank should score higher than an identical frame without."""
    cache = L1ElysiumCache()
    cache.admit(make_frame(0))
    cache.admit(make_frame(1))

    cache.update_pagerank({0: 1.0, 1: 0.0})

    score_0 = cache._keep_score(cache._frames[0])
    score_1 = cache._keep_score(cache._frames[1])

    assert score_0 > score_1


# ── 5. Dual-vector representation (Contribution 3) ────────────────────────

def make_frame_with_motion(
    frame_idx: int,
    action_score: float = 0.5,
    persistence_value: float = 0.5,
    luma_diff_energy: float = 1.0,
    divergence: float = 0.5,
    curl: float = 0.3,
    jacobian_frobenius: float = 0.8,
    hessian: float = 0.6,
    motion_entropy: float = 0.7,
    embedding: np.ndarray | None = None,
) -> CachedFrame:
    """Build a CachedFrame with non-trivial FrameMotionDescriptor for dual-vector tests."""
    motion = FrameMotionDescriptor(
        frame_idx=frame_idx,
        timestamp_sec=float(frame_idx),
        luma_diff_energy=luma_diff_energy,
        divergence=divergence,
        curl=curl,
        jacobian_frobenius=jacobian_frobenius,
        hessian_max_eigenvalue=hessian,
        motion_entropy=motion_entropy,
    )
    frame = CachedFrame(
        frame_idx=frame_idx,
        timestamp_sec=float(frame_idx),
        action_score=action_score,
        persistence_value=persistence_value,
        is_peak=action_score >= 0.5,
        motion=motion,
        embedding=embedding,
    )
    frame.build_motion_embedding()
    return frame


def test_build_motion_embedding_shape_and_norm():
    """build_motion_embedding() should produce a 6-D unit vector."""
    frame = make_frame_with_motion(0)
    assert frame.motion_embedding is not None
    assert frame.motion_embedding.shape == (6,)
    assert frame.motion_embedding.dtype == np.float32

    norm = float(np.linalg.norm(frame.motion_embedding))
    assert abs(norm - 1.0) < 1e-5, f"Expected unit vector, got norm={norm}"


def test_build_motion_embedding_zero_vector():
    """All-zero FrameMotionDescriptor should produce a zero motion_embedding."""
    motion = FrameMotionDescriptor(
        frame_idx=0,
        timestamp_sec=0.0,
        luma_diff_energy=0.0,
        divergence=0.0,
        curl=0.0,
        jacobian_frobenius=0.0,
        hessian_max_eigenvalue=0.0,
        motion_entropy=0.0,
    )
    frame = CachedFrame(
        frame_idx=0,
        timestamp_sec=0.0,
        action_score=0.5,
        persistence_value=0.5,
        is_peak=False,
        motion=motion,
    )
    frame.build_motion_embedding()
    assert frame.motion_embedding is not None
    assert float(np.linalg.norm(frame.motion_embedding)) < 1e-8


def test_dual_vector_query_combines_similarities():
    """When query_motion_embedding is provided, ranking should reflect
    both visual and motion similarity."""
    config = IRISConfig(l1_visual_query_weight=0.50, l1_motion_query_weight=0.50)
    cache = L1ElysiumCache(config=config)

    # Frame 0: good visual match, bad motion match
    f0 = make_frame_with_motion(
        0,
        luma_diff_energy=0.0, divergence=0.0, curl=0.0,
        jacobian_frobenius=0.0, hessian=0.0, motion_entropy=1.0,
        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    cache.admit(f0)

    # Frame 1: bad visual match, good motion match
    f1 = make_frame_with_motion(
        1,
        luma_diff_energy=1.0, divergence=0.5, curl=0.3,
        jacobian_frobenius=0.8, hessian=0.6, motion_entropy=0.7,
        embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    cache.admit(f1)

    query_visual = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # Motion query that matches frame 1's motion signature
    query_motion = f1.motion_embedding.copy()

    results = cache.query(query_visual, top_k=2, query_motion_embedding=query_motion)

    # With 50/50 weighting:
    # Frame 0: visual=1.0, motion=low → blended ~0.5
    # Frame 1: visual=0.0, motion=1.0 → blended ~0.5
    # The key test: both similarities were actually computed (non-zero query_similarity)
    assert all(r.query_similarity > 0.0 for r in results), (
        "Both frames should have non-zero combined similarity"
    )


def test_query_backward_compatible_without_motion():
    """Existing callers without query_motion_embedding should work identically."""
    cache = L1ElysiumCache()

    emb = np.array([1.0, 0.0], dtype=np.float32)
    cache.admit(make_frame(0, embedding=emb))

    query_emb = np.array([1.0, 0.0], dtype=np.float32)
    results = cache.query(query_emb, top_k=1)

    assert results[0].query_similarity == pytest.approx(1.0, abs=1e-5)


def test_dual_vector_gepa_weight_shift():
    """Shifting weights toward motion should change rankings."""
    # Visual-heavy config
    config_v = IRISConfig(
        l1_visual_query_weight=0.90, l1_motion_query_weight=0.10,
        l1_w_recency=0.0, l1_w_entropy=0.0, l1_w_hessian=0.0
    )
    # Motion-heavy config
    config_m = IRISConfig(
        l1_visual_query_weight=0.10, l1_motion_query_weight=0.90,
        l1_w_recency=0.0, l1_w_entropy=0.0, l1_w_hessian=0.0
    )

    for cfg, expected_first in [(config_v, 0), (config_m, 1)]:
        cache = L1ElysiumCache(config=cfg)

        # Frame 0: perfect visual match, zero motion
        f0 = make_frame_with_motion(
            0,
            luma_diff_energy=0.0, divergence=0.0, curl=0.0,
            jacobian_frobenius=0.0, hessian=0.0, motion_entropy=0.0,
            embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        # Force motion embedding to zero (all-zero descriptor)
        cache.admit(f0)

        # Frame 1: zero visual match, strong motion
        f1 = make_frame_with_motion(
            1,
            luma_diff_energy=1.0, divergence=0.5, curl=0.3,
            jacobian_frobenius=0.8, hessian=0.6, motion_entropy=0.7,
            embedding=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        cache.admit(f1)

        query_visual = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query_motion = f1.motion_embedding.copy()

        results = cache.query(query_visual, top_k=2, query_motion_embedding=query_motion)
        assert results[0].frame_idx == expected_first, (
            f"With visual={cfg.l1_visual_query_weight}, motion={cfg.l1_motion_query_weight}, "
            f"expected frame {expected_first} first but got frame {results[0].frame_idx}"
        )
