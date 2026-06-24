"""
Unit tests for L2TieredIndex — codec-tier-aware FAISS indexing.

Tests verify:
    1. Tier routing (PEAK → FlatIP, SALIENT → HNSW, CANDIDATE → PQ/FlatIP)
    2. Search merging across tiers
    3. PQ training and fallback behavior
    4. Memory reduction vs flat index
    5. Edge cases (empty index, single frame)

Owner: Track A
"""
from __future__ import annotations

import numpy as np
import pytest

from iris_config import IRISConfig
from l2_index import L2TieredIndex, FrameTier


# ── Helpers ────────────────────────────────────────────────────────────────

DIM = 64  # Use small dim for fast tests (must be divisible by pq_m=8)


def make_config(**overrides) -> IRISConfig:
    """Build an IRISConfig suitable for L2 index tests."""
    defaults = {
        "l2_embed_dim": DIM,
        "l2_hnsw_m": 16,
        "l2_hnsw_ef_search": 32,
        "l2_pq_m": 8,
        "l2_pq_nbits": 8,
        "l2_pq_min_train": 10,  # Low threshold for test speed
        "l2_salient_action_thresh": 0.35,
    }
    defaults.update(overrides)
    return IRISConfig(**defaults)


def random_embedding(dim: int = DIM) -> np.ndarray:
    """Generate a random float32 embedding of given dimensionality."""
    vec = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec


# ── 1. Tier routing ───────────────────────────────────────────────────────

def test_add_peak_to_flat_index():
    """Peak frames should be routed to the PEAK (FlatIP) tier."""
    cfg = make_config()
    idx = L2TieredIndex(cfg)

    tier = idx.add(
        frame_idx=10,
        embedding=random_embedding(),
        action_score=0.95,
        is_peak=True,
    )
    assert tier == FrameTier.PEAK

    stats = idx.stats()
    assert stats["peak"]["count"] == 1
    assert stats["salient"]["count"] == 0
    assert stats["candidate"]["count"] == 0


def test_add_salient_to_hnsw_index():
    """Non-peak frames above salient threshold should go to HNSW."""
    cfg = make_config(l2_salient_action_thresh=0.35)
    idx = L2TieredIndex(cfg)

    tier = idx.add(
        frame_idx=20,
        embedding=random_embedding(),
        action_score=0.50,
        is_peak=False,
    )
    assert tier == FrameTier.SALIENT

    stats = idx.stats()
    assert stats["salient"]["count"] == 1


def test_add_candidate_to_candidate_tier():
    """Low action_score non-peak frames should go to CANDIDATE."""
    cfg = make_config(l2_salient_action_thresh=0.35)
    idx = L2TieredIndex(cfg)

    tier = idx.add(
        frame_idx=30,
        embedding=random_embedding(),
        action_score=0.10,
        is_peak=False,
    )
    assert tier == FrameTier.CANDIDATE

    stats = idx.stats()
    assert stats["candidate"]["count"] == 1


def test_tier_routing_boundary():
    """Frame at exactly salient_thresh (non-peak) should be SALIENT."""
    cfg = make_config(l2_salient_action_thresh=0.35)
    idx = L2TieredIndex(cfg)

    tier = idx.add(
        frame_idx=40,
        embedding=random_embedding(),
        action_score=0.35,  # exactly at threshold
        is_peak=False,
    )
    assert tier == FrameTier.SALIENT


def test_peak_overrides_action_score():
    """is_peak=True should always route to PEAK, regardless of action_score."""
    cfg = make_config()
    idx = L2TieredIndex(cfg)

    tier = idx.add(
        frame_idx=50,
        embedding=random_embedding(),
        action_score=0.10,  # Low score but flagged as peak
        is_peak=True,
    )
    assert tier == FrameTier.PEAK


# ── 2. Search merging ────────────────────────────────────────────────────

def test_search_returns_results_from_peak():
    """Search should return results from the PEAK tier."""
    cfg = make_config()
    idx = L2TieredIndex(cfg)

    emb = random_embedding()
    idx.add(frame_idx=10, embedding=emb, action_score=0.95, is_peak=True)

    results = idx.search(emb, top_k=5)
    assert len(results) == 1
    assert results[0]["frame_idx"] == 10
    assert results[0]["tier"] == "PEAK"
    assert results[0]["score"] > 0.0


def test_search_merges_across_tiers():
    """Search should merge results from PEAK and SALIENT tiers."""
    cfg = make_config()
    idx = L2TieredIndex(cfg)

    # Add one PEAK frame
    peak_emb = random_embedding()
    idx.add(frame_idx=10, embedding=peak_emb, action_score=0.95, is_peak=True)

    # Add one SALIENT frame (similar embedding)
    idx.add(frame_idx=20, embedding=peak_emb * 0.99, action_score=0.50, is_peak=False)

    results = idx.search(peak_emb, top_k=5)
    frame_ids = {r["frame_idx"] for r in results}
    assert 10 in frame_ids
    assert 20 in frame_ids


def test_search_empty_index():
    """Searching an empty index should return an empty list."""
    cfg = make_config()
    idx = L2TieredIndex(cfg)

    results = idx.search(random_embedding(), top_k=5)
    assert results == []


def test_search_respects_top_k():
    """Should not return more than top_k results."""
    cfg = make_config()
    idx = L2TieredIndex(cfg)

    for i in range(10):
        idx.add(frame_idx=i, embedding=random_embedding(), action_score=0.95, is_peak=True)

    results = idx.search(random_embedding(), top_k=3)
    assert len(results) <= 3


# ── 3. PQ training ───────────────────────────────────────────────────────

def test_pq_auto_trains_at_threshold():
    """PQ should auto-train when buffer reaches effective minimum (2^nbits for 8-bit = 256)."""
    cfg = make_config(l2_pq_min_train=256)  # matches 2^8
    idx = L2TieredIndex(cfg)

    # Add 255 candidates — should stay in buffer (below 2^8 = 256)
    for i in range(255):
        idx.add(frame_idx=i, embedding=random_embedding(), action_score=0.05, is_peak=False)

    stats = idx.stats()
    assert not stats["candidate"]["pq_trained"]
    assert stats["candidate"]["buffer_pending"] == 255

    # Add 256th — should trigger PQ training
    idx.add(frame_idx=255, embedding=random_embedding(), action_score=0.05, is_peak=False)

    stats = idx.stats()
    assert stats["candidate"]["pq_trained"]
    assert stats["candidate"]["buffer_pending"] == 0
    assert stats["candidate"]["count"] == 256


def test_force_train_pq_fallback_to_flat():
    """force_train_pq() should fall back to FlatIP when buffer is too small for PQ."""
    cfg = make_config(l2_pq_min_train=1000)  # Impossibly high
    idx = L2TieredIndex(cfg)

    # Add just 5 candidates
    for i in range(5):
        idx.add(frame_idx=i, embedding=random_embedding(), action_score=0.05, is_peak=False)

    idx.force_train_pq()

    stats = idx.stats()
    assert stats["candidate"]["pq_trained"]
    assert stats["candidate"]["count"] == 5

    # Search should still work
    results = idx.search(random_embedding(), top_k=3)
    assert len(results) <= 3


def test_candidate_searchable_after_force_train():
    """Candidate frames should be searchable after force_train_pq()."""
    cfg = make_config(l2_pq_min_train=1000)
    idx = L2TieredIndex(cfg)

    target_emb = random_embedding()
    idx.add(frame_idx=42, embedding=target_emb, action_score=0.05, is_peak=False)

    # Before force_train, search triggers it automatically
    results = idx.search(target_emb, top_k=1)
    assert len(results) == 1
    assert results[0]["frame_idx"] == 42


# ── 4. Batch operations ──────────────────────────────────────────────────

def test_add_batch():
    """add_batch should correctly route multiple frames."""
    cfg = make_config()
    idx = L2TieredIndex(cfg)

    embeddings = np.vstack([random_embedding() for _ in range(5)])
    frame_indices = [100, 101, 102, 103, 104]
    action_scores = [0.95, 0.50, 0.10, 0.80, 0.02]
    is_peaks = [True, False, False, False, False]

    tiers = idx.add_batch(frame_indices, embeddings, action_scores, is_peaks)

    assert tiers[0] == FrameTier.PEAK       # 0.95, is_peak
    assert tiers[1] == FrameTier.SALIENT    # 0.50
    assert tiers[2] == FrameTier.CANDIDATE  # 0.10
    assert tiers[3] == FrameTier.SALIENT    # 0.80
    assert tiers[4] == FrameTier.CANDIDATE  # 0.02


# ── 5. Stats and memory ──────────────────────────────────────────────────

def test_stats_reports_correct_counts():
    """stats() should accurately reflect frame counts per tier."""
    cfg = make_config()
    idx = L2TieredIndex(cfg)

    idx.add(frame_idx=1, embedding=random_embedding(), action_score=0.95, is_peak=True)
    idx.add(frame_idx=2, embedding=random_embedding(), action_score=0.95, is_peak=True)
    idx.add(frame_idx=3, embedding=random_embedding(), action_score=0.50, is_peak=False)
    idx.add(frame_idx=4, embedding=random_embedding(), action_score=0.10, is_peak=False)

    stats = idx.stats()
    assert stats["total_frames"] == 4
    assert stats["peak"]["count"] == 2
    assert stats["salient"]["count"] == 1
    assert stats["candidate"]["count"] == 1


def test_memory_reduction_with_pq():
    """PQ tier should use significantly less memory than flat equivalent."""
    cfg = make_config(l2_pq_min_train=256, l2_embed_dim=DIM)
    idx = L2TieredIndex(cfg)

    # Add 300 candidate frames (>256 to trigger PQ auto-training)
    for i in range(300):
        idx.add(frame_idx=i, embedding=random_embedding(), action_score=0.05, is_peak=False)

    stats = idx.stats()
    assert stats["candidate"]["pq_trained"]

    # PQ memory: 300 × pq_m(8) = 2400 bytes
    # Flat memory: 300 × dim(64) × 4 = 76800 bytes
    candidate_mem = stats["candidate"]["memory_bytes"]
    flat_mem = 300 * DIM * 4

    assert candidate_mem < flat_mem, (
        f"PQ memory ({candidate_mem}) should be much less than flat ({flat_mem})"
    )
    reduction = flat_mem / candidate_mem
    assert reduction > 10, (
        f"Expected > 10× reduction, got {reduction:.1f}×"
    )


def test_total_indexed_property():
    """total_indexed should track all frames added."""
    cfg = make_config()
    idx = L2TieredIndex(cfg)

    assert idx.total_indexed == 0
    idx.add(frame_idx=1, embedding=random_embedding(), action_score=0.95, is_peak=True)
    assert idx.total_indexed == 1
    idx.add(frame_idx=2, embedding=random_embedding(), action_score=0.05, is_peak=False)
    assert idx.total_indexed == 2
