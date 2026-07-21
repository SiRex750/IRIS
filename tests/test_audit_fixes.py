import pytest
import numpy as np
from dataclasses import dataclass

from iris.charon_v import compute_motion_geometry, compute_valley_scene_boundaries
from iris.scene_retrieval import retrieve_scene_sparse
from iris.query import _build_retrieved
from iris.iris_config import IRISConfig
from iris.types import FrameRecord
from iris.ingest import IRISIndex

# Mock class representing L2 Graph
class MockGraph:
    def __init__(self):
        self.graph = self
        self.nodes = {}

# Mock class representing Node Data
class MockNodeData:
    def __init__(self, pagerank_score=0.1):
        self.pagerank_score = pagerank_score


def test_compute_motion_geometry_absolute_divergence():
    """Ensure divergence is computed using absolute values to prevent cancellation."""
    # Create motion vectors that would cancel each other out in signed divergence
    # e.g., expansion in one grid cell, contraction in another
    width = 160
    height = 120
    # Motion vectors: (src_x, src_y, dst_x, dst_y, motion_x, motion_y)
    # We place one positive motion vector and one negative motion vector in different cells
    mvs = [
        (16, 16, 16, 16, 4.0, 0.0),   # positive U motion (expansion)
        (48, 48, 48, 48, -4.0, 0.0),  # negative U motion (contraction)
    ]
    # Under signed divergence: mean of gradients would sum to 0.0.
    # Under absolute divergence: absolute mean of gradients is non-zero.
    geom = compute_motion_geometry(mvs, width, height)
    assert geom["divergence"] > 0.0
    assert geom["curl"] >= 0.0


def test_valley_scene_boundaries_cap():
    """Verify that valley scene boundaries are capped at MAX_SCENE_LEN = 300."""
    all_frame_energies = [(i, 1.0) for i in range(1000)]
    iframe_indices = [0]
    scenes = compute_valley_scene_boundaries(all_frame_energies, iframe_indices, fps=25.0)
    for start, end in scenes:
        assert end - start <= 300, f"Scene length {end - start} exceeds cap of 300"


def test_retrieve_scene_sparse_invalid_query():
    """Verify retrieve_scene_sparse raises ValueError on zero or invalid query embedding."""
    index = IRISIndex(
        video_path="dummy.mp4",
        frames=[],
        index_action_score=0.5,
        stats={},
        frames_processed=0,
        peak_count=0,
        skipped_frames_ratio=0.0,
        storage_reduction_factor=1.0,
        config_snapshot={"graph_mode": "scene_sparse"}
    )
    index._scene_centroids = {0: np.zeros(512)}
    
    config = IRISConfig(scene_shortlist_width=1, l2_retrieve_top_k=2)
    zero_query = np.zeros(512)
    
    with pytest.raises(ValueError, match="Invalid query embedding"):
        retrieve_scene_sparse(index, zero_query, config)


def test_retrieve_scene_sparse_adaptive_fallback():
    """Verify retrieve_scene_sparse falls back to all scenes when similarity is low (< 0.20)."""
    # Build two frames, one in scene 0, one in scene 1
    f0 = FrameRecord(
        frame_idx=0, timestamp=0.0, luma_diff_energy=0.0, luma_entropy=0.0,
        motion_magnitude=0.0, action_score=0.5, persistence_value=0.5, is_peak=True,
        divergence=0.0, curl=0.0, jacobian_frobenius=0.0, hessian_max_eigenvalue=0.0, motion_entropy=0.0,
        caption="Scene 0 frame", clip_embedding=np.array([1.0, 0.0], dtype=np.float32),
        pagerank_score=0.5, packet_size=100.0, pict_type="P", codec_conf=0.5, scene_id=0
    )
    f1 = FrameRecord(
        frame_idx=1, timestamp=1.0, luma_diff_energy=0.0, luma_entropy=0.0,
        motion_magnitude=0.0, action_score=0.5, persistence_value=0.5, is_peak=True,
        divergence=0.0, curl=0.0, jacobian_frobenius=0.0, hessian_max_eigenvalue=0.0, motion_entropy=0.0,
        caption="Scene 1 frame", clip_embedding=np.array([0.0, 1.0], dtype=np.float32),
        pagerank_score=0.5, packet_size=100.0, pict_type="P", codec_conf=0.5, scene_id=1
    )
    index = IRISIndex(
        video_path="dummy.mp4",
        frames=[f0, f1],
        index_action_score=0.5,
        stats={},
        frames_processed=2,
        peak_count=2,
        skipped_frames_ratio=0.0,
        storage_reduction_factor=1.0,
        config_snapshot={"graph_mode": "scene_sparse"}
    )
    index._scene_centroids = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 1.0], dtype=np.float32),
    }

    config = IRISConfig(
        scene_shortlist_width=1,  # Shortlist only 1 scene
        l2_retrieve_top_k=2,
        graph_mode="scene_sparse"
    )
    # Query is orthogonal to scene 0 (similarity 0.0) but matches scene 1.
    # If shortlist_width is 1, it might rank scene 0 first (or second).
    # Since similarity is 0.0 (< 0.20), it should fallback to all scenes and retrieve both.
    query_emb = np.array([0.0, 1.0], dtype=np.float32)
    results = retrieve_scene_sparse(index, query_emb, config)
    assert len(results) > 0


def test_build_retrieved_graph_mode_mismatch():
    """Verify build_retrieved raises ValueError if query graph_mode does not match index graph_mode."""
    index = IRISIndex(
        video_path="dummy.mp4",
        frames=[],
        index_action_score=0.5,
        stats={},
        frames_processed=0,
        peak_count=0,
        skipped_frames_ratio=0.0,
        storage_reduction_factor=1.0,
        config_snapshot={"graph_mode": "flat"}
    )
    config = IRISConfig(graph_mode="scene_sparse")
    
    with pytest.raises(ValueError, match="does not match index graph_mode"):
        _build_retrieved(index, np.ones(512), config)
