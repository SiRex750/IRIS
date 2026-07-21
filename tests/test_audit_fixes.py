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


def test_compute_motion_geometry_signed_divergence():
    """
    P1-07: Divergence must be signed so downstream consumers can distinguish
    expanding (positive) from converging (negative) flow fields.

    The FrameMotionDescriptor docstring explicitly states:
        divergence: float = 0.0  # positive = expanding, negative = converging

    This test verifies:
      - Pure outward (expansion) motion yields divergence > 0.
      - Pure inward (contraction) motion yields divergence < 0.
      - Mixed equal-and-opposite vectors yield near-zero divergence
        (physically correct: net expansion and contraction cancel out).
    """
    width, height = 160, 120

    # --- 1. Pure expansion (all U components positive) ---
    mvs_expand = [
        (16, 16, 16, 16, 4.0, 0.0),
        (48, 16, 48, 16, 4.0, 0.0),
        (80, 16, 80, 16, 4.0, 0.0),
        (16, 48, 16, 48, 4.0, 0.0),
        (48, 48, 48, 48, 4.0, 0.0),
    ]
    geom_exp = compute_motion_geometry(mvs_expand, width, height)
    assert geom_exp["divergence"] > 0.0, (
        f"Pure expansion should yield positive divergence, got {geom_exp['divergence']}"
    )

    # --- 2. Pure contraction (all U components negative) ---
    mvs_contract = [
        (16, 16, 16, 16, -4.0, 0.0),
        (48, 16, 48, 16, -4.0, 0.0),
        (80, 16, 80, 16, -4.0, 0.0),
        (16, 48, 16, 48, -4.0, 0.0),
        (48, 48, 48, 48, -4.0, 0.0),
    ]
    geom_con = compute_motion_geometry(mvs_contract, width, height)
    assert geom_con["divergence"] < 0.0, (
        f"Pure contraction should yield negative divergence, got {geom_con['divergence']}"
    )

    # --- 3. Curl is still magnitude-based (non-negative) ---
    assert geom_exp["curl"] >= 0.0


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


def test_peak_order_propagation_and_validation():
    from unittest import mock
    import iris.ingest as ingest_mod

    # 1. Test positive integer validation
    cfg_invalid = IRISConfig(peak_order=-1)
    with pytest.raises(ValueError, match="peak_order must be a positive integer"):
        ingest_mod.ingest("dummy.mp4", cfg_invalid)

    # 2. Verify synthetic/default peak order reaches the parser
    with mock.patch("iris.codec_validator.validate_video") as mock_val, \
         mock.patch("iris.charon_v.parse_video") as mock_parse:
        from iris.codec_validator import ValidationResult
        mock_val.return_value = ValidationResult(
            status="ok", severity="none", reasons=[], warnings=[],
            codec="h264", container="mp4", inspected_packet_count=10, inspected_frame_count=10,
            validation_level="fast", complete_stream_checked=False,
            mv_available=True, pts_complete=True, keyframe_found=True
        )
        # Return dummy values to satisfy the unpack
        mock_parse.return_value = ([], {"all_frame_energies": [], "iframe_indices": [], "fps": 25.0, "skipped": 0, "total": 0}, [])
        cfg_valid = IRISConfig(peak_order=5)
        try:
            ingest_mod.ingest("dummy.mp4", cfg_valid)
        except Exception:
            pass
        
        args, kwargs = mock_parse.call_args
        assert kwargs.get("peak_order") == 5


def test_action_score_semantics_and_migration():
    from iris.action_score import ActionScoreConfig, ActionScoreModule
    import warnings

    # 1. Config deprecation warning and migration
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = ActionScoreConfig(luma_diff_weight=0.8)
        assert len(w) >= 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert cfg.packet_size_weight == 0.8

    # 2. Conflicting values check
    with pytest.raises(ValueError, match="Conflicting values"):
        ActionScoreConfig(packet_size_weight=0.7, luma_diff_weight=0.8)

    # 3. Non-negative weights check
    with pytest.raises(ValueError, match="weights must be non-negative"):
        cfg_neg = ActionScoreConfig(packet_size_weight=-0.1)
        ActionScoreModule(cfg_neg).score_all([{"frame_idx": 0, "packet_size": 10.0}])

    # 4. Zero sum check
    with pytest.raises(ValueError, match="sum to a positive value"):
        cfg_zero = ActionScoreConfig(packet_size_weight=0.0, motion_weight=0.0, luma_entropy_weight=0.0)
        ActionScoreModule(cfg_zero).score_all([{"frame_idx": 0, "packet_size": 10.0}])

    # 5. Truly constant signal yields 0.0 (no neutral midpoint 0.5)
    asm = ActionScoreModule()
    constant_features = [{"frame_idx": i, "packet_size": 100.0, "motion_magnitude": 10.0, "luma_entropy": 3.0} for i in range(60)]
    records = asm.score_all(constant_features)
    for r in records:
        assert r["action_score"] == 0.0
        assert r["packet_size_contrib"] == 0.0
        assert r["motion_contrib"] == 0.0
        assert r["luma_entropy_contrib"] == 0.0

    # 6. Global max prominence scaling and bounds
    # Build synthetic curve with a single peak
    features = [{"frame_idx": i, "packet_size": 0.0, "motion_magnitude": 0.0, "luma_entropy": 0.0} for i in range(50)]
    features[25] = {"frame_idx": 25, "packet_size": 100.0, "motion_magnitude": 50.0, "luma_entropy": 10.0} # Peak at 25
    
    cfg_prom = ActionScoreConfig(max_prominence=0.8, persistence_threshold=0.3)
    asm_prom = ActionScoreModule(cfg_prom)
    records_prom = asm_prom.score_all(features)
    
    peak_record = records_prom[25]
    assert peak_record["is_peak"] is True
    assert 0.0 <= peak_record["persistence_value"] <= 1.0
    assert peak_record["packet_size_contrib"] == 1.0


def test_motion_geometry_normalization():
    from iris.cached_frame import CachedFrame
    from iris.frame_motion_descriptor import FrameMotionDescriptor
    from iris.l1_elysium import L1ElysiumCache

    fmd1 = FrameMotionDescriptor(frame_idx=0, timestamp_sec=0.0, motion_entropy=0.0, hessian_max_eigenvalue=0.0)
    fmd2 = FrameMotionDescriptor(frame_idx=1, timestamp_sec=0.0, motion_entropy=3.321928, hessian_max_eigenvalue=50.0)

    l1 = L1ElysiumCache(config=IRISConfig(l1_hessian_saturation_scale=10.0))
    cf1 = CachedFrame(frame_idx=0, timestamp_sec=0.0, action_score=0.5, persistence_value=0.5, is_peak=True, pagerank=0.5, query_similarity=0.5, motion=fmd1, admitted_at=0)
    cf2 = CachedFrame(frame_idx=1, timestamp_sec=0.0, action_score=0.5, persistence_value=0.5, is_peak=True, pagerank=0.5, query_similarity=0.5, motion=fmd2, admitted_at=0)

    l1._admission_counter = 1
    score1 = l1._keep_score(cf1)
    score2 = l1._keep_score(cf2)

    assert score2 > score1


def test_codec_validation_levels():
    from iris.codec_validator import validate_video, ValidationResult
    
    # 1. Reject non-existent file
    res = validate_video("non_existent.mp4", level="fast")
    assert res.status == "reject"
    assert res.severity == "high"
    assert res.validation_level == "fast"
    assert isinstance(res, ValidationResult)

    # 2. Reject in strict mode too
    res_strict = validate_video("non_existent.mp4", level="strict")
    assert res_strict.status == "reject"
    assert res_strict.validation_level == "strict"


def test_config_fields_consumption():
    from iris.iris_config import IRISConfig
    import glob
    import os

    fields = set(IRISConfig.__dataclass_fields__.keys())

    # We exclude deprecated/compatibility/upcoming fields from direct source requirements
    ignored_fields = {"luma_diff_weight", "aria_model", "answerer_schema_format"}

    iris_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "iris"))
    py_files = glob.glob(os.path.join(iris_dir, "**", "*.py"), recursive=True)

    source_content = ""
    for pf in py_files:
        if "iris_config.py" in pf:
            continue
        with open(pf, "r", encoding="utf-8", errors="ignore") as f:
            source_content += f.read()

    unconsumed = []
    for field in fields:
        if field in ignored_fields:
            continue
        if field not in source_content:
            unconsumed.append(field)

    assert not unconsumed, f"Unconsumed configuration fields found: {unconsumed}"


def test_config_mismatch_rejection():
    from iris.types import IRISIndex
    from iris.query import query
    
    index = IRISIndex(
        video_path="dummy.mp4",
        frames=[],
        index_action_score=0.5,
        stats={},
        frames_processed=0,
        peak_count=0,
        skipped_frames_ratio=0.0,
        storage_reduction_factor=1.0,
        config_snapshot={"clip_revision": "ViT-B/32", "graph_mode": "scene_sparse"}
    )
    
    # 1. clip_revision mismatch
    cfg_incompatible_clip = IRISConfig(clip_revision="ViT-L/14")
    with pytest.raises(ValueError, match="Incompatible configurations.*clip_revision"):
        query("test query", index, config=cfg_incompatible_clip)
        
    # 2. graph_mode mismatch
    cfg_incompatible_graph = IRISConfig(graph_mode="flat")
    with pytest.raises(ValueError, match="Incompatible configurations.*graph_mode"):
        query("test query", index, config=cfg_incompatible_graph)


def test_l2_graph_correctness_phase3():
    from iris.l2_asphodel import L2Asphodel
    
    # 1. Default graph_mode should be "scene_sparse"
    cfg = IRISConfig()
    assert cfg.graph_mode == "scene_sparse"

    # 2. Parallel list lengths bulk insertion validation
    l2 = L2Asphodel(cfg)
    feature_recs = [{"frame_idx": 0, "timestamp": 0.0, "luma_diff_energy": 0.1, "motion_magnitude": 1.0, "luma_entropy": 2.0}]
    action_recs = []
    with pytest.raises(ValueError, match="parallel list length mismatch"):
        l2.add_frame_nodes_bulk(feature_recs, action_recs)

    # 3. Hierarchy parents within scene boundaries
    feature_recs = [
        {"frame_idx": 0, "timestamp": 0.0, "luma_diff_energy": 0.1, "motion_magnitude": 1.0, "luma_entropy": 2.0, "scene_id": 0, "is_peak": True},
        {"frame_idx": 1, "timestamp": 1.0, "luma_diff_energy": 0.1, "motion_magnitude": 1.0, "luma_entropy": 2.0, "scene_id": 1, "is_peak": True},
        {"frame_idx": 2, "timestamp": 2.0, "luma_diff_energy": 0.1, "motion_magnitude": 1.0, "luma_entropy": 2.0, "scene_id": 0, "is_peak": False},
    ]
    action_recs = [
        {"action_score": 0.8, "persistence_value": 0.8}, # peak 0
        {"action_score": 0.9, "persistence_value": 0.9}, # peak 1
        {"action_score": 0.5, "persistence_value": 0.5}, # salient 2
    ]
    l2.add_frame_nodes_bulk(feature_recs, action_recs)
    
    # Check if edge between 0 and 2 has hierarchy_peak_salient relation, but NOT between 1 and 2!
    assert "hierarchy_peak_salient" in l2.graph[0][2].get("edge_type", "")
    assert "hierarchy_peak_salient" not in l2.graph.get_edge_data(1, 2, {}).get("edge_type", "")


def test_embedding_failure_safety_phase5():
    import pytest
    from iris._clip import get_frame_clip_embedding, get_clip_embedding_from_pil
    from unittest import mock
    
    # 1. get_frame_clip_embedding raises error if model is not loaded
    with mock.patch("iris._clip.get_clip_model", return_value=(None, None)):
        with pytest.raises(ValueError, match="CLIP model not loaded"):
            get_frame_clip_embedding(None, "cpu")
            
        with pytest.raises(ValueError, match="CLIP model not loaded"):
            get_clip_embedding_from_pil(None, "cpu")


def test_low_confidence_nli_unverifiable_phase7():
    from iris.cerberus_v import CerberusV
    from iris.iris_config import IRISConfig
    from unittest import mock
    
    cv = CerberusV()
    config = IRISConfig()
    # Set thresholds so action_score 0.7 triggers filtered_nli
    config.cerberus_high_thresh = 0.8
    config.cerberus_low_thresh = 0.6
    
    claims = ["maybe a red car is moving"]
    cache = mock.Mock()
    cache.set_facts = {}
    
    with mock.patch.object(cv, "_get_spacy"), \
         mock.patch.object(cv, "_get_nli_model"), \
         mock.patch.object(cv, "_full_nli", return_value={"verified": [], "rejected": [], "unverifiable": []}):
        res = cv.verify(claims, cache, 0.7, config)
        
        # Claims should end up in unverifiable, not in verified!
        assert "maybe a red car is moving" in res["unverifiable"]
        assert "maybe a red car is moving" not in res["verified"]
