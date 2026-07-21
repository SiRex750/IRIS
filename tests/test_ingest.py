import numpy as np
import pytest

import iris.ingest as ingest_mod
from iris.types import IRISIndex, FrameRecord
from iris.iris_config import IRISConfig

FIXED_EMB = np.full(512, 1.0 / np.sqrt(512), dtype=np.float32)
FIXED_CAPTION = {"clip_label": "x", "semantic_caption": "a test frame"}


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(ingest_mod, "get_clip_embedding_from_pil",
                        lambda pil, device: FIXED_EMB.copy())
    monkeypatch.setattr(ingest_mod, "get_semantic_and_clip_caption",
                        lambda pil, frame, emb, device: dict(FIXED_CAPTION))


def _synthetic():
    # 3 non-SKIP output frames (each with a pil_image sentinel -> fast path)
    # plus matching raw records. output_frames intentionally carry NO
    # motion_magnitude key, mirroring real charon_v output.
    output_frames, raw_records = [], []
    for i in range(3):
        output_frames.append({
            "frame_idx": i, "timestamp": float(i), "tier": "CANDIDATE",
            "luma_diff_energy": 0.1 * i, "motion_vectors": [],
            "pil_image": object(),
            "divergence": 0.0, "curl": 0.0, "jacobian_frobenius": 0.0,
            "hessian_max_eigenvalue": 0.0, "motion_entropy": 0.0,
        })
        raw_records.append({
            "frame_idx": i, "frame_type": "P",
            "luma_diff_energy": 0.1 * i, "motion_magnitude": 0.1 * i,
            "luma_entropy": 0.2 * i,
        })
    stats = {"total": 10, "i_frames": 1, "peaks": 1, "salient": 1,
             "candidate": 3, "skipped": 6,
             "salient_thresh_used": 0.35, "candidate_thresh_used": 0.08}
    return output_frames, raw_records, stats


def test_build_index_hermetic(patched):
    of, rr, st = _synthetic()
    idx = ingest_mod._build_index_from_records(of, rr, st, "synthetic.mp4", IRISConfig(graph_mode="flat"), 10)

    assert isinstance(idx, IRISIndex)
    assert idx._graph is not None
    assert len(idx.frames) == 3
    assert idx._graph.graph.number_of_nodes() == 3
    assert idx.video_path == "synthetic.mp4"
    assert idx.frames_processed == 3
    assert idx.index_action_score >= 0.0
    assert isinstance(idx.config_snapshot, dict) and idx.config_snapshot
    # stats-derived scalars
    assert abs(idx.skipped_frames_ratio - 0.6) < 1e-9
    assert abs(idx.storage_reduction_factor - (10 / 3)) < 1e-9


def test_framerecords_enriched(patched):
    of, rr, st = _synthetic()
    idx = ingest_mod._build_index_from_records(of, rr, st, "synthetic.mp4", IRISConfig(graph_mode="flat"), 10)
    for fr in idx.frames:
        assert isinstance(fr, FrameRecord)
        assert fr.clip_embedding is not None
        assert fr.clip_embedding.shape == (512,)
        assert fr.caption is None
        assert isinstance(fr.pagerank_score, float)
