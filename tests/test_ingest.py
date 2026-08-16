import math
from pathlib import Path

import numpy as np
import pytest

import iris.ingest as ingest_mod
from iris.types import IRISIndex, FrameRecord
from iris.iris_config import IRISConfig

FIXED_EMB = np.full(512, 1.0 / np.sqrt(512), dtype=np.float32)
FIXED_CAPTION = {"clip_label": "x", "semantic_caption": "a test frame"}

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE_VIDEO = (
    _REPO_ROOT / "eval" / "data" / "ucf" / "videos" / "Anomaly-Videos-Part-1"
    / "Abuse" / "Abuse029_x264.mp4"
)


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
    idx = ingest_mod._build_index_from_records(of, rr, st, "synthetic.mp4", IRISConfig(), 10)

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
    idx = ingest_mod._build_index_from_records(of, rr, st, "synthetic.mp4", IRISConfig(), 10)
    for fr in idx.frames:
        assert isinstance(fr, FrameRecord)
        assert fr.clip_embedding is not None
        assert fr.clip_embedding.shape == (512,)
        assert fr.caption is None
        assert isinstance(fr.pagerank_score, float)


@pytest.mark.skipif(not _SAMPLE_VIDEO.exists(), reason="sample video not present in this checkout")
def test_scene_segmentation_modes(patched):
    """scene_segmentation ablation knob (config-selected, default 'codec'):
    - fixed_count yields the SAME realized scene count as codec.
    - codec-mode scene_id assignment is exactly the historical (pre-knob)
      containment-lookup-against-scene_spans behavior -- verified directly
      here (not just "unchanged" by assertion, but recomputed from the same
      packet curve and diffed frame-by-frame).
    - fixed_seconds yields ceil(duration_seconds / fixed_scene_seconds) scenes.
    """
    import av
    import iris.charon_v as charon_v

    idx_codec = ingest_mod.ingest(str(_SAMPLE_VIDEO), config=IRISConfig(scene_segmentation="codec"))
    idx_fixed_count = ingest_mod.ingest(str(_SAMPLE_VIDEO), config=IRISConfig(scene_segmentation="fixed_count"))
    idx_fixed_seconds = ingest_mod.ingest(str(_SAMPLE_VIDEO), config=IRISConfig(scene_segmentation="fixed_seconds"))

    # -- codec-mode scene_id unchanged: recompute the historical containment
    #    lookup independently from the same packet curve and diff exactly.
    all_frame_energies, iframe_indices, _, _ = charon_v._demux_packet_curve(str(_SAMPLE_VIDEO))
    fps = charon_v.get_stream_fps(str(_SAMPLE_VIDEO))
    scene_spans = charon_v.compute_valley_scene_boundaries(all_frame_energies, iframe_indices, fps)

    def _expected_codec_scene_id(fi: int) -> int:
        for scene_idx, (start, end) in enumerate(scene_spans):
            if start <= fi < end:
                return scene_idx
        return -1

    for fr in idx_codec.frames:
        assert fr.scene_id == _expected_codec_scene_id(fr.frame_idx), (
            f"codec-mode scene_id changed for frame_idx={fr.frame_idx}: "
            f"got {fr.scene_id}, expected {_expected_codec_scene_id(fr.frame_idx)}"
        )

    # -- fixed_count: same realized scene COUNT as codec.
    codec_scene_count = len(set(fr.scene_id for fr in idx_codec.frames))
    fixed_count_scene_count = len(set(fr.scene_id for fr in idx_fixed_count.frames))
    assert fixed_count_scene_count == codec_scene_count, (
        f"fixed_count scene count ({fixed_count_scene_count}) != "
        f"codec scene count ({codec_scene_count})"
    )

    # -- fixed_seconds: ceil(duration / T) scenes, T = fixed_scene_seconds (default 60.0).
    container = av.open(str(_SAMPLE_VIDEO))
    stream = container.streams.video[0]
    duration_seconds = float(stream.duration * stream.time_base)
    container.close()
    expected_scenes = math.ceil(duration_seconds / IRISConfig().fixed_scene_seconds)
    fixed_seconds_scene_count = len(set(fr.scene_id for fr in idx_fixed_seconds.frames))
    assert fixed_seconds_scene_count == expected_scenes, (
        f"fixed_seconds scene count ({fixed_seconds_scene_count}) != "
        f"ceil(duration/T) ({expected_scenes})"
    )
