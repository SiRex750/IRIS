"""Part 3c unit tests for the span-construction methods, run before
trusting Method B/C/D's numbers in the family comparison."""
from eval.metrics import (
    predicted_span_from_frames,
    predicted_span_from_frames_clustered,
    predicted_span_from_frames_scene,
    predicted_span_from_frames_peak,
)


def test_method_b_reduces_to_method_a_single_cluster_no_trim():
    """Single cluster (all gaps < gap_threshold_s), tail_trim_pct=0 -> must
    reduce to exactly Method A's [min, max] over the same frames."""
    frames = [
        {"timestamp": 0.0, "pagerank_score": 0.05},
        {"timestamp": 1.0, "pagerank_score": 0.30},
        {"timestamp": 2.0, "pagerank_score": 0.10},
        {"timestamp": 3.5, "pagerank_score": 0.40},
        {"timestamp": 4.0, "pagerank_score": 0.15},
    ]
    method_a = predicted_span_from_frames([f["timestamp"] for f in frames])
    method_b = predicted_span_from_frames_clustered(frames, gap_threshold_s=3.0, tail_trim_pct=0)
    assert method_b == method_a == (0.0, 4.0)


def test_method_b_reduces_to_method_a_when_trim_does_not_touch_extremes():
    """Default tail_trim_pct=20 with the extremes as the highest-scored
    frames: trimming removes only a low-score middle frame, min/max
    unchanged -> still equals Method A's span."""
    frames = [
        {"timestamp": 0.0, "pagerank_score": 0.9},   # kept: extreme, high score
        {"timestamp": 1.0, "pagerank_score": 0.8},
        {"timestamp": 2.0, "pagerank_score": 0.01},  # lowest score -> trimmed (1/5=20%)
        {"timestamp": 3.0, "pagerank_score": 0.7},
        {"timestamp": 4.0, "pagerank_score": 0.85},  # kept: extreme, high score
    ]
    method_a = predicted_span_from_frames([f["timestamp"] for f in frames])
    method_b = predicted_span_from_frames_clustered(frames, gap_threshold_s=3.0, tail_trim_pct=20)
    assert method_a == (0.0, 4.0)
    assert method_b == (0.0, 4.0) == method_a


def test_method_b_splits_on_gap_and_picks_higher_scoring_cluster():
    frames = [
        {"timestamp": 0.0, "pagerank_score": 0.1},
        {"timestamp": 1.0, "pagerank_score": 0.1},
        {"timestamp": 20.0, "pagerank_score": 0.9},  # gap > 3.0 -> new cluster, higher total score
        {"timestamp": 21.0, "pagerank_score": 0.9},
    ]
    span = predicted_span_from_frames_clustered(frames, gap_threshold_s=3.0, tail_trim_pct=0)
    assert span == (20.0, 21.0)


def test_method_b_falls_back_to_last_retrieval_score():
    frames = [
        {"timestamp": 0.0, "last_retrieval_score": 0.1},
        {"timestamp": 1.0, "last_retrieval_score": 0.9},
    ]
    span = predicted_span_from_frames_clustered(frames, gap_threshold_s=3.0, tail_trim_pct=0)
    assert span == (0.0, 1.0)


def test_method_c_looks_up_top_frame_scene():
    frames = [
        {"timestamp": 5.0, "scene_id": 2},
        {"timestamp": 50.0, "scene_id": 7},
    ]
    scene_spans = {2: (4.0, 6.5), 7: (49.0, 51.0)}
    span, fallback = predicted_span_from_frames_scene(frames, scene_spans)
    assert span == (4.0, 6.5)  # top-ranked frame (index 0) -> scene 2
    assert fallback is False


def test_method_c_falls_back_when_scene_unassigned():
    frames = [
        {"timestamp": 5.0, "retrieval_contributions": {"scene_id": -1}},
        {"timestamp": 6.0, "retrieval_contributions": {"scene_id": -1}},
    ]
    span, fallback = predicted_span_from_frames_scene(frames, {})
    assert span == (5.0, 6.0)  # Method A fallback over all retrieved frames
    assert fallback is True


def test_method_c_falls_back_when_scene_not_in_map():
    frames = [{"timestamp": 5.0, "retrieval_contributions": {"scene_id": 99}}]
    span, fallback = predicted_span_from_frames_scene(frames, {0: (0.0, 1.0)})
    assert span == (5.0, 5.0)
    assert fallback is True


def test_method_d_centers_on_clip_best_frame():
    frames = [
        {"frame_idx": 0, "timestamp": 0.0, "pagerank_score": 0.9, "clip_embedding": [1, 0, 0]},
        {"frame_idx": 1, "timestamp": 10.0, "pagerank_score": 0.05, "clip_embedding": [0, 1, 0]},
        {"frame_idx": 2, "timestamp": 20.0, "pagerank_score": 0.05, "clip_embedding": [0, 0, 1]},
    ]
    query_embedding = [0, 1, 0]  # closest to frame_idx=1, which is neither rank-1 nor top pagerank
    span, used_clip_anchor = predicted_span_from_frames_peak(frames, query_embedding, half_width_s=2.2)
    assert span == (10.0 - 2.2, 10.0 + 2.2)
    assert used_clip_anchor is True


def test_method_d_falls_back_to_rank1_when_no_query_embedding():
    frames = [
        {"frame_idx": 0, "timestamp": 5.0, "clip_embedding": [1, 0, 0]},
        {"frame_idx": 1, "timestamp": 15.0, "clip_embedding": [0, 1, 0]},
    ]
    span, used_clip_anchor = predicted_span_from_frames_peak(frames, None, half_width_s=2.2)
    assert used_clip_anchor is False
    assert span == (5.0 - 2.2, 5.0 + 2.2)


def test_method_d_clamps_to_duration():
    frames = [{"frame_idx": 0, "timestamp": 9.5, "clip_embedding": [1, 0, 0]}]
    query_embedding = [1, 0, 0]
    span, used_clip_anchor = predicted_span_from_frames_peak(
        frames, query_embedding, half_width_s=2.2, duration_s=10.0,
    )
    assert used_clip_anchor is True
    assert span[1] == 10.0


def test_method_d_clamps_at_zero():
    frames = [{"frame_idx": 0, "timestamp": 1.0, "clip_embedding": [1, 0, 0]}]
    query_embedding = [1, 0, 0]
    span, used_clip_anchor = predicted_span_from_frames_peak(frames, query_embedding, half_width_s=2.2)
    assert used_clip_anchor is True
    assert span[0] == 0.0


def test_method_d_empty_frames():
    span, used_clip_anchor = predicted_span_from_frames_peak([], [1, 0, 0], half_width_s=2.2)
    assert span == (0.0, 0.0)
    assert used_clip_anchor is False
