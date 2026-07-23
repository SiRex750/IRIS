"""Family 5 Step 0 regression: scripts/part3_tune.py's default_predicted_span
must route to Method D (predicted_span_from_frames_peak), not Method B
(predicted_span_from_frames_clustered) -- Part 3e superseded Method B in
favor of Method D (tuning/family2_k_span_method_report.md). Running Family 5
under the stale Method B default would contaminate the
peak_distance/peak_prominence selection with a span method already proven
inferior.

This test fails against the pre-fix code (default_predicted_span called
predicted_span_from_frames_clustered) and passes post-fix."""
from scripts.part3_tune import default_predicted_span, SPAN_METHOD_D_HALF_WIDTH_S


def test_default_predicted_span_centers_on_clip_peak_not_cluster_span():
    # Three well-separated frames; query embedding picks out frame_idx=1
    # (neither rank-1 nor highest pagerank_score), same setup as
    # test_span_methods.py's Method D unit test. Method B (clustered
    # min/max with tail-trim) would NOT collapse this to a narrow window
    # centered on a single frame -- it would return a span shaped by
    # cluster membership/score-trimming over all three frames instead.
    frames = [
        {"frame_idx": 0, "timestamp": 0.0, "pagerank_score": 0.9, "clip_embedding": [1, 0, 0]},
        {"frame_idx": 1, "timestamp": 10.0, "pagerank_score": 0.05, "clip_embedding": [0, 1, 0]},
        {"frame_idx": 2, "timestamp": 20.0, "pagerank_score": 0.05, "clip_embedding": [0, 0, 1]},
    ]
    query_embedding = [0, 1, 0]

    span = default_predicted_span(frames, query_embedding)

    assert span == (10.0 - SPAN_METHOD_D_HALF_WIDTH_S, 10.0 + SPAN_METHOD_D_HALF_WIDTH_S)


def test_default_predicted_span_falls_back_to_rank1_without_query_embedding():
    # Method D's documented degrade-to-rank-1 behavior when no query
    # embedding is available -- exercises the query_embedding=None path
    # that a stale Method-B-based implementation wouldn't even accept
    # the same way (Method B never used query_embedding to anchor a peak).
    frames = [
        {"frame_idx": 0, "timestamp": 5.0, "clip_embedding": [1, 0, 0]},
        {"frame_idx": 1, "timestamp": 15.0, "clip_embedding": [0, 1, 0]},
    ]
    span = default_predicted_span(frames)
    assert span == (5.0 - SPAN_METHOD_D_HALF_WIDTH_S, 5.0 + SPAN_METHOD_D_HALF_WIDTH_S)
