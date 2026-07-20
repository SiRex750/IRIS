from iris.action_score import ActionScoreConfig, ActionScoreModule


def test_uniform_pan_not_peak():
    config = ActionScoreConfig(
        peak_distance=2,
        peak_prominence=0.05,
        persistence_threshold=0.4,
    )
    scorer = ActionScoreModule(config)

    frames = []
    for i in range(20):
        frames.append(
            {
                "frame_idx": i,
                "packet_size": 0.4,
                "motion_magnitude": 0.4,
                "luma_entropy": 0.05,
            }
        )

    records = scorer.score_all(frames)

    assert all(not record["is_peak"] for record in records)


def test_action_spike_becomes_peak():
    config = ActionScoreConfig(
        peak_distance=2,
        peak_prominence=0.05,
        persistence_threshold=0.4,
    )
    scorer = ActionScoreModule(config)

    values = [0.1, 0.2, 0.3, 0.9, 0.3, 0.2, 0.1]

    frames = []
    for i, value in enumerate(values):
        frames.append(
            {
                "frame_idx": i,
                "packet_size": value,
                "motion_magnitude": value,
                "luma_entropy": value,
            }
        )

    records = scorer.score_all(frames)
    peaks = [record for record in records if record["is_peak"]]

    assert len(peaks) == 1
    assert peaks[0]["frame_idx"] == 3
    assert peaks[0]["action_score"] > 0.6
    assert peaks[0]["persistence_value"] >= 0.4


def test_dominant_peak_normalizes_to_one():
    # Per-video normalization: the strongest peak always gets persistence_value == 1.0,
    # regardless of its raw prominence value.
    config = ActionScoreConfig(
        peak_distance=2,
        peak_prominence=0.05,
        persistence_threshold=0.4,
    )
    scorer = ActionScoreModule(config)

    values = [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    frames = [
        {"frame_idx": i, "packet_size": v, "motion_magnitude": v, "luma_entropy": v}
        for i, v in enumerate(values)
    ]

    records = scorer.score_all(frames)
    peaks = [r for r in records if r["is_peak"]]

    assert len(peaks) == 1
    assert peaks[0]["frame_idx"] == 3
    # Sole peak is the maximum: prominence / max_prominence = 1.0.
    assert peaks[0]["persistence_value"] == 1.0

    # Two-peak case: dominant at frame 6 (prominence 1.0), weak at frame 3 (prominence 0.1875).
    # data-derived max_prominence = 1.0.
    # Frame 3 persistence = 0.1875 < threshold 0.4 → not a peak.
    # Frame 6 persistence = 1.0 >= threshold → is a peak.
    values_weak = [0.1, 0.1, 0.1, 0.25, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
    frames_weak = [
        {"frame_idx": i, "packet_size": v, "motion_magnitude": v, "luma_entropy": v}
        for i, v in enumerate(values_weak)
    ]
    records_weak = scorer.score_all(frames_weak)
    peaks_weak = [r for r in records_weak if r["is_peak"]]

    assert len(peaks_weak) == 1
    assert peaks_weak[0]["frame_idx"] == 6
    assert peaks_weak[0]["persistence_value"] == 1.0


def test_persistence_clustering_regression():
    # Two distinct peaks with different sharpness; verify per-video normalization.
    config = ActionScoreConfig(
        peak_distance=5,
        peak_prominence=0.05,
        persistence_threshold=0.15,
    )
    scorer = ActionScoreModule(config)

    # Base: 0.1. Sharp peak at 6: 0.4. Gentler peak at 16: 0.22.
    values = [
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.22, 0.1, 0.1, 0.1, 0.1
    ]

    frames = []
    for i, val in enumerate(values):
        frames.append({
            "frame_idx": i,
            "packet_size": val,
            "motion_magnitude": val,
            "luma_entropy": val,
        })

    records = scorer.score_all(frames)

    # Non-peak frames must have persistence_value == 0.0.
    for idx, r in enumerate(records):
        if idx not in (6, 16):
            assert r["persistence_value"] == 0.0, f"Expected 0.0 persistence at non-peak frame {idx}"

    p1 = records[6]["persistence_value"]
    p2 = records[16]["persistence_value"]

    # Peak 6: normalized prominence 1.0, data-derived max_prominence 1.0 → persistence = 1.0.
    # Peak 16: normalized prominence 0.4, max_prominence 1.0 → persistence = 0.4.
    assert p1 == 1.0, f"Dominant peak should normalize to 1.0, got {p1}"
    assert 0.0 < p2 < 1.0, f"Expected fractional persistence for weaker peak, got {p2}"
    assert p1 > p2, f"Expected dominant peak persistence {p1} > weaker peak persistence {p2}"


def test_configured_max_prominence_divisor():
    """
    P1-06 regression: persistence_value must be normalized by the *configured*
    max_prominence, not the local video-specific maximum.

    Using a data-derived divisor made persistence values incomparable across
    videos processed in the same pipeline run.  A configured global bound
    ensures consistency.

    Setup:
        config.max_prominence = 0.5  (default)
        Prominences:  main peak = 0.8,  secondary = 0.4
        Expected:
            main peak:      min(0.8 / 0.5, 1.0) = 1.0   (clamped)
            secondary peak: min(0.4 / 0.5, 1.0) = 0.8
    """
    config = ActionScoreConfig(
        peak_distance=5,
        peak_prominence=0.05,
        persistence_threshold=0.3,
        max_prominence=0.5,   # explicit configured bound
    )
    scorer = ActionScoreModule(config)

    # Dip at frame 0 (0.0), baseline 0.2, main peak 1.0 at frame 5, secondary 0.6 at frame 11.
    # After min-max normalization the signal is unchanged (min=0.0, max=1.0).
    # Prominences: main peak = 1.0 - 0.2 = 0.8, secondary = 0.6 - 0.2 = 0.4.
    values = [0.0, 0.2, 0.2, 0.2, 0.2, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.6, 0.2, 0.2, 0.2]
    frames = [
        {"frame_idx": i, "packet_size": v, "motion_magnitude": v, "luma_entropy": v}
        for i, v in enumerate(values)
    ]

    records = scorer.score_all(frames)

    peak5  = records[5]
    peak11 = records[11]

    assert peak5["is_peak"]
    assert peak11["is_peak"]

    # Main peak: prominence 0.8 / configured_max 0.5 = 1.6 → clamped to 1.0.
    assert peak5["persistence_value"] == 1.0, (
        f"Main peak persistence should be clamped to 1.0, got {peak5['persistence_value']}"
    )
    # Weaker peak: 0.4 / 0.5 = 0.8 (not 0.5 which the old data-derived max 0.8 would give).
    assert abs(peak11["persistence_value"] - 0.8) < 0.05, (
        f"Secondary peak persistence should be ~0.8, got {peak11['persistence_value']}"
    )

