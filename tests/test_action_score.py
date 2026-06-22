from action_acore import ActionScoreConfig, ActionScoreModule


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
                "residual_energy": 0.4,
                "motion_magnitude": 0.4,
                "entropy": 0.05,
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
                "residual_energy": value,
                "motion_magnitude": value,
                "entropy": value,
            }
        )

    records = scorer.score_all(frames)
    peaks = [record for record in records if record["is_peak"]]

    assert len(peaks) == 1
    assert peaks[0]["frame_idx"] == 3
    assert peaks[0]["action_score"] > 0.6
    assert peaks[0]["persistence_value"] >= 0.4