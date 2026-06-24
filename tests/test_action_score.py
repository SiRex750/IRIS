from action_score import ActionScoreConfig, ActionScoreModule


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


def test_single_dominant_peak_option_b():
    # Test that with Option B, we use max_prominence as the divisor
    # and a peak's persistence is relative to config.max_prominence.
    config = ActionScoreConfig(
        peak_distance=2,
        peak_prominence=0.05,
        persistence_threshold=0.4,
        max_prominence=0.5,
    )
    scorer = ActionScoreModule(config)
    
    values = [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    frames = []
    for i, val in enumerate(values):
        frames.append({
            "frame_idx": i,
            "residual_energy": val,
            "motion_magnitude": val,
            "entropy": val,
        })
        
    records = scorer.score_all(frames)
    peaks = [record for record in records if record["is_peak"]]
    
    assert len(peaks) == 1
    assert peaks[0]["frame_idx"] == 3
    # Prominence of index 3 (value 1.0 after normalization vs 0.0 neighbors) is 1.0.
    # persistence_value = min(1.0 / 0.5, 1.0) = 1.0.
    assert peaks[0]["persistence_value"] == 1.0
    
    # If the peak is weaker, e.g. prominence is 0.1875, persistence is 0.1875 / 0.5 = 0.375.
    # 0.375 < 0.4 persistence_threshold, so it should not be considered a peak.
    values_weak = [0.1, 0.1, 0.1, 0.25, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
    frames_weak = []
    for i, val in enumerate(values_weak):
        frames_weak.append({
            "frame_idx": i,
            "residual_energy": val,
            "motion_magnitude": val,
            "entropy": val,
        })
    records_weak = scorer.score_all(frames_weak)
    peaks_weak = [record for record in records_weak if record["is_peak"]]
    assert len(peaks_weak) == 1
    assert peaks_weak[0]["frame_idx"] == 6


def test_persistence_clustering_regression():
    # Build synthetic residual energy series with 2 distinct peaks of different sharpness
    config = ActionScoreConfig(
        peak_distance=5,
        peak_prominence=0.05,
        persistence_threshold=0.15,  # Lower to admit both as peaks
        max_prominence=2.0,          # Set higher to ensure fractional values without clipping
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
            "residual_energy": val,
            "motion_magnitude": val,
            "entropy": val,
        })
        
    records = scorer.score_all(frames)
    
    # Assert (a): non-peak candidate indices (not in find_peaks) have persistence_value == 0.0
    for idx, r in enumerate(records):
        if idx not in (6, 16):
            assert r["persistence_value"] == 0.0, f"Expected 0.0 persistence at non-peak frame {idx}"
            
    # Assert (b): peak frames have distinct, non-degenerate fractional persistence values
    p1 = records[6]["persistence_value"]
    p2 = records[16]["persistence_value"]
    
    # Peak 1 (normalized value 1.0, prominence 1.0): 1.0 / 2.0 = 0.5
    # Peak 2 (normalized value 0.4, prominence 0.4): 0.4 / 2.0 = 0.2
    assert 0.0 < p1 < 1.0, f"Expected fractional persistence for peak 1, got {p1}"
    assert 0.0 < p2 < 1.0, f"Expected fractional persistence for peak 2, got {p2}"
    assert p1 != p2, "Expected distinct persistence values"
    
    # Assert (c): sharper peak gets higher persistence_value than gentler one
    assert p1 > p2, f"Expected sharper peak persistence {p1} to be > gentler peak persistence {p2}"