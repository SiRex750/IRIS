from benchmarks.exp1a_baselines import (
    uniform_budget_matched,
    random_budget_matched,
    iframe_only_budget_matched,
    scene_change_detection,
    luma_diff_topk
)

def _make_dummy_records(n=20):
    records = []
    for i in range(n):
        # Every 5th frame is an I-frame, others P
        ftype = "I" if i % 5 == 0 else "P"
        records.append({
            "frame_idx": i,
            "timestamp": float(i) * 0.5,
            "frame_type": ftype,
            "luma_diff_energy": float(i % 3) * 0.1,
            "packet_size": float(i) * 100.0,
            "motion_magnitude": 0.5,
            "luma_entropy": 0.3
        })
    return records

def test_uniform_exact_k():
    # 1. uniform returns exactly K frames
    records = _make_dummy_records(20)
    for k in [1, 5, 10, 20]:
        res = uniform_budget_matched(records, k)
        assert len(res["selected_frames"]) == k
        assert len(res["selected_timestamps"]) == k
        assert len(res["ranked_timestamps"]) == k

def test_random_exact_k():
    # 2. random returns exactly K frames
    records = _make_dummy_records(20)
    for k in [1, 5, 10, 20]:
        res = random_budget_matched(records, k, seed=0)
        assert len(res["selected_frames"]) == k
        assert len(res["selected_timestamps"]) == k
        assert len(res["ranked_timestamps"]) == k

def test_random_reproducibility():
    # 3. random is reproducible for the same seed
    records = _make_dummy_records(20)
    res1 = random_budget_matched(records, 8, seed=42)
    res2 = random_budget_matched(records, 8, seed=42)
    res3 = random_budget_matched(records, 8, seed=43)
    
    assert res1["selected_frames"] == res2["selected_frames"]
    assert res1["selected_frames"] != res3["selected_frames"]

def test_fill_count_iframe():
    # 4. fill_count is recorded when a baseline has fewer than K frames
    records = _make_dummy_records(20) # Has 4 keyframes (indices 0, 5, 10, 15)
    
    # K = 6 is more than 4 keyframes -> should fill 2 frames
    res = iframe_only_budget_matched(records, 6)
    assert len(res["selected_frames"]) == 6
    assert res["fill_count"] == 2

def test_no_more_than_k():
    # 5. no baseline is allowed to return more than K frames
    records = _make_dummy_records(20)
    
    # Test all baselines for multiple budgets
    for k in [3, 8]:
        assert len(uniform_budget_matched(records, k)["selected_frames"]) == k
        assert len(random_budget_matched(records, k)["selected_frames"]) == k
        assert len(iframe_only_budget_matched(records, k)["selected_frames"]) == k
        assert len(scene_change_detection(records, k)["selected_frames"]) == k
        assert len(luma_diff_topk(records, k)["selected_frames"]) == k
