from benchmarks.exp1a_metrics import (
    is_temporal_hit,
    compute_temporal_distance,
    compute_temporal_iou_proxy,
    evaluate_selection_coverage,
    evaluate_ranked_retrieval
)
import numpy as np

def test_temporal_hit():
    # 1. temporal hit inside interval
    assert is_temporal_hit(5.0, 4.0, 6.0) is True
    assert is_temporal_hit(4.0, 4.0, 6.0) is True
    assert is_temporal_hit(6.0, 4.0, 6.0) is True

def test_temporal_miss():
    # 2. temporal miss outside interval
    assert is_temporal_hit(3.9, 4.0, 6.0) is False
    assert is_temporal_hit(6.1, 4.0, 6.0) is False

def test_temporal_distance_inside():
    # 3. temporal distance = 0 inside interval
    assert compute_temporal_distance(5.0, 4.0, 6.0) == 0.0

def test_temporal_distance_outside():
    # 4. temporal distance positive outside interval
    assert compute_temporal_distance(3.0, 4.0, 6.0) == 1.0
    assert abs(compute_temporal_distance(7.2, 4.0, 6.0) - 1.2) < 1e-6

def test_temporal_iou_proxy():
    # 5. temporal_iou_proxy works
    # [4.5, 5.5] vs [4.0, 6.0]
    # intersection: 1.0, union: 2.0 -> IoU: 0.5
    iou = compute_temporal_iou_proxy(5.0, 4.0, 6.0)
    assert iou == 0.5

def test_recall_at_k():
    # 6. recall@k works
    ranked = [2.0, 5.0, 8.0]
    # At K=1, top-1 is 2.0 -> hit=0, recall=0
    res_k1 = evaluate_ranked_retrieval(ranked, 4.0, 6.0, k=1)
    assert res_k1["recall"] == 0.0
    assert res_k1["temporal_hit"] == 0.0
    
    # At K=2, top-2 are [2.0, 5.0] -> hit=1, recall=1
    res_k2 = evaluate_ranked_retrieval(ranked, 4.0, 6.0, k=2)
    assert res_k2["recall"] == 1.0
    assert res_k2["temporal_hit"] == 1.0

def test_mrr():
    # 7. mrr works
    ranked = [2.0, 8.0, 5.0, 9.0]
    # Hit is at index 2 (5.0), which is index 2 in 0-based, rank 3 (1-based)
    # MRR should be 1/3
    res = evaluate_ranked_retrieval(ranked, 4.0, 6.0, k=4)
    assert abs(res["mrr"] - 0.3333333333333333) < 1e-6
    
    # No hit -> MRR = 0
    res_nohit = evaluate_ranked_retrieval(ranked, 12.0, 15.0, k=4)
    assert res_nohit["mrr"] == 0.0

def test_missing_ground_truth():
    # 8. missing ground truth returns None / N/A, not 0
    ranked = [5.0]
    res_none = evaluate_ranked_retrieval(ranked, None, 6.0, k=1)
    assert res_none["recall"] is None
    assert res_none["temporal_hit"] is None
    assert res_none["mrr"] is None
    
    res_nan = evaluate_ranked_retrieval(ranked, float('nan'), 6.0, k=1)
    assert res_nan["recall"] is None
    
    res_neg = evaluate_ranked_retrieval(ranked, -1.0, 6.0, k=1)
    assert res_neg["recall"] is None

    cov_none = evaluate_selection_coverage(ranked, None, 6.0)
    assert cov_none["event_hit_any_selected"] is None
    assert cov_none["zero_overlap"] is None
