from benchmarks.exp0_metrics import (
    compute_compression_rate,
    is_temporal_hit,
    compute_temporal_distance,
    compute_temporal_iou_proxy,
    evaluate_event_retrieval
)

def test_compression_rate():
    # 1. compression rate calculation
    ratio, comp = compute_compression_rate(100, 20)
    assert ratio == 0.2
    assert comp == 0.8
    
    ratio2, comp2 = compute_compression_rate(0, 0)
    assert ratio2 == 0.0
    assert comp2 == 1.0

def test_temporal_hit_inside_interval():
    # 2. temporal hit inside interval
    assert is_temporal_hit(5.0, 4.0, 6.0) is True
    assert is_temporal_hit(4.0, 4.0, 6.0) is True
    assert is_temporal_hit(6.0, 4.0, 6.0) is True

def test_temporal_miss_outside_interval():
    # 3. temporal miss outside interval
    assert is_temporal_hit(3.9, 4.0, 6.0) is False
    assert is_temporal_hit(6.1, 4.0, 6.0) is False

def test_temporal_distance_inside():
    # 4. temporal distance is 0 inside interval
    assert compute_temporal_distance(5.0, 4.0, 6.0) == 0.0
    assert compute_temporal_distance(4.0, 4.0, 6.0) == 0.0
    assert compute_temporal_distance(6.0, 4.0, 6.0) == 0.0

def test_temporal_distance_outside():
    # 5. temporal distance positive outside interval
    assert compute_temporal_distance(3.0, 4.0, 6.0) == 1.0
    assert compute_temporal_distance(7.5, 4.0, 6.0) == 1.5

def test_temporal_iou_proxy():
    # 6. temporal_iou_proxy for timestamp interval
    # timestamp interval: [4.5, 5.5], event: [4.0, 6.0]
    # intersection: 1.0 (from 4.5 to 5.5)
    # union: 2.0 (event length is 2, intersection is 1.0, timestamp interval is 1.0)
    # IoU = 1.0 / 2.0 = 0.5
    iou = compute_temporal_iou_proxy(5.0, 4.0, 6.0)
    assert iou == 0.5

def test_recall_at_k():
    # 7. recall@k behavior
    selected = [2.0, 5.0, 8.0]
    # For k=1, top-1 is 2.0, which is outside [4.0, 6.0] -> hit=0, recall=0
    res_k1 = evaluate_event_retrieval(selected, 4.0, 6.0, k=1)
    assert res_k1["temporal_hit"] == 0.0
    assert res_k1["recall"] == 0.0
    
    # For k=2, top-2 are [2.0, 5.0], 5.0 is inside [4.0, 6.0] -> hit=1, recall=1
    res_k2 = evaluate_event_retrieval(selected, 4.0, 6.0, k=2)
    assert res_k2["temporal_hit"] == 1.0
    assert res_k2["recall"] == 1.0
