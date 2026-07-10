import numpy as np

def is_nan_or_none(val):
    if val is None:
        return True
    try:
        return np.isnan(float(val))
    except (ValueError, TypeError):
        return True

def is_temporal_hit(timestamp: float, start_time: float, end_time: float) -> bool:
    """
    Returns True if the timestamp falls within [start_time, end_time] (inclusive).
    """
    return start_time <= timestamp <= end_time

def compute_temporal_distance(timestamp: float, start_time: float, end_time: float) -> float:
    """
    Computes temporal distance from timestamp to event interval.
    If inside, distance is 0. Otherwise, min distance to start or end.
    """
    if start_time <= timestamp <= end_time:
        return 0.0
    return min(abs(timestamp - start_time), abs(timestamp - end_time))

def compute_interval_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """
    Computes Intersection over Union of two temporal intervals.
    """
    intersection = max(0.0, min(end1, end2) - max(start1, start2))
    union = (end1 - start1) + (end2 - start2) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union

def compute_temporal_iou_proxy(timestamp: float, start_time: float, end_time: float) -> float:
    """
    Converts timestamp to interval [timestamp - 0.5, timestamp + 0.5] and computes IoU
    with ground-truth event interval [start_time, end_time].
    """
    return compute_interval_iou(timestamp - 0.5, timestamp + 0.5, start_time, end_time)

def evaluate_selection_coverage(selected_timestamps: list[float], start_time: float, end_time: float) -> dict:
    """
    Computes Selection Coverage metrics for an event.
    If ground truth is missing, returns None for all metric values.
    """
    if is_nan_or_none(start_time) or is_nan_or_none(end_time) or start_time < 0 or end_time < 0:
        return {
            "event_hit_any_selected": None,
            "min_temporal_distance_seconds": None,
            "best_temporal_iou_proxy": None,
            "zero_overlap": None
        }
        
    if not selected_timestamps:
        return {
            "event_hit_any_selected": 0.0,
            "min_temporal_distance_seconds": 999.0,
            "best_temporal_iou_proxy": 0.0,
            "zero_overlap": 1.0
        }
        
    hits = [is_temporal_hit(t, start_time, end_time) for t in selected_timestamps]
    hit_any = 1.0 if any(hits) else 0.0
    zero_ol = 1.0 - hit_any
    
    distances = [compute_temporal_distance(t, start_time, end_time) for t in selected_timestamps]
    min_dist = min(distances) if distances else 999.0
    
    ious = [compute_temporal_iou_proxy(t, start_time, end_time) for t in selected_timestamps]
    best_iou = max(ious) if ious else 0.0
    
    return {
        "event_hit_any_selected": hit_any,
        "min_temporal_distance_seconds": min_dist,
        "best_temporal_iou_proxy": best_iou,
        "zero_overlap": zero_ol
    }

def evaluate_ranked_retrieval(ranked_timestamps: list[float], start_time: float, end_time: float, k: int) -> dict:
    """
    Computes Ranked Retrieval metrics for an event at top-k.
    If ground truth is missing, returns None for all metric values.
    """
    if is_nan_or_none(start_time) or is_nan_or_none(end_time) or start_time < 0 or end_time < 0:
        return {
            "recall": None,
            "temporal_hit": None,
            "mean_temporal_distance_seconds": None,
            "temporal_iou_proxy": None,
            "mrr": None
        }
        
    top_k = ranked_timestamps[:k]
    
    if not top_k:
        return {
            "recall": 0.0,
            "temporal_hit": 0.0,
            "mean_temporal_distance_seconds": 999.0,
            "temporal_iou_proxy": 0.0,
            "mrr": 0.0
        }
        
    # Recall & Hits
    hits = [is_temporal_hit(t, start_time, end_time) for t in top_k]
    hit = 1.0 if any(hits) else 0.0
    recall = hit
    
    # Distance
    distances = [compute_temporal_distance(t, start_time, end_time) for t in top_k]
    mean_dist = sum(distances) / len(distances) if distances else 999.0
    
    # IoU Proxy
    ious = [compute_temporal_iou_proxy(t, start_time, end_time) for t in top_k]
    best_iou = max(ious) if ious else 0.0
    
    # MRR (over all ranked timestamps, not just top-K)
    mrr_val = 0.0
    for idx, t in enumerate(ranked_timestamps):
        if is_temporal_hit(t, start_time, end_time):
            mrr_val = 1.0 / (idx + 1)
            break
            
    return {
        "recall": recall,
        "temporal_hit": hit,
        "mean_temporal_distance_seconds": mean_dist,
        "temporal_iou_proxy": best_iou,
        "mrr": mrr_val
    }
