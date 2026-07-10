def compute_compression_rate(total_frames: int, selected_frames: int) -> tuple[float, float]:
    """
    Computes frame retention ratio and frame compression rate.
    """
    if total_frames <= 0:
        return 0.0, 1.0
    retention_ratio = selected_frames / total_frames
    compression_rate = 1.0 - retention_ratio
    return retention_ratio, compression_rate

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

def evaluate_event_retrieval(selected_timestamps: list[float], start_time: float, end_time: float, k: int) -> dict:
    """
    Evaluates retrieval for a single event and a specific top_k (k).
    
    Returns:
        {
            "temporal_hit": 1.0 or 0.0,
            "recall": 1.0 or 0.0,
            "mean_temporal_distance_seconds": float,
            "temporal_iou_proxy": float
        }
    """
    top_k_timestamps = selected_timestamps[:k]
    
    if not top_k_timestamps:
        return {
            "temporal_hit": 0.0,
            "recall": 0.0,
            "mean_temporal_distance_seconds": 999.0, # large default distance
            "temporal_iou_proxy": 0.0
        }
        
    hits = [is_temporal_hit(t, start_time, end_time) for t in top_k_timestamps]
    hit = 1.0 if any(hits) else 0.0
    recall = hit # Since single-event recall in this top-k context is 1 if event was hit, 0 otherwise.
    
    distances = [compute_temporal_distance(t, start_time, end_time) for t in top_k_timestamps]
    mean_distance = sum(distances) / len(distances)
    
    ious = [compute_temporal_iou_proxy(t, start_time, end_time) for t in top_k_timestamps]
    max_iou = max(ious) if ious else 0.0
    
    return {
        "temporal_hit": hit,
        "recall": recall,
        "mean_temporal_distance_seconds": mean_distance,
        "temporal_iou_proxy": max_iou
    }
