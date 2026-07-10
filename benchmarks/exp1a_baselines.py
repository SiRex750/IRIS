import os
import time
import random
import numpy as np
import av
from scipy.signal import find_peaks

# Optional imports
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import clip
    import torch
    from sklearn.cluster import KMeans
    from PIL import Image
except ImportError:
    clip = None

def get_fill_frames(all_indices: list[int], selected_indices: list[int], needed: int) -> list[int]:
    """
    Selects 'needed' additional indices uniformly from the pool of remaining unselected indices.
    """
    selected_set = set(selected_indices)
    pool = [idx for idx in all_indices if idx not in selected_set]
    if not pool:
        return []
    
    if len(pool) <= needed:
        return pool
    
    # Uniformly space additional frames from the pool
    idxs = np.round(np.linspace(0, len(pool) - 1, needed)).astype(int)
    return [pool[i] for i in idxs]

def uniform_budget_matched(raw_records: list[dict], budget_k: int) -> dict:
    t0 = time.perf_counter()
    n_frames = len(raw_records)
    selected_frames = []
    
    if n_frames > 0 and budget_k > 0:
        idxs = np.round(np.linspace(0, n_frames - 1, budget_k)).astype(int)
        # Avoid duplicate indices
        unique_idxs = []
        for idx in idxs:
            if idx not in unique_idxs:
                unique_idxs.append(int(idx))
        
        # Fill up to K if linspace rounding caused fewer than K unique frames
        if len(unique_idxs) < budget_k:
            all_idxs = [rec["frame_idx"] for rec in raw_records]
            fill = get_fill_frames(all_idxs, unique_idxs, budget_k - len(unique_idxs))
            unique_idxs.extend(fill)
            
        selected_frames = sorted(unique_idxs)

    selected_timestamps = [raw_records[idx]["timestamp"] for idx in selected_frames]
    ranked_timestamps = list(selected_timestamps) # Uniform has no natural ranking, use time order

    return {
        "method_name": "uniform_budget_matched",
        "selected_frames": selected_frames,
        "selected_timestamps": selected_timestamps,
        "ranked_timestamps": ranked_timestamps,
        "frame_metadata": [raw_records[idx] for idx in selected_frames],
        "budget_k": budget_k,
        "fill_count": 0,
        "unavailable_reason": None,
        "selection_time_seconds": time.perf_counter() - t0
    }

def random_budget_matched(raw_records: list[dict], budget_k: int, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    n_frames = len(raw_records)
    selected_frames = []
    
    if n_frames > 0 and budget_k > 0:
        all_idxs = [rec["frame_idx"] for rec in raw_records]
        rng = random.Random(seed)
        selected_frames = sorted(rng.sample(all_idxs, min(budget_k, n_frames)))
        
    selected_timestamps = [raw_records[idx]["timestamp"] for idx in selected_frames]
    ranked_timestamps = list(selected_timestamps)

    return {
        "method_name": f"random_budget_matched_seed_{seed}",
        "selected_frames": selected_frames,
        "selected_timestamps": selected_timestamps,
        "ranked_timestamps": ranked_timestamps,
        "frame_metadata": [raw_records[idx] for idx in selected_frames],
        "budget_k": budget_k,
        "fill_count": 0,
        "unavailable_reason": None,
        "selection_time_seconds": time.perf_counter() - t0
    }

def iframe_only_budget_matched(raw_records: list[dict], budget_k: int) -> dict:
    t0 = time.perf_counter()
    all_idxs = [rec["frame_idx"] for rec in raw_records]
    
    # Filter keyframes
    keyframes = [rec["frame_idx"] for rec in raw_records if rec.get("frame_type") == "I"]
    fill_count = 0
    selected_frames = []
    
    if len(keyframes) >= budget_k:
        # Sample K keyframes uniformly
        idxs = np.round(np.linspace(0, len(keyframes) - 1, budget_k)).astype(int)
        unique_idxs = list(dict.fromkeys(idxs))
        selected_frames = [keyframes[i] for i in unique_idxs]
        
        # In case unique_idxs is less than budget_k due to rounding
        if len(selected_frames) < budget_k:
            fill_pool = [k for k in keyframes if k not in selected_frames]
            selected_frames.extend(fill_pool[:budget_k - len(selected_frames)])
    else:
        selected_frames = list(keyframes)
        fill_count = budget_k - len(selected_frames)
        fill = get_fill_frames(all_idxs, selected_frames, fill_count)
        selected_frames.extend(fill)
        
    selected_frames = sorted(selected_frames)
    selected_timestamps = [raw_records[idx]["timestamp"] for idx in selected_frames]
    ranked_timestamps = list(selected_timestamps)

    return {
        "method_name": "iframe_only_budget_matched",
        "selected_frames": selected_frames,
        "selected_timestamps": selected_timestamps,
        "ranked_timestamps": ranked_timestamps,
        "frame_metadata": [raw_records[idx] for idx in selected_frames],
        "budget_k": budget_k,
        "fill_count": fill_count,
        "unavailable_reason": None,
        "selection_time_seconds": time.perf_counter() - t0
    }

def scene_change_detection(raw_records: list[dict], budget_k: int) -> dict:
    t0 = time.perf_counter()
    all_idxs = [rec["frame_idx"] for rec in raw_records]
    luma_diffs = np.array([rec.get("luma_diff_energy", 0.0) for rec in raw_records])
    
    # Find peaks on luma diff curve as scene changes
    peaks, properties = find_peaks(luma_diffs, distance=5, prominence=0.01)
    prominences = properties.get("prominences", np.zeros_like(peaks))
    
    # Sort peaks by prominence descending
    sorted_peak_indices = [peaks[i] for i in np.argsort(prominences)[::-1]]
    
    fill_count = 0
    selected_frames = []
    
    if len(sorted_peak_indices) >= budget_k:
        selected_frames = sorted_peak_indices[:budget_k]
    else:
        selected_frames = list(sorted_peak_indices)
        fill_count = budget_k - len(selected_frames)
        fill = get_fill_frames(all_idxs, selected_frames, fill_count)
        selected_frames.extend(fill)
        
    selected_frames = sorted(selected_frames)
    selected_timestamps = [raw_records[idx]["timestamp"] for idx in selected_frames]
    
    # Rank by local luma diff peak prominence (for peak frames) or luma diff energy directly
    luma_diff_map = {rec["frame_idx"]: rec.get("luma_diff_energy", 0.0) for rec in raw_records}
    selected_frames_sorted = sorted(selected_frames, key=lambda idx: luma_diff_map.get(idx, 0.0), reverse=True)
    ranked_timestamps = [raw_records[idx]["timestamp"] for idx in selected_frames_sorted]

    return {
        "method_name": "scene_change_detection",
        "selected_frames": selected_frames,
        "selected_timestamps": selected_timestamps,
        "ranked_timestamps": ranked_timestamps,
        "frame_metadata": [raw_records[idx] for idx in selected_frames],
        "budget_k": budget_k,
        "fill_count": fill_count,
        "unavailable_reason": None,
        "selection_time_seconds": time.perf_counter() - t0
    }

def optical_flow_sampling(video_path: str, raw_records: list[dict], budget_k: int) -> dict:
    t0 = time.perf_counter()
    if cv2 is None:
        return {
            "method_name": "optical_flow_sampling",
            "selected_frames": [],
            "selected_timestamps": [],
            "ranked_timestamps": [],
            "frame_metadata": [],
            "budget_k": budget_k,
            "fill_count": 0,
            "unavailable_reason": "cv2 (OpenCV) not installed",
            "selection_time_seconds": 0.0
        }
        
    flow_magnitudes = {}
    container = None
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        prev_gray = None
        
        idx = 0
        for frame in container.decode(video=0):
            # Downsize for speed on CPU
            try:
                Y = frame.to_ndarray(format='gray')
            except Exception:
                arr = frame.to_ndarray(format='yuv420p')
                Y = arr[0:frame.height, :]
            small_Y = cv2.resize(Y, (160, 120))
            
            if prev_gray is None:
                flow_mag = 0.0
            else:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, small_Y, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_mag = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
                
            flow_magnitudes[idx] = flow_mag
            prev_gray = small_Y
            idx += 1
    except Exception as e:
        if container:
            container.close()
        return {
            "method_name": "optical_flow_sampling",
            "selected_frames": [],
            "selected_timestamps": [],
            "ranked_timestamps": [],
            "frame_metadata": [],
            "budget_k": budget_k,
            "fill_count": 0,
            "unavailable_reason": f"Optical flow computation error: {e}",
            "selection_time_seconds": 0.0
        }
    finally:
        if container:
            container.close()
            
    # Rank frames by flow magnitude
    sorted_by_flow = sorted(flow_magnitudes.keys(), key=lambda k: flow_magnitudes[k], reverse=True)
    selected_frames = sorted(sorted_by_flow[:budget_k])
    selected_timestamps = [raw_records[i]["timestamp"] for i in selected_frames if i < len(raw_records)]
    
    # Fill if needed
    fill_count = 0
    if len(selected_frames) < budget_k:
        all_idxs = [rec["frame_idx"] for rec in raw_records]
        fill_count = budget_k - len(selected_frames)
        fill = get_fill_frames(all_idxs, selected_frames, fill_count)
        selected_frames.extend(fill)
        selected_frames = sorted(selected_frames)
        selected_timestamps = [raw_records[i]["timestamp"] for i in selected_frames]

    # Ranked timestamps
    selected_frames_sorted = sorted(selected_frames, key=lambda k: flow_magnitudes.get(k, 0.0), reverse=True)
    ranked_timestamps = [raw_records[i]["timestamp"] for i in selected_frames_sorted]

    return {
        "method_name": "optical_flow_sampling",
        "selected_frames": selected_frames,
        "selected_timestamps": selected_timestamps,
        "ranked_timestamps": ranked_timestamps,
        "frame_metadata": [raw_records[i] for i in selected_frames],
        "budget_k": budget_k,
        "fill_count": fill_count,
        "unavailable_reason": None,
        "selection_time_seconds": time.perf_counter() - t0
    }

def clip_clustering(video_path: str, raw_records: list[dict], budget_k: int) -> dict:
    t0 = time.perf_counter()
    if clip is None:
        return {
            "method_name": "clip_clustering",
            "selected_frames": [],
            "selected_timestamps": [],
            "ranked_timestamps": [],
            "frame_metadata": [],
            "budget_k": budget_k,
            "fill_count": 0,
            "unavailable_reason": "CLIP, PyTorch or scikit-learn not installed",
            "selection_time_seconds": 0.0
        }
        
    container = None
    try:
        # Load model on CPU
        device = "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        # Subsample frames to run clustering faster (e.g. at most 100 candidate frames)
        n_frames = len(raw_records)
        step = max(1, n_frames // 100)
        sampled_idxs = list(range(0, n_frames, step))
        if n_frames - 1 not in sampled_idxs:
            sampled_idxs.append(n_frames - 1)
            
        sampled_set = set(sampled_idxs)
        embeddings = {}
        
        container = av.open(video_path)
        idx = 0
        for frame in container.decode(video=0):
            if idx in sampled_set:
                pil_image = frame.to_image()
                image_input = preprocess(pil_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model.encode_image(image_input)
                    features /= features.norm(dim=-1, keepdim=True)
                    embeddings[idx] = features.cpu().numpy().flatten()
            idx += 1
        container.close()
        container = None
        
        # Perform KMeans clustering
        idxs = list(embeddings.keys())
        features_matrix = np.array([embeddings[i] for i in idxs])
        
        n_clusters = min(budget_k, len(idxs))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        kmeans.fit(features_matrix)
        
        # For each cluster, pick the frame closest to the center
        selected_frames = []
        for cluster_idx in range(n_clusters):
            center = kmeans.cluster_centers_[cluster_idx]
            cluster_members = [idxs[i] for i in range(len(idxs)) if kmeans.labels_[i] == cluster_idx]
            dists = [np.linalg.norm(embeddings[m] - center) for m in cluster_members]
            closest_member = cluster_members[np.argmin(dists)]
            selected_frames.append(closest_member)
            
        selected_frames = sorted(selected_frames)
        fill_count = 0
        if len(selected_frames) < budget_k:
            all_idxs = [rec["frame_idx"] for rec in raw_records]
            fill_count = budget_k - len(selected_frames)
            fill = get_fill_frames(all_idxs, selected_frames, fill_count)
            selected_frames.extend(fill)
            selected_frames = sorted(selected_frames)
            
    except Exception as e:
        if container:
            container.close()
        return {
            "method_name": "clip_clustering",
            "selected_frames": [],
            "selected_timestamps": [],
            "ranked_timestamps": [],
            "frame_metadata": [],
            "budget_k": budget_k,
            "fill_count": 0,
            "unavailable_reason": f"CLIP clustering execution error: {e}",
            "selection_time_seconds": 0.0
        }
        
    selected_timestamps = [raw_records[i]["timestamp"] for i in selected_frames]
    ranked_timestamps = list(selected_timestamps) # Default time order for clusters

    return {
        "method_name": "clip_clustering",
        "selected_frames": selected_frames,
        "selected_timestamps": selected_timestamps,
        "ranked_timestamps": ranked_timestamps,
        "frame_metadata": [raw_records[i] for i in selected_frames],
        "budget_k": budget_k,
        "fill_count": fill_count,
        "unavailable_reason": None,
        "selection_time_seconds": time.perf_counter() - t0
    }

def luma_diff_topk(raw_records: list[dict], budget_k: int) -> dict:
    t0 = time.perf_counter()
    
    # Sort all frames by luma diff energy descending
    sorted_by_diff = sorted(raw_records, key=lambda x: x.get("luma_diff_energy", 0.0), reverse=True)
    
    selected_records = sorted(sorted_by_diff[:budget_k], key=lambda x: x["frame_idx"])
    selected_frames = [r["frame_idx"] for r in selected_records]
    selected_timestamps = [r["timestamp"] for r in selected_records]
    
    # Ranked timestamps
    ranked_timestamps = [r["timestamp"] for r in sorted_by_diff[:budget_k]]

    return {
        "method_name": "luma_diff_topk",
        "selected_frames": selected_frames,
        "selected_timestamps": selected_timestamps,
        "ranked_timestamps": ranked_timestamps,
        "frame_metadata": selected_records,
        "budget_k": budget_k,
        "fill_count": 0,
        "unavailable_reason": None,
        "selection_time_seconds": time.perf_counter() - t0
    }
