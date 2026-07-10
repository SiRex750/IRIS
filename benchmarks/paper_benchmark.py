import os
import sys
import time
import random
import argparse
import yaml
import json
import pandas as pd
import numpy as np
import av
import torch
import clip
import cv2
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import linear_sum_assignment

# Local project imports
from iris.charon_v import parse_video
from iris.action_score import ActionScoreModule, ActionScoreConfig
from iris.pipeline import wrapper_l2_retrieve
from iris.iris_config import IRISConfig
from iris.l2_asphodel import L2Asphodel
import benchmarks.exp1a_metrics as metrics
from benchmarks.exp1a_baselines import (
    uniform_budget_matched,
    iframe_only_budget_matched,
    luma_diff_topk,
    get_fill_frames
)

def fix_path(p):
    if not isinstance(p, str):
        return p
    if "\\" in p or p.startswith("c:") or p.startswith("C:"):
        p = p.replace("c:\\Users\\swara\\IRIS\\", "/home/ccbd/IRIS/")
        p = p.replace("C:\\Users\\swara\\IRIS\\", "/home/ccbd/IRIS/")
        p = p.replace("\\", "/")
    return p

_CLIP_EMBEDDINGS_CACHE = {}
_CLIP_MODEL_CACHE = None
_CLIP_PREPROCESS_CACHE = None

def get_cached_clip(device="cuda"):
    global _CLIP_MODEL_CACHE, _CLIP_PREPROCESS_CACHE
    if _CLIP_MODEL_CACHE is None:
        import clip
        _CLIP_MODEL_CACHE, _CLIP_PREPROCESS_CACHE = clip.load("ViT-B/32", device=device)
        _CLIP_MODEL_CACHE.eval()
    return _CLIP_MODEL_CACHE, _CLIP_PREPROCESS_CACHE

def encode_video_frames_clip(video_path: str, frame_indices: list[int], device: str = "cuda") -> dict[int, np.ndarray]:
    global _CLIP_EMBEDDINGS_CACHE
    if video_path not in _CLIP_EMBEDDINGS_CACHE:
        _CLIP_EMBEDDINGS_CACHE[video_path] = {}
        
    cache = _CLIP_EMBEDDINGS_CACHE[video_path]
    missing_indices = [idx for idx in frame_indices if idx not in cache]
    
    if missing_indices:
        model, preprocess = get_cached_clip(device=device)
        
        container = av.open(video_path)
        target_set = set(missing_indices)
        batch_images = []
        batch_idxs = []
        batch_size = 128
        
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in target_set:
                pil_img = frame.to_image()
                img_tensor = preprocess(pil_img)
                batch_images.append(img_tensor)
                batch_idxs.append(idx)
                
                if len(batch_images) >= batch_size:
                    with torch.no_grad():
                        inp = torch.stack(batch_images).to(device)
                        feats = model.encode_image(inp)
                        feats /= feats.norm(dim=-1, keepdim=True)
                        feats_np = feats.cpu().numpy()
                        for b_idx, feat in zip(batch_idxs, feats_np):
                            cache[b_idx] = feat
                    batch_images = []
                    batch_idxs = []
                    
        if batch_images:
            with torch.no_grad():
                inp = torch.stack(batch_images).to(device)
                feats = model.encode_image(inp)
                feats /= feats.norm(dim=-1, keepdim=True)
                feats_np = feats.cpu().numpy()
                for b_idx, feat in zip(batch_idxs, feats_np):
                    cache[b_idx] = feat
                    
        container.close()
        
    return {idx: cache[idx] for idx in frame_indices if idx in cache}

def fast_wrapper_l2_retrieve(video_path: str, query: str, frames_to_index: list[dict], config=None) -> list[dict]:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import clip
    model, _ = get_cached_clip(device=device)
    text_input = clip.tokenize([query]).to(device)
    with torch.no_grad():
        query_features = model.encode_text(text_input)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        query_embedding = query_features.cpu().numpy().flatten().astype(np.float32)
        
    graph = L2Asphodel(config=config)
    feature_recs = []
    score_recs = []
    enrichment_map = {}
    
    frame_idxs = [f["frame_idx"] for f in frames_to_index]
    embeddings_dict = encode_video_frames_clip(video_path, frame_idxs, device=device)
    
    for f in frames_to_index:
        feature_rec = {
            "frame_idx": f["frame_idx"],
            "timestamp": f["timestamp"],
            "luma_diff_energy": f.get("luma_diff_energy", 0.0),
            "motion_magnitude": 0.0,
            "luma_entropy": 0.0,
            "refined_motion_tensor": np.zeros(1, dtype=np.float32)
        }
        score_rec = {
            "action_score": f["action_score"],
            "persistence_value": f["persistence_value"]
        }
        feature_recs.append(feature_rec)
        score_recs.append(score_rec)
        emb = embeddings_dict.get(f["frame_idx"], np.zeros(512, dtype=np.float32))
        enrichment_map[f["frame_idx"]] = emb
        
    graph.add_frame_nodes_bulk(feature_recs, score_recs)
    graph.enrich_nodes_bulk(enrichment_map)
    
    if frames_to_index:
        query_action_score = max(f.get("action_score", 0.0) for f in frames_to_index)
    else:
        query_action_score = 0.5
        
    retrieved_nodes = graph.retrieve(query_embedding, query_action_score=query_action_score, top_k=config.l2_retrieve_top_k)
    
    retrieved = []
    for node in retrieved_nodes:
        retrieved.append({
            "frame_idx": node.frame_idx,
            "timestamp": node.timestamp,
            "luma_diff_energy": node.luma_diff_energy,
            "action_score": node.action_score,
            "persistence_value": node.persistence_value
        })
    return retrieved

def scene_change_detection(video_path: str, raw_records: list[dict], budget_k: int) -> dict:
    t0 = time.perf_counter()
    from scenedetect import detect, ContentDetector
    all_idxs = [rec["frame_idx"] for rec in raw_records]
    try:
        scene_list = detect(video_path, ContentDetector())
        scene_frames = [scene[0].frame_num for scene in scene_list]
    except Exception:
        scene_frames = []
        
    selected_frames = []
    fill_count = 0
    if len(scene_frames) >= budget_k:
        selected_frames = scene_frames[:budget_k]
    else:
        selected_frames = list(scene_frames)
        fill_count = budget_k - len(selected_frames)
        fill = get_fill_frames(all_idxs, selected_frames, fill_count)
        selected_frames.extend(fill)
        
    selected_frames = sorted(list(set(selected_frames)))
    if len(selected_frames) < budget_k:
        fill = get_fill_frames(all_idxs, selected_frames, budget_k - len(selected_frames))
        selected_frames.extend(fill)
        selected_frames = sorted(selected_frames)
        
    selected_timestamps = [raw_records[idx]["timestamp"] for idx in selected_frames]
    ranked_timestamps = list(selected_timestamps)
    
    return {
        "method_name": "scene_change",
        "selected_frames": selected_frames,
        "selected_timestamps": selected_timestamps,
        "ranked_timestamps": ranked_timestamps,
        "frame_metadata": [raw_records[idx] for idx in selected_frames],
        "budget_k": budget_k,
        "fill_count": fill_count,
        "unavailable_reason": None,
        "selection_time_seconds": time.perf_counter() - t0
    }

def optical_flow_farneback(video_path: str, raw_records: list[dict], budget_k: int) -> dict:
    t0 = time.perf_counter()
    flow_magnitudes = {}
    container = None
    all_idxs = [rec["frame_idx"] for rec in raw_records]
    num_frames = len(all_idxs)
    stride = max(5, num_frames // 50)  # Target ~50 flow calculations per video, min stride 5
    try:
        container = av.open(video_path)
        prev_gray = None
        for idx, frame in enumerate(container.decode(video=0)):
            if idx % stride != 0 and idx != num_frames - 1:
                continue
                
            h, w = frame.height, frame.width
            if w >= h:
                new_w = 160
                new_h = int(h * (160.0 / w))
            else:
                new_h = 160
                new_w = int(w * (160.0 / h))
            
            try:
                Y = frame.to_ndarray(format='gray')
            except Exception:
                arr = frame.to_ndarray(format='yuv420p')
                Y = arr[0:h, :]
            small_Y = cv2.resize(Y, (new_w, new_h))
            
            if prev_gray is None:
                flow_mag = 0.0
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, small_Y, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                flow_mag = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
                
            flow_magnitudes[idx] = flow_mag
            prev_gray = small_Y
    except Exception as e:
        return {
            "method_name": "optical_flow_farneback",
            "selected_frames": [],
            "selected_timestamps": [],
            "ranked_timestamps": [],
            "frame_metadata": [],
            "budget_k": budget_k,
            "fill_count": 0,
            "unavailable_reason": f"Farneback computation error: {e}",
            "selection_time_seconds": 0.0
        }
    finally:
        if container:
            container.close()
            
    # Interpolate flow magnitudes for unsampled indices
    last_mag = 0.0
    for idx in range(num_frames):
        if idx not in flow_magnitudes:
            flow_magnitudes[idx] = last_mag
        else:
            last_mag = flow_magnitudes[idx]
            
    flow_magnitudes[0] = 0.0
    sorted_by_flow = sorted(flow_magnitudes.keys(), key=lambda k: flow_magnitudes[k], reverse=True)
    selected_frames = sorted(sorted_by_flow[:budget_k])
    
    fill_count = 0
    all_idxs = [rec["frame_idx"] for rec in raw_records]
    if len(selected_frames) < budget_k:
        fill_count = budget_k - len(selected_frames)
        fill = get_fill_frames(all_idxs, selected_frames, fill_count)
        selected_frames.extend(fill)
        selected_frames = sorted(selected_frames)
        
    selected_timestamps = [raw_records[i]["timestamp"] for i in selected_frames if i < len(raw_records)]
    selected_frames_sorted = sorted(selected_frames, key=lambda k: flow_magnitudes.get(k, 0.0), reverse=True)
    ranked_timestamps = [raw_records[i]["timestamp"] for i in selected_frames_sorted if i < len(raw_records)]
    
    return {
        "method_name": "optical_flow_farneback",
        "selected_frames": selected_frames,
        "selected_timestamps": selected_timestamps,
        "ranked_timestamps": ranked_timestamps,
        "frame_metadata": [raw_records[i] for i in selected_frames if i < len(raw_records)],
        "budget_k": budget_k,
        "fill_count": fill_count,
        "unavailable_reason": None,
        "selection_time_seconds": time.perf_counter() - t0
    }

def clip_kmeans_diversity(video_path: str, raw_records: list[dict], budget_k: int, device: str = "cuda") -> dict:
    t0 = time.perf_counter()
    all_idxs = [rec["frame_idx"] for rec in raw_records]
    stride = max(3, len(all_idxs) // (budget_k * 2))
    strided_idxs = all_idxs[::stride]
    if all_idxs and all_idxs[-1] not in strided_idxs:
        strided_idxs.append(all_idxs[-1])
    try:
        embeddings_dict = encode_video_frames_clip(video_path, strided_idxs, device=device)
    except Exception as e:
        return {
            "method_name": "clip_kmeans_diversity",
            "selected_frames": [],
            "selected_timestamps": [],
            "ranked_timestamps": [],
            "frame_metadata": [],
            "budget_k": budget_k,
            "fill_count": 0,
            "unavailable_reason": f"CLIP encoding error: {e}",
            "selection_time_seconds": 0.0
        }
        
    idxs = sorted(list(embeddings_dict.keys()))
    if not idxs:
        return {
            "method_name": "clip_kmeans_diversity",
            "selected_frames": [],
            "selected_timestamps": [],
            "ranked_timestamps": [],
            "frame_metadata": [],
            "budget_k": budget_k,
            "fill_count": 0,
            "unavailable_reason": "No frames encoded",
            "selection_time_seconds": 0.0
        }
        
    features_matrix = np.array([embeddings_dict[i] for i in idxs])
    n_clusters = min(budget_k, len(idxs))
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init=10,
        max_iter=100,
        batch_size=min(4096, len(idxs)),
        reassignment_ratio=0
    )
    kmeans.fit(features_matrix)
    
    dists = np.linalg.norm(kmeans.cluster_centers_[:, None, :] - features_matrix[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(dists)
    selected_frames = sorted([idxs[j] for j in col_ind])
    
    fill_count = 0
    if len(selected_frames) < budget_k:
        fill_count = budget_k - len(selected_frames)
        fill = get_fill_frames(all_idxs, selected_frames, fill_count)
        selected_frames.extend(fill)
        selected_frames = sorted(selected_frames)
        
    selected_timestamps = [raw_records[i]["timestamp"] for i in selected_frames if i < len(raw_records)]
    ranked_timestamps = list(selected_timestamps)
    
    return {
        "method_name": "clip_kmeans_diversity",
        "selected_frames": selected_frames,
        "selected_timestamps": selected_timestamps,
        "ranked_timestamps": ranked_timestamps,
        "frame_metadata": [raw_records[i] for i in selected_frames if i < len(raw_records)],
        "budget_k": budget_k,
        "fill_count": fill_count,
        "unavailable_reason": None,
        "selection_time_seconds": time.perf_counter() - t0
    }

def clip_query_topk(video_path: str, raw_records: list[dict], query: str, budget_k: int, device: str = "cuda") -> dict:
    t0 = time.perf_counter()
    all_idxs = [rec["frame_idx"] for rec in raw_records]
    strided_idxs = all_idxs[::3]
    if all_idxs and all_idxs[-1] not in strided_idxs:
        strided_idxs.append(all_idxs[-1])
    try:
        model, _ = get_cached_clip(device=device)
        text_input = clip.tokenize([query]).to(device)
        with torch.no_grad():
            query_features = model.encode_text(text_input)
            query_features /= query_features.norm(dim=-1, keepdim=True)
            query_emb = query_features.cpu().numpy().flatten()
            
        embeddings_dict = encode_video_frames_clip(video_path, strided_idxs, device=device)
    except Exception as e:
        return {
            "method_name": "clip_query_topk",
            "selected_frames": [],
            "selected_timestamps": [],
            "ranked_timestamps": [],
            "frame_metadata": [],
            "budget_k": budget_k,
            "fill_count": 0,
            "unavailable_reason": f"CLIP query topk error: {e}",
            "selection_time_seconds": 0.0
        }
        
    similarities = {}
    for idx, emb in embeddings_dict.items():
        norm_val = np.linalg.norm(emb)
        if norm_val > 0:
            similarities[idx] = float(np.dot(emb, query_emb) / norm_val)
        else:
            similarities[idx] = 0.0
            
    sorted_by_sim = sorted(similarities.keys(), key=lambda k: similarities[k], reverse=True)
    selected_frames = sorted(sorted_by_sim[:budget_k])
    
    fill_count = 0
    if len(selected_frames) < budget_k:
        fill_count = budget_k - len(selected_frames)
        fill = get_fill_frames(all_idxs, selected_frames, fill_count)
        selected_frames.extend(fill)
        selected_frames = sorted(selected_frames)
        
    selected_timestamps = [raw_records[i]["timestamp"] for i in selected_frames if i < len(raw_records)]
    selected_frames_sorted = sorted(selected_frames, key=lambda k: similarities.get(k, 0.0), reverse=True)
    ranked_timestamps = [raw_records[i]["timestamp"] for i in selected_frames_sorted if i < len(raw_records)]
    
    return {
        "method_name": "clip_query_topk",
        "selected_frames": selected_frames,
        "selected_timestamps": selected_timestamps,
        "ranked_timestamps": ranked_timestamps,
        "frame_metadata": [raw_records[i] for i in selected_frames if i < len(raw_records)],
        "budget_k": budget_k,
        "fill_count": fill_count,
        "unavailable_reason": None,
        "selection_time_seconds": time.perf_counter() - t0
    }

def packet_size_only(raw_records: list[dict], budget_k: int) -> dict:
    t0 = time.perf_counter()
    sorted_by_size = sorted(raw_records, key=lambda x: x.get("packet_size", 0.0), reverse=True)
    selected_records = sorted(sorted_by_size[:budget_k], key=lambda x: x["frame_idx"])
    selected_frames = [r["frame_idx"] for r in selected_records]
    selected_timestamps = [r["timestamp"] for r in selected_records]
    ranked_timestamps = [r["timestamp"] for r in sorted_by_size[:budget_k]]
    
    return {
        "method_name": "packet_size_only",
        "selected_frames": selected_frames,
        "selected_timestamps": selected_timestamps,
        "ranked_timestamps": ranked_timestamps,
        "frame_metadata": selected_records,
        "budget_k": budget_k,
        "fill_count": 0,
        "unavailable_reason": None,
        "selection_time_seconds": time.perf_counter() - t0
    }

def main():
    parser = argparse.ArgumentParser(description="Remote-GPU Paper-Grade Video Benchmark")
    parser.add_argument("--config", type=str, default="configs/paper_benchmark.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--pilot-test", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    import iris.pipeline
    iris.pipeline.get_semantic_and_clip_caption = lambda pil_img, frame, clip_emb, device: {"clip_label": "", "semantic_caption": ""}
    iris.pipeline.wrapper_l2_retrieve = fast_wrapper_l2_retrieve
    global wrapper_l2_retrieve
    wrapper_l2_retrieve = fast_wrapper_l2_retrieve
    os.makedirs(config["out_dir"], exist_ok=True)
    
    df_data = pd.read_csv(config["dataset_csv"])
    df_data["path"] = df_data["path"].apply(fix_path)
    
    unique_vids = df_data["video_id"].unique()
    if args.smoke_test:
        unique_vids = unique_vids[:1]
        print(f"Smoke test mode: running on 1 video ({unique_vids[0]})")
    elif args.pilot_test:
        unique_vids = unique_vids[:int(max(1, len(unique_vids) * 0.01))]
        print(f"Pilot test mode: running on {len(unique_vids)} videos")
        
    # Build a lookup for queries
    video_groups = df_data[df_data["video_id"].isin(unique_vids)].groupby("video_id")
    
    device = config.get("device", "cuda")
    print(f"Using device: {device}")
    
    per_event_records = []
    
    for video_id, group in video_groups:
        video_path = group.iloc[0]["path"]
        if not os.path.exists(video_path):
            print(f"Skip missing video {video_path}")
            continue
            
        print(f"Processing video {video_id}...")
        
        # Parse video
        output_frames, stats, raw_records = parse_video(
            video_path,
            return_stats=True,
            return_raw=True,
            adaptive=True
        )
        
        total_frames = stats["total"]
        
        # Proposed selection K
        selected_idxs = sorted([f["frame_idx"] for f in output_frames])
        budget_k = len(selected_idxs)
        if budget_k == 0:
            budget_k = 1
            selected_idxs = [0]
            
        # Get frame metadata mappings
        scorer = ActionScoreModule(ActionScoreConfig())
        feature_dicts = [{
            "frame_idx": rec["frame_idx"],
            "packet_size": rec.get("packet_size", 0.0),
            "motion_magnitude": rec.get("motion_magnitude", 0.0),
            "luma_entropy": rec.get("luma_entropy", 0.0)
        } for rec in raw_records]
        scored_records = scorer.score_all(feature_dicts)
        action_map = {r["frame_idx"]: r for r in scored_records}
        for rec in raw_records:
            fidx = rec["frame_idx"]
            if fidx in action_map:
                action_map[fidx]["timestamp"] = rec["timestamp"]
                action_map[fidx]["luma_diff_energy"] = rec.get("luma_diff_energy", 0.0)
                action_map[fidx]["frame_type"] = rec.get("frame_type", "P")
                
        # Fill selected frames with scored details
        iris_selected_sorted = sorted(
            [action_map[idx] for idx in selected_idxs if idx in action_map],
            key=lambda x: x["action_score"] + x["persistence_value"],
            reverse=True
        )
        selected_idxs = sorted([f["frame_idx"] for f in iris_selected_sorted])
        selected_ts = [action_map[idx]["timestamp"] for idx in selected_idxs]
        ranked_ts = [f["timestamp"] for f in iris_selected_sorted]
        
        selections = {}
        
        # 1. Proposed model (selection only)
        selections["proposed_model_selection"] = {
            "method_name": "proposed_model_selection",
            "selected_frames": selected_idxs,
            "selected_timestamps": selected_ts,
            "ranked_timestamps": ranked_ts,
            "frame_metadata": [action_map[idx] for idx in selected_idxs],
            "budget_k": budget_k
        }
        
        # Run baselines
        selections["uniform"] = uniform_budget_matched(raw_records, budget_k)
        selections["iframe_prior"] = iframe_only_budget_matched(raw_records, budget_k)
        selections["luma_difference"] = luma_diff_topk(raw_records, budget_k)
        selections["packet_size_only"] = packet_size_only(raw_records, budget_k)
        
        # Random across 30 seeds
        random_selections = []
        for s in config.get("random_seeds", range(30)):
            r_sel = random_budget_matched(raw_records, budget_k, seed=s)
            random_selections.append(r_sel)
            
        # Optional external tools (scene_change, optical_flow, clip)
        selections["scene_change"] = scene_change_detection(video_path, raw_records, budget_k)
        selections["optical_flow_farneback"] = optical_flow_farneback(video_path, raw_records, budget_k)
        selections["clip_kmeans_diversity"] = clip_kmeans_diversity(video_path, raw_records, budget_k, device=device)
        
        # Iterate over event questions
        for idx, event_row in group.iterrows():
            start_time = event_row["start_time"]
            end_time = event_row["end_time"]
            query = event_row["query"]
            event_label = event_row["event_label"]
            
            # Query specific selections
            selections["clip_query_topk"] = clip_query_topk(video_path, raw_records, query, budget_k, device=device)
            
            # Full proposed system (with graph retrieval)
            t_full_start = time.perf_counter()
            full_retrieved_ts = []
            try:
                frames_to_index = [{
                    "frame_idx": f["frame_idx"],
                    "timestamp": f["timestamp"],
                    "luma_diff_energy": f.get("luma_diff_energy", 0.0),
                    "action_score": f.get("action_score", 0.0),
                    "persistence_value": f.get("persistence_value", 0.0)
                } for f in selections["proposed_model_selection"]["frame_metadata"]]
                
                iris_cfg = IRISConfig(l2_retrieve_top_k=budget_k)
                retrieved_nodes = wrapper_l2_retrieve(video_path, query, frames_to_index, config=iris_cfg)
                full_retrieved_ts = [r["timestamp"] for r in retrieved_nodes]
            except Exception as e:
                print(f"Proposed system graph retrieval failed: {e}")
                
            if len(full_retrieved_ts) < budget_k:
                full_retrieved_ts.extend(selections["proposed_model_selection"]["ranked_timestamps"][:budget_k - len(full_retrieved_ts)])
            full_retrieved_ts = full_retrieved_ts[:budget_k]
            
            selections["proposed_system"] = {
                "method_name": "proposed_system",
                "selected_frames": selections["proposed_model_selection"]["selected_frames"],
                "selected_timestamps": selections["proposed_model_selection"]["selected_timestamps"],
                "ranked_timestamps": full_retrieved_ts,
                "frame_metadata": selections["proposed_model_selection"]["frame_metadata"],
                "budget_k": budget_k
            }
            
            # Track C Graph Ablations (using retrieve_ppr on Asphodel)
            ablations = run_track_c_ablations(video_path, query, selections["proposed_model_selection"]["frame_metadata"], budget_k)
            selections.update(ablations)
            
            # Evaluate all methods
            methods_to_eval = [
                "proposed_model_selection", "proposed_system", "uniform", "iframe_prior",
                "scene_change", "luma_difference", "optical_flow_farneback",
                "clip_kmeans_diversity", "clip_query_topk", "packet_size_only",
                "semantic_only_ppr", "codec_only_ppr", "hybrid_ppr"
            ]
            
            # Add random seeds
            for s_idx, r_sel in enumerate(random_selections):
                m_name = f"random_seed_{s_idx}"
                selections[m_name] = r_sel
                
            for m_name in methods_to_eval + [f"random_seed_{i}" for i in range(len(random_selections))]:
                if m_name not in selections or selections[m_name].get("unavailable_reason"):
                    continue
                    
                sel = selections[m_name]
                cov_res = metrics.evaluate_selection_coverage(sel["selected_timestamps"], start_time, end_time)
                
                # Frames in Window calculation
                hits = [metrics.is_temporal_hit(t, start_time, end_time) for t in sel["selected_timestamps"]]
                fiw = sum(hits) / len(hits) if hits else 0.0
                
                # Retrieve top-K metrics
                k_list = [1, 5, 10]
                recalls = {}
                for k in k_list:
                    recalls[k] = metrics.evaluate_ranked_retrieval(sel["ranked_timestamps"], start_time, end_time, k)["recall"]
                    
                per_event_records.append({
                    "video_id": video_id,
                    "query": query,
                    "event_label": event_label,
                    "start_time": start_time,
                    "end_time": end_time,
                    "method_name": m_name,
                    "budget_k": budget_k,
                    "window_coverage": cov_res["event_hit_any_selected"],
                    "fiw": fiw,
                    "min_temp_dist": cov_res["min_temporal_distance_seconds"],
                    "recall_at_1": recalls[1],
                    "recall_at_5": recalls[5],
                    "recall_at_10": recalls[10]
                })
        # Save CSV outputs incrementally
        df_temp = pd.DataFrame(per_event_records)
        df_temp.to_csv(os.path.join(config["out_dir"], "per_event_results.csv"), index=False)
        import sys
        sys.stdout.flush()
        
    # Save CSV outputs
    df_results = pd.DataFrame(per_event_records)
    df_results.to_csv(os.path.join(config["out_dir"], "per_event_results.csv"), index=False)
    
    # Statistical analysis & report generation
    generate_statistics_and_plots(df_results, config)
    print("Benchmark completed successfully.")

def run_track_c_ablations(video_path: str, query: str, frame_metadata: list[dict], budget_k: int) -> dict:
    # 1. Build the Asphodel graph
    frames_to_index = [{
        "frame_idx": f["frame_idx"],
        "timestamp": f["timestamp"],
        "luma_diff_energy": f.get("luma_diff_energy", 0.0),
        "action_score": f.get("action_score", 0.0),
        "persistence_value": f.get("persistence_value", 0.0)
    } for f in frame_metadata]
    
    import clip
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, _ = get_cached_clip(device=device)
    text_input = clip.tokenize([query]).to(device)
    with torch.no_grad():
        query_features = model.encode_text(text_input)
        query_features /= query_features.norm(dim=-1, keepdim=True)
        query_embedding = query_features.cpu().numpy().flatten().astype(np.float32)
        
    embeddings_dict = encode_video_frames_clip(video_path, [f["frame_idx"] for f in frame_metadata], device=device)
    
    ablations = {}
    
    # helper function to run retrieve_ppr under different weights
    def run_ppr_with_weights(alpha, beta, lambda_val, method_name):
        graph = L2Asphodel(config={"alpha": alpha, "beta": beta})
        # Add frame nodes
        node_records = []
        enrichment_records = []
        for f in frames_to_index:
            feature_rec = {
                "frame_idx": f["frame_idx"],
                "timestamp": f["timestamp"],
                "luma_diff_energy": f["luma_diff_energy"],
                "motion_magnitude": 0.0,
                "luma_entropy": 0.0,
                "refined_motion_tensor": np.zeros(1, dtype=np.float32)
            }
            score_rec = {
                "action_score": f["action_score"],
                "persistence_value": f["persistence_value"]
            }
            node_records.append((feature_rec, score_rec))
            emb = embeddings_dict.get(f["frame_idx"], np.zeros(512, dtype=np.float32))
            enrichment_records.append((f["frame_idx"], [], emb))
            
        feature_recs = [rec[0] for rec in node_records]
        score_recs = [rec[1] for rec in node_records]
        enrichment_map = {rec[0]: rec[2] for rec in enrichment_records}
        graph.add_frame_nodes_bulk(feature_recs, score_recs)
        graph.enrich_nodes_bulk(enrichment_map)
        
        ret_nodes = graph.retrieve_ppr(query_embedding, top_k=budget_k, lambda_=lambda_val)
        ret_ts = [n.timestamp for n in ret_nodes]
        if len(ret_ts) < budget_k:
            ret_ts.extend([f["timestamp"] for f in frames_to_index[:budget_k - len(ret_ts)]])
        ret_ts = ret_ts[:budget_k]
        
        return {
            "method_name": method_name,
            "selected_frames": [f["frame_idx"] for f in frames_to_index],
            "selected_timestamps": [f["timestamp"] for f in frames_to_index],
            "ranked_timestamps": ret_ts,
            "budget_k": budget_k
        }
        
    # Semantic-only PPR
    ablations["semantic_only_ppr"] = run_ppr_with_weights(1.0, 0.0, 1.0, "semantic_only_ppr")
    # Codec-only PPR
    ablations["codec_only_ppr"] = run_ppr_with_weights(0.0, 1.0, 0.0, "codec_only_ppr")
    # Hybrid PPR
    ablations["hybrid_ppr"] = run_ppr_with_weights(0.4, 0.6, 0.5, "hybrid_ppr")
    
    return ablations

def random_budget_matched(raw_records: list[dict], budget_k: int, seed: int = 0) -> dict:
    n_frames = len(raw_records)
    selected_frames = []
    if n_frames > 0 and budget_k > 0:
        all_idxs = [rec["frame_idx"] for rec in raw_records]
        rng = random.Random(seed)
        selected_frames = sorted(rng.sample(all_idxs, min(budget_k, n_frames)))
        
    selected_timestamps = [raw_records[idx]["timestamp"] for idx in selected_frames]
    ranked_timestamps = list(selected_timestamps)
    
    return {
        "method_name": f"random_seed_{seed}",
        "selected_frames": selected_frames,
        "selected_timestamps": selected_timestamps,
        "ranked_timestamps": ranked_timestamps,
        "frame_metadata": [raw_records[idx] for idx in selected_frames],
        "budget_k": budget_k,
        "fill_count": 0,
        "unavailable_reason": None,
        "selection_time_seconds": 0.0
    }

def get_sha256(filepath):
    import hashlib
    h = hashlib.sha256()
    if not os.path.exists(filepath):
        return ""
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def generate_statistics_and_plots(df_results: pd.DataFrame, config: dict):
    random_cols = [c for c in df_results["method_name"].unique() if c.startswith("random_seed_")]
    
    df_non_random = df_results[~df_results["method_name"].str.startswith("random_seed_")]
    df_random = df_results[df_results["method_name"].str.startswith("random_seed_")]
    
    df_random_mean = df_random.groupby(["video_id", "query", "event_label", "start_time", "end_time"]).mean(numeric_only=True).reset_index()
    df_random_mean["method_name"] = "random"
    
    df_all = pd.concat([df_non_random, df_random_mean], ignore_index=True)
    summary = df_all.groupby("method_name")[["window_coverage", "fiw", "recall_at_1", "recall_at_5", "recall_at_10"]].mean().reset_index()
    
    bootstrap_seed = config.get("bootstrap_seed", 20260710)
    replicates = config.get("bootstrap_replicates", 10000)
    
    unique_vids = df_all["video_id"].unique()
    if len(unique_vids) <= 1:
        replicates = 5
        
    np.random.seed(bootstrap_seed)
    
    boot_stats = []
    for _ in range(replicates):
        boot_vids = np.random.choice(unique_vids, size=len(unique_vids), replace=True)
        boot_sample = pd.concat([df_all[df_all["video_id"] == vid] for vid in boot_vids], ignore_index=True)
        boot_mean = boot_sample.groupby("method_name")["window_coverage"].mean()
        boot_stats.append(boot_mean)
        
    df_boot = pd.DataFrame(boot_stats)
    ci_lower = df_boot.quantile(0.025)
    ci_upper = df_boot.quantile(0.975)
    
    summary["ci_lower"] = summary["method_name"].map(ci_lower)
    summary["ci_upper"] = summary["method_name"].map(ci_upper)
    
    rename_map = {
        "proposed_model_selection": "Proposed Model (Selection Only)",
        "proposed_system": "Proposed System (Graph Retrieval)",
        "uniform": "Uniform",
        "iframe_prior": "I-frame Prior",
        "scene_change": "Scene Change",
        "luma_difference": "Luma Difference",
        "optical_flow_farneback": "Classical Optical Flow (Farneback)",
        "clip_kmeans_diversity": "CLIP K-Means Diversity",
        "clip_query_topk": "CLIP Query Top-K",
        "packet_size_only": "Coded Packet Size Only",
        "semantic_only_ppr": "Semantic-Only PPR (Ablation)",
        "codec_only_ppr": "Codec-Only PPR (Ablation)",
        "hybrid_ppr": "Hybrid PPR (Ablation)",
        "random": "Random (Mean across 30 seeds)"
    }
    
    raw_summary = summary.copy()
    summary["method_name"] = summary["method_name"].map(rename_map)
    out_dir = config["out_dir"]
    summary.to_csv(os.path.join(out_dir, "per_method_summary.csv"), index=False)
    
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("# Publication-Ready Benchmark Summary Report\n\n")
        f.write("## Primary Endpoint: WindowCoverage@K at 10% retention\n\n")
        f.write("| Method | Window Coverage | 95% Confidence Interval | Recall@5 | Recall@10 |\n")
        f.write("|---|---|---|---|---|\n")
        for idx, row in summary.iterrows():
            f.write(f"| {row['method_name']} | {row['window_coverage']:.4f} | [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | {row['recall_at_5']:.4f} | {row['recall_at_10']:.4f} |\n")
            
    import shutil
    shutil.copy(os.path.join(out_dir, "summary.md"), os.path.join(out_dir, "summary_report.md"))
            
    with open(os.path.join(out_dir, "tables", "primary_results.tex"), "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Method & Window Coverage & 95\\% CI & Recall@5 & Recall@10 \\\\\n\\hline\n")
        for idx, row in summary.iterrows():
            f.write(f"{row['method_name']} & {row['window_coverage']:.4f} & [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] & {row['recall_at_5']:.4f} & {row['recall_at_10']:.4f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\caption{Matched-Budget Selection and Retrieval performance comparison.}\n\\label{tab:primary_results}\n\\end{table}\n")

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    summary_plot = summary.dropna(subset=["method_name"])
    plt.barh(summary_plot["method_name"], summary_plot["window_coverage"], xerr=[summary_plot["window_coverage"] - summary_plot["ci_lower"], summary_plot["ci_upper"] - summary_plot["window_coverage"]], capsize=5, color="skyblue")
    plt.xlabel("Window Coverage @ 10% Budget")
    plt.title("Matched-Budget Selection Window Coverage comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figures", "window_coverage.png"))
    plt.savefig(os.path.join(out_dir, "figures", "window_coverage.pdf"))
    plt.savefig(os.path.join(out_dir, "window_coverage.png"))
    plt.savefig(os.path.join(out_dir, "window_coverage.pdf"))
    
    baselines_to_check = [
        "uniform", "iframe_prior", "scene_change", "luma_difference", "optical_flow_farneback",
        "clip_kmeans_diversity", "packet_size_only", "random"
    ]
    raw_bas_sum = raw_summary[raw_summary["method_name"].isin(baselines_to_check)]
    
    strongest_baseline_name = "random"
    strongest_baseline_cov = 0.0
    if not raw_bas_sum.empty:
        idx_max = raw_bas_sum["window_coverage"].idxmax()
        strongest_baseline_name = raw_bas_sum.loc[idx_max, "method_name"]
        strongest_baseline_cov = raw_bas_sum.loc[idx_max, "window_coverage"]
        
    proposed_cov = raw_summary[raw_summary["method_name"] == "proposed_model_selection"]["window_coverage"].values[0]
    improvement = proposed_cov - strongest_baseline_cov
    
    boot_diffs = []
    for stat in boot_stats:
        prop = stat.get("proposed_model_selection", 0.0)
        base = stat.get(strongest_baseline_name, 0.0)
        boot_diffs.append(prop - base)
        
    boot_diff_lower = np.quantile(boot_diffs, 0.025)
    boot_diff_upper = np.quantile(boot_diffs, 0.975)
    
    exclude_zero = (boot_diff_lower > 0) or (boot_diff_upper < 0)
    passes_gate = exclude_zero and (improvement >= 0.03)
    
    parent_runs_dir = os.path.dirname(out_dir)
    
    claims = {
        "dataset": "nextqa",
        "primary_metric": "WindowCoverage@K",
        "proposed_model_coverage": float(proposed_cov),
        "strongest_baseline_name": rename_map.get(strongest_baseline_name, strongest_baseline_name),
        "strongest_baseline_coverage": float(strongest_baseline_cov),
        "absolute_improvement": float(improvement),
        "difference_95_ci": [float(boot_diff_lower), float(boot_diff_upper)],
        "exclude_zero_ci": bool(exclude_zero),
        "passes_claim_gate": bool(passes_gate)
    }
    with open(os.path.join(parent_runs_dir, "claims.json"), "w") as f:
        json.dump(claims, f, indent=2)
        
    status_data = {
        "status": "COMPLETED",
        "last_update": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "nextqa",
        "videos_evaluated": len(unique_vids),
        "replicates_run": replicates
    }
    with open(os.path.join(parent_runs_dir, "LATEST_STATUS.json"), "w") as f:
        json.dump(status_data, f, indent=2)
        
    with open(os.path.join(parent_runs_dir, "CURRENT_DATASET.json"), "w") as f:
        json.dump({"active_dataset": "nextqa", "stage": "STATISTICS", "status": "COMPLETE"}, f, indent=2)
        
    with open(os.path.join(parent_runs_dir, "RESULTS_INDEX.md"), "w") as f:
        f.write("# Publication-Ready Benchmark Results Index\n\n")
        f.write("This is the top-level index for the paper-grade evaluation run.\n\n")
        f.write("## Datasets State\n\n")
        f.write("| Dataset | Evaluated Videos | Primary Endpoint (Proposed vs Strongest Baseline) | Claim Gate Status |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| NExT-GQA | {len(unique_vids)} | {proposed_cov:.4f} vs {strongest_baseline_cov:.4f} | {'PASSED' if passes_gate else 'FAILED/MIXED'} |\n")
        f.write("\n## Folders & Artifacts\n\n")
        f.write(f"- [NExT-GQA Results Bundle Folder](file://{out_dir})\n")
        f.write(f"- [LaTeX Tables](file://{os.path.join(out_dir, 'tables')})\n")
        f.write(f"- [Generated Figures](file://{os.path.join(out_dir, 'figures')})\n")
        
    with open(os.path.join(parent_runs_dir, "RESULTS_INDEX.html"), "w") as f:
        f.write(f"<html><body><h1>Results Index</h1><p>NExT-GQA status: {'PASSED' if passes_gate else 'FAILED/MIXED'}</p></body></html>")
        
    artifacts_to_log = [
        ("nextqa/per_method_summary.csv", "CSV Table", "Summary of metrics per selection/retrieval method"),
        ("nextqa/summary.md", "Markdown Report", "Human-readable summary report"),
        ("nextqa/tables/primary_results.tex", "LaTeX Table", "LaTeX formatted results table for LaTeX draft"),
        ("nextqa/figures/window_coverage.png", "PNG Figure", "Plot of selection window coverage"),
        ("nextqa/figures/window_coverage.pdf", "PDF Figure", "Vector PDF of selection window coverage plot"),
        ("claims.json", "JSON Data", "Claim gate verification metrics"),
        ("RESULTS_INDEX.md", "Markdown Report", "Main benchmark entrypoint index")
    ]
    
    manifest_rows = []
    for rel_path, art_type, desc in artifacts_to_log:
        full_p = os.path.join(parent_runs_dir, rel_path)
        if os.path.exists(full_p):
            size = os.path.getsize(full_p)
            sha = get_sha256(full_p)
            manifest_rows.append({
                "Relative Path": rel_path,
                "Artifact Type": art_type,
                "Dataset": "nextqa" if "nextqa" in rel_path else "all",
                "Description": desc,
                "File Size (Bytes)": size,
                "SHA-256": sha
            })
            
    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(os.path.join(parent_runs_dir, "artifact_manifest.csv"), index=False)
    
if __name__ == "__main__":
    main()
