import os
import sys
import time
import datetime
import yaml
import json
import argparse
import psutil
import pandas as pd
import numpy as np

# Force CPU-only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import iris.charon_v as charon_v
from iris.action_score import ActionScoreConfig, ActionScoreModule
from iris.iris_config import IRISConfig
from iris.pipeline import wrapper_l2_retrieve
import iris.pipeline
iris.pipeline.get_semantic_and_clip_caption = lambda pil_img, frame, clip_emb, device: {
    "clip_label": "mock_label",
    "semantic_caption": "mock_caption"
}

import benchmarks.exp1a_baselines as baselines
import benchmarks.exp1a_metrics as metrics
from benchmarks.exp1a_report import generate_exp1a_report

def get_peak_ram():
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        peak = getattr(mem, 'peak_wset', getattr(mem, 'rss', 0))
        return peak / (1024 * 1024)
    except Exception:
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Benchmark 1A: Budget-Matched Frame Selection baseline comparison.")
    parser.add_argument("--dataset_csv", type=str, required=True, help="Path to prepared dataset CSV")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--out", type=str, default="benchmark_results/exp1a_budget_matched", help="Output directory")
    parser.add_argument("--cpu-only", action="store_true", default=True, help="Force CPU-only execution")
    args = parser.parse_args()

    # Handle folder exists logic: never silently overwrite, create timestamped subfolder
    if os.path.exists(args.out):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"{args.out}_{timestamp}"
        print(f"Output directory {args.out} exists. Creating timestamped folder: {out_dir}")
    else:
        out_dir = args.out
        
    os.makedirs(out_dir, exist_ok=True)

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config: {config}")

    iris_sel = config.get("iris_selection", {})
    action_thresh = iris_sel.get("action_score_threshold", 0.4)
    pers_thresh = iris_sel.get("persistence_threshold", 0.4)
    top_ks = config.get("top_k", [1, 5, 10])
    random_seeds = config.get("random_seeds", list(range(10)))
    span_buckets = config.get("span_buckets", {"short_max_seconds": 2.0, "medium_max_seconds": 5.0})

    # Load dataset
    if not os.path.exists(args.dataset_csv):
        print(f"Dataset CSV not found at {args.dataset_csv}")
        sys.exit(1)
        
    df_data = pd.read_csv(args.dataset_csv)
    print(f"Loaded dataset with {len(df_data)} event rows.")

    video_groups = df_data.groupby("video_id")
    
    unavailable_methods = {}
    per_video_records = []
    per_event_records = []
    random_sampling_runs = []
    
    # Check if cv2 and CLIP are available
    if baselines.cv2 is None:
        unavailable_methods["optical_flow_sampling"] = "cv2 (OpenCV) not installed"
    if baselines.clip is None:
        unavailable_methods["clip_clustering"] = "CLIP, PyTorch or scikit-learn not installed"

    for video_id, group in video_groups:
        video_path = group.iloc[0]["path"]
        if not os.path.exists(video_path):
            print(f"Warning: Video file for {video_id} not found at {video_path}. Skipping.")
            continue
            
        print(f"Processing video {video_id}...")
        
        # 1. Run IRIS selection first to determine K
        t_parse_start = time.perf_counter()
        try:
            output_frames, stats, raw_records = charon_v.parse_video(
                video_path,
                return_stats=True,
                return_raw=True,
                adaptive=True
            )
        except Exception as e:
            print(f"Failed to parse video {video_id}: {e}")
            continue
            
        t_parse = time.perf_counter() - t_parse_start
        total_frames = stats["total"]

        t_action_start = time.perf_counter()
        score_config = ActionScoreConfig(
            persistence_threshold=pers_thresh
        )
        scorer = ActionScoreModule(score_config)
        feature_dicts = []
        for rec in raw_records:
            feature_dicts.append({
                "frame_idx": rec["frame_idx"],
                "packet_size": rec.get("packet_size", 0.0),
                "motion_magnitude": rec.get("motion_magnitude", 0.0),
                "luma_entropy": rec.get("luma_entropy", 0.0)
            })
        scored_records = scorer.score_all(feature_dicts)
        t_action = time.perf_counter() - t_action_start

        # Map scores and timestamps
        action_map = {r["frame_idx"]: r for r in scored_records}
        for rec in raw_records:
            fidx = rec["frame_idx"]
            if fidx in action_map:
                action_map[fidx]["timestamp"] = rec["timestamp"]
                action_map[fidx]["luma_diff_energy"] = rec.get("luma_diff_energy", 0.0)
                action_map[fidx]["frame_type"] = rec.get("frame_type", "P")

        # Determine K frames selected by IRIS using native adaptive Charon-V tiers
        selected_idxs = sorted([f["frame_idx"] for f in output_frames])
        budget_k = len(selected_idxs)
        if budget_k == 0:
            print(f"Warning: IRIS selected 0 frames for {video_id}. Forcing K = 1.")
            budget_k = 1
            selected_idxs = [0]
            
        iris_selected_frames = [action_map[idx] for idx in selected_idxs if idx in action_map]

        print(f"Budget K = {budget_k} (out of {total_frames} frames)")
        
        # --- RUN ALL BASELINE SELECTION METHODS ---
        selections = {}
        
        # 1. iris_action_score
        t_sel_iris = time.perf_counter()
        iris_selected_sorted = sorted(
            iris_selected_frames,
            key=lambda x: x["action_score"] + x["persistence_value"],
            reverse=True
        )
        selected_idxs = sorted([f["frame_idx"] for f in iris_selected_sorted])
        selected_ts = [action_map[idx]["timestamp"] for idx in selected_idxs]
        ranked_ts = [f["timestamp"] for f in iris_selected_sorted]
        
        selections["iris_action_score"] = {
            "method_name": "iris_action_score",
            "selected_frames": selected_idxs,
            "selected_timestamps": selected_ts,
            "ranked_timestamps": ranked_ts,
            "frame_metadata": [action_map[idx] for idx in selected_idxs],
            "budget_k": budget_k,
            "fill_count": 0,
            "unavailable_reason": None,
            "selection_time_seconds": t_action + (time.perf_counter() - t_sel_iris),
            "total_method_time_seconds": t_parse + t_action + (time.perf_counter() - t_sel_iris)
        }

        # 2. uniform_budget_matched
        t_start = time.perf_counter()
        res_uniform = baselines.uniform_budget_matched(raw_records, budget_k)
        res_uniform["total_method_time_seconds"] = t_parse + (time.perf_counter() - t_start)
        selections["uniform_budget_matched"] = res_uniform

        # 3. random_budget_matched (10 seeds)
        for seed in random_seeds:
            t_start = time.perf_counter()
            res_random = baselines.random_budget_matched(raw_records, budget_k, seed=seed)
            res_random["total_method_time_seconds"] = t_parse + (time.perf_counter() - t_start)
            selections[f"random_budget_matched_seed_{seed}"] = res_random

        # 4. iframe_only_budget_matched
        t_start = time.perf_counter()
        res_iframe = baselines.iframe_only_budget_matched(raw_records, budget_k)
        res_iframe["total_method_time_seconds"] = t_parse + (time.perf_counter() - t_start)
        selections["iframe_only_budget_matched"] = res_iframe

        # 5. scene_change_detection
        t_start = time.perf_counter()
        res_scene = baselines.scene_change_detection(raw_records, budget_k)
        res_scene["total_method_time_seconds"] = t_parse + (time.perf_counter() - t_start)
        selections["scene_change_detection"] = res_scene

        # 6. optical_flow_sampling
        if "optical_flow_sampling" not in unavailable_methods:
            t_start = time.perf_counter()
            res_flow = baselines.optical_flow_sampling(video_path, raw_records, budget_k)
            if res_flow["unavailable_reason"]:
                unavailable_methods["optical_flow_sampling"] = res_flow["unavailable_reason"]
            else:
                res_flow["total_method_time_seconds"] = (time.perf_counter() - t_start) # Includes decode inside flow
                selections["optical_flow_sampling"] = res_flow

        # 7. clip_clustering
        if "clip_clustering" not in unavailable_methods:
            t_start = time.perf_counter()
            res_clip = baselines.clip_clustering(video_path, raw_records, budget_k)
            if res_clip["unavailable_reason"]:
                unavailable_methods["clip_clustering"] = res_clip["unavailable_reason"]
            else:
                res_clip["total_method_time_seconds"] = (time.perf_counter() - t_start)
                selections["clip_clustering"] = res_clip

        # 8. luma_diff_topk
        t_start = time.perf_counter()
        res_luma = baselines.luma_diff_topk(raw_records, budget_k)
        res_luma["total_method_time_seconds"] = t_parse + (time.perf_counter() - t_start)
        selections["luma_diff_topk"] = res_luma

        # 9. graphless_iris
        t_start = time.perf_counter()
        selections["graphless_iris"] = {
            "method_name": "graphless_iris",
            "selected_frames": selected_idxs,
            "selected_timestamps": selected_ts,
            "ranked_timestamps": ranked_ts,
            "frame_metadata": [action_map[idx] for idx in selected_idxs],
            "budget_k": budget_k,
            "fill_count": 0,
            "unavailable_reason": None,
            "selection_time_seconds": t_action + (time.perf_counter() - t_start),
            "total_method_time_seconds": t_parse + t_action + (time.perf_counter() - t_start)
        }

        # 10. full_iris (Evaluated per-event query)
        # Note: full_iris is run per event query inside the event loop since graph rankings are query-specific.
        
        # Log per video results for all completed methods (excluding seeds, we average random)
        methods_to_log = [
            "iris_action_score", "uniform_budget_matched", "iframe_only_budget_matched",
            "scene_change_detection", "luma_diff_topk", "graphless_iris"
        ]
        if "optical_flow_sampling" in selections:
            methods_to_log.append("optical_flow_sampling")
        if "clip_clustering" in selections:
            methods_to_log.append("clip_clustering")

        for m_name in methods_to_log:
            sel = selections[m_name]
            ret_ratio = len(sel["selected_frames"]) / total_frames
            per_video_records.append({
                "video_id": video_id,
                "method_name": m_name,
                "total_frames": total_frames,
                "selected_frames": len(sel["selected_frames"]),
                "budget_k": budget_k,
                "frame_retention_ratio": ret_ratio,
                "frame_compression_rate": 1.0 - ret_ratio,
                "fill_count": sel["fill_count"],
                "selection_time_seconds": sel["selection_time_seconds"],
                "total_method_time_seconds": sel["total_method_time_seconds"]
            })

        # Average random sampling across 10 seeds for per_video_results
        rand_selected_frames = []
        rand_fill = []
        rand_sel_time = []
        rand_tot_time = []
        for seed in random_seeds:
            sel = selections[f"random_budget_matched_seed_{seed}"]
            rand_selected_frames.append(len(sel["selected_frames"]))
            rand_fill.append(sel["fill_count"])
            rand_sel_time.append(sel["selection_time_seconds"])
            rand_tot_time.append(sel["total_method_time_seconds"])

        rand_ret_ratio = np.mean(rand_selected_frames) / total_frames
        per_video_records.append({
            "video_id": video_id,
            "method_name": "random_budget_matched",
            "total_frames": total_frames,
            "selected_frames": np.mean(rand_selected_frames),
            "budget_k": budget_k,
            "frame_retention_ratio": rand_ret_ratio,
            "frame_compression_rate": 1.0 - rand_ret_ratio,
            "fill_count": np.mean(rand_fill),
            "selection_time_seconds": np.mean(rand_sel_time),
            "total_method_time_seconds": np.mean(rand_tot_time)
        })

        # --- EVENT LEVEL EVALUATION ---
        for idx, event_row in group.iterrows():
            start_time = event_row["start_time"]
            end_time = event_row["end_time"]
            query = event_row["query"]
            event_label = event_row["event_label"]
            
            # Run full_iris per event query
            t_full_iris_start = time.perf_counter()
            full_iris_retrieved_ts = []
            try:
                # We reuse the iris_action_score selected frames (frame_metadata) to build the index
                # Convert frame_metadata back to format expected by wrapper_l2_retrieve
                frames_to_index = []
                for f in selections["iris_action_score"]["frame_metadata"]:
                    frames_to_index.append({
                        "frame_idx": f["frame_idx"],
                        "timestamp": f["timestamp"],
                        "luma_diff_energy": f.get("luma_diff_energy", 0.0),
                        "action_score": f.get("action_score", 0.0),
                        "persistence_value": f.get("persistence_value", 0.0),
                    })
                # Build configuration
                iris_cfg = IRISConfig(l2_retrieve_top_k=budget_k)
                retrieved_nodes = wrapper_l2_retrieve(video_path, query, frames_to_index, config=iris_cfg)
                full_iris_retrieved_ts = [r["timestamp"] for r in retrieved_nodes]
            except Exception as e:
                print(f"Full IRIS graph retrieval failed for {video_id} (query: {query}): {e}")
                
            t_full_iris = time.perf_counter() - t_full_iris_start
            
            # Fill or clip retrieved list to exactly K
            if len(full_iris_retrieved_ts) < budget_k:
                full_iris_retrieved_ts.extend(selections["iris_action_score"]["ranked_timestamps"][:budget_k - len(full_iris_retrieved_ts)])
            full_iris_retrieved_ts = full_iris_retrieved_ts[:budget_k]
            
            selections["full_iris"] = {
                "method_name": "full_iris",
                "selected_frames": selections["iris_action_score"]["selected_frames"],
                "selected_timestamps": selections["iris_action_score"]["selected_timestamps"],
                "ranked_timestamps": full_iris_retrieved_ts,
                "frame_metadata": selections["iris_action_score"]["frame_metadata"],
                "budget_k": budget_k,
                "fill_count": 0,
                "unavailable_reason": None,
                "selection_time_seconds": t_action + t_full_iris,
                "total_method_time_seconds": t_parse + t_action + t_full_iris
            }
            
            # Log full_iris video stats once per event query
            # We add a per-video row for full_iris corresponding to this event query, or handle it downstream
            # Let's add it to per_video_records
            full_ret_ratio = len(selections["full_iris"]["selected_frames"]) / total_frames
            per_video_records.append({
                "video_id": video_id,
                "method_name": f"full_iris_{event_label[:10]}", # query-specific row
                "total_frames": total_frames,
                "selected_frames": len(selections["full_iris"]["selected_frames"]),
                "budget_k": budget_k,
                "frame_retention_ratio": full_ret_ratio,
                "frame_compression_rate": 1.0 - full_ret_ratio,
                "fill_count": 0,
                "selection_time_seconds": selections["full_iris"]["selection_time_seconds"],
                "total_method_time_seconds": selections["full_iris"]["total_method_time_seconds"]
            })

            # Evaluate each method
            active_methods = list(methods_to_log) + ["full_iris"]
            for m_name in active_methods:
                sel = selections[m_name]
                cov_res = metrics.evaluate_selection_coverage(sel["selected_timestamps"], start_time, end_time)
                
                for k in top_ks:
                    ret_res = metrics.evaluate_ranked_retrieval(sel["ranked_timestamps"], start_time, end_time, k)
                    
                    per_event_records.append({
                        "video_id": video_id,
                        "query": query,
                        "event_label": event_label,
                        "start_time": start_time,
                        "end_time": end_time,
                        "method_name": m_name,
                        "top_k": k,
                        "selected_timestamps": ",".join([f"{t:.2f}" for t in sel["ranked_timestamps"][:k]]),
                        "event_hit_any_selected": cov_res["event_hit_any_selected"],
                        "min_temporal_distance_seconds": cov_res["min_temporal_distance_seconds"],
                        "best_temporal_iou_proxy": cov_res["best_temporal_iou_proxy"],
                        "zero_overlap": cov_res["zero_overlap"],
                        "recall": ret_res["recall"],
                        "temporal_hit": ret_res["temporal_hit"],
                        "mean_temporal_distance_seconds": ret_res["mean_temporal_distance_seconds"],
                        "temporal_iou_proxy": ret_res["temporal_iou_proxy"],
                        "mrr": ret_res["mrr"]
                    })

            # Evaluate random seeds separately for random_sampling_runs.csv
            for seed in random_seeds:
                sel = selections[f"random_budget_matched_seed_{seed}"]
                for k in top_ks:
                    ret_res = metrics.evaluate_ranked_retrieval(sel["ranked_timestamps"], start_time, end_time, k)
                    random_sampling_runs.append({
                        "seed": seed,
                        "video_id": video_id,
                        "query": query,
                        "event_label": event_label,
                        "start_time": start_time,
                        "end_time": end_time,
                        "top_k": k,
                        "recall": ret_res["recall"],
                        "temporal_hit": ret_res["temporal_hit"],
                        "mean_temporal_distance_seconds": ret_res["mean_temporal_distance_seconds"],
                        "temporal_iou_proxy": ret_res["temporal_iou_proxy"],
                        "mrr": ret_res["mrr"]
                    })
                    
            # Compute average random metrics for per_event_results
            for k in top_ks:
                seed_recalls = []
                seed_hits = []
                seed_distances = []
                seed_ious = []
                seed_mrrs = []
                
                # Retrieve stats for this event and top_k across seeds
                for seed in random_seeds:
                    sel = selections[f"random_budget_matched_seed_{seed}"]
                    cov_res = metrics.evaluate_selection_coverage(sel["selected_timestamps"], start_time, end_time)
                    ret_res = metrics.evaluate_ranked_retrieval(sel["ranked_timestamps"], start_time, end_time, k)
                    
                    seed_recalls.append(ret_res["recall"])
                    seed_hits.append(ret_res["temporal_hit"])
                    seed_distances.append(ret_res["mean_temporal_distance_seconds"])
                    seed_ious.append(ret_res["temporal_iou_proxy"])
                    seed_mrrs.append(ret_res["mrr"])
                    
                cov_res_seed = metrics.evaluate_selection_coverage(selections[f"random_budget_matched_seed_0"]["selected_timestamps"], start_time, end_time)
                
                per_event_records.append({
                    "video_id": video_id,
                    "query": query,
                    "event_label": event_label,
                    "start_time": start_time,
                    "end_time": end_time,
                    "method_name": "random_budget_matched",
                    "top_k": k,
                    "selected_timestamps": "averaged",
                    "event_hit_any_selected": cov_res_seed["event_hit_any_selected"], # representative
                    "min_temporal_distance_seconds": cov_res_seed["min_temporal_distance_seconds"],
                    "best_temporal_iou_proxy": cov_res_seed["best_temporal_iou_proxy"],
                    "zero_overlap": cov_res_seed["zero_overlap"],
                    "recall": np.mean(seed_recalls),
                    "temporal_hit": np.mean(seed_hits),
                    "mean_temporal_distance_seconds": np.mean(seed_distances),
                    "temporal_iou_proxy": np.mean(seed_ious),
                    "mrr": np.mean(seed_mrrs)
                })

    # Save outputs
    df_video_out = pd.DataFrame(per_video_records)
    # Filter query-specific full_iris rows from output video CSV to keep it clean (or average full_iris rows)
    # Let's map full_iris rows back to a single method 'full_iris'
    # and average their selection_time / total_method_time
    full_iris_mask = df_video_out["method_name"].str.startswith("full_iris_")
    full_iris_df = df_video_out[full_iris_mask]
    
    clean_video_records = df_video_out[~full_iris_mask].to_dict(orient="records")
    if not full_iris_df.empty:
        # Group by video_id
        for vid, sub in full_iris_df.groupby("video_id"):
            clean_video_records.append({
                "video_id": vid,
                "method_name": "full_iris",
                "total_frames": sub.iloc[0]["total_frames"],
                "selected_frames": sub.iloc[0]["selected_frames"],
                "budget_k": sub.iloc[0]["budget_k"],
                "frame_retention_ratio": sub.iloc[0]["frame_retention_ratio"],
                "frame_compression_rate": sub.iloc[0]["frame_compression_rate"],
                "fill_count": 0,
                "selection_time_seconds": sub["selection_time_seconds"].mean(),
                "total_method_time_seconds": sub["total_method_time_seconds"].mean()
            })
            
    df_video_clean = pd.DataFrame(clean_video_records)
    per_video_csv_path = os.path.join(out_dir, "per_video_results.csv")
    df_video_clean.to_csv(per_video_csv_path, index=False)
    print(f"Per-video results saved to {per_video_csv_path}")

    df_event_out = pd.DataFrame(per_event_records)
    per_event_csv_path = os.path.join(out_dir, "per_event_results.csv")
    df_event_out.to_csv(per_event_csv_path, index=False)
    print(f"Per-event results saved to {per_event_csv_path}")

    df_random_out = pd.DataFrame(random_sampling_runs)
    random_runs_csv_path = os.path.join(out_dir, "random_sampling_runs.csv")
    df_random_out.to_csv(random_runs_csv_path, index=False)
    print(f"Random runs saved to {random_runs_csv_path}")

    # Write unavailable methods JSON
    unavailable_path = os.path.join(out_dir, "unavailable_methods.json")
    with open(unavailable_path, "w") as f:
        json.dump(unavailable_methods, f, indent=2)
    print(f"Unavailable methods logged to {unavailable_path}")

    # ── GENERATE METRIC SUMMARIES ──────────────────────────────────────────
    # Group by method to generate summary metrics
    summary_list = []
    
    unique_methods = df_event_out["method_name"].unique()
    for m in unique_methods:
        sub_evt_k5 = df_event_out[(df_event_out["method_name"] == m) & (df_event_out["top_k"] == 5)]
        sub_evt_k10 = df_event_out[(df_event_out["method_name"] == m) & (df_event_out["top_k"] == 10)]
        sub_vid = df_video_clean[df_video_clean["method_name"] == m]
        
        summary_list.append({
            "method_name": m,
            "avg_selected_frames": float(sub_vid["selected_frames"].mean()) if not sub_vid.empty else 0.0,
            "avg_retention_ratio": float(sub_vid["frame_retention_ratio"].mean()) if not sub_vid.empty else 0.0,
            "avg_compression_rate": float(sub_vid["frame_compression_rate"].mean()) if not sub_vid.empty else 0.0,
            "event_hit_any_selected": float(sub_evt_k10["event_hit_any_selected"].mean()) if not sub_evt_k10.empty else 0.0,
            "recall_at_5": float(sub_evt_k5["recall"].mean()) if not sub_evt_k5.empty else 0.0,
            "recall_at_10": float(sub_evt_k10["recall"].mean()) if not sub_evt_k10.empty else 0.0,
            "mrr": float(sub_evt_k10["mrr"].mean()) if not sub_evt_k10.empty else 0.0,
            "temporal_iou_proxy": float(sub_evt_k5["temporal_iou_proxy"].mean()) if not sub_evt_k5.empty else 0.0,
            "selection_time_seconds": float(sub_vid["selection_time_seconds"].mean()) if not sub_vid.empty else 0.0,
            "total_method_time_seconds": float(sub_vid["total_method_time_seconds"].mean()) if not sub_vid.empty else 0.0
        })
        
    df_summary = pd.DataFrame(summary_list)
    per_method_summary_path = os.path.join(out_dir, "per_method_summary.csv")
    df_summary.to_csv(per_method_summary_path, index=False)
    print(f"Per-method summary CSV saved to {per_method_summary_path}")

    # Generate summary JSON
    peak_ram = get_peak_ram()
    summary_metrics = {
        "dataset": "ActivityNet Captions subset",
        "videos_processed": len(video_groups),
        "skipped_videos_count": len(df_data["video_id"].unique()) - len(video_groups),
        "peak_ram_mb": peak_ram,
        "unavailable_methods": unavailable_methods,
        "overall_summary": summary_list
    }
    
    summary_json_path = os.path.join(out_dir, "summary_metrics.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"Summary metrics JSON saved to {summary_json_path}")

    # Compile the final report and plots
    generate_exp1a_report(
        out_dir,
        per_video_csv_path,
        per_event_csv_path,
        summary_json_path,
        random_runs_csv_path
    )
    
    print(f"Benchmark 1A execution finished successfully. Output in: {out_dir}")

if __name__ == "__main__":
    main()
