import os
import sys
import time
import yaml
import json
import argparse
import psutil
import pandas as pd
import numpy as np

# Force CPU-only as early as possible
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import iris.charon_v as charon_v
from iris.action_score import ActionScoreConfig, ActionScoreModule
from benchmarks.exp0_metrics import (
    compute_compression_rate,
    evaluate_event_retrieval
)
from benchmarks.exp0_report import generate_report_and_plots

def get_peak_ram():
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        # On Windows peak_wset represents peak working set size. Fall back to RSS.
        peak = getattr(mem, 'peak_wset', getattr(mem, 'rss', 0))
        return peak / (1024 * 1024)
    except Exception:
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Experiment 0: Compression vs Accuracy Grid Sweep.")
    parser.add_argument("--dataset_csv", type=str, required=True, help="Path to prepared dataset CSV")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--out", type=str, default="benchmark_results/exp0_compression_accuracy", help="Output directory")
    parser.add_argument("--cpu-only", action="store_true", default=True, help="Force CPU-only execution")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config: {config}")

    # Set up threshold grid and params
    grid_cfg = config.get("threshold_grid", {})
    action_thresholds = grid_cfg.get("action_score_threshold", [0.2, 0.4, 0.6])
    persistence_thresholds = grid_cfg.get("persistence_threshold", [0.2, 0.4, 0.6])
    top_ks = config.get("top_k", [1, 5, 10])
    
    # Load dataset
    if not os.path.exists(args.dataset_csv):
        print(f"Dataset CSV not found at {args.dataset_csv}")
        sys.exit(1)
        
    df = pd.read_csv(args.dataset_csv)
    print(f"Dataset contains {len(df)} event rows across {df['video_id'].nunique()} unique videos.")

    video_groups = df.groupby("video_id")
    
    # Runtimes tracking
    total_parse_time = 0.0
    total_action_time = 0.0
    total_selection_time = 0.0
    total_videos_processed = 0

    per_video_records = []
    per_event_records = []

    # Process video by video
    for video_id, group in video_groups:
        video_path = group.iloc[0]["path"]
        if not os.path.exists(video_path):
            print(f"Warning: Video file for {video_id} not found at {video_path}. Skipping.")
            continue
            
        print(f"Processing video {video_id}...")
        
        # 1. Video Parsing (Charon-V)
        t0 = time.perf_counter()
        try:
            # We run with adaptive=True since it's the standard path in Charon-V
            output_frames, stats, raw_records = charon_v.parse_video(
                video_path,
                return_stats=True,
                return_raw=True,
                adaptive=True
            )
        except Exception as e:
            print(f"Failed to parse video {video_id}: {e}")
            continue
            
        t_parse = time.perf_counter() - t0
        total_parse_time += t_parse
        total_frames = stats["total"]
        
        # 2. Action Score Scoring
        t1 = time.perf_counter()
        score_config = ActionScoreConfig()
        scorer = ActionScoreModule(score_config)
        
        # Prepare feature dicts
        feature_dicts = []
        for rec in raw_records:
            feature_dicts.append({
                "frame_idx": rec["frame_idx"],
                "packet_size": rec.get("packet_size", 0.0),
                "motion_magnitude": rec.get("motion_magnitude", 0.0),
                "luma_entropy": rec.get("luma_entropy", 0.0)
            })
            
        scored_records = scorer.score_all(feature_dicts)
        t_action = time.perf_counter() - t1
        total_action_time += t_action

        # Map scores and timestamps
        frame_map = {r["frame_idx"]: r for r in scored_records}
        for rec in raw_records:
            fidx = rec["frame_idx"]
            if fidx in frame_map:
                frame_map[fidx]["timestamp"] = rec["timestamp"]

        # 3. Sweep Threshold Grid
        t2 = time.perf_counter()
        for act_thresh in action_thresholds:
            for pers_thresh in persistence_thresholds:
                # Select frames meeting thresholds
                selected_frames = []
                for fidx, f in frame_map.items():
                    act_score = f["action_score"]
                    pers_val = f["persistence_value"]
                    if act_score >= act_thresh or pers_val >= pers_thresh:
                        selected_frames.append(f)
                
                num_selected = len(selected_frames)
                ret_ratio, comp_rate = compute_compression_rate(total_frames, num_selected)
                
                # Rank selected frames by (action_score + persistence_value) descending
                # Stable sort in Python maintains index order as tie breaker
                selected_frames_sorted = sorted(
                    selected_frames,
                    key=lambda x: x["action_score"] + x["persistence_value"],
                    reverse=True
                )
                
                selected_timestamps = [f["timestamp"] for f in selected_frames_sorted]
                
                # Write per video record
                # We attribute runtimes proportionally or record them for this threshold configuration run
                # Selection time is computed per sweep
                t_select = time.perf_counter() - t2
                total_select_for_video = t_select
                
                per_video_records.append({
                    "video_id": video_id,
                    "threshold_action": act_thresh,
                    "threshold_persistence": pers_thresh,
                    "total_frames": total_frames,
                    "selected_frames": num_selected,
                    "frame_retention_ratio": ret_ratio,
                    "frame_compression_rate": comp_rate,
                    "video_parse_time_seconds": t_parse,
                    "action_score_time_seconds": t_action,
                    "selection_time_seconds": total_select_for_video,
                    "total_video_time_seconds": t_parse + t_action + total_select_for_video
                })
                
                # Evaluate against each event query in the group
                for idx, event_row in group.iterrows():
                    start_time = event_row["start_time"]
                    end_time = event_row["end_time"]
                    query = event_row["query"]
                    event_label = event_row["event_label"]
                    
                    for k in top_ks:
                        eval_res = evaluate_event_retrieval(selected_timestamps, start_time, end_time, k)
                        
                        per_event_records.append({
                            "video_id": video_id,
                            "query": query,
                            "event_label": event_label,
                            "start_time": start_time,
                            "end_time": end_time,
                            "threshold_action": act_thresh,
                            "threshold_persistence": pers_thresh,
                            "top_k": k,
                            "selected_timestamps": ",".join([f"{ts:.2f}" for ts in selected_timestamps[:k]]),
                            "temporal_hit": eval_res["temporal_hit"],
                            "recall": eval_res["recall"],
                            "mean_temporal_distance_seconds": eval_res["mean_temporal_distance_seconds"],
                            "temporal_iou_proxy": eval_res["temporal_iou_proxy"]
                        })
        
        total_videos_processed += 1

    # Check that we processed something
    if not per_video_records:
        print("No videos processed successfully.")
        sys.exit(1)

    # Save CSVs
    per_video_df = pd.DataFrame(per_video_records)
    per_video_csv_path = os.path.join(args.out, "per_video_threshold_results.csv")
    per_video_df.to_csv(per_video_csv_path, index=False)
    print(f"Per-video results saved to {per_video_csv_path}")

    per_event_df = pd.DataFrame(per_event_records)
    per_event_csv_path = os.path.join(args.out, "per_event_threshold_results.csv")
    per_event_df.to_csv(per_event_csv_path, index=False)
    print(f"Per-event results saved to {per_event_csv_path}")

    # Compute averages for summary JSON
    avg_parse_time = per_video_df["video_parse_time_seconds"].mean()
    avg_action_time = per_video_df["action_score_time_seconds"].mean()
    avg_select_time = per_video_df["selection_time_seconds"].mean()
    avg_total_time = per_video_df["total_video_time_seconds"].mean()
    peak_ram = get_peak_ram()

    # Identify dataset details
    total_downloaded = len(df["video_id"].unique())
    # Count event rows
    total_events = len(df)
    
    summary_metrics = {
        "dataset": "ActivityNet Captions subset",
        "videos_downloaded": total_downloaded,
        "videos_processed": total_videos_processed,
        "skipped_videos_count": total_downloaded - total_videos_processed,
        "total_event_spans": total_events,
        "peak_ram_mb": peak_ram,
        "average_runtimes_seconds": {
            "video_parse_time_seconds": float(avg_parse_time),
            "action_score_time_seconds": float(avg_action_time),
            "selection_time_seconds": float(avg_select_time),
            "total_video_time_seconds": float(avg_total_time)
        },
        "threshold_sweeps": []
    }

    # Aggregate threshold sweep results
    for act_thresh in action_thresholds:
        for pers_thresh in persistence_thresholds:
            sweep_video = per_video_df[
                (per_video_df["threshold_action"] == act_thresh) & 
                (per_video_df["threshold_persistence"] == pers_thresh)
            ]
            sweep_event = per_event_df[
                (per_event_df["threshold_action"] == act_thresh) & 
                (per_event_df["threshold_persistence"] == pers_thresh)
            ]
            
            sweep_data = {
                "threshold_action": act_thresh,
                "threshold_persistence": pers_thresh,
                "avg_selected_frames": float(sweep_video["selected_frames"].mean()),
                "avg_frame_retention_ratio": float(sweep_video["frame_retention_ratio"].mean()),
                "avg_frame_compression_rate": float(sweep_video["frame_compression_rate"].mean()),
                "top_k_metrics": {}
            }
            
            for k in top_ks:
                k_event = sweep_event[sweep_event["top_k"] == k]
                sweep_data["top_k_metrics"][str(k)] = {
                    "recall_at_k": float(k_event["recall"].mean()),
                    "temporal_hit_at_k": float(k_event["temporal_hit"].mean()),
                    "mean_temporal_distance_seconds": float(k_event["mean_temporal_distance_seconds"].mean()),
                    "temporal_iou_proxy": float(k_event["temporal_iou_proxy"].mean())
                }
                
            summary_metrics["threshold_sweeps"].append(sweep_data)

    summary_json_path = os.path.join(args.out, "summary_metrics.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"Summary metrics JSON saved to {summary_json_path}")

    # Generate the report and plots
    best_pair = generate_report_and_plots(
        args.out,
        per_video_csv_path,
        per_event_csv_path,
        summary_json_path
    )
    print(f"Experiment 0 run complete. Best tradeoff pair (Action, Persistence): {best_pair}")

if __name__ == "__main__":
    main()
