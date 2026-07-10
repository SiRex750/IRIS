import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_report_and_plots(out_dir, per_video_csv, per_event_csv, summary_json):
    # Load data
    df_video = pd.read_csv(per_video_csv)
    df_event = pd.read_csv(per_event_csv)
    with open(summary_json, "r") as f:
        summary_data = json.load(f)

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Pre-process for plots
    # We want to group by threshold pairs to get mean compression rate and mean metrics.
    # Group per_video by threshold_action and threshold_persistence
    group_vid = df_video.groupby(["threshold_action", "threshold_persistence"]).agg({
        "frame_compression_rate": "mean",
        "frame_retention_ratio": "mean",
        "selected_frames": "mean",
        "total_video_time_seconds": "mean"
    }).reset_index()

    # Group per_event by threshold_action, threshold_persistence, and top_k
    group_evt = df_event.groupby(["threshold_action", "threshold_persistence", "top_k"]).agg({
        "recall": "mean",
        "temporal_iou_proxy": "mean",
        "mean_temporal_distance_seconds": "mean"
    }).reset_index()

    # Merge to get compression vs accuracy
    df_merged = pd.merge(group_evt, group_vid, on=["threshold_action", "threshold_persistence"])

    # 1. compression_vs_recall_at_5.png
    plt.figure(figsize=(8, 6))
    df_k5 = df_merged[df_merged["top_k"] == 5]
    plt.scatter(df_k5["frame_compression_rate"], df_k5["recall"], color="blue", marker="o", s=80)
    for idx, row in df_k5.iterrows():
        plt.annotate(f"({row['threshold_action']},{row['threshold_persistence']})", 
                     (row["frame_compression_rate"], row["recall"]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    plt.title("Compression Rate vs Recall@5")
    plt.xlabel("Frame Compression Rate (1 - Retention)")
    plt.ylabel("Recall@5")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "compression_vs_recall_at_5.png"))
    plt.close()

    # 2. compression_vs_recall_at_10.png
    plt.figure(figsize=(8, 6))
    df_k10 = df_merged[df_merged["top_k"] == 10]
    plt.scatter(df_k10["frame_compression_rate"], df_k10["recall"], color="red", marker="s", s=80)
    for idx, row in df_k10.iterrows():
        plt.annotate(f"({row['threshold_action']},{row['threshold_persistence']})", 
                     (row["frame_compression_rate"], row["recall"]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    plt.title("Compression Rate vs Recall@10")
    plt.xlabel("Frame Compression Rate (1 - Retention)")
    plt.ylabel("Recall@10")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "compression_vs_recall_at_10.png"))
    plt.close()

    # 3. compression_vs_temporal_iou.png
    plt.figure(figsize=(8, 6))
    # We will use top_k = 5 for IoU proxy comparison, or average over top-ks. Let's use top_k = 5.
    df_iou = df_merged[df_merged["top_k"] == 5]
    plt.scatter(df_iou["frame_compression_rate"], df_iou["temporal_iou_proxy"], color="green", marker="^", s=80)
    for idx, row in df_iou.iterrows():
        plt.annotate(f"({row['threshold_action']},{row['threshold_persistence']})", 
                     (row["frame_compression_rate"], row["temporal_iou_proxy"]), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    plt.title("Compression Rate vs Temporal IoU Proxy (at K=5)")
    plt.xlabel("Frame Compression Rate (1 - Retention)")
    plt.ylabel("Temporal IoU Proxy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "compression_vs_temporal_iou.png"))
    plt.close()

    # 4. threshold_heatmap_recall_at_5.png
    # Create 2D grid for Recall@5
    action_thresholds = [float(x) for x in sorted(df_k5["threshold_action"].unique())]
    persistence_thresholds = [float(x) for x in sorted(df_k5["threshold_persistence"].unique())]
    
    grid = np.zeros((len(action_thresholds), len(persistence_thresholds)))
    for i, act in enumerate(action_thresholds):
        for j, pers in enumerate(persistence_thresholds):
            val = df_k5[(df_k5["threshold_action"] == act) & (df_k5["threshold_persistence"] == pers)]["recall"]
            grid[i, j] = val.values[0] if len(val) > 0 else 0.0

    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap="YlGnBu", origin="lower", aspect="auto")
    plt.colorbar(label="Recall@5")
    plt.xticks(range(len(persistence_thresholds)), persistence_thresholds)
    plt.yticks(range(len(action_thresholds)), action_thresholds)
    plt.xlabel("Persistence Threshold")
    plt.ylabel("Action Score Threshold")
    plt.title("Recall@5 Heatmap")
    
    # Annotate values
    for i in range(len(action_thresholds)):
        for j in range(len(persistence_thresholds)):
            plt.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="black" if grid[i, j] < 0.7 else "white")
            
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "threshold_heatmap_recall_at_5.png"))
    plt.close()

    # 5. runtime_vs_selected_frames.png
    plt.figure(figsize=(8, 6))
    # We will plot the raw per-video threshold runtime vs selected frames
    plt.scatter(df_video["selected_frames"], df_video["total_video_time_seconds"], color="purple", alpha=0.7, edgecolors="k")
    plt.title("Runtime vs Selected Frames (CPU)")
    plt.xlabel("Number of Selected Frames")
    plt.ylabel("Total CPU Processing Time (seconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "runtime_vs_selected_frames.png"))
    plt.close()

    # Determine best tradeoff (highest Recall@5 with highest compression)
    # We define best tradeoff as maximum Recall@5. If there's a tie, we pick the one with highest compression rate.
    best_row = None
    best_recall = -1.0
    best_comp = -1.0
    for idx, row in df_k5.iterrows():
        if row["recall"] > best_recall:
            best_recall = row["recall"]
            best_comp = row["frame_compression_rate"]
            best_row = row
        elif abs(row["recall"] - best_recall) < 1e-5:
            if row["frame_compression_rate"] > best_comp:
                best_comp = row["frame_compression_rate"]
                best_row = row

    best_threshold_pair = (float(best_row["threshold_action"]), float(best_row["threshold_persistence"])) if best_row is not None else (0.4, 0.4)

    # Identify failure cases: events where recall@10 is 0.0 (meaning no retrieved frame fell in gt span)
    failure_cases = df_event[(df_event["top_k"] == 10) & (df_event["temporal_hit"] == 0.0)]
    unique_failures = failure_cases[["video_id", "event_label", "start_time", "end_time", "threshold_action", "threshold_persistence"]].drop_duplicates().to_dict(orient="records")

    # Generate markdown report
    # We need to construct tables
    # Compression table
    comp_lines = [
        "| Action Thresh | Persistence Thresh | Avg Selected Frames | Avg Retention Ratio | Avg Compression Rate |",
        "|---|---|---|---|---|",
    ]
    for idx, row in group_vid.iterrows():
        comp_lines.append(f"| {row['threshold_action']:.2f} | {row['threshold_persistence']:.2f} | {row['selected_frames']:.2f} | {row['frame_retention_ratio']:.4f} | {row['frame_compression_rate']:.4f} |")

    # Accuracy table
    acc_lines = [
        "| Action Thresh | Persistence Thresh | K | Recall@K | Temporal IoU Proxy | Mean Distance (s) |",
        "|---|---|---|---|---|---|",
    ]
    # We sort group_evt by thresh and K
    group_evt_sorted = group_evt.sort_values(["threshold_action", "threshold_persistence", "top_k"])
    for idx, row in group_evt_sorted.iterrows():
        acc_lines.append(f"| {row['threshold_action']:.2f} | {row['threshold_persistence']:.2f} | {int(row['top_k'])} | {row['recall']:.4f} | {row['temporal_iou_proxy']:.4f} | {row['mean_temporal_distance_seconds']:.4f} |")

    # Runtime table
    # Stage runtimes
    runtime_summary = summary_data.get("average_runtimes_seconds", {})
    runtime_lines = [
        "| Stage | Average CPU Time (seconds) |",
        "|---|---|",
        f"| Video parsing (Charon-V) | {runtime_summary.get('video_parse_time_seconds', 0.0):.4f} |",
        f"| Action scoring (ActionScoreModule) | {runtime_summary.get('action_score_time_seconds', 0.0):.4f} |",
        f"| Frame selection & evaluation | {runtime_summary.get('selection_time_seconds', 0.0):.4f} |",
        f"| Total video pipeline | {runtime_summary.get('total_video_time_seconds', 0.0):.4f} |",
    ]

    report_content = f"""# Experiment 0 — Compression vs Accuracy

> [!WARNING]
> EXPERIMENT 0 SMOKE TEST — NOT PAPER-GRADE RESULT

## Dataset
- **Dataset Source**: ActivityNet Captions subset (`friedrichor/ActivityNet_Captions` fallback loaded successfully)
- **Number of downloaded videos**: {len(df_video["video_id"].unique())}
- **Number of event spans**: {len(df_event["event_label"].unique())}
- **Number of unavailable/skipped videos**: {summary_data.get("skipped_videos_count", 0)}

## What was tested
- **Action Score threshold sweep**: {action_thresholds}
- **Persistence threshold sweep**: {persistence_thresholds}
- **CPU-only execution**: Forced via `CUDA_VISIBLE_DEVICES=""`.

## Compression Table
{chr(10).join(comp_lines)}

## Accuracy Table
{chr(10).join(acc_lines)}

## Runtime Table (CPU-only)
{chr(10).join(runtime_lines)}
- **Peak RAM consumption**: {summary_data.get("peak_ram_mb", 0.0):.2f} MB

## Best Tradeoff
- **Best threshold pair (Action, Persistence)**: `{best_threshold_pair}`
- This pair achieved a Recall@5 of `{best_recall:.4f}` with a frame compression rate of `{best_comp:.4f}`.

## Failure Notes
Here is a list of events where no selected frame hit the ground-truth interval at K=10:
"""

    if unique_failures:
        report_content += "\n| Video ID | Event Label | Start Time | End Time | Thresh (Action, Persist) |\n|---|---|---|---|---|\n"
        for fail in unique_failures[:15]: # cap to 15 entries for readability
            report_content += f"| {fail['video_id']} | {fail['event_label']} | {fail['start_time']:.2f} | {fail['end_time']:.2f} | ({fail['threshold_action']:.2f}, {fail['threshold_persistence']:.2f}) |\n"
        if len(unique_failures) > 15:
            report_content += f"\n*(and {len(unique_failures) - 15} more failures)*\n"
    else:
        report_content += "\nNo failures recorded at K=10.\n"

    report_content += """
## Limitations
- Small smoke-test subset (5 videos, maximum 3 events per video).
- Not final paper-grade result.
- ActivityNet videos may be unavailable due to YouTube removal/blocking.
- The current signal uses codec packet sizes (`packet_size` bytes) for the scoring computation. The report acknowledges that we refer to this as the current implementation's standard packet size energy signal, and we make no claims of custom compressed-domain codec parsing until packet-size/coded-size signal is fully integrated.
- No QA, Cerberus, or graph verification has been tested yet.
"""

    report_path = os.path.join(out_dir, "summary_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Summary report generated at {report_path}")

    # Return best tradeoff details to runner
    return best_threshold_pair
