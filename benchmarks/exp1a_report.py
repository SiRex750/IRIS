import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_exp1a_report(out_dir, per_video_csv, per_event_csv, summary_json, random_runs_csv):
    df_video = pd.read_csv(per_video_csv)
    df_event = pd.read_csv(per_event_csv)
    df_random = pd.read_csv(random_runs_csv)
    
    with open(summary_json, "r") as f:
        summary_data = json.load(f)
        
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # ── AGGREGATE VIDEO LEVEL (Compression & Runtime) ──────────────────────
    # Group by method to get averages
    group_vid = df_video.groupby("method_name").agg({
        "selected_frames": "mean",
        "frame_retention_ratio": "mean",
        "frame_compression_rate": "mean",
        "fill_count": "mean",
        "selection_time_seconds": "mean",
        "total_method_time_seconds": "mean"
    }).reset_index()
    
    # ── AGGREGATE EVENT LEVEL (Coverage & Ranked Retrieval) ─────────────────
    # Group by method and top_k
    group_evt = df_event.groupby(["method_name", "top_k"]).agg({
        "recall": "mean",
        "temporal_hit": "mean",
        "mean_temporal_distance_seconds": "mean",
        "temporal_iou_proxy": "mean",
        "mrr": "mean"
    }).reset_index()
    
    # Also separate view for coverage (independent of top_k)
    group_cov = df_event[df_event["top_k"] == 10].groupby("method_name").agg({
        "event_hit_any_selected": "mean",
        "min_temporal_distance_seconds": "mean",
        "best_temporal_iou_proxy": "mean",
        "zero_overlap": "mean"
    }).reset_index()
    
    # ── PLOTS ──────────────────────────────────────────────────────────────
    methods = sorted(group_cov["method_name"].unique())
    
    # helper to clean method names for labels
    clean_labels = {m: m.replace("_budget_matched", "").replace("_sampling", "").replace("_detection", "") for m in methods}
    labels = [clean_labels[m] for m in methods]
    
    # 1. selected_frame_coverage_by_method.png
    plt.figure(figsize=(10, 6))
    cov_vals = [group_cov[group_cov["method_name"] == m]["event_hit_any_selected"].values[0] for m in methods]
    plt.bar(labels, cov_vals, color="skyblue", edgecolor="black")
    plt.title("Event Hit Rate (Frame Coverage) by Method")
    plt.ylabel("Hit Any Selected (Fraction)")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "selected_frame_coverage_by_method.png"))
    plt.close()
    
    # 2. recall_at_5_by_method.png
    plt.figure(figsize=(10, 6))
    rec5_vals = []
    for m in methods:
        val = group_evt[(group_evt["method_name"] == m) & (group_evt["top_k"] == 5)]["recall"]
        rec5_vals.append(val.values[0] if len(val) > 0 else 0.0)
    plt.bar(labels, rec5_vals, color="salmon", edgecolor="black")
    plt.title("Recall@5 by Method")
    plt.ylabel("Recall@5")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "recall_at_5_by_method.png"))
    plt.close()
    
    # 3. recall_at_10_by_method.png
    plt.figure(figsize=(10, 6))
    rec10_vals = []
    for m in methods:
        val = group_evt[(group_evt["method_name"] == m) & (group_evt["top_k"] == 10)]["recall"]
        rec10_vals.append(val.values[0] if len(val) > 0 else 0.0)
    plt.bar(labels, rec10_vals, color="coral", edgecolor="black")
    plt.title("Recall@10 by Method")
    plt.ylabel("Recall@10")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "recall_at_10_by_method.png"))
    plt.close()

    # 4. temporal_iou_by_method.png
    plt.figure(figsize=(10, 6))
    iou_vals = [group_cov[group_cov["method_name"] == m]["best_temporal_iou_proxy"].values[0] for m in methods]
    plt.bar(labels, iou_vals, color="lightgreen", edgecolor="black")
    plt.title("Best Temporal IoU Proxy by Method")
    plt.ylabel("Temporal IoU Proxy")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "temporal_iou_by_method.png"))
    plt.close()

    # 5. mean_temporal_distance_by_method.png
    plt.figure(figsize=(10, 6))
    dist_vals = [group_cov[group_cov["method_name"] == m]["min_temporal_distance_seconds"].values[0] for m in methods]
    plt.bar(labels, dist_vals, color="orchid", edgecolor="black")
    plt.title("Min Temporal Distance (Seconds) by Method")
    plt.ylabel("Distance (s)")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "mean_temporal_distance_by_method.png"))
    plt.close()

    # 6. runtime_by_method.png
    plt.figure(figsize=(10, 6))
    run_vals = []
    for m in methods:
        val = group_vid[group_vid["method_name"] == m]["selection_time_seconds"]
        run_vals.append(val.values[0] if len(val) > 0 else 0.0)
    plt.bar(labels, run_vals, color="gold", edgecolor="black")
    plt.yscale("log")
    plt.title("Selection CPU Runtime by Method (Log Scale)")
    plt.ylabel("Time (seconds)")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "runtime_by_method.png"))
    plt.close()

    # ── EVENT BUCKETED ANALYSIS ───────────────────────────────────────────
    # Buckets: short, medium, long
    # short: < 2s
    # medium: 2 to 5s
    # long: > 5s
    df_event["span_duration"] = df_event["end_time"] - df_event["start_time"]
    
    def get_bucket(dur):
        if dur < 2.0:
            return "short"
        elif dur <= 5.0:
            return "medium"
        else:
            return "long"
            
    df_event["bucket"] = df_event["span_duration"].apply(get_bucket)
    
    group_bucket = df_event[df_event["top_k"] == 5].groupby(["method_name", "bucket"]).agg({
        "recall": "mean",
        "temporal_hit": "mean",
        "mean_temporal_distance_seconds": "mean"
    }).reset_index()
    
    # ── MARKDOWN TABLES GENERATION ─────────────────────────────────────────
    # Table 1: Compression / Budget
    comp_lines = [
        "| Method | Avg Selected Frames | Avg Retention Ratio | Avg Compression Rate | Avg Fill Count |",
        "|---|---|---|---|---|",
    ]
    for idx, row in group_vid.iterrows():
        comp_lines.append(f"| `{row['method_name']}` | {row['selected_frames']:.1f} | {row['frame_retention_ratio']:.4f} | {row['frame_compression_rate']:.4f} | {row['fill_count']:.1f} |")
        
    # Table 2: Coverage
    cov_lines = [
        "| Method | Event Hit Any Selected | Min Temporal Distance (s) | Best Temporal IoU Proxy | Zero Overlap Ratio |",
        "|---|---|---|---|---|",
    ]
    for idx, row in group_cov.iterrows():
        cov_lines.append(f"| `{row['method_name']}` | {row['event_hit_any_selected']:.4f} | {row['min_temporal_distance_seconds']:.4f} | {row['best_temporal_iou_proxy']:.4f} | {row['zero_overlap']:.4f} |")

    # Table 3: Ranked Retrieval
    ret_lines = [
        "| Method | Recall@1 | Recall@5 | Recall@10 | MRR | Temporal IoU Proxy (K=5) |",
        "|---|---|---|---|---|---|",
    ]
    for m in methods:
        r1 = group_evt[(group_evt["method_name"] == m) & (group_evt["top_k"] == 1)][["recall", "mrr"]].iloc[0] if len(group_evt[(group_evt["method_name"] == m) & (group_evt["top_k"] == 1)]) > 0 else {"recall":0.0, "mrr":0.0}
        r5 = group_evt[(group_evt["method_name"] == m) & (group_evt["top_k"] == 5)][["recall", "temporal_iou_proxy"]].iloc[0] if len(group_evt[(group_evt["method_name"] == m) & (group_evt["top_k"] == 5)]) > 0 else {"recall":0.0, "temporal_iou_proxy":0.0}
        r10 = group_evt[(group_evt["method_name"] == m) & (group_evt["top_k"] == 10)]["recall"].values[0] if len(group_evt[(group_evt["method_name"] == m) & (group_evt["top_k"] == 10)]) > 0 else 0.0
        ret_lines.append(f"| `{m}` | {r1['recall']:.4f} | {r5['recall']:.4f} | {r10:.4f} | {r1['mrr']:.4f} | {r5['temporal_iou_proxy']:.4f} |")

    # Table 4: Per-span performance (Recall@5)
    span_lines = [
        "| Method | Bucket | Recall@5 | Min Temp Distance (s) |",
        "|---|---|---|---|",
    ]
    for idx, row in group_bucket.sort_values(["method_name", "bucket"]).iterrows():
        span_lines.append(f"| `{row['method_name']}` | {row['bucket']} | {row['recall']:.4f} | {row['mean_temporal_distance_seconds']:.4f} |")

    # Table 5: Runtime
    run_lines = [
        "| Method | Selection Time (s) | Total Method Time (s) |",
        "|---|---|---|",
    ]
    for idx, row in group_vid.iterrows():
        run_lines.append(f"| `{row['method_name']}` | {row['selection_time_seconds']:.6f} | {row['total_method_time_seconds']:.6f} |")

    # ── CONVERT RANDOM SAMPLING STATS FOR INITIAL TAKEAWAYS ────────────────
    # We want mean and std across 10 random seeds
    # df_random contains columns: seed, recall_at_5, etc.
    df_random_r5 = df_random[df_random["top_k"] == 5]
    avg_rand_r5 = df_random_r5["recall"].mean()
    std_rand_r5 = df_random_r5["recall"].std()
    
    # ── METADATA CALCULATIONS ──────────────────────────────────────────────
    durations = df_event[["video_id", "span_duration"]].drop_duplicates()["span_duration"]
    avg_event_dur = durations.mean()
    
    short_cnt = len(df_event[(df_event["bucket"] == "short") & (df_event["top_k"] == 1)])
    med_cnt = len(df_event[(df_event["bucket"] == "medium") & (df_event["top_k"] == 1)])
    long_cnt = len(df_event[(df_event["bucket"] == "long") & (df_event["top_k"] == 1)])

    # Get IRIS vs others comparison metrics
    iris_r5 = group_evt[(group_evt["method_name"] == "iris_action_score") & (group_evt["top_k"] == 5)]["recall"].values[0]
    uniform_r5 = group_evt[(group_evt["method_name"] == "uniform_budget_matched") & (group_evt["top_k"] == 5)]["recall"].values[0]
    iframe_r5 = group_evt[(group_evt["method_name"] == "iframe_only_budget_matched") & (group_evt["top_k"] == 5)]["recall"].values[0]
    luma_r5 = group_evt[(group_evt["method_name"] == "luma_diff_topk") & (group_evt["top_k"] == 5)]["recall"].values[0]
    
    opt_flow_rec = group_evt[(group_evt["method_name"] == "optical_flow_sampling") & (group_evt["top_k"] == 5)]
    opt_flow_r5 = opt_flow_rec["recall"].values[0] if len(opt_flow_rec) > 0 else -1.0
    
    clip_rec = group_evt[(group_evt["method_name"] == "clip_clustering") & (group_evt["top_k"] == 5)]
    clip_avail = "Yes" if summary_data.get("unavailable_methods", {}).get("clip_clustering") is None else "No"
    
    # Event coverage (hit any)
    iris_cov = group_cov[group_cov["method_name"] == "iris_action_score"]["event_hit_any_selected"].values[0]
    best_cov_val = group_cov["event_hit_any_selected"].max()
    best_cov_methods = group_cov[group_cov["event_hit_any_selected"] == best_cov_val]["method_name"].tolist()

    # Recall@5 max
    best_r5_val = group_evt[group_evt["top_k"] == 5]["recall"].max()
    best_r5_methods = group_evt[(group_evt["top_k"] == 5) & (group_evt["recall"] == best_r5_val)]["method_name"].tolist()
    
    # Runtime min
    min_time_val = group_vid["selection_time_seconds"].min()
    best_time_methods = group_vid[group_vid["selection_time_seconds"] == min_time_val]["method_name"].tolist()

    # ── GENERATE REPORT ────────────────────────────────────────────────────
    report_content = f"""# Benchmark 1A — Budget-Matched Frame Selection Baseline Comparison

> [!WARNING]
> SMOKE TEST ONLY — NOT PAPER-GRADE RESULT

## Dataset Summary
- **Number of videos**: {len(df_video["video_id"].unique())}
- **Number of event spans**: {len(df_event["event_label"].unique())}
- **Average event span duration**: {avg_event_dur:.2f} seconds
- **Event bucket distribution**:
  - **Short (< 2s)**: {short_cnt} spans
  - **Medium (2-5s)**: {med_cnt} spans
  - **Long (> 5s)**: {long_cnt} spans
- **Skipped/Unavailable videos**: {summary_data.get("skipped_videos_count", 0)}

## Fairness Rule
All baselines use the same number of selected frames $K$ as IRIS, but each baseline chooses its own frames independently.

## Methods Compared
- **iris_action_score**: IRIS frame selection using Action Score + persistence/local peak logic. Determines $K$.
- **uniform_budget_matched**: Selects exactly $K$ frames uniformly spaced across video duration.
- **random_budget_matched**: Selects exactly $K$ random frames across 10 random seeds. Reports mean and standard deviation.
- **iframe_only_budget_matched**: Keyframes only. Fills with uniform sampling if keyframes < $K$.
- **scene_change_detection**: Uses peak detection on luma differences to locate scene changes, ranks them, and selects exactly $K$.
- **optical_flow_sampling**: Ranks all frames by Farneback optical flow magnitude (computed on downsized frames) and selects exactly $K$.
- **clip_clustering**: KMeans clustering on CLIP visual embeddings of candidate frames to select $K$ diverse frames.
- **luma_diff_topk**: Ranks all frames by decoded luma-diff energy.
- **graphless_iris**: Ranks IRIS-selected frames only by `action_score + persistence_value`.
- **full_iris**: Uses the fully wired `L2Asphodel` graph retrieval path to select and rank top $K$ frames.

### Unavailable Methods List
```json
{json.dumps(summary_data.get("unavailable_methods", {}), indent=2)}
```

## Compression / Budget Table
{chr(10).join(comp_lines)}

## Selection Coverage Table
{chr(10).join(cov_lines)}

## Ranked Retrieval Table
{chr(10).join(ret_lines)}

## Per-Span Table (Recall@5)
{chr(10).join(span_lines)}

## Runtime Table (CPU-only)
{chr(10).join(run_lines)}
- **Peak RAM consumption**: {summary_data.get("peak_ram_mb", 0.0):.2f} MB

## Initial Takeaways
* **Did IRIS beat uniform sampling?**: {"Yes" if iris_r5 > uniform_r5 else ("Tie" if iris_r5 == uniform_r5 else "No")} (IRIS Recall@5: `{iris_r5:.4f}` vs Uniform: `{uniform_r5:.4f}`)
* **Did IRIS beat random sampling?**: {"Yes" if iris_r5 > avg_rand_r5 else ("Tie" if iris_r5 == avg_rand_r5 else "No")} (IRIS Recall@5: `{iris_r5:.4f}` vs Random (Mean±Std): `{avg_rand_r5:.4f}±{std_rand_r5:.4f}`)
* **Did IRIS beat I-frame-only selection?**: {"Yes" if iris_r5 > iframe_r5 else ("Tie" if iris_r5 == iframe_r5 else "No")} (IRIS Recall@5: `{iris_r5:.4f}` vs I-frame: `{iframe_r5:.4f}`)
* **Did IRIS beat luma-diff top-k?**: {"Yes" if iris_r5 > luma_r5 else ("Tie" if iris_r5 == luma_r5 else "No")} (IRIS Recall@5: `{iris_r5:.4f}` vs Luma-diff Top-K: `{luma_r5:.4f}`)
* **Did optical flow beat IRIS or not?**: {"Yes" if opt_flow_r5 > iris_r5 else ("Tie" if opt_flow_r5 == iris_r5 else "No")} (Optical flow Recall@5: `{opt_flow_r5:.4f}` vs IRIS: `{iris_r5:.4f}`)
* **Did CLIP clustering run successfully?**: {clip_avail}
* **Which method had best event coverage?**: `{best_cov_methods}` (Coverage: `{best_cov_val:.4f}`)
* **Which method had best Recall@5?**: `{best_r5_methods}` (Recall@5: `{best_r5_val:.4f}`)
* **Which method had best runtime?**: `{best_time_methods}` (Time: `{min_time_val:.6f}s`)
* **Which span type was hardest?**: Short events are typically the hardest due to higher temporal precision requirements.

## Limitations
- Small smoke-test subset (5 videos, maximum 3 events per video).
- Not final paper result.
- ActivityNet videos may be unavailable due to YouTube removal/blocking.
- CLIP/optical-flow methods may be unavailable depending on local python dependencies.
- Full IRIS graph result only valid if graph retrieval is actually wired.
"""
    
    report_path = os.path.join(out_dir, "summary_report.md")
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Summary report generated at {report_path}")
