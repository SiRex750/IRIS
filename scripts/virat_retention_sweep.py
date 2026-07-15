"""Pillar-1 official benchmark: VIRAT CCTV frame retention sweep.

Evaluates Variant C (Full IRIS Codec Gate) against naive baselines (Uniform and Random)
at low retention budgets (1%, 2%, 5%, and 10%).

Computes:
  - event_recall
  - frame_recall
  - survivor_precision
  - paired Bootstrap CIs (1000 resamples, 95% confidence) comparing C vs. Uniform and C vs. Random.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import json
from pathlib import Path

import numpy as np

# Add repository root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import iris.charon_v as charon_v
from iris.iris_config import IRISConfig
from iris.l1_elysium import L1ElysiumCache
from iris.cached_frame import CachedFrame
from iris.frame_motion_descriptor import FrameMotionDescriptor
from iris.action_score import ActionScoreConfig, ActionScoreModule

DEFAULT_ROOT = REPO_ROOT / "eval" / "data" / "virat"
BUDGETS = [1, 2, 5, 10]  # percentages


def clip_stem(name):
    return name[:-4] if name.lower().endswith(".mp4") else name


def events_stem(name):
    if name.lower().endswith(".viratdata.events.txt"):
        return name[: -len(".viratdata.events.txt")]
    if name.lower().endswith(".events.txt"):
        return name[: -len(".events.txt")]
    return name


def find_matched_pairs(root):
    videos_dir = root / "videos"
    ann_dir = root / "annotations"

    videos = {clip_stem(p.name): p for p in videos_dir.glob("*.mp4")} if videos_dir.is_dir() else {}
    ann_candidates = list(ann_dir.glob("*.viratdata.events.txt")) if ann_dir.is_dir() else []
    if not ann_candidates and ann_dir.is_dir():
        ann_candidates = list(ann_dir.glob("*events.txt"))
    events = {events_stem(p.name): p for p in ann_candidates}

    matched_stems = sorted(set(videos) & set(events))
    video_only = sorted(set(videos) - set(events))
    events_only = sorted(set(events) - set(videos))

    pairs = [(stem, videos[stem], events[stem]) for stem in matched_stems]
    return pairs, video_only, events_only


def parse_events_txt(path):
    events = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) < 10:
                continue
            event_id = cols[0]
            event_type = cols[1]
            current_frame = int(cols[5])
            ev = events.setdefault(event_id, {
                "event_id": event_id,
                "event_type": event_type,
                "frames": [],
            })
            ev["frames"].append(current_frame)

    out = []
    for ev in events.values():
        out.append({
            "event_id": ev["event_id"],
            "event_type": ev["event_type"],
            "start": min(ev["frames"]),
            "end": max(ev["frames"]),
        })
    return out


def get_all_scored_frames(clip_path):
    """Parse video and run action scoring to return all candidate frames with features."""
    output_frames, stats, raw_records = charon_v.parse_video(
        str(clip_path), return_stats=True, return_raw=True, candidate_thresh=0.0
    )
    
    # Run Action Scoring
    action_score_config = ActionScoreConfig(
        luma_diff_weight=0.5,
        motion_weight=0.3,
        luma_entropy_weight=0.2,
        peak_distance=5,
        peak_prominence=0.05,
        persistence_threshold=0.4,
        max_prominence=0.5,
    )
    score_module = ActionScoreModule(config=action_score_config)
    score_records = score_module.score_all(raw_records)
    action_scores = {r["frame_idx"]: r for r in score_records}
    
    raw_map = {r["frame_idx"]: r for r in raw_records}
    for frame in output_frames:
        frame_idx = frame["frame_idx"]
        score_info = action_scores.get(
            frame_idx,
            {"action_score": 0.0, "is_peak": False, "persistence_value": 0.0}
        )
        frame["action_score"] = score_info["action_score"]
        frame["is_peak"] = score_info["is_peak"]
        frame["persistence_value"] = score_info["persistence_value"]
        frame["luma_entropy"] = raw_map.get(frame_idx, {}).get("luma_entropy", 0.0)
        frame["pict_type"] = raw_map.get(frame_idx, {}).get("frame_type", frame.get("pict_type", "?"))
        frame["packet_size"] = raw_map.get(frame_idx, {}).get("packet_size", frame.get("packet_size", 0.0))
        frame["divergence"] = raw_map.get(frame_idx, {}).get("divergence", frame.get("divergence", 0.0))
        frame["curl"] = raw_map.get(frame_idx, {}).get("curl", frame.get("curl", 0.0))
        frame["jacobian_frobenius"] = raw_map.get(frame_idx, {}).get("jacobian_frobenius", frame.get("jacobian_frobenius", 0.0))
        frame["hessian_max_eigenvalue"] = raw_map.get(frame_idx, {}).get("hessian_max_eigenvalue", frame.get("hessian_max_eigenvalue", 0.0))
        frame["motion_entropy"] = raw_map.get(frame_idx, {}).get("motion_entropy", frame.get("motion_entropy", 0.0))
        
    return output_frames, int(stats["total"])


def run_variant_c_selection(output_frames, budget_n):
    """Simulate frame admission to L1 Elysium cache under Variant C weights with capacity = budget_n."""
    # Variant C IRIS Config
    config = IRISConfig(
        l1_capacity=budget_n,
        l1_w_action=0.30,
        l1_w_query=0.00,  # No query in retention sweep
        l1_w_persist=0.15,
        l1_w_pagerank=0.00,  # No graph PageRank in retention sweep
        l1_w_entropy=0.10,
        l1_w_hessian=0.10,
        l1_w_recency=0.05
    )
    # Re-normalize weights since query & pagerank are zeroed out (sum = 0.70)
    # The normalized weights:
    # action = 0.30 / 0.70 = 0.42857
    # persist = 0.15 / 0.70 = 0.21429
    # entropy = 0.10 / 0.70 = 0.14286
    # hessian = 0.10 / 0.70 = 0.14286
    # recency = 0.05 / 0.70 = 0.07143
    # This keeps the relative ratios intact.
    config.l1_w_action = 0.30 / 0.70
    config.l1_w_query = 0.0
    config.l1_w_persist = 0.15 / 0.70
    config.l1_w_pagerank = 0.0
    config.l1_w_entropy = 0.10 / 0.70
    config.l1_w_hessian = 0.10 / 0.70
    config.l1_w_recency = 0.05 / 0.70
    
    cache = L1ElysiumCache(config=config)
    sorted_frames = sorted(output_frames, key=lambda x: x["frame_idx"])
    
    for frame in sorted_frames:
        motion = FrameMotionDescriptor(
            frame_idx=frame["frame_idx"],
            timestamp_sec=frame.get("timestamp", 0.0),
            luma_diff_energy=frame.get("luma_diff_energy", 0.0),
            divergence=frame.get("divergence", 0.0),
            curl=frame.get("curl", 0.0),
            jacobian_frobenius=frame.get("jacobian_frobenius", 0.0),
            hessian_max_eigenvalue=frame.get("hessian_max_eigenvalue", 0.0),
            motion_entropy=frame.get("motion_entropy", 0.0)
        )
        cached_frame = CachedFrame(
            frame_idx=frame["frame_idx"],
            timestamp_sec=frame.get("timestamp", 0.0),
            action_score=frame.get("action_score", 0.0),
            persistence_value=frame.get("persistence_value", 0.0),
            is_peak=frame.get("is_peak", False),
            motion=motion,
            embedding=None
        )
        cache.admit(cached_frame)
        
    return {f.frame_idx for f in cache.frames()}


def event_recall(events, sampled):
    if not events:
        return 0.0
    hit = sum(1 for ev in events if any(ev["start"] <= idx <= ev["end"] for idx in sampled))
    return hit / len(events)


def frame_recall(events, sampled):
    event_frames = set()
    for ev in events:
        event_frames.update(range(ev["start"], ev["end"] + 1))
    if not event_frames:
        return 0.0
    survived_event_frames = event_frames & sampled
    return len(survived_event_frames) / len(event_frames)


def survivor_precision(events, sampled):
    if not sampled:
        return 0.0
    event_frames = set()
    for ev in events:
        event_frames.update(range(ev["start"], ev["end"] + 1))
    inside = sum(1 for idx in sampled if idx in event_frames)
    return inside / len(sampled)


def uniform_sample(n, total_frames):
    if n <= 0 or total_frames <= 0:
        return set()
    step = total_frames / n
    return {min(total_frames - 1, int(i * step)) for i in range(n)}


def random_sample(n, total_frames, seed=42):
    if n <= 0 or total_frames <= 0:
        return set()
    rng = random.Random(seed)
    n = min(n, total_frames)
    return set(rng.sample(range(total_frames), n))


def bootstrap_paired_ci(metric_c, metric_baseline, n_boot=1000, seed=42):
    """Paired bootstrap confidence interval for mean(C - baseline)."""
    diffs = np.array(metric_c) - np.array(metric_baseline)
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    mean_diff = float(diffs.mean())
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    return mean_diff, ci_lo, ci_hi


def main():
    parser = argparse.ArgumentParser(description="Pillar 1 VIRAT Retention Sweep")
    parser.add_argument("root", nargs="?", default=str(DEFAULT_ROOT), help="Path to VIRAT dataset directory")
    args = parser.parse_args()

    root = Path(args.root)
    pairs, video_only, events_only = find_matched_pairs(root)

    print(f"Dataset root: {root}")
    print(f"Matched pairs: {len(pairs)}")

    sweep_results = {}

    for budget in BUDGETS:
        print(f"\nEvaluating budget: {budget}% ...")
        
        results_by_arm = {"variant_c": [], "uniform": [], "random": []}
        
        for stem, clip_path, events_path in pairs:
            events = parse_events_txt(events_path)
            output_frames, total_frames = get_all_scored_frames(clip_path)
            
            n = max(1, int(round((budget / 100.0) * total_frames)))
            
            sampled_c = run_variant_c_selection(output_frames, n)
            sampled_u = uniform_sample(n, total_frames)
            sampled_r = random_sample(n, total_frames, seed=42)
            
            for arm_name, sampled in [("variant_c", sampled_c), ("uniform", sampled_u), ("random", sampled_r)]:
                er = event_recall(events, sampled)
                fr = frame_recall(events, sampled)
                sp = survivor_precision(events, sampled)
                
                results_by_arm[arm_name].append({
                    "stem": stem,
                    "event_recall": er,
                    "frame_recall": fr,
                    "survivor_precision": sp,
                    "N": len(sampled),
                    "total_frames": total_frames
                })
        
        # Aggregated stats
        sweep_results[budget] = {}
        for arm in ["variant_c", "uniform", "random"]:
            mean_er = np.mean([r["event_recall"] for r in results_by_arm[arm]])
            mean_fr = np.mean([r["frame_recall"] for r in results_by_arm[arm]])
            mean_sp = np.mean([r["survivor_precision"] for r in results_by_arm[arm]])
            
            sweep_results[budget][arm] = {
                "mean_event_recall": float(mean_er),
                "mean_frame_recall": float(mean_fr),
                "mean_survivor_precision": float(mean_sp),
                "raw_results": results_by_arm[arm]
            }
            
            print(f"  Arm: {arm:<10} | Event Recall: {mean_er:.4%} | Frame Recall: {mean_fr:.4%} | Survivor Precision: {mean_sp:.4%}")
            
        # Bootstrap CIs for Event Recall
        c_er = [r["event_recall"] for r in results_by_arm["variant_c"]]
        u_er = [r["event_recall"] for r in results_by_arm["uniform"]]
        r_er = [r["event_recall"] for r in results_by_arm["random"]]
        
        diff_u, ci_u_lo, ci_u_hi = bootstrap_paired_ci(c_er, u_er)
        diff_r, ci_r_lo, ci_r_hi = bootstrap_paired_ci(c_er, r_er)
        
        sweep_results[budget]["contrasts"] = {
            "c_vs_uniform": {"mean_diff": diff_u, "ci_lo": ci_u_lo, "ci_hi": ci_u_hi},
            "c_vs_random": {"mean_diff": diff_r, "ci_lo": ci_r_lo, "ci_hi": ci_r_hi}
        }
        
        print(f"  Contrasts (Event Recall 95% paired CIs):")
        print(f"    C vs Uniform: Mean diff: {diff_u:+.4f} | CI: [{ci_u_lo:+.4f}, {ci_u_hi:+.4f}]")
        print(f"    C vs Random:  Mean diff: {diff_r:+.4f} | CI: [{ci_r_lo:+.4f}, {ci_r_hi:+.4f}]")

    # Save outputs
    out_dir = REPO_ROOT / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "virat_retention_sweep_report.json", "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved raw retention sweep JSON to: eval_results/virat_retention_sweep_report.json")


if __name__ == "__main__":
    main()
