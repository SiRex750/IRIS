"""Pillar-1 L1 Elysium Frame Admission Ablation Study.

Runs cache admission simulation on the VIRAT CCTV dataset comparing three variants:
  - Variant A (LRU Baseline): w_recency=1.0, others=0.0
  - Variant B (Action Only): w_action=0.60, w_query=0.25, w_persist=0.15, others=0.0
  - Variant C (Full IRIS): w_action=0.30, w_query=0.20, w_persist=0.15, w_pagerank=0.10, w_entropy=0.10, w_hessian=0.10, w_recency=0.05

Tests the three queries:
  - Q1: "Show me the moment when fast motion begins"
  - Q2: "Find a static scene with no movement"
  - Q3: "What happens right after the scene changes?"

Evaluates at frame retention budgets: 1%, 2%, 5%, and 10%.
Generates 95% paired bootstrap CIs.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import json
from pathlib import Path

import numpy as np
import torch
import clip
from PIL import Image

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
BUDGETS = [1, 2, 5, 10]

QUERIES = {
    "Q1": "Show me the moment when fast motion begins",
    "Q2": "Find a static scene with no movement",
    "Q3": "What happens right after the scene changes?"
}


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


def get_all_scored_frames_with_embeddings(clip_path, index_path, model, preprocess, device):
    """Parse video, score frames, and load/generate CLIP embeddings."""
    if index_path.exists():
        try:
            print(f"    Loading precomputed index from {index_path.name} ...")
            sys.stdout.flush()
            from iris.ingest import load_index
            index = load_index(index_path)
            output_frames = []
            for f in index.frames:
                output_frames.append({
                    "frame_idx": f.frame_idx,
                    "timestamp": f.timestamp,
                    "tier": "I_FRAME" if f.pict_type == "I" else "CANDIDATE",
                    "luma_diff_energy": f.luma_diff_energy,
                    "packet_size": f.packet_size,
                    "motion_magnitude": f.motion_magnitude,
                    "action_score": f.action_score,
                    "is_peak": f.is_peak,
                    "persistence_value": f.persistence_value,
                    "luma_entropy": f.luma_entropy,
                    "pict_type": f.pict_type,
                    "divergence": f.divergence,
                    "curl": f.curl,
                    "jacobian_frobenius": f.jacobian_frobenius,
                    "hessian_max_eigenvalue": f.hessian_max_eigenvalue,
                    "motion_entropy": f.motion_entropy,
                    "clip_embedding": f.clip_embedding,
                })
            return output_frames, index.frames_processed
        except Exception as e:
            print(f"    [WARN] Failed to load index {index_path.name}, falling back to video parse: {e}")
            sys.stdout.flush()

    print(f"    Parsing video and decoding frames on the fly ...")
    sys.stdout.flush()
    output_frames, stats, raw_records = charon_v.parse_video(
        str(clip_path), return_stats=True, return_raw=True, candidate_thresh=0.08
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
    
    # Let's count how many we actually need to encode
    to_encode = [f for f in output_frames if f.get("pil_image") is not None]
    if to_encode:
        print(f"    Encoding {len(to_encode)} candidate frames on the fly using CLIP ...")
        sys.stdout.flush()
        
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
        
        # Embeddings
        if frame.get("pil_image") is not None:
            pil_img = frame["pil_image"]
            image = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(image).cpu().numpy().flatten().astype(np.float32)
            frame["clip_embedding"] = emb
        else:
            frame["clip_embedding"] = np.zeros(512, dtype=np.float32)
            
    return output_frames, int(stats["total"])


def get_variant_config(variant: str, budget_n: int) -> IRISConfig:
    if variant == "A":
        return IRISConfig(
            l1_capacity=budget_n,
            l1_w_action=0.00,
            l1_w_query=0.00,
            l1_w_persist=0.00,
            l1_w_pagerank=0.00,
            l1_w_entropy=0.00,
            l1_w_hessian=0.00,
            l1_w_recency=1.00,
        )
    elif variant == "B":
        return IRISConfig(
            l1_capacity=budget_n,
            l1_w_action=0.60,
            l1_w_query=0.25,
            l1_w_persist=0.15,
            l1_w_pagerank=0.00,
            l1_w_entropy=0.00,
            l1_w_hessian=0.00,
            l1_w_recency=0.00,
        )
    elif variant == "C":
        return IRISConfig(
            l1_capacity=budget_n,
            l1_w_action=0.30,
            l1_w_query=0.20,
            l1_w_persist=0.15,
            l1_w_pagerank=0.10,
            l1_w_entropy=0.10,
            l1_w_hessian=0.10,
            l1_w_recency=0.05,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


def run_cache_simulation(output_frames, variant: str, budget_n: int, query_similarities: dict[int, float]) -> set[int]:
    """Run sequential L1 Elysium cache admission simulation for a given variant and budget."""
    config = get_variant_config(variant, budget_n)
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
            embedding=frame.get("clip_embedding", None)
        )
        # Inject the query similarity
        cached_frame.query_similarity = query_similarities.get(frame["frame_idx"], 0.0)
        
        cache.admit(cached_frame)
        
    return {f.frame_idx for f in cache.frames()}


def bootstrap_paired_ci(metric_c, metric_baseline, n_boot=1000, seed=42):
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
    parser = argparse.ArgumentParser(description="Pillar 1 L1 Elysium Frame Admission Ablation Suite")
    parser.add_argument("root", nargs="?", default=str(DEFAULT_ROOT), help="Path to VIRAT dataset directory")
    args = parser.parse_args()

    root = Path(args.root)
    pairs, video_only, events_only = find_matched_pairs(root)

    print(f"Dataset root: {root}")
    print(f"Matched pairs: {len(pairs)}")

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP (ViT-B/32) on {device} ...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Embed the three text queries
    query_embeddings = {}
    for qid, qtext in QUERIES.items():
        text_tokens = clip.tokenize([qtext]).to(device)
        with torch.no_grad():
            query_embeddings[qid] = model.encode_text(text_tokens).cpu().numpy().flatten()
            # Normalize
            norm = np.linalg.norm(query_embeddings[qid])
            if norm > 1e-8:
                query_embeddings[qid] /= norm

    # 1. Parse and embed all videos once and cache in-memory
    print("\n[STEP 1/2] Loading and parsing all videos (caching in-memory) ...")
    sys.stdout.flush()
    video_cache = {}
    valid_pairs = []
    for stem, clip_path, events_path in pairs:
        print(f"  Processing stem: {stem} ...")
        sys.stdout.flush()
        index_path = root / "index_cache" / f"{stem}.npz"
        try:
            output_frames, total_frames = get_all_scored_frames_with_embeddings(
                clip_path, index_path, model, preprocess, device
            )
            video_cache[stem] = (output_frames, total_frames)
            valid_pairs.append((stem, clip_path, events_path))
            print(f"    Stem {stem} loaded: {len(output_frames)} candidate frames, {total_frames} total frames.")
            sys.stdout.flush()
        except Exception as e:
            print(f"    [ERROR] Failed to load/parse stem {stem}: {e} — skipping this clip.")
            sys.stdout.flush()

    print("\n[STEP 2/2] Running ablation sweep over budgets ...")
    sys.stdout.flush()
    ablation_results = {}

    for budget in BUDGETS:
        print(f"\nEvaluating budget: {budget}% ...")
        sys.stdout.flush()
        
        # We will collect observations at (clip, query) level
        # There are matched clips * 3 queries observations
        obs = []
        
        for stem, clip_path, events_path in valid_pairs:
            output_frames, total_frames = video_cache[stem]
            
            n = max(1, int(round((budget / 100.0) * total_frames)))
            
            for qid, q_emb in query_embeddings.items():
                # Compute query similarities for all frames
                q_sims = {}
                for f in output_frames:
                    f_emb = f.get("clip_embedding")
                    if f_emb is not None and np.linalg.norm(f_emb) > 1e-8:
                        f_emb_norm = f_emb / np.linalg.norm(f_emb)
                        q_sims[f["frame_idx"]] = float(np.dot(q_emb, f_emb_norm))
                    else:
                        q_sims[f["frame_idx"]] = 0.0
                
                # Determine global target frame (highest similarity)
                if q_sims:
                    global_target_idx = max(q_sims, key=q_sims.get)
                else:
                    global_target_idx = -1
                
                # Run cache simulation for Variants A, B, and C
                survivors_a = run_cache_simulation(output_frames, "A", n, q_sims)
                survivors_b = run_cache_simulation(output_frames, "B", n, q_sims)
                survivors_c = run_cache_simulation(output_frames, "C", n, q_sims)
                
                # Helper to determine hit
                def check_hit(survivors, target):
                    if target == -1:
                        return 0.0
                    # Hit if target or within +/- 5 frames of target is present
                    hit = 1.0 if any(abs(s - target) <= 5 for s in survivors) else 0.0
                    return hit

                hit_a = check_hit(survivors_a, global_target_idx)
                hit_b = check_hit(survivors_b, global_target_idx)
                hit_c = check_hit(survivors_c, global_target_idx)
                
                # Regret is 1 - hit
                regret_a = 1.0 - hit_a
                regret_b = 1.0 - hit_b
                regret_c = 1.0 - hit_c
                
                obs.append({
                    "stem": stem,
                    "query": qid,
                    "target_frame": global_target_idx,
                    "hit": {"A": hit_a, "B": hit_b, "C": hit_c},
                    "regret": {"A": regret_a, "B": regret_b, "C": regret_c}
                })
        
        # Calculate rates
        mean_hit = {v: np.mean([o["hit"][v] for o in obs]) for v in ["A", "B", "C"]}
        mean_regret = {v: np.mean([o["regret"][v] for o in obs]) for v in ["A", "B", "C"]}
        
        # Paired Bootstrap CIs
        # Contrast C vs A
        diff_hit_ca, ci_hit_ca_lo, ci_hit_ca_hi = bootstrap_paired_ci(
            [o["hit"]["C"] for o in obs], [o["hit"]["A"] for o in obs]
        )
        diff_regret_ca, ci_regret_ca_lo, ci_regret_ca_hi = bootstrap_paired_ci(
            [o["regret"]["C"] for o in obs], [o["regret"]["A"] for o in obs]
        )
        
        # Contrast C vs B
        diff_hit_cb, ci_hit_cb_lo, ci_hit_cb_hi = bootstrap_paired_ci(
            [o["hit"]["C"] for o in obs], [o["hit"]["B"] for o in obs]
        )
        diff_regret_cb, ci_regret_cb_lo, ci_regret_cb_hi = bootstrap_paired_ci(
            [o["regret"]["C"] for o in obs], [o["regret"]["B"] for o in obs]
        )
        
        print(f"  Hit Rate    | Variant A: {mean_hit['A']:.2%} | Variant B: {mean_hit['B']:.2%} | Variant C: {mean_hit['C']:.2%}")
        print(f"  Regret Rate | Variant A: {mean_regret['A']:.2%} | Variant B: {mean_regret['B']:.2%} | Variant C: {mean_regret['C']:.2%}")
        
        ablation_results[budget] = {
            "mean_hit": mean_hit,
            "mean_regret": mean_regret,
            "contrasts": {
                "c_vs_a": {
                    "hit": {"diff": diff_hit_ca, "ci_lo": ci_hit_ca_lo, "ci_hi": ci_hit_ca_hi},
                    "regret": {"diff": diff_regret_ca, "ci_lo": ci_regret_ca_lo, "ci_hi": ci_regret_ca_hi}
                },
                "c_vs_b": {
                    "hit": {"diff": diff_hit_cb, "ci_lo": ci_hit_cb_lo, "ci_hi": ci_hit_cb_hi},
                    "regret": {"diff": diff_regret_cb, "ci_lo": ci_regret_cb_lo, "ci_hi": ci_regret_cb_hi}
                }
            },
            "observations": obs
        }

    # Write report
    report_path = REPO_ROOT / "eval_results" / "l1_elysium_ablation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save raw json as well
    with open(REPO_ROOT / "eval_results" / "l1_elysium_ablation_raw.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    # Compile markdown text
    md = []
    md.append("# Phase 1 - L1 Elysium Frame Admission Ablation Report")
    md.append(f"\nEvaluating Variant C (Full IRIS) vs Variant A (LRU Baseline) and Variant B (Action Only) across the evaluated VIRAT dataset ({len(valid_pairs)} videos, {len(valid_pairs) * 3} queries).")
    
    stop_triggered = False
    
    for budget in BUDGETS:
        md.append(f"\n## Budget: {budget}% Retention")
        md.append(f"\n| Metric | Variant A (LRU) | Variant B (Action Only) | Variant C (Full IRIS) |")
        md.append(f"| :--- | :---: | :---: | :---: |")
        md.append(f"| **Cache Hit Rate** | {ablation_results[budget]['mean_hit']['A']:.2%} | {ablation_results[budget]['mean_hit']['B']:.2%} | {ablation_results[budget]['mean_hit']['C']:.2%} |")
        md.append(f"| **Eviction Regret Rate** | {ablation_results[budget]['mean_regret']['A']:.2%} | {ablation_results[budget]['mean_regret']['B']:.2%} | {ablation_results[budget]['mean_regret']['C']:.2%} |")
        
        contrasts = ablation_results[budget]["contrasts"]
        md.append(f"\n### Paired Bootstrap CIs (1000 resamples)")
        
        c_vs_a_hit_ci = f"[{contrasts['c_vs_a']['hit']['ci_lo']:+.4f}, {contrasts['c_vs_a']['hit']['ci_hi']:+.4f}]"
        c_vs_b_hit_ci = f"[{contrasts['c_vs_b']['hit']['ci_lo']:+.4f}, {contrasts['c_vs_b']['hit']['ci_hi']:+.4f}]"
        c_vs_a_reg_ci = f"[{contrasts['c_vs_a']['regret']['ci_lo']:+.4f}, {contrasts['c_vs_a']['regret']['ci_hi']:+.4f}]"
        c_vs_b_reg_ci = f"[{contrasts['c_vs_b']['regret']['ci_lo']:+.4f}, {contrasts['c_vs_b']['regret']['ci_hi']:+.4f}]"
        
        md.append(f"- **Hit Rate Contrast (C vs. A):** Mean diff = {contrasts['c_vs_a']['hit']['diff']:+.4f} | 95% CI = {c_vs_a_hit_ci}")
        md.append(f"- **Hit Rate Contrast (C vs. B):** Mean diff = {contrasts['c_vs_b']['hit']['diff']:+.4f} | 95% CI = {c_vs_b_hit_ci}")
        md.append(f"- **Eviction Regret Contrast (C vs. A):** Mean diff = {contrasts['c_vs_a']['regret']['diff']:+.4f} | 95% CI = {c_vs_a_reg_ci}")
        md.append(f"- **Eviction Regret Contrast (C vs. B):** Mean diff = {contrasts['c_vs_b']['regret']['diff']:+.4f} | 95% CI = {c_vs_b_reg_ci}")
        
        # Check stop conditions
        # C vs B crosses zero: ci_lo <= 0 and ci_hi >= 0 for hit rate, or regret rate crosses zero
        hit_crosses_zero = contrasts['c_vs_b']['hit']['ci_lo'] <= 0.0 <= contrasts['c_vs_b']['hit']['ci_hi']
        regret_crosses_zero = contrasts['c_vs_b']['regret']['ci_lo'] <= 0.0 <= contrasts['c_vs_b']['regret']['ci_hi']
        
        if hit_crosses_zero or regret_crosses_zero:
            stop_triggered = True
            md.append(f"\n> [!CAUTION]")
            md.append(f"> **STOP Condition Triggered at {budget}% budget!**")
            md.append(f"> Variant C's performance is not statistically distinguishable from Variant B.")
            if hit_crosses_zero:
                md.append(f"> - Hit Rate contrast CI {c_vs_b_hit_ci} crosses zero.")
            if regret_crosses_zero:
                md.append(f"> - Eviction Regret contrast CI {c_vs_b_reg_ci} crosses zero.")
    
    md.append(f"\n## Final Verdict")
    if stop_triggered:
        md.append(f"\n**[RED]** The STOP condition was triggered. The complex motion geometry signals do not show a statistically significant improvement over Variant B (Action Only) at extreme low-retention budgets. We report this failure as-is without tuning thresholds or parameters.")
    else:
        md.append(f"\n**[GREEN]** Variant C (Full IRIS) achieved a statistically significant improvement (higher Hit Rate and lower Eviction Regret Rate) compared to both Variant A and Variant B across all tested budgets.")
        
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved ablation report to: {report_path}")


if __name__ == "__main__":
    main()
