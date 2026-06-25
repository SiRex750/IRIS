"""
IRIS ablation evaluation harness.

Runs all 4 ablation conditions against test videos
and outputs metrics for the paper.

Ablation table:
    Condition       | Codec gating | NLI verification
    Baseline        | None         | None
    Ablation 1      | KG only      | None
    Ablation 2      | None         | Uniform NLI
    Full IRIS       | Both jointly | Risk-proportional

Metrics: accuracy, compression_ratio, latency_ms

Owner: Track D
"""
from __future__ import annotations

import time
import json
import argparse
from pathlib import Path
from iris.iris_config import IRISConfig
from iris.pipeline import run_pipeline

ABLATION_CONDITIONS = ["baseline", "ablation_1", "ablation_2", "full_iris"]


def get_config_for_condition(condition: str) -> IRISConfig:
    """Returns the IRISConfig custom-tailored for each ablation condition."""
    if condition == "baseline":
        # No codec gating (threshold=0), no NLI
        return IRISConfig(
            candidate_thresh=0.0,
            cerberus_low_thresh=0.35,
            cerberus_high_thresh=0.70,
            disable_nli=True,
            adaptive=False
        )
    elif condition == "ablation_1":
        # Codec gating active, no NLI
        return IRISConfig(
            candidate_thresh=0.08,
            cerberus_low_thresh=0.35,
            cerberus_high_thresh=0.70,
            disable_nli=True,
            adaptive=True
        )
    elif condition == "ablation_2":
        # No codec gating (threshold=0), uniform NLI (thresholds set to minimum to force full_nli)
        return IRISConfig(
            candidate_thresh=0.0,
            cerberus_low_thresh=0.01,
            cerberus_high_thresh=0.02,
            disable_nli=False,
            adaptive=False
        )
    elif condition == "full_iris":
        # Both active with default configurations
        return IRISConfig(
            candidate_thresh=0.08,
            cerberus_low_thresh=0.35,
            cerberus_high_thresh=0.70,
            disable_nli=False,
            adaptive=True
        )
    else:
        raise ValueError(f"Unknown ablation condition: {condition}")


def run_ablation(video_path: str, query: str, condition: str) -> dict:
    """Run one ablation condition. Returns metrics dict."""
    config = get_config_for_condition(condition)
    
    t0 = time.time()
    res = run_pipeline(video_path, query, config=config, verbose=True)
    elapsed_ms = (time.time() - t0) * 1000.0
    
    # Extract verification details
    claims_verified = res.get("verified_claims", []) or []
    claims_rejected = res.get("rejected_claims", []) or []
    claims_unverifiable = res.get("unverifiable_claims", []) or []
    
    debug_info = res.get("debug_info", {})
    retrieved_frames = debug_info.get("retrieved_frames", [])
    retrieved_idxs = [f["frame_idx"] for f in retrieved_frames]
    retrieved_scores = [round(f.get("action_score", 0.0), 3) for f in retrieved_frames]
    
    # Print detailed debug info for analysis
    print(f"  [DEBUG {condition}] Retrieved frames: {retrieved_idxs} with action scores {retrieved_scores}")
    print(f"  [DEBUG {condition}] Raw answer: {res.get('raw_answer')}")
    print(f"  [DEBUG {condition}] Verified claims: {claims_verified}")
    print(f"  [DEBUG {condition}] Rejected claims: {claims_rejected}")
    print(f"  [DEBUG {condition}] Unverifiable claims: {claims_unverifiable}")
    
    total_claims = len(claims_verified) + len(claims_rejected) + len(claims_unverifiable)
    nli_calls_estimated = len(claims_verified) + len(claims_rejected) if condition in ("ablation_2", "full_iris") else 0
    
    return {
        "condition": condition,
        "frames_processed": res.get("frames_processed", 0),
        "compression_ratio": res.get("compression_ratio", 0.0),
        "skipped_ratio": res.get("skipped_frames_ratio", 0.0),
        "latency_ms": elapsed_ms,
        "total_claims": total_claims,
        "verified_claims_count": len(claims_verified),
        "rejected_claims_count": len(claims_rejected),
        "unverifiable_claims_count": len(claims_unverifiable)
    }


def run_full_eval(video_path: str, query: str) -> list[dict]:
    """Run all conditions and return comparative results list."""
    results = []
    print(f"\nEvaluating E2E Ablation on Video: {video_path}")
    print(f"Query: '{query}'\n")
    
    for condition in ABLATION_CONDITIONS:
        print(f"-> Running ablation condition: {condition}...")
        try:
            res = run_ablation(video_path, query, condition)
            results.append(res)
        except Exception as e:
            print(f"Error running condition {condition}: {e}")
            
    return results


def print_comparison_table(results: list[dict]) -> None:
    """Prints a structured ASCII comparison table for the paper."""
    header = f"  {'Condition':<15} | {'Frames Sent':<12} | {'Skip Ratio':<10} | {'Latency (ms)':<12} | {'Verified/Rejected/Unverifiable':<32}"
    print("\n" + "=" * 90)
    print("  IRIS ABLATION COMPARISON TABLE")
    print("=" * 90)
    print(header)
    print("  " + "-" * 86)
    for r in results:
        verified_str = f"{r['verified_claims_count']}/{r['rejected_claims_count']}/{r['unverifiable_claims_count']}"
        print(f"  {r['condition']:<15} | "
              f"{r['frames_processed']:<12} | "
              f"{r['skipped_ratio']:<10.2%} | "
              f"{r['latency_ms']:<12.1f} | "
              f"{verified_str:<32}")
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="IRIS Ablation Evaluation Suite")
    parser.add_argument("video_path", nargs="?", default="benchmark_results/mov_bbb.mp4",
                        help="Path to the test video file.")
    parser.add_argument("--query", default="What action events happen in this video?",
                        help="Query to run against the video.")
    parser.add_argument("--out", default="benchmark_results/ablation_results.json",
                        help="Path to save output JSON report.")
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        # Fallback to static_video.mp4 if mov_bbb.mp4 doesn't exist
        fallback = Path("benchmark_results/static_video.mp4")
        if fallback.exists():
            video_path = fallback
        else:
            print(f"Error: Neither {args.video_path} nor {fallback} exists.")
            return
            
    results = run_full_eval(str(video_path), args.query)
    print_comparison_table(results)
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved ablation results to: {out_path}")


if __name__ == "__main__":
    main()
