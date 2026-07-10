import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
import numpy as np

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from iris.iris_config import IRISConfig
from iris.pipeline import run_pipeline
import iris.aria as aria

CONDITIONS = ["semantic_only", "hybrid_legacy", "ppr"]

def get_config_for_condition(condition: str) -> IRISConfig:
    if condition == "semantic_only":
        return IRISConfig(
            ranking_mode="legacy",
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
            disable_nli=False  # Keep NLI active so we can measure abstentions
        )
    elif condition == "hybrid_legacy":
        return IRISConfig(
            ranking_mode="legacy",
            alpha=0.4,
            beta=0.3,
            gamma=0.3,
            disable_nli=False
        )
    elif condition == "ppr":
        return IRISConfig(
            ranking_mode="ppr",
            disable_nli=False
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")

def evaluate_grounding(retrieved_frames: list[dict], gold_segments: list[list[float]]) -> tuple[float, float]:
    if not retrieved_frames:
        return 0.0, 0.0

    hits = 0
    for f in retrieved_frames:
        ts = f.get("timestamp", 0.0)
        in_gold = any(start <= ts <= end for start, end in gold_segments)
        if in_gold:
            hits += 1
            
    precision = hits / len(retrieved_frames)
    
    covered_segments = 0
    for start, end in gold_segments:
        has_frame = any(start <= f.get("timestamp", 0.0) <= end for f in retrieved_frames)
        if has_frame:
            covered_segments += 1
            
    recall = covered_segments / len(gold_segments) if gold_segments else 0.0
    return precision, recall

def run_single(clip_id: str, condition: str) -> None:
    """Run a single clip and condition in-process, outputting JSON to stdout."""
    # Set backend to local gemma3:4b
    aria.set_backend(aria.LlamaBackend(text_model="gemma3:4b"))

    gt_path = Path("test_data/ground_truth.json")
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    item = next((x for x in data if x["clip_id"] == clip_id), None)
    if not item:
        sys.stderr.write(f"Clip {clip_id} not found in ground truth.\n")
        sys.exit(1)

    video_path = item["video_path"]
    query_text = item["question"]
    gold_segments = item.get("gold_segments", [])

    config = get_config_for_condition(condition)
    t0 = time.time()
    res = run_pipeline(video_path, query_text, config=config, verbose=True)
    elapsed_ms = (time.time() - t0) * 1000.0

    answer = res.get("answer", "")
    raw_answer = res.get("raw_answer", "")
    abstains = "insufficient" in answer.lower() or "insufficient" in raw_answer.lower()

    retrieved_frames = res.get("debug_info", {}).get("retrieved_frames", [])
    precision, recall = evaluate_grounding(retrieved_frames, gold_segments)

    metrics = {
        "clip_id": clip_id,
        "condition": condition,
        "latency_ms": elapsed_ms,
        "abstains": bool(abstains),
        "grounding_precision": float(precision),
        "grounding_recall": float(recall),
        "frames_retrieved": int(len(retrieved_frames)),
        "retrieved_timestamps": [f.get("timestamp", 0.0) for f in retrieved_frames],
        "answer": answer
    }
    
    # Print clean JSON wrapper to stdout for parent process to capture
    print("===RESULT_JSON_START===")
    print(json.dumps(metrics))
    print("===RESULT_JSON_END===")

def run_eval(dry_run: bool = False) -> None:
    gt_path = Path("test_data/ground_truth.json")
    if not gt_path.exists():
        print(f"[ERROR] ground_truth.json not found at {gt_path}!")
        return

    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if dry_run:
        data = data[:1]
        print(f"[INFO] Dry-run active. Running on 1 clip only: {data[0]['clip_id']}\n")

    results = {c: [] for c in CONDITIONS}
    python_exe = sys.executable

    for item in data:
        clip_id = item["clip_id"]
        video_path = item["video_path"]
        query_text = item["question"]
        gold_segments = item.get("gold_segments", [])

        if not os.path.exists(video_path):
            print(f"[WARN] Video file missing: {video_path}. Skipping.")
            continue

        print(f"\n=========================================")
        print(f"Evaluating Clip: {clip_id}")
        print(f"Query: '{query_text}'")
        print(f"Gold segments (sec): {gold_segments}")
        print(f"=========================================")

        for condition in CONDITIONS:
            print(f" -> Running condition: {condition} (via subprocess to protect VRAM)...")
            
            # Execute subprocess to isolate GPU memory/VRAM
            cmd = [python_exe, __file__, "--run-single", "--clip-id", clip_id, "--condition", condition]
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                
                # Extract the result JSON from the subprocess output
                stdout_str = proc.stdout
                if "===RESULT_JSON_START===" in stdout_str:
                    parts = stdout_str.split("===RESULT_JSON_START===")[1].split("===RESULT_JSON_END===")
                    metrics = json.loads(parts[0].strip())
                    
                    results[condition].append(metrics)
                    print(f"    [RESULT] Precision: {metrics['grounding_precision']:.2%}")
                    print(f"             Recall:    {metrics['grounding_recall']:.2%}")
                    print(f"             Abstain:   {metrics['abstains']}")
                    print(f"             Latency:   {metrics['latency_ms']:.1f}ms")
                    print(f"             Retrieved Timestamps: {metrics['retrieved_timestamps']}")
                    print(f"             Answer:    {metrics['answer']}")
                else:
                    print(f"    [ERROR] Subprocess ran but output could not be parsed.")
                    print(f"    [STDOUT] {stdout_str}")
                    print(f"    [STDERR] {proc.stderr}")
                    
            except subprocess.CalledProcessError as e:
                print(f"    [ERROR] Subprocess failed with exit code {e.returncode}")
                print(f"    [STDOUT] {e.stdout}")
                print(f"    [STDERR] {e.stderr}")
            except Exception as ex:
                print(f"    [ERROR] Failed to run condition {condition}: {ex}")

    # Aggregated analysis
    print("\n" + "=" * 90)
    print("  IRIS GROUNDING & ABSTENTION ABLATION COMPARISON")
    print("=" * 90)
    header = f"  {'Condition':<15} | {'Avg Precision':<15} | {'Avg Recall':<15} | {'Abstention%':<12} | {'Avg Latency (ms)':<16}"
    print(header)
    print("  " + "-" * 86)
    
    report_data = {}
    for condition in CONDITIONS:
        metrics_list = results[condition]
        if not metrics_list:
            continue
            
        avg_precision = np.mean([m["grounding_precision"] for m in metrics_list])
        avg_recall = np.mean([m["grounding_recall"] for m in metrics_list])
        abstention_rate = np.mean([1.0 if m["abstains"] else 0.0 for m in metrics_list])
        avg_latency = np.mean([m["latency_ms"] for m in metrics_list])
        
        print(f"  {condition:<15} | "
              f"{avg_precision:<15.2%} | "
              f"{avg_recall:<15.2%} | "
              f"{abstention_rate:<12.2%} | "
              f"{avg_latency:<16.1f}")
              
        report_data[condition] = {
            "avg_precision": float(avg_precision),
            "avg_recall": float(avg_recall),
            "abstention_rate": float(abstention_rate),
            "avg_latency_ms": float(avg_latency),
            "runs": metrics_list
        }
    print("=" * 90 + "\n")
    
    # Save results
    out_dir = Path("eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "grounding_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[INFO] Saved grounding report to: eval_results/grounding_report.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRIS Grounding Eval Suite")
    parser.add_argument("--dry-run", action="store_true", help="Run on only 1 clip for debugging")
    parser.add_argument("--run-single", action="store_true", help="Internal flag to run a single clip/condition")
    parser.add_argument("--clip-id", type=str, help="Clip ID for run-single")
    parser.add_argument("--condition", type=str, help="Condition for run-single")
    args = parser.parse_args()
    
    if args.run_single:
        if not args.clip_id or not args.condition:
            sys.stderr.write("Error: --clip-id and --condition are required with --run-single\n")
            sys.exit(1)
        run_single(args.clip_id, args.condition)
    else:
        run_eval(dry_run=args.dry_run)
