import urllib.request
import os
import sys
import time
import numpy as np

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import iris.aria as aria
from iris.pipeline import run_pipeline

def load_env():
    """Manually load .env file from the same directory as the test script."""
    env_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")

def main():
    load_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[ERROR] No OPENAI_API_KEY found in .env! This script requires the real OpenAI API backend.")
        sys.exit(1)

    url = "https://www.w3schools.com/html/mov_bbb.mp4"
    temp_video = "bbb_integration_test.mp4"

    if os.path.exists(temp_video):
        print(f"[INFO] Using existing test video: {temp_video}")
    else:
        print(f"Downloading test video from {url}...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, temp_video)
        print("Download successful.")

    try:
        runs = []
        for i in range(1, 3):
            print(f"\n=========================================")
            print(f"       RUN {i} OF THE INTEGRATION TEST")
            print(f"=========================================")

            result = run_pipeline(
                temp_video,
                "Summarize the action events seen in the video.",
                verbose=True,
                nms_window=10
            )
            runs.append(result)

            is_mocked = result.get("nli_mocked", False)
            gate_label = 'MOCKED' if is_mocked else 'REAL NLI GATE'

            print(f"\n=== Run {i} Pipeline Execution Results ===")
            print(f"Final Answer (verified claims only): {result['answer']}")
            print(f"Raw Answer (all claims before gate): {result['raw_answer']}")
            print(f"")
            print(f"Claims Verified (gate passed?): {result['verified']} ({gate_label})")

            # Fix 9b/9c: show the 3-way claim breakdown explicitly
            v  = result.get("verified_claims", [])
            r  = result.get("rejected_claims", [])
            u  = result.get("unverifiable_claims", [])
            print(f"  Verified claims    ({len(v)}): {v}")
            print(f"  Rejected claims    ({len(r)}): {r}")
            print(f"  Unverifiable claims({len(u)}): {u}")

            print(f"\nFrames Processed:   {result['frames_processed']} (non-SKIP)")
            print(f"Continuous Peaks:   {result['peak_count']} (NMS Gated)")
            print(f"Compression Ratio (Skipped/Total):  {result['skipped_frames_ratio']:.6f}")
            print(f"Compression Ratio (Total/Stored):   {result['storage_reduction_factor']:.6f}")

            print("\n--- Latency Breakdown (Seconds) ---")
            timings = result.get("timings", {})
            for stage, elapsed in timings.items():
                print(f"  {stage:15s}: {elapsed:.4f}s")

            debug = result.get("debug_info", {})

            # Fix 9a proof: print the actual context_text passed to ARIA so we can see captions
            print("\n--- Context Text Passed to ARIA (fact pool, showing captions) ---")
            print(debug.get("context_text", "<not available>"))

            # Print raw, unrounded values for the top 10 frames
            all_scores = list(debug.get("action_scores", {}).items())
            all_scores.sort(key=lambda x: x[1]["action_score"], reverse=True)
            print("\n--- Top 10 Frames by Action Score (Raw & Unrounded) ---")
            for idx, score_dict in all_scores[:10]:
                print(f"  Frame {idx:3d}: Action Score = {score_dict['action_score']:.8f}, Persistence = {score_dict['persistence_value']:.8f}")

            # Print frames passed into ARIA prompt context
            retrieved = debug.get("retrieved_frames", [])
            print("\n--- Frames Passed into ARIA Prompt Context ---")
            for f in retrieved:
                caption_str = f.get("caption") or "no caption"
                print(f"  Frame {f['frame_idx']:3d} at {f['timestamp']:.2f}s | caption='{caption_str}' | action={f['action_score']:.6f} persist={f['persistence_value']:.6f}")

            # Print final NMS-gated peak list
            peaks = [idx for idx, score_info in debug.get("action_scores", {}).items() if score_info["is_peak"]]
            peaks.sort()
            print(f"\n--- Final NMS-Gated Peak List (Total: {len(peaks)}) ---")
            print(f"  Peak Frame Indices: {peaks}")

        # Programmatically compare Run 1 and Run 2 to check determinism
        print("\n=========================================")
        print("    COMPARISON BETWEEN RUN 1 AND RUN 2")
        print("=========================================")
        run1, run2 = runs[0], runs[1]

        # Compare retrieved frame lists
        r1_retrieved = [f['frame_idx'] for f in run1['debug_info']['retrieved_frames']]
        r2_retrieved = [f['frame_idx'] for f in run2['debug_info']['retrieved_frames']]
        retrieved_match = r1_retrieved == r2_retrieved
        print(f"Retrieved Frames Deterministic?   {retrieved_match} (Run 1: {r1_retrieved} | Run 2: {r2_retrieved})")

        # Compare peaks list
        r1_peaks = sorted([idx for idx, score_info in run1['debug_info']['action_scores'].items() if score_info['is_peak']])
        r2_peaks = sorted([idx for idx, score_info in run2['debug_info']['action_scores'].items() if score_info['is_peak']])
        peaks_match = r1_peaks == r2_peaks
        print(f"Peak Detection Deterministic?     {peaks_match} (Run 1: {r1_peaks} | Run 2: {r2_peaks})")

        # Compare NLI verification result (boolean)
        nli_match = run1['verified'] == run2['verified']
        print(f"NLI Verification Deterministic?   {nli_match} (Run 1: {run1['verified']} | Run 2: {run2['verified']})")

        # 3-way breakdown determinism
        def fmt_breakdown(res):
            v = res.get("verified_claims", [])
            r = res.get("rejected_claims", [])
            u = res.get("unverifiable_claims", [])
            return f"{len(v)} verified / {len(r)} rejected / {len(u)} unverifiable"
        print(f"Claim Breakdown Run 1:            {fmt_breakdown(run1)}")
        print(f"Claim Breakdown Run 2:            {fmt_breakdown(run2)}")

        # Compare raw answers (before gate)
        raw_match = run1['raw_answer'] == run2['raw_answer']
        print(f"Raw Answers Identical?            {raw_match}")

        # Compare final answers (after gate)
        answers_match = run1['answer'] == run2['answer']
        print(f"Final Answers Identical?          {answers_match}")
        if not answers_match:
            print("\n--- Final Answer from Run 1 ---")
            print(run1['answer'])
            print("\n--- Final Answer from Run 2 ---")
            print(run2['answer'])

    finally:
        if os.path.exists(temp_video):
            try:
                os.remove(temp_video)
                print("\nCleaned up test video file.")
            except Exception as e:
                print(f"Warning: Could not clean up test video: {e}")

if __name__ == "__main__":
    main()
