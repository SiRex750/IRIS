import urllib.request
import os
import sys
import time
import numpy as np

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import aria
from aria import LLMBackend
from pipeline import run_pipeline


class MockLLMBackend(LLMBackend):
    def generate(self, prompt: str, context: str, model: str | None = None) -> str:
        print("\n--- [TEST] Mock LLM Generation ---")
        print("User Prompt:", prompt)
        print("Context lines count:", len(context.splitlines()))
        print("----------------------------------\n")
        return (
            "The video depicts a big buck bunny standing in a vibrant green meadow. "
            "It shows local motion spikes and residual energy changes corresponding to action moments."
        )


def main():
    # 1. Select the appropriate LLM Backend
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[INFO] No OPENAI_API_KEY env var found. Setting up Mock LLM Backend for isolated testing.")
        aria.set_backend(MockLLMBackend())
    else:
        print("[INFO] OPENAI_API_KEY found. Running testing with real OpenAI Backend.")

    # 2. Download or locate test video
    url = "https://www.w3schools.com/html/mov_bbb.mp4"
    temp_video = "track_b_test_video.mp4"
    
    if os.path.exists(temp_video):
        print(f"[INFO] Using existing test video: {temp_video}")
    else:
        print(f"Downloading test video from {url}...")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        success = False
        for attempt in range(1, 4):
            try:
                print(f"Attempt {attempt}...")
                urllib.request.urlretrieve(url, temp_video)
                print("Download successful.")
                success = True
                break
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                time.sleep(2)

        if not success:
            print("[ERROR] Could not download test video. Exiting test.")
            sys.exit(1)

    try:
        # 3. Run E2E pipeline for Track B
        print("\n--- Running Track B End-to-End Pipeline Check ---")
        # Run with verbose=True to print action scores, retrieved frames, raw answer, and claims
        result = run_pipeline(temp_video, "Summarize the action events seen in the video.", verbose=True)
        
        print("\n=== Track B Pipeline Execution Results ===")
        print(f"Final Answer:       {result['answer']}")
        print(f"Claims Verified:    {result['verified']}")
        print(f"Frames Processed:   {result['frames_processed']} (non-SKIP)")
        print(f"Continuous Peaks:   {result['peak_count']}")
        print(f"Compression Ratio:  {result['compression_ratio']:.3f} (SKIP% of total)")
        
        debug = result.get("debug_info", {})
        print("\n=== Top 3 Detected Action Peaks ===")
        peaks_found = []
        for idx, score_dict in debug.get("action_scores", {}).items():
            if score_dict["is_peak"]:
                peaks_found.append((idx, score_dict["action_score"], score_dict["persistence_value"]))
        
        # Sort by action score descending
        peaks_found.sort(key=lambda x: x[1], reverse=True)
        for idx, score, persistence in peaks_found[:3]:
            print(f"  Frame {idx:3d}: Score={score:.4f}, Persistence={persistence:.4f}")

        # Assertions
        assert len(result["answer"]) > 0, "Pipeline answer should not be empty"
        assert result["frames_processed"] > 0, "Should have processed non-SKIP frames"
        assert result["peak_count"] > 0, "Should have detected at least one peak frame"
        assert 0.0 < result["compression_ratio"] < 1.0, "Compression ratio should be a valid percentage"
        
        print("\n[SUCCESS] Track B isolation test completed and all assertions passed!")

    finally:
        if os.path.exists(temp_video):
            try:
                os.remove(temp_video)
                print("Cleaned up test video file.")
            except Exception as e:
                print(f"Warning: Could not clean up test video: {e}")

if __name__ == "__main__":
    main()
