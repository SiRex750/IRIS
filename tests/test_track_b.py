import urllib.request
import os
import sys
import time
import numpy as np

# Ensure root directory is in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import iris.aria as aria
from iris.aria import LLMBackend
from iris.pipeline import run_pipeline


class MockLLMBackend(LLMBackend):
    def generate(self, prompt: str, context: str, model: str | None = None, *args, **kwargs) -> str:
        print("\n--- [TEST] Mock LLM Generation ---")
        print("User Prompt:", prompt)
        print("Context lines count:", len(context.splitlines()))
        print("----------------------------------\n")
        return (
            "The video depicts a big buck bunny standing in a vibrant green meadow. "
            "It shows local motion spikes and residual energy changes corresponding to action moments."
        )


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
    # 1. Load the environment key from .env
    load_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("[INFO] No OPENAI_API_KEY env var found. Setting up Mock LLM Backend for isolated testing.")
        aria.set_backend(MockLLMBackend())
        model_in_use = "Mock-LLM-Model"
    else:
        print("[INFO] OPENAI_API_KEY found. Running E2E testing with real OpenAI Backend.")
        model_in_use = "gpt-4o-mini"

    # 2. Download or locate test video
    local_path = "mov_bbb.mp4"
    temp_video = "track_b_test_video.mp4"
    import shutil
    
    if os.path.exists(local_path):
        print(f"[INFO] Found local video copy '{local_path}'. Copying to '{temp_video}' for offline test.")
        shutil.copy(local_path, temp_video)
        success = True
    elif os.path.exists(temp_video):
        print(f"[INFO] Using existing test video: {temp_video}")
        success = True
    else:
        url = "https://www.w3schools.com/html/mov_bbb.mp4"
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
        # Default nms_window=10 is used to suppress adjacent peak frames
        result = run_pipeline(temp_video, "Summarize the action events seen in the video.", verbose=True, nms_window=10)
        
        print("\n=== Track B Pipeline Execution Results ===")
        print(f"Active LLM Model:   {model_in_use}")
        print(f"Final Answer:       {result['answer']}")
        
        # Clarify Mock vs. Real verification
        if result.get("nli_mocked", False):
            print("Claims Verified:    True (MOCKED - Cerberus-V is currently a stub)")
        else:
            print(f"Claims Verified:    {result['verified']}")
            
        print(f"Frames Processed:   {result['frames_processed']} (non-SKIP)")
        print(f"Continuous Peaks:   {result['peak_count']} (NMS Gated)")
        
        # Report both compression metrics clearly
        print(f"Compression Ratio (Skipped/Total):  {result['skipped_frames_ratio']:.6f} (portion of frames completely skipped)")
        print(f"Compression Ratio (Total/Stored):   {result['storage_reduction_factor']:.6f} (reduction factor in stored frames)")
        
        debug = result.get("debug_info", {})
        
        # 4. Print raw, unrounded values for the top 10 frames by action score to check actual spread
        all_scores = list(debug.get("action_scores", {}).items())
        all_scores.sort(key=lambda x: x[1]["action_score"], reverse=True)
        print("\n=== Top 10 Frames by Action Score (Raw & Unrounded) ===")
        for idx, score_dict in all_scores[:10]:
            print(f"  Frame {idx:3d}: Action Score = {score_dict['action_score']:.8f}, Persistence = {score_dict['persistence_value']:.8f}")

        # 5. Print top 3 peaks (genuinely separate events due to NMS)
        peaks_found = []
        for idx, score_dict in debug.get("action_scores", {}).items():
            if score_dict["is_peak"]:
                peaks_found.append((idx, score_dict["action_score"], score_dict["persistence_value"]))
        
        # Sort by action score descending
        peaks_found.sort(key=lambda x: x[1], reverse=True)
        print("\n=== Top 3 Peaks (NMS Gated & Suppressed) ===")
        if not peaks_found:
            print("  No peaks detected.")
        for idx, score, persistence in peaks_found[:3]:
            print(f"  Frame {idx:3d}: Score={score:.8f}, Persistence={persistence:.8f}")

        # Assertions
        assert len(result["answer"]) > 0, "Pipeline answer should not be empty"
        assert result["frames_processed"] > 0, "Should have processed non-SKIP frames"
        assert result["peak_count"] > 0, "Should have detected at least one peak frame"
        assert 0.0 < result["skipped_frames_ratio"] < 1.0, "Compression ratio should be a valid percentage"
        
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
