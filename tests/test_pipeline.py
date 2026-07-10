"""
Integration tests for the IRIS end-to-end pipeline.

Owner: Track B
"""
from __future__ import annotations
import os
import urllib.request
import pytest
import numpy as np
import iris.aria as aria
from iris.aria import LLMBackend
from iris.pipeline import run_pipeline, run


class MockLLMBackend(LLMBackend):
    def generate(self, prompt: str, context: str, model: str | None = None) -> str:
        return (
            "The video depicts a big buck bunny standing in a vibrant green meadow. "
            "It shows local motion spikes and residual energy changes corresponding to action moments."
        )


@pytest.fixture(scope="module")
def bbb_video():
    local_path = "mov_bbb.mp4"
    temp_video = "test_pipeline_video.mp4"
    import shutil
    import time
    
    # Prioritize local offline video file to bypass network sandbox restrictions
    if os.path.exists(local_path):
        print(f"[INFO] Found local video copy '{local_path}'. Copying to '{temp_video}' for offline test.")
        shutil.copy(local_path, temp_video)
        success = True
    else:
        url = "https://www.w3schools.com/html/mov_bbb.mp4"
        # Fallback to download with retries and timeout if local copy is missing
        success = False
        for attempt in range(1, 4):
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                with opener.open(url, timeout=15) as response:
                    with open(temp_video, 'wb') as f:
                        f.write(response.read())
                success = True
                break
            except Exception as e:
                print(f"Download attempt {attempt} failed: {e}")
                if os.path.exists(temp_video):
                    try:
                        os.remove(temp_video)
                    except Exception:
                        pass
                time.sleep(2)
            
    if not success:
        pytest.skip("Could not locate or download test video for integration test")
        
    yield temp_video
    
    # Cleanup
    if os.path.exists(temp_video):
        try:
            os.remove(temp_video)
        except Exception:
            pass


def test_pipeline_integration(bbb_video):
    # Set up Mock LLM backend to avoid using API credits
    original_backend = aria.get_backend()
    aria.set_backend(MockLLMBackend())
    
    try:
        # Run end-to-end pipeline check
        result = run_pipeline(bbb_video, "Summarize the action events seen in the video.", verbose=True, nms_window=10)
        
        # Assertions
        assert "answer" in result
        assert "verified" in result
        assert "frames_processed" in result
        assert "peak_count" in result
        assert "skipped_frames_ratio" in result
        
        assert len(result["answer"]) > 0
        assert result["frames_processed"] > 0
        assert result["peak_count"] > 0
        assert 0.0 < result["skipped_frames_ratio"] < 1.0

        # Fix 9b/9c: new result fields must always be present
        assert "raw_answer" in result
        assert "verified_claims" in result
        assert "rejected_claims" in result
        assert "unverifiable_claims" in result
        assert isinstance(result["verified_claims"], list)
        assert isinstance(result["rejected_claims"], list)
        assert isinstance(result["unverifiable_claims"], list)

        # Fix 9b: is_verified must be consistent with the 3-way breakdown
        expected_verified = (
            len(result["rejected_claims"]) == 0
            and len(result["unverifiable_claims"]) == 0
        )
        assert result["verified"] == expected_verified, (
            f"is_verified={result['verified']} but rejected={result['rejected_claims']} "
            f"unverifiable={result['unverifiable_claims']}"
        )

        # verbose debug_info must include context_text and unverifiable_claims
        assert "context_text" in result["debug_info"]
        assert "unverifiable_claims" in result["debug_info"]

        # Test backward-compatible interface
        compat_result = run(bbb_video, "Summarize the action events seen in the video.")
        assert compat_result["answer"] == result["answer"]
        assert compat_result["verified"] == result["verified"]
        assert compat_result["frames_processed"] == result["frames_processed"]
        assert compat_result["peak_count"] == result["peak_count"]
        assert compat_result["compression_ratio"] == result["skipped_frames_ratio"]
        
    finally:
        # Restore original backend
        aria.set_backend(original_backend)


def test_aria_frame_mismatch_regression(bbb_video):
    import re
    # Set up Mock LLM backend to avoid using API credits
    original_backend = aria.get_backend()
    aria.set_backend(MockLLMBackend())
    
    try:
        # Run pipeline with verbose=True so we get debug_info
        result = run_pipeline(bbb_video, "Summarize the action events seen in the video.", verbose=True, nms_window=10)
        
        # 1. Extract frame indices from the returned retrieved_frames list in debug_info
        retrieved_indices = {int(f["frame_idx"]) for f in result["debug_info"]["retrieved_frames"]}
        
        # 2. Extract frame indices from the context_text fed into ARIA
        context_text = result["debug_info"]["context_text"]
        context_indices = {int(m) for m in re.findall(r"Frame (\d+)", context_text)}
        
        # 3. Assert no frame in ARIA's context is absent from retrieved_frames, and vice versa
        assert retrieved_indices == context_indices, (
            f"Mismatch! retrieved_frames={retrieved_indices} vs context_text={context_indices}"
        )
        
    finally:
        # Restore original backend
        aria.set_backend(original_backend)


def test_pipeline_visual_debug_mode(bbb_video):
    # Set up Mock LLM backend
    original_backend = aria.get_backend()
    aria.set_backend(MockLLMBackend())
    
    from unittest.mock import patch
    from iris.iris_config import IRISConfig
    
    # Create a config with visual_debug_mode enabled
    debug_config = IRISConfig()
    debug_config.visual_debug_mode = True
    
    try:
        # Patch ConfigManager.get_config to return our debug_config
        with patch('iris.iris_config.ConfigManager.get_config', return_value=debug_config):
            # Run the pipeline with verbose=True
            result = run_pipeline(bbb_video, "Summarize the action events.", verbose=True)
            
            # Since visual_debug_mode is enabled, there should be a "debug_frames" directory
            # created next to the bbb_video path
            debug_frames_dir = os.path.join(os.path.dirname(bbb_video), "debug_frames")
            assert os.path.exists(debug_frames_dir)
            
            # Check that there are annotated frame PNG images saved inside it
            saved_frames = os.listdir(debug_frames_dir)
            assert len(saved_frames) > 0
            for f in saved_frames:
                assert f.startswith("frame_")
                assert f.endswith(".png")
                
            # Clean up the generated debug frames
            import shutil
            shutil.rmtree(debug_frames_dir)
            
    finally:
        # Restore original backend
        aria.set_backend(original_backend)

