"""
Integration tests for the IRIS end-to-end pipeline.

Owner: Track B
"""
from __future__ import annotations
import os
import urllib.request
import pytest
import numpy as np
import aria
from aria import LLMBackend
from pipeline import run_pipeline, run


class MockLLMBackend(LLMBackend):
    def generate(self, prompt: str, context: str, model: str | None = None) -> str:
        return (
            "The video depicts a big buck bunny standing in a vibrant green meadow. "
            "It shows local motion spikes and residual energy changes corresponding to action moments."
        )


@pytest.fixture(scope="module")
def bbb_video():
    url = "https://www.w3schools.com/html/mov_bbb.mp4"
    temp_video = "test_pipeline_video.mp4"
    
    # Download the video
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, temp_video)
    except Exception as e:
        pytest.skip(f"Could not download test video for integration test: {e}")
        
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
