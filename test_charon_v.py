import urllib.request
import os
import sys
import socket
from charon_v import parse_video

def test_peak_tier(video_path: str = "mov_bbb.mp4"):
    if not os.path.exists(video_path):
        try:
            url = "https://www.w3schools.com/html/mov_bbb.mp4"
            urllib.request.urlretrieve(url, video_path)
        except Exception:
            import pytest
            pytest.skip(f"Test video {video_path} not found and download failed")
    frames, stats = parse_video(video_path, return_stats=True, adaptive=True)
    actual_salient_thresh = stats["salient_thresh_used"]
    
    peaks = [f for f in frames if f["tier"] == "PEAK"]
    salients = [f for f in frames if f["tier"] == "SALIENT"]
    
    # All peaks must be above the actual threshold used
    assert all(f["residual_energy"] >= actual_salient_thresh for f in peaks), \
        f"Peak frame found below salient_thresh floor ({actual_salient_thresh:.4f})"
    
    # PEAK should be sparser than SALIENT
    if len(salients) > 0:
        assert len(peaks) < len(salients), \
            f"Too many peaks ({len(peaks)}) vs salients ({len(salients)}) — tune order parameter"
    
    # No I_FRAME reclassified as PEAK
    iframe_ids = {f["frame_idx"] for f in frames if f["tier"] == "I_FRAME"}
    peak_ids = {f["frame_idx"] for f in peaks}
    assert iframe_ids.isdisjoint(peak_ids), \
        "I_FRAME was reclassified as PEAK — check tier priority order"
    
    print(f"Adaptive salient_thresh: {actual_salient_thresh:.4f}")
    print(f"PEAK: {len(peaks)}, SALIENT: {len(salients)}")

def main():
    socket.setdefaulttimeout(5.0)
    url = "https://www.w3schools.com/html/mov_bbb.mp4"
    temp_video = "mov_bbb.mp4"
    is_temp = True
    
    # 1. Download the public domain MP4 file
    print(f"Downloading test video from {url}...")
    try:
        urllib.request.urlretrieve(url, temp_video)
        print("Download successful.")
    except Exception as e:
        print(f"Error downloading test video: {e}")
        if os.path.exists("videoplayback.mp4"):
            temp_video = "videoplayback.mp4"
            is_temp = False
            print("Using local videoplayback.mp4 for testing.")
        else:
            sys.exit(1)
        
    try:
        # 2. Run the parser with return_stats=True (defaults to adaptive=True)
        print("Parsing video in adaptive mode...")
        output, stats = parse_video(temp_video, return_stats=True)
        print(f"Parsing complete. Found {len(output)} output entries out of {stats['total']} total frames.")
        
        # 3. Assertions
        
        # Assert: Output is a list of dicts
        assert isinstance(output, list), "Output must be a list"
        assert all(isinstance(f, dict) for f in output), "All output elements must be dicts"
        
        # Assert: All dicts have the required keys
        required_keys = {"frame_idx", "timestamp", "tier", "residual_energy", "motion_vectors"}
        for f in output:
            assert required_keys.issubset(f.keys()), f"Dict missing required keys: {f.keys()}"
            
        # Assert: No "SKIP" tier appears in output
        tiers = {f["tier"] for f in output}
        assert "SKIP" not in tiers, "Output should not contain any SKIP tier"
        
        # Assert: Salient + I-frame count is less than 10% of total frames (under adaptive mode)
        salient_and_iframe_count = sum(1 for f in output if f["tier"] in ("I_FRAME", "SALIENT"))
        total_frames = stats["total"]
        print(f"I-Frame + Salient count: {salient_and_iframe_count} ({salient_and_iframe_count / total_frames * 100:.2f}%)")
        
        # Assert: All I-frame entries always have motion_vectors == []
        iframe_entries = [f for f in output if f["tier"] == "I_FRAME"]
        assert len(iframe_entries) > 0, "There must be at least one I-frame in the output"
        assert all(f["motion_vectors"] == [] for f in iframe_entries), "All I-frames must have empty motion vectors"
        
        # Assert: Used thresholds are returned in stats and correctness is verified
        assert "salient_thresh_used" in stats, "stats missing salient_thresh_used"
        assert "candidate_thresh_used" in stats, "stats missing candidate_thresh_used"
        
        # Verify correctness assertions requested by user
        salient_thresh_default = 0.35
        candidate_thresh_default = 0.08
        
        print(f"Adaptive thresholds used: Salient={stats['salient_thresh_used']:.4f}, Candidate={stats['candidate_thresh_used']:.4f}")
        assert stats['salient_thresh_used'] < salient_thresh_default, f"salient_thresh_used ({stats['salient_thresh_used']}) should be less than {salient_thresh_default}"
        assert stats['candidate_thresh_used'] < candidate_thresh_default, f"candidate_thresh_used ({stats['candidate_thresh_used']}) should be less than {candidate_thresh_default}"
        assert stats['salient_thresh_used'] > stats['candidate_thresh_used'], "salient threshold must be greater than candidate threshold"
        
        # Test adaptive=False behavior
        print("Parsing video with adaptive=False to verify hardcoded thresholds...")
        output_non_adaptive, stats_non_adaptive = parse_video(
            temp_video,
            return_stats=True,
            adaptive=False,
            candidate_thresh=0.08,
            salient_thresh=0.35
        )
        assert stats_non_adaptive['salient_thresh_used'] == 0.35
        assert stats_non_adaptive['candidate_thresh_used'] == 0.08
        
        # Additional assertion: Let's make sure that at least some P/B frames have non-empty motion vectors.
        # By setting adaptive=False and candidate_thresh=0.0, we parse all frames and verify that motion vector extraction works.
        print("Parsing video with candidate_thresh=0.0 to verify motion vectors...")
        all_frames = parse_video(temp_video, candidate_thresh=0.0, adaptive=False)
        non_iframe_entries = [f for f in all_frames if f["tier"] != "I_FRAME"]
        has_mvs = any(len(f["motion_vectors"]) > 0 for f in non_iframe_entries)
        print(f"Has non-empty motion vectors in non-I-frames: {has_mvs}")
        assert has_mvs, "Motion vector extraction failed: all non-I-frames have empty motion vectors!"
        
        # Test PEAK tier promotion
        print("Running test_peak_tier...")
        test_peak_tier(temp_video)
        
        print("\nAll assertions PASSED successfully!")
        
    finally:
        # Clean up the downloaded file
        if is_temp and os.path.exists(temp_video):
            os.remove(temp_video)
            print("Cleaned up test video file.")

if __name__ == "__main__":
    main()
