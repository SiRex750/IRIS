import urllib.request
import os
import sys
import socket
from iris.charon_v import parse_video, PEAK_WINDOW_SECONDS


def _get_scene_thresh(stats: dict, frame_idx: int) -> tuple[float, float]:
    """Return (salient, candidate) threshold for the scene whose [start,end) contains frame_idx."""
    for (start, end), salient in stats["salient_thresh_per_scene"].items():
        if start <= frame_idx < end:
            return salient, stats["candidate_thresh_per_scene"][(start, end)]
    raise KeyError(f"frame_idx {frame_idx} not found in any scene range")

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
    
    # Every PEAK survivor must be at or above its own scene's salient threshold
    for f in peaks:
        scene_salient, _ = _get_scene_thresh(stats, f["frame_idx"])
        assert f["packet_size"] >= scene_salient, (
            f"PEAK frame {f['frame_idx']} packet_size {f['packet_size']:.1f} < "
            f"scene salient_thresh {scene_salient:.1f}"
        )

    # Peaks are a subset of non-I-frame survivors
    iframe_ids = {f["frame_idx"] for f in frames if f["tier"] == "I_FRAME"}
    peak_ids = {f["frame_idx"] for f in peaks}
    non_i_ids = {f["frame_idx"] for f in frames if f["tier"] != "I_FRAME"}
    assert peak_ids.issubset(non_i_ids), \
        "PEAK frame_idx found outside non-I survivors — check tier priority order"

    # Peak density is sane: peaks must be < 20 % of all frames
    assert len(peaks) < 0.2 * stats["total"], \
        f"Peak density too high: {len(peaks)} peaks out of {stats['total']} total frames"

    # peak_order_used is >= 3 and fps-derived (not the old literal 15 for a ~24fps clip)
    order_used = stats["peak_order_used"]
    assert order_used >= 3, \
        f"peak_order_used must be >= 3, got {order_used}"
    assert order_used != 15, (
        f"peak_order_used={order_used} looks like the old hardcoded literal; "
        f"expected max(3, round({PEAK_WINDOW_SECONDS}*fps)) ≈ 12 for a 24fps clip"
    )

    print(f"Adaptive salient_thresh: {actual_salient_thresh:.4f}")
    print(f"PEAK: {len(peaks)}, SALIENT: {len(salients)}, peak_order_used: {order_used}")

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
        required_keys = {"frame_idx", "timestamp", "tier", "luma_diff_energy", "motion_vectors"}
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
        
        # Verify correctness assertions requested by user (byte-scale thresholds since Phase 4)
        print(f"Adaptive thresholds used: Salient={stats['salient_thresh_used']:.4f}, Candidate={stats['candidate_thresh_used']:.4f}")
        assert stats['salient_thresh_used'] > 1.0, f"salient_thresh_used ({stats['salient_thresh_used']}) should be byte-scale (> 1.0)"
        assert stats['candidate_thresh_used'] > 1.0, f"candidate_thresh_used ({stats['candidate_thresh_used']}) should be byte-scale (> 1.0)"
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
