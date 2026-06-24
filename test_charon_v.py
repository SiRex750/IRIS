import urllib.request
import os
import sys
import socket
from charon_v import parse_video

def main():
    socket.setdefaulttimeout(15.0)
    url = "https://www.w3schools.com/html/mov_bbb.mp4"
    temp_video = "mov_bbb.mp4"
    is_temp = True
    
    # 1. Download the public domain MP4 file
    print(f"Downloading test video from {url}...")
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
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
        # 2. Run the parser
        print("Parsing video...")
        output, stats = parse_video(temp_video, return_stats=True)
        print(f"Parsing complete. Found {len(output)} output entries out of {stats['total']} total frames.")
        
        # 3. Assertions
        assert isinstance(output, list), "Output must be a list"
        assert len(output) > 0, "Output list should not be empty"
        assert all(isinstance(f, dict) for f in output), "All output elements must be dicts"
        
        # Assert: All dicts have the raw signals and metadata
        required_keys = {"frame_idx", "timestamp", "residual_energy", "motion_magnitude", "entropy", "motion_vectors", "frame_type"}
        for f in output:
            assert required_keys.issubset(f.keys()), f"Dict missing required keys: {f.keys()}"
            
        # Assert: All frames are returned (length of output equals stats['total'])
        assert len(output) == stats["total"], f"Output length {len(output)} must match stats total {stats['total']}"
        
        # Assert: I-frame entries always have motion_vectors == []
        iframe_entries = [f for f in output if f["frame_type"] == "I"]
        assert len(iframe_entries) > 0, "There must be at least one I-frame in the output"
        assert all(f["motion_vectors"] == [] for f in iframe_entries), "All I-frames must have empty motion vectors"
        
        # Assert: At least some non-I-frames have non-empty motion vectors if there was motion
        non_iframe_entries = [f for f in output if f["frame_type"] != "I"]
        has_mvs = any(len(f["motion_vectors"]) > 0 for f in non_iframe_entries)
        print(f"Has non-empty motion vectors in non-I-frames: {has_mvs}")
        assert has_mvs, "Motion vector extraction failed: all non-I-frames have empty motion vectors!"
        
        print("\nAll Charon-V raw signal extraction assertions PASSED successfully!")
        
    finally:
        # Clean up the downloaded file
        if is_temp and os.path.exists(temp_video):
            try:
                os.remove(temp_video)
                print("Cleaned up test video file.")
            except Exception:
                pass

if __name__ == "__main__":
    main()
