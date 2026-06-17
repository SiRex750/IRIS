import av
import numpy as np
import sys
import pprint
from scipy.signal import argrelextrema

def detect_peaks(
    all_frame_energies: list[tuple[int, float]],
    salient_thresh: float,
    order: int = 3,
) -> set[int]:
    """
    Find local maxima of the residual energy curve over ALL frames
    (including SKIP frames) to preserve signal continuity.
    Only promotes frames above salient_thresh to avoid noise peaks
    in quiet segments being misclassified as significant.
    
    Args:
        all_frame_energies: list of (frame_idx, residual_energy) for
                            every frame in decode order, including SKIPs.
        salient_thresh: floor for PEAK promotion.
        order: frames on each side that must be lower to qualify as
               a local maximum. Default 3.
    
    Returns:
        Set of frame_idx values that qualify as PEAK.
    """
    indices = [fe[0] for fe in all_frame_energies]
    energies = np.array([fe[1] for fe in all_frame_energies])
    
    local_max_positions = argrelextrema(energies, np.greater, order=order)[0]
    
    peak_frame_ids = set()
    for pos in local_max_positions:
        if energies[pos] >= salient_thresh:
            peak_frame_ids.add(indices[pos])
    
    return peak_frame_ids

def parse_video(video_path: str, return_stats: bool = False, candidate_thresh: float = 0.08, salient_thresh: float = 0.35, adaptive: bool = True):
    """
    Parses an H.264 video stream using PyAV and numpy without full RGB decoding.
    Returns a list of dicts for salient frames (I_FRAME, SALIENT, CANDIDATE) only.
    
    If return_stats is True, returns a tuple: (output_frames, stats_dict).
    """
    # Pass 1: Collect luma-diff energies of P/B frames
    container = av.open(video_path)
    if not container.streams.video:
        container.close()
        raise ValueError("No video stream found in the container.")
    
    all_frame_energies: list[tuple[int, float]] = []  # (frame_idx, residual_energy)
    energies = []  # existing list, kept for threshold calculation
    total_frames_pass1 = 0
    prev_Y_pass1 = None
    
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format='yuv420p')
        current_Y_pass1 = arr[0:frame.height, :]
        
        if total_frames_pass1 == 0:
            residual_energy = 1.0
        else:
            if prev_Y_pass1 is not None:
                diff = np.abs(current_Y_pass1.astype(float) - prev_Y_pass1.astype(float))
                residual_energy = float(np.mean(diff) / 255.0)
            else:
                residual_energy = 1.0
                
        prev_Y_pass1 = current_Y_pass1.copy()
        
        all_frame_energies.append((total_frames_pass1, residual_energy))
        if not frame.key_frame:
            energies.append(residual_energy)
            
        total_frames_pass1 += 1
        
    container.close()
    
    if adaptive:
        if energies:
            salient_thresh = float(np.percentile(energies, 97))
            candidate_thresh = float(np.percentile(energies, 90))

    peak_frame_ids = detect_peaks(all_frame_energies, salient_thresh=salient_thresh)

    # Second pass (or only pass)
    container = av.open(video_path)
    
    # Locate the first video stream
    if not container.streams.video:
        container.close()
        raise ValueError("No video stream found in the container.")
        
    stream = container.streams.video[0]
    
    # Export motion vectors (must set before container.decode starts)
    stream.codec_context.options = {"flags2": "+export_mvs"}
    
    output_frames = []
    
    total_frames = 0
    i_frame_count = 0
    peak_count = 0
    salient_count = 0
    candidate_count = 0
    skipped_count = 0
    
    prev_Y = None
    
    for frame in container.decode(video=0):
        # Y plane extraction
        # yuv420p layout has the Y channel as the first height rows of the numpy array
        arr = frame.to_ndarray(format='yuv420p')
        current_Y = arr[0:frame.height, :]
        
        if total_frames == 0:
            residual_energy = 1.0
        else:
            if prev_Y is not None:
                # Cast to float to prevent underflow before calculating mean absolute diff
                diff = np.abs(current_Y.astype(float) - prev_Y.astype(float))
                residual_energy = float(np.mean(diff) / 255.0)
            else:
                residual_energy = 1.0
        
        # Keep a copy of current Y for the next iteration's residual proxy calculation
        prev_Y = current_Y.copy()
        
        # Tier classification
        if frame.key_frame:
            tier = "I_FRAME"
        elif total_frames in peak_frame_ids:
            tier = "PEAK"   # local maximum above salient_thresh — overrides threshold tier
        elif residual_energy > salient_thresh:
            tier = "SALIENT"
        elif residual_energy >= candidate_thresh:
            tier = "CANDIDATE"
        else:
            tier = "SKIP"
                
        # Update statistics counters
        if tier == "I_FRAME":
            i_frame_count += 1
        elif tier == "PEAK":
            peak_count += 1
        elif tier == "SALIENT":
            salient_count += 1
        elif tier == "CANDIDATE":
            candidate_count += 1
        elif tier == "SKIP":
            skipped_count += 1
            
        # Extract motion vectors (empty for I-frames or if not exported/available)
        if tier == "I_FRAME":
            motion_vectors = []
        else:
            motion_vectors = []
            try:
                for sd in frame.side_data:
                    if getattr(sd.type, 'name', None) == 'MOTION_VECTORS':
                        for mv in sd:
                            motion_vectors.append((
                                int(mv.src_x),
                                int(mv.src_y),
                                int(mv.dst_x),
                                int(mv.dst_y),
                                int(mv.motion_x),
                                int(mv.motion_y)
                            ))
            except (AttributeError, TypeError):
                motion_vectors = []
                
        # Calculate robust timestamp
        if frame.time is not None:
            timestamp = float(frame.time)
        elif frame.pts is not None and stream.time_base is not None:
            timestamp = float(frame.pts * stream.time_base)
        else:
            timestamp = 0.0

        # Append to output only if the frame tier is not SKIP
        if tier != "SKIP":
            output_frames.append({
                "frame_idx": total_frames,
                "timestamp": timestamp,
                "tier": tier,
                "residual_energy": residual_energy,
                "motion_vectors": motion_vectors
            })
            
        total_frames += 1
        
    container.close()
    
    stats = {
        "total": total_frames,
        "i_frames": i_frame_count,
        "peaks": peak_count,
        "salient": salient_count,
        "candidate": candidate_count,
        "skipped": skipped_count,
        "salient_thresh_used": salient_thresh,
        "candidate_thresh_used": candidate_thresh
    }
    
    if return_stats:
        return output_frames, stats
    return output_frames

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse H.264 video streams.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    parser.add_argument("--no-adaptive", action="store_true", help="Force hardcoded thresholds instead of adaptive ones.")
    
    args = parser.parse_args()
    
    try:
        output_frames, stats = parse_video(args.video_path, return_stats=True, adaptive=not args.no_adaptive)
    except Exception as e:
        print(f"Error parsing video: {e}")
        sys.exit(1)
        
    total = stats["total"]
    def pct(count):
        return (count / total * 100) if total > 0 else 0.0
        
    print(f"Total frames: {total}")
    print(f"I-frames: {stats['i_frames']} ({pct(stats['i_frames']):.1f}%)")
    print(f"Peaks: {stats['peaks']} ({pct(stats['peaks']):.1f}%)")
    print(f"Salient: {stats['salient']} ({pct(stats['salient']):.1f}%)")
    print(f"Candidate: {stats['candidate']} ({pct(stats['candidate']):.1f}%)")
    print(f"Skipped: {stats['skipped']} ({pct(stats['skipped']):.1f}%)")
    print(f"Salient threshold used: {stats['salient_thresh_used']:.4f}")
    print(f"Candidate threshold used: {stats['candidate_thresh_used']:.4f}")
    print(f"Output entries: {len(output_frames)}")
    
    print("\nFirst 5 output entries:")
    for entry in output_frames[:5]:
        pprint.pprint(entry)
