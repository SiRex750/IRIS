import av
import numpy as np
import sys
import pprint
from scipy.signal import argrelextrema

def compute_motion_geometry(motion_vectors: list, width: int, height: int) -> dict:
    """
    Computes physical motion geometry descriptors from raw H.264 motion vectors.
    """
    if not motion_vectors or width <= 0 or height <= 0:
        return {
            "divergence": 0.0,
            "curl": 0.0,
            "jacobian_frobenius": 0.0,
            "hessian_max_eigenvalue": 0.0,
            "motion_entropy": 0.0
        }
        
    grid_w = max(1, width // 16)
    grid_h = max(1, height // 16)
    
    U = np.zeros((grid_h, grid_w), dtype=np.float32)
    V = np.zeros((grid_h, grid_w), dtype=np.float32)
    counts = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    for mv in motion_vectors:
        # mv: (src_x, src_y, dst_x, dst_y, motion_x, motion_y)
        gx = min(max(0, mv[2] // 16), grid_w - 1)
        gy = min(max(0, mv[3] // 16), grid_h - 1)
        U[gy, gx] += mv[4]
        V[gy, gx] += mv[5]
        counts[gy, gx] += 1.0
        
    mask = counts > 0
    U[mask] /= counts[mask]
    V[mask] /= counts[mask]
    
    if grid_h > 1 and grid_w > 1:
        U_y, U_x = np.gradient(U)
        V_y, V_x = np.gradient(V)
    else:
        U_y = np.zeros_like(U)
        U_x = np.zeros_like(U)
        V_y = np.zeros_like(V)
        V_x = np.zeros_like(V)
        
    div = U_x + V_y
    divergence = float(np.mean(div))
    
    rot = V_x - U_y
    curl = float(np.mean(np.abs(rot)))
    
    jac_norm = np.sqrt(U_x**2 + U_y**2 + V_x**2 + V_y**2)
    jacobian_frobenius = float(np.mean(jac_norm))
    
    M = np.sqrt(U**2 + V**2)
    if grid_h > 1 and grid_w > 1:
        M_y, M_x = np.gradient(M)
        M_yy, M_yx = np.gradient(M_y)
        M_xy, M_xx = np.gradient(M_x)
    else:
        M_xx = np.zeros_like(M)
        M_yy = np.zeros_like(M)
        M_xy = np.zeros_like(M)
        M_yx = np.zeros_like(M)
        
    trace = M_xx + M_yy
    diff = M_xx - M_yy
    det_term = np.sqrt(diff**2 + 4 * M_yx * M_xy)
    eigenvalue_field = 0.5 * (np.abs(trace) + det_term)
    hessian_max_eigenvalue = float(np.mean(eigenvalue_field))
    
    flat_mags = M.flatten()
    max_mag = np.max(flat_mags)
    if max_mag > 1e-5:
        hist, _ = np.histogram(flat_mags, bins=10, range=(0, max_mag), density=True)
        hist = hist[hist > 0]
        p = hist / np.sum(hist)
        motion_entropy = float(-np.sum(p * np.log2(p)))
    else:
        motion_entropy = 0.0
        
    return {
        "divergence": divergence,
        "curl": curl,
        "jacobian_frobenius": jacobian_frobenius,
        "hessian_max_eigenvalue": hessian_max_eigenvalue,
        "motion_entropy": motion_entropy
    }


def detect_peaks(
    all_frame_energies: list[tuple[int, float]],
    salient_thresh: float | dict[tuple[int, int], float],
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
        salient_thresh: floor for PEAK promotion (or a dict mapping scene ranges to floors).
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
        idx = indices[pos]
        energy = energies[pos]
        
        # Determine the threshold for this frame
        thresh = salient_thresh
        if isinstance(salient_thresh, dict):
            for (start, end), val in salient_thresh.items():
                if start <= idx < end:
                    thresh = val
                    break
            if isinstance(thresh, dict):  # safety check
                thresh = 0.35
                
        if energy >= thresh:
            peak_frame_ids.add(idx)
    
    return peak_frame_ids

def parse_video(video_path: str, return_stats: bool = False, return_raw: bool = False, candidate_thresh: float = 0.08, salient_thresh: float = 0.35, adaptive: bool = True):
    """
    Parses an H.264 video stream using PyAV and numpy without full RGB decoding.
    Returns a list of dicts for salient frames (I_FRAME, SALIENT, CANDIDATE) only.
    
    If return_stats is True, returns a tuple: (output_frames, stats_dict).
    If return_raw is True, additionally returns a list of raw per-frame records.
    """
    # Pass 1: Collect luma-diff energies of P/B frames
    container = av.open(video_path)
    if not container.streams.video:
        container.close()
        raise ValueError("No video stream found in the container.")
    
    all_frame_energies: list[tuple[int, float]] = []  # (frame_idx, residual_energy)
    iframe_indices = []
    energies = []  # existing list, kept for threshold calculation
    total_frames_pass1 = 0
    prev_Y_pass1 = None
    
    try:
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
            if frame.key_frame:
                iframe_indices.append(total_frames_pass1)
            else:
                energies.append(residual_energy)
                
            total_frames_pass1 += 1
    finally:
        container.close()
    
    scene_thresholds = {}
    if adaptive:
        num_frames_total = len(all_frame_energies)
        if not iframe_indices:
            iframe_indices = [0]
        elif iframe_indices[0] != 0:
            iframe_indices.insert(0, 0)
            
        scenes = []
        for i, start_idx in enumerate(iframe_indices):
            end_idx = iframe_indices[i+1] if i+1 < len(iframe_indices) else num_frames_total
            scenes.append((start_idx, end_idx))
            
        scene_salient_vals = []
        scene_candidate_vals = []
        
        for start_idx, end_idx in scenes:
            scene_energies = [
                energy for idx, energy in all_frame_energies
                if start_idx <= idx < end_idx and idx not in iframe_indices
            ]
            if scene_energies:
                salient = max(0.002, float(np.percentile(scene_energies, 95)))
                candidate = max(0.001, float(np.percentile(scene_energies, 90)))
            else:
                if energies:
                    salient = max(0.002, float(np.percentile(energies, 95)))
                    candidate = max(0.001, float(np.percentile(energies, 90)))
                else:
                    salient = 0.35
                    candidate = 0.08
            scene_thresholds[(start_idx, end_idx)] = (salient, candidate)
            scene_salient_vals.append(salient)
            scene_candidate_vals.append(candidate)
            
        if scene_salient_vals:
            salient_thresh = float(np.min(scene_salient_vals))
            candidate_thresh = float(np.min(scene_candidate_vals))
            
        peak_frame_ids = detect_peaks(
            all_frame_energies,
            salient_thresh={k: v[0] for k, v in scene_thresholds.items()},
        )
    else:
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
    raw_records = []
    
    total_frames = 0
    i_frame_count = 0
    peak_count = 0
    salient_count = 0
    candidate_count = 0
    skipped_count = 0
    
    prev_Y = None
    
    try:
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
            
            # Determine thresholds for the current frame
            curr_salient = salient_thresh
            curr_candidate = candidate_thresh
            if adaptive:
                for (start, end), (s_thresh, c_thresh) in scene_thresholds.items():
                    if start <= total_frames < end:
                        curr_salient = s_thresh
                        curr_candidate = c_thresh
                        break
            
            # Tier classification
            if frame.key_frame:
                tier = "I_FRAME"
            elif total_frames in peak_frame_ids:
                tier = "PEAK"   # local maximum overrides threshold tier
            elif residual_energy > curr_salient:
                tier = "SALIENT"
            elif residual_energy >= curr_candidate:
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
                geom = compute_motion_geometry(motion_vectors, frame.width, frame.height)
                # Capture PIL image here so pipeline.py can extract CLIP embeddings
                # without opening the video a 3rd time.  Guard with try/except so
                # any format issue doesn't drop the frame — it just falls back to
                # the legacy re-decode path in wrapper_l2_retrieve.
                try:
                    pil_image = frame.to_image()
                except Exception:
                    pil_image = None
                output_frames.append({
                    "frame_idx": total_frames,
                    "timestamp": timestamp,
                    "tier": tier,
                    "residual_energy": residual_energy,
                    "motion_vectors": motion_vectors,
                    "pil_image": pil_image,
                    **geom
                })

            # Expose raw records non-breakingly if requested
            if return_raw:
                pict_type = getattr(frame, "pict_type", None)
                if pict_type is None:
                    frame_type = "I" if frame.key_frame else "P"
                elif isinstance(pict_type, int):
                    try:
                        frame_type = av.video.frame.PictureType(pict_type).name
                    except Exception:
                        frame_type = "I" if frame.key_frame else "P"
                elif hasattr(pict_type, "name"):
                    frame_type = pict_type.name
                else:
                    frame_type = str(pict_type)

                mvs_mags = [np.sqrt(mv[4]**2 + mv[5]**2) for mv in motion_vectors]
                motion_magnitude = float(np.mean(mvs_mags)) if mvs_mags else 0.0
                try:
                    hist, _ = np.histogram(current_Y, bins=256, range=(0, 256), density=True)
                    hist = hist[hist > 0]
                    entropy = float(-np.sum(hist * np.log2(hist))) / 8.0
                except Exception:
                    entropy = 0.0
                
                raw_records.append({
                    "frame_idx": total_frames,
                    "frame_type": frame_type,
                    "residual_energy": residual_energy,
                    "motion_magnitude": motion_magnitude,
                    "entropy": entropy
                })
                
            total_frames += 1
    finally:
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
    
    if return_raw:
        if return_stats:
            return output_frames, stats, raw_records
        return output_frames, raw_records
        
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
