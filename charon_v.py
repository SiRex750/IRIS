import av
import numpy as np
import sys
import pprint
from scipy.signal import argrelextrema

PEAK_WINDOW_SECONDS = 0.5


def get_stream_fps(video_path: str) -> float:
    """Zero-decode: read the video stream's average frame rate from container
    metadata (no packet demux, no pixel decode)."""
    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0
    finally:
        container.close()
    return fps

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
        all_frame_energies: list of (frame_idx, luma_diff_energy) for
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

def parse_video(video_path: str, return_stats: bool = False, return_raw: bool = False, candidate_thresh: float = 0.08, salient_thresh: float = 0.35, adaptive: bool = True, visual_debug_mode: bool = False, peak_order: int | None = None, full_decode: bool = False):
    """
    Parses an H.264 video stream using PyAV and numpy without full RGB decoding.
    Returns a list of dicts for salient frames (I_FRAME, SALIENT, CANDIDATE) only.
    
    If return_stats is True, returns a tuple: (output_frames, stats_dict).
    If return_raw is True, additionally returns a list of raw per-frame records.
    """
    # Pass 1: Build codec saliency curve from packet sizes (zero decode).
    all_frame_energies, iframe_indices, energies = _demux_packet_curve(video_path)
    
    scene_thresholds = {}
    scene_salient_vals = []
    scene_candidate_vals = []
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
        
        for start_idx, end_idx in scenes:
            scene_energies = [
                energy for idx, energy in all_frame_energies
                if start_idx <= idx < end_idx and idx not in iframe_indices
            ]
            if scene_energies:
                salient = max(1.0, float(np.percentile(scene_energies, 95)))
                candidate = max(1.0, float(np.percentile(scene_energies, 90)))
            else:
                if energies:
                    salient = max(1.0, float(np.percentile(energies, 95)))
                    candidate = max(1.0, float(np.percentile(energies, 90)))
                else:
                    salient = 0.0
                    candidate = 0.0
            scene_thresholds[(start_idx, end_idx)] = (salient, candidate)
            scene_salient_vals.append(salient)
            scene_candidate_vals.append(candidate)
            
        if scene_salient_vals:
            salient_thresh = float(np.median(scene_salient_vals))
            candidate_thresh = float(np.median(scene_candidate_vals))
            
        # Exclude I-frame entries so their ~10× larger packet sizes do not warp
        # argrelextrema for neighboring P/B frames.
        iframe_set = set(iframe_indices)
        # detect_peaks call deferred to Pass 2 where stream.average_rate gives fps.
    else:
        # Non-adaptive mode: candidate_thresh/salient_thresh are passed through verbatim.
        # With Phase-4 packet-size gate they must be byte-scale (e.g. 0.0 = keep all).
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

    fps = float(stream.average_rate) if stream.average_rate else 25.0
    effective_order = peak_order if peak_order is not None else max(3, round(PEAK_WINDOW_SECONDS * fps))
    if adaptive:
        peak_frame_ids = detect_peaks(
            [(idx, size) for idx, size in all_frame_energies if idx not in iframe_set],
            salient_thresh={k: v[0] for k, v in scene_thresholds.items()},
            order=effective_order,
        )

    packet_size_by_idx = {idx: size for idx, size in all_frame_energies}
    output_frames = []
    raw_records = []
    
    total_frames = 0
    i_frame_count = 0
    peak_count = 0
    salient_count = 0
    candidate_count = 0
    skipped_count = 0
    
    prev_processed_Y = None
    expensive_processed = 0

    try:
        for frame in container.decode(video=0):
            # ── TIER CLASSIFICATION (metadata only, no pixel access) ──────────
            ps = packet_size_by_idx.get(total_frames, 0.0)

            curr_salient = salient_thresh
            curr_candidate = candidate_thresh
            if adaptive:
                for (start, end), (s_thresh, c_thresh) in scene_thresholds.items():
                    if start <= total_frames < end:
                        curr_salient = s_thresh
                        curr_candidate = c_thresh
                        break

            # LEGACY / UNUSED: tier kept for backward compatibility with Track A/C.
            # Track B uses continuous action scoring (action_score.py).
            if frame.key_frame:
                tier = "I_FRAME"
            elif total_frames in peak_frame_ids:
                tier = "PEAK"
            elif ps > curr_salient:
                tier = "SALIENT"
            elif ps >= curr_candidate:
                tier = "CANDIDATE"
            else:
                tier = "SKIP"

            if tier == "I_FRAME":
                i_frame_count += 1
            elif tier == "PEAK":
                peak_count += 1
            elif tier == "SALIENT":
                salient_count += 1
            elif tier == "CANDIDATE":
                candidate_count += 1
            else:
                skipped_count += 1

            # Timestamp from metadata (no pixel access required)
            if frame.time is not None:
                timestamp = float(frame.time)
            elif frame.pts is not None and stream.time_base is not None:
                timestamp = float(frame.pts * stream.time_base)
            else:
                timestamp = 0.0

            process = full_decode or (tier != "SKIP")

            if process:
                expensive_processed += 1

                # Y plane extraction
                arr = frame.to_ndarray(format='yuv420p')
                Y = arr[0:frame.height, :]

                # Luma diff vs previous retained frame.
                # In selective mode this is a gap-diff; in full_decode mode it is
                # an adjacent-frame diff.  The difference is intentional.
                if prev_processed_Y is None:
                    luma_diff_energy = 0.0
                else:
                    luma_diff_energy = float(
                        np.mean(np.abs(Y.astype(float) - prev_processed_Y.astype(float))) / 255.0
                    )
                prev_processed_Y = Y.copy()

                # Luma entropy from Y-plane histogram
                try:
                    hist, _ = np.histogram(Y, bins=256, range=(0, 256), density=True)
                    hist = hist[hist > 0]
                    luma_entropy = float(-np.sum(hist * np.log2(hist))) / 8.0
                except Exception:
                    luma_entropy = 0.0

                # Motion vector extraction (export_mvs already set on codec context)
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

                mvs_mags = [np.sqrt(mv[4]**2 + mv[5]**2) for mv in motion_vectors]
                motion_magnitude = float(np.mean(mvs_mags)) if mvs_mags else 0.0

                # Append survivor to output_frames
                if tier != "SKIP":
                    geom = compute_motion_geometry(motion_vectors, frame.width, frame.height)
                    # Capture PIL image so pipeline.py can extract CLIP embeddings
                    # without opening the video a third time.
                    try:
                        pil_image = frame.to_image()
                    except Exception:
                        pil_image = None
                    output_frames.append({
                        "frame_idx": total_frames,
                        "timestamp": timestamp,
                        "tier": tier,
                        "luma_diff_energy": luma_diff_energy,
                        "packet_size": ps,
                        "motion_vectors": motion_vectors,
                        "pil_image": pil_image,
                        **geom
                    })

                # Full raw_record
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

                    rec = {
                        "frame_idx": total_frames,
                        "timestamp": timestamp,
                        "frame_type": frame_type,
                        "luma_diff_energy": luma_diff_energy,
                        "packet_size": ps,
                        "motion_magnitude": motion_magnitude,
                        "luma_entropy": luma_entropy
                    }
                    if visual_debug_mode and tier != "SKIP":
                        rec["frame"] = frame.to_ndarray(format='bgr24')
                    raw_records.append(rec)

            else:
                # SKIP frame in selective mode — no pixel work
                if return_raw:
                    raw_records.append({
                        "frame_idx": total_frames,
                        "timestamp": timestamp,
                        "packet_size": ps,
                        "luma_diff_energy": 0.0,
                        "motion_magnitude": 0.0,
                        "luma_entropy": 0.0,
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
        "candidate_thresh_used": candidate_thresh,
        "salient_thresh_per_scene": {k: v[0] for k, v in scene_thresholds.items()},
        "candidate_thresh_per_scene": {k: v[1] for k, v in scene_thresholds.items()},
        "num_scenes": len(scene_thresholds),
        "peak_order_used": effective_order,
        "frames_expensive_processed": expensive_processed,
    }
    
    if return_raw:
        if return_stats:
            return output_frames, stats, raw_records
        return output_frames, raw_records
        
    if return_stats:
        return output_frames, stats
    return output_frames


def _demux_packet_curve(
    video_path: str,
) -> tuple[list[tuple[int, float]], list[int], list[float]]:
    """
    Zero-decode codec saliency curve for Phase-4 Tier 0.

    Demuxes coded packet sizes (no pixel decode), re-sorts to DISPLAY order by pts,
    and returns the same three structures parse_video Pass 1 currently builds from
    luma-diff energies — so the gate can swap signals with a minimal diff.

    Returns:
      all_frame_energies: list of (display_idx, packet_size_bytes) for EVERY frame,
                          display_idx assigned 0..N-1 in pts-ascending order.
      iframe_indices:     display_idx values whose packet is a keyframe.
      energies:           packet_size_bytes for NON-keyframe frames only
                          (the pool used for percentile thresholds).
    Packet sizes are returned RAW (bytes, as float). No normalization.
    """
    container = av.open(video_path)
    if not container.streams.video:
        container.close()
        raise ValueError("No video stream found in the container.")

    stream = container.streams.video[0]
    raw: list[tuple[int | None, float, bool]] = []

    try:
        for pkt in container.demux(stream):
            if pkt.size == 0:  # flush packet
                continue
            raw.append((pkt.pts, float(pkt.size), bool(pkt.is_keyframe)))
    finally:
        container.close()

    # Re-sort to display order by pts; fall back to demux order on None pts.
    has_none_pts = any(p[0] is None for p in raw)
    if has_none_pts:
        sorted_raw = raw  # stable demux-order fallback
    else:
        sorted_raw = sorted(raw, key=lambda p: p[0])

    all_frame_energies: list[tuple[int, float]] = []
    iframe_indices: list[int] = []
    energies: list[float] = []

    for display_idx, (_, size, is_kf) in enumerate(sorted_raw):
        all_frame_energies.append((display_idx, size))
        if is_kf:
            iframe_indices.append(display_idx)
        else:
            energies.append(size)

    return all_frame_energies, iframe_indices, energies


def compute_valley_scene_boundaries(
    all_frame_energies: list[tuple[int, float]],
    iframe_indices: list[int],
    fps: float,
    valley_percentile: float = 25.0,
) -> list[tuple[int, int]]:
    """
    Zero-decode scene boundaries from the packet-size curve: a boundary is a
    local minimum of the non-keyframe packet curve below valley_percentile of
    that curve. Pure function of the curve (and the stated percentile/fps) —
    survivor-independent, deterministic, no size cap (callers apply the cap
    downstream since it depends on survivor population).

    fps: real stream average_rate (see get_stream_fps), used only to size the
    argrelextrema order window (mirrors detect_peaks' PEAK_WINDOW_SECONDS
    window) — not read from the curve itself.

    Returns (start_idx, end_idx) spans in display order covering [0, N).
    """
    n_frames = len(all_frame_energies)
    if n_frames == 0:
        return []

    iframe_set = set(iframe_indices)
    non_kf = [(idx, size) for idx, size in all_frame_energies if idx not in iframe_set]

    if not non_kf:
        return [(0, n_frames)]

    indices = np.array([idx for idx, _ in non_kf])
    sizes = np.array([size for _, size in non_kf])

    order = max(3, round(PEAK_WINDOW_SECONDS * fps))
    local_min_positions = argrelextrema(sizes, np.less, order=order)[0]
    thresh = np.percentile(sizes, valley_percentile)

    boundaries = sorted(
        int(indices[pos]) for pos in local_min_positions if sizes[pos] < thresh
    )

    ends = sorted(set([0] + boundaries + [n_frames]))
    scenes: list[tuple[int, int]] = []
    for i in range(len(ends) - 1):
        start, end = ends[i], ends[i + 1]
        if end > start:
            scenes.append((start, end))
    return scenes


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
