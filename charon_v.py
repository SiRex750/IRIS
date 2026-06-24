import av
import numpy as np
import sys
import pprint

def parse_video(
    video_path: str,
    return_stats: bool = False,
    return_raw: bool = False,
    candidate_thresh: float = 0.08,
    salient_thresh: float = 0.35,
    adaptive: bool = True,
    visual_debug_mode: bool = False
):
    """
    Parses an H.264 video stream using PyAV and numpy in a single pass.
    Acts purely as a raw signal extractor, returning signals for ALL decoded frames.
    
    Returns:
        If return_raw is True and return_stats is True: (records, stats, records)
        If return_raw is True and return_stats is False: (records, records)
        If return_raw is False and return_stats is True: (records, stats)
        Otherwise: records
    """
    container = av.open(video_path)
    if not container.streams.video:
        container.close()
        raise ValueError("No video stream found in the container.")
        
    stream = container.streams.video[0]
    # Export motion vectors (must set before container.decode starts)
    stream.codec_context.options = {"flags2": "+export_mvs"}
    
    records = []
    total_frames = 0
    i_frame_count = 0
    prev_Y = None
    
    try:
        for frame in container.decode(video=0):
            # Y plane extraction
            arr = frame.to_ndarray(format='yuv420p')
            current_Y = arr[0:frame.height, :]
            
            # Calculate residual energy
            if total_frames == 0:
                residual_energy = 1.0
            else:
                if prev_Y is not None:
                    diff = np.abs(current_Y.astype(float) - prev_Y.astype(float))
                    residual_energy = float(np.mean(diff) / 255.0)
                else:
                    residual_energy = 1.0
            
            prev_Y = current_Y.copy()
            
            # Count I-frames
            is_iframe = frame.key_frame
            if is_iframe:
                i_frame_count += 1
                pict_type = "I"
            else:
                pict_type = "P"  # default placeholder
                
            # Extract motion vectors
            motion_vectors = []
            if not is_iframe:
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
                    
            # Calculate motion magnitude
            mvs_mags = [np.sqrt(mv[4]**2 + mv[5]**2) for mv in motion_vectors]
            motion_magnitude = float(np.mean(mvs_mags)) if mvs_mags else 0.0
            
            # Calculate entropy
            try:
                hist, _ = np.histogram(current_Y, bins=256, range=(0, 256), density=True)
                hist = hist[hist > 0]
                entropy = float(-np.sum(hist * np.log2(hist))) / 8.0
            except Exception:
                entropy = 0.0
                
            # Calculate robust timestamp
            if frame.time is not None:
                timestamp = float(frame.time)
            elif frame.pts is not None and stream.time_base is not None:
                timestamp = float(frame.pts * stream.time_base)
            else:
                timestamp = 0.0
                
            # Generate small base64 thumbnail for frontend visibility
            thumbnail = ""
            try:
                img = frame.to_image()
                img.thumbnail((160, 90))
                import io, base64
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=40)
                thumbnail = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                pass

            # Build unified frame record containing both raw signals and metadata
            frame_rec = {
                "frame_idx": total_frames,
                "timestamp": timestamp,
                "residual_energy": residual_energy,
                "motion_magnitude": motion_magnitude,
                "entropy": entropy,
                "motion_vectors": motion_vectors,
                "frame_type": pict_type,
                "thumbnail": thumbnail
            }
            if visual_debug_mode:
                frame_rec["frame"] = frame.to_ndarray(format='bgr24')
                
            records.append(frame_rec)
            total_frames += 1
    finally:
        container.close()
        
    stats = {
        "total": total_frames,
        "i_frames": i_frame_count,
        # Keep keys for backward compatibility but set to 0 as they are deprecated
        "peaks": 0,
        "salient": 0,
        "candidate": 0,
        "skipped": 0,
        "salient_thresh_used": salient_thresh,
        "candidate_thresh_used": candidate_thresh
    }
    
    if return_raw:
        if return_stats:
            return records, stats, records
        return records, records
        
    if return_stats:
        return records, stats
    return records

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract raw signals from H.264 video streams.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    
    args = parser.parse_args()
    
    try:
        output_frames, stats = parse_video(args.video_path, return_stats=True)
    except Exception as e:
        print(f"Error parsing video: {e}")
        sys.exit(1)
        
    print(f"Total frames processed: {stats['total']}")
    print(f"I-frames: {stats['i_frames']}")
    print(f"Output entries: {len(output_frames)}")
    
    print("\nFirst 5 output entries:")
    for entry in output_frames[:5]:
        # Exclude raw motion vectors from quick print for clean output
        print({k: v for k, v in entry.items() if k != "motion_vectors" and k != "frame"})
