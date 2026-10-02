"""Dump the packet-size curve, valley scene boundaries and admitted frames for one video.

Data source for the paper's segmentation figure (paper/latex/iris_mmsys27.tex, Section 3).
Read-only: it calls exactly the functions ingest calls, at production defaults, and writes one
JSON file. It decodes no pixels, loads no model, and touches no cache or result file.

  charon_v._demux_packet_curve       per-frame packet sizes (display order) and I-frame indices
  charon_v.get_stream_fps            stream frame rate (sizes the local-minimum window)
  charon_v.compute_valley_scene_boundaries   scene spans, as ingest uses them
  charon_v.parse_video               admitted frames (production thresholds, adaptive on)

The valley threshold and the raw valley list (before the 300-frame scene cap) are recomputed
here with the same formula as compute_valley_scene_boundaries, purely so the figure can mark
them; the scene spans reported are the function's own output.

Usage:
  python scripts/fig_valley_dump.py --video <path to .mp4> --out eval_results/fig_valley_<name>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import argrelextrema

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from iris import charon_v  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    video = str(args.video)

    all_frame_energies, iframe_indices, _, _ = charon_v._demux_packet_curve(video)
    fps = charon_v.get_stream_fps(video)
    scenes = charon_v.compute_valley_scene_boundaries(all_frame_energies, iframe_indices, fps)

    # Same formula as compute_valley_scene_boundaries, recomputed only to annotate the figure.
    iframe_set = set(iframe_indices)
    non_kf = [(idx, size) for idx, size in all_frame_energies if idx not in iframe_set]
    idx_arr = np.array([i for i, _ in non_kf])
    size_arr = np.array([s for _, s in non_kf])
    order = max(3, round(charon_v.PEAK_WINDOW_SECONDS * fps))
    local_min = argrelextrema(size_arr, np.less, order=order)[0]
    valley_threshold = float(np.percentile(size_arr, 25.0))
    valleys = sorted(int(idx_arr[p]) for p in local_min if size_arr[p] < valley_threshold)

    output_frames, stats = charon_v.parse_video(video, return_stats=True)
    admitted = [{"frame_idx": int(f["frame_idx"]), "tier": str(f.get("tier", ""))} for f in output_frames]

    result = {
        "video": args.video.name,
        "fps": float(fps),
        "n_frames": len(all_frame_energies),
        "packet_bytes": [[int(i), float(s)] for i, s in all_frame_energies],
        "iframe_indices": sorted(int(i) for i in iframe_indices),
        "valley_window_order": int(order),
        "valley_threshold_bytes": valley_threshold,
        "valleys_before_cap": valleys,
        "scene_spans": [[int(a), int(b)] for a, b in scenes],
        "admitted": admitted,
        "parse_stats": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in dict(stats).items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result))
    print(f"{args.video.name}: {result['n_frames']} frames, {len(iframe_indices)} I-frames, "
          f"{len(valleys)} valleys, {len(scenes)} scenes, {len(admitted)} admitted -> {args.out}")


if __name__ == "__main__":
    main()
