"""Survivor-count measurement at real CCTV scale (selection only — no edits
under iris/).

Runs Charon-V's production gate (parse_video, full_decode=False) on a
15-minute, ~1.5GB VIRAT clip to see how many frames survive the salience
gate at realistic scale, then projects (does NOT run) what a flat,
un-gated build would cost at that N:
  - edges = n_surv*(n_surv-1)/2        (fully-connected O(N^2) graph)
  - projected_caption_sec = n_surv * 1.38   (measured in-build caption rate,
    ~0.855s/frame from the 220-frame run, roughed up for real-model variance)

Deliberately does NOT embed, caption, or build the graph — enrichment over
thousands of survivors would run for hours. Selection + estimates only.

Usage: python phase6_survivor_scale.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import iris.charon_v as charon_v
from iris.iris_config import IRISConfig

VIDEO_PATH = Path.home() / "Downloads" / "VIRAT_S_000102.mp4"

CAPTION_SEC_PER_FRAME = 1.38  # measured in-build caption rate (see amort_units.json)


def _probe_container_meta(vpath: Path) -> tuple[float, float]:
    """Cheap metadata-only probe (no frame decode): duration_sec, fps."""
    import av

    container = av.open(str(vpath))
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0
        if stream.duration is not None and stream.time_base is not None:
            duration_sec = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_sec = float(container.duration) / 1_000_000.0  # av.time_base = 1e6
        else:
            duration_sec = 0.0
    finally:
        container.close()
    return duration_sec, fps


def main() -> None:
    if not VIDEO_PATH.exists():
        print(f"ERROR: {VIDEO_PATH} not found")
        sys.exit(1)

    cfg = IRISConfig(
        ranking_mode="ppr",
        codec_conf_source="packet_size",
        codec_conf_pictype_norm=True,
        ppr_lambda=0.5,
        ppr_damping=0.5,
        l2_retrieve_top_k=8,
    )

    duration_sec, fps = _probe_container_meta(VIDEO_PATH)

    print(f"Running Charon-V parse_video (full_decode=False) on {VIDEO_PATH.name} ...")
    print("SELECTION ONLY — no embed, no caption, no graph build.")
    sys.stdout.flush()

    t0 = time.perf_counter()
    output_frames, stats = charon_v.parse_video(
        str(VIDEO_PATH),
        return_stats=True,
        return_raw=False,
        candidate_thresh=cfg.candidate_thresh,
        salient_thresh=cfg.salient_thresh,
        adaptive=getattr(cfg, "adaptive", True),
        visual_debug_mode=getattr(cfg, "visual_debug_mode", False),
        full_decode=False,
    )
    parse_wall_sec = time.perf_counter() - t0

    total_frame_count = int(stats["total"])
    n_surv = len(output_frames)
    survivor_ratio = n_surv / total_frame_count if total_frame_count else 0.0

    edges = n_surv * (n_surv - 1) // 2
    projected_caption_sec = n_surv * CAPTION_SEC_PER_FRAME

    print("\n===SURVIVOR_SCALE===")
    print(f"total_frame_count={total_frame_count}")
    print(f"duration_sec={duration_sec:.4f}")
    print(f"fps={fps:.4f}")
    print(f"n_surv={n_surv}")
    print(f"survivor_ratio={survivor_ratio:.6f}")
    print(f"(parse_video wall time: {parse_wall_sec:.2f}s)")

    print("\n===FLAT_BUILD_PROJECTION (not run) ===")
    print(f"edges (N^2, fully-connected) = n_surv*(n_surv-1)/2 = {edges:,}")
    print(
        f"projected_caption_sec = n_surv * {CAPTION_SEC_PER_FRAME} = "
        f"{projected_caption_sec:,.1f} sec (~{projected_caption_sec / 3600:.2f} hours)"
    )
    print("\nWhat a flat build would cost at this scale:")
    print(f"  - {edges:,} edges in a fully-connected graph")
    print(
        f"  - ~{projected_caption_sec:,.0f}s (~{projected_caption_sec / 3600:.2f}h) "
        f"of captioning alone, before embedding or graph construction"
    )


if __name__ == "__main__":
    main()
