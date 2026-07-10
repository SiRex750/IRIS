"""
Phase-4 curve sanity tool.

Compares the codec packet-size curve against the legacy luma-diff curve to
establish a correctness tripwire before Phase 4 changes the gate signal.

Usage:
    python scripts/phase4_curve_sanity.py [video_path]
"""
from __future__ import annotations

import os
import socket
import sys
import urllib.request

import av
import numpy as np
from scipy.stats import spearmanr


def build_curves(video_path: str) -> dict:
    """
    Build and compare the luma-diff curve (legacy) and the packet-size curve
    (codec) for the given video, re-sorted to display order.

    Returns a dict containing the raw curves, alignment stats, correlation
    metrics over non-I frames, and survivor-fraction sanity numbers.
    """

    # ── 1. LUMA-DIFF CURVE (display/PTS order via decode) ─────────────────
    luma_curve: list[tuple[float, bool]] = []  # (luma_diff, is_keyframe)
    container = av.open(video_path)
    prev_Y = None
    try:
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="yuv420p")
            Y = arr[0 : frame.height, :]
            if len(luma_curve) == 0:
                luma_diff = 0.0
            else:
                luma_diff = float(
                    np.mean(np.abs(Y.astype(float) - prev_Y.astype(float))) / 255.0
                )
            prev_Y = Y.copy()
            luma_curve.append((luma_diff, bool(frame.key_frame)))
    finally:
        container.close()

    # ── 2. PACKET-SIZE CURVE, re-sorted to display order via PTS ──────────
    raw_packets: list[tuple[int | None, int, bool]] = []
    container = av.open(video_path)
    stream = container.streams.video[0]
    try:
        for pkt in container.demux(stream):
            if pkt.size == 0:  # flush packet — skip
                continue
            raw_packets.append((pkt.pts, pkt.size, bool(pkt.is_keyframe)))
    finally:
        container.close()

    has_none_pts = any(p[0] is None for p in raw_packets)
    if has_none_pts:
        print(
            "WARNING: one or more packets have pts=None; pts-resort to display order is "
            "impossible for this stream. codec_validator will reject this case in a later "
            "step. Falling back to demux order."
        )
        packet_curve = raw_packets  # keep demux order as-is
    else:
        packet_curve = sorted(raw_packets, key=lambda p: p[0])

    # ── 3. ALIGNMENT CHECK ─────────────────────────────────────────────────
    assert len(luma_curve) == len(packet_curve), (
        f"Frame count mismatch: {len(luma_curve)} decoded frames vs "
        f"{len(packet_curve)} non-flush packets"
    )
    N = len(luma_curve)
    agree = sum(
        1 for i in range(N) if luma_curve[i][1] == packet_curve[i][2]
    )
    print(f"Keyframe alignment: {agree}/{N} positions agree between pts-sorted packets and decoded frames")

    # ── 4. METRICS over non-I frames only ─────────────────────────────────
    non_i_pkt  = np.array([packet_curve[i][1]  for i in range(N) if not packet_curve[i][2]], dtype=float)
    non_i_luma = np.array([luma_curve[i][0]    for i in range(N) if not packet_curve[i][2]], dtype=float)
    i_pkt      = np.array([packet_curve[i][1]  for i in range(N) if     packet_curve[i][2]], dtype=float)
    i_luma     = np.array([luma_curve[i][0]    for i in range(N) if     packet_curve[i][2]], dtype=float)

    rho, rho_p = spearmanr(non_i_pkt, non_i_luma)
    pearson_r  = float(np.corrcoef(non_i_pkt, non_i_luma)[0, 1])

    mean_pkt_i    = float(np.mean(i_pkt))    if len(i_pkt)    > 0 else float("nan")
    mean_pkt_noni = float(np.mean(non_i_pkt)) if len(non_i_pkt) > 0 else float("nan")
    mean_luma_i   = float(np.mean(i_luma))   if len(i_luma)   > 0 else float("nan")
    mean_luma_noni= float(np.mean(non_i_luma))if len(non_i_luma)> 0 else float("nan")

    print(f"Spearman rho (non-I): {rho:.4f}  (p={rho_p:.4e})")
    print(f"Pearson  r   (non-I): {pearson_r:.4f}")
    print(f"Mean packet size  — I-frames: {mean_pkt_i:.1f}  non-I: {mean_pkt_noni:.1f}")
    print(f"Mean luma-diff    — I-frames: {mean_luma_i:.5f}  non-I: {mean_luma_noni:.5f}")

    # ── 5. SURVIVOR-FRACTION sanity (mirrors candidate/salient logic) ──────
    p90 = float(np.percentile(non_i_pkt, 90))
    p95 = float(np.percentile(non_i_pkt, 95))

    n_above_p90 = int(np.sum(non_i_pkt > p90))
    n_above_p95 = int(np.sum(non_i_pkt > p95))
    frac_p90 = n_above_p90 / N
    frac_p95 = n_above_p95 / N

    print(f"Non-I pkt p90={p90:.0f}  survivors={n_above_p90}  fraction of total={frac_p90:.4f}")
    print(f"Non-I pkt p95={p95:.0f}  survivors={n_above_p95}  fraction of total={frac_p95:.4f}")

    return {
        "luma_curve":       luma_curve,
        "packet_curve":     packet_curve,
        "alignment_agree":  agree,
        "alignment_total":  N,
        "spearman_rho":     float(rho),
        "spearman_p":       float(rho_p),
        "pearson_r":        pearson_r,
        "mean_pkt_i":       mean_pkt_i,
        "mean_pkt_noni":    mean_pkt_noni,
        "mean_luma_i":      mean_luma_i,
        "mean_luma_noni":   mean_luma_noni,
        "p90_pkt_noni":     p90,
        "p95_pkt_noni":     p95,
        "n_above_p90":      n_above_p90,
        "n_above_p95":      n_above_p95,
        "frac_above_p90":   frac_p90,
        "frac_above_p95":   frac_p95,
    }


def _resolve_video() -> str | None:
    """Return a usable video path, or None if unavailable."""
    local = "mov_bbb.mp4"
    if os.path.exists(local):
        return local
    url = "https://www.w3schools.com/html/mov_bbb.mp4"
    print(f"Downloading {url} ...")
    socket.setdefaulttimeout(5.0)
    try:
        urllib.request.urlretrieve(url, local)
        print("Download complete.")
        return local
    except Exception as exc:
        print(f"Download failed: {exc}")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = _resolve_video()
        if video_path is None:
            print("SKIP: test video unavailable and download failed — nothing to do.")
            sys.exit(0)

    print(f"\n=== Phase-4 curve sanity: {video_path} ===\n")
    result = build_curves(video_path)
    print("\n--- Summary dict (scalar fields) ---")
    for k, v in result.items():
        if not isinstance(v, list):
            print(f"  {k}: {v}")
