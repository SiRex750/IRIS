"""
Standalone, zero-decode scene-distribution calibration on a real clip.

Answers: how many valley-scenes, how big, and does sum(scene_size^2)
actually beat flat N^2 -- before any structure code (graph, PPR, descent)
exists. No iris/ changes. Measurement only.
"""
import argparse
import sys

import numpy as np
from scipy.signal import find_peaks

from iris.charon_v import _demux_packet_curve, compute_valley_scene_boundaries, get_stream_fps

# REPORT-NOT-TUNE: stated defaults. Do NOT adjust either to hit a target
# scene count or edge count. If the result looks wrong, surface it and STOP.
VALLEY_PERCENTILE = 25.0   # a boundary is a local min of the P/B packet curve BELOW this percentile of that curve
SCENE_SIZE_CAP    = 128    # max survivor-proxy frames per scene; larger scenes are split

# REPORT-NOT-TUNE: stated defaults. Do NOT adjust either to hit a target
# multi-burst fraction. If the result looks wrong, surface it, do not retune.
BURST_PROMINENCE_PCTILE = 75.0   # a burst = packet-curve local max above this pctile of THAT SCENE's non-kf sizes
BURST_MIN_SEP_FRAMES    = 15     # two maxima closer than this count as one burst (dedupe noise)


def bucket_histogram(values, edges):
    counts = [0] * (len(edges) + 1)
    for v in values:
        placed = False
        for i, e in enumerate(edges):
            if v <= e:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    labels = []
    prev = 0
    for e in edges:
        labels.append(f"<={e}")
        prev = e
    labels.append(f">{edges[-1]}")
    return list(zip(labels, counts))


def scene_survivor_proxy(scene, non_kf_by_idx):
    start, end = scene
    scene_sizes = [size for idx, size in non_kf_by_idx if start <= idx < end]
    if not scene_sizes:
        return 0, 0.0
    candidate_thresh = float(np.percentile(scene_sizes, 90))
    survivors = sum(1 for s in scene_sizes if s >= candidate_thresh)
    return survivors, candidate_thresh


def deepest_interior_valley(scene, non_kf_by_idx):
    start, end = scene
    interior = [(idx, size) for idx, size in non_kf_by_idx if start < idx < end]
    if len(interior) < 3:
        return None
    sizes = np.array([size for _, size in interior])
    idxs = np.array([idx for idx, _ in interior])
    pos = int(np.argmin(sizes))
    return int(idxs[pos])


def count_scene_bursts(scene, non_kf_by_idx):
    """Prominent sub-peaks in a scene's non-kf packet curve. Structural
    (packet-domain) proxy only -- not the embedding-domain centroid decision."""
    start, end = scene
    interior = [(idx, size) for idx, size in non_kf_by_idx if start <= idx < end]
    if len(interior) < 2:
        return 0
    sizes = np.array([size for _, size in interior])
    thresh = np.percentile(sizes, BURST_PROMINENCE_PCTILE)
    peaks, _ = find_peaks(sizes, height=thresh, distance=BURST_MIN_SEP_FRAMES)
    return int(len(peaks))


def split_oversized_scenes(scenes, non_kf_by_idx, cap):
    result = []
    splits = 0
    queue = list(scenes)
    while queue:
        scene = queue.pop(0)
        survivors, _ = scene_survivor_proxy(scene, non_kf_by_idx)
        if survivors <= cap:
            result.append(scene)
            continue
        split_point = deepest_interior_valley(scene, non_kf_by_idx)
        start, end = scene
        if split_point is None:
            split_point = (start + end) // 2
        splits += 1
        queue.insert(0, (split_point, end))
        queue.insert(0, (start, split_point))
    return sorted(result), splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    all_frame_energies, iframe_indices, non_kf_energies = _demux_packet_curve(args.video)
    n_frames = len(all_frame_energies)
    num_iframes = len(iframe_indices)

    iframe_set = set(iframe_indices)
    non_kf_by_idx = [(idx, size) for idx, size in all_frame_energies if idx not in iframe_set]

    fps = get_stream_fps(args.video)
    uncoarsened_scenes = compute_valley_scene_boundaries(
        all_frame_energies, iframe_indices, fps, valley_percentile=VALLEY_PERCENTILE
    )
    scenes, num_splits = split_oversized_scenes(uncoarsened_scenes, non_kf_by_idx, SCENE_SIZE_CAP)
    num_scenes = len(scenes)

    survivor_proxies = []
    for scene in scenes:
        survivors, _ = scene_survivor_proxy(scene, non_kf_by_idx)
        survivor_proxies.append(survivors)

    n_surv = sum(survivor_proxies)

    edges_scene_sparse = sum(s * s for s in survivor_proxies)
    edges_flat = n_surv * (n_surv - 1) // 2
    ratio = (edges_flat / edges_scene_sparse) if edges_scene_sparse > 0 else float("inf")

    span_edges = [10, 30, 60, 120, 300, 600, 1200]
    spans = [end - start for start, end in scenes]
    span_hist = bucket_histogram(spans, span_edges)

    surv_edges = [4, 8, 16, 32, 64, 128]
    surv_hist = bucket_histogram(survivor_proxies, surv_edges)

    print(f"N_frames = {n_frames}")
    print(f"num_iframes = {num_iframes}")
    print(f"num_scenes (post-cap) = {num_scenes}")
    print(f"num_scenes_split = {num_splits}")
    print()
    print("scene FRAME-SPAN histogram:")
    for label, count in span_hist:
        print(f"  {label}: {count}")
    print()
    print("NOTE: survivor_proxy is a zero-decode PROJECTION (per-scene 90th-pctile packet-size")
    print("gate on the non-keyframe curve). Authoritative survivor count comes from the full")
    print("gate in phase6_survivor_scale.py.")
    print(f"survivor_proxy total = {n_surv}")
    print("per-scene survivor_proxy histogram:")
    for label, count in surv_hist:
        print(f"  {label}: {count}")
    print()
    print(f"edges_scene_sparse = sum(survivor_proxy_i^2) = {edges_scene_sparse}")
    print(f"edges_flat = N_surv*(N_surv-1)//2 = {edges_flat}")
    print(f"ratio edges_flat / edges_scene_sparse = {ratio}")
    print()
    print(f"VALLEY_PERCENTILE = {VALLEY_PERCENTILE}")
    print(f"SCENE_SIZE_CAP = {SCENE_SIZE_CAP}")
    print(f"scenes that hit the cap (split) = {num_splits}")

    # --- MULTI-BURST (structural, packet-domain, zero-decode) ---
    burst_counts = [count_scene_bursts(scene, non_kf_by_idx) for scene in uncoarsened_scenes]
    num_uncoarsened_scenes = len(uncoarsened_scenes)
    multi_burst_count = sum(1 for b in burst_counts if b >= 2)
    multi_burst_frac = (multi_burst_count / num_uncoarsened_scenes) if num_uncoarsened_scenes else 0.0

    burst_hist = {"1": 0, "2": 0, "3": 0, ">=4": 0}
    for b in burst_counts:
        if b <= 1:
            burst_hist["1"] += 1
        elif b == 2:
            burst_hist["2"] += 1
        elif b == 3:
            burst_hist["3"] += 1
        else:
            burst_hist[">=4"] += 1

    print()
    print("=== MULTI-BURST (structural, packet-domain, zero-decode) ===")
    print(f"num_uncoarsened_scenes = {num_uncoarsened_scenes}")
    print(f"multi_burst_scene_count = {multi_burst_count}")
    print(f"multi_burst_fraction = {multi_burst_frac}")
    print("bursts-per-scene histogram:")
    for label in ("1", "2", "3", ">=4"):
        print(f"  {label}: {burst_hist[label]}")
    print(f"BURST_PROMINENCE_PCTILE = {BURST_PROMINENCE_PCTILE}")
    print(f"BURST_MIN_SEP_FRAMES = {BURST_MIN_SEP_FRAMES}")
    print("NOTE: this is a STRUCTURAL (packet-domain) proxy; the centroid-mush decision")
    print("(single vs sub-centroid) is embedding-domain and is deferred to the index-build subtask.")

    if num_scenes <= 1:
        print()
        print("DEGENERATE SCENE DISTRIBUTION — surface to architecture, do not proceed to graph")
        sys.exit(1)
    if n_surv > 0 and num_scenes > n_surv / 2:
        print()
        print("DEGENERATE SCENE DISTRIBUTION — surface to architecture, do not proceed to graph")
        sys.exit(1)


if __name__ == "__main__":
    main()
