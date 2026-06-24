"""
IRIS Compression Analysis — measures frame reduction at every pipeline stage.

Traces a video through:
    Stage 0: Raw decode (total frames in video)
    Stage 1: Charon-V tier filtering (SKIP frames dropped)
    Stage 2: ActionScore continuous scoring + peak detection
    Stage 3: L1 Elysium cache (fixed capacity, keep_score eviction)
    Stage 4: L2 Tiered Index (memory compression via FlatIP/HNSW/PQ)

Reports compression ratios, memory estimates, and tier distribution.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ── IRIS imports ──────────────────────────────────────────────────────────
import charon_v
from action_score import ActionScoreModule, ActionScoreConfig
from frame_motion_descriptor import FrameMotionDescriptor
from cached_frame import CachedFrame
from iris_config import IRISConfig
from l1_elysium import L1ElysiumCache
from l2_index import L2TieredIndex, FrameTier


def run_compression_analysis(video_path: str, l1_capacity: int = 32) -> dict:
    """Run the full IRIS pipeline and measure compression at every stage."""

    results = {}
    config = IRISConfig(l1_capacity=l1_capacity)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 0 + 1: Charon-V — decode video and tier-filter
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print(f"IRIS Compression Analysis: {Path(video_path).name}")
    print("=" * 70)

    t0 = time.perf_counter()
    output_frames, stats, raw_records = charon_v.parse_video(
        video_path,
        return_stats=True,
        return_raw=True,
        adaptive=config.adaptive,
        salient_thresh=config.salient_thresh,
        candidate_thresh=config.candidate_thresh,
    )
    t_charon = time.perf_counter() - t0

    total_frames = stats["total"]
    kept_after_charon = len(output_frames)
    skipped = stats["skipped"]

    results["stage0_total_frames"] = total_frames
    results["stage1_kept"] = kept_after_charon
    results["stage1_skipped"] = skipped
    results["stage1_compression_ratio"] = total_frames / max(kept_after_charon, 1)
    results["stage1_time_sec"] = t_charon

    print(f"\n{'─' * 70}")
    print(f"STAGE 0 — Raw Video Decode")
    print(f"{'─' * 70}")
    print(f"  Total frames decoded:     {total_frames}")
    print(f"  Video file:               {Path(video_path).name}")

    print(f"\n{'─' * 70}")
    print(f"STAGE 1 — Charon-V Tier Filtering (SKIP frames dropped)")
    print(f"{'─' * 70}")
    print(f"  I-Frames:                 {stats['i_frames']:>5}  ({stats['i_frames']/total_frames*100:5.1f}%)")
    print(f"  PEAK:                     {stats['peaks']:>5}  ({stats['peaks']/total_frames*100:5.1f}%)")
    print(f"  SALIENT:                  {stats['salient']:>5}  ({stats['salient']/total_frames*100:5.1f}%)")
    print(f"  CANDIDATE:                {stats['candidate']:>5}  ({stats['candidate']/total_frames*100:5.1f}%)")
    print(f"  SKIP (dropped):           {skipped:>5}  ({skipped/total_frames*100:5.1f}%)")
    print(f"  ─────────────────────────────────────")
    print(f"  Kept after Charon-V:      {kept_after_charon:>5}  ({kept_after_charon/total_frames*100:5.1f}%)")
    print(f"  Compression ratio:        {results['stage1_compression_ratio']:.2f}×")
    print(f"  Time:                     {t_charon:.3f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2: ActionScore — continuous scoring + peak detection
    # ══════════════════════════════════════════════════════════════════════
    t1 = time.perf_counter()

    action_config = ActionScoreConfig(
        residual_weight=config.residual_weight,
        motion_weight=config.motion_weight,
        entropy_weight=config.entropy_weight,
        peak_distance=config.peak_distance,
        peak_prominence=config.peak_prominence,
        persistence_threshold=config.persistence_threshold,
        max_prominence=config.max_prominence,
    )
    scorer = ActionScoreModule(action_config)

    # Build feature dicts for ActionScore from raw records
    feature_dicts = []
    for rec in raw_records:
        feature_dicts.append({
            "frame_idx": rec["frame_idx"],
            "residual_energy": rec["residual_energy"],
            "motion_magnitude": rec["motion_magnitude"],
            "entropy": rec["entropy"],
        })

    action_records = scorer.score_all(feature_dicts)
    action_map = {r["frame_idx"]: r for r in action_records}
    t_action = time.perf_counter() - t1

    # Analyze action score distribution
    scores = [r["action_score"] for r in action_records]
    peaks = [r for r in action_records if r["is_peak"]]
    non_peaks = [r for r in action_records if not r["is_peak"]]

    results["stage2_total_scored"] = len(action_records)
    results["stage2_peaks"] = len(peaks)
    results["stage2_mean_score"] = float(np.mean(scores)) if scores else 0.0
    results["stage2_median_score"] = float(np.median(scores)) if scores else 0.0
    results["stage2_std_score"] = float(np.std(scores)) if scores else 0.0
    results["stage2_time_sec"] = t_action

    print(f"\n{'─' * 70}")
    print(f"STAGE 2 — ActionScore Continuous Scoring")
    print(f"{'─' * 70}")
    print(f"  Frames scored:            {len(action_records)}")
    print(f"  Peaks detected:           {len(peaks)}")
    print(f"  Score distribution:")
    print(f"    Mean:                   {results['stage2_mean_score']:.4f}")
    print(f"    Median:                 {results['stage2_median_score']:.4f}")
    print(f"    Std:                    {results['stage2_std_score']:.4f}")
    print(f"    Min:                    {min(scores):.4f}")
    print(f"    Max:                    {max(scores):.4f}")

    # Score histogram
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(scores, bins=bins)
    print(f"  Score histogram:")
    for i in range(len(bins) - 1):
        bar = "#" * hist[i]
        print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:>4}  {bar}")
    print(f"  Time:                     {t_action:.3f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 3: L1 Elysium — fixed-capacity cache with keep_score eviction
    # ══════════════════════════════════════════════════════════════════════
    t2 = time.perf_counter()

    cache = L1ElysiumCache(config=config)
    evicted_count = 0

    for of in output_frames:
        fidx = of["frame_idx"]
        ar = action_map.get(fidx)
        if ar is None:
            continue

        motion = FrameMotionDescriptor(
            frame_idx=fidx,
            timestamp_sec=of.get("timestamp", 0.0),
            residual_energy=of.get("residual_energy", 0.0),
            divergence=of.get("divergence", 0.0),
            curl=of.get("curl", 0.0),
            jacobian_frobenius=of.get("jacobian_frobenius", 0.0),
            hessian_max_eigenvalue=of.get("hessian_max_eigenvalue", 0.0),
            motion_entropy=of.get("motion_entropy", 0.0),
        )

        cf = CachedFrame(
            frame_idx=fidx,
            timestamp_sec=of.get("timestamp", 0.0),
            action_score=ar["action_score"],
            persistence_value=ar["persistence_value"],
            is_peak=ar["is_peak"],
            motion=motion,
        )
        cf.build_motion_embedding()

        was_full = cache.is_full
        cache.admit(cf)
        if was_full and fidx not in cache:
            pass  # New frame immediately evicted (shouldn't happen often)
        elif was_full:
            evicted_count += 1

    t_l1 = time.perf_counter() - t2

    l1_final = len(cache)
    results["stage3_l1_capacity"] = config.l1_capacity
    results["stage3_frames_in_cache"] = l1_final
    results["stage3_evictions"] = evicted_count
    results["stage3_compression_ratio"] = total_frames / max(l1_final, 1)
    results["stage3_time_sec"] = t_l1

    # Analyze what survived in L1
    survivors = list(cache.frames())
    survivor_scores = [f.action_score for f in survivors]
    survivor_peaks = sum(1 for f in survivors if f.is_peak)

    print(f"\n{'─' * 70}")
    print(f"STAGE 3 — L1 Elysium Cache (capacity={config.l1_capacity})")
    print(f"{'─' * 70}")
    print(f"  Frames admitted:          {kept_after_charon}")
    print(f"  Evictions triggered:      {evicted_count}")
    print(f"  Frames in cache now:      {l1_final}")
    print(f"  Peaks retained:           {survivor_peaks}/{len(peaks)}")
    print(f"  Compression ratio:        {results['stage3_compression_ratio']:.1f}× (vs raw)")
    print(f"  Cache hit rate:           {cache.hits}/{cache.hits + cache.misses} ({cache.hits/(cache.hits + cache.misses)*100:.1f}%)" if (cache.hits + cache.misses) > 0 else "  Cache hit rate:           N/A")
    if survivor_scores:
        print(f"  Survivor score range:     [{min(survivor_scores):.4f} — {max(survivor_scores):.4f}]")
        print(f"  Survivor mean score:      {np.mean(survivor_scores):.4f}")
    print(f"  Time:                     {t_l1:.3f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 4: L2 Tiered Index — memory compression
    # ══════════════════════════════════════════════════════════════════════
    t3 = time.perf_counter()

    # Simulate 512-D embeddings for all non-SKIP frames
    embed_dim = config.l2_embed_dim
    tiered_index = L2TieredIndex(config=config)

    tier_assignments = {"PEAK": 0, "SALIENT": 0, "CANDIDATE": 0}

    for of in output_frames:
        fidx = of["frame_idx"]
        ar = action_map.get(fidx)
        if ar is None:
            continue

        # Simulate a normalized embedding
        rng = np.random.RandomState(fidx)
        emb = rng.randn(embed_dim).astype(np.float32)
        emb /= max(np.linalg.norm(emb), 1e-8)

        tier = tiered_index.add(
            frame_idx=fidx,
            embedding=emb,
            action_score=ar["action_score"],
            is_peak=ar["is_peak"],
            persistence_value=ar["persistence_value"],
        )
        tier_assignments[tier.value] += 1

    t_l2 = time.perf_counter() - t3

    l2_stats = tiered_index.stats()

    results["stage4_tier_distribution"] = tier_assignments
    results["stage4_stats"] = l2_stats
    results["stage4_time_sec"] = t_l2

    # Memory analysis
    flat_mem = l2_stats["flat_equivalent_memory_bytes"]
    actual_mem = (
        l2_stats["peak"]["memory_bytes"]
        + l2_stats["salient"]["memory_bytes"]
        + l2_stats["candidate"]["memory_bytes"]
    )

    print(f"\n{'─' * 70}")
    print(f"STAGE 4 — L2 Tiered FAISS Index")
    print(f"{'─' * 70}")
    print(f"  Total indexed:            {l2_stats['total_frames']}")
    print(f"  Tier distribution:")
    print(f"    PEAK → FlatIP:          {tier_assignments['PEAK']:>5}  ({tier_assignments['PEAK']/max(l2_stats['total_frames'],1)*100:5.1f}%)")
    print(f"    SALIENT → HNSW:         {tier_assignments['SALIENT']:>5}  ({tier_assignments['SALIENT']/max(l2_stats['total_frames'],1)*100:5.1f}%)")
    print(f"    CANDIDATE → PQ:         {tier_assignments['CANDIDATE']:>5}  ({tier_assignments['CANDIDATE']/max(l2_stats['total_frames'],1)*100:5.1f}%)")
    print(f"  PQ trained:               {l2_stats['candidate']['pq_trained']}")
    print(f"  Memory usage:")
    print(f"    PEAK (FlatIP):          {l2_stats['peak']['memory_bytes']/1024:.1f} KB")
    print(f"    SALIENT (HNSW):         {l2_stats['salient']['memory_bytes']/1024:.1f} KB")
    print(f"    CANDIDATE (PQ/Flat):    {l2_stats['candidate']['memory_bytes']/1024:.1f} KB")
    print(f"    Total actual:           {actual_mem/1024:.1f} KB")
    print(f"    Flat equivalent:        {flat_mem/1024:.1f} KB")
    if actual_mem > 0:
        print(f"    Memory reduction:       {flat_mem/actual_mem:.1f}×")
    print(f"  Time:                     {t_l2:.3f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    total_time = t_charon + t_action + t_l1 + t_l2

    # Estimate per-frame memory in L1 (dual-vector + metadata)
    # Visual embedding: 512 * 2 bytes (bfloat16) = 1024 bytes
    # Motion embedding: 6 * 4 bytes (float32) = 24 bytes
    # FrameMotionDescriptor: 6 * 8 bytes (float64) = 48 bytes
    # Metadata (scores, indices, etc.): ~64 bytes
    per_frame_l1_bytes = 1024 + 24 + 48 + 64
    l1_total_memory = l1_final * per_frame_l1_bytes

    # Raw video: all frames would need full embeddings
    raw_memory = total_frames * embed_dim * 4  # float32 embeddings for all

    print(f"\n{'═' * 70}")
    print(f"COMPRESSION SUMMARY")
    print(f"{'═' * 70}")
    print(f"  Raw video frames:         {total_frames}")
    print(f"  After Charon-V filter:    {kept_after_charon}  ({kept_after_charon/total_frames*100:.1f}% of raw)")
    print(f"  In L1 active cache:       {l1_final}  ({l1_final/total_frames*100:.1f}% of raw)")
    print(f"  ─────────────────────────────────────")
    print(f"  Frame reduction:          {total_frames} → {l1_final}  ({results['stage3_compression_ratio']:.1f}× compression)")
    print(f"  ─────────────────────────────────────")
    print(f"  Memory comparison:")
    print(f"    Naive (all frames):     {raw_memory/1024:.1f} KB  (512-D float32 per frame)")
    print(f"    L1 cache:               {l1_total_memory/1024:.1f} KB  (dual-vector + metadata)")
    print(f"    L2 tiered index:        {actual_mem/1024:.1f} KB  (FlatIP+HNSW+PQ)")
    print(f"    Total IRIS:             {(l1_total_memory + actual_mem)/1024:.1f} KB")
    print(f"    Memory reduction:       {raw_memory/max(l1_total_memory + actual_mem, 1):.1f}×")
    print(f"  ─────────────────────────────────────")
    print(f"  Pipeline time:            {total_time:.3f}s")
    print(f"{'═' * 70}")

    results["total_time_sec"] = total_time
    results["memory_raw_kb"] = raw_memory / 1024
    results["memory_iris_kb"] = (l1_total_memory + actual_mem) / 1024
    results["memory_reduction"] = raw_memory / max(l1_total_memory + actual_mem, 1)

    return results


if __name__ == "__main__":
    import glob

    videos = glob.glob("*.mp4")
    if not videos:
        print("No .mp4 files found in the current directory.")
        sys.exit(1)

    all_results = {}
    for vpath in sorted(videos):
        print(f"\n\n{'#' * 70}")
        print(f"  Analyzing: {vpath}")
        print(f"{'#' * 70}\n")

        try:
            r = run_compression_analysis(vpath, l1_capacity=32)
            all_results[vpath] = r
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ── Cross-video comparison table ──────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n\n{'═' * 70}")
        print(f"CROSS-VIDEO COMPARISON")
        print(f"{'═' * 70}")
        print(f"{'Video':<25} {'Total':>6} {'Kept':>6} {'L1':>4} {'Frame ×':>8} {'Mem ×':>8} {'Time':>7}")
        print(f"{'─' * 25} {'─' * 6} {'─' * 6} {'─' * 4} {'─' * 8} {'─' * 8} {'─' * 7}")
        for vpath, r in all_results.items():
            print(
                f"{Path(vpath).name:<25} "
                f"{r['stage0_total_frames']:>6} "
                f"{r['stage1_kept']:>6} "
                f"{r['stage3_frames_in_cache']:>4} "
                f"{r['stage3_compression_ratio']:>7.1f}× "
                f"{r['memory_reduction']:>7.1f}× "
                f"{r['total_time_sec']:>6.2f}s"
            )
