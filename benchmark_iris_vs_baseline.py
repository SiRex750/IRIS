"""
benchmark_iris_vs_baseline.py

Runs a REAL, live comparison between IRIS's codec-gated frame selection and the
standard industry baseline (uniform frame sampling at a fixed rate) on the SAME
test video. Produces a terminal report and a saved PNG chart.

This does NOT claim an accuracy/quality improvement -- uniform sampling has no
verification step to compare against, so that comparison isn't apples-to-apples
yet. This measures what IS fairly comparable today: how many frames each
approach has to send downstream to an LLM, and the resulting cost/latency
implications, which is IRIS's actual, current, defensible claim.

Usage:
    python benchmark_iris_vs_baseline.py [video_path] [--query "..."] [--fps 1.0]

If no video_path is given, downloads the same w3schools mov_bbb.mp4 test clip
used in all prior integration tests, for consistency with today's results.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_VIDEO_URL = "https://www.w3schools.com/html/mov_bbb.mp4"
DEFAULT_QUERY = "What action events happen in this video?"
OUTPUT_DIR = Path("benchmark_results")


def download_test_video(dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Test video already exists locally at: {dest}")
        return dest
    print(f"Downloading test video from {DEFAULT_VIDEO_URL} ...")
    urllib.request.urlretrieve(DEFAULT_VIDEO_URL, dest)
    print("Download successful.")
    return dest


def get_video_stats(video_path: str) -> dict:
    """Independent, IRIS-agnostic frame count / fps / duration probe via PyAV."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames
    fps = float(stream.average_rate) if stream.average_rate else 25.0
    duration_sec = float(stream.duration * stream.time_base) if stream.duration else (total_frames / fps if fps else 0.0)

    if not total_frames or total_frames <= 0:
        # Some containers don't report frame count in metadata; count by decoding.
        total_frames = 0
        for _ in container.decode(stream):
            total_frames += 1
        container.close()
        container = av.open(video_path)

    container.close()
    return {"total_frames": int(total_frames), "fps": fps, "duration_sec": duration_sec}


def uniform_baseline_frame_count(total_frames: int, fps: float, sample_fps: float) -> int:
    """
    Standard industry baseline: sample frames at a fixed rate (sample_fps),
    independent of content. This is the approach used as the baseline in
    KFFocus / AKS / Q-Frame / KeyVideoLLM and most production video-LLM
    pipelines that don't do content-adaptive selection.
    """
    if fps <= 0:
        return total_frames
    duration_sec = total_frames / fps
    return max(1, int(round(duration_sec * sample_fps)))


def run_iris_pipeline(video_path: str, query: str) -> dict:
    """Calls the real IRIS pipeline and returns its actual measured results."""
    import pipeline
    t0 = time.time()
    result = pipeline.run_pipeline(video_path, query, verbose=False)
    elapsed = time.time() - t0
    result["_wall_clock_total_sec"] = elapsed
    return result


def estimate_llm_cost(frame_count: int, cost_per_frame_usd: float = 0.0015) -> float:
    """
    Rough order-of-magnitude cost estimate for sending N frames through a
    vision-capable LLM call (using a representative per-image token cost).
    This is illustrative, not a precise billing figure -- labeled as such
    in the report.
    """
    return frame_count * cost_per_frame_usd


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark IRIS vs uniform-sampling baseline")
    parser.add_argument("video_path", nargs="?", default=None)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--fps", type=float, default=1.0,
                         help="Uniform sampling rate for the baseline, in frames/sec. "
                              "1.0 fps is a common default in production video-LLM pipelines.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.video_path:
        video_path = Path(args.video_path)
    else:
        video_path = download_test_video(OUTPUT_DIR / "mov_bbb.mp4")

    print("\n=== Probing video (independent of IRIS) ===")
    vstats = get_video_stats(str(video_path))
    print(f"  Total frames : {vstats['total_frames']}")
    print(f"  FPS          : {vstats['fps']:.2f}")
    print(f"  Duration     : {vstats['duration_sec']:.2f}s")

    baseline_frames = uniform_baseline_frame_count(
        vstats["total_frames"], vstats["fps"], args.fps
    )

    print(f"\n=== Running IRIS pipeline (live, real run) ===")
    iris_result = run_iris_pipeline(str(video_path), args.query)

    iris_frames_processed = iris_result.get("frames_processed")
    if iris_frames_processed is None:
        # fall back to whatever count fields exist in the result dict
        iris_frames_processed = iris_result.get("non_skip_frame_count") \
            or len(iris_result.get("raw_records", [])) \
            or len(iris_result.get("output_frames", []))

    total_frames = vstats["total_frames"]
    iris_reduction = total_frames / iris_frames_processed if iris_frames_processed else float("nan")
    baseline_reduction = total_frames / baseline_frames if baseline_frames else float("nan")

    iris_cost = estimate_llm_cost(iris_frames_processed)
    baseline_cost = estimate_llm_cost(baseline_frames)
    cost_savings_pct = (1 - iris_cost / baseline_cost) * 100 if baseline_cost else float("nan")

    report = {
        "video": str(video_path),
        "video_total_frames": total_frames,
        "video_fps": vstats["fps"],
        "video_duration_sec": vstats["duration_sec"],
        "uniform_baseline": {
            "sample_rate_fps": args.fps,
            "frames_sent_to_llm": baseline_frames,
            "reduction_factor_vs_raw": round(baseline_reduction, 3),
            "estimated_llm_cost_usd": round(baseline_cost, 5),
        },
        "iris": {
            "frames_sent_to_llm": iris_frames_processed,
            "reduction_factor_vs_raw": round(iris_reduction, 3),
            "estimated_llm_cost_usd": round(iris_cost, 5),
            "skip_ratio_reported_by_pipeline": iris_result.get("compression_ratio")
                or iris_result.get("skip_ratio"),
            "wall_clock_total_sec": round(iris_result["_wall_clock_total_sec"], 3),
            "claims_verified": iris_result.get("is_verified"),
            "verified_claim_count": len(iris_result.get("verified_claims", []) or []),
            "rejected_claim_count": len(iris_result.get("rejected_claims", []) or []),
            "unverifiable_claim_count": len(iris_result.get("unverifiable_claims", []) or []),
        },
        "comparison": {
            "frames_iris_vs_baseline_ratio": round(baseline_frames / iris_frames_processed, 3)
                if iris_frames_processed else None,
            "estimated_cost_savings_pct_vs_uniform_baseline": round(cost_savings_pct, 1),
        },
        "caveats": [
            "Cost figures are illustrative order-of-magnitude estimates "
            "(per-frame vision-token cost assumption), not exact billing numbers.",
            "This compares FRAME-SELECTION EFFICIENCY only (how many frames "
            "each approach sends to an LLM). It does NOT compare answer "
            "accuracy/quality -- uniform sampling has no verification step, "
            "so an apples-to-apples accuracy comparison isn't available yet.",
            "Single test video (mov_bbb.mp4). Not yet validated across a "
            "dataset -- see eval_suite.py for planned multi-clip ablation.",
        ],
    }

    print("\n" + "=" * 60)
    print("  IRIS vs. UNIFORM-SAMPLING BASELINE -- LIVE BENCHMARK")
    print("=" * 60)
    print(f"  Video: {video_path.name}  ({total_frames} frames, "
          f"{vstats['fps']:.1f} fps, {vstats['duration_sec']:.1f}s)\n")

    print(f"  {'Metric':<38}{'Baseline (uniform)':<22}{'IRIS':<14}")
    print(f"  {'-'*38}{'-'*22}{'-'*14}")
    print(f"  {'Frames sent to LLM':<38}{baseline_frames:<22}{iris_frames_processed:<14}")
    print(f"  {'Reduction vs. raw frame count':<38}{baseline_reduction:<22.2f}{iris_reduction:<14.2f}")
    print(f"  {'Est. LLM cost (illustrative, USD)':<38}${baseline_cost:<21.5f}${iris_cost:<13.5f}")
    print()
    if iris_frames_processed < baseline_frames:
        print(f"  IRIS sends {report['comparison']['frames_iris_vs_baseline_ratio']}x fewer frames "
              f"than uniform {args.fps} fps sampling on this clip.")
        print(f"  Estimated cost reduction vs. baseline: "
              f"{report['comparison']['estimated_cost_savings_pct_vs_uniform_baseline']}%")
    else:
        print(f"  NOTE: On this specific clip, fixed {args.fps} fps sampling sends FEWER frames "
              f"than IRIS ({baseline_frames} vs {iris_frames_processed}).")
        print(f"  This is expected on short/low-motion test clips -- a low fixed rate happens to")
        print(f"  skip a lot here too. IRIS's actual advantage is being CONTENT-ADAPTIVE rather")
        print(f"  than fixed: a fixed-fps baseline either misses real events on sparse footage")
        print(f"  (rate too low) or wastes compute during dead time on longer/denser footage")
        print(f"  (rate too high), and there's no single fixed rate that avoids both failure")
        print(f"  modes across different videos. Try --fps 0.5 or --fps 2.0 to see this baseline's")
        print(f"  frame count swing widely on the SAME clip, while IRIS's count is driven by")
        print(f"  actual content regardless of any chosen rate.")
    print()
    print(f"  Claims Verified (this run): {iris_result.get('is_verified')}  "
          f"({report['iris']['verified_claim_count']} verified / "
          f"{report['iris']['rejected_claim_count']} rejected / "
          f"{report['iris']['unverifiable_claim_count']} unverifiable)")
    print()
    print("  CAVEATS (read before presenting this):")
    for c in report["caveats"]:
        print(f"   - {c}")
    print("=" * 60)

    report_path = OUTPUT_DIR / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved JSON report to: {report_path}")

    make_chart(report, OUTPUT_DIR / "iris_vs_baseline_compression.png")
    make_sweep_chart(vstats, iris_frames_processed, OUTPUT_DIR / "iris_vs_baseline_fps_sweep.png")


def make_sweep_chart(vstats: dict, iris_frames: int, out_path: Path) -> None:
    """
    The real, defensible claim isn't 'IRIS always beats a fixed rate' --
    it's that IRIS's frame count is driven by content, while a fixed-fps
    baseline's frame count swings with whatever rate you happen to pick,
    with no single rate that's correct across different videos. This
    sweep makes that visible on the same clip.
    """
    rates = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    baseline_counts = [
        uniform_baseline_frame_count(vstats["total_frames"], vstats["fps"], r)
        for r in rates
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rates, baseline_counts, marker="o", color="#9a9a93",
            label="Uniform sampling (varies with chosen rate)", linewidth=2)
    ax.axhline(iris_frames, color="#1D9E75", linewidth=2.5,
                label=f"IRIS (content-driven, {iris_frames} frames)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{r}" for r in rates])
    ax.set_xlabel("Uniform sampling rate (frames/sec) -- a value you have to guess")
    ax.set_ylabel("Frames sent to LLM")
    ax.set_title("Baseline Frame Count Depends on a Guessed Rate;\nIRIS's Does Not")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved sweep chart to: {out_path}")


def make_chart(report: dict, out_path: Path) -> None:
    baseline_frames = report["uniform_baseline"]["frames_sent_to_llm"]
    iris_frames = report["iris"]["frames_sent_to_llm"]
    total_frames = report["video_total_frames"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left: frames sent to LLM ---
    ax = axes[0]
    bars = ax.bar(
        [f"Uniform sampling\n({report['uniform_baseline']['sample_rate_fps']} fps)", "IRIS\n(codec-gated)"],
        [baseline_frames, iris_frames],
        color=["#9a9a93", "#1D9E75"],
        width=0.55,
    )
    ax.axhline(total_frames, color="#cc4444", linestyle="--", linewidth=1, label=f"Total frames in clip ({total_frames})")
    ax.set_ylabel("Frames sent to LLM")
    ax.set_title("Frames Sent Downstream")
    for bar, val in zip(bars, [baseline_frames, iris_frames]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(total_frames * 0.02, 1),
                 str(val), ha="center", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Right: reduction factor ---
    ax2 = axes[1]
    baseline_reduction = report["uniform_baseline"]["reduction_factor_vs_raw"]
    iris_reduction = report["iris"]["reduction_factor_vs_raw"]
    bars2 = ax2.bar(
        ["Uniform sampling", "IRIS"],
        [baseline_reduction, iris_reduction],
        color=["#9a9a93", "#1D9E75"],
        width=0.55,
    )
    ax2.set_ylabel("Reduction factor (raw frames / frames sent)")
    ax2.set_title("Compression vs. Raw Video")
    for bar, val in zip(bars2, [baseline_reduction, iris_reduction]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + max(iris_reduction * 0.02, 0.1),
                  f"{val:.2f}x", ha="center", fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("IRIS vs. Uniform-Sampling Baseline -- Frame Selection Efficiency", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.01,
              "Compares frame-selection efficiency only. Does not compare answer accuracy (no baseline verification step exists).",
              ha="center", fontsize=8, color="#666666")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to: {out_path}")


if __name__ == "__main__":
    main()
