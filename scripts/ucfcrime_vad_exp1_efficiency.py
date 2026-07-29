"""Step 5 efficiency measurement for UCF-Crime VAD Experiment 1.

Runs iris.charon_v.parse_video (the codec-level L1 ingest pass) on a sample of
>=30 UCF-Crime videos spanning the duration range found in
ucfcrime_dataset_validation.json, measuring wall time, peak RSS, and
frames-decoded-vs-retained. Also instruments every torch.nn.Module.__call__ to
prove no neural-network forward pass occurs during this ingest path -- this is
verified by monkeypatching, not asserted from reading the source.

No ground-truth annotations exist for this dataset on this box (see
ucfcrime_dataset_validation.json / the experiment report), so this script
computes NO AUC and NO anomaly score interpretation -- efficiency only.

Writes:
  tuning/ucfcrime_vad_exp1/efficiency_measurements.json
  tuning/ucfcrime_vad_exp1/efficiency_per_video.csv
  tuning/ucfcrime_vad_exp1/per_frame_action_scores/<video_stem>.csv  (raw Stage-A
      codec action scores for the sampled videos, frozen weights, for independent
      recomputation -- NOT an AUC, no ground truth available)
"""
from __future__ import annotations

import csv
import gc
import json
import random
import sys
import threading
import time
from pathlib import Path

import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = REPO_ROOT / "tuning" / "ucfcrime_vad_exp1" / "ucfcrime_dataset_validation.json"
OUT_DIR = REPO_ROOT / "tuning" / "ucfcrime_vad_exp1"
PER_FRAME_DIR = OUT_DIR / "per_frame_action_scores"

FROZEN_STATE_PATH = REPO_ROOT / "tuning" / "frozen_state.json"

SAMPLE_SIZE = 32
RANDOM_SEED = 20260729  # today's date (2026-07-29), fixed so the sample is reproducible


class _NNForwardPassMonitor:
    """Monkeypatches torch.nn.Module.__call__ for the duration of a `with` block
    and counts every invocation, so "no NN forward pass" is measured, not assumed."""

    def __init__(self) -> None:
        self.call_count = 0
        self._orig = None
        self._torch_available = False

    def __enter__(self):
        try:
            import torch.nn as nn
            self._nn = nn
            self._orig = nn.Module.__call__

            def _counting_call(mod_self, *args, **kwargs):
                self.call_count += 1
                return self._orig(mod_self, *args, **kwargs)

            nn.Module.__call__ = _counting_call
            self._torch_available = True
        except Exception:
            self._torch_available = False
        return self

    def __exit__(self, *exc):
        if self._torch_available:
            self._nn.Module.__call__ = self._orig


def _peak_rss_sampler(proc: psutil.Process, stop_event: threading.Event, interval_s: float = 0.02):
    peak = [proc.memory_info().rss]

    def _run():
        while not stop_event.is_set():
            try:
                rss = proc.memory_info().rss
                if rss > peak[0]:
                    peak[0] = rss
            except Exception:
                pass
            stop_event.wait(interval_s)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return peak, t


OUTLIER_FRAME_CUTOFF = 20000  # excludes the single 126,553-frame Arson019 stress outlier
                              # (per eval_results/ucf_inventory.md: "not part of a smooth
                              # curve -- treat as a separate stress test, not a curve anchor")
                              # from the main sample so 32-video wall time stays bounded;
                              # noted explicitly, not silently dropped.


def pick_sample(records: list[dict], n: int) -> list[dict]:
    valid = [
        r for r in records
        if r["open_ok"] and r["is_h264_or_hevc"] and r["nb_frames_meta"]
        and r["nb_frames_meta"] <= OUTLIER_FRAME_CUTOFF
    ]
    valid_sorted = sorted(valid, key=lambda r: r["nb_frames_meta"])
    if len(valid_sorted) <= n:
        return valid_sorted
    # Evenly spaced across the sorted duration range (deterministic, reproducible;
    # not a uniform-random sample, so it's stated explicitly as stratified-by-size).
    idxs = [round(i * (len(valid_sorted) - 1) / (n - 1)) for i in range(n)]
    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(valid_sorted[i])
    return out


def main() -> None:
    if not FROZEN_STATE_PATH.exists():
        print(f"BLOCKER: {FROZEN_STATE_PATH} missing.", file=sys.stderr)
        sys.exit(1)
    frozen = json.loads(FROZEN_STATE_PATH.read_text())["frozen"]

    validation = json.loads(VALIDATION_PATH.read_text())
    records = validation["per_file_records"]

    sample = pick_sample(records, SAMPLE_SIZE)
    print(f"Sampled {len(sample)} videos (stratified by frame count, full range).")

    import iris.charon_v as charon_v
    from iris.action_score import ActionScoreConfig, ActionScoreModule

    action_cfg = ActionScoreConfig(
        packet_size_weight=frozen["packet_size_weight"],
        motion_weight=frozen["motion_weight"],
        luma_entropy_weight=frozen["luma_entropy_weight"],
        peak_distance=frozen["peak_distance"],
        peak_prominence=frozen["peak_prominence"],
        persistence_threshold=frozen["persistence_threshold"],
        max_prominence=frozen["max_prominence"],
    )
    action_module = ActionScoreModule(config=action_cfg)

    proc = psutil.Process()
    PER_FRAME_DIR.mkdir(parents=True, exist_ok=True)

    per_video_results = []
    total_nn_calls = 0

    for i, rec in enumerate(sample):
        video_path = REPO_ROOT / rec["path"]
        print(f"[{i + 1}/{len(sample)}] {rec['path']} ({rec['nb_frames_meta']} frames meta)")

        gc.collect()
        baseline_rss = proc.memory_info().rss
        stop_event = threading.Event()
        peak_holder, sampler_thread = _peak_rss_sampler(proc, stop_event)

        wall_t0 = time.perf_counter()
        error = None
        output_frames = None
        stats = None
        raw_records = None
        with _NNForwardPassMonitor() as mon:
            try:
                output_frames, stats, raw_records = charon_v.parse_video(
                    str(video_path),
                    return_stats=True,
                    return_raw=True,
                    full_decode=False,
                )
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
        wall_elapsed = time.perf_counter() - wall_t0

        stop_event.set()
        sampler_thread.join(timeout=1.0)
        peak_rss = peak_holder[0]
        nn_calls_this_video = mon.call_count
        total_nn_calls += nn_calls_this_video

        result = {
            "path": rec["path"],
            "size_bytes": rec["size_bytes"],
            "nb_frames_meta": rec["nb_frames_meta"],
            "fps": rec["fps"],
            "duration_s_meta": rec["duration_s"],
            "wall_time_s": wall_elapsed,
            "error": error,
            "peak_rss_bytes": peak_rss,
            "baseline_rss_bytes": baseline_rss,
            "nn_forward_pass_calls_during_ingest": nn_calls_this_video,
            "torch_available_for_monitor": mon._torch_available,
        }

        if error is None and stats is not None:
            total_decoded = stats["total"]
            retained = stats["i_frames"] + stats["peaks"] + stats["salient"] + stats["candidate"]
            skipped = stats["skipped"]
            result.update({
                "frames_total_decoded": total_decoded,
                "frames_retained_full_feature": retained,
                "frames_skipped_pixelwork": skipped,
                "retention_pct": 100.0 * retained / total_decoded if total_decoded else None,
                "wall_time_s_per_1000_frames": (
                    wall_elapsed / (total_decoded / 1000.0) if total_decoded else None
                ),
            })

            if raw_records:
                scored = action_module.score_all(
                    [
                        {
                            "frame_idx": r["frame_idx"],
                            "packet_size": r.get("packet_size", 0.0),
                            "motion_magnitude": r.get("motion_magnitude", 0.0),
                            "luma_entropy": r.get("luma_entropy", 0.0),
                        }
                        for r in raw_records
                    ]
                )
                stem = Path(rec["path"]).stem
                csv_path = PER_FRAME_DIR / f"{stem}.csv"
                with csv_path.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["frame_idx", "action_score", "is_peak", "persistence_value"])
                    for s in scored:
                        w.writerow([s["frame_idx"], s["action_score"], s["is_peak"], s["persistence_value"]])
                result["per_frame_action_score_csv"] = str(csv_path.relative_to(REPO_ROOT))
                result["n_scored_frames"] = len(scored)

        per_video_results.append(result)

    n_ok = sum(1 for r in per_video_results if r["error"] is None)
    n_failed = len(per_video_results) - n_ok
    wall_times = [r["wall_time_s"] for r in per_video_results if r["error"] is None]
    peak_rsses = [r["peak_rss_bytes"] for r in per_video_results if r["error"] is None]
    retentions = [r["retention_pct"] for r in per_video_results if r.get("retention_pct") is not None]

    try:
        import torch
        device_used = "cuda" if torch.cuda.is_available() else "cpu"
        gpu_available = torch.cuda.is_available()
    except Exception:
        device_used = "cpu (torch not importable)"
        gpu_available = False

    summary = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_config_used": frozen,
        "sample_size": len(sample),
        "sample_selection": (
            "deterministic, stratified by nb_frames_meta across the full duration "
            "range of the 350 locally-available videos (evenly spaced ranks after "
            "sorting by frame count) -- not uniform-random, chosen to span short/"
            "long videos as instructed; noted explicitly here per non-negotiable #1."
        ),
        "n_ingest_ok": n_ok,
        "n_ingest_failed": n_failed,
        "failed_videos": [
            {"path": r["path"], "error": r["error"]} for r in per_video_results if r["error"]
        ],
        "wall_time_s_mean": sum(wall_times) / len(wall_times) if wall_times else None,
        "wall_time_s_min": min(wall_times) if wall_times else None,
        "wall_time_s_max": max(wall_times) if wall_times else None,
        "peak_rss_bytes_mean": sum(peak_rsses) / len(peak_rsses) if peak_rsses else None,
        "peak_rss_bytes_max": max(peak_rsses) if peak_rsses else None,
        "retention_pct_mean": sum(retentions) / len(retentions) if retentions else None,
        "retention_pct_min": min(retentions) if retentions else None,
        "retention_pct_max": max(retentions) if retentions else None,
        "device_used": device_used,
        "gpu_available_on_this_box": gpu_available,
        "gpu_required_at_any_point": False,
        "total_nn_forward_pass_calls_across_all_sampled_videos": total_nn_calls,
        "no_nn_forward_pass_verified": total_nn_calls == 0,
        "note_on_verification_method": (
            "torch.nn.Module.__call__ was monkeypatched for the duration of each "
            "parse_video() call and every invocation counted; the counter is "
            "reported per-video and summed here, not just asserted from source "
            "reading."
        ),
    }

    (OUT_DIR / "efficiency_measurements.json").write_text(json.dumps(summary, indent=2))

    csv_path = OUT_DIR / "efficiency_per_video.csv"
    fieldnames = sorted({k for r in per_video_results for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_video_results:
            w.writerow(r)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_DIR / 'efficiency_measurements.json'} and {csv_path}")


if __name__ == "__main__":
    main()
