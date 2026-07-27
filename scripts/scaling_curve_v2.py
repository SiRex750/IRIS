"""Scaling curve v2: flat vs scene_sparse across N, UNIFORM watchdog policy.

Re-runs eval_results/scaling_curve.{json,md} (v1, uncommitted) with the
confound v1 flagged fixed: v1 capped UCF flat arms at 480s wall time but ran
the reused VIRAT flat point uncapped, so v1's "flat fails past N~3k" was a
budget artifact, not a real infeasibility frontier. This run applies ONE
watchdog (WALL_CAP_SEC wall time, MEM_CAP_BYTES RSS) identically to EVERY
flat measurement, including a fresh VIRAT N=4,892 remeasurement, and records
three distinct outcomes (completed / oom / timed_out) instead of collapsing
non-completion into "failed".

Also settles the open fairness question: does building the flat graph from
frames already loaded by the scene_sparse ingest (v1's method, and this run's
default) give a different flat latency than an INDEPENDENT flat ingest from
raw video? Tested once, on the mid-size Abuse042 clip (N=2,963).

Each (clip, mode) pair runs in its own subprocess via
scripts/_scaling_curve_v2_worker.py so a flat OOM/timeout cannot kill the
scene_sparse measurement for the same clip.

VERIFY: python scripts/scaling_curve_v2.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\Siddanth Anil\IRIS")
WORKER = REPO / "scripts" / "_scaling_curve_v2_worker.py"
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"

OUT_RAW = REPO / "eval_results" / "scaling_curve_v2_raw.json"
OUT_MD = REPO / "eval_results" / "scaling_curve_v2.md"
TMP_DIR = REPO / "eval_results" / "_v2_tmp"
TMP_DIR.mkdir(exist_ok=True)

NEXTQA_CACHE_DIR = REPO / "eval" / "data" / "nextqa" / "index_cache"

QUERY_SEED = 20260726
N_Q = 50

WALL_CAP_SEC = 3600.0
MEM_CAP_BYTES = int(29.0 * 1e9)  # system has 33.4GB total; identical to v1's cap value

FAIRNESS_CHECK_LABEL = "Abuse042"  # N=2,963, mid-size per pre-registration

# label -> (dataset, cache_path, video_path or None, target_n_selection_basis)
CLIPS = [
    ("Normal_Videos_289", "ucf", REPO / "eval/data/ucf/index_cache/Normal_Videos_289",
     REPO / "eval/data/ucf/videos/Testing_Normal_Videos_Anomaly/Normal_Videos_289_x264.mp4", "~800 (raw-frame proxy)"),
    ("Abuse025", "ucf", REPO / "eval/data/ucf/index_cache/Abuse025",
     REPO / "eval/data/ucf/videos/Anomaly-Videos-Part-1/Abuse/Abuse025_x264.mp4", "~2k (raw-frame proxy)"),
    ("Arrest016", "ucf", REPO / "eval/data/ucf/index_cache/Arrest016",
     REPO / "eval/data/ucf/videos/Anomaly-Videos-Part-1/Arrest/Arrest016_x264.mp4", "~10k (raw-frame proxy)"),
    ("Abuse042", "ucf", REPO / "eval/data/ucf/index_cache/Abuse042",
     REPO / "eval/data/ucf/videos/Anomaly-Videos-Part-1/Abuse/Abuse042_x264.mp4", "~30k (raw-frame proxy)"),
    ("VIRAT_S_040001_01_000448_001101", "virat", REPO / "eval/data/virat/index_cache/VIRAT_S_040001_01_000448_001101",
     REPO / "eval/data/virat/videos/VIRAT_S_040001_01_000448_001101.mp4", "reused committed point (~5k), remeasured under uniform policy"),
    ("Arrest047", "ucf", REPO / "eval/data/ucf/index_cache/Arrest047",
     REPO / "eval/data/ucf/videos/Anomaly-Videos-Part-1/Arrest/Arrest047_x264.mp4", "~80k (raw-frame proxy)"),
    ("Arson019", "ucf", REPO / "eval/data/ucf/index_cache/Arson019",
     REPO / "eval/data/ucf/videos/Anomaly-Videos-Part-1/Arson/Arson019_x264.mp4", "~126k (raw-frame proxy)"),
]


def _dir_fingerprint(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(d.glob("*.npz")):
        st = f.stat()
        h.update(f.name.encode()); h.update(str(st.st_mtime_ns).encode()); h.update(str(st.st_size).encode())
    return h.hexdigest()


def run_worker(mode: str, cache_path: Path, out_json: Path, *, video: Path | None = None,
                source: str = "cached_frames") -> dict:
    if out_json.exists():
        out_json.unlink()
    cmd = [
        str(PYTHON), str(WORKER),
        "--mode", mode,
        "--cache-path", str(cache_path),
        "--source", source,
        "--out-json", str(out_json),
        "--wall-cap-sec", str(WALL_CAP_SEC),
        "--mem-cap-bytes", str(MEM_CAP_BYTES),
        "--query-seed", str(QUERY_SEED),
        "--n-queries", str(N_Q),
    ]
    if video is not None:
        cmd += ["--video", str(video)]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), timeout=WALL_CAP_SEC + 180,
                               capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        # external safety net -- the in-process watchdog should have tripped first
        if out_json.exists():
            data = json.loads(out_json.read_text())
        else:
            data = {"outcome": "timed_out", "note": "external subprocess timeout, in-process watchdog did not write a result"}
        data["wall_sec_observed_by_driver"] = time.time() - t0
        return data

    wall_sec = time.time() - t0
    if out_json.exists():
        data = json.loads(out_json.read_text())
        data["wall_sec_observed_by_driver"] = wall_sec
        return data

    # subprocess exited without writing JSON: crash not caught by the watchdog
    # (e.g. a hard MemoryError / OS-level kill before the watchdog thread's
    # next poll). Distinguish "oom" (memory-shaped exit) from a generic crash.
    stderr_tail = (proc.stderr or "")[-4000:]
    is_mem = ("MemoryError" in stderr_tail) or (proc.returncode in (-9, 137, 3221225477))
    return {
        "outcome": "oom" if is_mem else "crashed",
        "returncode": proc.returncode,
        "wall_sec_observed_by_driver": wall_sec,
        "stderr_tail": stderr_tail,
        "stdout_tail": (proc.stdout or "")[-2000:],
    }


def normalize_outcome(raw: dict) -> str:
    o = raw.get("outcome")
    if o == "SUCCESS":
        return "completed"
    if o in ("timed_out",):
        return "timed_out"
    if o in ("oom_cap", "oom"):
        return "oom"
    return "crashed"


def main():
    nextqa_before = _dir_fingerprint(NEXTQA_CACHE_DIR)

    clips_out = []
    fairness_result = None

    for label, dataset, cache_path, video_path, basis in CLIPS:
        print(f"\n=== {label} (dataset={dataset}) ===", flush=True)
        clip_record = {"label": label, "dataset": dataset, "video": str(video_path.relative_to(REPO)),
                        "target_n_selection_basis": basis}

        print(f"[{label}] scene_sparse ...", flush=True)
        ss_raw = run_worker("scene_sparse", cache_path, TMP_DIR / f"{label}_scenesparse.json")
        ss_raw["outcome_normalized"] = normalize_outcome(ss_raw)
        clip_record["scene_sparse"] = ss_raw
        n_survivors = ss_raw.get("n_survivors")
        clip_record["n_survivors"] = n_survivors
        print(f"[{label}] scene_sparse -> {clip_record['scene_sparse']['outcome_normalized']} "
              f"(N={n_survivors}, wall_observed={ss_raw.get('wall_sec_observed_by_driver'):.1f}s)", flush=True)

        print(f"[{label}] flat (cached_frames, wall_cap={WALL_CAP_SEC}s, mem_cap={MEM_CAP_BYTES/1e9:.0f}GB) ...", flush=True)
        flat_raw = run_worker("flat", cache_path, TMP_DIR / f"{label}_flat.json")
        flat_raw["outcome_normalized"] = normalize_outcome(flat_raw)
        clip_record["flat"] = flat_raw
        print(f"[{label}] flat -> {flat_raw['outcome_normalized']} "
              f"(wall_observed={flat_raw.get('wall_sec_observed_by_driver'):.1f}s)", flush=True)

        clips_out.append(clip_record)

        # incremental checkpoint -- long total runtime, don't lose progress
        _write_checkpoint(clips_out, nextqa_before, fairness_result, done=False)

    # ── Fairness check: rebuild flat for FAIRNESS_CHECK_LABEL via an
    # independent ingest (no scene_sparse detour) and compare to the
    # cached_frames flat measurement already collected above. ──────────────
    fc_clip = next(c for c in clips_out if c["label"] == FAIRNESS_CHECK_LABEL)
    fc_cache, fc_video = None, None
    for label, dataset, cache_path, video_path, basis in CLIPS:
        if label == FAIRNESS_CHECK_LABEL:
            fc_cache, fc_video = cache_path, video_path
    print(f"\n=== FAIRNESS CHECK: {FAIRNESS_CHECK_LABEL} flat, independent_ingest ===", flush=True)
    indep_raw = run_worker("flat", fc_cache, TMP_DIR / f"{FAIRNESS_CHECK_LABEL}_flat_independent.json",
                            video=fc_video, source="independent_ingest")
    indep_raw["outcome_normalized"] = normalize_outcome(indep_raw)
    print(f"[fairness] independent_ingest flat -> {indep_raw['outcome_normalized']} "
          f"(wall_observed={indep_raw.get('wall_sec_observed_by_driver'):.1f}s)", flush=True)

    cached_lat = fc_clip["flat"].get("median_total_retrieval_s")
    indep_lat = indep_raw.get("median_total_retrieval_s")
    both_completed = fc_clip["flat"]["outcome_normalized"] == "completed" and indep_raw["outcome_normalized"] == "completed"
    if both_completed and cached_lat and indep_lat:
        rel_diff = abs(cached_lat - indep_lat) / max(cached_lat, indep_lat)
    else:
        rel_diff = None

    fairness_result = {
        "clip": FAIRNESS_CHECK_LABEL,
        "n_survivors_cached_frames": fc_clip["flat"].get("n_survivors"),
        "n_survivors_independent_ingest": indep_raw.get("n_survivors"),
        "median_total_retrieval_s_cached_frames": cached_lat,
        "median_total_retrieval_s_independent_ingest": indep_lat,
        "node_count_cached_frames": fc_clip["flat"].get("node_count"),
        "node_count_independent_ingest": indep_raw.get("node_count"),
        "edge_count_cached_frames": fc_clip["flat"].get("edge_count"),
        "edge_count_independent_ingest": indep_raw.get("edge_count"),
        "relative_latency_diff": rel_diff,
        "both_completed": both_completed,
        "independent_ingest_raw": indep_raw,
        "verdict": (
            "MATCH (within 15% noise band): same-frames construction validated as fair"
            if (rel_diff is not None and rel_diff < 0.15) else
            "MISMATCH: material difference between constructions -- see relative_latency_diff"
            if rel_diff is not None else
            "INCONCLUSIVE: one or both arms did not complete (see outcomes)"
        ),
    }
    print(f"[fairness] verdict: {fairness_result['verdict']}", flush=True)

    nextqa_after = _dir_fingerprint(NEXTQA_CACHE_DIR)
    _write_checkpoint(clips_out, nextqa_before, fairness_result, done=True, nextqa_after=nextqa_after)
    print(f"\nWrote {OUT_RAW}")


def _fit_exponent(points: list[tuple[float, float]]) -> dict | None:
    import math
    pts = [(n, t) for n, t in points if n and t and n > 0 and t > 0]
    if len(pts) < 2:
        return None
    xs = [math.log(n) for n, _ in pts]
    ys = [math.log(t) for _, t in pts]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    slope = num / den
    intercept = mean_y - slope * mean_x
    return {"exponent_k": slope, "log_intercept": intercept, "n_points": n,
            "points_used": [p[0] for p in pts]}


def _write_checkpoint(clips_out, nextqa_before, fairness_result, done: bool, nextqa_after=None):
    ucf_flat_pts = []
    ucf_ss_pts = []
    for c in clips_out:
        if c["dataset"] != "ucf":
            continue
        n = c["n_survivors"]
        if c["flat"]["outcome_normalized"] == "completed":
            ucf_flat_pts.append((n, c["flat"].get("median_total_retrieval_s")))
        if c["scene_sparse"]["outcome_normalized"] == "completed":
            ucf_ss_pts.append((n, c["scene_sparse"].get("median_total_retrieval_s")))

    fits = {
        "flat_ucf_only": _fit_exponent(ucf_flat_pts),
        "scene_sparse_ucf_only": _fit_exponent(ucf_ss_pts),
    }

    provenance = {
        "harness": "scripts/scaling_curve_v2.py + scripts/_scaling_curve_v2_worker.py",
        "query_seed": QUERY_SEED,
        "n_queries_per_clip": N_Q,
        "uniform_watchdog": {
            "wall_time_cap_sec": WALL_CAP_SEC,
            "mem_cap_bytes": MEM_CAP_BYTES,
            "note": "IDENTICAL wall_cap_sec and mem_cap_bytes applied to EVERY flat arm "
                    "in this run, including the VIRAT remeasurement. Wraps index load + "
                    "flat graph build + the full N_WARMUP+n_queries*N_REPS timed query loop.",
        },
        "nextqa_cache_sha256_before": nextqa_before,
        "nextqa_cache_sha256_after": nextqa_after,
        "nextqa_cache_untouched": (nextqa_after is not None and nextqa_before == nextqa_after),
        "fairness_check_clip": FAIRNESS_CHECK_LABEL,
        "run_complete": done,
    }

    out = {"provenance": provenance, "clips": clips_out, "fairness_check": fairness_result, "fits_ucf_only": fits}
    OUT_RAW.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
