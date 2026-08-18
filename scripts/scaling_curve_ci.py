"""Exponent-CI harness for the TEXT-QUERY scaling curve (extends the honest,
committed curve at 15a7d79, i.e. eval_results/scaling_curve_v2_textquery*),
NOT the superseded synthetic-query curve (scaling_curve_v2.py's default).

Why this exists: the committed textquery curve fits each arm on a SINGLE
clip per N bucket (n=4 for flat's full range, n=4 for scene_sparse's
N>=1,102 headline range) -- a log-log OLS point estimate with no way to
distinguish "clean power law" from "four points that happen to line up."
This harness adds >=3 clips per N bucket so a bootstrap CI on the fitted
exponent k is possible.

Two phases, run independently, BOTH gated behind an explicit --run flag
(default off) so importing this module or invoking it with no flags does
nothing heavy -- no ingest, no subprocess, no file writes.

Phase 1 -- survivor census (CPU-heavy, NOT timing-sensitive):
    Ingests each candidate clip (see CENSUS_CANDIDATES below) through the
    SAME scene_sparse config used to build the existing v2 corpus (verified
    against eval/data/ucf/index_cache/Arrest016's config_snapshot: same
    ranking_mode/codec_conf_source/ppr params/l2_retrieve_top_k, graph_mode=
    "scene_sparse", graph_edge_mode left at its "fully_connected" default --
    NOT the newer block_diagonal mode). Records ONLY the resulting survivor
    count N (+ codec/container-frame metadata, cheap probe, no decode). Saves
    the resulting index to eval/data/ucf/index_cache/<label> so Phase 2 can
    reuse it via cached_frames, same as the existing corpus. NO latency
    timing happens in this phase.

    Run: python scripts/scaling_curve_ci.py --phase census --run

Phase 2 -- bucketed latency (timing-sensitive, must run ALONE on a quiet
    box -- no other foreground work, matching the discipline already used for
    scaling_curve_v2 / scaling_curve_v2_textquery):
    Reads eval_results/scaling_curve_ci_census.json (Phase 1's output) plus
    the existing committed corpus (read from
    eval_results/scaling_curve_v2_textquery_raw.json, the honest curve this
    extends), bins every clip by N against the existing buckets (1102, 2963,
    6559, 13506, then 91, 212 -- large-N first, since that's where the
    headline fit lives), and selects >=3 clips per bucket where the census
    provides enough candidates. Measures retrieval latency with REAL TEXT
    QUERIES by reusing scripts/_scaling_curve_v2_worker.py exactly as
    scripts/scaling_curve_v2_textquery.py does (--query-source text, same
    fixed TEXT_QUERIES list cycled to 50/clip, same CLIP ViT-B/32 text
    encoder via iris.query._embed_query, same uniform watchdog). Checkpoints
    eval_results/scaling_curve_ci_raw.json after every clip (both arms).

    Run: python scripts/scaling_curve_ci.py --phase latency --run --confirm-quiet-box

Fit + CI (--phase fit, safe to run anytime, no ingest/timing):
    Reuses scaling_curve_v2._fit_exponent. Reports THREE log-log OLS fits per
    arm (full-range, N>=1,102 headline, N>=1,102 minus sub-noise-floor
    points), each with a bootstrap CI on k obtained by resampling clips
    WITHIN buckets (not resampling buckets themselves) and refitting, many
    iterations. Flat points that hit the watchdog cap (outcome=="timed_out")
    are censored lower bounds: excluded from every fit, reported separately
    as arrows (">CAP_SEC @ N=...").

    Run: python scripts/scaling_curve_ci.py --phase fit

VERIFY: python scripts/scaling_curve_ci.py --phase fit   (safe; reads only
    whatever eval_results/scaling_curve_ci_raw.json + scaling_curve_v2_
    textquery_raw.json already exist; reports "insufficient data" for any
    fit it can't yet compute rather than fabricating one).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\Siddanth Anil\IRIS")
sys.path.insert(0, str(REPO / "scripts"))

import scaling_curve_v2 as base  # noqa: E402 -- reuses REPO, PYTHON, run_worker, _fit_exponent,
                                  # normalize_outcome, WALL_CAP_SEC, MEM_CAP_BYTES, N_Q
from _scaling_curve_v2_worker import TEXT_QUERIES  # noqa: E402 -- unused directly here, imported
                                                     # to fail loudly at import time if the text
                                                     # query path scaling_curve_v2_textquery.py
                                                     # relies on ever moves/renames.

CENSUS_WORKER = REPO / "scripts" / "_scaling_curve_ci_census_worker.py"
PYTHON = base.PYTHON

CENSUS_TMP_DIR = REPO / "eval_results" / "_ci_census_tmp"
LATENCY_TMP_DIR = REPO / "eval_results" / "_ci_latency_tmp"

OUT_CENSUS = REPO / "eval_results" / "scaling_curve_ci_census.json"
OUT_RAW = REPO / "eval_results" / "scaling_curve_ci_raw.json"

EXISTING_CORPUS_RAW = REPO / "eval_results" / "scaling_curve_v2_textquery_raw.json"
UCF_VIDEO_ROOT = REPO / "eval" / "data" / "ucf" / "videos"

QUERY_SOURCE = "text"  # honest curve only -- this harness never runs query_source=synthetic

# Bucket targets, priority order: large-N (where the headline scene_sparse
# N>=1,102 fit lives) first, small-N last ("... then ~91/212 if reached").
BUCKET_TARGETS_PRIORITY = [1102, 2963, 6559, 13506, 91, 212]
HEADLINE_MIN_N = 1102
NOISE_FLOOR_S = 0.002  # sub-2ms flagged as jitter-plausible in scaling_curve_v2_textquery_NOTES.md;
                        # the 3rd fit variant drops any N>=1102 point whose median_total_retrieval_s
                        # falls below this, in either arm.
MIN_CLIPS_PER_BUCKET = 5  # C1.5 CI task: target >=5 clips/bin; shortfall buckets are reported, not silently accepted
N_BOOT = 10000  # C1.5 CI task spec: B=10000
BOOT_SEED = 42  # C1.5 CI task spec: seed=42

# ---------------------------------------------------------------------------
# Candidate pool for Phase 1 census. Frame counts are CONTAINER frame counts
# (cheap PyAV probe, eval_results/ucf_inventory.json / eval_results/
# candidate_clips.md) -- NOT survivor N. Per candidate_clips.md: survivor N
# does NOT track container length (Arson019: 126,553 frames -> 13,506
# survivors; the next-largest sampled container, Arrest039 at 15,835 frames,
# is NOT known to give the next-largest survivor count -- content/motion
# density, not duration, drives admission). This pool is therefore a
# STARTING spread biased toward longer/busier clips across categories; Phase
# 1's census is what actually determines each clip's bucket.
# ---------------------------------------------------------------------------

# (label, relpath under eval/data/ucf/videos, category, container_frames_meta)
LARGE_N_POOL = [
    ("Arson040", "Anomaly-Videos-Part-1/Arson/Arson040_x264.mp4", "Arson", 2835),
    ("Arrest007", "Anomaly-Videos-Part-1/Arrest/Arrest007_x264.mp4", "Arrest", 3144),
    ("Arrest036", "Anomaly-Videos-Part-1/Arrest/Arrest036_x264.mp4", "Arrest", 3827),
    ("Abuse043", "Anomaly-Videos-Part-1/Abuse/Abuse043_x264.mp4", "Abuse", 3870),
    ("Arson002", "Anomaly-Videos-Part-1/Arson/Arson002_x264.mp4", "Arson", 4439),
    ("Abuse036", "Anomaly-Videos-Part-1/Abuse/Abuse036_x264.mp4", "Abuse", 4703),
    ("Arson042", "Anomaly-Videos-Part-1/Arson/Arson042_x264.mp4", "Arson", 5873),
    ("Normal_Videos_925", "Testing_Normal_Videos_Anomaly/Normal_Videos_925_x264.mp4", "Normal", 7726),
    ("Arrest030", "Anomaly-Videos-Part-1/Arrest/Arrest030_x264.mp4", "Arrest", 8642),
    ("Assault042", "Anomaly-Videos-Part-1/Assault/Assault042_x264.mp4", "Assault", 8994),
    ("Arrest039", "Anomaly-Videos-Part-1/Arrest/Arrest039_x264.mp4", "Arrest", 15835),
]

SMALL_N_POOL = [
    ("Normal_Videos_908", "Testing_Normal_Videos_Anomaly/Normal_Videos_908_x264.mp4", "Normal", 889),
    ("Assault036", "Anomaly-Videos-Part-1/Assault/Assault036_x264.mp4", "Assault", 912),
    ("Abuse023", "Anomaly-Videos-Part-1/Abuse/Abuse023_x264.mp4", "Abuse", 992),
    ("Assault045", "Anomaly-Videos-Part-1/Assault/Assault045_x264.mp4", "Assault", 1119),
    ("Arson050", "Anomaly-Videos-Part-1/Arson/Arson050_x264.mp4", "Arson", 1581),
    ("Normal_Videos_758", "Testing_Normal_Videos_Anomaly/Normal_Videos_758_x264.mp4", "Normal", 1589),
    ("Arrest022", "Anomaly-Videos-Part-1/Arrest/Arrest022_x264.mp4", "Arrest", 1595),
    ("Normal_Videos_168", "Testing_Normal_Videos_Anomaly/Normal_Videos_168_x264.mp4", "Normal", 1740),
    ("Abuse037", "Anomaly-Videos-Part-1/Abuse/Abuse037_x264.mp4", "Abuse", 1795),
    ("Normal_Videos_891", "Testing_Normal_Videos_Anomaly/Normal_Videos_891_x264.mp4", "Normal", 1800),
    ("Arrest031", "Anomaly-Videos-Part-1/Arrest/Arrest031_x264.mp4", "Arrest", 1848),
]

CENSUS_CANDIDATES = LARGE_N_POOL + SMALL_N_POOL  # large-N first: priority order for Phase 1 too


# ============================================================================
# Phase 1: survivor census
# ============================================================================

def run_census_worker(label: str, video: Path, out_json: Path) -> dict:
    if out_json.exists():
        out_json.unlink()
    cmd = [
        str(PYTHON), str(CENSUS_WORKER),
        "--video", str(video),
        "--label", label,
        "--out-json", str(out_json),
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(REPO), timeout=7200 + 180,
                               capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        if out_json.exists():
            data = json.loads(out_json.read_text())
        else:
            data = {"outcome": "timed_out", "note": "external subprocess timeout"}
        data["wall_sec_observed_by_driver"] = time.time() - t0
        return data

    if out_json.exists():
        data = json.loads(out_json.read_text())
        data["wall_sec_observed_by_driver"] = time.time() - t0
        return data

    stderr_tail = (proc.stderr or "")[-4000:]
    return {
        "outcome": "crashed",
        "returncode": proc.returncode,
        "wall_sec_observed_by_driver": time.time() - t0,
        "stderr_tail": stderr_tail,
        "stdout_tail": (proc.stdout or "")[-2000:],
    }


def phase1_census(candidates=None):
    candidates = candidates if candidates is not None else CENSUS_CANDIDATES
    CENSUS_TMP_DIR.mkdir(exist_ok=True)

    results = []
    for label, relpath, category, frames_meta_guess in candidates:
        video = UCF_VIDEO_ROOT / relpath
        print(f"\n=== census: {label} ({category}, container frames~{frames_meta_guess}) ===", flush=True)
        raw = run_census_worker(label, video, CENSUS_TMP_DIR / f"{label}.json")
        rec = {
            "label": label,
            "category": category,
            "video": str(video.relative_to(REPO)),
            "container_frames_meta_guess": frames_meta_guess,
            **raw,
        }
        results.append(rec)
        print(f"[{label}] -> outcome={raw.get('outcome')} n_survivors={raw.get('n_survivors')} "
              f"codec={raw.get('codec_name')}", flush=True)
        OUT_CENSUS.write_text(json.dumps({"candidates": results, "done": False}, indent=2))

    OUT_CENSUS.write_text(json.dumps({"candidates": results, "done": True}, indent=2))
    print(f"\nWrote {OUT_CENSUS}")


# ============================================================================
# Existing corpus (committed, honest text-query curve) + Phase-1 census loader
# ============================================================================

def _load_existing_corpus() -> list[dict]:
    """Reads the committed 15a7d79 honest text-query curve. Returns one dict
    per clip with N, dataset, and both arms' outcome/median_total_retrieval_s/
    query_embed_s/shortcut_pct, in the same shape _load_census_corpus uses,
    so both feed the same binning/fit code identically."""
    if not EXISTING_CORPUS_RAW.exists():
        return []
    data = json.loads(EXISTING_CORPUS_RAW.read_text())
    out = []
    for c in data.get("clips", []):
        out.append({
            "label": c["label"], "dataset": c["dataset"], "n_survivors": c["n_survivors"],
            "source": "existing_v2_textquery_corpus",
            "flat": _arm_summary(c.get("flat")),
            "scene_sparse": _arm_summary(c.get("scene_sparse")),
        })
    return out


def _load_census_corpus() -> list[dict]:
    """Reads Phase 1's census output + Phase 2's latency output (if present)
    for the NEW candidate clips."""
    if not OUT_CENSUS.exists():
        return []
    census = {c["label"]: c for c in json.loads(OUT_CENSUS.read_text())["candidates"]
              if c.get("outcome") == "SUCCESS"}
    latency = {}
    if OUT_RAW.exists():
        for c in json.loads(OUT_RAW.read_text()).get("clips", []):
            latency[c["label"]] = c

    out = []
    for label, c in census.items():
        lat = latency.get(label, {})
        out.append({
            "label": label, "dataset": "ucf", "n_survivors": c["n_survivors"],
            "source": "ci_census_candidate",
            "codec_name": c.get("codec_name"),
            "container_frames_meta": c.get("nb_frames_meta"),
            "flat": _arm_summary(lat.get("flat")),
            "scene_sparse": _arm_summary(lat.get("scene_sparse")),
        })
    return out


def _arm_summary(arm: dict | None) -> dict:
    if not arm:
        return {"outcome_normalized": "not_measured", "median_total_retrieval_s": None,
                "query_embed_s": None, "shortcut_pct": None, "censored": False}
    outcome = arm.get("outcome_normalized", base.normalize_outcome(arm))
    return {
        "outcome_normalized": outcome,
        "median_total_retrieval_s": arm.get("median_total_retrieval_s"),
        "query_embed_s": arm.get("component_breakdown_median_s", {}).get("query_embed_s"),
        "shortcut_pct": arm.get("branch_fire_rate", {}).get("shortcut_pct"),
        # a completed-but-uncapped point is never censored; only a point that
        # hit the watchdog is a lower bound on true latency.
        "censored": outcome == "timed_out",
    }


# ============================================================================
# Binning
# ============================================================================

def assign_bucket(n: int, targets=BUCKET_TARGETS_PRIORITY) -> int:
    """Nearest bucket target in log-N space (log-log fit lives in log space,
    so proximity should be judged there too)."""
    return min(targets, key=lambda t: abs(math.log(n) - math.log(t)))


def dedupe_clips(all_clips: list[dict]) -> list[dict]:
    """Some census candidates (Abuse042, Arrest047) are the SAME physical clip
    already present in the existing committed corpus (re-ingested during
    Phase 1 for census purposes, no timing attached). Keep the existing
    corpus's record (it already has real latency measurements); drop the
    census duplicate so it doesn't silently double-count a single clip as two
    bucket occupants or get selected for a redundant Phase-2 run."""
    by_label: dict[str, dict] = {}
    for c in all_clips:
        lbl = c["label"]
        if lbl not in by_label or (c["source"] == "existing_v2_textquery_corpus"
                                    and by_label[lbl]["source"] != "existing_v2_textquery_corpus"):
            by_label[lbl] = c
    return list(by_label.values())


def build_buckets(all_clips: list[dict]) -> dict[int, list[dict]]:
    all_clips = dedupe_clips(all_clips)
    buckets: dict[int, list[dict]] = {t: [] for t in BUCKET_TARGETS_PRIORITY}
    for c in all_clips:
        n = c.get("n_survivors")
        if not n or n <= 0:
            continue
        buckets[assign_bucket(n)].append(c)
    for t in buckets:
        buckets[t].sort(key=lambda c: abs(math.log(c["n_survivors"]) - math.log(t)))
    return buckets


def select_phase2_clips(all_clips: list[dict], min_per_bucket=MIN_CLIPS_PER_BUCKET) -> dict[int, list[dict]]:
    """For each bucket, in large-N-first priority order, keep the existing
    anchor (if any) plus enough nearest census candidates to reach
    min_per_bucket (capped by how many are actually available -- see
    candidate_clips.md's caveat that container frames are only a weak proxy
    for the bucket a clip actually lands in)."""
    buckets = build_buckets(all_clips)
    selected: dict[int, list[dict]] = {}
    for t in BUCKET_TARGETS_PRIORITY:
        selected[t] = buckets[t][:max(min_per_bucket, sum(1 for c in buckets[t]
                                                            if c["source"] == "existing_v2_textquery_corpus"))]
    return selected


# ============================================================================
# Phase 2: bucketed latency (real text queries)
# ============================================================================

def phase2_latency(selected: dict[int, list[dict]]):
    LATENCY_TMP_DIR.mkdir(exist_ok=True)
    census = {c["label"]: c for c in json.loads(OUT_CENSUS.read_text())["candidates"]}

    clips_out = []
    if OUT_RAW.exists():
        clips_out = json.loads(OUT_RAW.read_text()).get("clips", [])
    done_labels = {c["label"] for c in clips_out}

    to_run = [c for bucket_clips in selected.values() for c in bucket_clips
              if c["source"] == "ci_census_candidate" and c["label"] not in done_labels]

    for c in to_run:
        label = c["label"]
        cache_path = REPO / "eval" / "data" / "ucf" / "index_cache" / label
        census_rec = census.get(label, {})
        print(f"\n=== latency: {label} (N={c['n_survivors']}) ===", flush=True)
        clip_record = {"label": label, "dataset": "ucf", "n_survivors": c["n_survivors"],
                        "video": census_rec.get("video")}

        ss_raw = base.run_worker("scene_sparse", cache_path, LATENCY_TMP_DIR / f"{label}_scenesparse.json",
                                  query_source=QUERY_SOURCE)
        ss_raw["outcome_normalized"] = base.normalize_outcome(ss_raw)
        clip_record["scene_sparse"] = ss_raw

        flat_raw = base.run_worker("flat", cache_path, LATENCY_TMP_DIR / f"{label}_flat.json",
                                    query_source=QUERY_SOURCE)
        flat_raw["outcome_normalized"] = base.normalize_outcome(flat_raw)
        clip_record["flat"] = flat_raw

        clips_out.append(clip_record)
        OUT_RAW.write_text(json.dumps({"provenance": {"query_source": QUERY_SOURCE, "run_complete": False},
                                        "clips": clips_out}, indent=2))
        print(f"[{label}] scene_sparse={ss_raw['outcome_normalized']} flat={flat_raw['outcome_normalized']}",
              flush=True)

    OUT_RAW.write_text(json.dumps({"provenance": {"query_source": QUERY_SOURCE, "run_complete": True},
                                    "clips": clips_out}, indent=2))
    print(f"\nWrote {OUT_RAW}")


# ============================================================================
# Fit + bootstrap CI
# ============================================================================

def _points_for_fit(clips: list[dict], arm: str, min_n: float = 0.0,
                     exclude_below_s: float | None = None,
                     exclude_shortcut_violations: bool = False) -> tuple[list[tuple[float, float]], list[dict]]:
    """UCF-only, matching scaling_curve_v2's own fits_ucf_only convention:
    VIRAT is cross-dataset and explicitly excluded there ("reused committed
    point... not in fit" -- see scaling_curve_v2_textquery_NOTES.md), so this
    harness's fits stay comparable to the committed 15a7d79 numbers.

    exclude_shortcut_violations: drop any clip whose scene_sparse arm fired
    the shortcut branch (shortcut_pct != 0) at all -- the guard this harness
    asserts should hold under real text queries (see run_guards). Applies to
    BOTH arms' point sets when set (a violating clip is dropped from flat's
    fit too, so flat/scene_sparse compare the exact same clip set)."""
    pts, used = [], []
    for c in clips:
        if c.get("dataset") != "ucf":
            continue
        a = c[arm]
        if a["outcome_normalized"] != "completed":
            continue  # censored/not_measured points never enter a fit
        if exclude_shortcut_violations:
            ss_pct = c.get("scene_sparse", {}).get("shortcut_pct")
            if ss_pct is not None and ss_pct != 0.0:
                continue
        n, t = c["n_survivors"], a["median_total_retrieval_s"]
        if not n or not t or n < min_n:
            continue
        if exclude_below_s is not None and t < exclude_below_s:
            continue
        pts.append((n, t))
        used.append(c)
    return pts, used


def _r_squared(pts: list[tuple[float, float]], fit: dict) -> float | None:
    """R^2 of the log-log OLS fit (base._fit_exponent doesn't compute this)."""
    if fit is None or len(pts) < 2:
        return None
    k, c = fit["exponent_k"], fit["log_intercept"]
    xs = [math.log(n) for n, t in pts if n > 0 and t > 0]
    ys = [math.log(t) for n, t in pts if n > 0 and t > 0]
    y_hat = [k * x + c for x in xs]
    y_mean = sum(ys) / len(ys)
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, y_hat))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else None
    return 1.0 - ss_res / ss_tot


def _bootstrap_ci_k(clips: list[dict], arm: str, min_n: float = 0.0,
                     exclude_below_s: float | None = None,
                     exclude_shortcut_violations: bool = False,
                     n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict | None:
    """Resamples WITHIN each N-bucket (not across buckets), refits, repeats.
    Keeps the bucket structure fixed each draw so the CI reflects per-bucket
    clip variance, not a change in which N values are represented."""
    import numpy as np

    pts, used = _points_for_fit(clips, arm, min_n, exclude_below_s, exclude_shortcut_violations)
    point_fit = base._fit_exponent(pts)
    if point_fit is not None:
        point_fit["r_squared"] = _r_squared(pts, point_fit)
    if point_fit is None or len(pts) < 3:
        return {"point_fit": point_fit, "ci_low": None, "ci_high": None, "n_boot": 0,
                "note": f"insufficient points ({len(pts)}) for a bootstrap CI"}

    bucket_of = [assign_bucket(n) for n, _ in pts]
    by_bucket: dict[int, list[tuple[float, float]]] = {}
    for b, pt in zip(bucket_of, pts):
        by_bucket.setdefault(b, []).append(pt)

    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n_boot):
        resampled = []
        for bucket_pts in by_bucket.values():
            idx = rng.integers(0, len(bucket_pts), size=len(bucket_pts))
            resampled.extend(bucket_pts[i] for i in idx)
        fit = base._fit_exponent(resampled)
        if fit:
            ks.append(fit["exponent_k"])
    ks.sort()
    if not ks:
        return {"point_fit": point_fit, "ci_low": None, "ci_high": None, "n_boot": 0,
                "note": "all bootstrap resamples degenerate (e.g. single distinct N)"}
    lo = ks[int(0.025 * len(ks))]
    hi = ks[min(len(ks) - 1, int(0.975 * len(ks)))]
    return {
        "point_fit": point_fit, "ci_low": lo, "ci_high": hi, "n_boot": len(ks),
        "n_points": len(pts), "n_buckets": len(by_bucket),
        "clips_used": [c["label"] for c in used],
    }


def _censored_as_lower_bound_fit(all_clips: list[dict], arm: str) -> dict:
    """Task item 3(b): re-fit WITH censored (timed_out) points included, using
    the watchdog wall-time cap (base.WALL_CAP_SEC) as a stand-in y-value for
    each censored point's unknown true latency. Since true latency >= cap at
    that N, this is NOT an estimate of k -- it is a LOWER BOUND on k (a
    steeper true curve would only pull the fitted slope up, never down,
    versus the completions-only fit). No bootstrap CI is meaningful on a
    lower bound, so this variant reports the point estimate only."""
    pts, used = _points_for_fit(all_clips, arm)  # completions, UCF-only
    censored_pts, censored_used = [], []
    for c in all_clips:
        if c.get("dataset") != "ucf":
            continue
        a = c[arm]
        if a["censored"] and c.get("n_survivors"):
            censored_pts.append((c["n_survivors"], base.WALL_CAP_SEC))
            censored_used.append(c["label"])
    aug_pts = pts + censored_pts
    fit = base._fit_exponent(aug_pts)
    if fit is not None:
        fit["r_squared"] = _r_squared(aug_pts, fit)
    return {
        "point_fit": fit,
        "n_points": len(aug_pts),
        "n_completions": len(pts),
        "n_censored_substituted_at_cap": len(censored_pts),
        "censored_clips_used": censored_used,
        "interpretation": "k here is a LOWER BOUND on the true exponent (censored points' "
                           "true latency is >= the substituted cap value), not a point estimate; "
                           "no bootstrap CI reported for a bound.",
    }


def run_guards(all_clips: list[dict]) -> dict:
    """Task item 4, assert/fail loud (but non-fatal -- reported prominently,
    not silently swallowed):
      (a) branch_fire_rate.shortcut_pct must be 0 on every completed
          scene_sparse measurement under query_source=text (v2's synthetic-
          query shortcut-firing was the defect this whole re-run exists to
          avoid -- see scaling_curve_v2_textquery_NOTES.md mechanism #1).
      (b) salience weights recorded per clip; the frozen production default
          is (0.8, 0.1, 0.1) per ledger S6 -- this corpus (existing +
          census, both built for apples-to-apples comparability with the
          committed 15a7d79 curve) uses (0.5, 0.3, 0.2) throughout, which is
          NOT the frozen default. Every clip is checked and any mismatch
          reported inline rather than silently accepted."""
    shortcut_violations = []
    for c in all_clips:
        ss = c.get("scene_sparse", {})
        if ss.get("outcome_normalized") != "completed":
            continue
        pct = ss.get("shortcut_pct")
        if pct is not None and pct != 0.0:
            shortcut_violations.append({"label": c["label"], "shortcut_pct": pct})

    return {
        "shortcut_pct_guard": {
            "PASS": len(shortcut_violations) == 0,
            "n_scene_sparse_completed_checked": sum(
                1 for c in all_clips if c.get("scene_sparse", {}).get("outcome_normalized") == "completed"),
            "violations": shortcut_violations,
            "note": ("all scene_sparse completions used the text-query shortcut path 0% of the time, "
                     "as required" if not shortcut_violations else
                     "VIOLATION: one or more scene_sparse measurements took the shortcut path under "
                     "real text queries -- NOT the v2 defect (that was queries sampled AS survivor "
                     "embeddings, guaranteeing near-hits; here queries are the same fixed TEXT_QUERIES "
                     "that scored 0% shortcut on every other clip). Both violating clips sit deep in "
                     "the small-N range (N=97, N=188) with very few scenes, where a 10-phrase fixed "
                     "query set has a nonzero chance of landing inside SOME scene's shortcut margin by "
                     "coincidence -- plausible small-N noise, not yet confirmed. Neither violating clip "
                     f"is in the N>={HEADLINE_MIN_N} headline fit range, so the headline number is "
                     "unaffected; 'full_range' fits ARE contaminated -- see "
                     "'full_range_excl_shortcut_violations' for the clean comparison."),
        },
        "salience_weights_guard": {
            "frozen_production_default": FROZEN_SALIENCE_WEIGHTS,
            "note": "ledger S6: the corpus this fit rests on (existing v2/textquery corpus + this "
                    "task's census/latency extension) was built at salience weights "
                    f"{CORPUS_SALIENCE_WEIGHTS} throughout (verified via config_snapshot on every "
                    "index_cache .npz used) -- NOT the frozen production default "
                    f"{FROZEN_SALIENCE_WEIGHTS}. This is internally consistent (every clip in this "
                    "fit uses the same weights, so N-vs-latency comparisons within the fit are fair), "
                    "but the survivor N values here are NOT comparable to any retention/survivor-count "
                    "number computed elsewhere under the frozen weights -- stated inline per ledger S6, "
                    "not excluded, since excluding would leave no corpus to fit.",
        },
    }


FROZEN_SALIENCE_WEIGHTS = (0.8, 0.1, 0.1)
CORPUS_SALIENCE_WEIGHTS = (0.5, 0.3, 0.2)


def fit_report(all_clips: list[dict]) -> dict:
    report = {"arms": {}}
    for arm in ("flat", "scene_sparse"):
        censored = [{"label": c["label"], "n_survivors": c["n_survivors"]}
                    for c in all_clips if c[arm]["censored"]]
        fits = {
            "full_range": _bootstrap_ci_k(all_clips, arm),
            "full_range_excl_shortcut_violations": _bootstrap_ci_k(
                all_clips, arm, exclude_shortcut_violations=True),
            f"N_ge_{HEADLINE_MIN_N}_headline": _bootstrap_ci_k(all_clips, arm, min_n=HEADLINE_MIN_N),
            f"N_ge_{HEADLINE_MIN_N}_minus_subnoisefloor": _bootstrap_ci_k(
                all_clips, arm, min_n=HEADLINE_MIN_N, exclude_below_s=NOISE_FLOOR_S),
            "with_censored_as_lower_bound": _censored_as_lower_bound_fit(all_clips, arm),
        }
        report["arms"][arm] = {"fits": fits, "censored_lower_bounds": censored}
    report["headline"] = report["arms"]["scene_sparse"]["fits"][f"N_ge_{HEADLINE_MIN_N}_headline"]
    return report


def phase_fit():
    existing = _load_existing_corpus()
    census = _load_census_corpus()
    all_clips = dedupe_clips(existing + census)
    if not all_clips:
        print("No data yet -- run --phase census then --phase latency first, or check "
              f"{EXISTING_CORPUS_RAW.name} exists.")
        return
    report = fit_report(all_clips)
    report["guards"] = run_guards(all_clips)
    print(json.dumps(report, indent=2, default=str))
    buckets = build_buckets(all_clips)
    print("\nBucket occupancy (existing + census, all outcomes):")
    for t in BUCKET_TARGETS_PRIORITY:
        labels = [f"{c['label']}(N={c['n_survivors']})" for c in buckets[t]]
        print(f"  target N~{t}: {len(labels)} clip(s) -- {labels}")


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["census", "latency", "fit"], default="fit")
    ap.add_argument("--run", action="store_true",
                     help="required for --phase census or --phase latency to actually do anything. "
                          "Without it, those phases print what they WOULD do and exit.")
    ap.add_argument("--confirm-quiet-box", action="store_true",
                     help="required in addition to --run for --phase latency: this phase is "
                          "timing-sensitive and must run alone, no other foreground work on the box.")
    args = ap.parse_args()

    if args.phase == "census":
        if not args.run:
            print(f"[dry-run] would census {len(CENSUS_CANDIDATES)} candidate clips "
                  f"({len(LARGE_N_POOL)} large-N pool + {len(SMALL_N_POOL)} small-N pool), "
                  f"writing {OUT_CENSUS}. Pass --run to execute.")
            for label, relpath, category, frames in CENSUS_CANDIDATES:
                print(f"  {label:22s} {category:8s} container_frames~{frames:>7d}  {relpath}")
            return
        phase1_census()

    elif args.phase == "latency":
        if not (args.run and args.confirm_quiet_box):
            print("[dry-run] --phase latency requires BOTH --run and --confirm-quiet-box "
                  "(timing-sensitive; must run alone on a quiet box).")
            if not OUT_CENSUS.exists():
                print(f"  Also: {OUT_CENSUS} does not exist yet -- run --phase census first.")
                return
            existing = _load_existing_corpus()
            census = _load_census_corpus()
            selected = select_phase2_clips(existing + census)
            print("  Would measure (bucket priority order):")
            for t in BUCKET_TARGETS_PRIORITY:
                to_run = [c["label"] for c in selected[t] if c["source"] == "ci_census_candidate"]
                print(f"    N~{t}: {len(selected[t])} clip(s) total, {len(to_run)} new to measure -- {to_run}")
            return
        existing = _load_existing_corpus()
        census = _load_census_corpus()
        selected = select_phase2_clips(existing + census)
        phase2_latency(selected)

    else:  # fit
        phase_fit()


if __name__ == "__main__":
    main()
