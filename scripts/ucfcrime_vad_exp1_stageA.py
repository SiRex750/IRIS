"""Step 1d verification + Step 2/3 (Stage A) for UCF-Crime VAD Experiment 1.

The official annotation file (Temporal_Anomaly_Annotation_ForTestVideos, 290
rows) was located at:
  eval/data/ucf/annotations/Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate/Temporal_Anomaly_Annotation.txt
after the user extracted it locally (not present anywhere reachable before that).

Of the 290 official test videos, only 169 have a matching local .mp4 file
under eval/data/ucf/videos/ (19 anomalous -- 2 Abuse, 5 Arrest, 9 Arson,
3 Assault -- plus all 150 Normal test videos). The other 121 rows (9 entire
categories: Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting,
Shoplifting, Stealing, Vandalism) have NO local video file and are excluded
from every computation below -- reported, not silently dropped, and not
downloaded per instructions.

This means every AUC number in this script's output is a PARTIAL-CORPUS
result over 169/290 official test videos, NOT the standard 290-video pooled
AUC that LAVAD/EventVAD/ZS-CLIP report. It is not directly comparable to
those cited numbers and the report must say so explicitly every time it is
shown.

Protocol (Step 2), applied here:
  - Ground truth: 1 inside an annotated anomaly span, 0 elsewhere; all-0 for
    Normal videos.
  - Frame-level Stage-A score = ActionScoreModule.score_all() on codec
    features (packet_size, motion_magnitude, luma_entropy), frozen weights
    from tuning/frozen_state.json, computed only on retained-tier frames
    (I_FRAME/PEAK/SALIENT/CANDIDATE -- the frames iris.charon_v.parse_video
    returns in `output_frames`).
  - Every non-retained (SKIP-tier) frame is assigned its nearest **preceding**
    retained frame's score (piecewise-constant hold-forward); frames before
    the first retained frame hold the first retained frame's score backward
    (stated explicitly, not hidden).
  - Retention rate is measured per video, not assumed.
  - AUC reported over (a) all frames (with hold-forward fill) and (b)
    retained frames only (no fill), pooled and per-video macro.

Writes:
  tuning/ucfcrime_vad_exp1/annotation_validation.json
  tuning/ucfcrime_vad_exp1/stageA_results.json
  tuning/ucfcrime_vad_exp1/stageA_per_video.csv
  tuning/ucfcrime_vad_exp1/per_frame_ground_truth_scores/<video>.csv
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_PATH = (
    REPO_ROOT / "eval" / "data" / "ucf" / "annotations"
    / "Temporal_Anomaly_Annotation_For_Testing_Videos" / "Txt_formate"
    / "Temporal_Anomaly_Annotation.txt"
)
VIDEO_ROOT = REPO_ROOT / "eval" / "data" / "ucf" / "videos"
FROZEN_STATE_PATH = REPO_ROOT / "tuning" / "frozen_state.json"
OUT_DIR = REPO_ROOT / "tuning" / "ucfcrime_vad_exp1"
GT_SCORE_DIR = OUT_DIR / "per_frame_ground_truth_scores"


def parse_annotation_file(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        name, cls = parts[0], parts[1]
        nums = [int(x) for x in parts[2:6]]
        spans = []
        for s, e in zip(nums[0::2], nums[1::2]):
            if s != -1 and e != -1:
                spans.append((s, e))
        rows.append({"video_name": name, "class": cls, "spans": spans})
    return rows


def build_ground_truth(n_frames: int, spans: list[tuple[int, int]]) -> np.ndarray:
    gt = np.zeros(n_frames, dtype=np.uint8)
    for s, e in spans:
        s_c = max(0, min(s, n_frames - 1))
        e_c = max(0, min(e, n_frames - 1))
        gt[s_c:e_c + 1] = 1
    return gt


def main() -> None:
    if not ANNOTATION_PATH.exists():
        print(f"BLOCKER: {ANNOTATION_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    frozen = json.loads(FROZEN_STATE_PATH.read_text())["frozen"]

    rows = parse_annotation_file(ANNOTATION_PATH)
    assert len(rows) == 290, f"expected 290 annotation rows, got {len(rows)}"
    unique_names = {r["video_name"] for r in rows}
    assert len(unique_names) == 290, "duplicate video_name in annotation file"
    n_normal = sum(1 for r in rows if r["class"] == "Normal")
    n_anom = len(rows) - n_normal
    assert n_normal == 150 and n_anom == 140, f"expected 140/150 split, got {n_anom}/{n_normal}"

    local_files = {p.name: p for p in VIDEO_ROOT.rglob("*.mp4")}

    matched_rows = [r for r in rows if r["video_name"] in local_files]
    missing_rows = [r for r in rows if r["video_name"] not in local_files]

    from collections import Counter
    missing_by_class = Counter(r["class"] for r in missing_rows)
    matched_by_class = Counter(r["class"] for r in matched_rows)

    annotation_validation = {
        "annotation_path": str(ANNOTATION_PATH.relative_to(REPO_ROOT)),
        "total_rows": len(rows),
        "unique_video_names": len(unique_names),
        "n_anomalous_rows": n_anom,
        "n_normal_rows": n_normal,
        "matches_official_290_140_150_spec": True,
        "n_matched_to_local_video_file": len(matched_rows),
        "n_missing_local_video_file": len(missing_rows),
        "matched_by_class": dict(matched_by_class),
        "missing_by_class": dict(missing_by_class),
        "missing_video_names": [r["video_name"] for r in missing_rows],
        "note": (
            f"{len(matched_rows)}/290 official test videos have a local .mp4 "
            f"file; the other {len(missing_rows)}/290 (spanning 9 entirely "
            "absent categories) are excluded from every AUC computation in "
            "this run -- listed here, not silently dropped, not downloaded."
        ),
        "frame_index_out_of_bounds": [],  # filled in below
    }

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

    GT_SCORE_DIR.mkdir(parents=True, exist_ok=True)

    per_video_results = []
    pooled_all_scores, pooled_all_gt = [], []
    pooled_retained_scores, pooled_retained_gt = [], []
    per_video_auc_all = {}  # video_name -> auc (all-frame, propagated)
    decode_failures = []

    t_start = time.time()
    for i, row in enumerate(matched_rows):
        name = row["video_name"]
        path = local_files[name]
        print(f"[{i + 1}/{len(matched_rows)}] {name} ({row['class']})", file=sys.stderr)
        try:
            output_frames, stats, raw_records = charon_v.parse_video(
                str(path), return_stats=True, return_raw=True, full_decode=False,
            )
        except Exception as e:
            decode_failures.append({"video_name": name, "error": f"{type(e).__name__}: {e}"})
            continue

        n_frames = stats["total"]

        # Ground-truth bound check (Step 1d): report every mismatch, don't clamp silently.
        for s, e in row["spans"]:
            if s >= n_frames or e >= n_frames:
                annotation_validation["frame_index_out_of_bounds"].append({
                    "video_name": name,
                    "span": [s, e],
                    "decoded_n_frames": n_frames,
                })

        gt = build_ground_truth(n_frames, row["spans"])

        # Retained-tier scores from real features. raw_records is index-aligned
        # 1:1 with frame_idx (parse_video guarantees this via its size-match
        # assertion), so pull real packet_size/motion/luma straight from it.
        retained_idx_set = {f["frame_idx"] for f in output_frames}
        scored_input = [
            {
                "frame_idx": r["frame_idx"],
                "packet_size": r.get("packet_size", 0.0),
                "motion_magnitude": r.get("motion_magnitude", 0.0),
                "luma_entropy": r.get("luma_entropy", 0.0),
            }
            for r in raw_records
            if r["frame_idx"] in retained_idx_set
        ]
        scored_input.sort(key=lambda r: r["frame_idx"])
        scored = action_module.score_all(scored_input)
        retained_scores_by_idx = {s["frame_idx"]: s["action_score"] for s in scored}

        # Hold-forward fill for all frames.
        full_scores = np.zeros(n_frames, dtype=np.float64)
        retained_sorted_idx = sorted(retained_scores_by_idx.keys())
        if retained_sorted_idx:
            first_val = retained_scores_by_idx[retained_sorted_idx[0]]
            full_scores[:retained_sorted_idx[0]] = first_val  # hold first value backward
            last_val = first_val
            ptr = 0
            for fi in range(n_frames):
                if ptr < len(retained_sorted_idx) and fi == retained_sorted_idx[ptr]:
                    last_val = retained_scores_by_idx[fi]
                    ptr += 1
                full_scores[fi] = last_val
        retention_pct = 100.0 * len(retained_sorted_idx) / n_frames if n_frames else 0.0

        # Per-video AUC (all-frame, propagated) -- only meaningful if there's
        # at least one positive frame (i.e. an anomalous video).
        auc_all = None
        if gt.sum() > 0 and gt.sum() < n_frames:
            auc_all = float(roc_auc_score(gt, full_scores))
            per_video_auc_all[name] = auc_all

        pooled_all_scores.append(full_scores)
        pooled_all_gt.append(gt)
        if retained_sorted_idx:
            pooled_retained_scores.append(np.array([retained_scores_by_idx[fi] for fi in retained_sorted_idx]))
            pooled_retained_gt.append(gt[retained_sorted_idx])

        # Write per-frame CSV (frame_idx, ground_truth, action_score_propagated, is_retained_tier).
        csv_path = GT_SCORE_DIR / f"{Path(name).stem}.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "ground_truth", "action_score_propagated", "is_retained_tier"])
            for fi in range(n_frames):
                w.writerow([fi, int(gt[fi]), full_scores[fi], int(fi in retained_idx_set)])

        per_video_results.append({
            "video_name": name,
            "class": row["class"],
            "n_frames_decoded": n_frames,
            "n_spans": len(row["spans"]),
            "n_positive_frames": int(gt.sum()),
            "retention_pct": retention_pct,
            "auc_all_frames_propagated": auc_all,
            "per_frame_csv": str(csv_path.relative_to(REPO_ROOT)),
        })

    elapsed = time.time() - t_start
    annotation_validation["decode_failures"] = decode_failures
    (OUT_DIR / "annotation_validation.json").write_text(json.dumps(annotation_validation, indent=2))

    # Pooled (micro) AUC -- all frames, propagated.
    pooled_all_scores_cat = np.concatenate(pooled_all_scores) if pooled_all_scores else np.array([])
    pooled_all_gt_cat = np.concatenate(pooled_all_gt) if pooled_all_gt else np.array([])
    pooled_auc_all = float(roc_auc_score(pooled_all_gt_cat, pooled_all_scores_cat)) if pooled_all_gt_cat.sum() > 0 else None

    # Pooled (micro) AUC -- retained frames only, no fill.
    pooled_retained_scores_cat = np.concatenate(pooled_retained_scores) if pooled_retained_scores else np.array([])
    pooled_retained_gt_cat = np.concatenate(pooled_retained_gt) if pooled_retained_gt else np.array([])
    pooled_auc_retained = float(roc_auc_score(pooled_retained_gt_cat, pooled_retained_scores_cat)) if pooled_retained_gt_cat.sum() > 0 else None

    # Macro AUC: mean of per-video AUC over anomalous videos only (exclude normal --
    # AUC undefined, no positives).
    macro_aucs = list(per_video_auc_all.values())
    macro_auc = float(np.mean(macro_aucs)) if macro_aucs else None
    n_excluded_normal = sum(1 for r in per_video_results if r["auc_all_frames_propagated"] is None)

    # Per-category pooled AUC: each anomaly category's matched videos + all matched Normal videos.
    normal_scores = [np.array(r) for r, res in zip(pooled_all_scores, per_video_results) if res["class"] == "Normal"]
    normal_gts = [np.array(r) for r, res in zip(pooled_all_gt, per_video_results) if res["class"] == "Normal"]
    per_category_auc = {}
    for cls in sorted({r["class"] for r in per_video_results if r["class"] != "Normal"}):
        cls_scores = [s for s, res in zip(pooled_all_scores, per_video_results) if res["class"] == cls]
        cls_gts = [g for g, res in zip(pooled_all_gt, per_video_results) if res["class"] == cls]
        combined_scores = np.concatenate(cls_scores + normal_scores)
        combined_gt = np.concatenate(cls_gts + normal_gts)
        if combined_gt.sum() > 0:
            per_category_auc[cls] = {
                "pooled_auc_vs_all_matched_normal": float(roc_auc_score(combined_gt, combined_scores)),
                "n_videos": len(cls_scores),
            }

    retention_pcts = [r["retention_pct"] for r in per_video_results]

    results = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "IMPORTANT_CAVEAT": (
            "This AUC is computed over 169/290 official UCF-Crime test videos "
            "(19 anomalous + all 150 normal) -- the 121 videos from 9 entirely "
            "absent categories are excluded. It is NOT the standard 290-video "
            "pooled benchmark number and is NOT directly comparable to the "
            "cited ZS CLIP / ZS ImageBind / LAVAD / EventVAD figures, which "
            "were computed over the full 290-video set."
        ),
        "frozen_config_used": frozen,
        "n_videos_scored": len(per_video_results),
        "n_decode_failures": len(decode_failures),
        "decode_failures": decode_failures,
        "elapsed_s": elapsed,
        "retention_pct_mean": float(np.mean(retention_pcts)) if retention_pcts else None,
        "retention_pct_min": float(np.min(retention_pcts)) if retention_pcts else None,
        "retention_pct_max": float(np.max(retention_pcts)) if retention_pcts else None,
        "pooled_auc_all_frames_propagated": pooled_auc_all,
        "pooled_auc_retained_frames_only": pooled_auc_retained,
        "macro_auc_anomalous_videos_only": macro_auc,
        "n_anomalous_videos_in_macro_avg": len(macro_aucs),
        "n_normal_videos_excluded_from_macro": n_excluded_normal,
        "per_category_pooled_auc_vs_matched_normal": per_category_auc,
        "interpretation_gate": (
            "gt_0.65_codec_native_real" if (pooled_auc_all or 0) > 0.65 else
            "0.53_to_0.65_weak_zs_clip_level" if (pooled_auc_all or 0) >= 0.53 else
            "approx_0.50_premise_dead"
        ) if pooled_auc_all is not None else "undefined_no_positive_frames",
    }

    (OUT_DIR / "stageA_results.json").write_text(json.dumps(results, indent=2))

    fieldnames = sorted({k for r in per_video_results for k in r.keys()})
    with (OUT_DIR / "stageA_per_video.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_video_results:
            w.writerow(r)

    print(json.dumps(results, indent=2))
    print(f"Wrote {OUT_DIR / 'stageA_results.json'}, {OUT_DIR / 'stageA_per_video.csv'}, "
          f"{OUT_DIR / 'annotation_validation.json'}")


if __name__ == "__main__":
    main()
