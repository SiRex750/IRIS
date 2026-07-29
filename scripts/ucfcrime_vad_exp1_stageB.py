"""Step 4 (Stage B) for UCF-Crime VAD Experiment 1.

Stage A (scripts/ucfcrime_vad_exp1_stageA.py) landed pooled AUC = 0.7232 over
the 169/290-video available subset, clearing the task's own >0.65 gate --
with the caveat that this is a partial-corpus result, not the standard
290-video benchmark. Per the task's decision rule, Stage B is attempted.

FORMULATION (fixed here, before running, per instructions -- "node isolation",
one of the three examples the task text offers):

    score_B(frame) = 1 - rank_percentile(pagerank_score) among that video's own
    scene-sparse graph nodes (i.e. the retained/output frames IRIS indexes
    under retrieval_strategy="hybrid" -- ALL retained-tier frames, per the
    frozen config). A frame whose PageRank rank is low (poorly connected /
    structurally isolated in its own video's graph) scores HIGH on this
    anomaly axis. Rank-percentile (not raw PageRank) is used so scores are
    comparable in [0, 1] across videos of very different graph sizes.

Non-retained (SKIP-tier) frames get the same piecewise-constant hold-forward
treatment as Stage A (§ Step 2), using the SAME Stage-A ground-truth/frame
bookkeeping already written to
tuning/ucfcrime_vad_exp1/per_frame_ground_truth_scores/<video>.csv (no
re-decode needed for ground truth or Stage-A scores; only the scene-sparse
graph + PageRank computation, which requires CLIP embeddings, is new work
here).

Combination: fixed 50/50 average of Stage A and Stage B (both already in
[0, 1] by construction) -- no fusion-weight sweep, as instructed.

Writes:
  tuning/ucfcrime_vad_exp1/stageB_results.json
  tuning/ucfcrime_vad_exp1/stageB_per_video.csv
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
OUT_DIR = REPO_ROOT / "tuning" / "ucfcrime_vad_exp1"
GT_SCORE_DIR = OUT_DIR / "per_frame_ground_truth_scores"
FROZEN_STATE_PATH = REPO_ROOT / "tuning" / "frozen_state.json"
VIDEO_ROOT = REPO_ROOT / "eval" / "data" / "ucf" / "videos"


def main() -> None:
    frozen = json.loads(FROZEN_STATE_PATH.read_text())["frozen"]
    stageA_per_video = list(csv.DictReader((OUT_DIR / "stageA_per_video.csv").open()))
    local_files = {p.name: p for p in VIDEO_ROOT.rglob("*.mp4")}

    import iris.ingest as ingest
    from iris.iris_config import IRISConfig
    from iris.ingest import _rank_percentile

    cfg = IRISConfig(
        packet_size_weight=frozen["packet_size_weight"],
        motion_weight=frozen["motion_weight"],
        luma_entropy_weight=frozen["luma_entropy_weight"],
        peak_distance=frozen["peak_distance"],
        peak_prominence=frozen["peak_prominence"],
        persistence_threshold=frozen["persistence_threshold"],
        max_prominence=frozen["max_prominence"],
        retrieval_strategy=frozen["retrieval_strategy"],
        l2_retrieve_top_k=frozen["l2_retrieve_top_k"],
        graph_mode="scene_sparse",
    )

    per_video_results = []
    pooled_B_scores, pooled_gt = [], []
    pooled_combined_scores = []
    per_video_auc_B = {}
    per_video_auc_combined = {}
    ingest_failures = []

    t_start = time.time()
    for i, row in enumerate(stageA_per_video):
        name = row["video_name"]
        cls = row["class"]
        n_frames = int(row["n_frames_decoded"])
        print(f"[{i + 1}/{len(stageA_per_video)}] {name} ({cls})", file=sys.stderr)

        gt_csv_path = REPO_ROOT / row["per_frame_csv"]
        gt_rows = list(csv.DictReader(gt_csv_path.open()))
        gt = np.array([int(r["ground_truth"]) for r in gt_rows], dtype=np.uint8)
        stageA_scores = np.array([float(r["action_score_propagated"]) for r in gt_rows], dtype=np.float64)
        assert len(gt) == n_frames, f"{name}: gt length {len(gt)} != n_frames {n_frames}"

        try:
            idx = ingest.ingest(str(local_files[name]), config=cfg)
        except Exception as e:
            ingest_failures.append({"video_name": name, "error": f"{type(e).__name__}: {e}"})
            continue

        pagerank_by_frame = {fr.frame_idx: fr.pagerank_score for fr in idx.frames}
        rank_pct = _rank_percentile(pagerank_by_frame)
        retained_B_scores = {fi: 1.0 - rank_pct[fi] for fi in rank_pct}

        full_B = np.zeros(n_frames, dtype=np.float64)
        retained_sorted_idx = sorted(retained_B_scores.keys())
        if retained_sorted_idx:
            first_val = retained_B_scores[retained_sorted_idx[0]]
            last_val = first_val
            ptr = 0
            for fi in range(n_frames):
                if ptr < len(retained_sorted_idx) and fi == retained_sorted_idx[ptr]:
                    last_val = retained_B_scores[fi]
                    ptr += 1
                full_B[fi] = last_val
        retention_pct_B = 100.0 * len(retained_sorted_idx) / n_frames if n_frames else 0.0

        combined = 0.5 * stageA_scores + 0.5 * full_B

        auc_B = None
        auc_combined = None
        if gt.sum() > 0 and gt.sum() < n_frames:
            auc_B = float(roc_auc_score(gt, full_B))
            auc_combined = float(roc_auc_score(gt, combined))
            per_video_auc_B[name] = auc_B
            per_video_auc_combined[name] = auc_combined

        pooled_B_scores.append(full_B)
        pooled_gt.append(gt)
        pooled_combined_scores.append(combined)

        per_video_results.append({
            "video_name": name,
            "class": cls,
            "n_frames": n_frames,
            "n_graph_nodes": len(idx.frames),
            "retention_pct_graph_nodes": retention_pct_B,
            "auc_stageB_alone": auc_B,
            "auc_stageA_plus_B_50_50": auc_combined,
        })

    elapsed = time.time() - t_start

    pooled_gt_cat = np.concatenate(pooled_gt) if pooled_gt else np.array([])
    pooled_B_cat = np.concatenate(pooled_B_scores) if pooled_B_scores else np.array([])
    pooled_combined_cat = np.concatenate(pooled_combined_scores) if pooled_combined_scores else np.array([])

    pooled_auc_B = float(roc_auc_score(pooled_gt_cat, pooled_B_cat)) if pooled_gt_cat.sum() > 0 else None
    pooled_auc_combined = float(roc_auc_score(pooled_gt_cat, pooled_combined_cat)) if pooled_gt_cat.sum() > 0 else None

    macro_auc_B = float(np.mean(list(per_video_auc_B.values()))) if per_video_auc_B else None
    macro_auc_combined = float(np.mean(list(per_video_auc_combined.values()))) if per_video_auc_combined else None

    results = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "IMPORTANT_CAVEAT": (
            "Same 169/290-video partial-corpus caveat as Stage A -- NOT "
            "comparable to the cited 290-video published numbers."
        ),
        "formulation": (
            "score_B(frame) = 1 - rank_percentile(pagerank_score) within that "
            "video's own scene-sparse graph nodes (node isolation). "
            "Non-retained frames hold-forward propagated identically to "
            "Stage A. Combined score = fixed 0.5*StageA + 0.5*StageB, no "
            "fusion-weight sweep."
        ),
        "frozen_config_used": frozen,
        "n_videos_attempted": len(stageA_per_video),
        "n_ingest_failures": len(ingest_failures),
        "ingest_failures": ingest_failures,
        "elapsed_s": elapsed,
        "pooled_auc_stageB_alone": pooled_auc_B,
        "pooled_auc_stageA_plus_B_50_50": pooled_auc_combined,
        "pooled_auc_stageA_alone_for_reference": None,  # filled from stageA_results.json by report
        "macro_auc_stageB_alone": macro_auc_B,
        "macro_auc_stageA_plus_B_50_50": macro_auc_combined,
    }
    stageA_results = json.loads((OUT_DIR / "stageA_results.json").read_text())
    results["pooled_auc_stageA_alone_for_reference"] = stageA_results["pooled_auc_all_frames_propagated"]

    (OUT_DIR / "stageB_results.json").write_text(json.dumps(results, indent=2))

    fieldnames = sorted({k for r in per_video_results for k in r.keys()})
    with (OUT_DIR / "stageB_per_video.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_video_results:
            w.writerow(r)

    print(json.dumps(results, indent=2))
    print(f"Wrote {OUT_DIR / 'stageB_results.json'}, {OUT_DIR / 'stageB_per_video.csv'}")


if __name__ == "__main__":
    main()
