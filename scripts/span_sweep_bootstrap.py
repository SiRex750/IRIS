"""Video-clustered paired bootstrap for the span-construction sweep's
acceptance criteria (criteria 2 and 3): resample the 450 val_tune VIDEOS
(not questions), 2000 resamples, seed 20260728, on the paired per-question
mIoP and mIoU differences (candidate - anchor). CI must exclude zero for
both metrics for a candidate to pass criteria 2+3.

Re-runs the retrieval+profile pass (fast, ~15s, zero fresh ingests
expected -- identical guarantee as span_sweep.py) rather than trying to
serialize the full per-question record set for every one of the 50 cells.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import iris.ingest as iris_ingest  # noqa: E402
import span_sweep as s  # noqa: E402
from part3_tune import make_config, load_frozen_state, load_val_tune_questions  # noqa: E402
from eval.metrics import assert_no_confirm_videos  # noqa: E402

OUT = REPO / "tuning" / "span_sweep"
SEED = 20260728
N_RESAMPLES = 2000


def build_cell_fn(method: str, params: dict):
    if method == "D":
        return lambda records: s.eval_cell_D(records, half_width_s=params["half_width_s"], duration_s_mode="pass")[0]
    if method == "E":
        return lambda records: s.eval_cell_E(
            records, tau=params["tau"], w_min=params["w_min"],
            gap=params.get("gap", 1), anchor_source=params.get("anchor_source", "topk"),
        )[0]
    if method == "F":
        return lambda records: s.eval_cell_F(records, alpha=params["alpha"], w_min=params["w_min"])[0]
    raise ValueError(method)


def paired_bootstrap(rows_anchor, rows_candidate, video_ids_per_row, rng) -> dict:
    """rows_* are per-question dicts aligned 1:1 (same question order).
    Resamples videos with replacement; each resample's per-question set is
    every question belonging to a resampled video (a video can appear
    multiple times, each occurrence contributing all its questions)."""
    by_video = {}
    for i, vid in enumerate(video_ids_per_row):
        by_video.setdefault(vid, []).append(i)
    unique_videos = sorted(by_video.keys())
    n_videos = len(unique_videos)

    iop_a = np.array([r["iop"] for r in rows_anchor])
    iop_c = np.array([r["iop"] for r in rows_candidate])
    iou_a = np.array([r["iou"] for r in rows_anchor])
    iou_c = np.array([r["iou"] for r in rows_candidate])

    diffs_miop = np.empty(N_RESAMPLES)
    diffs_miou = np.empty(N_RESAMPLES)
    for b in range(N_RESAMPLES):
        sampled = rng.choice(unique_videos, size=n_videos, replace=True)
        idxs = np.concatenate([by_video[v] for v in sampled])
        diffs_miop[b] = iop_c[idxs].mean() - iop_a[idxs].mean()
        diffs_miou[b] = iou_c[idxs].mean() - iou_a[idxs].mean()

    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    ci_miop = ci(diffs_miop)
    ci_miou = ci(diffs_miou)
    return {
        "point_estimate_mIoP_diff": float(iop_c.mean() - iop_a.mean()),
        "point_estimate_mIoU_diff": float(iou_c.mean() - iou_a.mean()),
        "ci95_mIoP_diff": ci_miop, "ci95_mIoU_diff": ci_miou,
        "mIoP_excludes_zero": not (ci_miop[0] <= 0 <= ci_miop[1]),
        "mIoU_excludes_zero": not (ci_miou[0] <= 0 <= ci_miou[1]),
        "n_resamples": N_RESAMPLES, "n_videos": n_videos,
    }


def main():
    frozen = load_frozen_state()["frozen"]
    questions = load_val_tune_questions()
    video_ids = sorted({q["video"] for q in questions})
    split = json.loads((REPO / "split_manifest.json").read_text())
    confirm_videos = set(split["confirm_videos"])
    assert_no_confirm_videos(video_ids, confirm_videos)

    overrides = {k: frozen[k] for k in (
        "retrieval_strategy", "ppr_lambda", "ppr_damping", "l2_retrieve_top_k",
        "peak_distance", "peak_prominence", "packet_size_weight", "motion_weight",
        "luma_entropy_weight", "persistence_threshold", "max_prominence",
    )}
    cfg = make_config(overrides)
    paths, missing, config_hash = s.check_index_cache(video_ids, cfg)
    assert not missing, f"missing cached videos: {missing}"
    index_cache = {vid: iris_ingest.load_index(paths[vid]) for vid in video_ids}
    records, pass_stats = s.run_retrieval_and_profile_pass(questions, cfg, index_cache)
    assert_no_confirm_videos([r["video"] for r in records], confirm_videos)
    print(f"[setup] {len(records)} records ready for bootstrap, gold_at_4={pass_stats['gold_at_4_rate']:.5f}")

    video_ids_per_row = [r["video"] for r in records]

    anchors = {
        "best_D_by_mIoP": {"method": "D", "params": {"half_width_s": 0.5}},
        "frozen_D_2.2": {"method": "D", "params": {"half_width_s": 2.2}},
    }
    candidates = {
        "best_D_by_mIoP": [
            {"method": "E", "params": {"tau": 0.7, "w_min": 0.6}},
            {"method": "E", "params": {"tau": 0.7, "w_min": 1.0}},
            {"method": "E", "params": {"tau": 0.7, "w_min": 1.4}},
            {"method": "E", "params": {"tau": 0.7, "w_min": 1.8}},
            {"method": "E", "params": {"tau": 0.7, "w_min": 2.2}},
            {"method": "E", "params": {"tau": 0.8, "w_min": 0.6}},
            {"method": "E", "params": {"tau": 0.8, "w_min": 1.0}},
            {"method": "E", "params": {"tau": 0.8, "w_min": 1.4}},
            {"method": "E", "params": {"tau": 0.8, "w_min": 1.8}},
            {"method": "E", "params": {"tau": 0.8, "w_min": 2.2}},
        ],
        "frozen_D_2.2": [
            {"method": "E", "params": {"tau": 0.6, "w_min": 2.2}},
        ],
    }

    results = {}
    for anchor_name, anchor_spec in anchors.items():
        anchor_fn = build_cell_fn(anchor_spec["method"], anchor_spec["params"])
        rows_anchor = anchor_fn(records)
        results[anchor_name] = {"anchor": anchor_spec, "candidates": {}}
        for cand in candidates[anchor_name]:
            cand_fn = build_cell_fn(cand["method"], cand["params"])
            rows_cand = cand_fn(records)
            rng = np.random.default_rng(SEED)
            boot = paired_bootstrap(rows_anchor, rows_cand, video_ids_per_row, rng)
            key = f"{cand['method']}_{json.dumps(cand['params'], sort_keys=True)}"
            results[anchor_name]["candidates"][key] = {"params": cand, "bootstrap": boot}
            print(f"[{anchor_name}] vs {key}: "
                  f"mIoP_diff={boot['point_estimate_mIoP_diff']:+.5f} CI={boot['ci95_mIoP_diff']} excl0={boot['mIoP_excludes_zero']} | "
                  f"mIoU_diff={boot['point_estimate_mIoU_diff']:+.5f} CI={boot['ci95_mIoU_diff']} excl0={boot['mIoU_excludes_zero']}")

    (OUT / "bootstrap_ci.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {OUT / 'bootstrap_ci.json'}")


if __name__ == "__main__":
    main()
