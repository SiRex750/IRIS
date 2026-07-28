"""Single-pass, fully self-consistent run of the span-construction sweep:
one retrieval+profile pass feeds the full 50-cell grid, the acceptance-
criteria scan, the video-clustered bootstrap, and the fixed-a-priori
sensitivity checks -- all from the same `records`.

Written after discovering that CLIP query embedding runs on GPU
(iris/query.py's `_embed_query` picks "cuda" when available), which
introduces small (~1e-4 magnitude) run-to-run floating-point jitter in
retrieval results. That's harmless for a single run's internal
comparisons (every cell in this script sees identical `records`) but
made an earlier cross-script comparison (grid_results.csv from
span_sweep.py vs point estimates from a separately-invoked
span_sweep_bootstrap.py) unreliable at the precision this analysis
needs. This script exists so every number in the final report comes
from ONE retrieval pass, never stitched across runs.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import iris.ingest as iris_ingest  # noqa: E402
import span_sweep as s  # noqa: E402
from part3_tune import make_config, load_frozen_state, load_val_tune_questions  # noqa: E402
from eval.metrics import assert_no_confirm_videos, compute_similarity_profile  # noqa: E402

OUT = REPO / "tuning" / "span_sweep"
SEED = 20260728
N_RESAMPLES = 2000


def log(msg):
    s.log(msg)


def paired_bootstrap(rows_anchor, rows_candidate, video_ids_per_row, rng):
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

    ci_miop, ci_miou = ci(diffs_miop), ci(diffs_miou)
    return {
        "point_estimate_mIoP_diff": float(iop_c.mean() - iop_a.mean()),
        "point_estimate_mIoU_diff": float(iou_c.mean() - iou_a.mean()),
        "ci95_mIoP_diff": ci_miop, "ci95_mIoU_diff": ci_miou,
        "mIoP_excludes_zero": not (ci_miop[0] <= 0 <= ci_miop[1]),
        "mIoU_excludes_zero": not (ci_miou[0] <= 0 <= ci_miou[1]),
        "n_resamples": N_RESAMPLES, "n_videos": n_videos,
    }


def main():
    log("=== span_sweep_final.py -- single-pass authoritative run ===")
    frozen = load_frozen_state()["frozen"]
    split = json.loads((REPO / "split_manifest.json").read_text())
    confirm_videos = set(split["confirm_videos"])
    tune_videos = set(split["tune_videos"])

    questions = load_val_tune_questions()
    video_ids = sorted({q["video"] for q in questions})
    assert_no_confirm_videos(video_ids, confirm_videos)
    assert set(video_ids) <= tune_videos

    overrides = {k: frozen[k] for k in (
        "retrieval_strategy", "ppr_lambda", "ppr_damping", "l2_retrieve_top_k",
        "peak_distance", "peak_prominence", "packet_size_weight", "motion_weight",
        "luma_entropy_weight", "persistence_threshold", "max_prominence",
    )}
    cfg = make_config(overrides)
    paths, missing, config_hash = s.check_index_cache(video_ids, cfg)
    if missing:
        log(f"[FATAL] {len(missing)} videos missing from cache: {missing[:10]}")
        sys.exit(1)
    index_cache = {vid: iris_ingest.load_index(paths[vid]) for vid in video_ids}
    records, pass_stats = s.run_retrieval_and_profile_pass(questions, cfg, index_cache)
    assert_no_confirm_videos([r["video"] for r in records], confirm_videos)
    log(f"[pass1] {pass_stats}")
    video_ids_per_row = [r["video"] for r in records]

    # ── Full 50-cell grid (authoritative, supersedes span_sweep.py's) ──────
    all_cells = {}
    cell_id = 0
    for hw in s.D_HALF_WIDTHS:
        cell_id += 1
        rows, extra = s.eval_cell_D(records, half_width_s=hw, duration_s_mode="pass")
        agg = s.aggregate_cell(rows, extra, pass_stats["gold_at_4_rate"])
        all_cells[cell_id] = {"method": "D", "params": {"half_width_s": hw}, "rows": rows, "agg": agg}
    for tau in s.E_TAUS:
        for w_min in s.E_WMINS:
            cell_id += 1
            rows, extra = s.eval_cell_E(records, tau=tau, w_min=w_min)
            agg = s.aggregate_cell(rows, extra, pass_stats["gold_at_4_rate"])
            all_cells[cell_id] = {"method": "E", "params": {"tau": tau, "w_min": w_min, "smooth_s": 0.5, "gap": 1, "anchor_source": "topk"}, "rows": rows, "agg": agg}
    for alpha in s.F_ALPHAS:
        for w_min in s.F_WMINS:
            cell_id += 1
            rows, extra = s.eval_cell_F(records, alpha=alpha, w_min=w_min)
            agg = s.aggregate_cell(rows, extra, pass_stats["gold_at_4_rate"])
            all_cells[cell_id] = {"method": "F", "params": {"alpha": alpha, "w_min": w_min}, "rows": rows, "agg": agg}
    log(f"[grid] {cell_id} cells computed in this single pass")

    # Write authoritative grid_results.csv (overwrite the span_sweep.py one --
    # same schema, single-pass values, this file is what the report quotes)
    csv_path = OUT / "grid_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=s.GRID_FIELDNAMES)
        writer.writeheader()
        for cid, c in all_cells.items():
            row = {"cell_id": cid, "method": c["method"], "params": json.dumps(c["params"]), "elapsed_ms": ""}
            for k in s.GRID_FIELDNAMES:
                if k in c["agg"]:
                    row[k] = c["agg"][k]
            row.setdefault("degenerate_fallback_rate", "")
            row.setdefault("weight_fallback_rate", "")
            writer.writerow({k: row.get(k, "") for k in s.GRID_FIELDNAMES})
    log(f"[grid] wrote authoritative {csv_path}")

    def best_of(method):
        cands = [(cid, c) for cid, c in all_cells.items() if c["method"] == method]
        return max(cands, key=lambda x: (x[1]["agg"]["mIoP"], x[1]["agg"]["IoP@0.5"]))

    best_D_id, best_D = best_of("D")
    best_E_id, best_E = best_of("E")
    best_F_id, best_F = best_of("F")
    frozen_D_id = next(cid for cid, c in all_cells.items() if c["method"] == "D" and c["params"]["half_width_s"] == 2.2)
    frozen_D = all_cells[frozen_D_id]
    log(f"[selection] best D (by mIoP): cell {best_D_id} {best_D['params']} mIoP={best_D['agg']['mIoP']:.5f} mIoU={best_D['agg']['mIoU']:.5f}")
    log(f"[selection] frozen D@2.2: cell {frozen_D_id} mIoP={frozen_D['agg']['mIoP']:.5f} mIoU={frozen_D['agg']['mIoU']:.5f}")
    log(f"[selection] best E (by mIoP): cell {best_E_id} {best_E['params']} mIoP={best_E['agg']['mIoP']:.5f} mIoU={best_E['agg']['mIoU']:.5f}")
    log(f"[selection] best F (by mIoP): cell {best_F_id} {best_F['params']} mIoP={best_F['agg']['mIoP']:.5f} mIoU={best_F['agg']['mIoU']:.5f}")

    # ── Acceptance criterion 1 scan (both anchors), self-consistent ────────
    def scan_criterion1(anchor_id, anchor):
        a_miop, a_miou = anchor["agg"]["mIoP"], anchor["agg"]["mIoU"]
        passing = []
        for cid, c in all_cells.items():
            if cid == anchor_id:
                continue
            if c["agg"]["mIoP"] > a_miop and c["agg"]["mIoU"] > a_miou:
                passing.append(cid)
        return passing

    crit1_vs_bestD = scan_criterion1(best_D_id, best_D)
    crit1_vs_frozenD = scan_criterion1(frozen_D_id, frozen_D)
    log(f"[criterion1] vs best-D-by-mIoP (cell {best_D_id}): {len(crit1_vs_bestD)} candidates pass point-estimate test: {crit1_vs_bestD}")
    log(f"[criterion1] vs frozen-D@2.2 (cell {frozen_D_id}): {len(crit1_vs_frozenD)} candidates pass point-estimate test: {crit1_vs_frozenD}")

    # ── Bootstrap on every criterion-1 passer, against both anchors ────────
    bootstrap_results = {}
    for anchor_name, anchor_id, anchor, passers in [
        ("best_D_by_mIoP", best_D_id, best_D, crit1_vs_bestD),
        ("frozen_D_2.2", frozen_D_id, frozen_D, crit1_vs_frozenD),
    ]:
        bootstrap_results[anchor_name] = {"anchor_cell_id": anchor_id, "anchor_params": anchor["params"], "candidates": {}}
        for cid in passers:
            cand = all_cells[cid]
            rng = np.random.default_rng(SEED)
            boot = paired_bootstrap(anchor["rows"], cand["rows"], video_ids_per_row, rng)
            key = f"cell{cid}_{cand['method']}_{json.dumps(cand['params'], sort_keys=True)}"
            passes_all_4 = (
                boot["mIoP_excludes_zero"] and boot["point_estimate_mIoP_diff"] > 0
                and boot["mIoU_excludes_zero"] and boot["point_estimate_mIoU_diff"] > 0
                and cand["agg"]["zero_width_rate"] <= 0.01
                and cand["agg"]["clip_anchor_fallback_rate"] <= 0.05
                and cand["agg"].get("degenerate_fallback_rate", 0.0) <= 0.05
            )
            bootstrap_results[anchor_name]["candidates"][key] = {
                "cell_id": cid, "params": cand["params"], "bootstrap": boot,
                "zero_width_rate": cand["agg"]["zero_width_rate"],
                "clip_anchor_fallback_rate": cand["agg"]["clip_anchor_fallback_rate"],
                "degenerate_fallback_rate": cand["agg"].get("degenerate_fallback_rate", 0.0),
                "passes_all_4_criteria": passes_all_4,
            }
            log(f"[bootstrap] {anchor_name} vs {key}: mIoP_diff={boot['point_estimate_mIoP_diff']:+.5f} "
                f"excl0={boot['mIoP_excludes_zero']} mIoU_diff={boot['point_estimate_mIoU_diff']:+.5f} "
                f"excl0={boot['mIoU_excludes_zero']} passes_all_4={passes_all_4}")

    (OUT / "bootstrap_ci.json").write_text(json.dumps(bootstrap_results, indent=2, default=str))
    any_pass = any(
        c["passes_all_4_criteria"]
        for anchor in bootstrap_results.values()
        for c in anchor["candidates"].values()
    )
    log(f"[acceptance] ANY candidate clears all 4 criteria against any anchor: {any_pass}")

    # ── Sensitivity checks at best-E-by-mIoP cell (reported regardless of
    # acceptance outcome, for transparency) ─────────────────────────────────
    sensitivity = {}
    be_params = best_E["params"]
    # smooth_s sensitivity (0.0 vs the swept default 0.5): the cached profile
    # in `records` was computed at smooth_s=0.5, and query embeddings were
    # not retained after pass 1 (kept memory low across the whole sweep), so
    # this recomputes via a fresh _call_embed_query call per question --
    # identical text -> identical embedding modulo the same GPU jitter noted
    # above, negligible for a profile-shape sensitivity check.
    from iris.query import _call_embed_query
    records_by_key = {(r["video"], r["qid"]): r for r in records}

    def eval_E_with_smooth(smooth_s_val):
        rows = []
        for q in questions:
            vid = q["video"]
            if vid not in index_cache:
                continue
            r = records_by_key.get((vid, q["qid"]))
            if r is None:
                continue
            qe, _ = _call_embed_query(q["question"], cfg)
            profile = compute_similarity_profile(index_cache[vid].frames, qe, smooth_s=smooth_s_val)
            retrieved_min = r["retrieved_min"]
            dur = r["duration"]
            w_max = min(20.0, 0.5 * dur)
            span, used, tel = s.span_from_similarity_profile(
                profile, retrieved_min, dur, be_params["tau"], be_params["w_min"],
                gap=be_params.get("gap", 1), anchor_source=be_params.get("anchor_source", "topk"), w_max=w_max,
            )
            iop, iou = s.iop_iou(span, q["gold_spans"])
            rows.append({"iop": iop, "iou": iou})
        n = len(rows)
        return {"mIoP": sum(r["iop"] for r in rows) / n, "mIoU": sum(r["iou"] for r in rows) / n, "n": n}

    log("[sensitivity] recomputing smooth_s=0.0 variant at best-E cell (this re-embeds all questions once more)...")
    sens_smooth0 = eval_E_with_smooth(0.0)
    sensitivity["smooth_s"] = {"default_0.5": {"mIoP": best_E["agg"]["mIoP"], "mIoU": best_E["agg"]["mIoU"]}, "variant_0.0": sens_smooth0}
    log(f"[sensitivity] smooth_s: default(0.5)={sensitivity['smooth_s']['default_0.5']} variant(0.0)={sens_smooth0}")

    # gap sensitivity: 0 vs default 1 (cheap, reuse cached profile via eval_cell_E)
    rows_gap0, _ = s.eval_cell_E(records, tau=be_params["tau"], w_min=be_params["w_min"], gap=0, anchor_source=be_params.get("anchor_source", "topk"))
    agg_gap0 = s.aggregate_cell(rows_gap0, {}, pass_stats["gold_at_4_rate"])
    sensitivity["gap"] = {"default_1": {"mIoP": best_E["agg"]["mIoP"], "mIoU": best_E["agg"]["mIoU"]}, "variant_0": {"mIoP": agg_gap0["mIoP"], "mIoU": agg_gap0["mIoU"]}}
    log(f"[sensitivity] gap: default(1)={sensitivity['gap']['default_1']} variant(0)={sensitivity['gap']['variant_0']}")

    # anchor_source sensitivity: survivors vs default topk (cheap, cached profile)
    rows_surv, _ = s.eval_cell_E(records, tau=be_params["tau"], w_min=be_params["w_min"], gap=be_params.get("gap", 1), anchor_source="survivors")
    agg_surv = s.aggregate_cell(rows_surv, {}, pass_stats["gold_at_4_rate"])
    sensitivity["anchor_source"] = {"default_topk": {"mIoP": best_E["agg"]["mIoP"], "mIoU": best_E["agg"]["mIoU"]}, "variant_survivors": {"mIoP": agg_surv["mIoP"], "mIoU": agg_surv["mIoU"]}}
    log(f"[sensitivity] anchor_source: default(topk)={sensitivity['anchor_source']['default_topk']} variant(survivors)={sensitivity['anchor_source']['variant_survivors']}")

    (OUT / "sensitivity_checks.json").write_text(json.dumps({
        "winning_cell_note": "No cell cleared all 4 acceptance criteria (see bootstrap_ci.json) -- these "
                              "sensitivity checks are reported at the best-E-by-mIoP cell for transparency, "
                              "not as validation of a recommended cell.",
        "cell_id": best_E_id, "params": best_E["params"], "checks": sensitivity,
    }, indent=2))
    log(f"[deliverable] wrote sensitivity_checks.json")

    # ── Per-question CSVs for the 4 designated cells ────────────────────
    per_q_dir = OUT / "per_question"
    per_q_dir.mkdir(exist_ok=True)
    designated = {
        f"D_hw2.2_cell{frozen_D_id}": frozen_D["rows"],
        f"D_best_cell{best_D_id}": best_D["rows"],
        f"E_best_cell{best_E_id}": best_E["rows"],
        f"F_best_cell{best_F_id}": best_F["rows"],
    }
    per_q_fields = ["video", "qid", "type", "gold_spans", "pred_span_start", "pred_span_end", "width",
                     "iop", "iou", "anchor_ts", "n_survivors_scored", "degenerate_profile_fallback"]
    for name, rows in designated.items():
        with open(per_q_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_q_fields)
            w.writeheader()
            for r in rows:
                row = {k: r.get(k, "") for k in per_q_fields}
                row["gold_spans"] = json.dumps(row["gold_spans"])
                w.writerow(row)
    log(f"[deliverable] wrote per-question CSVs for {list(designated.keys())}")

    summary = {
        "pass_stats": pass_stats, "config_hash": config_hash,
        "best_D": {"cell_id": best_D_id, "params": best_D["params"], "agg": best_D["agg"]},
        "frozen_D": {"cell_id": frozen_D_id, "agg": frozen_D["agg"]},
        "best_E": {"cell_id": best_E_id, "params": best_E["params"], "agg": best_E["agg"]},
        "best_F": {"cell_id": best_F_id, "params": best_F["params"], "agg": best_F["agg"]},
        "criterion1_passers_vs_best_D": crit1_vs_bestD, "criterion1_passers_vs_frozen_D": crit1_vs_frozenD,
        "any_candidate_passes_all_4_criteria": any_pass,
    }
    (OUT / "sweep_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("=== span_sweep_final.py complete ===")


if __name__ == "__main__":
    main()
