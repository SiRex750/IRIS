"""Family 5 sanity re-check: peak_distance x peak_prominence under the
action_score_weights winner (0.8, 0.1, 0.1), not the (0.5, 0.3, 0.2)
weights the original Family 5 run (commit 5d7c2b1) used.

action_score_weights (commit 5058e56) found a real winner away from the
default weights and is the more upstream parameter (it shapes the curve
find_peaks operates on), so its own report flagged that peak_dist_prom's
selection needed re-confirming under the new weights before treating
(peak_distance=5, peak_prominence=0.05) as settled for good. This script
is that re-check.

Deliberately NOT run through part3_tune.py's run_family()/FAMILIES
machinery: "peak_dist_prom" is already in completed_families (skipped by
run_family's guard), and this re-check uses a distinct family label
("peak_dist_prom_recheck_under_action_score_weights") in all_trials.csv
so a later reader can tell which weight regime each run was under without
cross-referencing config-hashes. Reuses part3_tune.py's ingest/evaluate/
CSV-append/select_best machinery directly instead of duplicating it.
"""
from __future__ import annotations

import time

from part3_tune import (
    DEFAULTS, TUPLE_FAMILIES, FAMILIES,
    load_frozen_state, make_config, ingest_config_hash,
    ensure_indexes, evaluate_config, append_trial_row, select_best,
    load_val_tune_questions, TUNING_DIR,
)

FAMILY_LABEL = "peak_dist_prom_recheck_under_action_score_weights"
GRID = FAMILIES["peak_dist_prom"]  # identical grid: [(3,0.03),(5,0.05),(8,0.05),(5,0.10)]
FIELD_NAMES = TUPLE_FAMILIES["peak_dist_prom"]  # ["peak_distance", "peak_prominence"]


def main() -> None:
    state = load_frozen_state()
    frozen = state["frozen"]
    print(f"[recheck] frozen_so_far={frozen}", flush=True)
    expected = {"packet_size_weight": 0.8, "motion_weight": 0.1, "luma_entropy_weight": 0.1,
                "peak_distance": 5, "peak_prominence": 0.05}
    for k, v in expected.items():
        if frozen.get(k) != v:
            raise SystemExit(f"[recheck] PREFLIGHT MISMATCH: frozen['{k}']={frozen.get(k)!r}, expected {v!r}. Stopping.")
    print("[recheck] preflight OK -- weights and prior peak_dist_prom selection confirmed as expected", flush=True)

    questions = load_val_tune_questions()
    print(f"[setup] val_tune questions loaded: {len(questions)} across {len(set(q['video'] for q in questions))} videos", flush=True)
    video_ids = sorted({q["video"] for q in questions})

    trials = []
    index_cache: dict[str, object] = {}

    for val in GRID:
        t_start = time.perf_counter()
        overrides = {**frozen, **dict(zip(FIELD_NAMES, val))}
        label = ",".join(f"{fn}={v}" for fn, v in zip(FIELD_NAMES, val))

        cfg = make_config(overrides)
        print(f"  [trial] {FAMILY_LABEL}={label} "
              f"(packet_size_weight={cfg.packet_size_weight}, motion_weight={cfg.motion_weight}, "
              f"luma_entropy_weight={cfg.luma_entropy_weight})", flush=True)
        index_paths = ensure_indexes(video_ids, cfg)
        index_cache.clear()
        metrics = evaluate_config(cfg, questions, index_paths, index_cache)
        wall_s = time.perf_counter() - t_start
        print(f"    -> mIoP={metrics['mIoP']:.4f} IoP@0.5={metrics['IoP@0.5']:.4f} "
              f"mIoU={metrics['mIoU']:.4f} n={metrics['n_questions_scored']} wall={wall_s:.0f}s", flush=True)

        trials.append({"value": val, "label": label, "metrics": metrics,
                        "cfg_hash": ingest_config_hash(cfg), "wall_s": wall_s})
        append_trial_row({
            "family": FAMILY_LABEL, "trial_value": label, "config_hash": ingest_config_hash(cfg),
            "n_questions_scored": metrics["n_questions_scored"], "mIoP": round(metrics["mIoP"], 5),
            "mIoU": round(metrics["mIoU"], 5), "IoP@0.3": round(metrics["IoP@0.3"], 5),
            "IoP@0.5": round(metrics["IoP@0.5"], 5), "IoU@0.3": round(metrics["IoU@0.3"], 5),
            "IoU@0.5": round(metrics["IoU@0.5"], 5), "median_retrieval_ms": round(metrics["median_retrieval_ms"], 2),
            "p95_retrieval_ms": round(metrics["p95_retrieval_ms"], 2), "wall_s": round(wall_s, 1),
            "selected": "",
        })

    default_val = tuple(DEFAULTS[fn] for fn in FIELD_NAMES)
    best = select_best(trials, default_val)
    print(f"[recheck] SELECTED under new weights: {best['label']} (mIoP={best['metrics']['mIoP']:.4f})", flush=True)

    prior_selected = (frozen["peak_distance"], frozen["peak_prominence"])
    prior_label = f"peak_distance={prior_selected[0]},peak_prominence={prior_selected[1]}"
    superseded = best["value"] != prior_selected
    print(f"[recheck] prior Family 5 selection: {prior_label}. "
          f"{'SUPERSEDED' if superseded else 'RECONFIRMED'} under new weights.", flush=True)

    with open(TUNING_DIR / "selection_decisions.md", "a") as f:
        f.write("\n### Family 5 re-check under action_score_weights\n\n")
        f.write("Re-ran the identical peak_dist_prom grid "
                "[(3, 0.03), (5, 0.05), (8, 0.05), (5, 0.10)] with "
                "packet_size_weight/motion_weight/luma_entropy_weight held at the "
                "action_score_weights winner (0.8, 0.1, 0.1) instead of the "
                "(0.5, 0.3, 0.2) the original Family 5 run (commit 5d7c2b1) used.\n\n")
        f.write("| value | mIoP | IoP@0.5 | mIoU | median_retrieval_ms | n_scored |\n")
        f.write("|---|---|---|---|---|---|\n")
        for t in trials:
            mark = " **<- selected**" if t is best else ""
            f.write(f"| {t['label']}{mark} | {t['metrics']['mIoP']:.4f} | {t['metrics']['IoP@0.5']:.4f} | "
                    f"{t['metrics']['mIoU']:.4f} | {t['metrics']['median_retrieval_ms']:.1f} | {t['metrics']['n_questions_scored']} |\n")
        f.write(f"\n**{'SUPERSEDED' if superseded else 'RECONFIRMED'}**: ")
        if superseded:
            f.write(f"under the new (0.8, 0.1, 0.1) weights, the winner is **{best['label']}**, "
                    f"different from the original Family 5 selection ({prior_label}, mIoP=0.29516 "
                    "under the old (0.5, 0.3, 0.2) weights). "
                    f"peak_distance/peak_prominence's frozen values are updated accordingly.\n")
        else:
            f.write(f"the original Family 5 selection ({prior_label}) is still the best (or "
                    "statistically tied-best, same 0.005 mIoP tie-break rule) combo under the new "
                    "(0.8, 0.1, 0.1) weights. No change to the frozen peak_distance/peak_prominence "
                    "values.\n")

    if superseded:
        state["frozen"]["peak_distance"] = best["value"][0]
        state["frozen"]["peak_prominence"] = best["value"][1]
        state["peak_dist_prom_recheck_under_action_score_weights"] = {
            "superseded": True, "prior_selection": prior_label, "new_selection": best["label"],
            "all_trials": [{"value": t["label"], "mIoP": t["metrics"]["mIoP"], "IoP@0.5": t["metrics"]["IoP@0.5"],
                             "median_retrieval_ms": t["metrics"]["median_retrieval_ms"]} for t in trials],
        }
        from part3_tune import save_frozen_state
        save_frozen_state(state)
        print("[recheck] frozen_state.json updated: peak_distance/peak_prominence changed.", flush=True)
    else:
        state["peak_dist_prom_recheck_under_action_score_weights"] = {
            "superseded": False, "prior_selection": prior_label, "new_selection": best["label"],
            "all_trials": [{"value": t["label"], "mIoP": t["metrics"]["mIoP"], "IoP@0.5": t["metrics"]["IoP@0.5"],
                             "median_retrieval_ms": t["metrics"]["median_retrieval_ms"]} for t in trials],
        }
        from part3_tune import save_frozen_state
        save_frozen_state(state)
        print("[recheck] frozen_state.json: peak_distance/peak_prominence left unchanged "
              "(reconfirmation recorded in a new top-level key, not touching 'frozen').", flush=True)

    print("PEAK_DIST_PROM_RECHECK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
