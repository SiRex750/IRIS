"""Part 2c: empirical metric-implementation parity check.

Regenerates (predicted_span, gold_spans) for every question in every
already-frozen Family 1 (retrieval_strategy) and Family 2 (ppr_lambda)
trial -- same 11 configs, same cached indexes, retrieval-only, fully
deterministic -- since the original tuning run never persisted per-question
raw spans to disk (only trial-level aggregates). This is NOT a re-tuning
run: no new configs are evaluated and no selection logic runs again.

Then scores every (pred, gold) pair with BOTH eval/metrics.py (what
scripts/part3_tune.py actually uses) and the canonical
benchmark_runs/paper_setup_.../scripts/nextgqa_metrics.py (what
metric_registry.json designates as validated), and diffs every field.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import part3_tune as pt  # noqa: E402
import iris.ingest as iris_ingest  # noqa: E402
from iris.query import _call_embed_query, _retrieve_with_l1  # noqa: E402
import eval.metrics as eval_metrics  # noqa: E402

NEXTGQA_METRICS_PATH = REPO / "benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts/nextgqa_metrics.py"
spec = importlib.util.spec_from_file_location("nextgqa_metrics_canonical", NEXTGQA_METRICS_PATH)
nextgqa_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nextgqa_metrics)

OUT_DIFF_CSV = REPO / "metric_parity_diff.csv"
OUT_SYNTH_CSV = REPO / "metric_parity_synthetic.csv"


def score_eval_metrics(pred_span: tuple[float, float], gold_spans: list[list[float]]) -> dict:
    iou, iop = eval_metrics.best_over_gold_spans(gold_spans, pred_span)
    return {"IoP": iop, "IoU": iou, "IoP@0.3": iop >= 0.3, "IoP@0.5": iop >= 0.5,
            "IoU@0.3": iou >= 0.3, "IoU@0.5": iou >= 0.5}


def score_nextgqa_metrics(pred_span: tuple[float, float], gold_spans: list[list[float]]) -> dict:
    gold_tuples = [(g[0], g[1]) for g in gold_spans]
    iop = nextgqa_metrics.iop(pred_span[0], pred_span[1], gold_tuples)
    iou = nextgqa_metrics.iou(pred_span[0], pred_span[1], gold_tuples)
    return {"IoP": iop, "IoU": iou, "IoP@0.3": iop >= 0.3, "IoP@0.5": iop >= 0.5,
            "IoU@0.3": iou >= 0.3, "IoU@0.5": iou >= 0.5}


def build_trial_configs() -> list[dict]:
    trials = []
    for val in pt.FAMILIES["retrieval_strategy"]:
        trials.append({"family": "retrieval_strategy", "value": str(val), "overrides": {"retrieval_strategy": val}})
    for val in pt.FAMILIES["ppr_lambda"]:
        trials.append({"family": "ppr_lambda", "value": str(val),
                        "overrides": {"retrieval_strategy": "hybrid", "ppr_lambda": val}})
    return trials


def main():
    questions = pt.load_val_tune_questions()
    print(f"[setup] {len(questions)} val_tune questions", flush=True)
    trials = build_trial_configs()
    print(f"[setup] {len(trials)} trials to regenerate (4 retrieval_strategy + 7 ppr_lambda)", flush=True)

    diff_fields = ["IoP", "IoU", "IoP@0.3", "IoP@0.5", "IoU@0.3", "IoU@0.5"]
    fieldnames = (["family", "trial_value", "video", "qid", "pred_start", "pred_end", "gold_spans"]
                  + [f"eval_metrics_{f}" for f in diff_fields]
                  + [f"nextgqa_metrics_{f}" for f in diff_fields]
                  + ["all_fields_match"])

    n_total = 0
    n_mismatch = 0
    mismatches = []
    trial_recompute = {}  # (family,value) -> {"iops":[], "iops_ng":[]}

    with open(OUT_DIFF_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for trial in trials:
            cfg = pt.make_config(trial["overrides"])
            video_ids = sorted({q["video"] for q in questions})
            index_paths = pt.ensure_indexes(video_ids, cfg, n_workers=8)
            index_cache: dict = {}
            t0 = time.perf_counter()
            key = (trial["family"], trial["value"])
            trial_recompute[key] = {"em_iop": [], "em_iou": [], "ng_iop": [], "ng_iou": []}

            for q in questions:
                vid = q["video"]
                if vid not in index_paths:
                    continue
                if vid not in index_cache:
                    index_cache[vid] = iris_ingest.load_index(index_paths[vid])
                index = index_cache[vid]
                try:
                    qe, _ = _call_embed_query(q["question"], cfg)
                    frames, _ = _retrieve_with_l1(index, qe, cfg)
                except Exception:
                    continue
                pred_span = eval_metrics.predicted_span_from_frames([fr["timestamp"] for fr in frames])
                gold_spans = q["gold_spans"]

                em = score_eval_metrics(pred_span, gold_spans)
                ng = score_nextgqa_metrics(pred_span, gold_spans)

                match = all(em[k] == ng[k] if not isinstance(em[k], float) else abs(em[k] - ng[k]) < 1e-9
                            for k in diff_fields)

                n_total += 1
                trial_recompute[key]["em_iop"].append(em["IoP"])
                trial_recompute[key]["em_iou"].append(em["IoU"])
                trial_recompute[key]["ng_iop"].append(ng["IoP"])
                trial_recompute[key]["ng_iou"].append(ng["IoU"])

                row = {
                    "family": trial["family"], "trial_value": trial["value"],
                    "video": vid, "qid": q["qid"],
                    "pred_start": round(pred_span[0], 4), "pred_end": round(pred_span[1], 4),
                    "gold_spans": json.dumps(gold_spans),
                    **{f"eval_metrics_{k}": em[k] for k in diff_fields},
                    **{f"nextgqa_metrics_{k}": ng[k] for k in diff_fields},
                    "all_fields_match": match,
                }
                w.writerow(row)

                if not match:
                    n_mismatch += 1
                    mismatches.append(row)

            dt = time.perf_counter() - t0
            print(f"  [trial] {trial['family']}={trial['value']}: done in {dt:.0f}s", flush=True)

    print(f"[summary] {n_total} question-scorings, {n_mismatch} mismatches", flush=True)

    json.dump({
        "n_total": n_total, "n_mismatch": n_mismatch,
        "mismatches": mismatches[:200],
        "trial_recompute_aggregates": {
            f"{k[0]}={k[1]}": {
                "eval_metrics_mIoP": sum(v["em_iop"]) / len(v["em_iop"]) if v["em_iop"] else None,
                "eval_metrics_mIoU": sum(v["em_iou"]) / len(v["em_iou"]) if v["em_iou"] else None,
                "nextgqa_metrics_mIoP": sum(v["ng_iop"]) / len(v["ng_iop"]) if v["ng_iop"] else None,
                "nextgqa_metrics_mIoU": sum(v["ng_iou"]) / len(v["ng_iou"]) if v["ng_iou"] else None,
                "n_scored": len(v["em_iop"]),
            } for k, v in trial_recompute.items()
        },
    }, open(REPO / ".part2c_recompute_summary.json", "w"), indent=2)

    print("METRIC_PARITY_REAL_DATA_DONE" if n_mismatch == 0 else f"METRIC_PARITY_REAL_DATA_MISMATCHES={n_mismatch}")


if __name__ == "__main__":
    main()
