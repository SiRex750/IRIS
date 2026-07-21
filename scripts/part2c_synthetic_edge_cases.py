"""Part 2c task 4: hand-verified synthetic edge cases through both metric
implementations, side by side."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import eval.metrics as eval_metrics  # noqa: E402

NEXTGQA_METRICS_PATH = REPO / "benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts/nextgqa_metrics.py"
spec = importlib.util.spec_from_file_location("nextgqa_metrics_canonical", NEXTGQA_METRICS_PATH)
nextgqa_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nextgqa_metrics)


def score_em(pred, gold_spans):
    iou, iop = eval_metrics.best_over_gold_spans(gold_spans, pred)
    return iop, iou


def score_ng(pred, gold_spans):
    gold_tuples = [(g[0], g[1]) for g in gold_spans]
    iop = nextgqa_metrics.iop(pred[0], pred[1], gold_tuples)
    iou = nextgqa_metrics.iou(pred[0], pred[1], gold_tuples)
    return iop, iou


CASES = [
    {
        "name": "multiple_gold_spans_max_overlap",
        "pred": (11.0, 13.0), "gold_spans": [[0.0, 2.0], [10.0, 14.0]],
        "expected_IoP": 1.0, "expected_IoU": 0.5,
        "rationale": "pred entirely inside 2nd gold span [10,14]; 1st gold span [0,2] has 0 overlap -- max-over-spans must pick the 2nd, never concatenate.",
    },
    {
        "name": "zero_duration_pred_inside_gold",
        "pred": (5.0, 5.0), "gold_spans": [[3.0, 8.0]],
        "expected_IoP": None,  # implementation-divergent case -- see report
        "expected_IoU": None,
        "rationale": "Official eval_ground.py special-cases span[0]==span[-1]: (IoU=0,IoP=1) if the point is inside the gold span. This is the case eval/metrics.py replicates and nextgqa_metrics.py does not (it unconditionally returns IoP=0 for any non-positive predicted length).",
    },
    {
        "name": "reversed_span_start_gt_end",
        "pred": (10.0, 5.0), "gold_spans": [[3.0, 8.0]],
        "expected_IoP": 0.0, "expected_IoU": 0.0,
        "rationale": "malformed predicted span (start>end) -- both implementations should degrade to 0,0 rather than a negative/nonsensical score or a crash.",
    },
    {
        "name": "pred_fully_outside_video_duration",
        "pred": (25.0, 30.0), "gold_spans": [[5.0, 10.0]],
        "expected_IoP": 0.0, "expected_IoU": 0.0,
        "rationale": "trivial no-overlap case.",
    },
    {
        "name": "exact_boundary_IoP_0.3",
        "pred": (0.0, 10.0), "gold_spans": [[0.0, 3.0]],
        "expected_IoP": 0.3, "expected_IoU": 0.3,
        "rationale": "intersection=3, pred_len=10 -> IoP=0.3 exactly (threshold boundary); gold fully inside pred so union=pred_len too, IoU=0.3 as well.",
    },
    {
        "name": "exact_boundary_IoP_0.5",
        "pred": (0.0, 10.0), "gold_spans": [[0.0, 5.0]],
        "expected_IoP": 0.5, "expected_IoU": 0.5,
        "rationale": "intersection=5, pred_len=10 -> IoP=0.5 exactly; gold fully inside pred so IoU=0.5 too.",
    },
    {
        "name": "gold_fully_inside_pred",
        "pred": (0.0, 20.0), "gold_spans": [[5.0, 8.0]],
        "expected_IoP": 0.15, "expected_IoU": 0.15,
        "rationale": "gold (len 3) entirely inside pred (len 20): union=pred_len, so IoU==IoP==3/20=0.15 -- a sanity identity both implementations must satisfy.",
    },
    {
        "name": "pred_fully_inside_gold",
        "pred": (5.0, 8.0), "gold_spans": [[0.0, 20.0]],
        "expected_IoP": 1.0, "expected_IoU": 0.15,
        "rationale": "pred (len 3) entirely inside gold (len 20): IoP=1.0 (100% of prediction overlaps), IoU=3/20=0.15 (union=gold_len).",
    },
    {
        "name": "empty_prediction",
        "pred": (0.0, 0.0), "gold_spans": [[5.0, 10.0]],
        "expected_IoP": 0.0, "expected_IoU": 0.0,
        "rationale": "no retrieved frames -> predicted_span_from_frames([]) = (0,0); the zero-width point (0,0) falls outside gold [5,10], so both implementations should agree on 0,0 (distinct from the zero-duration-INSIDE-gold divergence case above).",
    },
]

rows = []
for c in CASES:
    em_iop, em_iou = score_em(c["pred"], c["gold_spans"])
    ng_iop, ng_iou = score_ng(c["pred"], c["gold_spans"])
    match = (abs(em_iop - ng_iop) < 1e-9) and (abs(em_iou - ng_iou) < 1e-9)
    rows.append({
        "name": c["name"], "pred": c["pred"], "gold_spans": c["gold_spans"],
        "expected_IoP": c["expected_IoP"], "expected_IoU": c["expected_IoU"],
        "eval_metrics_IoP": em_iop, "eval_metrics_IoU": em_iou,
        "nextgqa_metrics_IoP": ng_iop, "nextgqa_metrics_IoU": ng_iou,
        "match": match, "rationale": c["rationale"],
    })
    print(f"{c['name']}: eval_metrics=({em_iop},{em_iou}) nextgqa_metrics=({ng_iop},{ng_iou}) match={match}")

with open(REPO / "metric_parity_synthetic.csv", "w", newline="") as f:
    fieldnames = ["name", "pred", "gold_spans", "expected_IoP", "expected_IoU",
                  "eval_metrics_IoP", "eval_metrics_IoU", "nextgqa_metrics_IoP", "nextgqa_metrics_IoU",
                  "match", "rationale"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

n_mismatch = sum(1 for r in rows if not r["match"])
print(f"\n{n_mismatch}/{len(rows)} synthetic cases mismatch")
