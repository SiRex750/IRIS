"""val_confirm reproduction check for the --split parameterisation.

Re-runs the retrieval + span-construction path for every val_confirm question
through the GENERALISED loader (load_split_questions("val_confirm")) and diffs
the result against the committed tuning/val_confirm_e2e_per_question.csv.

Why not simply re-run the whole script and diff the CSV byte-for-byte:

  1. Two columns (retrieval_span_ms, caption_answer_ms) are wall-clock
     measurements. They cannot be byte-identical between any two runs, so a
     literal byte-for-byte comparison of the full CSV is unsatisfiable by
     construction, independent of this change.

  2. The captioner silently changed after the committed baseline was recorded
     (minicpm-v -> minicpm-v4.6; see iris/aria.py:130-134, which picks whichever
     ollama tag happens to be present rather than pinning one). Every
     caption-derived column would therefore differ for reasons that have
     nothing to do with --split.

So this check targets exactly the columns the captioner and the answerer cannot
influence: retrieval output, span construction, and gold. If --split is a true
no-op for val_confirm, these must match to the last decimal place. Anything
that differs here IS attributable to the parameterisation.

Read-only: opens the existing val_confirm index cache, writes nothing to it.
No answerer call, no captioner call.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import iris.ingest as iris_ingest  # noqa: E402
import iris.query as iris_query  # noqa: E402
from iris.retrieval_entry import retrieve_for_question  # noqa: E402
from eval.metrics import predicted_span_from_frames_peak  # noqa: E402
import val_confirm_e2e_eval as ev  # noqa: E402
from part3_tune import load_frozen_state  # noqa: E402

nextgqa_metrics = ev.nextgqa_metrics

COMMITTED = REPO / "tuning" / "val_confirm_e2e_per_question.csv"
OUT = Path(__file__).resolve().parent / "val_confirm_reproduction.json"

# Columns the retrieval+span path fully determines. The captioner and answerer
# cannot touch any of these.
COMPARED = ["pred_span_start", "pred_span_end", "iop", "iou", "used_clip_anchor",
            "gold_spans", "gold_answer_idx", "type", "question"]
EXCLUDED_TIMING = ["retrieval_span_ms", "caption_answer_ms"]
EXCLUDED_MODEL = ["pred_answer_idx", "pred_answer_label", "acc_qa",
                  "acc_gqa_unverified", "raw_answer_nonempty"]


def main() -> None:
    frozen = load_frozen_state()["frozen"]
    cfg = ev.make_e2e_config(frozen, ev.build_arg_parser().parse_args([]))
    half_width_s = float(frozen["span_method_half_width_s"])
    h = ev.ingest_config_hash(cfg)
    cache_dir = Path(ev.SPLIT_SPECS["val_confirm"]["index_cache_dir"])

    with open(COMMITTED, newline="") as f:
        committed = {(r["video"], r["qid"]): r for r in csv.DictReader(f)}

    questions = ev.load_split_questions("val_confirm")
    print(f"[repro] loader returned {len(questions)} questions; committed CSV has {len(committed)}")

    order_match = [(q["video"], q["qid"]) for q in questions] == list(committed.keys())
    print(f"[repro] question identity+order match: {order_match}")

    index_cache: dict = {}
    mismatches: list[dict] = []
    checked = 0
    missing_index = 0
    t0 = time.perf_counter()

    for q in questions:
        vid = q["video"]
        key = (vid, q["qid"])
        if key not in committed:
            mismatches.append({"key": key, "reason": "absent from committed CSV"})
            continue
        idx_path = cache_dir / f"{vid}__{h}"
        if not idx_path.with_suffix(idx_path.suffix + ".npz").exists():
            missing_index += 1
            continue
        if vid not in index_cache:
            index_cache[vid] = iris_ingest.load_index(str(idx_path))
        index = index_cache[vid]

        frames, _plan, _tel = retrieve_for_question(
            q["question"], index, cfg, type_code=q.get("type"), family=q.get("family"),
        )
        qe, _ = iris_query._call_embed_query(q["question"], cfg)
        pred_span, used_clip_anchor = predicted_span_from_frames_peak(
            frames, qe, half_width_s=half_width_s, duration_s=q["duration"],
        )
        gold_tuples = [(g[0], g[1]) for g in q["gold_spans"]]
        iop = nextgqa_metrics.iop(pred_span[0], pred_span[1], gold_tuples)
        iou = nextgqa_metrics.iou(pred_span[0], pred_span[1], gold_tuples)

        now = {
            "pred_span_start": f"{round(pred_span[0], 3)}",
            "pred_span_end": f"{round(pred_span[1], 3)}",
            "iop": f"{round(iop, 5)}",
            "iou": f"{round(iou, 5)}",
            "used_clip_anchor": str(used_clip_anchor),
            "gold_spans": json.dumps(q["gold_spans"]),
            "gold_answer_idx": str(q["gold_answer_idx"]),
            "type": str(q.get("type")),
            "question": q["question"],
        }
        old = committed[key]
        diff = {}
        for col in COMPARED:
            a, b = str(old[col]), now[col]
            if a != b:
                # Numeric columns: compare as floats so 4.4 vs 4.400 is not a
                # false positive from CSV float formatting.
                try:
                    if abs(float(a) - float(b)) < 1e-9:
                        continue
                except (TypeError, ValueError):
                    pass
                diff[col] = {"committed": a, "now": b}
        if diff:
            mismatches.append({"video": vid, "qid": q["qid"], "diff": diff})
        checked += 1

    elapsed = time.perf_counter() - t0
    result = {
        "committed_csv": str(COMMITTED),
        "n_questions_loaded": len(questions),
        "n_rows_committed": len(committed),
        "question_identity_and_order_match": order_match,
        "n_compared": checked,
        "n_skipped_missing_index": missing_index,
        "columns_compared": COMPARED,
        "columns_excluded_timing": EXCLUDED_TIMING,
        "columns_excluded_model_dependent": EXCLUDED_MODEL,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:50],
        "verdict": (
            "IDENTICAL on every retrieval/span/gold column"
            if not mismatches and order_match else "MISMATCH -- see mismatches"
        ),
        "elapsed_s": round(elapsed, 1),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "mismatches"}, indent=2))
    print(f"[repro] wrote {OUT}")


if __name__ == "__main__":
    main()
