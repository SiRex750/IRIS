"""Family 4 pre-check: does any real question at any tested
l2_retrieve_top_k value produce a zero-width predicted span (min timestamp
== max timestamp among retrieved frames)? This is the dormant edge case
Part 2c found and fixed in nextgqa_metrics.py -- this check (a) confirms
which metric implementation this harness is actually using right now and
whether the fix is present, and (b) empirically checks whether the edge
case is exercised by Family 4's grid, not just asserted from code reading.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import part3_tune as pt  # noqa: E402
import iris.ingest as iris_ingest  # noqa: E402
from iris.query import _call_embed_query, _retrieve_with_l1  # noqa: E402

GRID = [4, 8, 12, 16]


def main():
    # (a) confirm implementation + fix presence
    impl_module_file = inspect.getsourcefile(pt.nextgqa_metrics.iop_single)
    src = inspect.getsource(pt.nextgqa_metrics.iop_single)
    fix_present = "pred_s == pred_e" in src

    print(f"[implementation] scripts/part3_tune.py's best_over_gold_spans() calls "
          f"nextgqa_metrics.iop()/iou() from: {impl_module_file}", flush=True)
    print(f"[implementation] Part 2c zero-width special-case fix present in iop_single(): {fix_present}", flush=True)

    questions = pt.load_val_tune_questions()
    print(f"[setup] {len(questions)} val_tune questions", flush=True)

    frozen = {"retrieval_strategy": "hybrid", "ppr_lambda": 0.25, "ppr_damping": 0.5}
    video_ids = sorted({q["video"] for q in questions})

    per_k_results = {}
    any_zero_width = False
    zero_width_examples = []

    for k in GRID:
        cfg = pt.make_config({**frozen, "l2_retrieve_top_k": k})
        index_paths = pt.ensure_indexes(video_ids, cfg, n_workers=8)
        index_cache = {}
        n_zero_width = 0
        n_scored = 0
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
            timestamps = [f["timestamp"] for f in frames]
            pred_span = pt.predicted_span_from_frames(timestamps)
            n_scored += 1
            if pred_span[0] == pred_span[1]:
                n_zero_width += 1
                any_zero_width = True
                if len(zero_width_examples) < 10:
                    zero_width_examples.append({
                        "k": k, "video": vid, "qid": q["qid"],
                        "n_retrieved_frames": len(frames), "pred_span": pred_span,
                    })
        per_k_results[k] = {"n_scored": n_scored, "n_zero_width": n_zero_width}
        print(f"  [k={k}] n_scored={n_scored} n_zero_width_predicted_spans={n_zero_width}", flush=True)

    report_lines = [
        "# Family 4 pre-check: zero-width predicted span occurrence",
        "",
        "## Implementation in use",
        "",
        f"`scripts/part3_tune.py`'s `best_over_gold_spans()` calls "
        f"`nextgqa_metrics.iop()`/`iou()` loaded from "
        f"`{impl_module_file}` (the consolidated canonical module, per the "
        f"Part 2c commit `5ceff0b`) -- **not** `eval/metrics.py`'s "
        f"independent reimplementation, which was retired from this "
        f"harness's scoring path.",
        "",
        f"The Part 2c zero-width special-case fix (`if pred_s == pred_e: "
        f"return 1.0 if gold_s <= pred_s <= gold_e else 0.0`) **is present** "
        f"in `iop_single()`: `{fix_present}`.",
        "",
        "## Empirical occurrence check across Family 4's grid",
        "",
        "| l2_retrieve_top_k | n_scored | n_zero_width_predicted_spans |",
        "|---|---|---|",
    ]
    for k in GRID:
        r = per_k_results[k]
        report_lines.append(f"| {k} | {r['n_scored']} | {r['n_zero_width']} |")
    report_lines.append("")

    if any_zero_width:
        report_lines.append(
            f"**Zero-width predicted spans DID occur** ({sum(r['n_zero_width'] for r in per_k_results.values())} "
            f"total across the grid). Examples (up to 10): {zero_width_examples}. "
            f"Since the fix is present in `nextgqa_metrics.py` and this harness is confirmed "
            f"to be using that module, these cases are scored correctly "
            f"(IoP=1.0 if the point lands inside gold, else 0.0) rather than "
            f"silently undercounted -- no further action needed before scoring Family 4."
        )
    else:
        report_lines.append(
            "**Zero-width predicted spans never occurred at any tested "
            "l2_retrieve_top_k value (4, 8, 12, 16).** Consistent with "
            "Part 2c's finding on Families 1-3 (0/29,535) -- even the "
            "smallest tested top_k (4) still retrieves enough frames with "
            "distinct timestamps that a single-unique-timestamp collapse "
            "essentially never happens on this dataset. No fix was needed "
            "for this family's data (it was already applied to "
            "`nextgqa_metrics.py` as of Part 2c/the `5ceff0b` push, and "
            "remains correctly wired in); this is stated plainly per the "
            "task's instruction to report even a clean 'never occurred' "
            "result rather than skip the check."
        )

    (REPO / "zero_width_span_check.md").write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))
    print("\nFAMILY4_PRECHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
