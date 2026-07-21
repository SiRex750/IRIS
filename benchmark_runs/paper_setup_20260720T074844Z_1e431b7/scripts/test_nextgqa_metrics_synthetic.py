"""Synthetic, handwritten unit tests for nextgqa_metrics.py. No real dataset predictions used.

Each expected value below was computed by hand (shown in the comment) and independently
cross-checked by re-deriving intersection/union arithmetic, not copied from the implementation.
Run: python test_nextgqa_metrics_synthetic.py
"""
import sys
from nextgqa_metrics import iop, iou, iop_single, iou_single, acc_gqa, frame_index_to_seconds, temporal_hit_at_k

FAILURES = []


def check(name, got, expected, tol=1e-9):
    ok = abs(got - expected) <= tol if isinstance(expected, float) else got == expected
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: got={got} expected={expected}")
    if not ok:
        FAILURES.append(name)


def main():
    # Exact match: pred == gold == [2,6] -> intersection=4, pred_len=4, union=4
    check("exact_match_iop", iop_single(2, 6, 2, 6), 4 / 4)   # = 1.0
    check("exact_match_iou", iou_single(2, 6, 2, 6), 4 / 4)   # = 1.0

    # No overlap: pred=[0,2], gold=[5,8] -> intersection=0
    check("no_overlap_iop", iop_single(0, 2, 5, 8), 0.0)
    check("no_overlap_iou", iou_single(0, 2, 5, 8), 0.0)

    # Partial overlap: pred=[0,10], gold=[5,15] -> intersection=[5,10]=5, pred_len=10, union=[0,15]=15
    check("partial_overlap_iop", iop_single(0, 10, 5, 15), 5 / 10)   # = 0.5
    check("partial_overlap_iou", iou_single(0, 10, 5, 15), 5 / 15)   # = 0.3333333333333333

    # Prediction fully inside gold: pred=[4,6], gold=[0,10] -> intersection=2, pred_len=2, union=10
    check("pred_inside_gold_iop", iop_single(4, 6, 0, 10), 2 / 2)    # = 1.0 (whole prediction is inside)
    check("pred_inside_gold_iou", iou_single(4, 6, 0, 10), 2 / 10)   # = 0.2

    # Gold fully inside prediction: pred=[0,10], gold=[4,6] -> intersection=2, pred_len=10, union=10
    check("gold_inside_pred_iop", iop_single(0, 10, 4, 6), 2 / 10)   # = 0.2
    check("gold_inside_pred_iou", iou_single(0, 10, 4, 6), 2 / 10)   # = 0.2 (union = pred span = 10)

    # Prediction at video boundary: video [0,20], pred=[18,20], gold=[19,20]
    # intersection=1, pred_len=2, union=(2)+(1)-(1)=2
    check("boundary_pred_iop", iop_single(18, 20, 19, 20), 1 / 2)    # = 0.5
    check("boundary_pred_iou", iou_single(18, 20, 19, 20), 1 / 2)    # = 0.5

    # Equality at 0.3: pred=[0,10], gold=[7,10] -> IoU intersection=3, union=10+3-3=10 -> 0.3 exactly
    check("equality_at_0.3_iou", iou_single(0, 10, 7, 10), 0.3)

    # Equality at 0.5: pred=[0,10], gold=[5,10] -> IoP intersection=5,pred_len=10 -> 0.5 exactly
    check("equality_at_0.5_iop", iop_single(0, 10, 5, 10), 0.5)

    # Multiple gold spans: pred=[4,6] vs gold=[[0,1],[4,5.5]] -> best IoP against 2nd span:
    # intersection=[4,5.5]=1.5, pred_len=2 -> 0.75; against 1st span intersection=0 -> 0.0; max=0.75
    check("multi_gold_best_iop", iop(4, 6, [(0, 1), (4, 5.5)]), 0.75)

    # Zero-duration predicted span INSIDE gold: pred=[5,5], gold=[0,10].
    # CORRECTED (Part 2c metric parity check, cross-referenced against the
    # actual official doc-doc/NExT-GQA scorer, code/TempGQA/eval_ground.py::
    # get_tIoU): a zero-width predicted point special-cases to (IoU=0,
    # IoP=1) when it falls inside the gold span -- perfect precision, zero
    # measurable union overlap. The previous expectation here (IoP=0.0) was
    # this test's own assumption, written without consulting the official
    # source, and was NOT what the real scorer does -- it undercounted a
    # single-timestamp retrieval landing exactly inside the gold window.
    check("zero_duration_pred_inside_gold_iop", iop_single(5, 5, 0, 10), 1.0)
    check("zero_duration_pred_inside_gold_iou", iou_single(5, 5, 0, 10), 0.0)  # union=10+0-0=10, inter=0 -> 0/10=0.0

    # Zero-duration predicted span OUTSIDE gold: pred=[15,15], gold=[0,10] -> point not in gold -> IoP=0.0
    check("zero_duration_pred_outside_gold_iop", iop_single(15, 15, 0, 10), 0.0)
    check("zero_duration_pred_outside_gold_iou", iou_single(15, 15, 0, 10), 0.0)

    # Reversed span: pred_s=8, pred_e=3 (e<s) -> clamped to zero-duration span -> IoP=0.0
    check("reversed_span_iop", iop_single(8, 3, 0, 10), 0.0)

    # Zero-duration gold span: gold=[5,5], pred=[0,10] -> intersection=0 (measure zero) -> IoP=0.0
    check("zero_duration_gold_iop", iop_single(0, 10, 5, 5), 0.0)

    # Out-of-bounds span (prediction extends past a nominal video end, e.g. video len=10 pred=[8,15])
    # against gold=[9,10]: intersection=[9,10]=1, pred_len=7 -> IoP=1/7
    check("out_of_bounds_iop", iop_single(8, 15, 9, 10), 1 / 7)

    # Acc@GQA: correct answer AND IoP>=0.5 -> True; correct answer but IoP<0.5 -> False
    check("acc_gqa_true", acc_gqa("A", "A", 0, 10, [(5, 10)]), True)   # IoP=0.5 -> passes (>=0.5)
    check("acc_gqa_false_low_iop", acc_gqa("A", "A", 0, 10, [(9, 10)]), False)  # IoP=0.1
    check("acc_gqa_false_wrong_answer", acc_gqa("A", "B", 0, 10, [(0, 10)]), False)

    # Frame-index-to-seconds: frame 30 at fps=30 -> 1.0s exactly; frame 45 at fps=29.97 -> 1.501835...
    check("frame_to_seconds_exact", frame_index_to_seconds(30, 30.0), 1.0)
    check("frame_to_seconds_fractional_fps", frame_index_to_seconds(45, 29.97), 45 / 29.97)
    try:
        frame_index_to_seconds(10, 0.0)
        print("[FAIL] frame_to_seconds_zero_fps: expected ValueError, none raised")
        FAILURES.append("frame_to_seconds_zero_fps")
    except ValueError:
        print("[PASS] frame_to_seconds_zero_fps: raised ValueError as expected")

    # Diagnostic-only Temporal Hit@K (explicitly separate from official IoU/IoP; sanity-checked
    # here so it's never silently mislabeled as NExT-GQA Recall@K downstream)
    check("temporal_hit_at_1_hit", temporal_hit_at_k([3.0, 20.0], [(0, 5)], k=1), 1.0)
    check("temporal_hit_at_1_miss", temporal_hit_at_k([20.0, 3.0], [(0, 5)], k=1), 0.0)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("ALL SYNTHETIC METRIC TESTS PASSED (handwritten expected values, no real predictions used)")


if __name__ == "__main__":
    main()
