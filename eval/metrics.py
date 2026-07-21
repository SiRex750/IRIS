"""NExT-GQA grounding metrics (IoP/IoU) and predicted-span derivation.

IoP/IoU definitions match the official doc-doc/NExT-GQA scorer
(code/TempGQA/eval_ground.py get_tIoU): for gold spans loc=[s,e] and a
predicted span pred=[s',e'],
  intersection = max(0, min(e,e') - max(s,s'))
  union        = max(e,e') - min(s,s')
  IoU = intersection / union
  IoP = intersection / (e' - s')   (predicted-span duration)
For multiple gold spans, take the max-overlap span (never concatenate).
"""
from __future__ import annotations


def get_tIoU(gold_span: tuple[float, float], pred_span: tuple[float, float]) -> tuple[float, float]:
    g0, g1 = gold_span
    p0, p1 = pred_span
    if p0 == p1:
        return (0.0, 1.0) if g0 <= p0 <= g1 else (0.0, 0.0)
    union_lo, union_hi = min(g0, p0), max(g1, p1)
    inter_lo, inter_hi = max(g0, p0), min(g1, p1)
    inter = max(0.0, inter_hi - inter_lo)
    iou = inter / (union_hi - union_lo) if union_hi > union_lo else 0.0
    iop = inter / (p1 - p0) if p1 > p0 else 0.0
    return iou, iop


def best_over_gold_spans(gold_spans: list[list[float]], pred_span: tuple[float, float]) -> tuple[float, float]:
    """Max-overlap gold span (never concatenate multiple gold spans)."""
    best_iou, best_iop = 0.0, 0.0
    for g in gold_spans:
        iou, iop = get_tIoU((g[0], g[1]), pred_span)
        best_iou = max(best_iou, iou)
        best_iop = max(best_iop, iop)
    return best_iou, best_iop


def predicted_span_from_frames(timestamps: list[float]) -> tuple[float, float]:
    """Predicted grounding span = [min, max] timestamp among retrieved frames.
    A single retrieved frame yields a zero-width span (point prediction)."""
    if not timestamps:
        return (0.0, 0.0)
    return (min(timestamps), max(timestamps))
