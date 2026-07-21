"""Official NExT-GQA metric implementations, audited against eval/grounding_scorer.py.

eval/grounding_scorer.py already implements `iop()` (single best predicted span vs union of
gold spans) and `frames_in_window()` (diagnostic-only "fraction retrieved in window", NOT an
official NExT-GQA metric despite the superficial resemblance to Hit-rate). This module is a
SEPARATE, setup-only artifact that implements the *complete* official metric set for audit and
synthetic testing; it deliberately does not modify eval/grounding_scorer.py (production code is
out of scope for a setup-only task). If/when these are adopted into eval/, the two `iop()`
implementations should be reconciled and one deleted.

No real dataset question is scored by this module during setup -- it is exercised only against
handwritten synthetic spans in tests/test_nextgqa_metrics_synthetic.py.
"""
from __future__ import annotations


def _clip_span(s: float, e: float):
    """Defensive clamp: a reversed span (e<s) has zero measure by convention here."""
    if e < s:
        return s, s
    return s, e


def intersection(a_s: float, a_e: float, b_s: float, b_e: float) -> float:
    lo = max(a_s, b_s)
    hi = min(a_e, b_e)
    return max(0.0, hi - lo)


def iop_single(pred_s: float, pred_e: float, gold_s: float, gold_e: float) -> float:
    """IoP = intersection(pred, gold) / duration(pred). 0.0 if pred has zero duration."""
    pred_s, pred_e = _clip_span(pred_s, pred_e)
    pred_len = pred_e - pred_s
    if pred_len <= 0:
        return 0.0
    return intersection(pred_s, pred_e, gold_s, gold_e) / pred_len


def iou_single(pred_s: float, pred_e: float, gold_s: float, gold_e: float) -> float:
    """IoU = intersection(pred, gold) / union(pred, gold). 0.0 if union is zero."""
    pred_s, pred_e = _clip_span(pred_s, pred_e)
    gold_s, gold_e = _clip_span(gold_s, gold_e)
    inter = intersection(pred_s, pred_e, gold_s, gold_e)
    union = (pred_e - pred_s) + (gold_e - gold_s) - inter
    if union <= 0:
        return 0.0
    return inter / union


def best_over_gold_spans(pred_s: float, pred_e: float, gold_spans: list[tuple[float, float]], fn):
    """Official multi-gold-span protocol: score against EACH gold span independently and take
    the maximum -- never concatenate/union unrelated gold spans into one interval before scoring.
    """
    if not gold_spans:
        return 0.0
    return max(fn(pred_s, pred_e, g_s, g_e) for g_s, g_e in gold_spans)


def iop(pred_s: float, pred_e: float, gold_spans: list[tuple[float, float]]) -> float:
    return best_over_gold_spans(pred_s, pred_e, gold_spans, iop_single)


def iou(pred_s: float, pred_e: float, gold_spans: list[tuple[float, float]]) -> float:
    return best_over_gold_spans(pred_s, pred_e, gold_spans, iou_single)


def acc_qa(pred_answer, gold_answer) -> bool:
    return pred_answer == gold_answer


def acc_gqa(pred_answer, gold_answer, pred_s: float, pred_e: float, gold_spans: list[tuple[float, float]]) -> bool:
    """Acc@GQA = answered correctly AND IoP >= 0.5 (official NExT-GQA definition)."""
    return acc_qa(pred_answer, gold_answer) and iop(pred_s, pred_e, gold_spans) >= 0.5


def frame_index_to_seconds(frame_idx: int, fps: float) -> float:
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return frame_idx / fps


# ---- Diagnostic-only metrics (explicitly NOT NExT-GQA official metrics) -------------------
# Registered separately in metric_registry.json with is_official=false. In particular
# Temporal Hit@K must never be reported as NExT-GQA Recall@K -- Hit@K here means "at least one
# of the top-K retrieved frame timestamps falls inside a gold span", which is a retrieval
# diagnostic over discrete frames, not the official continuous-span IoU/IoP protocol.

def temporal_hit_at_k(ts_list: list[float], gold_spans: list[tuple[float, float]], k: int) -> float:
    topk = ts_list[:k]
    if not topk:
        return 0.0
    return 1.0 if any(g_s <= t <= g_e for t in topk for g_s, g_e in gold_spans) else 0.0
