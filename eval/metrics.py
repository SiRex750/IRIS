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

import numpy as np


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
    """Method A (existing, unchanged): predicted span = [min, max] timestamp
    among retrieved frames. A single retrieved frame yields a zero-width
    span (point prediction)."""
    if not timestamps:
        return (0.0, 0.0)
    return (min(timestamps), max(timestamps))


def _frame_score(frame: dict) -> float:
    score = frame.get("pagerank_score")
    if score is None:
        score = frame.get("last_retrieval_score", 0.0)
    return score


def _pick_peak_by_clip(retrieved_frames: list[dict], query_embedding) -> dict | None:
    """Highest CLIP-cosine frame WITHIN the already-retrieved pool.
    Returns None when unavailable (caller must fall back to retrieved_frames[0]
    explicitly and record that fallback -- never let this fail silently)."""
    if query_embedding is None:
        return None
    q = np.asarray(query_embedding, dtype=np.float32).ravel()
    qn = float(np.linalg.norm(q))
    if qn == 0.0:
        return None
    q = q / qn
    best, best_sim = None, -1.0
    for fr in retrieved_frames:
        emb = fr.get("clip_embedding")
        if emb is None:
            continue
        v = np.asarray(emb, dtype=np.float32).ravel()
        vn = float(np.linalg.norm(v))
        if vn == 0.0:
            continue
        sim = float(np.dot(q, v / vn))
        if sim > best_sim:
            best, best_sim = fr, sim
    return best


def predicted_span_from_frames_clustered(
    retrieved_frames: list[dict], gap_threshold_s: float = 3.0, tail_trim_pct: float = 20.0,
    query_embedding=None,
) -> tuple[float, float]:
    """Method B (Part 3c): score-weighted temporal clustering + tail-trim.

    1. Sort by timestamp ascending.
    2. Split into clusters wherever the gap between consecutive timestamps
       exceeds gap_threshold_s.
    3. Sum pagerank_score (falling back to last_retrieval_score) per
       cluster; keep the cluster with the highest total.
    4. Within that cluster, drop the bottom tail_trim_pct% of frames by
       score (GranAlign-style refinement, avoids a cluster being diluted by
       low-relevance frames sitting between two good ones).
    5. Predicted span = [min, max] timestamp of what remains.

    gap_threshold_s=3.0 and tail_trim_pct=20 are first-pass reasoned
    defaults for this comparison pass, not tuned -- flagged as a future
    micro-tuning candidate if this method wins.

    With tail_trim_pct=0 and a single cluster (no gap exceeds
    gap_threshold_s), this reduces to exactly Method A's output -- see
    tests/test_span_methods.py::test_method_b_reduces_to_method_a.
    """
    if not retrieved_frames:
        return (0.0, 0.0)

    frames_sorted = sorted(retrieved_frames, key=lambda f: f["timestamp"])
    clusters: list[list[dict]] = [[frames_sorted[0]]]
    for f in frames_sorted[1:]:
        if f["timestamp"] - clusters[-1][-1]["timestamp"] > gap_threshold_s:
            clusters.append([f])
        else:
            clusters[-1].append(f)

    anchor = _pick_peak_by_clip(retrieved_frames, query_embedding)
    if anchor is not None:
        anchor_idx = anchor["frame_idx"]
        best_cluster = next((c for c in clusters if any(f["frame_idx"] == anchor_idx for f in c)),
                             max(clusters, key=lambda c: sum(_frame_score(f) for f in c)))
    else:
        best_cluster = max(clusters, key=lambda c: sum(_frame_score(f) for f in c))

    n = len(best_cluster)
    n_drop = int(n * tail_trim_pct / 100.0)
    n_keep = max(1, n - n_drop)
    kept = sorted(best_cluster, key=_frame_score, reverse=True)[:n_keep] if n_drop > 0 else best_cluster

    timestamps = [f["timestamp"] for f in kept]
    return (min(timestamps), max(timestamps))


def predicted_span_from_frames_scene(
    retrieved_frames: list[dict], scene_spans_map: dict[int, tuple[float, float]],
    query_embedding=None,
) -> tuple[tuple[float, float], bool]:
    """Method C (Part 3c): look up the real scene boundary of the top-ranked
    retrieved frame (retrieved_frames[0], guaranteed rank-1 by
    iris.l2_asphodel.retrieve_ppr's ordering contract) instead of inventing
    a span from arbitrary retrieved timestamps.

    Returns (predicted_span, fallback_triggered). Falls back to Method A
    (over ALL retrieved frames, not just the top one) when the top frame's
    scene_id is unassigned (-1, e.g. the hermetic/synthetic-records or
    audio-only ingest path) or has no entry in scene_spans_map (e.g. an
    index built before the Part 3c scene_spans persistence change)."""
    if not retrieved_frames:
        return (0.0, 0.0), False

    top = _pick_peak_by_clip(retrieved_frames, query_embedding) or retrieved_frames[0]
    scene_id = top.get("scene_id")
    if scene_id is not None and scene_id >= 0 and scene_id in scene_spans_map:
        return scene_spans_map[scene_id], False

    timestamps = [f["timestamp"] for f in retrieved_frames]
    return predicted_span_from_frames(timestamps), True


def predicted_span_from_frames_peak(
    retrieved_frames: list[dict], query_embedding,
    half_width_s: float = 2.2, duration_s: float | None = None,
) -> tuple[tuple[float, float], bool]:
    """Method D (Part 3c): fixed-width window centred on the highest
    CLIP-similarity frame within the retrieved pool.

    Retrieval is NOT changed -- only which retrieved frame anchors the span.
    Reuses the same query embedding retrieval already used; no second embed
    path, no new model call.

    Returns (predicted_span, used_clip_anchor). used_clip_anchor=False means
    _pick_peak_by_clip returned None (no query embedding, zero-norm vector,
    or no frame carries clip_embedding) and this fell back to
    retrieved_frames[0] -- the same degrade-to-rank-1 discipline Methods B/C
    already use, so the caller can log it the same way as Method C's
    `fallback_triggered`.

    half_width_s=2.2 is a provisional constant, NOT derived from data or
    tuned -- a starting guess pending the val_tune sweep. Do not treat it as
    validated until that run reports back.
    """
    if not retrieved_frames:
        return (0.0, 0.0), False

    peak = _pick_peak_by_clip(retrieved_frames, query_embedding)
    used_clip_anchor = peak is not None
    if peak is None:
        peak = retrieved_frames[0]

    t = float(peak["timestamp"])
    lo = max(0.0, t - half_width_s)
    hi = t + half_width_s
    if duration_s is not None:
        hi = min(float(duration_s), hi)
    return (lo, hi), used_clip_anchor


def is_zero_width_span(span: tuple[float, float]) -> bool:
    """True when a predicted span collapses to a single instant
    (min timestamp == max timestamp). Methods B and D can both degenerate
    to this -- a single-frame cluster for B, or an anchor frame sitting at
    duration_s for D's clamp -- and a zero-width span mints IoP=1.0 whenever
    it happens to land inside the gold span, which silently inflates the
    metric unless callers track how often it happens."""
    return span[0] == span[1]
