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


def _cosine(q: np.ndarray, v) -> float | None:
    """Cosine similarity, or None if either vector is missing/zero-norm."""
    if v is None:
        return None
    vv = np.asarray(v, dtype=np.float32).ravel()
    vn = float(np.linalg.norm(vv))
    if vn == 0.0:
        return None
    return float(np.dot(q, vv / vn))


def _normalize_query(query_embedding) -> np.ndarray | None:
    if query_embedding is None:
        return None
    q = np.asarray(query_embedding, dtype=np.float32).ravel()
    qn = float(np.linalg.norm(q))
    if qn == 0.0:
        return None
    return q / qn


def compute_similarity_profile(
    survivor_frames: list, query_embedding, smooth_s: float = 0.5,
) -> dict:
    """Method E, stage 1 (spec steps 1-3): per-video CLIP-similarity profile
    over ALL survivor frames (index.frames), not just the retrieved top-K.

    survivor_frames elements may be FrameRecord objects (attribute access:
    .timestamp/.clip_embedding/.frame_idx) or dicts (key access) -- both are
    supported so this can run against either index.frames directly or a
    dict-projected copy.

    smooth_s: centered time-window moving average, in seconds (not an
    index-count window, since survivor frames are not evenly spaced in
    time). smooth_s=0 means no smoothing (s~ = s).

    Returns a dict with sorted timestamps/frame_idxs, the smoothed+
    normalized z array, n_skipped (no embedding or zero-norm), n_scored,
    and degenerate (True iff max(s~) == min(s~), e.g. <2 scored frames or a
    perfectly flat profile -- callers must fall back to Method D on this).
    Empty/degenerate results always still populate n_skipped/n_scored so
    telemetry is never silently lost.
    """
    q = _normalize_query(query_embedding)
    if q is None:
        return {"timestamps": np.array([]), "frame_idxs": [], "z": np.array([]),
                "n_skipped": len(survivor_frames), "n_scored": 0, "degenerate": True}

    def _get(fr, name):
        return fr.get(name) if isinstance(fr, dict) else getattr(fr, name, None)

    pairs = []
    n_skipped = 0
    for fr in survivor_frames:
        sim = _cosine(q, _get(fr, "clip_embedding"))
        if sim is None:
            n_skipped += 1
            continue
        pairs.append((float(_get(fr, "timestamp")), _get(fr, "frame_idx"), sim))

    if len(pairs) < 2:
        return {"timestamps": np.array([p[0] for p in pairs]), "frame_idxs": [p[1] for p in pairs],
                "z": np.array([]), "n_skipped": n_skipped, "n_scored": len(pairs), "degenerate": True}

    pairs.sort(key=lambda p: p[0])
    timestamps = np.array([p[0] for p in pairs], dtype=np.float64)
    frame_idxs = [p[1] for p in pairs]
    s = np.array([p[2] for p in pairs], dtype=np.float64)

    if smooth_s and smooth_s > 0:
        s_smooth = np.empty_like(s)
        for i, t in enumerate(timestamps):
            window = np.abs(timestamps - t) <= (smooth_s / 2.0)
            s_smooth[i] = s[window].mean()
    else:
        s_smooth = s

    lo, hi = float(s_smooth.min()), float(s_smooth.max())
    if hi == lo:
        return {"timestamps": timestamps, "frame_idxs": frame_idxs, "z": np.array([]),
                "n_skipped": n_skipped, "n_scored": len(pairs), "degenerate": True}

    z = (s_smooth - lo) / (hi - lo)
    return {"timestamps": timestamps, "frame_idxs": frame_idxs, "z": z,
            "n_skipped": n_skipped, "n_scored": len(pairs), "degenerate": False}


def span_from_similarity_profile(
    profile: dict, retrieved_frames: list[dict], duration_s: float,
    tau: float, w_min: float, gap: int = 1, anchor_source: str = "topk",
    w_max: float | None = None,
) -> tuple[tuple[float, float], bool, dict]:
    """Method E, stage 2 (spec steps 4-7): extract a span from an already-
    computed similarity profile (see compute_similarity_profile). Kept
    separate from stage 1 so a tau/w_min sweep can reuse one profile
    computation per question across every (tau, w_min) cell.

    Degenerate-profile fallback: a profile only degenerates when fewer than
    2 survivor frames carry a usable embedding, or every survivor's
    (smoothed) cosine is identical -- both mean the profile carries no
    usable anchor signal. Falls back to the same discipline Method D uses
    when it can't rank frames (predicted_span_from_frames_peak with
    query_embedding=None, which forces the retrieved_frames[0] anchor and
    reports used_clip_anchor=False), width=w_min. In >2685 questions on
    val_tune this branch essentially never fires with real embeddings, but
    it must still be correct: never a bare (0.0, 0.0) span unrelated to the
    video's actual duration.

    Returns (span, used_clip_anchor, telemetry). telemetry always carries
    n_survivors_scored/degenerate_profile_fallback/final_width/
    n_frames_included so no cell can complete without full accounting.
    """
    telemetry = {
        "n_survivors_scored": profile["n_scored"], "n_skipped": profile["n_skipped"],
        "degenerate_profile_fallback": False, "final_width": None, "n_frames_included": 0,
    }
    if profile["degenerate"] or len(profile["z"]) == 0:
        telemetry["degenerate_profile_fallback"] = True
        span, used_clip_anchor = predicted_span_from_frames_peak(
            retrieved_frames, None, half_width_s=w_min / 2.0, duration_s=duration_s,
        )
        telemetry["final_width"] = span[1] - span[0]
        return span, used_clip_anchor, telemetry

    timestamps = profile["timestamps"]
    frame_idxs = profile["frame_idxs"]
    z = profile["z"]
    n = len(z)
    retrieved_frame_idxs = [f["frame_idx"] for f in retrieved_frames]

    if anchor_source == "topk" and retrieved_frame_idxs:
        idx_pos = {fi: i for i, fi in enumerate(frame_idxs)}
        candidates = [idx_pos[fi] for fi in retrieved_frame_idxs if fi in idx_pos]
        if not candidates:
            candidates = list(range(n))
    else:
        candidates = list(range(n))

    anchor_idx = max(candidates, key=lambda i: z[i])
    used_clip_anchor = True
    t_star = float(timestamps[anchor_idx])

    if w_max is None:
        w_max = min(20.0, 0.5 * float(duration_s))

    left_idx = anchor_idx
    consecutive_below = 0
    j = anchor_idx - 1
    while j >= 0:
        if z[j] >= tau:
            left_idx = j
            consecutive_below = 0
        else:
            consecutive_below += 1
            if consecutive_below > gap:
                break
        j -= 1

    right_idx = anchor_idx
    consecutive_below = 0
    j = anchor_idx + 1
    while j < n:
        if z[j] >= tau:
            right_idx = j
            consecutive_below = 0
        else:
            consecutive_below += 1
            if consecutive_below > gap:
                break
        j += 1

    lo = float(timestamps[left_idx])
    hi = float(timestamps[right_idx])
    n_included = right_idx - left_idx + 1

    lo = max(0.0, lo)
    hi = min(float(duration_s), hi)
    if hi < lo:
        hi = lo

    width = hi - lo
    if width < w_min:
        half = w_min / 2.0
        lo = max(0.0, t_star - half)
        hi = min(float(duration_s), t_star + half)
    elif width > w_max:
        half = w_max / 2.0
        lo = max(0.0, t_star - half)
        hi = min(float(duration_s), t_star + half)

    telemetry["final_width"] = hi - lo
    telemetry["n_frames_included"] = n_included
    telemetry["anchor_timestamp"] = t_star
    return (lo, hi), used_clip_anchor, telemetry


def predicted_span_from_frames_profile(
    survivor_frames: list, retrieved_frames: list[dict], query_embedding,
    duration_s: float, tau: float, w_min: float, smooth_s: float = 0.5,
    gap: int = 1, anchor_source: str = "topk", w_max: float | None = None,
) -> tuple[tuple[float, float], bool, dict]:
    """Method E (new): profile-thresholded adaptive-width span. Single-call
    convenience wrapper around compute_similarity_profile +
    span_from_similarity_profile -- see those for the two-stage
    implementation a tau/w_min sweep should call directly to avoid
    recomputing the CLIP profile once per cell.

    Falls back to Method D (predicted_span_from_frames_peak) whenever the
    profile is degenerate (fewer than 2 scored survivors, or a perfectly
    flat similarity profile) or retrieved_frames is empty; the fallback is
    always reported in the returned telemetry. See span_from_similarity_profile
    for the exact fallback discipline -- this wrapper delegates to it so
    there is exactly one fallback code path, not two that could drift.
    """
    profile = compute_similarity_profile(survivor_frames, query_embedding, smooth_s=smooth_s)
    return span_from_similarity_profile(
        profile, retrieved_frames, duration_s, tau, w_min,
        gap=gap, anchor_source=anchor_source, w_max=w_max,
    )


def weighted_std_topk(retrieved_frames: list[dict], query_embedding) -> tuple[float, bool]:
    """Weighted std-dev of retrieved top-K timestamps, weights = CLIP
    cosine similarity to the query (negative cosines clipped to 0 so the
    weighted-variance formula stays well-defined). Returns (std,
    weight_fallback) -- weight_fallback=True means every weight was zero
    (e.g. all cosines <= 0, or no frame carried an embedding) and an
    unweighted std was used instead."""
    if not retrieved_frames:
        return 0.0, False
    timestamps = np.array([f["timestamp"] for f in retrieved_frames], dtype=np.float64)
    if len(timestamps) == 1:
        return 0.0, False

    q = _normalize_query(query_embedding)
    weights = np.zeros(len(retrieved_frames), dtype=np.float64)
    if q is not None:
        for i, f in enumerate(retrieved_frames):
            sim = _cosine(q, f.get("clip_embedding"))
            weights[i] = max(0.0, sim) if sim is not None else 0.0

    weight_fallback = False
    if weights.sum() <= 0.0:
        weight_fallback = True
        weights = np.ones(len(retrieved_frames), dtype=np.float64)

    mean = float(np.average(timestamps, weights=weights))
    var = float(np.average((timestamps - mean) ** 2, weights=weights))
    return var ** 0.5, weight_fallback


def predicted_span_from_frames_weighted_spread(
    retrieved_frames: list[dict], query_embedding, duration_s: float,
    alpha: float, w_min: float, w_max: float | None = None,
) -> tuple[tuple[float, float], bool, dict]:
    """Method F (new): Method B's weighted-spread intuition with a width
    floor, fixing B's 29.8%-zero-width failure mode.

    w = clamp(alpha * weighted_std(top-K timestamps, weights=CLIP cosine), w_min, w_max)
    span = [t_peak - w/2, t_peak + w/2], clamped to [0, duration_s].
    t_peak is the same CLIP-similarity-peak anchor Method D uses (shared
    _pick_peak_by_clip), so a Method-D vs Method-F comparison isolates the
    width rule, not the anchor.

    w_max defaults to Method E's non-binding sanity clamp, min(20.0, 0.5*D)
    -- the spec does not give Method F its own w_max, and this is the only
    fixed value already established for a `clamp(..., w_min, w_max)` span
    formula in this task.
    """
    if w_max is None:
        w_max = min(20.0, 0.5 * float(duration_s))

    peak = _pick_peak_by_clip(retrieved_frames, query_embedding)
    used_clip_anchor = peak is not None
    if peak is None:
        if not retrieved_frames:
            return (0.0, 0.0), False, {"weight_fallback": False, "raw_std": 0.0, "final_width": 0.0}
        peak = retrieved_frames[0]
    t_peak = float(peak["timestamp"])

    std, weight_fallback = weighted_std_topk(retrieved_frames, query_embedding)
    width = min(max(alpha * std, w_min), w_max)

    lo = max(0.0, t_peak - width / 2.0)
    hi = min(float(duration_s), t_peak + width / 2.0)
    if hi < lo:
        hi = lo
    return (lo, hi), used_clip_anchor, {
        "weight_fallback": weight_fallback, "raw_std": std, "final_width": hi - lo,
    }


def is_structurally_barred(gold_spans: list[list[float]], pred_width: float) -> bool:
    """True iff no predicted span of this width, however placed, could
    reach IoP@0.5 against the longest gold span -- i.e. max_gold_len <
    0.5 * pred_width. Zero-width predictions are never "barred" by this
    definition (they follow the separate zero-width-span IoP=1.0 special
    case in get_tIoU, not a width-vs-gold-length comparison)."""
    if pred_width <= 0:
        return False
    max_gold_len = max((g[1] - g[0]) for g in gold_spans) if gold_spans else 0.0
    return max_gold_len < 0.5 * pred_width


def assert_no_confirm_videos(video_ids, confirm_videos: set) -> None:
    """Fail loudly if any video_ids entry is a val_confirm video. Intended
    to be called on every invocation of a val_tune-only harness (e.g. the
    span-construction sweep) so a split_manifest.json mixup can never
    silently leak confirm-split questions into a tune-only measurement."""
    leaked = sorted(set(video_ids) & set(confirm_videos))
    if leaked:
        raise AssertionError(
            f"val_confirm video(s) leaked into a val_tune-only run: {leaked[:10]}"
            f"{'...' if len(leaked) > 10 else ''} ({len(leaked)} total)"
        )


def is_zero_width_span(span: tuple[float, float]) -> bool:
    """True when a predicted span collapses to a single instant
    (min timestamp == max timestamp). Methods B and D can both degenerate
    to this -- a single-frame cluster for B, or an anchor frame sitting at
    duration_s for D's clamp -- and a zero-width span mints IoP=1.0 whenever
    it happens to land inside the gold span, which silently inflates the
    metric unless callers track how often it happens."""
    return span[0] == span[1]
