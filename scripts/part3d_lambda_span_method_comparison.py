"""Part 3d: joint ppr_lambda x span-method comparison. Family 2's earlier
ppr_lambda=0.25 winner was picked using Method A's min-max span, which
mechanically rewards low lambda regardless of true localization quality --
that confound was never removed by re-running with the corrected anchor
(commit 47ebce5) or a fourth span method. This script re-sweeps the original
Family 2 lambda grid with all four span-construction methods (A, B, C, D)
scored side by side, so lambda selection and span-method selection stop
contaminating each other.

retrieval_strategy/ppr_damping/l2_retrieve_top_k are held at their current
tuning/frozen_state.json values -- this run only varies ppr_lambda and the
span-construction method. ppr_lambda is NOT in INGEST_RELEVANT_KEYS (see
part3_tune.INGEST_RELEVANT_KEYS), so it does not affect the ingest
config-hash -- indexes are built ONCE and reused across every lambda value;
only the retrieval call (which lambda directly affects) is repeated per
(lambda, question).

Method C requires scene_spans, same as part3c -- reuses the same dedicated
fresh cache dir (tuning/index_cache_scenespans) so nothing here is
contaminated by stale pre-scene_spans .npz files.

Diagnostic only: does NOT write to tuning/frozen_state.json. No lambda
value is auto-frozen by this script -- that decision is made by a human
after reading tuning/lambda_span_method_comparison.csv.

NOT RUN as part of the task that wrote this file -- no video ingest, no
scoring, no pytest invocation happened. This is a complete, runnable
script staged for a later GPU pass.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import part3_tune as pt  # noqa: E402
import iris.ingest as iris_ingest  # noqa: E402
from iris.query import _call_embed_query, _retrieve_with_l1  # noqa: E402
from eval.metrics import (  # noqa: E402
    best_over_gold_spans, predicted_span_from_frames,
    predicted_span_from_frames_clustered, predicted_span_from_frames_scene,
    predicted_span_from_frames_peak, is_zero_width_span,
)

FRESH_CACHE_DIR = REPO / "tuning" / "index_cache_scenespans"
FRESH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Original Family 2 grid (see part3_tune.FAMILIES["ppr_lambda"]) -- re-swept
# here jointly against all four span methods instead of Method A alone.
LAMBDA_GRID = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]

# Everything else frozen at its current tuning/frozen_state.json value.
# Loaded live below (not hardcoded) so this script can't silently drift
# from whatever the frozen block actually says at run time.
_FROZEN_STATE = json.loads((REPO / "tuning" / "frozen_state.json").read_text())["frozen"]
FROZEN_BASE = {
    "retrieval_strategy": _FROZEN_STATE["retrieval_strategy"],
    "ppr_damping": _FROZEN_STATE["ppr_damping"],
    "l2_retrieve_top_k": _FROZEN_STATE["l2_retrieve_top_k"],
}

GAP_THRESHOLD_S = 3.0
TAIL_TRIM_PCT = 20.0
# Provisional, not tuned -- see eval.metrics.predicted_span_from_frames_peak.
HALF_WIDTH_S = 2.2


def ensure_fresh_indexes(video_ids: list[str], cfg, n_workers: int = 8) -> dict[str, str]:
    h = pt.ingest_config_hash(cfg)
    paths = {}
    todo = []
    for vid in video_ids:
        p = FRESH_CACHE_DIR / f"{vid}__{h}"
        if p.with_suffix(p.suffix + ".npz").exists():
            paths[vid] = str(p)
        else:
            todo.append(vid)
    print(f"    [ingest] {len(todo)}/{len(video_ids)} need fresh ingest (config-hash {h})", flush=True)

    def _do(vid):
        vpath = pt.VIDEO_DIR / f"{vid}.mp4"
        idx = iris_ingest.ingest(str(vpath), cfg)
        out_path = FRESH_CACHE_DIR / f"{vid}__{h}"
        iris_ingest.save_index(idx, str(out_path))
        return vid

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_do, vid): vid for vid in todo}
        for fut in as_completed(futs):
            vid = futs[fut]
            try:
                fut.result()
                paths[vid] = str(FRESH_CACHE_DIR / f"{vid}__{h}")
            except Exception as exc:  # noqa: BLE001
                print(f"    [ingest FAIL] {vid}: {type(exc).__name__}: {exc}", flush=True)
            done += 1
            if done % 100 == 0:
                print(f"    [ingest] {done}/{len(todo)} done", flush=True)
    return paths


def score_span(pred_span, gold_spans) -> dict:
    iou, iop = best_over_gold_spans(gold_spans, pred_span)
    return {"IoP": iop, "IoU": iou, "IoP@0.3": iop >= 0.3, "IoP@0.5": iop >= 0.5,
            "IoU@0.3": iou >= 0.3, "IoU@0.5": iou >= 0.5}


def main():
    questions = pt.load_val_tune_questions()
    print(f"[setup] {len(questions)} val_tune questions", flush=True)
    video_ids = sorted({q["video"] for q in questions})

    # Ingest ONCE: ppr_lambda is not in INGEST_RELEVANT_KEYS, so every
    # lambda value in the grid shares the same on-disk index.
    ingest_cfg = pt.make_config({**FROZEN_BASE, "ppr_lambda": LAMBDA_GRID[0]})
    print("[setup] ensuring fresh indexes (shared across all lambda values)...", flush=True)
    t_ingest0 = time.perf_counter()
    index_paths = ensure_fresh_indexes(video_ids, ingest_cfg, n_workers=8)
    print(f"[setup] ingest done in {time.perf_counter()-t_ingest0:.0f}s, "
          f"{len(index_paths)}/{len(video_ids)} indexes ready", flush=True)

    index_cache: dict = {}
    for vid, path in index_paths.items():
        index_cache[vid] = iris_ingest.load_index(path)

    per_question_rows = []
    trial_aggregates: dict[tuple, dict] = {}

    for lam in LAMBDA_GRID:
        cfg = pt.make_config({**FROZEN_BASE, "ppr_lambda": lam})

        method_scores = {"A": [], "B": [], "C": [], "D": []}
        n_fallback = 0          # Method C scene-lookup fallback
        n_clip_fallback = 0     # Method D CLIP-anchor fallback
        n_zero_width_b = 0
        n_zero_width_d = 0
        n_scored = 0
        t0 = time.perf_counter()

        for q in questions:
            vid = q["video"]
            if vid not in index_cache:
                continue
            index = index_cache[vid]
            try:
                qe, _ = _call_embed_query(q["question"], cfg)
                frames, _ = _retrieve_with_l1(index, qe, cfg)
            except Exception:
                continue
            if not frames:
                continue

            gold_spans = q["gold_spans"]
            timestamps = [f["timestamp"] for f in frames]

            span_a = predicted_span_from_frames(timestamps)
            span_b = predicted_span_from_frames_clustered(frames, GAP_THRESHOLD_S, TAIL_TRIM_PCT, query_embedding=qe)
            span_c, fallback = predicted_span_from_frames_scene(frames, index.scene_spans, query_embedding=qe)
            span_d, used_clip_anchor = predicted_span_from_frames_peak(
                frames, qe, half_width_s=HALF_WIDTH_S, duration_s=q.get("duration"),
            )

            sa = score_span(span_a, gold_spans)
            sb = score_span(span_b, gold_spans)
            sc = score_span(span_c, gold_spans)
            sd = score_span(span_d, gold_spans)

            zw_b = is_zero_width_span(span_b)
            zw_d = is_zero_width_span(span_d)

            method_scores["A"].append(sa)
            method_scores["B"].append(sb)
            method_scores["C"].append(sc)
            method_scores["D"].append(sd)
            if fallback:
                n_fallback += 1
            if not used_clip_anchor:
                n_clip_fallback += 1
            if zw_b:
                n_zero_width_b += 1
            if zw_d:
                n_zero_width_d += 1
            n_scored += 1

            per_question_rows.append({
                "lambda": lam, "video": vid, "qid": q["qid"],
                "span_a_start": round(span_a[0], 3), "span_a_end": round(span_a[1], 3),
                "span_b_start": round(span_b[0], 3), "span_b_end": round(span_b[1], 3),
                "span_c_start": round(span_c[0], 3), "span_c_end": round(span_c[1], 3),
                "span_d_start": round(span_d[0], 3), "span_d_end": round(span_d[1], 3),
                "method_c_fallback": fallback, "method_d_used_clip_anchor": used_clip_anchor,
                "method_b_zero_width": zw_b, "method_d_zero_width": zw_d,
                "IoP_a": round(sa["IoP"], 4), "IoP_b": round(sb["IoP"], 4),
                "IoP_c": round(sc["IoP"], 4), "IoP_d": round(sd["IoP"], 4),
                "IoU_a": round(sa["IoU"], 4), "IoU_b": round(sb["IoU"], 4),
                "IoU_c": round(sc["IoU"], 4), "IoU_d": round(sd["IoU"], 4),
            })

        dt = time.perf_counter() - t0
        fallback_rate = n_fallback / n_scored if n_scored else 0.0
        clip_fallback_rate = n_clip_fallback / n_scored if n_scored else 0.0
        zero_width_b_rate = n_zero_width_b / n_scored if n_scored else 0.0
        zero_width_d_rate = n_zero_width_d / n_scored if n_scored else 0.0

        for method, scores in method_scores.items():
            iops = [s["IoP"] for s in scores]
            ious = [s["IoU"] for s in scores]
            agg = {
                "mIoP": sum(iops) / len(iops) if iops else 0.0,
                "mIoU": sum(ious) / len(ious) if ious else 0.0,
                "IoP@0.3": sum(1 for s in scores if s["IoP@0.3"]) / len(scores) if scores else 0.0,
                "IoP@0.5": sum(1 for s in scores if s["IoP@0.5"]) / len(scores) if scores else 0.0,
                "IoU@0.3": sum(1 for s in scores if s["IoU@0.3"]) / len(scores) if scores else 0.0,
                "IoU@0.5": sum(1 for s in scores if s["IoU@0.5"]) / len(scores) if scores else 0.0,
                "n_scored": n_scored,
                "method_c_fallback_rate": fallback_rate if method == "C" else "",
                "method_b_zero_width_rate": zero_width_b_rate if method == "B" else "",
                "method_d_zero_width_rate": zero_width_d_rate if method == "D" else "",
                "method_d_clip_fallback_rate": clip_fallback_rate if method == "D" else "",
            }
            trial_aggregates[(lam, method)] = agg
            print(f"  [lambda={lam} method={method}] mIoP={agg['mIoP']:.4f} IoP@0.5={agg['IoP@0.5']:.4f} "
                  f"mIoU={agg['mIoU']:.4f}", flush=True)

        print(f"  [lambda={lam}] method_c_fallback_rate={fallback_rate:.4f} "
              f"method_d_clip_fallback_rate={clip_fallback_rate:.4f} "
              f"method_b_zero_width_rate={zero_width_b_rate:.4f} "
              f"method_d_zero_width_rate={zero_width_d_rate:.4f} "
              f"scoring_wall_s={dt:.0f}", flush=True)
        if n_clip_fallback:
            print(f"  [lambda={lam}] WARNING: method_d_clip_fallback_rate is non-zero "
                  f"({n_clip_fallback}/{n_scored}) -- qe should never be None here; "
                  f"investigate before trusting Method D numbers at this lambda.", flush=True)

    with open(REPO / "tuning" / "lambda_span_method_comparison.csv", "w", newline="") as f:
        fieldnames = [
            "lambda", "method", "mIoP", "IoP@0.3", "IoP@0.5", "mIoU", "IoU@0.3", "IoU@0.5", "n_scored",
            "method_c_fallback_rate", "method_b_zero_width_rate",
            "method_d_zero_width_rate", "method_d_clip_fallback_rate",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for lam in LAMBDA_GRID:
            for method in ["A", "B", "C", "D"]:
                agg = trial_aggregates[(lam, method)]
                row = {kk: round(vv, 5) if isinstance(vv, float) else vv for kk, vv in agg.items()}
                w.writerow({"lambda": lam, "method": method, **row})

    json.dump(per_question_rows[:500], open(REPO / ".part3d_per_question_sample.json", "w"), indent=2)

    print("LAMBDA_SPAN_METHOD_COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
