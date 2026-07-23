"""Part 3e: joint l2_retrieve_top_k x ppr_lambda x span-method comparison.

Extends part3d_lambda_span_method_comparison.py (which held K frozen at 4)
with the K axis. Family 4's original K=4 decision (part3_tune.py) predates
Methods B/C/D and predates the CLIP-anchor fix (commit 47ebce5) and the
_build_retrieved scene_id fix (this same PR) -- Method C's numbers under
the old K sweep were never trustworthy, so K deserves re-checking under
the same corrected conditions lambda was re-checked under in part3d.

Three axes:
  - l2_retrieve_top_k in [4, 5, 8, 12, 16]  (Family 4 / part3c's grid)
  - ppr_lambda in [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]  (Family 2's grid)
  - span method in {A, B, C, D}
140 cells total. retrieval_strategy=hybrid, ppr_damping=0.5 held fixed,
matching tuning/frozen_state.json.

Ingestion structure: K IS in part3_tune.INGEST_RELEVANT_KEYS (it changes
the retrieved pool / graph construction), so a fresh ingest config-hash is
needed per K value -- but NOT per lambda (lambda is retrieval-time only).
This script therefore ingests once per K value (5 ingests total across the
whole grid, not 35), then loops over all 7 lambda values against that
K's cached indexes before moving to the next K. Part 3c already
populated tuning/index_cache_scenespans for K=[4,5,8,12,16] at these same
frozen settings -- ensure_fresh_indexes checks the cache-hash first and
only ingests what's genuinely missing.

Diagnostic only: does NOT write to tuning/frozen_state.json.

Use --ks / --lambdas to restrict either grid (e.g. a cheap
K=4,lambda=0.50-only sanity pass). CSV rows are written and flushed after
each (K, lambda)'s four methods finish, not only at the very end.
"""
from __future__ import annotations

import argparse
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

K_GRID = [4, 5, 8, 12, 16]
LAMBDA_GRID = [0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 1.00]

_FROZEN_STATE = json.loads((REPO / "tuning" / "frozen_state.json").read_text())["frozen"]
FROZEN_BASE = {
    "retrieval_strategy": _FROZEN_STATE["retrieval_strategy"],
    "ppr_damping": _FROZEN_STATE["ppr_damping"],
}

GAP_THRESHOLD_S = 3.0
TAIL_TRIM_PCT = 20.0
HALF_WIDTH_S = 2.2

CSV_FIELDNAMES = [
    "K", "lambda", "method", "mIoP", "IoP@0.3", "IoP@0.5", "mIoU", "IoU@0.3", "IoU@0.5", "n_scored",
    "method_c_fallback_rate", "method_b_zero_width_rate",
    "method_d_zero_width_rate", "method_d_clip_fallback_rate",
]


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
    print(f"    [ingest] config-hash {h}: {len(todo)}/{len(video_ids)} need fresh ingest", flush=True)

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


def score_cell(K: int, lam: float, cfg, index_cache: dict, questions: list[dict]) -> dict:
    method_scores = {"A": [], "B": [], "C": [], "D": []}
    n_fallback = 0
    n_clip_fallback = 0
    n_zero_width_b = 0
    n_zero_width_d = 0
    n_scored = 0

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

    fallback_rate = n_fallback / n_scored if n_scored else 0.0
    clip_fallback_rate = n_clip_fallback / n_scored if n_scored else 0.0
    zero_width_b_rate = n_zero_width_b / n_scored if n_scored else 0.0
    zero_width_d_rate = n_zero_width_d / n_scored if n_scored else 0.0

    aggs = {}
    for method, scores in method_scores.items():
        iops = [s["IoP"] for s in scores]
        ious = [s["IoU"] for s in scores]
        aggs[method] = {
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
    return {
        "aggs": aggs, "fallback_rate": fallback_rate, "clip_fallback_rate": clip_fallback_rate,
        "zero_width_b_rate": zero_width_b_rate, "zero_width_d_rate": zero_width_d_rate,
        "n_scored": n_scored, "n_clip_fallback": n_clip_fallback,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ks", type=str, default=None,
                    help="Comma-separated K subset (default: full [4,5,8,12,16] grid). "
                         "Use e.g. --ks 4 for a cheap sanity pass.")
    p.add_argument("--lambdas", type=str, default=None,
                    help="Comma-separated lambda subset (default: full 7-value grid). "
                         "Use e.g. --lambdas 0.5 for a cheap sanity pass.")
    return p.parse_args()


def main():
    args = parse_args()
    k_grid = K_GRID if not args.ks else [int(x) for x in args.ks.split(",")]
    lambda_grid = LAMBDA_GRID if not args.lambdas else [float(x) for x in args.lambdas.split(",")]

    questions = pt.load_val_tune_questions()
    print(f"[setup] {len(questions)} val_tune questions", flush=True)
    print(f"[setup] K grid: {k_grid}  lambda grid: {lambda_grid}  "
          f"({len(k_grid) * len(lambda_grid) * 4} cells)", flush=True)
    video_ids = sorted({q["video"] for q in questions})

    csv_path = REPO / "tuning" / "lambda_k_span_method_comparison.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=CSV_FIELDNAMES)
    csv_w.writeheader()
    csv_f.flush()

    for K in k_grid:
        # Ingest once per K: K is INGEST_RELEVANT, lambda is not -- every
        # lambda value at this K shares the same on-disk index.
        ingest_cfg = pt.make_config({**FROZEN_BASE, "l2_retrieve_top_k": K, "ppr_lambda": lambda_grid[0]})
        print(f"[setup K={K}] ensuring fresh indexes (shared across all lambda values at this K)...", flush=True)
        t_ingest0 = time.perf_counter()
        index_paths = ensure_fresh_indexes(video_ids, ingest_cfg, n_workers=8)
        print(f"[setup K={K}] ingest done in {time.perf_counter()-t_ingest0:.0f}s, "
              f"{len(index_paths)}/{len(video_ids)} indexes ready", flush=True)

        index_cache: dict = {}
        for vid, path in index_paths.items():
            index_cache[vid] = iris_ingest.load_index(path)

        for lam in lambda_grid:
            cfg = pt.make_config({**FROZEN_BASE, "l2_retrieve_top_k": K, "ppr_lambda": lam})
            t0 = time.perf_counter()
            result = score_cell(K, lam, cfg, index_cache, questions)
            dt = time.perf_counter() - t0

            for method in ["A", "B", "C", "D"]:
                agg = result["aggs"][method]
                print(f"  [K={K} lambda={lam} method={method}] mIoP={agg['mIoP']:.4f} "
                      f"IoP@0.5={agg['IoP@0.5']:.4f} mIoU={agg['mIoU']:.4f}", flush=True)

            print(f"  [K={K} lambda={lam}] method_c_fallback_rate={result['fallback_rate']:.4f} "
                  f"method_d_clip_fallback_rate={result['clip_fallback_rate']:.4f} "
                  f"method_b_zero_width_rate={result['zero_width_b_rate']:.4f} "
                  f"method_d_zero_width_rate={result['zero_width_d_rate']:.4f} "
                  f"scoring_wall_s={dt:.0f}", flush=True)
            if result["n_clip_fallback"]:
                print(f"  [K={K} lambda={lam}] WARNING: method_d_clip_fallback_rate is non-zero "
                      f"({result['n_clip_fallback']}/{result['n_scored']}) -- qe should never be "
                      f"None here; investigate before trusting Method D numbers at this cell.", flush=True)

            # Checkpoint: flush this (K, lambda)'s four (K, lambda, method) rows
            # to disk immediately so a crash later in the sweep doesn't lose them.
            for method in ["A", "B", "C", "D"]:
                agg = result["aggs"][method]
                row = {kk: round(vv, 5) if isinstance(vv, float) else vv for kk, vv in agg.items()}
                csv_w.writerow({"K": K, "lambda": lam, "method": method, **row})
            csv_f.flush()

        # Drop this K's index cache before loading the next K's -- keeps
        # peak memory bounded to one K's worth of indexes at a time.
        index_cache.clear()

    csv_f.close()
    print("LAMBDA_K_SPAN_METHOD_COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
