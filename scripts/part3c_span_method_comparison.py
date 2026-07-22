"""Part 3c: three-way span-construction method comparison (A: min-max,
B: score-weighted clustering, C: real scene-boundary lookup) across the
same l2_retrieve_top_k grid Family 4 used. ONE retrieval call per (K,
question) -- all three methods scored from the same retrieved_frames list.

Does NOT touch tuning/frozen_state.json for retrieval_strategy/ppr_lambda/
ppr_damping/l2_retrieve_top_k -- those stay exactly as already frozen.
This run only decides the span-construction method.

Method C requires scene_spans, which pre-Part-3c cached indexes don't have
(the field didn't exist yet) -- uses a dedicated fresh cache dir so nothing
here is contaminated by stale pre-change .npz files.
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
)

FRESH_CACHE_DIR = REPO / "tuning" / "index_cache_scenespans"
FRESH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

K_GRID = [4, 5, 8, 12, 16]
FROZEN = {"retrieval_strategy": "hybrid", "ppr_lambda": 0.5, "ppr_damping": 0.5}
GAP_THRESHOLD_S = 3.0
TAIL_TRIM_PCT = 20.0


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


def n_clusters_for(retrieved_frames, gap_threshold_s):
    frames_sorted = sorted(retrieved_frames, key=lambda f: f["timestamp"])
    n = 1
    for i in range(1, len(frames_sorted)):
        if frames_sorted[i]["timestamp"] - frames_sorted[i - 1]["timestamp"] > gap_threshold_s:
            n += 1
    return n


def main():
    questions = pt.load_val_tune_questions()
    print(f"[setup] {len(questions)} val_tune questions", flush=True)
    video_ids = sorted({q["video"] for q in questions})

    per_question_rows = []
    peak_in_gold_rows = []
    trial_aggregates: dict[tuple, dict] = {}

    for k in K_GRID:
        cfg = pt.make_config({**FROZEN, "l2_retrieve_top_k": k})
        print(f"[K={k}] ensuring fresh indexes (needed for scene_spans)...", flush=True)
        t_ingest0 = time.perf_counter()
        index_paths = ensure_fresh_indexes(video_ids, cfg, n_workers=8)
        print(f"[K={k}] ingest done in {time.perf_counter()-t_ingest0:.0f}s, "
              f"{len(index_paths)}/{len(video_ids)} indexes ready", flush=True)

        index_cache: dict = {}
        method_scores = {"A": [], "B": [], "C": []}
        n_clusters_list = []
        n_fallback = 0
        n_peak_in_gold = 0
        n_scored = 0
        t0 = time.perf_counter()

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
            if not frames:
                continue

            gold_spans = q["gold_spans"]
            timestamps = [f["timestamp"] for f in frames]

            span_a = predicted_span_from_frames(timestamps)
            span_b = predicted_span_from_frames_clustered(frames, GAP_THRESHOLD_S, TAIL_TRIM_PCT, query_embedding=qe)
            span_c, fallback = predicted_span_from_frames_scene(frames, index.scene_spans, query_embedding=qe)

            sa = score_span(span_a, gold_spans)
            sb = score_span(span_b, gold_spans)
            sc = score_span(span_c, gold_spans)

            n_clust = n_clusters_for(frames, GAP_THRESHOLD_S)
            peak_top_ts = frames[0]["timestamp"]
            peak_in_gold = any(g[0] <= peak_top_ts <= g[1] for g in gold_spans)

            method_scores["A"].append(sa)
            method_scores["B"].append(sb)
            method_scores["C"].append(sc)
            n_clusters_list.append(n_clust)
            if fallback:
                n_fallback += 1
            if peak_in_gold:
                n_peak_in_gold += 1
            n_scored += 1

            per_question_rows.append({
                "k": k, "video": vid, "qid": q["qid"],
                "span_a_start": round(span_a[0], 3), "span_a_end": round(span_a[1], 3),
                "span_b_start": round(span_b[0], 3), "span_b_end": round(span_b[1], 3),
                "span_c_start": round(span_c[0], 3), "span_c_end": round(span_c[1], 3),
                "method_c_fallback": fallback, "n_clusters_b": n_clust,
                "IoP_a": round(sa["IoP"], 4), "IoP_b": round(sb["IoP"], 4), "IoP_c": round(sc["IoP"], 4),
                "IoU_a": round(sa["IoU"], 4), "IoU_b": round(sb["IoU"], 4), "IoU_c": round(sc["IoU"], 4),
            })
            peak_in_gold_rows.append({
                "k": k, "video": vid, "qid": q["qid"], "peak_in_gold": peak_in_gold,
                "top_frame_timestamp": round(peak_top_ts, 3),
            })

        dt = time.perf_counter() - t0
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
            }
            trial_aggregates[(k, method)] = agg
            print(f"  [K={k} method={method}] mIoP={agg['mIoP']:.4f} IoP@0.5={agg['IoP@0.5']:.4f} "
                  f"mIoU={agg['mIoU']:.4f}", flush=True)

        print(f"  [K={k}] peak_in_gold_rate={n_peak_in_gold/n_scored:.4f} "
              f"method_c_fallback_rate={n_fallback/n_scored:.4f} "
              f"mean_n_clusters_b={sum(n_clusters_list)/len(n_clusters_list):.2f} "
              f"scoring_wall_s={dt:.0f}", flush=True)

    with open(REPO / "tuning" / "span_method_comparison.csv", "w", newline="") as f:
        fieldnames = ["k", "method", "mIoP", "IoP@0.3", "IoP@0.5", "mIoU", "IoU@0.3", "IoU@0.5", "n_scored"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for k in K_GRID:
            for method in ["A", "B", "C"]:
                agg = trial_aggregates[(k, method)]
                w.writerow({"k": k, "method": method, **{kk: round(vv, 5) if isinstance(vv, float) else vv for kk, vv in agg.items()}})

    with open(REPO / "tuning" / "span_method_peak_in_gold.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(peak_in_gold_rows[0].keys()))
        w.writeheader()
        for r in peak_in_gold_rows:
            w.writerow(r)

    json.dump({str(k): trial_aggregates[(k, "A")] for k in K_GRID}, open(REPO / ".part3c_method_a_summary.json", "w"), indent=2)
    json.dump(per_question_rows[:500], open(REPO / ".part3c_per_question_sample.json", "w"), indent=2)

    print("SPAN_METHOD_COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
