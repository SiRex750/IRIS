"""Adaptive span-construction sweep (Methods D/E/F) -- val_tune only.

Measurement task, NOT a freeze: tuning/frozen_state.json is never written
here. Retrieval-only: no captioner, no answerer, no LLM call, no ingest
(the val_tune index cache is expected to be 100% warm; any fresh ingest is
a fatal error, not a silently-tolerated slow path). Every cell in the grid
runs on all 2685 val_tune questions -- no cell is ever dropped, sampled, or
interpolated.

Retrieval is run exactly once per question (span_method is not
ingest-relevant and is post-retrieval arithmetic); Method E's CLIP-
similarity profile (survivor cosine + smoothing) is likewise computed once
per question, since smooth_s is fixed a priori (only tau/w_min are swept)
-- the 50-cell grid itself is then cheap per-question arithmetic over
these two cached passes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import iris.ingest as iris_ingest  # noqa: E402
from iris.query import _call_embed_query, _retrieve_with_l1  # noqa: E402
from eval.metrics import (  # noqa: E402
    predicted_span_from_frames_peak,
    compute_similarity_profile,
    span_from_similarity_profile,
    weighted_std_topk,
    predicted_span_from_frames_weighted_spread,
    is_structurally_barred,
    is_zero_width_span,
    assert_no_confirm_videos,
    _pick_peak_by_clip,
)
from part3_tune import (  # noqa: E402
    make_config, load_frozen_state, load_val_tune_questions,
    INDEX_CACHE_DIR, VIDEO_DIR, ingest_config_hash,
)

import importlib.util as _importlib_util  # noqa: E402
_NEXTGQA_METRICS_PATH = REPO / "benchmark_runs/paper_setup_20260720T074844Z_1e431b7/scripts/nextgqa_metrics.py"
_spec = _importlib_util.spec_from_file_location("nextgqa_metrics_canonical", _NEXTGQA_METRICS_PATH)
nextgqa_metrics = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(nextgqa_metrics)

OUT = REPO / "tuning" / "span_sweep"
OUT.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT / "run.log"
_log_f = open(LOG_PATH, "a", buffering=1)


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_f.write(line + "\n")


def iop_iou(pred_span, gold_spans) -> tuple[float, float]:
    gold_tuples = [(g[0], g[1]) for g in gold_spans]
    iop = nextgqa_metrics.iop(pred_span[0], pred_span[1], gold_tuples)
    iou = nextgqa_metrics.iou(pred_span[0], pred_span[1], gold_tuples)
    return iop, iou


def in_any_gold(t: float, gold_spans) -> bool:
    return any(g[0] <= t <= g[1] for g in gold_spans)


DURATION_BUCKETS = [("<2s", 0.0, 2.0), ("2-5s", 2.0, 5.0), ("5-10s", 5.0, 10.0), (">=10s", 10.0, float("inf"))]


def duration_bucket(gold_spans) -> str:
    max_len = max((g[1] - g[0]) for g in gold_spans) if gold_spans else 0.0
    for name, lo, hi in DURATION_BUCKETS:
        if lo <= max_len < hi:
            return name
    return ">=10s"


# ── Stage 0: ingest-cache reuse (must be zero fresh ingests -- checked
# BEFORE any ingest work is attempted, not reactively after the fact) ──────

def check_index_cache(video_ids, cfg):
    """Pure read-only check against the existing val_tune index cache.
    Never ingests. Returns (paths_for_cached_videos, missing_video_ids,
    config_hash)."""
    h = ingest_config_hash(cfg)
    paths, missing = {}, []
    for vid in video_ids:
        p = INDEX_CACHE_DIR / f"{vid}__{h}"
        if p.with_suffix(p.suffix + ".npz").exists():
            paths[vid] = str(p)
        else:
            missing.append(vid)
    return paths, missing, h


# ── Stage 1: retrieval + profile pass (once) ────────────────────────────────

def run_retrieval_and_profile_pass(questions, cfg, index_cache: dict) -> tuple[list[dict], dict]:
    records = []
    n_retrieval_fail = 0
    t0 = time.perf_counter()
    t_retrieval_total = 0.0
    t_profile_total = 0.0

    for q in questions:
        vid = q["video"]
        index = index_cache[vid]

        tr0 = time.perf_counter()
        try:
            query_embedding, _ = _call_embed_query(q["question"], cfg)
            retrieved_frames, _ = _retrieve_with_l1(index, query_embedding, cfg)
        except Exception as exc:  # noqa: BLE001
            log(f"[retrieval FAIL] video={vid} qid={q['qid']}: {type(exc).__name__}: {exc}")
            n_retrieval_fail += 1
            continue
        t_retrieval_total += time.perf_counter() - tr0

        if not retrieved_frames:
            n_retrieval_fail += 1
            continue

        tp0 = time.perf_counter()
        peak = _pick_peak_by_clip(retrieved_frames, query_embedding)
        used_clip_anchor_D = peak is not None
        if peak is None:
            peak = retrieved_frames[0]
        peak_timestamp = float(peak["timestamp"])

        profile = compute_similarity_profile(index.frames, query_embedding, smooth_s=0.5)
        std_topk, weight_fallback_F = weighted_std_topk(retrieved_frames, query_embedding)
        gold_at_4 = any(in_any_gold(f["timestamp"], q["gold_spans"]) for f in retrieved_frames[:4])
        t_profile_total += time.perf_counter() - tp0

        retrieved_min = [{"frame_idx": f["frame_idx"], "timestamp": f["timestamp"]} for f in retrieved_frames]

        records.append({
            "video": vid, "qid": q["qid"], "type": q.get("type"), "gold_spans": q["gold_spans"],
            "duration": float(q["duration"]), "retrieved_min": retrieved_min,
            "peak_timestamp": peak_timestamp, "used_clip_anchor_D": used_clip_anchor_D,
            "profile": profile, "std_topk": std_topk, "weight_fallback_F": weight_fallback_F,
            "gold_at_4": gold_at_4,
        })

    wall_s = time.perf_counter() - t0
    stats = {
        "n_questions": len(questions), "n_scored": len(records), "n_retrieval_fail": n_retrieval_fail,
        "wall_s": wall_s, "retrieval_wall_s": t_retrieval_total, "profile_wall_s": t_profile_total,
        "gold_at_4_rate": (sum(1 for r in records if r["gold_at_4"]) / len(records)) if records else 0.0,
    }
    return records, stats


# ── Cell evaluators ──────────────────────────────────────────────────────

def eval_cell_D(records, half_width_s: float, duration_s_mode: str) -> tuple[list[dict], dict]:
    """duration_s_mode: 'pass' (post-S1, clamp hi to duration) or 'omit' (pre-S1)."""
    rows = []
    for r in records:
        dur = r["duration"] if duration_s_mode == "pass" else None
        lo = max(0.0, r["peak_timestamp"] - half_width_s)
        hi = r["peak_timestamp"] + half_width_s
        if dur is not None:
            hi = min(dur, hi)
        span = (lo, hi)
        rows.append(_score_row(r, span, r["used_clip_anchor_D"], {}))
    return rows, {}


def eval_cell_E(records, tau: float, w_min: float, gap: int = 1, anchor_source: str = "topk",
                 smooth_s_override: dict | None = None) -> tuple[list[dict], dict]:
    rows = []
    n_degenerate_fallback = 0
    for r in records:
        profile = smooth_s_override[r["video"], r["qid"]] if smooth_s_override is not None else r["profile"]
        w_max = min(20.0, 0.5 * r["duration"])
        span, used_clip_anchor, tel = span_from_similarity_profile(
            profile, r["retrieved_min"], r["duration"], tau, w_min,
            gap=gap, anchor_source=anchor_source, w_max=w_max,
        )
        if tel["degenerate_profile_fallback"]:
            n_degenerate_fallback += 1
        rows.append(_score_row(r, span, used_clip_anchor, tel))
    n = len(records)
    return rows, {"degenerate_fallback_rate": n_degenerate_fallback / n if n else 0.0}


def eval_cell_F(records, alpha: float, w_min: float) -> tuple[list[dict], dict]:
    rows = []
    n_weight_fallback = 0
    for r in records:
        w_max = min(20.0, 0.5 * r["duration"])
        width = min(max(alpha * r["std_topk"], w_min), w_max)
        lo = max(0.0, r["peak_timestamp"] - width / 2.0)
        hi = min(r["duration"], r["peak_timestamp"] + width / 2.0)
        if hi < lo:
            hi = lo
        span = (lo, hi)
        if r["weight_fallback_F"]:
            n_weight_fallback += 1
        rows.append(_score_row(r, span, r["used_clip_anchor_D"], {}))
    n = len(records)
    return rows, {"weight_fallback_rate": n_weight_fallback / n if n else 0.0}


def _score_row(r, span, used_clip_anchor, tel) -> dict:
    iop, iou = iop_iou(span, r["gold_spans"])
    width = span[1] - span[0]
    return {
        "video": r["video"], "qid": r["qid"], "type": r["type"],
        "gold_spans": r["gold_spans"], "pred_span_start": span[0], "pred_span_end": span[1],
        "width": width, "iop": iop, "iou": iou, "used_clip_anchor": used_clip_anchor,
        "zero_width": is_zero_width_span(span), "structurally_barred": is_structurally_barred(r["gold_spans"], width),
        "duration_bucket": duration_bucket(r["gold_spans"]), "anchor_ts": tel.get("anchor_timestamp", None),
        "n_survivors_scored": tel.get("n_survivors_scored", None),
        "degenerate_profile_fallback": tel.get("degenerate_profile_fallback", False),
    }


def aggregate_cell(rows: list[dict], extra: dict, gold_at_4_rate: float) -> dict:
    n = len(rows)
    iops = [x["iop"] for x in rows]
    ious = [x["iou"] for x in rows]
    widths = sorted(x["width"] for x in rows)
    zero_width_n = sum(1 for x in rows if x["zero_width"])
    fallback_clip_n = sum(1 for x in rows if not x["used_clip_anchor"])
    structurally_barred_n = sum(1 for x in rows if x["structurally_barred"])

    def pct(vals, p):
        if not vals:
            return 0.0
        k = (len(vals) - 1) * p
        f, c = int(k), min(int(k) + 1, len(vals) - 1)
        if f == c:
            return vals[f]
        return vals[f] + (vals[c] - vals[f]) * (k - f)

    agg = {
        "n_scored": n, "gold_at_4": gold_at_4_rate,
        "mIoP": sum(iops) / n if n else 0.0, "mIoU": sum(ious) / n if n else 0.0,
        "IoP@0.3": sum(1 for x in iops if x >= 0.3) / n if n else 0.0,
        "IoP@0.5": sum(1 for x in iops if x >= 0.5) / n if n else 0.0,
        "IoU@0.3": sum(1 for x in ious if x >= 0.3) / n if n else 0.0,
        "IoU@0.5": sum(1 for x in ious if x >= 0.5) / n if n else 0.0,
        "zero_width_count": zero_width_n, "zero_width_rate": zero_width_n / n if n else 0.0,
        "clip_anchor_fallback_rate": fallback_clip_n / n if n else 0.0,
        "mean_width": statistics.mean(widths) if widths else 0.0,
        "median_width": statistics.median(widths) if widths else 0.0,
        "p10_width": pct(widths, 0.10), "p90_width": pct(widths, 0.90),
        "structurally_barred_count": structurally_barred_n,
    }
    agg.update(extra)

    for bname, _, _ in DURATION_BUCKETS:
        sub = [x for x in rows if x["duration_bucket"] == bname]
        agg[f"mIoP_{bname}"] = (sum(x["iop"] for x in sub) / len(sub)) if sub else None
        agg[f"mIoU_{bname}"] = (sum(x["iou"] for x in sub) / len(sub)) if sub else None
        agg[f"n_{bname}"] = len(sub)

    for t in ["CW", "CH", "TN", "TC", "TP"]:
        sub = [x for x in rows if x["type"] == t]
        agg[f"mIoP_{t}"] = (sum(x["iop"] for x in sub) / len(sub)) if sub else None
        agg[f"mIoU_{t}"] = (sum(x["iou"] for x in sub) / len(sub)) if sub else None
        agg[f"n_{t}"] = len(sub)

    return agg


# ── Grid definitions ─────────────────────────────────────────────────────

D_HALF_WIDTHS = [0.5, 0.8, 1.1, 1.5, 2.2, 3.0, 4.0, 6.0]
E_TAUS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
E_WMINS = [0.6, 1.0, 1.4, 1.8, 2.2]
F_ALPHAS = [1.0, 1.5, 2.0, 3.0]
F_WMINS = [0.6, 1.4, 2.2]

GRID_FIELDNAMES = [
    "cell_id", "method", "params", "n_scored", "gold_at_4", "mIoP", "mIoU",
    "IoP@0.3", "IoP@0.5", "IoU@0.3", "IoU@0.5",
    "zero_width_count", "zero_width_rate", "clip_anchor_fallback_rate", "degenerate_fallback_rate",
    "weight_fallback_rate", "mean_width", "median_width", "p10_width", "p90_width",
    "structurally_barred_count",
    "mIoP_<2s", "mIoU_<2s", "n_<2s", "mIoP_2-5s", "mIoU_2-5s", "n_2-5s",
    "mIoP_5-10s", "mIoU_5-10s", "n_5-10s", "mIoP_>=10s", "mIoU_>=10s", "n_>=10s",
    "mIoP_CW", "mIoU_CW", "n_CW", "mIoP_CH", "mIoU_CH", "n_CH",
    "mIoP_TN", "mIoU_TN", "n_TN", "mIoP_TC", "mIoU_TC", "n_TC",
    "mIoP_TP", "mIoU_TP", "n_TP", "elapsed_ms",
]


def write_grid_row(csv_writer, csv_f, cell_id, method, params, agg, elapsed_ms):
    row = {"cell_id": cell_id, "method": method, "params": json.dumps(params), "elapsed_ms": round(elapsed_ms, 1)}
    for k in GRID_FIELDNAMES:
        if k in agg:
            row[k] = agg[k]
    row.setdefault("degenerate_fallback_rate", "")
    row.setdefault("weight_fallback_rate", "")
    csv_writer.writerow({k: row.get(k, "") for k in GRID_FIELDNAMES})
    csv_f.flush()
    headline = (f"mIoP={agg['mIoP']:.5f} mIoU={agg['mIoU']:.5f} IoP@0.5={agg['IoP@0.5']:.5f} "
                f"zero_width={agg['zero_width_rate']:.4f}")
    log(f"[cell {cell_id}] {method} {params} n={agg['n_scored']} {headline} elapsed_ms={elapsed_ms:.1f}")


def main() -> None:
    log("=== span_sweep.py start ===")
    frozen = load_frozen_state()["frozen"]
    log(f"[setup] frozen config read live: {frozen}")

    split = json.loads((REPO / "split_manifest.json").read_text())
    confirm_videos = set(split["confirm_videos"])
    tune_videos = set(split["tune_videos"])

    questions = load_val_tune_questions()
    video_ids = sorted({q["video"] for q in questions})
    log(f"[setup] val_tune questions loaded: {len(questions)} across {len(video_ids)} videos")

    assert_no_confirm_videos(video_ids, confirm_videos)
    assert_no_confirm_videos([q["video"] for q in questions], confirm_videos)
    assert set(video_ids) <= tune_videos, "video_ids must be a subset of split_manifest.json tune_videos"
    log(f"[guard] val_confirm exclusion assertion passed -- 0 confirm videos among {len(video_ids)} tune videos")

    for q in questions:
        if not q.get("duration"):
            raise AssertionError(f"missing duration for video={q['video']} (gsub_val.json)")
    log(f"[guard] duration_s available (gsub_val.json per-video 'duration') for all {len(video_ids)} videos")

    overrides = {k: frozen[k] for k in (
        "retrieval_strategy", "ppr_lambda", "ppr_damping", "l2_retrieve_top_k",
        "peak_distance", "peak_prominence", "packet_size_weight", "motion_weight",
        "luma_entropy_weight", "persistence_threshold", "max_prominence",
    )}
    cfg = make_config(overrides)
    log(f"[setup] config: query_reformulation_mode={cfg.query_reformulation_mode} "
        f"temporal_traversal_mode={cfg.temporal_traversal_mode} graph_mode={cfg.graph_mode}")

    paths, missing, config_hash = check_index_cache(video_ids, cfg)
    log(f"[cache] config_hash={config_hash} cache_dir={INDEX_CACHE_DIR} "
        f"cached={len(paths)}/{len(video_ids)} missing={len(missing)}")
    if missing:
        stop_report = {
            "status": "STOPPED -- fresh ingest would be required, which this retrieval-only task forbids",
            "config_hash": config_hash, "missing_videos": missing, "n_missing": len(missing),
            "n_total_videos": len(video_ids),
        }
        (OUT / "reproduction_gate.json").write_text(json.dumps(stop_report, indent=2))
        log(f"[FATAL] {len(missing)} videos missing from index cache under hash {config_hash} -- "
            f"stopping without ingesting. See reproduction_gate.json.")
        sys.exit(1)

    index_cache = {}
    for vid in video_ids:
        idx = iris_ingest.load_index(paths[vid])
        assert idx.frames, f"video {vid} loaded with zero survivor frames"
        index_cache[vid] = idx
    log(f"[setup] loaded {len(index_cache)} indexes from cache, zero fresh ingests confirmed")

    records, pass_stats = run_retrieval_and_profile_pass(questions, cfg, index_cache)
    log(f"[pass1] retrieval+profile pass complete: {pass_stats}")
    assert_no_confirm_videos([r["video"] for r in records], confirm_videos)

    # ── Reproduction gate ────────────────────────────────────────────────
    log("=== reproduction gate ===")
    rows_omit, _ = eval_cell_D(records, half_width_s=2.2, duration_s_mode="omit")
    agg_omit = aggregate_cell(rows_omit, {}, pass_stats["gold_at_4_rate"])
    rows_pass, _ = eval_cell_D(records, half_width_s=2.2, duration_s_mode="pass")
    agg_pass = aggregate_cell(rows_pass, {}, pass_stats["gold_at_4_rate"])

    expected = {
        "n": 2685, "gold_at_4": 0.5303538175046555, "mIoP": 0.29781971,
        "mIoU": 0.16086580, "IoP@0.5": 0.30093,
    }
    tol = {"n": 0, "gold_at_4": 1e-6, "mIoP": 1e-6, "mIoU": 1e-6, "IoP@0.5": 5e-5}
    actual = {"n": agg_omit["n_scored"], "gold_at_4": agg_omit["gold_at_4"],
              "mIoP": agg_omit["mIoP"], "mIoU": agg_omit["mIoU"], "IoP@0.5": agg_omit["IoP@0.5"]}
    mismatches = {}
    for k in expected:
        delta = abs(actual[k] - expected[k])
        if delta > tol[k]:
            mismatches[k] = {"expected": expected[k], "actual": actual[k], "delta": delta, "tolerance": tol[k]}

    s1_delta = {
        "mIoP_delta": agg_pass["mIoP"] - agg_omit["mIoP"], "IoP@0.5_delta": agg_pass["IoP@0.5"] - agg_omit["IoP@0.5"],
        "mIoU_delta": agg_pass["mIoU"] - agg_omit["mIoU"], "IoU@0.5_delta": agg_pass["IoU@0.5"] - agg_omit["IoU@0.5"],
    }
    n_clamp_fired = sum(1 for a, b in zip(rows_omit, rows_pass) if a["pred_span_end"] != b["pred_span_end"])

    gate_result = {
        "expected": expected, "actual": actual, "mismatches": mismatches,
        "reproduction_gate_passed": len(mismatches) == 0,
        "duration_s_omitted_full_agg": agg_omit, "duration_s_passed_full_agg": agg_pass,
        "s1_isolated_effect_val_tune": s1_delta, "n_spans_where_s1_clamp_changed_hi": n_clamp_fired,
    }
    (OUT / "reproduction_gate.json").write_text(json.dumps(gate_result, indent=2, default=str))
    log(f"[gate] reproduction_gate_passed={gate_result['reproduction_gate_passed']} mismatches={mismatches}")
    log(f"[gate] S1 isolated effect on val_tune: {s1_delta}, clamp fired on {n_clamp_fired}/{len(records)}")

    if not gate_result["reproduction_gate_passed"]:
        log("[FATAL] reproduction gate FAILED -- stopping before the sweep. "
            "No per-question historical baseline file was available to diff against; "
            "aggregate mismatch detail is in reproduction_gate.json.")
        sys.exit(1)

    # ── Full 50-cell grid ────────────────────────────────────────────────
    log("=== full 50-cell grid sweep ===")
    csv_path = OUT / "grid_results.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=GRID_FIELDNAMES)
    writer.writeheader()

    all_cells = {}  # cell_id -> (rows, agg, params, method)
    cell_id = 0
    t_sweep_start = time.perf_counter()

    for hw in D_HALF_WIDTHS:
        cell_id += 1
        t0 = time.perf_counter()
        rows, extra = eval_cell_D(records, half_width_s=hw, duration_s_mode="pass")
        agg = aggregate_cell(rows, extra, pass_stats["gold_at_4_rate"])
        elapsed = (time.perf_counter() - t0) * 1000
        params = {"half_width_s": hw}
        write_grid_row(writer, csv_f, cell_id, "D", params, agg, elapsed)
        all_cells[cell_id] = (rows, agg, params, "D")

    for tau in E_TAUS:
        for w_min in E_WMINS:
            cell_id += 1
            t0 = time.perf_counter()
            rows, extra = eval_cell_E(records, tau=tau, w_min=w_min, gap=1, anchor_source="topk")
            agg = aggregate_cell(rows, extra, pass_stats["gold_at_4_rate"])
            elapsed = (time.perf_counter() - t0) * 1000
            params = {"tau": tau, "w_min": w_min, "smooth_s": 0.5, "gap": 1, "anchor_source": "topk"}
            write_grid_row(writer, csv_f, cell_id, "E", params, agg, elapsed)
            all_cells[cell_id] = (rows, agg, params, "E")

    for alpha in F_ALPHAS:
        for w_min in F_WMINS:
            cell_id += 1
            t0 = time.perf_counter()
            rows, extra = eval_cell_F(records, alpha=alpha, w_min=w_min)
            agg = aggregate_cell(rows, extra, pass_stats["gold_at_4_rate"])
            elapsed = (time.perf_counter() - t0) * 1000
            params = {"alpha": alpha, "w_min": w_min}
            write_grid_row(writer, csv_f, cell_id, "F", params, agg, elapsed)
            all_cells[cell_id] = (rows, agg, params, "F")

    csv_f.close()
    sweep_wall_s = time.perf_counter() - t_sweep_start
    log(f"[sweep] all {cell_id} cells complete, wall_s={sweep_wall_s:.2f}")
    log(f"[timing] retrieval_pass_wall_s={pass_stats['wall_s']:.2f} "
        f"(retrieval={pass_stats['retrieval_wall_s']:.2f}, profile={pass_stats['profile_wall_s']:.2f}), "
        f"sweep_wall_s={sweep_wall_s:.2f}")

    # ── Selection: best per method by mIoP (primary), IoP@0.5 tie-break ────
    def best_of(method):
        cands = [(cid, agg, params) for cid, (rows, agg, params, m) in all_cells.items() if m == method]
        return max(cands, key=lambda x: (x[1]["mIoP"], x[1]["IoP@0.5"]))

    best_D = best_of("D")
    best_E = best_of("E")
    best_F = best_of("F")
    log(f"[selection] best D: cell {best_D[0]} {best_D[2]} mIoP={best_D[1]['mIoP']:.5f}")
    log(f"[selection] best E: cell {best_E[0]} {best_E[2]} mIoP={best_E[1]['mIoP']:.5f}")
    log(f"[selection] best F: cell {best_F[0]} {best_F[2]} mIoP={best_F[1]['mIoP']:.5f}")

    frozen_D_cell_id = next(cid for cid, (r, a, p, m) in all_cells.items() if m == "D" and p["half_width_s"] == 2.2)

    # ── Per-question CSVs for the 4 designated cells ────────────────────
    per_q_dir = OUT / "per_question"
    per_q_dir.mkdir(exist_ok=True)
    designated = {
        f"D_hw2.2_cell{frozen_D_cell_id}": all_cells[frozen_D_cell_id][0],
        f"D_best_cell{best_D[0]}": all_cells[best_D[0]][0],
        f"E_best_cell{best_E[0]}": all_cells[best_E[0]][0],
        f"F_best_cell{best_F[0]}": all_cells[best_F[0]][0],
    }
    per_q_fields = ["video", "qid", "type", "gold_spans", "pred_span_start", "pred_span_end", "width",
                     "iop", "iou", "anchor_ts", "n_survivors_scored", "degenerate_profile_fallback"]
    for name, rows in designated.items():
        with open(per_q_dir / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_q_fields)
            w.writeheader()
            for r in rows:
                row = {k: r.get(k, "") for k in per_q_fields}
                row["gold_spans"] = json.dumps(row["gold_spans"])
                w.writerow(row)
    log(f"[deliverable] wrote per-question CSVs for {list(designated.keys())}")

    summary = {
        "frozen_D_cell_id": frozen_D_cell_id, "best_D": {"cell_id": best_D[0], "params": best_D[2], "agg": best_D[1]},
        "best_E": {"cell_id": best_E[0], "params": best_E[2], "agg": best_E[1]},
        "best_F": {"cell_id": best_F[0], "params": best_F[2], "agg": best_F[1]},
        "pass_stats": pass_stats, "sweep_wall_s": sweep_wall_s,
    }
    (OUT / "sweep_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("=== span_sweep.py main grid complete ===")


if __name__ == "__main__":
    main()
