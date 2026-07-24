"""Latency A/B: flat vs scene_sparse (eval_results/P_latency_prereg.md). VAL only.

Measures retrieval-path latency only (query embedding + _build_retrieved).
No answer generation, no span prediction -- those are not part of the timed
path and do not affect it.

CPU only, 8 threads (9800X3D, 8 physical cores). 3 warm-up queries per video
(first question repeated, discarded) before timed reps. Each question is run
5x; MEDIAN is reported. Index load time is measured separately, once per
video per arm, and excluded from per-query numbers.

VERIFY: python scripts/latency_ab.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import contextlib
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:
    import torch
    torch.set_num_threads(8)
except Exception:
    torch = None

import iris.scene_retrieval as scene_retrieval
from iris import _perf
from iris.iris_config import IRISConfig
from iris.query import _embed_query, _build_retrieved
from scripts.scene_2x2x2_sweep import _load_val_rows, FLAT_CACHE, SSPARSE_CACHE

N_WARMUP = 3
N_REPS = 5
TOP_K = 8

BASE = dict(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=TOP_K,
    scene_diag=True,
)

TIMER_KEYS = [
    "query_embed_s", "total_retrieval_s", "flat_ppr_s",
    "scene_centroid_rank_s", "subgraph_induction_s", "cross_scene_edges_s", "scene_ppr_s",
]
COUNT_KEYS = ["ppr_nodes", "ppr_edges"]


_DEVNULL = open(os.devnull, "w")


def _timed_query(index, question: str, cfg) -> tuple[dict, dict]:
    # retrieve_scene_sparse prints per-query diagnostics (scene_diag=True) from
    # inside the timed region (before it returns) -- flat has no equivalent
    # per-query stdout write. Suppressing stdout here (real fd, not StringIO)
    # keeps the two arms' timed regions doing comparable I/O, not none-vs-print.
    _perf.reset()
    with contextlib.redirect_stdout(_DEVNULL):
        emb = _embed_query(question, cfg)
        _build_retrieved(index, emb, cfg)
    timings = {k: _perf.TIMINGS.get(k) for k in TIMER_KEYS}
    counts = {k: _perf.COUNTS.get(k) for k in COUNT_KEYS}
    return timings, counts


def _median_or_none(vals: list[float | None]) -> float | None:
    present = [v for v in vals if v is not None]
    return statistics.median(present) if present else None


def run_arm(graph_mode: str, grounded_rows: list[dict]) -> tuple[list[dict], float]:
    cache_dir = FLAT_CACHE if graph_mode == "flat" else SSPARSE_CACHE
    cfg = IRISConfig(**BASE, graph_mode=graph_mode)

    from eval.grounding_scorer import load_indexes
    _t_load0 = time.perf_counter()
    loaded = load_indexes(grounded_rows, cache_dir)
    index_load_s = time.perf_counter() - _t_load0

    by_video: dict[str, list[dict]] = {}
    for row in grounded_rows:
        by_video.setdefault(row["video"], []).append(row)

    per_query_records: list[dict] = []

    for vid, rows in by_video.items():
        index = loaded.get(vid)
        if index is None:
            continue

        warmup_q = rows[0]["question"]
        for _ in range(N_WARMUP):
            _timed_query(index, warmup_q, cfg)

        for row in rows:
            qid = str(row["qid"])
            rep_timings: list[dict] = []
            rep_counts: list[dict] = []
            branch = "n/a"
            for _ in range(N_REPS):
                scene_retrieval.SCENE_DIAG_RECORDS.clear()
                timings, counts = _timed_query(index, row["question"], cfg)
                rep_timings.append(timings)
                rep_counts.append(counts)
                if graph_mode == "scene_sparse" and scene_retrieval.SCENE_DIAG_RECORDS:
                    branch = scene_retrieval.SCENE_DIAG_RECORDS[-1]["branch"]

            median_timings = {k: _median_or_none([t[k] for t in rep_timings]) for k in TIMER_KEYS}
            median_counts = {k: _median_or_none([c[k] for c in rep_counts]) for k in COUNT_KEYS}

            per_query_records.append({
                "video": vid,
                "qid": qid,
                "graph_mode": graph_mode,
                "branch": branch,
                "n_frames": index.frames_processed,
                **median_timings,
                **median_counts,
                "reps": rep_timings,
            })

    return per_query_records, index_load_s


def _bucket_frames(n: int) -> str:
    if n < 100:
        return "<100"
    if n < 300:
        return "100-299"
    if n < 600:
        return "300-599"
    if n < 1000:
        return "600-999"
    return "1000+"


def main() -> None:
    grounded_rows = _load_val_rows()
    n_videos = len(set(r["video"] for r in grounded_rows))
    print(f"[DATA] VAL grounded questions: {len(grounded_rows)} across {n_videos} videos")
    assert len(grounded_rows) > 0

    all_records: list[dict] = []
    index_load_times: dict[str, float] = {}

    for graph_mode in ("flat", "scene_sparse"):
        print(f"\n[RUN] graph_mode={graph_mode} ...", flush=True)
        records, index_load_s = run_arm(graph_mode, grounded_rows)
        all_records.extend(records)
        index_load_times[graph_mode] = index_load_s
        print(f"[RUN] graph_mode={graph_mode} done: {len(records)} questions, "
              f"index_load_s={index_load_s:.4f}", flush=True)

    logs_dir = REPO / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / "latency_ab.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "n_warmup": N_WARMUP,
            "n_reps": N_REPS,
            "top_k": TOP_K,
            "index_load_s": index_load_times,
            "per_query": all_records,
        }, fh, indent=2)
    print(f"\n[LOG] per-query dump written to {out_path}")

    # ── REPORT (facts requested by pre-registration §4; no interpretation) ──
    def pooled(mode, key, branch=None):
        vals = [r[key] for r in all_records
                if r["graph_mode"] == mode and r[key] is not None
                and (branch is None or r["branch"] == branch)]
        return _median_or_none(vals)

    print("\n=== MEDIAN total_retrieval_s ===")
    for mode in ("flat", "scene_sparse"):
        print(f"  {mode:<13} pooled={pooled(mode, 'total_retrieval_s')}")
        if mode == "scene_sparse":
            print(f"    shortcut={pooled(mode, 'total_retrieval_s', 'shortcut')}")
            print(f"    descend ={pooled(mode, 'total_retrieval_s', 'descend')}")

    print("\n=== COMPONENT BREAKDOWN (median, seconds) ===")
    for mode in ("flat", "scene_sparse"):
        print(f"  {mode}:")
        for key in TIMER_KEYS:
            print(f"    {key:<22} {pooled(mode, key)}")

    print("\n=== MEDIAN nodes/edges into PPR ===")
    for mode in ("flat", "scene_sparse"):
        print(f"  {mode:<13} nodes={pooled(mode, 'ppr_nodes')} edges={pooled(mode, 'ppr_edges')}")

    print("\n=== LATENCY vs FRAME COUNT (bucketed, median total_retrieval_s) ===")
    buckets = ["<100", "100-299", "300-599", "600-999", "1000+"]
    for mode in ("flat", "scene_sparse"):
        print(f"  {mode}:")
        for b in buckets:
            vals = [r["total_retrieval_s"] for r in all_records
                    if r["graph_mode"] == mode and _bucket_frames(r["n_frames"]) == b
                    and r["total_retrieval_s"] is not None]
            med = _median_or_none(vals)
            print(f"    {b:<10} n={len(vals):<4} median={med}")

    print("\n=== INDEX LOAD TIME (per arm, excluded from per-query numbers) ===")
    for mode, t in index_load_times.items():
        print(f"  {mode:<13} {t:.4f}s")

    print("\n=== QUERY EMBEDDING TIME (control -- should match across arms) ===")
    for mode in ("flat", "scene_sparse"):
        print(f"  {mode:<13} median_query_embed_s={pooled(mode, 'query_embed_s')}")

    print("\nSTOP: per pre-registration, no shortlist_width/tau tuning follows this report.")


if __name__ == "__main__":
    main()
