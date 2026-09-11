"""Looped, machine-state-aware build-cost probe for ONE arm, ONE round.

Fixes two problems in the single-shot probe (scripts/blockdiag_build_probe.py):

  1. The block_diagonal ("new") arm times ~0.12s per build -- close to the
     timer resolution floor. Here we run N consecutive builds *in the same
     process* and time the whole loop, so the timed quantity is far above
     timer noise; report both the per-iteration times (loop split into
     N separately-timed builds, not just total/N) and the total.

  2. The fully_connected ("old") arm's wall time has read differently across
     separate CLI sessions (69.47s / 61.39s / 48.57s) while being <1% tight
     *within* a session -- consistent with machine-state drift between
     sessions rather than measurement noise. We record a machine-state
     snapshot (logical CPU count, a short cpu_percent sample, available RAM)
     immediately before each arm's loop starts, and the driver runs arms in
     interleaved (old, new, old, new, ...) order across rounds so any drift
     hits both arms symmetrically instead of biasing one.

Usage (one process per arm per round -- the driver interleaves the launches):

    python scripts/build_cost_final_probe.py --mode old --n 3  --round 1 --out-json eval_results/_build_cost_final_tmp/old_r1.json
    python scripts/build_cost_final_probe.py --mode new --n 50 --round 1 --out-json eval_results/_build_cost_final_tmp/new_r1.json

Loads the cached VIRAT N=4,892 frames (no re-decode); load_index's rebuild
and the machine-state snapshot are excluded from the timed loop, matching
the existing probe's convention of excluding npz load from the timed window.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\Siddanth Anil\IRIS")
sys.path.insert(0, str(REPO))

import psutil

import iris.ingest as iris_ingest

CACHE_PATH = REPO / "eval" / "data" / "virat" / "index_cache" / "VIRAT_S_040001_01_000448_001101"
EXPECTED_EDGE_COUNT = 23571


def machine_snapshot():
    snap = {
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "cpu_percent_sample": psutil.cpu_percent(interval=0.2),  # blocks 0.2s, sampled just before the loop
    }
    try:
        snap["getloadavg_1_5_15"] = list(psutil.getloadavg())
    except (AttributeError, OSError):
        snap["getloadavg_1_5_15"] = None  # not available on Windows
    vm = psutil.virtual_memory()
    snap["available_ram_bytes"] = vm.available
    snap["available_ram_gb"] = vm.available / 1e9
    snap["total_ram_bytes"] = vm.total
    snap["total_ram_gb"] = vm.total / 1e9
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["old", "new"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    cache_file = Path(str(CACHE_PATH) + ".npz")
    if not cache_file.exists():
        print(f"FATAL: cache not found at {cache_file}", file=sys.stderr)
        sys.exit(1)

    idx = iris_ingest.load_index(CACHE_PATH)  # rebuilds w/ saved config as a side effect; not timed
    frames = idx.frames
    n_survivors = len(frames)

    cfg = dict(idx.config_snapshot)
    cfg["graph_mode"] = "scene_sparse"
    cfg["graph_edge_mode"] = "fully_connected" if args.mode == "old" else "block_diagonal"

    snap = machine_snapshot()  # taken immediately before the timed loop starts

    per_iter_sec = []
    edge_counts = []
    t_loop0 = time.time()
    for i in range(args.n):
        t0 = time.time()
        graph = iris_ingest._build_graph(frames, cfg)
        per_iter_sec.append(time.time() - t0)
        ec = graph.graph.number_of_edges()
        edge_counts.append(ec)
        assert ec == EXPECTED_EDGE_COUNT, (
            f"{args.mode} round {args.round} iter {i}: edge_count={ec} != {EXPECTED_EDGE_COUNT}"
        )
        assert graph.graph.number_of_nodes() == n_survivors
    total_loop_sec = time.time() - t_loop0

    per_iter_sorted = sorted(per_iter_sec)
    mid = len(per_iter_sorted) // 2
    if len(per_iter_sorted) % 2:
        median = per_iter_sorted[mid]
    else:
        median = (per_iter_sorted[mid - 1] + per_iter_sorted[mid]) / 2

    result = {
        "mode": args.mode,
        "graph_edge_mode": cfg["graph_edge_mode"],
        "round": args.round,
        "n": args.n,
        "n_survivors": n_survivors,
        "machine_snapshot_before_loop": snap,
        "per_iter_sec": per_iter_sec,
        "total_loop_sec": total_loop_sec,
        "mean_per_iter_sec": total_loop_sec / args.n,
        "median_per_iter_sec": median,
        "min_per_iter_sec": min(per_iter_sec),
        "max_per_iter_sec": max(per_iter_sec),
        "edge_counts_all_iters": edge_counts,
        "edge_count_guard_passed": all(ec == EXPECTED_EDGE_COUNT for ec in edge_counts),
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
