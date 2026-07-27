"""Isolated peak-memory/time probe for ONE graph-build path.

Run twice, in separate processes (so the two builds' allocations never share
a heap), to get an apples-to-apples peak-RSS comparison:

    python scripts/blockdiag_build_probe.py --mode old --out-json eval_results/blockdiag_probe_old.json
    python scripts/blockdiag_build_probe.py --mode new --out-json eval_results/blockdiag_probe_new.json

"old"  = graph_edge_mode="fully_connected" (today's scene_sparse ingest path:
         materializes N(N-1)/2 pairs, then prunes cross-scene edges).
"new"  = graph_edge_mode="block_diagonal" (direct intra-scene build, no dense
         intermediate).

Loads the cached VIRAT N=4,892 frames (no re-decode) and times/measures ONLY
the iris_ingest._build_graph(...) call -- npz load is excluded from the
timed/measured window, matching the existing scripts/virat_smoke_flat.py
convention.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(r"C:\Users\Siddanth Anil\IRIS")
sys.path.insert(0, str(REPO))

import psutil

import iris.ingest as iris_ingest

CACHE_PATH = REPO / "eval" / "data" / "virat" / "index_cache" / "VIRAT_S_040001_01_000448_001101"


class MemMonitor:
    def __init__(self, pid, interval=0.05):
        self.proc = psutil.Process(pid)
        self.interval = interval
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self.proc.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak_rss = self.proc.memory_info().rss
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["old", "new"], required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    cache_file = Path(str(CACHE_PATH) + ".npz")
    if not cache_file.exists():
        print(f"FATAL: cache not found at {cache_file}", file=sys.stderr)
        sys.exit(1)

    idx = iris_ingest.load_index(CACHE_PATH)  # rebuilds w/ saved config as a side effect; not timed below
    frames = idx.frames
    n_survivors = len(frames)

    cfg = dict(idx.config_snapshot)
    cfg["graph_mode"] = "scene_sparse"
    cfg["graph_edge_mode"] = "fully_connected" if args.mode == "old" else "block_diagonal"

    mon = MemMonitor(psutil.Process().pid)
    mon.start()

    t0 = time.time()
    graph = iris_ingest._build_graph(frames, cfg)
    build_wall_sec = time.time() - t0

    mon.stop()

    result = {
        "mode": args.mode,
        "graph_edge_mode": cfg["graph_edge_mode"],
        "n_survivors": n_survivors,
        "build_wall_sec": build_wall_sec,
        "node_count": graph.graph.number_of_nodes(),
        "edge_count": graph.graph.number_of_edges(),
        "peak_rss_bytes": mon.peak_rss,
        "peak_rss_gb": mon.peak_rss / 1e9,
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
