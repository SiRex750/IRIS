"""VIRAT smoke test, arm 2: build the FLAT (fully-connected) graph.

Reuses the frames + embeddings already ingested and cached by the
scene_sparse arm (no re-decode) -- loads eval/data/virat/index_cache/
VIRAT_S_040001_01_000448_001101.npz, then calls iris.ingest._build_graph
directly with graph_mode="flat" (node_groups=None -> no cross-scene
pruning -> true N*(N-1)/2 fully-connected build). Separate process from
the scene_sparse arm so a crash here cannot take down that result.

Writes nothing to any cache -- this is a graph-build timing/memory probe
only, run against data that is already safely cached.
"""
from __future__ import annotations

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
OUT_JSON = Path(r"C:\Users\SIDDAN~1\AppData\Local\Temp\claude\C--Users-Siddanth-Anil-IRIS\711c2c5e-db63-4599-98bd-3b5d9ff16dd7\scratchpad\result_flat.json")

WALL_TIME_CAP_SEC = 480.0  # hard watchdog; characterize as timeout if exceeded
MEM_CAP_BYTES = int(29.0 * 1e9)  # ~29GB; system has 33.4GB total


class Watchdog:
    def __init__(self, pid, wall_cap_sec, mem_cap_bytes, interval=0.5):
        self.proc = psutil.Process(pid)
        self.wall_cap_sec = wall_cap_sec
        self.mem_cap_bytes = mem_cap_bytes
        self.interval = interval
        self.peak_rss = 0
        self.tripped = None  # None | "wall_timeout" | "mem_cap"
        self.t_start = time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self.proc.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
                elapsed = time.time() - self.t_start
                if rss > self.mem_cap_bytes:
                    self.tripped = "mem_cap"
                    self._write_and_die(elapsed)
                if elapsed > self.wall_cap_sec:
                    self.tripped = "wall_timeout"
                    self._write_and_die(elapsed)
            except Exception:
                pass
            time.sleep(self.interval)

    def _write_and_die(self, elapsed):
        result = {
            "outcome": "FAILED",
            "reason": self.tripped,
            "elapsed_sec_before_kill": elapsed,
            "peak_rss_bytes": self.peak_rss,
            "peak_rss_gb": self.peak_rss / 1e9,
            "wall_time_cap_sec": self.wall_cap_sec,
            "mem_cap_bytes": self.mem_cap_bytes,
        }
        OUT_JSON.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        sys.stdout.flush()
        import os
        os._exit(137)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


def main():
    if not (str(CACHE_PATH) + ".npz"):
        pass
    cache_file = Path(str(CACHE_PATH) + ".npz")
    if not cache_file.exists():
        print(f"FATAL: cache not found at {cache_file}; run the scene_sparse arm first.", file=sys.stderr)
        sys.exit(1)

    wd = Watchdog(psutil.Process().pid, WALL_TIME_CAP_SEC, MEM_CAP_BYTES)
    wd.start()

    t_load0 = time.time()
    idx = iris_ingest.load_index(CACHE_PATH)  # rebuilds graph with the SAVED (scene_sparse) config as a side effect
    load_wall_sec = time.time() - t_load0

    num_survivors = len(idx.frames)

    # Build the FLAT graph from the identical frame set (same embeddings,
    # same scene_ids, same everything) -- only graph_mode differs.
    # config_snapshot on the loaded index is a plain dict; _get() in ingest.py
    # supports dict configs directly, so patch graph_mode in-place.
    cfg_dict = dict(idx.config_snapshot)
    cfg_dict["graph_mode"] = "flat"
    cfg_dict["graph_edge_mode"] = "fully_connected"

    t0 = time.time()
    flat_graph = iris_ingest._build_graph(idx.frames, cfg_dict)
    build_wall_sec = time.time() - t0

    wd.stop()

    node_count = flat_graph.graph.number_of_nodes()
    edge_count = flat_graph.graph.number_of_edges()
    edges_theoretical = num_survivors * (num_survivors - 1) // 2

    result = {
        "outcome": "SUCCESS",
        "num_survivors": num_survivors,
        "load_wall_sec": load_wall_sec,
        "flat_build_wall_sec": build_wall_sec,
        "flat_node_count": node_count,
        "flat_edge_count": edge_count,
        "flat_edges_theoretical_N_choose_2": edges_theoretical,
        "edge_count_matches_theoretical": (edge_count == edges_theoretical),
        "peak_rss_bytes": wd.peak_rss,
        "peak_rss_gb": wd.peak_rss / 1e9,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
