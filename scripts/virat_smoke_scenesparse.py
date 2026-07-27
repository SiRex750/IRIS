"""VIRAT smoke test, arm 1: ingest + build scene_sparse graph.

Writes ONLY to eval/data/virat/index_cache/. Never touches nextqa cache.
Reports frame counts, ingest wall time, peak RSS, codec-log warnings,
and scene_sparse graph node/edge counts + build wall time.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[5] if False else None
REPO = Path(r"C:\Users\Siddanth Anil\IRIS")
sys.path.insert(0, str(REPO))

import psutil
import av.logging as av_logging

import iris.ingest as iris_ingest
from iris.iris_config import IRISConfig

VIDEO = REPO / "eval" / "data" / "virat" / "videos" / "VIRAT_S_040001_01_000448_001101.mp4"
CACHE_DIR = REPO / "eval" / "data" / "virat" / "index_cache"
CACHE_PATH = CACHE_DIR / "VIRAT_S_040001_01_000448_001101"
OUT_JSON = Path(r"C:\Users\SIDDAN~1\AppData\Local\Temp\claude\C--Users-Siddanth-Anil-IRIS\711c2c5e-db63-4599-98bd-3b5d9ff16dd7\scratchpad\result_scenesparse.json")

CFG = IRISConfig(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=8,
    graph_mode="scene_sparse",
    graph_edge_mode="fully_connected",
)


class MemMonitor:
    def __init__(self, pid, interval=1.0):
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
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


def main():
    if not VIDEO.exists():
        print(f"FATAL: {VIDEO} not found.", file=sys.stderr)
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    mon = MemMonitor(psutil.Process().pid)
    mon.start()

    av_logging.set_level(av_logging.WARNING)
    cap = av_logging.Capture(local=False)
    logs = cap.__enter__()

    t0 = time.time()
    idx = iris_ingest.ingest(str(VIDEO), config=CFG)
    ingest_wall_sec = time.time() - t0

    cap.__exit__(None, None, None)
    captured = [(lvl, name, msg) for (lvl, name, msg) in logs]

    peak_rss_ingest = mon.peak_rss

    # ── scene_sparse graph, authoritative from the just-built index ──────────
    num_survivors = len(idx.frames)
    scene_ids = [fr.scene_id for fr in idx.frames]
    unassigned = sum(1 for s in scene_ids if s < 0)
    num_scenes = len(set(s for s in scene_ids if s >= 0))
    edge_count = idx._graph.graph.number_of_edges()
    node_count = idx._graph.graph.number_of_nodes()

    per_scene_counts = {}
    for s in scene_ids:
        if s < 0:
            continue
        per_scene_counts[s] = per_scene_counts.get(s, 0) + 1
    survivors_per_scene = list(per_scene_counts.values())
    sum_c = sum(s * (s - 1) // 2 for s in survivors_per_scene)
    edges_flat_theoretical = num_survivors * (num_survivors - 1) // 2

    save_t0 = time.time()
    iris_ingest.save_index(idx, CACHE_PATH)
    save_wall_sec = time.time() - save_t0

    mon.stop()

    warn_records = [
        {"level": lvl, "name": name, "message": str(msg)} for (lvl, name, msg) in captured
    ]

    result = {
        "video": str(VIDEO),
        "container_frames_meta": None,
        "num_survivors": num_survivors,
        "num_scenes": num_scenes,
        "unassigned_scene_id_count": unassigned,
        "ingest_wall_sec": ingest_wall_sec,
        "save_wall_sec": save_wall_sec,
        "peak_rss_bytes_during_ingest": peak_rss_ingest,
        "peak_rss_gb_during_ingest": peak_rss_ingest / 1e9,
        "scene_sparse_node_count": node_count,
        "scene_sparse_edge_count": edge_count,
        "sum_C_scene_size_choose_2": sum_c,
        "block_diagonal_exact": (edge_count == sum_c),
        "edges_flat_theoretical_N_choose_2": edges_flat_theoretical,
        "codec_log_capture_count": len(warn_records),
        "codec_log_capture_sample": warn_records[:20],
        "frames_processed_total": idx.frames_processed,
        "skipped_frames_ratio": idx.skipped_frames_ratio,
        "storage_reduction_factor": idx.storage_reduction_factor,
        "cache_path": str(CACHE_PATH) + ".npz",
    }

    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
