"""Worker subprocess for scripts/scaling_curve_v2.py.

Runs ONE (clip, graph_mode) timing arm in its own process so a flat OOM/crash
cannot take down the scene_sparse measurement for the same clip, and so a
wall-time/memory watchdog can be applied uniformly without corrupting the
parent driver's state.

Two graph sources for the "flat" mode (fairness check, see driver):
  --source cached_frames      : build flat graph from the frames already
                                 loaded from the scene_sparse .npz cache
                                 (same construction as the v1 curve).
  --source independent_ingest : re-ingest the raw video from scratch with
                                 graph_mode="flat" set from the start (no
                                 scene_sparse detour). Requires --video.

scene_sparse mode always uses the cached .npz (it was ingested that way).

Watchdog wraps: index load + graph build (flat only) + the FULL timed query
loop (N_WARMUP + n_queries * N_REPS calls to _build_retrieved), matching the
v1 curve's "GRAPH BUILD + full 50-query timed loop combined" wrapping so v1
and v2 wall-time numbers are comparable. On trip, writes a FAILED result and
os._exit(137) so the parent sees an unambiguous signal distinct from a clean
crash.

VERIFY: python scripts/_scaling_curve_v2_worker.py --help
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import argparse
import dataclasses
import json
import sys
import threading
import time
from pathlib import Path

REPO = Path(r"C:\Users\Siddanth Anil\IRIS")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

try:
    import torch
    torch.set_num_threads(8)
except Exception:
    torch = None

import numpy as np
import psutil

import iris.ingest as iris_ingest
import iris.scene_retrieval as scene_retrieval
from iris import _perf
from iris.iris_config import IRISConfig
from iris.query import _build_retrieved, _embed_query
from latency_ab import N_WARMUP, N_REPS, TOP_K, TIMER_KEYS, COUNT_KEYS, _median_or_none  # noqa: E402

# Fixed surveillance-domain text queries for --query-source text. Cycled to
# match --n-queries so per-clip query count stays identical to the synthetic
# (sampled-survivor-embedding) run.
TEXT_QUERIES = [
    "a person running",
    "an unattended bag",
    "two people fighting",
    "a vehicle entering the scene",
    "a person falling",
    "someone climbing over a fence",
    "a person carrying a weapon",
    "a crowd gathering suddenly",
    "someone breaking a window",
    "a person loitering near a doorway",
]


class Watchdog:
    """Trips on wall time OR RSS; identical policy object for every clip/mode."""

    def __init__(self, pid, wall_cap_sec, mem_cap_bytes, out_json: Path, interval=0.5):
        self.proc = psutil.Process(pid)
        self.wall_cap_sec = wall_cap_sec
        self.mem_cap_bytes = mem_cap_bytes
        self.interval = interval
        self.out_json = out_json
        self.peak_rss = 0
        self.tripped = None
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
                    self.tripped = "oom_cap"
                    self._write_and_die(elapsed)
                if elapsed > self.wall_cap_sec:
                    self.tripped = "timed_out"
                    self._write_and_die(elapsed)
            except Exception:
                pass
            time.sleep(self.interval)

    def _write_and_die(self, elapsed):
        result = {
            "outcome": self.tripped,
            "elapsed_sec_before_kill": elapsed,
            "peak_rss_bytes": self.peak_rss,
            "peak_rss_gb": self.peak_rss / 1e9,
            "wall_time_cap_sec": self.wall_cap_sec,
            "mem_cap_bytes": self.mem_cap_bytes,
        }
        self.out_json.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        sys.stdout.flush()
        os._exit(137)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


def _timed_query_from_embedding(index, item, cfg) -> tuple[dict, dict]:
    """item is either a pre-computed np.ndarray embedding (query_source=synthetic)
    or a text string (query_source=text, embedded via the real CLIP text encoder
    inside the timed region, matching latency_ab.py's _timed_query)."""
    _perf.reset()
    if isinstance(item, str):
        emb = _embed_query(item, cfg)
    else:
        emb = item
    _build_retrieved(index, emb, cfg)
    timings = {k: _perf.TIMINGS.get(k) for k in TIMER_KEYS}
    counts = {k: _perf.COUNTS.get(k) for k in COUNT_KEYS}
    return timings, counts


def run_query_loop(mode: str, index, queries: list) -> list[dict]:
    cfg = IRISConfig(
        ranking_mode="ppr",
        codec_conf_source="packet_size",
        codec_conf_pictype_norm=True,
        ppr_lambda=0.5,
        ppr_damping=0.5,
        l2_retrieve_top_k=TOP_K,
        scene_diag=True,
        graph_mode=mode,
    )

    warmup_emb = queries[0]
    for _ in range(N_WARMUP):
        _timed_query_from_embedding(index, warmup_emb, cfg)

    records = []
    for qi, emb in enumerate(queries):
        rep_timings, rep_counts = [], []
        branch = "n/a"
        for _ in range(N_REPS):
            scene_retrieval.SCENE_DIAG_RECORDS.clear()
            timings, counts = _timed_query_from_embedding(index, emb, cfg)
            rep_timings.append(timings)
            rep_counts.append(counts)
            if mode == "scene_sparse" and scene_retrieval.SCENE_DIAG_RECORDS:
                branch = scene_retrieval.SCENE_DIAG_RECORDS[-1]["branch"]
        median_timings = {k: _median_or_none([t[k] for t in rep_timings]) for k in TIMER_KEYS}
        median_counts = {k: _median_or_none([c[k] for c in rep_counts]) for k in COUNT_KEYS}
        records.append({"qi": qi, "graph_mode": mode, "branch": branch, **median_timings, **median_counts})
    return records


def seeded_queries(frames, seed: int, n_q: int):
    rng = np.random.default_rng(seed)
    frames_with_emb = [fr for fr in frames if fr.clip_embedding is not None]
    assert len(frames_with_emb) >= n_q, f"only {len(frames_with_emb)} survivors have clip_embedding, need {n_q}"
    chosen_idx = rng.choice(len(frames_with_emb), size=n_q, replace=False)
    chosen_frame_idxs = [int(frames_with_emb[i].frame_idx) for i in chosen_idx]
    queries = []
    for i in chosen_idx:
        v = np.asarray(frames_with_emb[i].clip_embedding, dtype=np.float32)
        n = np.linalg.norm(v)
        queries.append(v / n if n > 0 else v)
    return queries, chosen_frame_idxs


def text_queries(n_q: int):
    """Cycle the fixed TEXT_QUERIES list to length n_q -- same query count as
    the synthetic run, no seed needed (list order is fixed)."""
    chosen = [TEXT_QUERIES[i % len(TEXT_QUERIES)] for i in range(n_q)]
    return chosen, None


def pooled(records, key, branch=None):
    vals = [r[key] for r in records if r[key] is not None and (branch is None or r["branch"] == branch)]
    return _median_or_none(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["flat", "scene_sparse"], required=True)
    ap.add_argument("--cache-path", required=True, help="path WITHOUT .npz suffix, scene_sparse index cache")
    ap.add_argument("--video", default=None, help="raw video path, required for --source independent_ingest")
    ap.add_argument("--source", choices=["cached_frames", "independent_ingest"], default="cached_frames")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--wall-cap-sec", type=float, required=True)
    ap.add_argument("--mem-cap-bytes", type=int, required=True)
    ap.add_argument("--query-seed", type=int, required=True)
    ap.add_argument("--n-queries", type=int, required=True)
    ap.add_argument("--query-source", choices=["synthetic", "text"], default="synthetic",
                     help="synthetic: seeded sample of survivor CLIP image embeddings (default, "
                          "matches v2). text: fixed list of real CLIP text-encoded queries.")
    args = ap.parse_args()

    out_json = Path(args.out_json)

    wd = Watchdog(psutil.Process().pid, args.wall_cap_sec, args.mem_cap_bytes, out_json)
    wd.start()

    t0 = time.time()
    if args.mode == "scene_sparse":
        idx = iris_ingest.load_index(Path(args.cache_path))
        assert idx.config_snapshot.get("graph_mode") == "scene_sparse"
        load_or_build_s = time.time() - t0
        n_survivors = len(idx.frames)
        graph = idx._graph
    else:
        if args.source == "cached_frames":
            idx_ss = iris_ingest.load_index(Path(args.cache_path))
            n_survivors = len(idx_ss.frames)
            flat_cfg = dict(idx_ss.config_snapshot)
            flat_cfg["graph_mode"] = "flat"
            flat_cfg["graph_edge_mode"] = "fully_connected"
            t_build0 = time.time()
            flat_graph = iris_ingest._build_graph(idx_ss.frames, flat_cfg)
            load_or_build_s = time.time() - t0  # includes the load above
            idx = dataclasses.replace(idx_ss, config_snapshot=flat_cfg, _graph=flat_graph)
            graph = flat_graph
        else:  # independent_ingest
            assert args.video, "--video required for --source independent_ingest"
            idx_ss = iris_ingest.load_index(Path(args.cache_path))
            flat_cfg_dict = dict(idx_ss.config_snapshot)
            flat_cfg_dict["graph_mode"] = "flat"
            flat_cfg_dict["graph_edge_mode"] = "fully_connected"
            cfg = IRISConfig(**flat_cfg_dict)
            idx = iris_ingest.ingest(args.video, config=cfg)
            load_or_build_s = time.time() - t0
            n_survivors = len(idx.frames)
            graph = idx._graph

    if args.query_source == "text":
        queries, chosen_frame_idxs = text_queries(args.n_queries)
    else:
        queries, chosen_frame_idxs = seeded_queries(idx.frames, args.query_seed, args.n_queries)
    records = run_query_loop(args.mode, idx, queries)

    wd.stop()

    node_count = graph.graph.number_of_nodes()
    edge_count = graph.graph.number_of_edges()

    result = {
        "outcome": "SUCCESS",
        "mode": args.mode,
        "source": args.source,
        "n_survivors": n_survivors,
        "n_queries": args.n_queries,
        "query_source": args.query_source,
        "query_seed": args.query_seed,
        "chosen_frame_idxs": chosen_frame_idxs,
        "load_or_build_wall_sec": load_or_build_s,
        "node_count": node_count,
        "edge_count": edge_count,
        "median_total_retrieval_s": pooled(records, "total_retrieval_s"),
        "component_breakdown_median_s": {k: pooled(records, k) for k in TIMER_KEYS},
        "median_ppr_nodes_edges": {k: pooled(records, k) for k in COUNT_KEYS},
        "peak_rss_bytes": wd.peak_rss,
        "peak_rss_gb": wd.peak_rss / 1e9,
        "wall_time_cap_sec": args.wall_cap_sec,
        "mem_cap_bytes": args.mem_cap_bytes,
    }
    if args.mode == "scene_sparse":
        shortcut_n = sum(1 for r in records if r["branch"] == "shortcut")
        descend_n = sum(1 for r in records if r["branch"] == "descend")
        result["branch_fire_rate"] = {
            "shortcut": shortcut_n,
            "descend": descend_n,
            "total": len(records),
            "shortcut_pct": shortcut_n / len(records) * 100.0 if records else None,
        }
        result["median_total_retrieval_s_shortcut"] = pooled(records, "total_retrieval_s", "shortcut")
        result["median_total_retrieval_s_descend"] = pooled(records, "total_retrieval_s", "descend")

    out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "chosen_frame_idxs"}, indent=2))


if __name__ == "__main__":
    main()
