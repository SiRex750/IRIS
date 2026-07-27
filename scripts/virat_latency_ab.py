"""VIRAT N=4892 latency A/B: flat vs scene_sparse, reusing scripts/latency_ab.py's
timing harness (_perf, TIMER_KEYS, COUNT_KEYS, N_WARMUP, N_REPS, TOP_K,
_median_or_none) -- no new timer written.

VIRAT has no NExT-GQA questions, so query embeddings come from a seeded
sample of survivor CLIP embeddings (recorded below), not from _embed_query
text encoding. That means query_embed_s is N/A here by design, not by scale
-- called out explicitly in the report, not hidden.

Read-only against both caches: loads the existing scene_sparse .npz, builds
the flat graph ONCE in memory from the same loaded frames (no re-ingest, no
re-decode, nothing written back to disk). Both one-time setup costs are
timed and reported separately from the per-query loop.

VERIFY: python virat_latency_ab.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\Siddanth Anil\IRIS")
sys.path.insert(0, str(REPO))

try:
    import torch
    torch.set_num_threads(8)
except Exception:
    torch = None

import numpy as np

import iris.ingest as iris_ingest
import iris.scene_retrieval as scene_retrieval
from iris import _perf
from iris.iris_config import IRISConfig
from iris.query import _build_retrieved

# Reuse the harness's constants and stats helper directly -- not redefined.
sys.path.insert(0, str(REPO / "scripts"))
from latency_ab import N_WARMUP, N_REPS, TOP_K, TIMER_KEYS, COUNT_KEYS, _median_or_none  # noqa: E402

CACHE_PATH = REPO / "eval" / "data" / "virat" / "index_cache" / "VIRAT_S_040001_01_000448_001101"
NEXTQA_CACHE_DIR = REPO / "eval" / "data" / "nextqa" / "index_cache"
VIRAT_CACHE_DIR = REPO / "eval" / "data" / "virat" / "index_cache"
OUT_JSON = Path(r"C:\Users\SIDDAN~1\AppData\Local\Temp\claude\C--Users-Siddanth-Anil-IRIS\711c2c5e-db63-4599-98bd-3b5d9ff16dd7\scratchpad\virat_latency_result.json")

N_Q = 50
QUERY_SEED = 20260726  # pinned, recorded

BASE = dict(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,
    ppr_damping=0.5,
    l2_retrieve_top_k=TOP_K,
    scene_diag=True,
)


def _dir_fingerprint(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(d.glob("*.npz")):
        st = f.stat()
        h.update(f.name.encode()); h.update(str(st.st_mtime_ns).encode()); h.update(str(st.st_size).encode())
    return h.hexdigest()


def _timed_query_from_embedding(index, emb, cfg) -> tuple[dict, dict]:
    """Same shape as latency_ab._timed_query, minus the _embed_query call --
    we already have the embedding (sampled survivor CLIP vector, seeded)."""
    _perf.reset()
    _build_retrieved(index, emb, cfg)
    timings = {k: _perf.TIMINGS.get(k) for k in TIMER_KEYS}
    counts = {k: _perf.COUNTS.get(k) for k in COUNT_KEYS}
    return timings, counts


def run_arm(mode: str, index, queries: list[np.ndarray]) -> list[dict]:
    cfg = IRISConfig(**BASE, graph_mode=mode)

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
        records.append({
            "qi": qi,
            "graph_mode": mode,
            "branch": branch,
            **median_timings,
            **median_counts,
        })
    return records


def main():
    nextqa_before = _dir_fingerprint(NEXTQA_CACHE_DIR)
    virat_before = _dir_fingerprint(VIRAT_CACHE_DIR)

    # ── ONE-TIME setup: load + build each graph mode ONCE, outside timing ────
    t0 = time.perf_counter()
    idx_scene_sparse = iris_ingest.load_index(CACHE_PATH)
    scene_sparse_setup_s = time.perf_counter() - t0
    assert idx_scene_sparse.config_snapshot.get("graph_mode") == "scene_sparse"

    n_survivors = len(idx_scene_sparse.frames)

    t0 = time.perf_counter()
    flat_cfg_dict = dict(idx_scene_sparse.config_snapshot)
    flat_cfg_dict["graph_mode"] = "flat"
    flat_graph = iris_ingest._build_graph(idx_scene_sparse.frames, flat_cfg_dict)
    flat_setup_s = time.perf_counter() - t0
    idx_flat = dataclasses.replace(
        idx_scene_sparse,
        config_snapshot=flat_cfg_dict,
        _graph=flat_graph,
    )

    nextqa_after_setup = _dir_fingerprint(NEXTQA_CACHE_DIR)
    virat_after_setup = _dir_fingerprint(VIRAT_CACHE_DIR)

    # ── Seeded synthetic queries: sample N_Q survivor CLIP embeddings ────────
    rng = np.random.default_rng(QUERY_SEED)
    frames_with_emb = [fr for fr in idx_scene_sparse.frames if fr.clip_embedding is not None]
    assert len(frames_with_emb) >= N_Q, (
        f"only {len(frames_with_emb)} survivors have clip_embedding, need {N_Q}"
    )
    chosen_idx = rng.choice(len(frames_with_emb), size=N_Q, replace=False)
    chosen_frame_idxs = [int(frames_with_emb[i].frame_idx) for i in chosen_idx]
    queries = []
    for i in chosen_idx:
        v = np.asarray(frames_with_emb[i].clip_embedding, dtype=np.float32)
        n = np.linalg.norm(v)
        queries.append(v / n if n > 0 else v)

    assert len(queries) == N_Q

    # ── Timed per-query loops (setup already done above) ────────────────────
    records_flat = run_arm("flat", idx_flat, queries)
    records_scene = run_arm("scene_sparse", idx_scene_sparse, queries)

    assert len(records_flat) == len(records_scene) == N_Q, "query count mismatch across modes"

    nextqa_after = _dir_fingerprint(NEXTQA_CACHE_DIR)
    virat_after = _dir_fingerprint(VIRAT_CACHE_DIR)

    def pooled(records, key, branch=None):
        vals = [r[key] for r in records if r[key] is not None and (branch is None or r["branch"] == branch)]
        return _median_or_none(vals)

    shortcut_n = sum(1 for r in records_scene if r["branch"] == "shortcut")
    descend_n = sum(1 for r in records_scene if r["branch"] == "descend")

    result = {
        "n_survivors": n_survivors,
        "n_queries": N_Q,
        "query_seed": QUERY_SEED,
        "query_source": "seeded sample of survivor CLIP embeddings (L2-normalized), not text-encoded",
        "chosen_frame_idxs": chosen_frame_idxs,
        "setup_costs_one_time_excluded_from_per_query": {
            "scene_sparse_load_s": scene_sparse_setup_s,
            "flat_build_s": flat_setup_s,
        },
        "median_total_retrieval_s": {
            "flat": pooled(records_flat, "total_retrieval_s"),
            "scene_sparse": pooled(records_scene, "total_retrieval_s"),
            "scene_sparse_shortcut": pooled(records_scene, "total_retrieval_s", "shortcut"),
            "scene_sparse_descend": pooled(records_scene, "total_retrieval_s", "descend"),
        },
        "component_breakdown_median_s": {
            "flat": {k: pooled(records_flat, k) for k in TIMER_KEYS},
            "scene_sparse": {k: pooled(records_scene, k) for k in TIMER_KEYS},
        },
        "median_ppr_nodes_edges": {
            "flat": {k: pooled(records_flat, k) for k in COUNT_KEYS},
            "scene_sparse": {k: pooled(records_scene, k) for k in COUNT_KEYS},
        },
        "scene_sparse_branch_fire_rate": {
            "shortcut": shortcut_n,
            "descend": descend_n,
            "total": len(records_scene),
            "shortcut_pct": shortcut_n / len(records_scene) * 100.0,
        },
        "assertions": {
            "query_count_matches_across_modes": len(records_flat) == len(records_scene) == N_Q,
            "nextqa_cache_untouched": (nextqa_before == nextqa_after),
            "virat_cache_untouched": (virat_before == virat_after),
            "nextqa_fingerprint_before": nextqa_before,
            "nextqa_fingerprint_after": nextqa_after,
            "virat_fingerprint_before": virat_before,
            "virat_fingerprint_after_setup": virat_after_setup,
            "virat_fingerprint_after": virat_after,
        },
        "per_query_records": {
            "flat": records_flat,
            "scene_sparse": records_scene,
        },
    }

    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_query_records"}, indent=2))


if __name__ == "__main__":
    main()
