"""Read-only: derive per-clip survivor census (N, scene count, duration,
config weights, graph_edge_mode) from every .npz in eval/data/ucf/index_cache
(+ VIRAT cache, if present) for the C1.5 CI task. Reads the '__manifest__'
JSON blob exactly as iris.ingest.load_index does, but does NOT rebuild the
live graph / decode embeddings -- cheap, no ingest, no writes outside
eval_results/_ci_survivor_census.json.
"""
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Users\Siddanth Anil\IRIS")

import numpy as np

CACHE_DIR = REPO / "eval" / "data" / "ucf" / "index_cache"
VIRAT_CACHE = REPO / "eval" / "data" / "virat" / "index_cache"

FROZEN_WEIGHTS = (0.8, 0.1, 0.1)  # ledger S6 frozen production default (luma_diff, motion, luma_entropy)

records = []
cache_files = sorted(CACHE_DIR.glob("*.npz"))
if VIRAT_CACHE.exists():
    cache_files += sorted(VIRAT_CACHE.glob("*.npz"))

for p in cache_files:
    label = p.stem
    try:
        data = np.load(p, allow_pickle=False)
        manifest = json.loads(data["__manifest__"].item())
        frame_list = manifest["frames"]
        cfg_snap = manifest.get("config_snapshot", {}) or {}
        timestamps = [float(f["timestamp"]) for f in frame_list]
        scene_ids = sorted({int(f.get("scene_id", -1)) for f in frame_list})
        n_survivors = len(frame_list)
        duration_s = (max(timestamps) - min(timestamps)) if timestamps else None
        weights = (
            cfg_snap.get("luma_diff_weight"),
            cfg_snap.get("motion_weight"),
            cfg_snap.get("luma_entropy_weight"),
        )
        records.append({
            "label": label,
            "n_survivors": n_survivors,
            "n_scenes": len([s for s in scene_ids if s >= 0]),
            "scene_ids_negative_present": any(s < 0 for s in scene_ids),
            "duration_s": duration_s,
            "graph_edge_mode": cfg_snap.get("graph_edge_mode"),
            "graph_mode": cfg_snap.get("graph_mode"),
            "salience_weights": weights,
            "weights_match_frozen_default": tuple(weights) == FROZEN_WEIGHTS,
            "cache_path": str(p.relative_to(REPO)),
        })
    except Exception as e:
        records.append({"label": label, "error": repr(e), "cache_path": str(p.relative_to(REPO))})

out = REPO / "eval_results" / "_ci_survivor_census.json"
out.write_text(json.dumps({"records": records}, indent=2))
print(json.dumps({"records": records}, indent=2, default=str))
print(f"\nwrote {out}", file=sys.stderr)
