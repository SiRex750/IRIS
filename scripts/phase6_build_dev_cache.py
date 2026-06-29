"""Phase 6 — Build ingest cache for all present dev videos.

Resumable: skips existing {id}.npz files. Per-video failures are caught and
logged as SKIP — never aborts the whole run, never substitutes a missing video.

Cache is lambda-independent: stores only ingested index (codec_conf, embeddings,
graph) — no query, no retrieval state, no lambda.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest
from iris.iris_config import IRISConfig

DATA_DIR  = REPO / "eval" / "data" / "nextqa"
FLAT_DIR  = DATA_DIR / "NExTVideo_flat"
CACHE_DIR = DATA_DIR / "index_cache"
DEV_JSONL = DATA_DIR / "dev_100.jsonl"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ingest config — query-independent fields only; lambda/damping not used at ingest
CFG = IRISConfig(
    ranking_mode="ppr",
    codec_conf_source="packet_size",
    codec_conf_pictype_norm=True,
    ppr_lambda=0.5,   # not used at ingest; stored in config_snapshot only
    ppr_damping=0.5,
)

# Assert nothing query-side gets serialized by checking save/load symmetry key
_QUERY_SIDE_KEYS = {"ppr_lambda", "ppr_damping", "ranking_mode"}


def main() -> None:
    dev_rows = [json.loads(l) for l in open(DEV_JSONL, encoding="utf-8")]
    distinct_vids = sorted({r["video"] for r in dev_rows})
    present = [v for v in distinct_vids if (FLAT_DIR / f"{v}.mp4").exists()]
    total = len(present)

    print(f"Dev distinct videos: {len(distinct_vids)}  present: {total}")
    print(f"Cache dir: {CACHE_DIR}")
    print()

    cached, skipped = [], []
    wall_start = time.time()

    for k, vid in enumerate(present, 1):
        cache_path = CACHE_DIR / vid
        npz = Path(str(cache_path) + ".npz")
        if npz.exists():
            print(f"[{k:3d}/{total}] SKIP (cached) {vid}")
            cached.append(vid)
            continue

        video_path = FLAT_DIR / f"{vid}.mp4"
        t0 = time.time()
        try:
            index = iris_ingest.ingest(str(video_path), config=CFG)
            iris_ingest.save_index(index, cache_path)
            elapsed = time.time() - t0
            n = len(index.frames)
            print(f"[{k:3d}/{total}] OK  {vid}  N={n}  t={elapsed:.1f}s")
            cached.append(vid)
        except Exception as e:
            elapsed = time.time() - t0
            reason = str(e)[:120]
            print(f"[{k:3d}/{total}] SKIP {vid}: {reason}  (t={elapsed:.1f}s)")
            skipped.append((vid, reason))

        sys.stdout.flush()

    wall_elapsed = time.time() - wall_start

    print()
    print("=== SUMMARY ===")
    print(f"Total wall-clock : {wall_elapsed:.1f}s ({wall_elapsed/60:.1f}min)")
    print(f"Cached           : {len(cached)}")
    print(f"Skipped (errors) : {len(skipped)}")
    if skipped:
        print("SKIP list:")
        for vid, reason in skipped:
            print(f"  {vid}: {reason}")
    else:
        print("SKIP list: (none)")


if __name__ == "__main__":
    main()
