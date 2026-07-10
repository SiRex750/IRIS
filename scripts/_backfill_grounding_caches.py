"""One-off backfill: ingest missing dev_100 grounded videos into BOTH
eval/data/nextqa/index_cache (flat) and index_cache_ssparse (scene_sparse)
so the grounding gate can run on the largest reachable N. Resumable (skips
existing .npz). Not part of any production path.
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

DATA_DIR = REPO / "eval" / "data" / "nextqa"
FLAT_DIR = DATA_DIR / "NExTVideo_flat"
FLAT_CACHE = DATA_DIR / "index_cache"
SSPARSE_CACHE = DATA_DIR / "index_cache_ssparse"
DEV_JSONL = DATA_DIR / "dev_100.jsonl"
GQA_JSON = DATA_DIR / "gsub_val.json"

FLAT_CFG = IRISConfig(
    ranking_mode="ppr", codec_conf_source="packet_size",
    codec_conf_pictype_norm=True, ppr_lambda=0.5, ppr_damping=0.5,
)
SSPARSE_CFG = IRISConfig(
    ranking_mode="ppr", codec_conf_source="packet_size",
    codec_conf_pictype_norm=True, ppr_lambda=0.5, ppr_damping=0.5,
    graph_mode="scene_sparse",
)


def backfill(cache_dir: Path, cfg: IRISConfig, vids: list[str], label: str) -> None:
    print(f"=== backfilling {label} ({len(vids)} candidates) ===")
    for i, vid in enumerate(vids, 1):
        npz = cache_dir / f"{vid}.npz"
        if npz.exists():
            print(f"[{label} {i}/{len(vids)}] SKIP (cached) {vid}")
            continue
        video_path = FLAT_DIR / f"{vid}.mp4"
        if not video_path.exists():
            print(f"[{label} {i}/{len(vids)}] MISSING video file {vid}")
            continue
        t0 = time.time()
        try:
            idx = iris_ingest.ingest(str(video_path), config=cfg)
            iris_ingest.save_index(idx, cache_dir / vid)
            print(f"[{label} {i}/{len(vids)}] OK {vid} N={len(idx.frames)} t={time.time()-t0:.1f}s")
        except Exception as e:
            print(f"[{label} {i}/{len(vids)}] FAIL {vid}: {str(e)[:120]}")
        sys.stdout.flush()


def main() -> None:
    dev_rows = [json.loads(l) for l in open(DEV_JSONL, encoding="utf-8")]
    gsub = json.load(open(GQA_JSON, encoding="utf-8"))
    grounded = [
        r for r in dev_rows
        if r["video"] in gsub and str(r["qid"]) in gsub[r["video"]]["location"]
    ]
    distinct_vids = sorted({r["video"] for r in grounded})
    print(f"dev_100 grounded distinct videos: {len(distinct_vids)}")

    flat_cached = {p.stem for p in FLAT_CACHE.glob("*.npz")}
    ssparse_cached = {p.stem for p in SSPARSE_CACHE.glob("*.npz")}

    missing_flat = [v for v in distinct_vids if v not in flat_cached]
    missing_ss = [v for v in distinct_vids if v not in ssparse_cached]

    backfill(FLAT_CACHE, FLAT_CFG, missing_flat, "flat")
    backfill(SSPARSE_CACHE, SSPARSE_CFG, missing_ss, "ssparse")

    print()
    flat_cached2 = {p.stem for p in FLAT_CACHE.glob("*.npz")}
    ssparse_cached2 = {p.stem for p in SSPARSE_CACHE.glob("*.npz")}
    both = set(distinct_vids) & flat_cached2 & ssparse_cached2
    print(f"After backfill: {len(both)}/{len(distinct_vids)} grounded videos cached in BOTH.")


if __name__ == "__main__":
    main()
