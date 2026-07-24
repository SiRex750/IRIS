"""One-off, decode-free rebuild of index_cache_ssparse/ from index_cache/.

Loads each flat-mode .npz index (frames + CLIP embeddings already cached,
scene_id already assigned from a prior real ingest), overrides
config_snapshot.graph_mode = "scene_sparse", rebuilds the graph via
iris.ingest._build_graph on the in-memory frame records, and saves into
index_cache_ssparse/. No video is opened, no CLIP call is made, no
captioning runs -- this only re-derives graph structure from records
already present in the flat cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import iris.ingest as iris_ingest

DATA_DIR = REPO / "eval" / "data" / "nextqa"
FLAT_CACHE = DATA_DIR / "index_cache"
SSPARSE_CACHE = DATA_DIR / "index_cache_ssparse"


def main() -> None:
    SSPARSE_CACHE.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(FLAT_CACHE.glob("*.npz"))
    print(f"Found {len(npz_files)} flat-cache .npz files")

    saved = 0
    failed = []
    for i, npz_path in enumerate(npz_files, 1):
        vid = npz_path.stem
        idx = iris_ingest.load_index(FLAT_CACHE / vid)

        ss_config = dict(idx.config_snapshot)
        ss_config["graph_mode"] = "scene_sparse"

        try:
            idx._graph = iris_ingest._build_graph(idx.frames, ss_config)
        except Exception as e:
            failed.append((vid, str(e)[:150]))
            print(f"[{i:2d}/{len(npz_files)}] FAIL {vid}: {str(e)[:150]}")
            continue

        idx.config_snapshot = ss_config
        iris_ingest.save_index(idx, SSPARSE_CACHE / vid)
        saved += 1
        print(f"[{i:2d}/{len(npz_files)}] OK {vid}  N={len(idx.frames)}")

    print()
    print(f"Saved: {saved}/{len(npz_files)}   Failed: {len(failed)}")
    if failed:
        print("Failures:", failed)


if __name__ == "__main__":
    main()
