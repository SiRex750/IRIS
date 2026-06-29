"""One-shot: ingest mov_bbb.mp4 and cache the index to .diag4_cache_mov_bbb.npz."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from iris.ingest import ingest, save_index

vp = REPO_ROOT / "mov_bbb.mp4"
cache = REPO_ROOT / ".diag4_cache_mov_bbb.npz"

print(f"video: {vp}  exists={vp.exists()}", flush=True)
if not vp.exists():
    print("ERROR: video not found")
    sys.exit(1)

print("Starting ingest...", flush=True)
idx = ingest(str(vp))
save_index(idx, cache)
print(f"Cache written: {cache.name}  frames={len(idx.frames)}", flush=True)
