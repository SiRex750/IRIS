"""Robust pre-cache: catch all errors and write to C:\IRIS\diag4_run.log."""
import sys, traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

log_path = REPO_ROOT / "diag4_run.log"
log = open(log_path, "w", buffering=1, encoding="utf-8")

def L(msg):
    log.write(msg + "\n")
    log.flush()

L("start")
L(f"REPO_ROOT={REPO_ROOT}")

try:
    from iris.ingest import ingest, save_index
    L("imports ok")

    vp = REPO_ROOT / "mov_bbb.mp4"
    L(f"video={vp}  exists={vp.exists()}  size={vp.stat().st_size if vp.exists() else 'N/A'}")

    L("calling ingest...")
    idx = ingest(str(vp))
    L(f"ingest done: frames={len(idx.frames)}")

    cache = REPO_ROOT / ".diag4_cache_mov_bbb.npz"
    save_index(idx, cache)
    L(f"cache saved: {cache}")

except Exception:
    L("EXCEPTION:\n" + traceback.format_exc())
    sys.exit(1)
finally:
    log.close()
