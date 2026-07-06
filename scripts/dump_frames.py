"""scripts/dump_frames.py — dump specific VIRAT_S_000102 frames to PNG for
manual visual inspection. Reuses the seek-based frame fetch already cloned
in scripts/diag_v4_moondream_captions.py (itself cloned from
iris/query.py:_ensure_captions's per-miss seek loop) -- no new decoder,
no iris/ edits.

Usage:
    python scripts/dump_frames.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.demo_cctv_query import CACHE_PATH, VIDEO, _load_index
from scripts.diag_v4_moondream_captions import _fetch_frames_pil

FRAMES = [258, 1263, 3243, 3334, 23748, 27184, 873]


def main() -> None:
    npz = Path(str(CACHE_PATH) + ".npz")
    if not npz.exists():
        print(f"FATAL: no index cache at {npz}", file=sys.stderr)
        sys.exit(1)
    if not VIDEO.exists():
        print(f"FATAL: no video at {VIDEO}", file=sys.stderr)
        sys.exit(1)

    idx = _load_index()
    out = REPO / "frame_dumps"
    out.mkdir(exist_ok=True)

    pil_by_frame = _fetch_frames_pil(idx, FRAMES)
    for frame_idx in FRAMES:
        img = pil_by_frame[frame_idx]
        img.save(out / f"frame_{frame_idx}.png")
        print(f"saved frame_{frame_idx}.png")


if __name__ == "__main__":
    main()
