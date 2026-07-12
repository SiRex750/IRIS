"""Dump frames around the two entrance-activity windows of VIRAT_S_000102
so a human can answer: does anyone visibly ENTER the building during the
recording? Ground-truth check for smoke #2 fixture polarity.
Standalone: cv2 only, no iris imports.

Run:  python scripts/dump_entrance_check.py
Then: open frame_dumps_entrance/ and eyeball in filename order.
"""
from pathlib import Path
import cv2
import sys

REPO = Path(__file__).resolve().parent.parent
VIDEO = REPO / "eval" / "data" / "virat" / "videos" / "VIRAT_S_000102.mp4"
OUT = REPO / "frame_dumps_entrance"

# Window 1: around frame 258 (~8.6s) — white-shirt person exiting building.
# Dense sampling BEFORE it too, in case the exit was preceded by an entry.
WINDOW_1 = list(range(0, 700, 50))          # frames 0..650, every ~1.7s

# Window 2: frames ~26268–27184 (~876–907s) — people near entrance late in video.
WINDOW_2 = list(range(26100, 27300, 60))    # every ~2s

FRAMES = sorted(set(WINDOW_1 + WINDOW_2))

def main() -> None:
    if not VIDEO.exists():
        print(f"FATAL: {VIDEO} not found", file=sys.stderr); sys.exit(1)
    OUT.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    n_ok = 0
    for idx in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            print(f"  frame {idx}: read failed (past EOF?)"); continue
        t = idx / fps
        name = OUT / f"f{idx:06d}_t{int(t//60):02d}m{t%60:04.1f}s.png"
        cv2.imwrite(str(name), img)
        n_ok += 1
        print(f"  saved {name.name}")
    cap.release()
    print(f"\n{n_ok}/{len(FRAMES)} frames -> {OUT}/")
    print("Eyeball in order. Question: does anyone cross the threshold INTO")
    print("the building? Exiting / standing near the door = NO (entry predates")
    print("recording). If yes, note the frame number from the filename.")

if __name__ == "__main__":
    main()
