"""Dump the 4 audit-closure frames (+1 neighbor pair for the count check)
so the witness-seat decision can close. Standalone: cv2 only.

Run:  python scripts/dump_audit_closure.py
Then: open frame_dumps_audit/ and check the questions printed below.
"""
from pathlib import Path
import cv2
import sys

REPO = Path(__file__).resolve().parent.parent
VIDEO = REPO / "eval" / "data" / "virat" / "videos" / "VIRAT_S_000102.mp4"
OUT = REPO / "frame_dumps_audit"

# frame_idx: question to answer
CHECKS = {
    23748: "Q1: Is anyone opening/adjusting the CAR door? And is the car gray or black?",
    24420: "Q2: Is a man in dark clothing stepping into/out of the car? (classifies exhibit 3)",
    873:   "Q3a: How many people visible? (minicpm says 2, your notes say 3)",
    1263:  "Q3b: How many people visible? (minicpm says 2, your notes say 3)",
    27184: "Q4: Striped-shirt person -- could it read as a safety vest, or is that invented?",
}

def main() -> None:
    if not VIDEO.exists():
        print(f"FATAL: {VIDEO} not found", file=sys.stderr); sys.exit(1)
    OUT.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    for idx, q in CHECKS.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            print(f"  frame {idx}: read FAILED"); continue
        t = idx / fps
        name = OUT / f"f{idx:06d}_t{int(t//60):02d}m{t%60:04.1f}s.png"
        cv2.imwrite(str(name), img)
        print(f"saved {name.name}\n   -> {q}\n")
    cap.release()
    print(f"Done -> {OUT}/  (5 frames, 4 questions)")

if __name__ == "__main__":
    main()
