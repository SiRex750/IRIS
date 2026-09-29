"""Pilot step 0: git check, static scan over all Sintel training sequences, manifest.

Static share per sequence = (valid, non-occluded pixels with |GT| < 0.5 px) / (valid, non-occluded
pixels), pooled over every flow file of the sequence, on the flow's native grid (frame t-1).
Writes p3/pilot/static_scan.csv and p3/pilot/manifest.json.
"""
import csv
import datetime
import json
import os
import re
import subprocess
import sys

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
REPO = os.path.dirname(P3)
sys.path.insert(0, P3)

from lib.flo import SintelSeq  # noqa: E402

SINTEL = r"C:\Users\akash\Documents\datasets\MPI-Sintel"
EXPECTED = {"ambush_7": 0.794, "bandage_2": 0.555, "market_2": 0.491, "bandage_1": 0.465,
            "shaman_2": 0.445, "alley_1": 0.0015, "ambush_5": 0.0059, "temple_2": 0.0047}
TOL = 0.005


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def x264_build():
    """x264 version string from the SEI of a 2-frame throwaway encode (kept in p3/work)."""
    p = os.path.join(P3, "work", "x264_probe.mp4")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with av.open(p, "w") as o:
        st = o.add_stream("libx264", rate=24)
        st.width, st.height, st.pix_fmt = 64, 64, "yuv420p"
        for _ in range(2):
            for pk in st.encode(av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")):
                o.mux(pk)
        for pk in st.encode(None):
            o.mux(pk)
    blob = open(p, "rb").read()
    m = re.search(rb"(x264 - core [ -~]+?) - H\.264", blob)
    return m.group(1).decode() if m else None


def main():
    # a. git
    dirty = git("status", "--porcelain", "--", "p3/lib")
    tracked = git("ls-files", "p3/lib")
    if dirty or not tracked:
        print("STOP 0a: p3/lib not committed/clean:\n", dirty or "(untracked)")
        sys.exit(1)
    print("0a OK: p3/lib committed and clean at", git("rev-parse", "HEAD"))

    # c. static scan
    root = os.path.join(SINTEL, "training", "final")
    rows = []
    for name in sorted(os.listdir(root)):
        s = SintelSeq(SINTEL, name, "final")
        n_valid = n_static = 0
        for t in range(1, s.n_frames):
            f = s.flow_into(t)
            ok = ~s.bad_mask_into(t)
            mag = np.hypot(f[..., 0], f[..., 1])
            n_valid += int(ok.sum())
            n_static += int((ok & (mag < 0.5)).sum())
        share = n_static / n_valid
        exp = EXPECTED.get(name)
        rows.append({"sequence": name, "n_flow_files": s.n_frames - 1, "valid_px": n_valid,
                     "static_px": n_static, "static_share": round(share, 6),
                     "expected": exp, "abs_diff": None if exp is None else round(abs(share - exp), 6)})
    rows.sort(key=lambda r: -r["static_share"])
    with open(os.path.join(HERE, "static_scan.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['sequence']:12s} {r['static_share']:.4f}  expected {r['expected']}  diff {r['abs_diff']}")
    bad = [r for r in rows if r["abs_diff"] is not None and r["abs_diff"] > TOL]
    if bad:
        print("STOP 0c: static share differs from expected by >", TOL, [r["sequence"] for r in bad])
        sys.exit(1)
    top2 = [r["sequence"] for r in rows if r["sequence"] not in ("alley_1", "ambush_5")][:2]
    seqs = ["alley_1", "ambush_5"] + top2
    share = {r["sequence"]: r["static_share"] for r in rows}

    # d. manifest
    drv = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    man = {
        "start_time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_hash": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "python": sys.version.split()[0], "pyav": av.__version__,
        "ffmpeg": getattr(av, "ffmpeg_version_info", None),
        "ffmpeg_libs": {k: ".".join(map(str, v)) for k, v in av.library_versions.items()},
        "x264_build": x264_build(), "nvidia": drv, "numpy": np.__version__,
        "sequences": seqs, "static_share": {s: share[s] for s in seqs}, "pass": "final",
        "gt": "lib/gt_grid.py forward splat onto frame t grid, 4x4 cells, gt_cover >= 0.5",
    }
    with open(os.path.join(HERE, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print("0d manifest:", json.dumps(man, indent=1))
    print("SEQUENCES:", ", ".join(f"{s} ({share[s]:.4f})" for s in seqs))


if __name__ == "__main__":
    main()
