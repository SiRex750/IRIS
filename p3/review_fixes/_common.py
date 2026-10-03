"""Shared helpers for review follow-ups B and C: Sintel GT on frame t's 4x4 cell grid, exactly as the sweep's
load_seq (p3/sweep/run_sweep.py) builds it: lib.gt_grid.GTGrid(flow_into(t), bad_mask_into(t)).cell_gt(),
valid = gt_cover >= 0.5 (run_pilot.COVER_MIN), |GT| = hypot(cgx, cgy). GT is pass-independent."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, P3)
sys.path.insert(0, os.path.join(P3, "pilot"))

from lib.flo import SintelSeq  # noqa: E402
from lib.gt_grid import GTGrid  # noqa: E402

SINTEL = r"C:\Users\akash\Documents\datasets\MPI-Sintel"
C = 4
COVER_MIN = 0.5
MOVING_GT = 2.0
CACHE = os.path.join(os.environ.get("REVIEW_CACHE", os.path.join(HERE, "_cache")))


def sequences():
    import pandas as pd
    r = pd.read_csv(os.path.join(P3, "sweep", "results.csv"), usecols=["pass", "sequence"])
    return sorted(r.loc[r["pass"] == "final", "sequence"].unique())


def gt_cells(seq):
    """{t: dict(cgx, cgy, valid, gmag)} for t = 1..n-1, flat (Hc*Wc,) arrays; cached as npz (not committed)."""
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f"gt_{seq}.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        n = int(z["n"])
        return {t: {"cgx": z["cgx"][t - 1], "cgy": z["cgy"][t - 1], "valid": z["valid"][t - 1],
                    "gmag": z["gmag"][t - 1]} for t in range(1, n)}, (int(z["Hc"]), int(z["Wc"]))
    s = SintelSeq(SINTEL, seq, "final")
    out = {}
    for t in range(1, s.n_frames):
        g = GTGrid(s.flow_into(t), s.bad_mask_into(t))
        cgx, cgy, ccov, _ = g.cell_gt()
        cgx, cgy = cgx.ravel(), cgy.ravel()
        out[t] = {"cgx": cgx, "cgy": cgy, "valid": ccov.ravel() >= COVER_MIN, "gmag": np.hypot(cgx, cgy)}
    shp = (g.Hc, g.Wc)
    ts = range(1, s.n_frames)
    np.savez_compressed(fn, n=s.n_frames, Hc=shp[0], Wc=shp[1],
                        **{k: np.stack([out[t][k] for t in ts]) for k in ("cgx", "cgy", "valid", "gmag")})
    return out, shp
