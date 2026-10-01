"""Sintel sweep analogue of p3/virat/check_mv_threading.py: re-extract every sweep encode with thread_type=None and
compare, frame by frame, with the block records stored by the sweep (blocks/<pass>/<eid>__<seq>.parquet:
x, y, w, h, dir, mv_x, mv_y; frame types from __frames.parquet). Writes a CSV of every arm file
(default p3/sweep/mv_threading_check.csv; the committed CSV is the 2026-10-02 run on the original sweep, summarised in
DEVIATIONS.md 1). Usage: python check_mv_threading.py [out.csv]"""
import glob
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

SWEEP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SWEEP))
from lib.mvs import extract_mvs, motion_prev_to_cur, dst_topleft  # noqa: E402

COLS = ["x", "y", "w", "h", "dirn", "mv_x", "mv_y"]


def key_rows(a):
    return a[np.lexsort(a.T[::-1])] if len(a) else a


def check(f):
    pass_ = os.path.basename(os.path.dirname(f))
    eid, seq = os.path.basename(f)[:-8].split("__")
    b = pd.read_parquet(f, columns=["frame", "x", "y", "w", "h", "dir", "mv_x", "mv_y"])
    b["dirn"] = np.where(b["dir"] == "past", -1, 1)
    fr = pd.read_parquet(f[:-8] + "__frames.parquet", columns=["frame", "ftype", "group"])
    frames, _ = extract_mvs(os.path.join(SWEEP, "encodes", pass_, f"{eid}__{seq}.mp4"), thread_type=None)
    types = [x["pict_type"] for x in frames]
    stored = dict(tuple(b.groupby("frame")))
    diff = []
    for x in frames:
        t = x["index"]
        if x["pict_type"] == "I" or t == 0:
            continue
        m = x["mvs"]
        tl, mv = dst_topleft(m), motion_prev_to_cur(m)
        new = np.column_stack([np.rint(tl[:, 0]), np.rint(tl[:, 1]), m["w"], m["h"],
                               np.where(m["source"] < 0, -1, 1), mv[:, 0], mv[:, 1]]).astype(float)
        old = stored[t][COLS].to_numpy(float) if t in stored else np.zeros((0, 7))
        if old.shape != new.shape or not np.array_equal(key_rows(old), key_rows(new)):
            diff.append(t)
    g = fr.set_index("frame")
    return {"pass": pass_, "encode_id": eid, "sequence": seq, "n_frames": len(frames),
            "types_match": types == fr.sort_values("frame")["ftype"].tolist(),
            "has_B": "B" in types,
            "differing_frames": " ".join(map(str, diff)),
            "differing_ftypes": "".join(g.loc[t, "ftype"] for t in diff),
            "differing_groups": " ".join(g.loc[t, "group"] for t in diff),
            "last4": "".join(types[-4:])}


if __name__ == "__main__":
    files = sorted(f for f in glob.glob(os.path.join(SWEEP, "blocks", "*", "*.parquet"))
                   if not f.endswith("__frames.parquet"))
    with mp.get_context("spawn").Pool(6) as pool:
        rows = pool.map(check, files, chunksize=4)
    df = pd.DataFrame(rows)
    df.to_csv(sys.argv[1] if len(sys.argv) > 1 else os.path.join(SWEEP, "mv_threading_check.csv"), index=False)
    bad = df[df.differing_frames != ""]
    print(f"{len(df)} arm files checked ({df.has_B.sum()} with B-frames); {len(bad)} with differing frames; "
          f"types mismatch {int((~df.types_match).sum())}")
    print(df.groupby("pass").size().to_string())
    if len(bad):
        print(bad.to_string())
