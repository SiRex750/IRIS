"""Check stored VIRAT MV cell arrays against a single-threaded re-extraction (health check, no gate metric).

Found during the analysis self-test: lib.mvs.extract_mvs with thread_type="AUTO" (frame threading, used by the
Sintel sweep and run_virat.py) can export different motion vectors for the LAST frame of a B-frame stream
(Sintel alley_1, x264_qp24_bf2_pb1 / nvenc_qp28_bf2_bqeq: frame 49 differs run to run; single-threaded
decoding is reproducible). This script re-extracts every stored arm with thread_type=None and lists every
frame whose stored mv_q, mv_fut_q or cell_dir differs. Writes mv_threading_check.csv.
With --fix: for every file listed as differing in mv_threading_check.csv, writes the single-threaded arrays
(all other keys copied unchanged) to data/<clip>/mv_fix/<arm>.npz; the collected file is not touched.
"""
import glob
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from lib.mvs import extract_mvs  # noqa: E402
from run_virat import cell_arrays, VIRAT  # noqa: E402


def check(f):
    clip, arm = os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-4]
    z = np.load(f)
    n, Hc, Wc = z["mv_q"].shape
    p = os.path.join(VIRAT, clip + ".mp4") if arm == "native" else os.path.join(HERE, "encodes", clip, arm + ".mp4")
    fr = extract_mvs(p, thread_type=None, max_frames=n + 5)[0][:n]
    ca = cell_arrays(fr, Hc, Wc)
    diff = np.zeros(n, bool)
    for k in ("mv_q", "mv_fut_q", "cell_dir"):
        diff |= (z[k] != ca[k]).reshape(n, -1).any(1)
    ft = [str(x) for x in z["ftype"]]
    return {"clip": clip, "arm": arm, "n_frames": n, "types_match": ft == ca["types"],
            "differing_frames": " ".join(map(str, np.nonzero(diff)[0])),
            "differing_ftypes": "".join(ft[i][0] for i in np.nonzero(diff)[0]), "last4": "".join(t[0] for t in ft[-4:])}


def fix(clip, arm):
    src = os.path.join(HERE, "data", clip, arm + ".npz")
    z = dict(np.load(src))
    n, Hc, Wc = z["mv_q"].shape
    p = os.path.join(VIRAT, clip + ".mp4") if arm == "native" else os.path.join(HERE, "encodes", clip, arm + ".mp4")
    ca = cell_arrays(extract_mvs(p, thread_type=None, max_frames=n + 5)[0][:n], Hc, Wc)
    for k in ("mv_q", "mv_fut_q", "cell_dir"):
        z[k] = ca[k]
    os.makedirs(os.path.join(HERE, "data", clip, "mv_fix"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "data", clip, "mv_fix", arm + ".npz"), **z)
    return clip, arm


if __name__ == "__main__" and "--fix" in sys.argv:
    df = pd.read_csv(os.path.join(HERE, "mv_threading_check.csv")).fillna("")
    for c, a in df.loc[df.differing_frames != "", ["clip", "arm"]].itertuples(index=False):
        print("fixed", *fix(c, a))
elif __name__ == "__main__":
    files = sorted(f for f in glob.glob(os.path.join(HERE, "data", "*", "*.npz"))
                   if os.path.basename(f) not in ("source.npz", "raft.npz"))
    with mp.get_context("spawn").Pool(6) as pool:
        rows = pool.map(check, files, chunksize=4)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "mv_threading_check.csv"), index=False)
    bad = df[df.differing_frames != ""]
    print(f"{len(df)} arm files checked; {len(bad)} with differing frames; types mismatch {int((~df.types_match).sum())}")
    if len(bad):
        print(bad.to_string())
