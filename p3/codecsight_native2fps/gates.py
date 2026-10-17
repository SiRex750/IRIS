"""RAFT-truth gates of PREREGISTRATION.md (amended): V0 (§5) and the drift gate (§3), per clip. Code only; run on
collected data by the analysis script. Neither function writes anything. The clip-level pass/fail is decided by the
analysis (ceil(0.75 N) of the counted clips); this module returns the per-clip numbers it is decided from.

Data layout (collect.py): <root>/data/<clip>/raft.npz  (row i-1 = native pair i-1 -> i, i = 1..L-1, window-relative)
                          <root>/data/<clip>/<arm>.npz, <root>/encodes/<clip>/<arm>.mp4
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "virat"))

import analyze_virat as av_  # noqa: E402
from analyze_virat import frame_groups, cell_texture, cell_vectors, TEX_BIN, TEX_MAX, STALE_GT, COVER_COUNT  # noqa: E402
from lib.mvs import extract_mvs  # noqa: E402
from run_virat import cell_arrays, decode_source, VIRAT  # noqa: E402

QUIET_STEP = 0.5   # px, drift gate §3: per-frame native step below this
FALSE_MOVE = 2.0   # px, |S| above this = "truly moved" (STALE_GT)
DRIFT_RATE = 0.01
V0_ERR = 1.5


def need(n, frac):
    import math
    return math.ceil(frac * n - 1e-9)


def load_truth(clip, root):
    """T (L-1, ncell, 2) float32 splatted RAFT per native pair, ok (L-1, ncell): cover rule and finite."""
    rz = np.load(os.path.join(root, "data", clip, "raft.npz"))
    sp = rz["splat_t"].astype(np.float32)
    nm1, Hc, Wc = sp.shape[:3]
    sp = sp.reshape(nm1, Hc * Wc, 2)
    ok = (rz["splat_count"].reshape(nm1, -1) >= COVER_COUNT) & np.isfinite(sp).all(-1)
    return sp, ok, Hc, Wc


def drift_false_moves(T, ok, k, keyint):
    """Drift gate §3 counts for one clip. Cell-frames: sampled frames s = r + m k (m >= 1, s not a refresh) of the
    GOP starting at refresh r (native, window-relative, multiples of keyint); cells valid at every step r+1..s; quiet =
    every step |T| < 0.5 px; false move = quiet and |S| > 2 px, S the vector sum of the steps at the fixed cell.
    Returns (quiet_cellframes, false_move_cellframes)."""
    L = T.shape[0] + 1
    mag = np.hypot(T[..., 0], T[..., 1])
    quiet_step = ok & (mag < QUIET_STEP)
    n_quiet = n_false = 0
    for r in range(0, L, keyint):
        S = np.zeros(T.shape[1:], np.float64)
        allq = np.ones(T.shape[1], bool)
        for i in range(r + 1, min(r + keyint - 1, L - 1) + 1):
            S += np.where(ok[i - 1][:, None], T[i - 1], 0.0)
            allq &= quiet_step[i - 1]
            if (i - r) % k == 0:
                n_quiet += int(allq.sum())
                n_false += int((allq & (np.hypot(S[:, 0], S[:, 1]) > FALSE_MOVE)).sum())
    return n_quiet, n_false


def v0_clip(clip, root, win_first, L, ref_arm="x264_crf12"):
    """V0 inputs for one clip (§5, as analyze_confirm.do_clip): texture histogram of all cells of window frames
    1..L-1, and for every valid d1 cell with |RAFT| > 2 px and a vector in the CRF 12 encode: its texture and
    |MV - RAFT| (vectors re-extracted, single-threaded)."""
    sp, ok, Hc, Wc = load_truth(clip, root)
    ncell = Hc * Wc
    n = L
    valid = np.zeros((n, ncell), bool)
    gmag = np.full((n, ncell), np.nan, np.float32)
    valid[1:] = ok
    gmag[1:] = np.hypot(sp[..., 0], sp[..., 1])
    src, _ = decode_source(os.path.join(VIRAT, clip + ".mp4"), n, start=win_first)
    H = src[0].shape[0] * 2 // 3
    tex = np.zeros((n, ncell), np.float32)
    for t in range(1, n):
        tex[t] = cell_texture(src[t][:H].astype(np.float64)).ravel()
    del src
    tex_hist = np.bincount(np.minimum((tex[1:] / TEX_BIN).astype(np.int64), int(TEX_MAX / TEX_BIN) - 1).ravel(),
                           minlength=int(TEX_MAX / TEX_BIN))
    z = np.load(os.path.join(root, "data", clip, f"{ref_arm}.npz"))
    mv_q = z["mv_q"].reshape(n, -1)
    ftype = [str(x) for x in z["ftype"]]
    fr, _ = extract_mvs(os.path.join(root, "encodes", clip, f"{ref_arm}.mp4"))
    vec = cell_vectors(fr, Hc, Wc)
    q_mismatch = int((cell_arrays(fr, Hc, Wc)["mv_q"].reshape(len(fr), -1) != mv_q).sum())
    d1 = frame_groups(ftype, z["d_past"], bool(z["d_known"])).get("d1", np.array([], int))
    sel = np.zeros((n, ncell), bool)
    sel[d1] = valid[d1] & (gmag[d1] > STALE_GT) & (mv_q[d1] != 255)
    tt, cc = np.nonzero(sel)
    err = np.hypot(vec[tt, cc, 0] - sp[tt - 1, cc, 0], vec[tt, cc, 1] - sp[tt - 1, cc, 1]).astype(np.float32)
    return {"tex_hist": tex_hist, "v0_tex": tex[tt, cc], "v0_err": err, "q_mismatch": q_mismatch}


def tex_q75(hists):
    cum = np.cumsum(np.sum(hists, axis=0))
    return (np.searchsorted(cum, 0.75 * cum[-1]) + 1) * TEX_BIN


def v0_median(v0_tex, v0_err, q75):
    m = v0_tex >= q75
    return float(np.median(v0_err[m])) if m.any() else np.nan
