"""Paper 3: VIRAT analysis, as fixed by p3/virat/PREREGISTRATION.md (commit 45f3e98). Outputs in p3/virat/:
results.csv (clip x arm x tau x group), v0_cells_summary.csv (per clip V0 numbers), summary.md.

Reuse of the Sintel code:
  - cells, |MV| per cell, vector choice (past overrides future), d, groups: already stored by run_virat.py
    with run_pilot_v2.run_one's mapping. The gate "|MV| < tau, no vector = recompute" is applied to the stored
    q = floor(4|MV|) as  q < 4*tau  (exact for tau in {0.5, 1, 2}; 255 = no vector is never < 4*tau).
  - per (arm, group, tau) counters: the accumulator of run_pilot_v2.run_one (valid, reuse on valid, stale,
    GT-static, missed, flip vs x264 CRF 12 on that arm's d1 frames), re-implemented on arrays in
    gate_counts() because run_one is fused with Sintel encoding and Sintel GT. --selftest runs gate_counts on
    Sintel sweep encodes with the Sintel GT and compares every counter with p3/sweep/results.csv.
Changed for VIRAT, and why:
  - GT: RAFT-estimated flow forward-splatted onto frame t's grid (raft.npz splat_t, lib.gt_grid), valid if
    cover = count/16 >= 0.5. There is no occlusion/invalid mask (Sintel's bad mask has no RAFT equivalent).
  - groups: "d1(assumed)" for P-frames of an encode whose d is unknown (NATIVE VIRAT_S_000002), as run_one did
    for d_known = False arms; B-frame group is the naive (unscaled) vector, as pre-registered.
  - V0 needs the MV vector, but the data files store |MV| only: V0 re-extracts the x264 CRF 12 vectors from the
    saved encode (lib.mvs.extract_mvs) and paints them with the same mapping; the repainted |MV| is checked
    against the stored q on every cell.
  - texture: cell_texture() of p3/pilot_v2/texture_check/texture_check.py (mean np.gradient magnitude of source
    luma over the cell), on the decoded source Y of frame t. Top-quartile threshold = 75th percentile over all
    cells of source frames 1..299 of all 54 clips (0.001-wide histogram).
  - pixel-change proxy: exact per-cell mean |Y_t - Y_(t-1)| recomputed from the decoded source (the stored
    uint8 is rounded); cross-checked against source.npz.
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
REPO = os.path.dirname(P3)
for p in (HERE, P3, os.path.join(P3, "sweep"), os.path.join(P3, "pilot_v2"), os.path.join(P3, "pilot"),
          os.path.join(P3, "validate")):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.mvs import extract_mvs, motion_prev_to_cur, dst_topleft  # noqa: E402
from run_pilot import paint  # noqa: E402
from run_virat import cell_arrays, decode_source, C, OUTLIER, VIRAT  # noqa: E402

PREREG = "45f3e988b37a21babdbb4b30f1f62422c9b82fb4"
DATA = os.path.join(HERE, "data")
ENC = os.path.join(HERE, "encodes")
TAUS = (0.5, 1.0, 2.0)
TAU = 1.0
STALE_GT, STATIC_GT, COVER_COUNT = 2.0, 0.5, 8  # cover >= 0.5  <=>  count >= 8 of 16
ELIG_MIN = 0.001
PIX = 8.0
TEX_BIN, TEX_MAX = 0.001, 200.0
V0_ERR = 1.5
DELTA = 0.0005
CRFS = (12, 18, 23, 28, 33, 38, 45)
NQPS = (18, 23, 28, 33, 38, 45)
REF = "x264_crf12"
B_ARMS = ("x264_qp24_bf2_pb1", "nvenc_qp28_bf2", "nvenc_qp28_bf2_bqeq")


def scene_of(clip):
    return "VIRAT_S_" + clip.split("_")[2]


# --------------------------------------------------------------------------- metric core
def frame_groups(ftype, d_past, d_known):
    """run_one's grouping: I-frames and frame 0 excluded; P -> d1 / dgt1 (d1(assumed) if d unknown); else B."""
    g = {}
    for t, ft in enumerate(ftype):
        if ft == "I" or t == 0:
            continue
        if ft == "P":
            key = ("d1" if d_past[t] == 1 else "dgt1") if d_known else "d1(assumed)"
        else:
            key = "B"
        g.setdefault(key, []).append(t)
    return {k: np.array(v) for k, v in g.items()}


def gate_counts(mv_q, groups, valid, gmag, ref=None, pix=None, taus=TAUS):
    """mv_q (n, ncell) uint8; valid / gmag / pix indexed [t] with t = display index (row 0 unused).
    ref: {tau: (d1 frame mask (n,), reuse (n, ncell))} of the flip reference. Returns list of dicts."""
    ncell = mv_q.shape[1]
    out = []
    for key, T in groups.items():
        q, v, g = mv_q[T], valid[T], gmag[T]
        moving = v & (g > STALE_GT)
        static = v & (g < STATIC_GT)
        pc = pix[T] if pix is not None else None
        for tau in taus:
            reuse = q < 4 * tau
            r = {"group": key, "tau": tau, "n_frames": len(T), "inframe_cells": len(T) * ncell,
                 "valid_cells": int(v.sum()), "reuse_inframe_cells": int(reuse.sum()),
                 "reuse_cells": int((reuse & v).sum()), "stale_cells": int((reuse & moving).sum()),
                 "moving_cells": int(moving.sum()), "gt_static_cells": int(static.sum()),
                 "missed_cells": int((static & ~reuse).sum())}
            if ref is not None:
                m, rd = ref[tau]
                sel = m[T]
                r["flip_cells"] = int((reuse[sel] != rd[T[sel]]).sum())
                r["flip_den"] = int(sel.sum()) * ncell
            if pc is not None:
                r["pix_cells"] = int((reuse & pc).sum())
            out.append(r)
    return out


def finish(r):
    V, R = r["valid_cells"], r["reuse_cells"]
    r["stale"] = r["stale_cells"] / V if V else np.nan
    r["missed"] = r["missed_cells"] / V if V else np.nan
    r["stale_of_reuse"] = r["stale_cells"] / R if R else np.nan
    r["reuse_share_inframe"] = r["reuse_inframe_cells"] / r["inframe_cells"] if r["inframe_cells"] else np.nan
    r["flip"] = r["flip_cells"] / r["flip_den"] if r.get("flip_den") else np.nan
    if "pix_cells" in r:
        r["pix_of_inframe"] = r["pix_cells"] / r["inframe_cells"] if r["inframe_cells"] else np.nan
        RI = r["reuse_inframe_cells"]
        r["pix_of_reuse"] = r["pix_cells"] / RI if RI else np.nan
    return r


def ref_decisions(mv_q, groups):
    n = mv_q.shape[0]
    m = np.zeros(n, bool)
    if "d1" in groups:
        m[groups["d1"]] = True
    return {tau: (m, mv_q < 4 * tau) for tau in TAUS}


# --------------------------------------------------------------------------- texture / V0
def cell_texture(y):
    """p3/pilot_v2/texture_check/texture_check.py, unchanged."""
    gy, gx = np.gradient(y)
    g = np.hypot(gx, gy)
    H, W = g.shape
    return g.reshape(H // C, C, W // C, C).mean(axis=(1, 3))


def cell_vectors(frames, Hc, Wc):
    """(n, ncell, 2) motion t-1 -> t per cell with run_one's mapping (NaN = no vector)."""
    n, ncell = len(frames), Hc * Wc
    out = np.full((n, ncell, 2), np.nan, np.float32)
    for fr in frames:
        t, m = fr["index"], fr["mvs"]
        if fr["pict_type"] == "I" or t == 0 or m.size == 0:
            continue
        tl = dst_topleft(m)
        x0 = np.rint(tl[:, 0]).astype(np.int64)
        y0 = np.rint(tl[:, 1]).astype(np.int64)
        w, h = m["w"].astype(np.int64), m["h"].astype(np.int64)
        src = m["source"].astype(np.int64)
        mv = motion_prev_to_cur(m)
        for sgn in (1, -1):
            sel = np.nonzero(src == sgn)[0]
            if sel.size == 0:
                continue
            ci, bi = paint(Hc, Wc, x0[sel], y0[sel], w[sel], h[sel])
            out[t, ci] = mv[sel][bi]
    return out


# --------------------------------------------------------------------------- one clip
MV_FIX = False  # set per run: use data/<clip>/mv_fix/<arm>.npz where present (see DEVIATIONS.md)


def load_arm(clip, arm):
    fx = os.path.join(DATA, clip, "mv_fix", f"{arm}.npz")
    z = np.load(fx if MV_FIX and os.path.exists(fx) else os.path.join(DATA, clip, f"{arm}.npz"))
    n = z["mv_q"].shape[0]
    return {"mv_q": z["mv_q"].reshape(n, -1), "ftype": [str(x) for x in z["ftype"]], "d_past": z["d_past"],
            "d_known": bool(z["d_known"]), "meta": json.loads(str(z["meta"]))}


def do_clip(clip):
    t0 = time.perf_counter()
    notes = []
    arms = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DATA, clip, "*.npz"))
                  if os.path.basename(p) not in ("source.npz", "raft.npz"))
    rz = np.load(os.path.join(DATA, clip, "raft.npz"))
    sp = rz["splat_t"].astype(np.float32)
    nm1, Hc, Wc = sp.shape[:3]
    n, ncell = nm1 + 1, Hc * Wc
    valid = np.zeros((n, ncell), bool)
    gmag = np.full((n, ncell), np.nan, np.float32)
    valid[1:] = rz["splat_count"].reshape(nm1, ncell) >= COVER_COUNT
    gmag[1:] = np.hypot(sp[..., 0], sp[..., 1]).reshape(nm1, ncell)
    del sp
    n_valid = int(valid[1:].sum())
    n_mov = int((valid[1:] & (gmag[1:] > STALE_GT)).sum())
    elig_share = n_mov / n_valid if n_valid else np.nan

    # source: texture of frame t and exact mean |Y_t - Y_(t-1)|
    src, _ = decode_source(os.path.join(VIRAT, clip + ".mp4"), n)
    H = src[0].shape[0] * 2 // 3
    tex = np.zeros((n, ncell), np.float32)
    absd = np.zeros((n, ncell), np.float32)
    yp = src[0][:H].astype(np.float64)
    for t in range(1, n):
        y = src[t][:H].astype(np.float64)
        tex[t] = cell_texture(y).ravel()
        absd[t] = np.abs(y - yp).reshape(H // C, C, -1, C).mean(axis=(1, 3)).ravel()
        yp = y
    del src, y, yp
    tex_hist = np.bincount(np.minimum((tex[1:] / TEX_BIN).astype(np.int64), int(TEX_MAX / TEX_BIN) - 1).ravel(),
                           minlength=int(TEX_MAX / TEX_BIN))
    stored = np.load(os.path.join(DATA, clip, "source.npz"))["absdiff"].reshape(nm1, ncell)
    ad_mismatch = int((np.minimum(np.floor(absd[1:] + 0.5), 255) != stored).sum())
    pix = absd > PIX

    # flip reference and arms
    ref_arm = load_arm(clip, REF)
    ref_groups = frame_groups(ref_arm["ftype"], ref_arm["d_past"], ref_arm["d_known"])
    ref = ref_decisions(ref_arm["mv_q"], ref_groups)
    rows = []
    for arm in arms:
        a = ref_arm if arm == REF else load_arm(clip, arm)
        if a["mv_q"].shape != (n, ncell):
            notes.append(f"{arm}: mv_q shape {a['mv_q'].shape} != {(n, ncell)}")
            continue
        grp = frame_groups(a["ftype"], a["d_past"], a["d_known"])
        for r in gate_counts(a["mv_q"], grp, valid, gmag, ref=ref, pix=pix):
            rows.append(finish({"clip": clip, "arm": arm, "codec": a["meta"].get("codec"), "d_known": a["d_known"],
                                **r}))

    # V0 cells: x264 CRF 12, d1 frames, valid, |RAFT| > 2, has a vector
    fr, _ = extract_mvs(os.path.join(ENC, clip, f"{REF}.mp4"), thread_type=None)
    vec = cell_vectors(fr, Hc, Wc)
    q_re = cell_arrays(fr, Hc, Wc)["mv_q"].reshape(len(fr), -1)
    q_mismatch = int((q_re != ref_arm["mv_q"]).sum())
    del fr, q_re
    d1 = ref_groups.get("d1", np.array([], int))
    sel = np.zeros((n, ncell), bool)
    sel[d1] = valid[d1] & (gmag[d1] > STALE_GT) & (ref_arm["mv_q"][d1] != 255)
    rz_flow = rz["splat_t"].reshape(nm1, ncell, 2)
    tt, cc = np.nonzero(sel)
    err = np.hypot(vec[tt, cc, 0] - rz_flow[tt - 1, cc, 0].astype(np.float32),
                   vec[tt, cc, 1] - rz_flow[tt - 1, cc, 1].astype(np.float32)).astype(np.float32)
    v0_nan = int(np.isnan(err).sum())
    return {"clip": clip, "rows": rows, "valid_cellframes": n_valid, "moving_cellframes": n_mov,
            "elig_share": elig_share, "tex_hist": tex_hist, "v0_tex": tex[tt, cc], "v0_err": err,
            "checks": {"absdiff_rounding_mismatch_cells": ad_mismatch, "crf12_q_repaint_mismatch_cells": q_mismatch,
                       "v0_nan_err": v0_nan, "n_arms": len(arms)},
            "notes": notes, "run_s": time.perf_counter() - t0}


def do_clip_safe(args):
    global MV_FIX
    clip, MV_FIX = args
    try:
        return do_clip(clip)
    except Exception as e:
        return {"clip": clip, "error": repr(e), "traceback": traceback.format_exc()}


# --------------------------------------------------------------------------- self-test on Sintel
def selftest():
    from run_sweep import load_seq
    seq = "alley_1"
    sq = load_seq(seq, "final")
    n, Hc, Wc = len(sq["imgs"]), sq["Hc"], sq["Wc"]
    ncell = Hc * Wc
    valid = np.zeros((n, ncell), bool)
    gmag = np.full((n, ncell), np.nan, np.float64)
    for t in range(1, n):
        g = sq["gt"][t]
        valid[t] = g["ccov"].ravel() >= 0.5
        gmag[t] = np.hypot(g["cgx"], g["cgy"]).ravel()

    def arrays(arm):
        fr, _ = extract_mvs(os.path.join(P3, "sweep", "encodes", "final", f"{arm}__{seq}.mp4"))
        ca = cell_arrays(fr, Hc, Wc)
        return ca["mv_q"].reshape(n, ncell), frame_groups(ca["types"], ca["d_past"], True)

    rq, rg = arrays(REF)
    ref = ref_decisions(rq, rg)
    res = pd.read_csv(os.path.join(P3, "sweep", "results.csv"))
    res = res[(res["pass"] == "final") & (res["sequence"] == seq)]
    name = {"d1": "d1", "dgt1": "dgt1_naive", "B": "B_naive"}
    cols = {"valid_cells": "valid_cells", "reuse_cells": "reuse_cells", "stale_cells": "stale_cells",
            "gt_static_cells": "gt_static_cells", "missed_cells": "missed_cells", "flip_cells": "flip_x264ref_cells",
            "flip_den": "flip_x264ref_den", "n_frames": "n_group_frames"}
    bad, expected, checked = [], [], 0
    for arm in ("x264_crf12", "x264_crf23", "x264_crf45", "nvenc_qp45", "mpeg4_q31", "x264_qp24_bf2_pb1",
                "nvenc_qp28_bf2_bqeq"):
        q, g = arrays(arm)
        for r in gate_counts(q, g, valid, gmag, ref=ref):
            s = res[(res.encode_id == arm) & (res.group == name[r["group"]]) & (res.tau == r["tau"])]
            if len(s) != 1:
                bad.append((arm, r["group"], r["tau"], "no sweep row"))
                continue
            for k, c in cols.items():
                checked += 1
                if int(s.iloc[0][c]) != r[k]:
                    m = (arm, r["group"], r["tau"], k, r[k], int(s.iloc[0][c]))
                    if arm == REF and k == "flip_den":
                        expected.append(m + ("reference arm: sweep stores flip den 0 for itself (flip = 0 either way)",))
                    elif r["group"] == "d1" and arm in B_ARMS:
                        expected.append(m + ("last frame of a B stream, frame-threaded MV export (DEVIATIONS.md)",))
                    else:
                        bad.append(m)
    return {"sequence": seq, "values_checked": checked, "mismatches": bad, "expected": expected}


# --------------------------------------------------------------------------- verdicts / summary
def need(n, frac):
    return math.ceil(frac * n - 1e-9)


def verdicts(res, clips_all, elig, v0med, tau=TAU):
    """Mechanical pre-registered verdicts from results.csv rows (res) over the given clip lists."""
    r = res[res.tau == tau]

    def val(arm, group, col):
        s = r[(r.arm == arm) & (r.group == group)].set_index("clip")[col]
        return s

    out = []
    e_ok = len(elig) >= 20
    v0c = [c for c in elig if c in v0med.index]
    v0_pass_clips = int((v0med.reindex(elig) < V0_ERR).sum())
    v0_pass = v0_pass_clips >= need(len(elig), 0.75) and len(elig) > 0
    out.append({"prediction": "V0 RAFT validity: median |MV - RAFT| < 1.5 px (x264 CRF 12, |RAFT| > 2, top-quartile "
                              "texture)", "n": len(elig), "value": f"{v0_pass_clips}/{len(elig)} clips "
                              f"(median of clip medians {np.nanmedian(v0med.reindex(v0c)):.2f} px)",
                "threshold": f">= 75% ({need(len(elig), 0.75)})", "verdict": "PASS" if v0_pass else "FAIL"})

    def raft_verdict(ok):
        if not e_ok:
            return "descriptive only (< 20 eligible)"
        if not v0_pass:
            return "no verdict (V0 failed)" + (" [would PASS]" if ok else " [would FAIL]")
        return "PASS" if ok else "FAIL"

    for lab, hi, lo, frac in (("V1-stale x264: STALE(CRF 45) - STALE(CRF 12) > 0.0005", "x264_crf45", "x264_crf12", .75),
                              ("V1-stale NVENC: STALE(QP 45) - STALE(QP 18) > 0.0005", "nvenc_qp45", "nvenc_qp18", .75)):
        d = (val(hi, "d1", "stale") - val(lo, "d1", "stale")).reindex(elig)
        k = int((d > DELTA).sum())
        out.append({"prediction": lab, "n": len(elig), "value": f"{k}/{len(elig)} clips (median diff {d.median():.5f})",
                    "threshold": f">= {int(frac * 100)}% ({need(len(elig), frac)})",
                    "verdict": raft_verdict(k >= need(len(elig), frac))})
    N = len(clips_all)
    d = (val("x264_crf45", "d1", "flip") - val("x264_crf18", "d1", "flip")).reindex(clips_all)
    k = int((d > 0).sum())
    out.append({"prediction": "V1-flip x264: FLIP(CRF 45) > FLIP(CRF 18)", "n": N,
                "value": f"{k}/{N} clips (median diff {d.median():.4f})", "threshold": f">= 75% ({need(N, .75)})",
                "verdict": "PASS" if k >= need(N, .75) else "FAIL"})
    fl = pd.DataFrame({c: val(f"x264_crf{c}", "d1", "flip") for c in CRFS[1:]}).reindex(clips_all)
    rho = fl.apply(lambda row: spearmanr(CRFS[1:], row.values).statistic if row.notna().all() and row.nunique() > 1
                   else np.nan, axis=1)
    k = int((rho >= 0.8).sum())
    out.append({"prediction": "V1-flip x264: Spearman(CRF, FLIP) over CRF 18..45 >= 0.8", "n": N,
                "value": f"{k}/{N} clips (median rho {rho.median():.3f})", "threshold": f">= 70% ({need(N, .70)})",
                "verdict": "PASS" if k >= need(N, .70) else "FAIL"})
    d = (val("nvenc_qp45", "d1", "flip") - val("nvenc_qp18", "d1", "flip")).reindex(clips_all)
    k = int((d > 0).sum())
    out.append({"prediction": "V1-flip NVENC: FLIP(QP 45) > FLIP(QP 18)", "n": N,
                "value": f"{k}/{N} clips (median diff {d.median():.4f})", "threshold": f">= 75% ({need(N, .75)})",
                "verdict": "PASS" if k >= need(N, .75) else "FAIL"})
    for lab, b, p, frac in (("V2 x264: STALE(B, QP 24 bf 2) - STALE(QP 24 bf 0, d1) > 0.0005", "x264_qp24_bf2_pb1",
                             "x264_qp24", .75),
                            ("V2 NVENC: STALE(B, QP 28 bf 2 B=P) - STALE(QP 28 bf 0, d1) > 0.0005",
                             "nvenc_qp28_bf2_bqeq", "nvenc_qp28", .65)):
        d = (val(b, "B", "stale") - val(p, "d1", "stale")).reindex(elig)
        k = int((d > DELTA).sum())
        out.append({"prediction": lab, "n": len(elig), "value": f"{k}/{len(elig)} clips (median diff {d.median():.5f})",
                    "threshold": f">= {int(frac * 100)}% ({need(len(elig), frac)})",
                    "verdict": raft_verdict(k >= need(len(elig), frac))})
    return pd.DataFrame(out), rho


def md_table(df, floatfmt="{:.4f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("–" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--clips", default="")
    ap.add_argument("--cache", default="", help="pickle of per-clip outputs (written if absent, read if present)")
    ap.add_argument("--mv-fix", action="store_true",
                    help="use data/<clip>/mv_fix/ single-threaded arrays where present (primary; DEVIATIONS.md)")
    a = ap.parse_args()
    sfx = "" if a.mv_fix else "_as_collected"
    variant = ("MV data: collected arrays, with the 9 frame-299 records of B-frame streams replaced by single-threaded "
               "re-extraction (data/<clip>/mv_fix/, DEVIATIONS.md 1). PRIMARY run." if a.mv_fix else
               "MV data: exactly as collected (frame-threaded extraction). Secondary run (DEVIATIONS.md 1).")
    T0 = time.perf_counter()
    print("self-test on Sintel (alley_1) ...", flush=True)
    st = selftest()
    print(f"self-test: {st['values_checked']} counters checked, {len(st['mismatches'])} mismatches", flush=True)
    t_self = time.perf_counter() - T0

    man = json.load(open(os.path.join(HERE, "manifest.json")))
    clips = sorted(man["clips"])
    if a.clips:
        clips = [c for c in clips if c in a.clips.split(",")]
    order = sorted(clips, key=lambda c: c != OUTLIER)
    outs, fails = {}, []
    if a.cache and os.path.exists(a.cache):  # per-clip outputs of an identical earlier invocation
        outs, fails = pickle.load(open(a.cache, "rb"))
    else:
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for o in pool.imap_unordered(do_clip_safe, [(c, a.mv_fix) for c in order]):
                if "error" in o:
                    fails.append(o)
                    print(f"FAILED {o['clip']}: {o['error']}", flush=True)
                else:
                    outs[o["clip"]] = o
                    print(f"{o['clip']} done in {o['run_s']:.0f}s", flush=True)
        if a.cache:
            pickle.dump((outs, fails), open(a.cache, "wb"))
    ok_clips = [c for c in clips if c in outs]

    # eligibility (RAFT + source only)
    el = pd.DataFrame({"clip": ok_clips, "valid_cellframes": [outs[c]["valid_cellframes"] for c in ok_clips],
                       "moving_cellframes": [outs[c]["moving_cellframes"] for c in ok_clips],
                       "moving_share": [outs[c]["elig_share"] for c in ok_clips]})
    el["eligible"] = el["moving_share"] >= ELIG_MIN
    elig = el.loc[el.eligible, "clip"].tolist()

    # V0
    hist = sum(outs[c]["tex_hist"] for c in ok_clips)
    cum = np.cumsum(hist)
    q75 = (np.searchsorted(cum, 0.75 * cum[-1]) + 1) * TEX_BIN
    v0 = []
    for c in ok_clips:
        m = outs[c]["v0_tex"] >= q75
        e = outs[c]["v0_err"][m]
        v0.append({"clip": c, "v0_candidate_cells": int(outs[c]["v0_err"].size), "v0_cells": int(m.sum()),
                   "median_err_px": float(np.nanmedian(e)) if e.size else np.nan})
    v0 = pd.DataFrame(v0)
    v0med = v0.set_index("clip")["median_err_px"]

    res = pd.DataFrame([r for c in ok_clips for r in outs[c]["rows"]])
    grp_map = {c: man["clips"][c]["group"] for c in clips}
    res.insert(1, "scene", res["clip"].map(scene_of))
    res.insert(2, "source_group", res["clip"].map(grp_map))
    res.insert(3, "eligible", res["clip"].isin(elig))
    res.to_csv(os.path.join(HERE, f"results{sfx}.csv"), index=False)
    v0.merge(el, on="clip").to_csv(os.path.join(HERE, f"v0_eligibility{sfx}.csv"), index=False)

    vt, rho = verdicts(res, ok_clips, elig, v0med)
    no2 = [c for c in ok_clips if c != OUTLIER]
    vt2, _ = verdicts(res, no2, [c for c in elig if c != OUTLIER], v0med)
    vt.to_csv(os.path.join(HERE, f"verdicts{sfx}.csv"), index=False)
    vt2.to_csv(os.path.join(HERE, f"verdicts_without_000002{sfx}.csv"), index=False)

    # ---------------- summary.md
    r1 = res[res.tau == TAU]

    def col(arm, group, c, clipset=None):
        s = r1[(r1.arm == arm) & (r1.group == group)].set_index("clip")[c]
        return s.reindex(clipset or ok_clips)

    L = []
    L.append("# Paper 3 — VIRAT analysis\n")
    L.append(f"Pre-registration `{PREREG}` (p3/virat/PREREGISTRATION.md). Data: runner `{man['runner_commit']}`, "
             f"collected {man['start_time']} → {man.get('end_time')}. Analysis code run at `{git('rev-parse', 'HEAD')}`"
             f"{' (uncommitted: ' + git('status', '--porcelain', '--', 'p3/virat/analyze_virat.py') + ')' if git('status', '--porcelain', '--', 'p3/virat/analyze_virat.py') else ''}.")
    L.append(variant + "\n")
    L.append("All flow-based quantities use RAFT-estimated flow (torchvision raft_large, DEFAULT weights), not ground "
             "truth. STALE and missed reuse are therefore RAFT-estimated. tau = 1 px unless stated. Every value is "
             "per clip; tables show counts of clips or medians over clips (no pooled means).\n")
    L.append("## 1. Eligibility and V0\n")
    L.append(f"Eligible clips (RAFT-estimated |flow| > 2 px in >= 0.1% of valid cell-frames): **{len(elig)}/{len(ok_clips)}**"
             f" (without VIRAT_S_000002: {len([c for c in elig if c != OUTLIER])}/{len(no2)}). "
             f"{'>= 20, so V1-stale and V2 carry verdicts (subject to V0).' if len(elig) >= 20 else '< 20: V1-stale and V2 are descriptive only.'}\n")
    L.append(f"Ineligible clips: {', '.join(c for c in ok_clips if c not in elig) or 'none'}.\n")
    L.append(f"V0: texture threshold (75th percentile of cell texture over all cells of source frames 1..299 of all "
             f"{len(ok_clips)} clips) = {q75:.3f} grey levels/px. {vt.iloc[0]['value']} have median |MV − RAFT| < 1.5 px "
             f"→ **V0 {vt.iloc[0]['verdict']}**.\n")
    L.append("## 2. Prediction verdict table (tau = 1 px)\n")
    L.append("All 54 clips:\n")
    L.append(md_table(vt))
    L.append("\nWithout VIRAT_S_000002 (texture threshold unchanged):\n")
    L.append(md_table(vt2))
    L.append("")
    other = os.path.join(HERE, "verdicts_as_collected.csv")
    if a.mv_fix and os.path.exists(other):
        oc = pd.read_csv(other)
        same = oc[["prediction", "value", "verdict"]].equals(vt[["prediction", "value", "verdict"]])
        L.append(f"Both runs (DEVIATIONS.md 1): the as-collected run's verdict table (summary_as_collected.md) is "
                 f"{'IDENTICAL to this one (every value and verdict).' if same else 'DIFFERENT from this one:'}\n")
        if not same:
            L.append(md_table(oc))
            L.append("")
    # kill rules
    def vd(df, key):
        return df[df.prediction.str.startswith(key)].verdict.tolist()
    x_stale, n_stale = vd(vt, "V1-stale x264")[0], vd(vt, "V1-stale NVENC")[0]
    x_flip = vd(vt, "V1-flip x264")
    n_flip = vd(vt, "V1-flip NVENC")[0]
    L.append("Kill-rule reading. The pre-registration says 'V1 fails for an encoder (stale and flip)' without defining "
             "whether one failing component suffices. Both readings, mechanically:\n")
    fails_ = {}
    for enc_, st_, fl_ in (("x264", x_stale, "PASS" if all(v == "PASS" for v in x_flip) else "FAIL"),
                           ("NVENC", n_stale, n_flip)):
        stf = None if not st_ in ("PASS", "FAIL") else st_ == "FAIL"  # no verdict / descriptive -> None
        flf = fl_ == "FAIL"
        a_ = flf and (stf if stf is not None else True)   # reading A: fails only if every available component fails
        b_ = flf or bool(stf)                             # reading B: fails if any component fails
        fails_[enc_] = (a_, b_)
        L.append(f"- {enc_}: V1-stale {st_}; V1-flip {fl_} (x264 flip = both flip criteria) → reading A "
                 f"(fail = all components fail): {'FAIL' if a_ else 'PASS'}; reading B (fail = any component "
                 f"fails): {'FAIL' if b_ else 'PASS'}.")
    L.append("- V1-stale without a verdict (V0 failed or < 20 eligible) is not counted as a failure in either reading.")
    L.append("")
    L.append("## 3. Per-clip and per-scene tables (tau = 1 px)\n")
    pc = pd.DataFrame({"scene": [scene_of(c) for c in ok_clips], "group": [grp_map[c] for c in ok_clips],
                       "eligible": [c in elig for c in ok_clips]}, index=ok_clips)
    stale_tab = pc.copy()
    for lab, arm, g in (("CRF12", "x264_crf12", "d1"), ("CRF45", "x264_crf45", "d1"), ("NV18", "nvenc_qp18", "d1"),
                        ("NV45", "nvenc_qp45", "d1"), ("QP24 bf0", "x264_qp24", "d1"),
                        ("QP24 bf2 B", "x264_qp24_bf2_pb1", "B"), ("NV28 bf0", "nvenc_qp28", "d1"),
                        ("NV28 B=P B", "nvenc_qp28_bf2_bqeq", "B")):
        stale_tab[lab] = col(arm, g, "stale").values
    stale_tab.insert(3, "moving share", el.set_index("clip")["moving_share"].reindex(ok_clips).values)
    stale_tab.insert(4, "V0 median err", v0med.reindex(ok_clips).values)
    L.append("### STALE (RAFT-estimated) per clip\n")
    L.append(md_table(stale_tab.reset_index().rename(columns={"index": "clip"}), "{:.5f}"))
    flip_tab = pc.copy()
    for c_ in CRFS[1:]:
        flip_tab[f"CRF{c_}"] = col(f"x264_crf{c_}", "d1", "flip").values
    flip_tab["rho(CRF)"] = rho.reindex(ok_clips).values
    for q_ in (18, 45):
        flip_tab[f"NV{q_}"] = col(f"nvenc_qp{q_}", "d1", "flip").values
    L.append("\n### FLIP vs x264 CRF 12 per clip\n")
    L.append(md_table(flip_tab.reset_index().rename(columns={"index": "clip"})))

    # per scene
    def clip_crit():
        d = pd.DataFrame(index=ok_clips)
        d["scene"] = [scene_of(c) for c in ok_clips]
        d["eligible"] = [c in elig for c in ok_clips]
        d["V0 ok"] = v0med.reindex(ok_clips) < V0_ERR
        d["V1s x264"] = (col("x264_crf45", "d1", "stale") - col("x264_crf12", "d1", "stale")) > DELTA
        d["V1s NV"] = (col("nvenc_qp45", "d1", "stale") - col("nvenc_qp18", "d1", "stale")) > DELTA
        d["V1f x264"] = (col("x264_crf45", "d1", "flip") - col("x264_crf18", "d1", "flip")) > 0
        d["rho>=0.8"] = rho.reindex(ok_clips) >= 0.8
        d["V1f NV"] = (col("nvenc_qp45", "d1", "flip") - col("nvenc_qp18", "d1", "flip")) > 0
        d["V2 x264"] = (col("x264_qp24_bf2_pb1", "B", "stale") - col("x264_qp24", "d1", "stale")) > DELTA
        d["V2 NV"] = (col("nvenc_qp28_bf2_bqeq", "B", "stale") - col("nvenc_qp28", "d1", "stale")) > DELTA
        return d
    cc = clip_crit()
    rows = []
    for s, d in cc.groupby("scene"):
        e = d[d.eligible]
        row = {"scene": s, "clips": len(d), "eligible": len(e)}
        for k in ("V0 ok", "V1s x264", "V1s NV", "V2 x264", "V2 NV"):
            row[k] = f"{int(e[k].sum())}/{len(e)}"
        for k in ("V1f x264", "rho>=0.8", "V1f NV"):
            row[k] = f"{int(d[k].sum())}/{len(d)}"
        row["med STALE CRF12"] = col("x264_crf12", "d1", "stale", list(d.index)).median()
        row["med STALE CRF45"] = col("x264_crf45", "d1", "stale", list(d.index)).median()
        row["med FLIP CRF18"] = col("x264_crf18", "d1", "flip", list(d.index)).median()
        row["med FLIP CRF45"] = col("x264_crf45", "d1", "flip", list(d.index)).median()
        rows.append(row)
    L.append("\n### Per scene (clip counts meeting each clip-level criterion; STALE criteria over eligible clips; "
             "medians over the scene's clips)\n")
    L.append(md_table(pd.DataFrame(rows), "{:.5f}"))

    # descriptive
    L.append("\n## 4. Descriptive (no verdicts)\n")

    def med_tab(arms_groups, cols_):
        rows_ = []
        for lab, arm, g in arms_groups:
            s = r1[(r1.arm == arm) & (r1.group == g)]
            row = {"arm": lab, "group": g, "clips": len(s)}
            for c_ in cols_:
                row["med " + c_] = s[c_].median()
            rows_.append(row)
        return pd.DataFrame(rows_)

    cols_d = ["stale", "flip", "missed", "reuse_share_inframe", "pix_of_inframe", "pix_of_reuse"]
    L.append("### mpeg4 series (median over clips)\n")
    L.append(md_table(med_tab([(f"mpeg4 q{q}", f"mpeg4_q{q}", "d1") for q in (2, 4, 8, 16, 31)], cols_d), "{:.5f}"))
    L.append("\n### NATIVE (camera encode) by source group (median over clips)\n")
    rows_ = []
    for gname in sorted(set(grp_map.values())):
        cl = [c for c in ok_clips if grp_map[c] == gname]
        s = r1[(r1.arm == "native") & (r1["clip"].isin(cl))]
        for g, sg in s.groupby("group"):
            rows_.append({"source group": gname, "group": g, "clips": len(sg), **{"med " + c_: sg[c_].median()
                                                                                   for c_ in cols_d}})
    L.append(md_table(pd.DataFrame(rows_), "{:.5f}"))
    L.append("\n### Source group (clip counts meeting each criterion; medians of per-clip differences)\n")
    rows_ = []
    for gname in sorted(set(grp_map.values())):
        cl = [c for c in ok_clips if grp_map[c] == gname]
        d = cc.loc[cl]
        e = d[d.eligible]
        rows_.append({"source group": gname, "clips": len(d), "eligible": len(e),
                      **{k: f"{int(e[k].sum())}/{len(e)}" for k in ("V0 ok", "V1s x264", "V1s NV", "V2 x264", "V2 NV")},
                      **{k: f"{int(d[k].sum())}/{len(d)}" for k in ("V1f x264", "rho>=0.8", "V1f NV")},
                      "med ΔSTALE x264 (45-12)": (col("x264_crf45", "d1", "stale", cl)
                                                  - col("x264_crf12", "d1", "stale", cl)).median(),
                      "med ΔFLIP x264 (45-18)": (col("x264_crf45", "d1", "flip", cl)
                                                 - col("x264_crf18", "d1", "flip", cl)).median()})
    L.append(md_table(pd.DataFrame(rows_), "{:.5f}"))
    L.append("\n### Missed reuse (RAFT-estimated static cells, |RAFT| < 0.5 px, that the gate recomputes / valid) and "
             "pixel-change proxy (reused cells whose exact source mean |ΔY| > 8 grey levels), all arms, median over clips\n")
    ag = [(f"x264 CRF {c}", f"x264_crf{c}", "d1") for c in CRFS] + [("x264 QP 24", "x264_qp24", "d1"),
          ("x264 QP 24 bf2 (B)", "x264_qp24_bf2_pb1", "B")] + [(f"NVENC QP {q}", f"nvenc_qp{q}", "d1") for q in NQPS] \
        + [("NVENC QP 28 bf2 (B)", "nvenc_qp28_bf2", "B"), ("NVENC QP 28 bf2 B=P (B)", "nvenc_qp28_bf2_bqeq", "B")]
    L.append(md_table(med_tab(ag, cols_d), "{:.5f}"))
    L.append("\n### tau = 0.5 and 2 px: the verdict quantities recomputed (no verdicts)\n")
    for tau in (0.5, 2.0):
        vtx, _ = verdicts(res, ok_clips, elig, v0med, tau=tau)
        vtx = vtx.iloc[1:].drop(columns=["verdict"])
        L.append(f"tau = {tau}:\n")
        L.append(md_table(vtx))
        L.append("")

    # failures / oddities / runtime
    L.append("## 5. Checks, failures, oddities, runtime\n")
    L.append(f"- Metric-code self-test on Sintel alley_1 (final): gate_counts on the sweep's saved encodes with "
             f"Sintel GT vs p3/sweep/results.csv: {st['values_checked']} counters compared, "
             f"{len(st['mismatches'])} unexplained mismatches{': ' + str(st['mismatches'][:5]) if st['mismatches'] else ''}"
             f"; {len(st['expected'])} explained differences: "
             + "; ".join(sorted({f"{m[0]} {m[1]}: {m[-1]}" for m in st["expected"]})) + ".")
    L.append(f"- Data variant: {variant}")
    chk = pd.DataFrame([{"clip": c, **outs[c]["checks"]} for c in ok_clips])
    L.append(f"- Clip failures: {len(fails)}{': ' + '; '.join(f['clip'] + ' ' + f['error'] for f in fails) if fails else ''}.")
    L.append(f"- Arms per clip: {sorted(chk.n_arms.unique().tolist())}.")
    L.append(f"- x264 CRF 12 vectors re-extracted for V0: |MV| repaint vs stored q mismatching cells, total over clips: "
             f"{int(chk.crf12_q_repaint_mismatch_cells.sum())}.")
    L.append(f"- Source mean |ΔY| recomputed vs stored uint8: mismatching cells, total: "
             f"{int(chk.absdiff_rounding_mismatch_cells.sum())}.")
    L.append(f"- V0 cells with NaN error (no RAFT landing): {int(chk.v0_nan_err.sum())}. V0 cells per clip "
             f"(median {int(v0.v0_cells.median())}, min {int(v0.v0_cells.min())}); clips with no V0 cell: "
             f"{', '.join(v0.loc[v0.v0_cells == 0, 'clip']) or 'none'}.")
    notes = [f"{c}: {n}" for c in ok_clips for n in outs[c]["notes"]]
    L.append(f"- Notes: {'; '.join(notes) if notes else 'none'}.")
    L.append("- Groups present per arm: " + "; ".join(
        f"{arm}: {', '.join(sorted(r1[r1.arm == arm].group.unique()))}" for arm in sorted(r1.arm.unique())
        if set(r1[r1.arm == arm].group.unique()) != {"d1"}) + ".")
    L.append(f"- NATIVE VIRAT_S_000002 has d unknown (4 reference frames, B-frames): its P-frames are group "
             f"'d1(assumed)'.")
    L.append(f"- Runtime: self-test {t_self:.0f}s; total {time.perf_counter() - T0:.0f}s with {a.workers} workers.")
    with open(os.path.join(HERE, f"summary{sfx}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(vt.to_string())
    print(vt2.to_string())
    print(f"done in {time.perf_counter() - T0:.0f}s")


if __name__ == "__main__":
    main()
