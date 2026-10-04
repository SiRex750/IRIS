"""Shared read-only helpers for p3/review_fixes2 (second internal review). Nothing here writes to a results file.

Sintel: p3/sweep/results.csv (the as-collected run; every arm/group a prediction uses is identical in
p3/sweep_rerun, see p3/sweep_rerun/COMPARISON.md). CCTV: p3/virat_confirm/results.csv (confirmatory test).
"""
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
CRFS = (12, 18, 23, 28, 33, 38, 45)
NQPS = (18, 23, 28, 33, 38, 45)
TAUS = (0.5, 1.0, 2.0)

_R = None
_V = None


def sintel():
    global _R
    if _R is None:
        _R = pd.read_csv(os.path.join(P3, "sweep", "results.csv"))
    return _R


def seqs():
    return sorted(sintel().sequence.unique())


def S(eid, group, pass_="final", tau=1.0, col="stale_of_valid", R=None):
    """Per-sequence Series (index = all 23 sequences) of one metric for one arm/group, as summarize_sweep.S."""
    R = sintel() if R is None else R
    d = R[(R["pass"] == pass_) & (R["encode_id"] == eid) & (R["group"] == group) & np.isclose(R["tau"], tau)]
    return d.set_index("sequence")[col].reindex(seqs())


def virat():
    global _V
    if _V is None:
        _V = pd.read_csv(os.path.join(P3, "virat_confirm", "results.csv"))
    return _V


def eligible():
    el = pd.read_csv(os.path.join(P3, "virat_confirm", "eligibility_v0.csv"))
    return sorted(el.loc[el.eligible.astype(bool), "clip"])


def V(arm, group, col="stale_mov", tau=1.0):
    r = virat()
    r = r[(r.arm == arm) & (r.group == group) & np.isclose(r.tau, tau)]
    return r.set_index("clip")[col]


# confirmatory predictions C1-C4: (key, high arm, high group, low arm, low group, share of eligible clips)
CPREDS = (("C1 x264 CRF 45 - CRF 12", "x264_crf45", "d1", "x264_crf12", "d1", .80),
          ("C2 NVENC QP 45 - QP 18", "nvenc_qp45", "d1", "nvenc_qp18", "d1", .75),
          ("C3 x264 QP 24 B - bf 0", "x264_qp24_bf2_pb1", "B", "x264_qp24", "d1", .80),
          ("C4 NVENC QP 28 B=P - bf 0", "nvenc_qp28_bf2_bqeq", "B", "nvenc_qp28", "d1", .75))


def need(n, frac):
    return math.ceil(frac * n - 1e-9)


def f(x, nd=4):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def table(header, lines):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in ln) + " |" for ln in lines]
    return "\n".join(out)


def write(name, lines):
    open(os.path.join(HERE, name), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


# --------------------------------------------------------------------------- CCTV per-clip recomputation
def cctv_base(clip):
    """valid (n, ncell) and |RAFT| (n, ncell) exactly as analyze_confirm.do_clip (row 0 unused), plus raft arrays."""
    rz = np.load(os.path.join(P3, "virat_confirm", "data", clip, "raft.npz"))
    sp = rz["splat_t"].astype(np.float32)
    nm1, Hc, Wc = sp.shape[:3]
    n, ncell = nm1 + 1, Hc * Wc
    valid = np.zeros((n, ncell), bool)
    gmag = np.full((n, ncell), np.nan, np.float32)
    valid[1:] = rz["splat_count"].reshape(nm1, ncell) >= 8  # analyze_virat.COVER_COUNT
    gmag[1:] = np.hypot(sp[..., 0], sp[..., 1]).reshape(nm1, ncell)
    return {"n": n, "Hc": Hc, "Wc": Wc, "valid": valid, "gmag": gmag, "splat": sp}


def cctv_counts(clip, base, masks, tau=1.0):
    """For each C1-C4 arm/group and each named extra mask (n, ncell) bool (None = no restriction): stale and moving
    cells of analyze_virat.gate_counts with valid := valid & mask. Returns rows (dicts)."""
    import sys
    sys.path.insert(0, os.path.join(P3, "virat"))
    from analyze_virat import frame_groups, gate_counts
    need_arms = sorted({(p[1], p[2]) for p in CPREDS} | {(p[3], p[4]) for p in CPREDS})
    rows = []
    for arm, grp in need_arms:
        z = np.load(os.path.join(P3, "virat_confirm", "data", clip, f"{arm}.npz"))
        n = z["mv_q"].shape[0]
        mv_q = z["mv_q"].reshape(n, -1)
        groups = frame_groups([str(x) for x in z["ftype"]], z["d_past"], bool(z["d_known"]))
        groups = {grp: groups[grp]} if grp in groups else {}
        for name, m in masks.items():
            v = base["valid"] if m is None else base["valid"] & m
            for r in gate_counts(mv_q, groups, v, base["gmag"], taus=(tau,)):
                rows.append({"clip": clip, "arm": arm, "group": grp, "mask": name, "moving_cells": r["moving_cells"],
                             "stale_cells": r["stale_cells"], "valid_cells": r["valid_cells"],
                             "stale_mov": r["stale_cells"] / r["moving_cells"] if r["moving_cells"] else np.nan})
    return rows


def cpred_table(df, masks, clips):
    """C1-C4 counts and medians per mask over `clips`. df: rows of cctv_counts. Returns (md lines, records)."""
    lines, recs = [], []
    for key, hi, gh, lo, gl, frac in CPREDS:
        for m in masks:
            d = df[df["mask"] == m]
            s = lambda a, g: d[(d.arm == a) & (d.group == g)].set_index("clip")["stale_mov"].reindex(clips)  # noqa
            mh, ml = s(hi, gh), s(lo, gl)
            diff = mh - ml
            mov = d[(d.arm == lo) & (d.group == gl)].set_index("clip")["moving_cells"].reindex(clips)
            k = int((diff > 0.01).sum())
            rec = {"prediction": key, "cells": m, "clips": len(clips), "count": k, "need": need(len(clips), frac),
                   "would_meet": k >= need(len(clips), frac), "median_low": ml.median(), "median_high": mh.median(),
                   "median_diff": diff.median(), "median_moving_cells_low_arm": mov.median(),
                   "min_moving_cells_low_arm": mov.min()}
            recs.append(rec)
            lines.append([key, m, f"{k}/{len(clips)}", f">= {rec['need']}", "yes" if rec["would_meet"] else "no",
                          f(ml.median()), f(mh.median()), f(diff.median()), f"{mov.median():.0f}",
                          f"{mov.min():.0f}"])
    hdr = ["prediction", "cells", "clips with rise > 0.01", "pre-reg count", "count met", "median low / bf 0",
           "median high / B", "median diff", "median moving cells (low arm)", "min moving cells"]
    return table(hdr, lines), recs
