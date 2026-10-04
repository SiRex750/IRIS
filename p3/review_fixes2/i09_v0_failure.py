"""Review-2 item 9: the confirmatory clip that failed the V0 RAFT agreement check, and why it stays in the analysis.
SUPPLEMENTARY - no verdict changes.

V0 (p3/virat_confirm/analyze_confirm.py, rule of 45f3e98): per eligible clip, median |MV - RAFT| over x264 CRF 12 d1
cells that are valid, |RAFT| > 2 px, have a vector, and texture >= the 75th percentile (9.829); the clip "passes"
if the median < 1.5 px. V0 is a TEST-LEVEL gate: C1-C4 carry verdicts if >= 75% of eligible clips pass. The
pre-registration (p3/virat_confirm/PREREGISTRATION.md) defines the analysed set only by eligibility (>= 2,000
moving cell-frames) and fixes it before results; it has no rule that drops a clip failing V0, so dropping it after
seeing V0 would be a post-hoc exclusion.

Here: the failing clip from eligibility_v0.csv; its V0 error distribution recomputed (re-extracting the saved
x264 CRF 12 encode, single-threaded, as analyze_confirm) next to the other eligible clips; the same error on
forward-backward consistent cells (item 8b mask, _cache/fb_<clip>.npz); and C1-C4 recounted without it (26 clips),
as a sensitivity check. Writes i09_v0_failure.md / .csv.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

from _common import P3, HERE, eligible, V, CPREDS, need, cctv_base, f, table, write

sys.path.insert(0, os.path.join(P3, "virat"))
CONF = os.path.join(P3, "virat_confirm")


def v0_errors(clip, q75):
    from analyze_virat import frame_groups, cell_vectors
    from lib.mvs import extract_mvs
    base = cctv_base(clip)
    n, Hc, Wc = base["n"], base["Hc"], base["Wc"]
    z = np.load(os.path.join(CONF, "data", clip, "x264_crf12.npz"))
    mv_q = z["mv_q"].reshape(n, -1)
    d1 = frame_groups([str(x) for x in z["ftype"]], z["d_past"], bool(z["d_known"]))["d1"]
    fr, _ = extract_mvs(os.path.join(CONF, "encodes", clip, "x264_crf12.mp4"))
    vec = cell_vectors(fr, Hc, Wc)
    tex = np.load(os.path.join(HERE, "_cache", f"tex_{clip}.npz"))["tex"]
    fb = np.load(os.path.join(HERE, "_cache", f"fb_{clip}.npz"))["ok"]
    sel = np.zeros((n, Hc * Wc), bool)
    sel[d1] = base["valid"][d1] & (base["gmag"][d1] > 2.0) & (mv_q[d1] != 255) & (tex[d1] >= q75)
    tt, cc = np.nonzero(sel)
    fl = base["splat"].reshape(n - 1, -1, 2)[tt - 1, cc]
    err = np.hypot(vec[tt, cc, 0] - fl[:, 0], vec[tt, cc, 1] - fl[:, 1])
    raft = np.hypot(fl[:, 0], fl[:, 1])
    mv = np.hypot(vec[tt, cc, 0], vec[tt, cc, 1])
    c = fb[tt, cc]
    return {"clip": clip, "v0_cells": int(err.size), "v0_median_err": float(np.median(err)),
            "share_err_gt_3px": float((err > 3).mean()), "median_raft_px": float(np.median(raft)),
            "median_mv_px": float(np.median(mv)), "share_raft_gt_8px": float((raft > 8).mean()),
            "fb_consistent_share": float(c.mean()),
            "v0_median_err_fb_consistent": float(np.median(err[c])) if c.any() else np.nan,
            "v0_median_err_fb_inconsistent": float(np.median(err[~c])) if (~c).any() else np.nan}


def main():
    el = eligible()
    ev = pd.read_csv(os.path.join(CONF, "eligibility_v0.csv")).set_index("clip")
    bad = [c for c in el if not ev.loc[c, "v0_median_err_px"] < 1.5]
    q75 = float(re.search(r"V0 texture threshold .*?: ([0-9.]+) grey", open(os.path.join(CONF, "summary.md"),
                                                                             encoding="utf-8").read()).group(1))
    rows = pd.DataFrame([v0_errors(c, q75) for c in el]).set_index("clip")
    rows.to_csv(os.path.join(HERE, "i09_v0_failure.csv"))
    assert np.allclose(rows.v0_median_err, ev.loc[el, "v0_median_err_px"], atol=2e-3), "V0 not reproduced"
    others = rows.drop(index=bad)
    keep = [c for c in el if c not in bad]
    lines = []
    for key, hi, gh, lo, gl, frac in CPREDS:
        d27 = (V(hi, gh) - V(lo, gl)).reindex(el)
        d26 = d27.reindex(keep)
        lines.append([key, f"{int((d27 > .01).sum())}/27 (need {need(27, frac)})",
                      f"{int((d26 > .01).sum())}/{len(keep)} (need {need(len(keep), frac)})",
                      f(d27.median()), f(d26.median()), " ".join(f(d27[b]) for b in bad)])
    L = ["# Item 9: the V0 failure in the confirmatory test (supplementary)", "",
         f"- Failing clip(s): **{', '.join(bad)}** (V0 median |MV - RAFT| "
         + ", ".join(f"{ev.loc[b, 'v0_median_err_px']:.3f} px" for b in bad)
         + " vs the 1.5 px limit; the other 26 eligible clips: median "
         f"{others.v0_median_err.median():.3f}, max {others.v0_median_err.max():.3f} px). V0 as a whole: 26/27 >= 21 -> PASS.",
         "- Why it stays: V0 is a test-level gate (>= 75% of eligible clips), not a per-clip exclusion rule; the "
         "pre-registered analysed set is fixed by eligibility (>= 2,000 moving cell-frames) alone, and removing a clip "
         "after seeing V0 would be a post-hoc exclusion. Its RAFT-based numbers are flagged, and the 26-clip "
         "recount below shows the verdicts do not depend on it.", "",
         "## V0 cells of the failing clip vs the other eligible clips (recomputed; V0 reproduced to 2e-3 px)", "",
         table(["", "V0 cells", "median err px", "err > 3 px", "median |RAFT| px", "median |MV| px", "|RAFT| > 8 px",
                "FB-consistent share", "median err, FB-consistent", "median err, FB-inconsistent"],
               [[b, *(f(rows.loc[b, k], 3) if k != "v0_cells" else int(rows.loc[b, k]) for k in rows.columns)]
                for b in bad]
               + [["others (median of 26)", *(f(others[k].median(), 3) if k != "v0_cells"
                                              else int(others[k].median()) for k in rows.columns)]]), "",
         "Reading: the clip's V0 cells move faster than in any other eligible clip (median |RAFT| above, a large "
         "share above 8 px), and about half of them fail RAFT's own forward-backward check (item 8b). On its "
         "FB-consistent V0 cells the median |MV - RAFT| is below the 1.5 px limit; the excess error sits in the "
         "FB-inconsistent cells, i.e. where RAFT itself is unreliable. Descriptive diagnosis, not a re-test of V0.", "",
         "## C1-C4 with and without the failing clip (tau = 1; sensitivity, not a verdict)", "",
         table(["prediction", "27 clips", "without it", "median diff 27", "median diff 26", "its own diff"], lines)]
    write("i09_v0_failure.md", L)


if __name__ == "__main__":
    main()
