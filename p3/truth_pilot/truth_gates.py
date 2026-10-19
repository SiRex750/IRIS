"""Truth pilot gates (PLAN.md, e7ac3b0 §3-§5), per clip per candidate, from run_truth.py outputs and the existing
native RAFT (p3/codecsight_native2fps/data/<clip>/raft.npz). Writes gates.csv and gates.md (gates + runtime only)."""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
NATIVE = os.path.join(P3, "codecsight_native2fps")
sys.path.insert(0, NATIVE)
import gates as G  # noqa: E402  (native2fps gates: load_truth)
sys.path.insert(0, HERE)
from run_truth import CLIPS, KEYINT, PLAN_COMMIT, periods, query_points, windows  # noqa: E402

K = 15            # sampled-frame stride of these clips (asserted in run_truth.windows)
QUIET = 0.5       # px, per-frame |T| never exceeds (<=)
MOVED = 2.0       # px
RATE = 0.01       # gate 1
SPAN = 16         # native frames, gate 2
AGREE = 1.0       # px, gate 2
CANDS = ["dot", "cotracker3"]
OUT = os.path.join(HERE, "out")


def gate1(T, ok, disp, L):
    """Quiet cell-frames (valid and |T_i| <= 0.5 at every step r+1..s) at s = r + 15 m; false move if |D(s)| > 2."""
    quiet_step = ok & (np.hypot(T[..., 0], T[..., 1]) <= QUIET)
    nq = nf = 0
    for r, end in periods(L):
        allq = np.ones(T.shape[1], bool)
        for i in range(r + 1, end + 1):
            allq &= quiet_step[i - 1]
            if (i - r) % K == 0:
                nq += int(allq.sum())
                nf += int((allq & (np.hypot(disp[i, :, 0], disp[i, :, 1]) > MOVED)).sum())
    return nq, nf


def gate2(flow_tm1, xy, disp, L):
    """RAFT chain from each query point at r, d = 1..16, bilinear on the cell grid of frame t-1; |R| > 2 points."""
    Hc, Wc = flow_tm1.shape[1:3]
    errs = []
    for r, end in periods(L):
        p = xy.astype(np.float64).copy()
        alive = np.ones(len(p), bool)
        for d in range(1, min(SPAN, end - r) + 1):
            t = r + d
            u = np.clip((p[:, 0] - 1.5) / 4, 0, Wc - 1)
            v = np.clip((p[:, 1] - 1.5) / 4, 0, Hc - 1)
            f = flow_tm1[t - 1].astype(np.float64)
            step = np.stack([map_coordinates(f[..., j], [v, u], order=1, mode="nearest") for j in (0, 1)], -1)
            alive &= np.isfinite(step).all(-1)
            p = p + np.where(np.isfinite(step), step, 0)
            alive &= (p[:, 0] >= 0) & (p[:, 0] <= 1279) & (p[:, 1] >= 0) & (p[:, 1] <= 719)
            R = p - xy
            mov = alive & (np.hypot(R[:, 0], R[:, 1]) > MOVED)
            e = disp[t, mov] - R[mov]
            errs.append(np.hypot(e[:, 0], e[:, 1]))
    e = np.concatenate(errs) if errs else np.zeros(0)
    return int(e.size), (float(np.median(e)) if e.size else np.nan)


def main():
    W_ = windows()
    cells, xy = query_points()
    rows, mans = [], {}
    src = {c: c for c in CANDS}
    for cand in CANDS:
        mp = os.path.join(OUT, f"{cand}_manifest.json")
        mans[cand] = json.load(open(mp)) if os.path.exists(mp) else None
    # PLAN §1: if DOT at 480x856 ran out of memory on any period, DOT is evaluated at 360x640 on every clip.
    first = mans["dot"]
    dot_oom = bool(first) and any("OutOfMemory" in v.get("error", "") for v in first["clips"].values())
    if dot_oom:
        fb = os.path.join(OUT, "dot_360x640_manifest.json")
        mans["dot"] = json.load(open(fb)) if os.path.exists(fb) else None
        src["dot"] = "dot_360x640"
    for clip in CLIPS:
        L = W_[clip]["L"]
        T, ok, Hc, Wc = G.load_truth(clip, NATIVE)
        assert T.shape[0] == L - 1
        T, ok = T[:, cells], ok[:, cells]
        flow = np.load(os.path.join(NATIVE, "data", clip, "raft.npz"))["flow_tm1"]
        for cand in CANDS:
            row = {"clip": clip, "candidate": src[cand]}
            f = os.path.join(OUT, "data", clip, f"{src[cand]}.npz")
            m = (mans[cand] or {}).get("clips", {}).get(clip, {})
            row.update(status=m.get("status", "missing"), infer_s=m.get("infer_s"), peak_vram_gb=m.get("peak_vram_gb"))
            if os.path.exists(f) and m.get("status") == "ok":
                z = np.load(f)
                assert (z["cells"] == cells).all()
                disp = z["disp"].astype(np.float64)
                nq, nf = gate1(T, ok, disp, L)
                n2, med = gate2(flow, xy, disp, L)
                row.update(g1_quiet=nq, g1_false=nf, g1_rate=nf / nq if nq else np.nan,
                           g2_points=n2, g2_median_px=med)
            row["g1_pass"] = bool(row.get("g1_rate", np.nan) <= RATE)      # NaN does not pass
            row["g2_pass"] = bool(row.get("g2_median_px", np.nan) < AGREE)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "gates.csv"), index=False)
    L_ = ["# Truth pilot: gates per clip per candidate", "",
          f"Plan {PLAN_COMMIT}. This pilot chooses a reference after the RAFT-based truth failed (reading B drift gate "
          "0/25, `p3/codecsight_native2fps/gates.md`). Only truth was computed on these clips.", "",
          "Gate 1 (drift): quiet cell-frames whose candidate displacement is > 2 px; pass if <= 0.01 on all 3 clips. "
          "Gate 2 (agreement): median |candidate - native RAFT chain| over spans <= 16 frames, points moving > 2 px; "
          "pass if < 1.0 px on all 3 clips. A candidate passes only if both pass.", ""]
    for cand in CANDS:
        d = df[df.candidate == src[cand]]
        g1, g2 = d.g1_pass.all(), d.g2_pass.all()
        man = mans[cand] or {}
        L_.append(f"- **{src[cand]}**: gate 1 {'PASS' if g1 else 'FAIL'}, gate 2 {'PASS' if g2 else 'FAIL'} -> "
                  f"**{'PASS' if g1 and g2 else 'FAIL'}**. Wall time {man.get('wall_s', float('nan')) / 60:.1f} min.")
    if dot_oom:
        L_.append("- DOT at 480x856 ran out of GPU memory (" + "; ".join(
            f"{c}: {v.get('status')}" for c, v in first["clips"].items()) + "); per PLAN §1 DOT is evaluated at "
            "360x640 on every clip (first attempt: `out/dot_manifest.json`).")
    L_ += ["", "| clip | candidate | g1 quiet | g1 false | g1 rate | g1 | g2 points | g2 median px | g2 | infer s | peak VRAM GB |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L_.append(f"| {r.clip} | {r.candidate} | {r.get('g1_quiet', '')} | {r.get('g1_false', '')} | "
                  f"{r.get('g1_rate', float('nan')):.5f} | {'pass' if r.g1_pass else 'fail'} | {r.get('g2_points', '')} | "
                  f"{r.get('g2_median_px', float('nan')):.3f} | {'pass' if r.g2_pass else 'fail'} | "
                  f"{(r.infer_s or float('nan')):.0f} | {(r.peak_vram_gb or float('nan')):.2f} |")
    open(os.path.join(HERE, "gates.md"), "w", encoding="utf-8").write("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
