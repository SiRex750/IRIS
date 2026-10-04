"""Review-2 item 6: B-frame cells with no past-pointing vector - how they are handled and how often. SUPPLEMENTARY.

How the code handles them (p3/pilot_v2/run_pilot_v2.py run_one, used unchanged by the Sintel sweep; the CCTV runner
p3/virat/run_virat.py cell_arrays copies the same mapping):
    for sgn in (1, -1):  # future first, past overrides
        ... paint(...) ; cmv[ci] = mv[sel][bi] ; cdir[ci] = sgn
    has = cdir != 0 ; reuse = has & (|use| < tau)
  - a cell covered by a past-pointing block (FFmpeg source = -1) uses that vector, whether or not a future-pointing
    block also covers it (a bi-predicted block is exported as one past and one future entry, so the past one wins);
  - a cell covered ONLY by future-pointing blocks (source = +1) uses the FUTURE vector: B_naive takes its raw length
    |mv|; B_scaled uses -mv / d_fut (run_one: `use = cmv / cd * where(cdir == 1, -1, 1)`). It is gated like any
    other cell - not recomputed;
  - a cell with no vector at all (intra block, or not covered) has cdir = 0 -> has = False -> never reused (recompute).
CCTV: mv_q holds the past vector if present, else the future one (cell_dir -1 / +1 / 0), gated as |MV| < tau
(naive); the confirmatory B arms use naive vectors only.

Counts, per B arm, over B-frames: share of valid cells that are past / future-only / no vector, and the share of
stale cells (B naive, tau = 1: valid, reused, |GT| > 2 px - or |RAFT| > 2 px for CCTV) that come from future-only
cells. Self-check: the recomputed B_naive stale cells equal results.csv `stale_cells`.
Sintel block records: p3/sweep_rerun/blocks (B-frame records are identical in p3/sweep and p3/sweep_rerun;
p3/sweep/DEVIATIONS.md 1). Writes i06_bframe_no_past.md / .csv.
"""
import importlib.util
import os

import numpy as np
import pandas as pd

from _common import P3, sintel, seqs, eligible, f, table, write, HERE

_s = importlib.util.spec_from_file_location("rf_common", os.path.join(P3, "review_fixes", "_common.py"))
rf = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rf)

C = 4
SINTEL_B = [("final", "x264_crf23_bf2"), ("final", "x264_qp24_bf2_pb1"), ("final", "nvenc_qp28_bf2"),
            ("final", "nvenc_qp28_bf2_bqeq"), ("clean", "x264_qp24_bf2_pb1"), ("clean", "nvenc_qp28_bf2_bqeq")]
CCTV_B = ["x264_qp24_bf2_pb1", "nvenc_qp28_bf2_bqeq"]


def paint(Hc, Wc, x0, y0, w, h):
    """p3/pilot/run_pilot.py paint, verbatim (importing run_pilot pulls in the whole pilot)."""
    assert np.all(w <= 16) and np.all(h <= 16)
    o = np.arange(4)
    cx = (x0 // C)[:, None, None] + o[None, None, :]
    cy = (y0 // C)[:, None, None] + o[None, :, None]
    msk = ((o[None, None, :] < (w // C)[:, None, None]) & (o[None, :, None] < (h // C)[:, None, None])
           & (cx < Wc) & (cy < Hc))
    bi = np.broadcast_to(np.arange(x0.size)[:, None, None], msk.shape)
    return (cy * Wc + cx)[msk], bi[msk]


def sintel_arm(pass_, eid, q, gt, Hc, Wc):
    b = pd.read_parquet(os.path.join(P3, "sweep_rerun", "blocks", pass_, f"{eid}__{q}.parquet"))
    b = b[b.ftype == "B"]
    n = Hc * Wc
    c = dict(frames=0, valid=0, past=0, fut=0, none=0, stale=0, stale_fut=0, moving=0, moving_fut=0)
    for t, bt in b.groupby("frame", sort=True):
        x0, y0, w, h = (bt[k].to_numpy(np.int64) for k in ("x", "y", "w", "h"))
        mv = np.hypot(bt.mv_x.to_numpy(), bt.mv_y.to_numpy())
        src = np.where(bt.dir.to_numpy() == "past", -1, 1)
        cdir = np.zeros(n, np.int8)
        cm = np.full(n, np.nan)
        for sgn in (1, -1):
            sel = np.nonzero(src == sgn)[0]
            if sel.size:
                ci, bi = paint(Hc, Wc, x0[sel], y0[sel], w[sel], h[sel])
                cm[ci], cdir[ci] = mv[sel][bi], sgn
        v, g = gt[t]["valid"], gt[t]["gmag"]
        stale = v & (cdir != 0) & (cm < 1.0) & (g > 2.0)
        mov = v & (g > 2.0)
        c["frames"] += 1
        c["valid"] += int(v.sum())
        c["past"] += int((v & (cdir == -1)).sum())
        c["fut"] += int((v & (cdir == 1)).sum())
        c["none"] += int((v & (cdir == 0)).sum())
        c["stale"] += int(stale.sum())
        c["stale_fut"] += int((stale & (cdir == 1)).sum())
        c["moving"] += int(mov.sum())
        c["moving_fut"] += int((mov & (cdir == 1)).sum())
    return c


def cctv_arm(clip, arm):
    d = os.path.join(P3, "virat_confirm", "data", clip)
    z = np.load(os.path.join(d, f"{arm}.npz"))
    r = np.load(os.path.join(d, "raft.npz"))
    n = z["mv_q"].shape[0]
    q, cd = z["mv_q"].reshape(n, -1), z["cell_dir"].reshape(n, -1)
    T = np.array([t for t, ft in enumerate(z["ftype"]) if str(ft) == "B" and t > 0])
    sp = r["splat_t"].astype(np.float32)
    valid = r["splat_count"].reshape(n - 1, -1)[T - 1] >= 8
    g = np.hypot(sp[..., 0], sp[..., 1]).reshape(n - 1, -1)[T - 1]
    qT, cT = q[T], cd[T]
    mov = valid & (g > 2.0)
    stale = mov & (qT < 4)
    return dict(frames=len(T), valid=int(valid.sum()), past=int((valid & (cT == -1)).sum()),
                fut=int((valid & (cT == 1)).sum()), none=int((valid & (cT == 0)).sum()), stale=int(stale.sum()),
                stale_fut=int((stale & (cT == 1)).sum()), moving=int(mov.sum()),
                moving_fut=int((mov & (cT == 1)).sum()))


def shares(c):
    V = c["valid"]
    return {"past_of_valid": c["past"] / V, "future_only_of_valid": c["fut"] / V, "no_vector_of_valid": c["none"] / V,
            "future_only_of_moving": c["moving_fut"] / c["moving"] if c["moving"] else np.nan,
            "future_only_of_stale": c["stale_fut"] / c["stale"] if c["stale"] else np.nan}


def main():
    R = sintel()
    rows = []
    gts = {q: rf.gt_cells(q) for q in seqs()}
    for pass_, eid in SINTEL_B:
        for q in seqs():
            gt, (Hc, Wc) = gts[q]
            c = sintel_arm(pass_, eid, q, gt, Hc, Wc)
            ref = R[(R["pass"] == pass_) & (R.encode_id == eid) & (R.group == "B_naive") & (R.tau == 1.0) &
                    (R.sequence == q)]
            rows.append({"data": f"Sintel {pass_}", "arm": eid, "unit": q, **c, **shares(c),
                         "stale_results_csv": int(ref.stale_cells.iloc[0]), "valid_results_csv": int(ref.valid_cells.iloc[0])})
    el = eligible()
    V = pd.read_csv(os.path.join(P3, "virat_confirm", "results.csv"))
    for arm in CCTV_B:
        for clip in sorted(V["clip"].unique()):
            c = cctv_arm(clip, arm)
            ref = V[(V["clip"] == clip) & (V.arm == arm) & (V.group == "B") & (V.tau == 1.0)]
            rows.append({"data": "CCTV confirmatory", "arm": arm, "unit": clip, "eligible": clip in el, **c, **shares(c),
                         "stale_results_csv": int(ref.stale_cells.iloc[0]), "valid_results_csv": int(ref.valid_cells.iloc[0])})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "i06_bframe_no_past.csv"), index=False)
    ok = (df.stale == df.stale_results_csv) & (df.valid == df.valid_results_csv)
    lines = []
    for (d, arm), g in df.groupby(["data", "arm"], sort=False):
        if d.startswith("CCTV"):
            g = g[g.eligible.astype(bool)]
        tot = g[["valid", "past", "fut", "none", "stale", "stale_fut", "moving", "moving_fut"]].sum()
        lines.append([d + (" (27 eligible)" if d.startswith("CCTV") else ""), arm, int(g.frames.sum()),
                      f(g.future_only_of_valid.median()), f(g.future_only_of_valid.min()),
                      f(g.future_only_of_valid.max()), f(g.no_vector_of_valid.median()),
                      f(tot.fut / tot.valid), f(g.future_only_of_stale.median()), f(tot.stale_fut / tot.stale)])
    doc = __doc__.split("Counts, per B arm")[0].strip().splitlines()[1:]
    write("i06_bframe_no_past.md", [
        "# Item 6: B-frame cells with no past-pointing vector (supplementary)", "", "## Handling, from the code", "",
        *doc, "", "## How often (B-frames only; tau = 1, B naive)", "",
        table(["data", "B arm", "B-frames", "future-only / valid, median", "min", "max", "no vector / valid, median",
               "future-only / valid, pooled", "share of stale cells that are future-only, median", "pooled"], lines),
        "", f"Self-check: recomputed valid and B_naive stale cells equal results.csv in {int(ok.sum())}/{len(df)} "
            "arm x sequence/clip rows."])


if __name__ == "__main__":
    main()
