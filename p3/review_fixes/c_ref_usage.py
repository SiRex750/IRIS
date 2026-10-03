"""Review follow-up C: how often do x264 ref 2 / ref 3 blocks point further back? EXPLORATORY - no verdicts.

The bitstream export gives no reference index, so it is inferred on Sintel (final pass) by comparing each cell's
vector with whole multiples of the GT motion:
  cells: P-frame cells (t >= 1) of x264_crf23 (ref 1, control), x264_crf23_ref2, x264_crf23_ref3 that are
         valid (GT cover >= 0.5), moving (|GT| > 2 px), textured (top quartile of cell texture) and have a vector.
  class: k in {1, 2, 3} minimising |MV - k*GT|; the cell is "k x" if that error is <= 0.5*k px, else "none".
         Primary GT = the frame-t cell GT (t-1 -> t) - this assumes constant velocity over k frames.
         Secondary ("trajectory GT"): k-frame displacement composed by following the cell back through the
         per-frame cell GT (nearest cell on frame t-1's grid, then t-2's), so it does not assume constant velocity;
         a cell whose back-traced cell is invalid or off-frame is left out of the secondary count.
Also: share of moving valid cells (all textures, and the textured subset) whose vector is near zero (|MV| < 1 px),
and the share with no vector, per encode.
Texture: cell_texture of p3/pilot_v2/texture_check (mean np.gradient magnitude of source luma, swscale yuv420p Y as
the encoder input); top-quartile threshold = 75th percentile over valid cells of final-pass source frames 1..n-1 of
all 23 sequences.
MVs: the saved block records p3/sweep/blocks/final/<encode>__<seq>.parquet (mv_x, mv_y = motion prev -> cur, as
run_one stores them), painted onto cells with run_pilot.paint. These bf 0 streams were not affected by the threaded
export bug and their block files are identical in the single-threaded rerun (p3/sweep_rerun/COMPARISON.md), so no
re-extraction is needed. Self-check: reuse and stale cells recomputed from the painted cells must equal
results.csv (final, tau 1) for every encode x sequence.
Outputs: p3/review_fixes/ref_usage.md, ref_usage.csv.
"""
import os

import numpy as np
import pandas as pd

from _common import HERE, P3, C, MOVING_GT, gt_cells, sequences, SINTEL

from run_pilot import paint, y_plane  # noqa: E402
from lib.flo import SintelSeq  # noqa: E402

ENCS = (("ref 1 (control)", "x264_crf23", "d1"), ("ref 2", "x264_crf23_ref2", "d1(assumed)"),
        ("ref 3", "x264_crf23_ref3", "d1(assumed)"))
KS = (1, 2, 3)
TAU = 1.0


def cell_texture(y):  # = p3/pilot_v2/texture_check/texture_check.py cell_texture
    gy, gx = np.gradient(y)
    g = np.hypot(gx, gy)
    H, W = g.shape
    return g.reshape(H // C, C, W // C, C).mean(axis=(1, 3))


def textures(seq):
    fn = os.path.join(HERE, "_cache", f"tex_{seq}.npy")
    if os.path.exists(fn):
        return np.load(fn)
    s = SintelSeq(SINTEL, seq, "final")
    H = s.image(0).shape[0]
    tex = np.stack([cell_texture(y_plane(s.image(t), H)).ravel() for t in range(s.n_frames)])
    np.save(fn, tex)
    return tex


def traj(gt, t, k, Hc, Wc):
    """k-frame displacement (frame t-k -> t) per cell of frame t by back-tracing through cell GT; NaN if lost."""
    n = Hc * Wc
    cy, cx = np.divmod(np.arange(n), Wc)
    px, py = cx * C + C / 2.0, cy * C + C / 2.0
    dx, dy = np.zeros(n), np.zeros(n)
    ok = np.ones(n, bool)
    for j in range(k):
        tt = t - j
        if tt < 1:
            return np.full(n, np.nan), np.full(n, np.nan)
        g = gt[tt]
        ix = np.clip(np.floor(px / C).astype(np.int64), 0, Wc - 1)
        iy = np.clip(np.floor(py / C).astype(np.int64), 0, Hc - 1)
        inside = (px >= 0) & (px < Wc * C) & (py >= 0) & (py < Hc * C)
        ci = iy * Wc + ix
        ok &= inside & g["valid"][ci]
        ux, uy = g["cgx"][ci], g["cgy"][ci]
        dx, dy = dx + ux, dy + uy
        px, py = px - ux, py - uy
    dx[~ok], dy[~ok] = np.nan, np.nan
    return dx, dy


def classify(mx, my, gx, gy):
    err = np.stack([np.hypot(mx - k * gx, my - k * gy) for k in KS])
    best = np.argmin(np.nan_to_num(err, nan=np.inf), axis=0)
    e = np.take_along_axis(err, best[None], 0)[0]
    k = np.array(KS)[best]
    return np.where(e <= 0.5 * k, k, 0)


def main():
    seqs = sequences()
    tex = {s: textures(s) for s in seqs}
    gts = {s: gt_cells(s) for s in seqs}
    vals = np.concatenate([tex[s][t][g["valid"]] for s in seqs for t, g in gts[s][0].items()])
    thr = float(np.percentile(vals, 75))
    res = pd.read_csv(os.path.join(P3, "sweep", "results.csv"))
    res = res[(res["pass"] == "final") & (res.tau == TAU)].set_index(["encode_id", "sequence", "group"])
    rows, mism = [], []
    for seq in seqs:
        gt, (Hc, Wc) = gts[seq]
        tr = {}
        for lab, eid, grp in ENCS:
            b = pd.read_parquet(os.path.join(P3, "sweep", "blocks", "final", f"{eid}__{seq}.parquet"),
                                columns=["frame", "ftype", "group", "x", "y", "w", "h", "dir", "mv_x", "mv_y"])
            b = b[(b.ftype == "P") & (b["dir"] == "past")]
            acc = {k: 0 for k in ("n_moving", "n_moving_vec", "n_moving_nearzero", "n_moving_novec", "n_tex",
                                  "n_tex_nearzero", "n_tex_novec", "n_class", "c1", "c2", "c3", "c0",
                                  "n_class_traj", "t1", "t2", "t3", "t0", "reuse", "stale")}
            for t, bt in b.groupby("frame"):
                g = gt[t]
                ci, bi = paint(Hc, Wc, bt.x.to_numpy(np.int64), bt.y.to_numpy(np.int64),
                               bt.w.to_numpy(np.int64), bt.h.to_numpy(np.int64))
                mx = np.full(Hc * Wc, np.nan)
                my = np.full(Hc * Wc, np.nan)
                mx[ci], my[ci] = bt.mv_x.to_numpy()[bi], bt.mv_y.to_numpy()[bi]
                has = ~np.isnan(mx)
                mag = np.hypot(mx, my)
                near0 = has & (np.nan_to_num(mag, nan=np.inf) < 1.0)
                valid, gmag = g["valid"], g["gmag"]
                acc["reuse"] += int((valid & near0).sum())  # tau = 1: reuse = has & |MV| < 1
                acc["stale"] += int((valid & near0 & (gmag > MOVING_GT)).sum())
                mov = valid & (gmag > MOVING_GT)
                txm = mov & (tex[seq][t] >= thr)
                acc["n_moving"] += int(mov.sum())
                acc["n_moving_vec"] += int((mov & has).sum())
                acc["n_moving_nearzero"] += int((mov & near0).sum())
                acc["n_moving_novec"] += int((mov & ~has).sum())
                acc["n_tex"] += int(txm.sum())
                acc["n_tex_nearzero"] += int((txm & near0).sum())
                acc["n_tex_novec"] += int((txm & ~has).sum())
                sel = txm & has
                cls = classify(mx[sel], my[sel], g["cgx"][sel], g["cgy"][sel])
                acc["n_class"] += int(sel.sum())
                for k in (0, 1, 2, 3):
                    acc[f"c{k}"] += int((cls == k).sum())
                if t not in tr:
                    tr[t] = {k: traj(gt, t, k, Hc, Wc) for k in KS}
                d = tr[t]
                ok = sel & np.all([~np.isnan(d[k][0]) for k in KS], axis=0)
                err = np.stack([np.hypot(mx[ok] - d[k][0][ok], my[ok] - d[k][1][ok]) for k in KS])
                best = np.argmin(err, axis=0)
                kk = np.array(KS)[best]
                ct = np.where(np.take_along_axis(err, best[None], 0)[0] <= 0.5 * kk, kk, 0)
                acc["n_class_traj"] += int(ok.sum())
                for k in (0, 1, 2, 3):
                    acc[f"t{k}"] += int((ct == k).sum())
            r = res.loc[(eid, seq, grp)]
            if acc["reuse"] != int(r.reuse_cells) or acc["stale"] != int(r.stale_cells):
                mism.append(f"{eid} {seq}: reuse {acc['reuse']} vs {int(r.reuse_cells)}, "
                            f"stale {acc['stale']} vs {int(r.stale_cells)}")
            rows.append({"sequence": seq, "encode": lab, "encode_id": eid, **acc})
        print(seq, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "ref_usage.csv"), index=False)
    write_md(df, thr, mism, len(seqs))


def sh(a, b):
    return a / b if b else np.nan


def write_md(df, thr, mism, nseq):
    L = ["# Reference usage of x264 ref 2 / ref 3 on Sintel, inferred from vectors (review follow-up C)\n",
         "**Exploratory, not a verdict.** The MV export has no reference index; which reference a block used is "
         "inferred by matching its vector to 1x, 2x or 3x the ground-truth motion. Generated by "
         "`p3/review_fixes/c_ref_usage.py` from the saved Sintel sweep block records (final pass, x264 CRF 23; "
         "no new encodes, no re-extraction).\n",
         f"- Cells: P-frame cells with a vector that are valid (GT cover >= 0.5), moving (|GT| > 2 px) and textured "
         f"(cell texture >= {thr:.3f} grey levels/px, the 75th percentile over valid cells of all {nseq} "
         "sequences' final-pass source frames).",
         "- Class: k in {1, 2, 3} minimising |MV - k·GT|; '1x/2x/3x' if that error is <= 0.5·k px, else 'none'. "
         "Primary GT is the frame's own t-1 -> t cell GT, so 2x/3x assume constant velocity over 2-3 frames. "
         "Secondary ('trajectory'): the k-frame displacement composed by back-tracing the cell through the per-frame "
         "GT, which drops that assumption; cells whose back-trace hits an invalid cell or leaves the frame are "
         "excluded from it.",
         "- A 2x/3x match is consistent with a block predicting from frame t-2/t-3; it is not proof (a 1x-reference "
         "block can carry a 2x vector by a wrong match, and a block on an older reference can land on 'none').",
         f"- Self-check: reuse and stale cells recomputed from the painted block records equal results.csv "
         f"(final, tau 1) for {len(df) - len(mism)}/{len(df)} encode x sequence pairs."
         + ("" if not mism else " Mismatches: " + "; ".join(mism)) + "\n"]

    tot = df.groupby("encode", sort=False).sum(numeric_only=True)
    L.append("## Overall (cells pooled over sequences) and median over sequences\n")
    L.append("| encode | classified cells | 1x | 2x | 3x | none | 2x+3x | median 2x+3x over seqs | "
             "trajectory: classified | 1x | 2x | 3x | none |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e, r in tot.iterrows():
        n, nt = r.n_class, r.n_class_traj
        sub = df[df.encode == e]
        m23 = ((sub.c2 + sub.c3) / sub.n_class).median()
        L.append(f"| {e} | {int(n)} | {sh(r.c1, n):.3f} | {sh(r.c2, n):.3f} | {sh(r.c3, n):.3f} | "
                 f"{sh(r.c0, n):.3f} | {sh(r.c2 + r.c3, n):.3f} | {m23:.3f} | {int(nt)} | {sh(r.t1, nt):.3f} | "
                 f"{sh(r.t2, nt):.3f} | {sh(r.t3, nt):.3f} | {sh(r.t0, nt):.3f} |")
    L.append("\n## Near-zero vectors on moving cells (|MV| < 1 px; these are the cells the tau = 1 gate reuses)\n")
    L.append("| encode | moving valid cells | near-zero share | no-vector share | median near-zero share over seqs | "
             "textured moving cells | near-zero share (textured) | no-vector share (textured) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for e, r in tot.iterrows():
        sub = df[df.encode == e]
        L.append(f"| {e} | {int(r.n_moving)} | {sh(r.n_moving_nearzero, r.n_moving):.4f} | "
                 f"{sh(r.n_moving_novec, r.n_moving):.4f} | {(sub.n_moving_nearzero / sub.n_moving).median():.4f} | "
                 f"{int(r.n_tex)} | {sh(r.n_tex_nearzero, r.n_tex):.4f} | {sh(r.n_tex_novec, r.n_tex):.4f} |")
    ref1 = df[df.encode == "ref 1 (control)"].set_index("sequence")
    for e in ("ref 2", "ref 3"):
        s = df[df.encode == e].set_index("sequence")
        d = s.n_moving_nearzero / s.n_moving - ref1.n_moving_nearzero / ref1.n_moving
        L.append(f"\n{e} minus ref 1, near-zero share of moving cells, per sequence: median {d.median():+.4f}; "
                 f"higher in {int((d > 0).sum())}/{len(d)}, lower in {int((d < 0).sum())}/{len(d)}, "
                 f"equal in {int((d == 0).sum())}/{len(d)}; above +0.01 in {int((d > 0.01).sum())}/{len(d)}.")

    L.append("\n## Per sequence (primary classification; shares of classified cells)\n")
    L.append("| sequence | " + " | ".join(f"{e}: n / 1x / 2x / 3x / none / near-zero (moving)"
                                         for e in tot.index) + " |")
    L.append("|---|" + "---|" * len(tot.index))
    for seq in sorted(df.sequence.unique()):
        cells = []
        for e in tot.index:
            r = df[(df.sequence == seq) & (df.encode == e)].iloc[0]
            n = r.n_class
            cells.append(f"{int(n)} / {sh(r.c1, n):.3f} / {sh(r.c2, n):.3f} / {sh(r.c3, n):.3f} / "
                         f"{sh(r.c0, n):.3f} / {sh(r.n_moving_nearzero, r.n_moving):.4f}")
        L.append(f"| {seq} | " + " | ".join(cells) + " |")
    with open(os.path.join(HERE, "ref_usage.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[7:22]))


if __name__ == "__main__":
    main()
