"""Paper 3 exploratory pilot: encode x sequence grid, codec MVs vs frame-t GT (lib/gt_grid.py).

Exploratory only. Outputs (all under p3/pilot/):
  encodes/<enc>__<seq>.mp4          bitstreams (gitignored)
  blocks/<enc>__<seq>.parquet       one row per exported vector (gitignored)
  blocks/<enc>__<seq>__frames.parquet  one row per display frame
  results.csv                       one row per encode x sequence x group x tau
  encode_log.json                   exact option strings, sizes, timings, failures
Run order: x264 CRF 12 (gate reference) first, x264 veryslow last.

Definitions
  motion        reference -> current = dst - src = -motion/scale (px)
  d             reference distance. ref=1, no B-pyramid: P and past-pointing B vectors -> previous
                I/P in display order; future-pointing B -> next I/P. ref 2/3 arms: d unknown,
                read as d=1 (group label "d1(assumed)").
  groups        frame-level: P-frames with d=1 -> d1; P-frames with d>1 -> dgt1_naive/_scaled;
                B-frames -> B_naive/_scaled. naive = raw vector vs 1-step GT; scaled = vector/d,
                sign flipped for future refs, vs 1-step GT.
  GT            lib/gt_grid (frame t grid). Blocks/cells with gt_cover < 0.5 excluded.
  cells         4x4 px, 256 x 109 in-frame. Each cell takes the vector of the block covering it;
                B cells take the past-pointing vector when present. No vector = "recompute".
  gate          reuse iff cell has a vector and |MV| < tau (MV naive or scaled per group).
  All rates are pixel-area weighted (cells are equal area; block rows weighted by w*h).
"""
import json
import os
import sys
import time
import traceback

import av
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, P3)

from lib.flo import SintelSeq  # noqa: E402
from lib.encode import encode_sequence  # noqa: E402
from lib.mvs import extract_mvs, motion_prev_to_cur, dst_topleft  # noqa: E402
from lib.compare import compare_frame  # noqa: E402
from lib.gt_grid import GTGrid  # noqa: E402

SINTEL = r"C:\Users\akash\Documents\datasets\MPI-Sintel"
ENC_DIR = os.path.join(HERE, "encodes")
BLK_DIR = os.path.join(HERE, "blocks")
TAUS = (0.5, 1.0, 2.0)
STATIC_GT, FALSE_STATIC_GT, STALE_GT = 0.5, 1.0, 2.0
EPE_BIG = 3.0
COVER_MIN = 0.5
C = 4
FPS = 24
REF_ID = "x264_crf12"

X = dict(preset="medium", bframes=0, keyint=250, ref=1)
N = dict(bframes=0, keyint=250, ref=1)
M = dict(bframes=0, keyint=250)


def _e(eid, codec, factors, d_known=True, **kn):
    return {"id": eid, "codec": codec, "factors": factors, "d_known": d_known, "knobs": kn}


ENCODES = (
    [_e("x264_crf12", "libx264", ["crf"], **{**X, "crf": 12})]
    + [_e(f"x264_crf{c}", "libx264", ["crf"] + (["ref", "preset", "bframes", "keyint", "encoder"] if c == 23 else []),
          **{**X, "crf": c}) for c in (18, 23, 28, 33, 38, 45)]
    + [_e("x264_crf23_bf2", "libx264", ["bframes"], **{**X, "crf": 23, "bframes": 2, "b_pyramid": "none"}),
       _e("x264_crf23_ref2", "libx264", ["ref"], d_known=False, **{**X, "crf": 23, "ref": 2}),
       _e("x264_crf23_ref3", "libx264", ["ref"], d_known=False, **{**X, "crf": 23, "ref": 3}),
       _e("x264_crf23_ultrafast", "libx264", ["preset"], **{**X, "crf": 23, "preset": "ultrafast"}),
       _e("x264_crf23_keyint30", "libx264", ["keyint"], **{**X, "crf": 23, "keyint": 30})]
    + [_e(f"nvenc_qp{q}", "h264_nvenc", ["nvenc_qp"] + (["encoder", "bframes"] if q in (23, 28) else []),
          **{**N, "qp": q, "preset": "p4"}) for q in (18, 23, 28, 33, 38, 45)]
    + [_e("nvenc_qp28_bf2", "h264_nvenc", ["bframes"], **{**N, "qp": 28, "preset": "p4", "bframes": 2,
                                                          "b_ref_mode": "disabled"})]
    + [_e(f"mpeg4_q{q}", "mpeg4", ["mpeg4_q", "encoder"], **{**M, "q": q}) for q in (2, 4, 8, 16, 31)]
    + [_e("x264_crf23_veryslow", "libx264", ["preset"], **{**X, "crf": 23, "preset": "veryslow"})]
)


# --------------------------------------------------------------------------- helpers
def wmedian(v, w):
    if v.size == 0:
        return np.nan
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, cw[-1] / 2.0)])


def y_plane(rgb_or_frame, H):
    f = rgb_or_frame if isinstance(rgb_or_frame, av.VideoFrame) else \
        av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb_or_frame), format="rgb24")
    return f.reformat(format="yuv420p").to_ndarray()[:H].astype(np.float64)


def load_seq(name):
    """Load images, source Y planes and per-frame GT once per sequence."""
    s = SintelSeq(SINTEL, name, "final")
    imgs = [s.image(i) for i in range(s.n_frames)]
    H, W = imgs[0].shape[:2]
    gt = {}
    for t in range(1, s.n_frames):
        flow, bad = s.flow_into(t), s.bad_mask_into(t)
        g = GTGrid(flow, bad)
        g._build_sat()
        cgx, cgy, ccov, _ = g.cell_gt()
        gt[t] = {"g": g, "flow": flow, "bad": bad, "cgx": cgx, "cgy": cgy, "ccov": ccov}
    return {"name": name, "imgs": imgs, "yref": [y_plane(im, H) for im in imgs], "gt": gt,
            "H": H, "W": W, "Hc": -(-H // C), "Wc": -(-W // C)}


def ref_distances(types):
    """prev / next I-or-P display index for each display index."""
    n = len(types)
    prev, nxt = [None] * n, [None] * n
    last = None
    for i in range(n):
        prev[i] = last
        if types[i] in ("I", "P"):
            last = i
    last = None
    for i in range(n - 1, -1, -1):
        nxt[i] = last
        if types[i] in ("I", "P"):
            last = i
    return prev, nxt


def paint(Hc, Wc, x0, y0, w, h):
    """Flat cell indices + owning block index for blocks (all on the 4-px grid, <= 16 px)."""
    assert np.all(w <= 16) and np.all(h <= 16)
    o = np.arange(4)
    cx = (x0 // C)[:, None, None] + o[None, None, :]
    cy = (y0 // C)[:, None, None] + o[None, :, None]
    msk = ((o[None, None, :] < (w // C)[:, None, None]) & (o[None, :, None] < (h // C)[:, None, None])
           & (cx < Wc) & (cy < Hc))
    bi = np.broadcast_to(np.arange(x0.size)[:, None, None], msk.shape)
    return (cy * Wc + cx)[msk], bi[msk]


# --------------------------------------------------------------------------- one run
def run_one(enc, sq, ref_dec, log):
    eid, seq = enc["id"], sq["name"]
    H, Hc, Wc = sq["H"], sq["Hc"], sq["Wc"]
    ncell = Hc * Wc
    path = os.path.join(ENC_DIR, f"{eid}__{seq}.mp4")
    t0 = time.perf_counter()
    rec = encode_sequence(sq["imgs"], path, enc["codec"], fps=FPS, **enc["knobs"])
    t_enc = time.perf_counter() - t0
    frames, t_mv = extract_mvs(path, want_qp=True)
    if len(frames) != len(sq["imgs"]):
        raise RuntimeError(f"decoded {len(frames)} frames, source has {len(sq['imgs'])}")

    # PSNR (Y, vs source converted by the same swscale path the encoder input used)
    psnr = []
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            mse = np.mean((y_plane(f, H) - sq["yref"][i]) ** 2)
            psnr.append(100.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))

    types = [f["pict_type"] for f in frames]
    prev, nxt = ref_distances(types)
    rows, frows = [], []
    # cell accumulators keyed by group variant
    acc = {}
    my_dec = {}

    def A(key):
        if key not in acc:
            acc[key] = {"frames": 0, "inframe": 0, "valid": 0, "valid_vec": 0, "zero": 0, "false_static": 0,
                        **{f"{k}@{t}": 0 for t in TAUS for k in ("reuse", "static", "tp", "stale", "flip", "flip_den")}}
        return acc[key]

    for fr in frames:
        t, ft = fr["index"], fr["pict_type"]
        qp = fr["qp"]
        frec = {"frame": t, "ftype": ft, "pkt_size": fr["pkt_size"], "psnr_y": psnr[t],
                "mean_qp": float(qp.mean()) if qp is not None else np.nan,
                "n_vectors": int(fr["mvs"].size)}
        if ft == "I" or t == 0:
            frec.update(no_vector_area=ncell * C * C, group="I")
            frows.append(frec)
            continue
        m = fr["mvs"]
        gtt = sq["gt"][t]
        tl = dst_topleft(m)
        x0 = np.rint(tl[:, 0]).astype(np.int64)
        y0 = np.rint(tl[:, 1]).astype(np.int64)
        w, h = m["w"].astype(np.int64), m["h"].astype(np.int64)
        src = m["source"].astype(np.int64)
        mv = motion_prev_to_cur(m)
        if ft == "P":
            dp = (t - prev[t]) if enc["d_known"] else 1
            d = np.full(m.size, float(dp))
            group = ("d1" if enc["d_known"] else "d1(assumed)") if dp == 1 else "dgt1"
        else:  # B
            dpast = t - prev[t] if prev[t] is not None else np.nan
            dfut = nxt[t] - t if nxt[t] is not None else np.nan
            d = np.where(src < 0, dpast, dfut).astype(float)
            group = "B"
        gx, gy, gcov, gstd, _ = gtt["g"].block_gt(x0, y0, w, h)
        gsx = np.full(m.size, np.nan)
        gsy = np.full(m.size, np.nan)
        if ft == "P" and group == "d1" and m.size:
            r = compare_frame(m, gtt["flow"], gtt["bad"])
            gsx[src == -1], gsy[src == -1] = r["gt_x"], r["gt_y"]
        bqp = qp[np.clip(y0 // 16, 0, qp.shape[0] - 1), np.clip(x0 // 16, 0, qp.shape[1] - 1)].astype(float) \
            if qp is not None else np.full(m.size, np.nan)
        rows.append(pd.DataFrame({
            "frame": t, "ftype": ft, "group": group, "x": x0, "y": y0, "w": w, "h": h,
            "dir": np.where(src < 0, "past", "future"), "d": d, "d_known": enc["d_known"],
            "mv_x": mv[:, 0], "mv_y": mv[:, 1], "qp": bqp,
            "gt_x": gx, "gt_y": gy, "gt_cover": gcov, "gt_std": gstd, "gt_src_x": gsx, "gt_src_y": gsy}))

        # ---- cells: future first, then past overrides
        cmv = np.full((ncell, 2), np.nan)
        cd = np.full(ncell, np.nan)
        cdir = np.zeros(ncell, np.int8)
        overlap = 0
        for sgn in (1, -1):
            sel = np.nonzero(src == sgn)[0]
            if sel.size == 0:
                continue
            ci, bi = paint(Hc, Wc, x0[sel], y0[sel], w[sel], h[sel])
            overlap += ci.size - np.unique(ci).size
            cmv[ci] = mv[sel][bi]
            cd[ci] = d[sel][bi]
            cdir[ci] = sgn
        has = cdir != 0
        cgx, cgy = gtt["cgx"].ravel(), gtt["cgy"].ravel()
        valid = gtt["ccov"].ravel() >= COVER_MIN
        gmag = np.hypot(cgx, cgy)
        zero = has & (cmv[:, 0] == 0) & (cmv[:, 1] == 0)
        frec.update(no_vector_area=int((~has).sum()) * C * C, overlap_cells=int(overlap),
                    excluded_cells=int((~valid).sum()), group=group)
        frows.append(frec)

        variants = [("naive", 1.0)] if group in ("d1", "d1(assumed)") else [("naive", None), ("scaled", None)]
        for vname, _ in variants:
            key = group if group in ("d1", "d1(assumed)") else f"{group}_{vname}"
            if vname == "scaled":
                use = cmv / cd[:, None] * np.where(cdir == 1, -1.0, 1.0)[:, None]
            else:
                use = cmv
            umag = np.hypot(use[:, 0], use[:, 1])
            a = A(key)
            a["frames"] += 1
            a["inframe"] += ncell
            a["valid"] += int(valid.sum())
            a["valid_vec"] += int((valid & has).sum())
            a["zero"] += int((valid & zero).sum())
            a["false_static"] += int((valid & zero & (gmag > FALSE_STATIC_GT)).sum())
            static = valid & (gmag < STATIC_GT)
            for tau in TAUS:
                reuse = has & (umag < tau)
                my_dec[(t, key, tau)] = reuse
                a[f"reuse@{tau}"] += int((valid & reuse).sum())
                a[f"static@{tau}"] += int(static.sum())
                a[f"tp@{tau}"] += int((valid & reuse & static).sum())
                a[f"stale@{tau}"] += int((valid & reuse & (gmag > STALE_GT)).sum())
                rd = ref_dec.get((seq, t, tau)) if ref_dec is not None else None
                if rd is not None:
                    a[f"flip@{tau}"] += int((reuse != rd).sum())
                    a[f"flip_den@{tau}"] += ncell

    blocks = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    fdf = pd.DataFrame(frows)
    for df in (blocks, fdf):
        df.insert(0, "encode_id", eid)
        df.insert(0, "sequence", seq)
    blocks.to_parquet(os.path.join(BLK_DIR, f"{eid}__{seq}.parquet"), index=False)
    fdf.to_parquet(os.path.join(BLK_DIR, f"{eid}__{seq}__frames.parquet"), index=False)

    # ---- encode-level
    n = len(frames)
    pq = [f["qp"].mean() for f in frames if f["pict_type"] == "P" and f["qp"] is not None]
    enc_level = {
        "file_bytes": rec["file_bytes"], "packet_bytes": rec["packet_bytes"],
        "bitrate_kbps": rec["packet_bytes"] * 8 * FPS / n / 1000.0,
        "psnr_y": float(np.mean(psnr)), "mean_p_qp": float(np.mean(pq)) if pq else np.nan,
        "n_I": types.count("I"), "n_P": types.count("P"), "n_B": types.count("B"), "n_frames": n,
        "overlap_cells_total": int(fdf.get("overlap_cells", pd.Series(dtype=float)).fillna(0).sum()),
    }

    # ---- block-level EPE per group variant
    out = []
    for key, a in acc.items():
        g = key.split("_")[0] if key.startswith(("dgt1", "B")) else key
        vname = "scaled" if key.endswith("_scaled") else "naive"
        b = blocks[blocks["group"] == g]
        area = (b["w"] * b["h"]).to_numpy(float)
        kept = (b["gt_cover"] >= COVER_MIN).to_numpy()
        mvx, mvy = b["mv_x"].to_numpy(), b["mv_y"].to_numpy()
        if vname == "scaled":
            sgn = np.where(b["dir"].to_numpy() == "future", -1.0, 1.0)
            mvx, mvy = mvx / b["d"].to_numpy() * sgn, mvy / b["d"].to_numpy() * sgn
        epe = np.hypot(mvx - b["gt_x"].to_numpy(), mvy - b["gt_y"].to_numpy())
        ek, ak = epe[kept], area[kept]
        base = {"encode_id": eid, "sequence": seq, "codec": enc["codec"], "factors": "|".join(enc["factors"]),
                "group": key, "d_known": enc["d_known"], **enc_level,
                "n_group_frames": a["frames"], "n_vectors": int(len(b)),
                "epe_median": wmedian(ek, ak), "epe_mean": float(np.average(ek, weights=ak)) if ak.sum() else np.nan,
                "epe_gt3_area_pct": 100 * float(ak[ek > EPE_BIG].sum() / ak.sum()) if ak.sum() else np.nan,
                "block_excluded_area_share": float(area[~kept].sum() / area.sum()) if area.sum() else np.nan,
                "cell_excluded_share": 1 - a["valid"] / a["inframe"] if a["inframe"] else np.nan,
                "mv_coverage_pct": 100 * a["valid_vec"] / a["valid"] if a["valid"] else np.nan,
                "zero_mv_share": a["zero"] / a["valid"] if a["valid"] else np.nan,
                "false_static_of_zero": a["false_static"] / a["zero"] if a["zero"] else np.nan,
                "false_static_of_valid": a["false_static"] / a["valid"] if a["valid"] else np.nan,
                "zero_cells": a["zero"], "false_static_cells": a["false_static"], "valid_cells": a["valid"]}
        for tau in TAUS:
            r, s, tp, st = (a[f"reuse@{tau}"], a[f"static@{tau}"], a[f"tp@{tau}"], a[f"stale@{tau}"])
            fd = a[f"flip_den@{tau}"]
            out.append({**base, "tau": tau,
                        "flip_rate": (a[f"flip@{tau}"] / fd) if fd else (0.0 if eid == REF_ID else np.nan),
                        "flip_cells": a[f"flip@{tau}"], "flip_den_cells": fd,
                        "reuse_cells": r, "gt_static_cells": s, "tp_cells": tp,
                        "precision": tp / r if r else np.nan, "recall": tp / s if s else np.nan,
                        "stale_cells": st, "stale_of_reuse": st / r if r else np.nan,
                        "stale_of_valid": st / a["valid"] if a["valid"] else np.nan})
    log.update({"options": rec["options"], "x264_sei": rec["x264_sei"], "encode_s": t_enc,
                "encoder_reported_s": rec["seconds"], "mv_extract_s": t_mv, "run_s": time.perf_counter() - t0,
                **{k: enc_level[k] for k in ("file_bytes", "bitrate_kbps", "psnr_y", "n_I", "n_P", "n_B",
                                             "overlap_cells_total")},
                "types": "".join(types)})
    return out, my_dec


def main():
    T0 = time.perf_counter()
    man_p = os.path.join(HERE, "manifest.json")
    man = json.load(open(man_p))
    seqs = man["sequences"]
    os.makedirs(ENC_DIR, exist_ok=True)
    os.makedirs(BLK_DIR, exist_ok=True)
    print("loading", seqs, flush=True)
    cache = {s: load_seq(s) for s in seqs}
    print(f"loaded in {time.perf_counter() - T0:.0f}s", flush=True)

    results, logs, failures = [], [], []
    ref_dec = {}
    for enc in ENCODES:
        for s in seqs:
            lg = {"encode_id": enc["id"], "sequence": s, "codec": enc["codec"], "knobs": enc["knobs"]}
            try:
                out, dec = run_one(enc, cache[s], None if enc["id"] == REF_ID else ref_dec, lg)
                if enc["id"] == REF_ID:
                    for (t, key, tau), v in dec.items():
                        ref_dec[(s, t, tau)] = v
                results += out
                lg["status"] = "ok"
                print(f"{enc['id']:22s} {s:10s} {lg['bitrate_kbps']:9.0f} kbps  PSNR {lg['psnr_y']:5.2f}  "
                      f"{lg['types'][:14]}  {lg['run_s']:.1f}s", flush=True)
            except Exception as e:
                lg["status"] = "FAILED"
                lg["error"] = repr(e)
                lg["traceback"] = traceback.format_exc()
                failures.append({"encode_id": enc["id"], "sequence": s, "error": repr(e)})
                print("FAILED", enc["id"], s, repr(e), flush=True)
            logs.append(lg)
            pd.DataFrame(results).to_csv(os.path.join(HERE, "results.csv"), index=False)
            json.dump({"logs": logs, "failures": failures}, open(os.path.join(HERE, "encode_log.json"), "w"),
                      indent=1, default=str)
    total = time.perf_counter() - T0
    man["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    man["pilot_runtime_s"] = total
    man["n_failures"] = len(failures)
    json.dump(man, open(man_p, "w"), indent=1)
    print(f"done in {total:.0f}s, failures: {len(failures)}")


if __name__ == "__main__":
    main()
