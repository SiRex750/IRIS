"""Paper 3 exploratory pilot v2 (fixes from the v1 audit). Outputs under p3/pilot_v2/.

Changes vs v1 (p3/pilot/run_pilot.py):
  - every libx264 encode: threads=1, sliced-threads=0 (v1 ran 7 slices/frame, CPU-dependent)
  - every mpeg4 encode: threads=1 (v1 mpeg4 ran 16 video packets/frame, one per thread)
  - B-frame arms: fixed pattern (x264 b-adapt=0) and matched-QP variants; decoded I/P/B QP logged
  - search-range arm: x264 me=umh merange 16/32/64
  - headline gate metrics: stale / valid and missed reuse / valid (GT-static cells not reused)
  - flip rate vs x264 CRF 12 and vs the encoder's own best setting (NVENC QP 18, mpeg4 q 2)
  - stale cells split by GT speed 2-8, 8-16, 16-32, >32 px
  - slices per frame counted from the bitstream for every encode; x264 SEI threads checked
Definitions otherwise as in v1 (see that docstring): d, vector groups, frame-t GT grid, cells, gate.
"""
import collections
import json
import os
import subprocess
import sys
import time
import traceback

import av
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
REPO = os.path.dirname(P3)
sys.path.insert(0, P3)
sys.path.insert(0, os.path.join(P3, "validate"))
sys.path.insert(0, os.path.join(P3, "pilot"))

from lib.encode import encode_sequence, sei_value  # noqa: E402
from lib.mvs import extract_mvs, motion_prev_to_cur, dst_topleft  # noqa: E402
from lib.compare import compare_frame  # noqa: E402
from val_D_gt_grid import slice_ref_idc  # noqa: E402
from run_pilot import load_seq, wmedian, y_plane, ref_distances, paint  # noqa: E402  (v1 helpers, unchanged)

ENC_DIR = os.path.join(HERE, "encodes")
BLK_DIR = os.path.join(HERE, "blocks")
TAUS = (0.5, 1.0, 2.0)
STATIC_GT, FALSE_STATIC_GT, STALE_GT = 0.5, 1.0, 2.0
SPEED_BINS = ((2, 8), (8, 16), (16, 32), (32, np.inf))
EPE_BIG, COVER_MIN, C, FPS = 3.0, 0.5, 4, 24
X264_REF = "x264_crf12"
OWN_REF = {"libx264": "x264_crf12", "h264_nvenc": "nvenc_qp18", "mpeg4": "mpeg4_q2"}

X = dict(preset="medium", bframes=0, keyint=250, ref=1, threads=1)
XB = dict(X, bframes=2, b_adapt=0, b_pyramid="none")
N = dict(bframes=0, keyint=250, ref=1, preset="p4")
NB = dict(N, bframes=2, b_ref_mode="disabled")
M = dict(bframes=0, keyint=250, threads=1)


def _e(eid, codec, factors, d_known=True, extra=False, **kn):
    return {"id": eid, "codec": codec, "factors": factors, "d_known": d_known, "extra": extra, "knobs": kn}


ENCODES = (
    [_e("x264_crf12", "libx264", ["crf"], crf=12, **X)]
    + [_e(f"x264_crf{c}", "libx264", ["crf"] + (["ref", "preset", "keyint", "bframes", "merange", "encoder"]
                                               if c == 23 else []), crf=c, **X) for c in (18, 23, 28, 33, 38, 45)]
    + [_e("x264_crf23_ref2", "libx264", ["ref"], d_known=False, **{**X, "crf": 23, "ref": 2}),
       _e("x264_crf23_ref3", "libx264", ["ref"], d_known=False, **{**X, "crf": 23, "ref": 3}),
       _e("x264_crf23_ultrafast", "libx264", ["preset"], **{**X, "crf": 23, "preset": "ultrafast"}),
       _e("x264_crf23_keyint30", "libx264", ["keyint"], **{**X, "crf": 23, "keyint": 30})]
    + [_e(f"x264_crf23_umh{r}", "libx264", ["merange"], crf=23, me="umh", merange=r, **X) for r in (16, 32, 64)]
    + [_e("x264_crf23_bf2", "libx264", ["bframes"], crf=23, **XB),
       _e("x264_crf23_bf2_pb1", "libx264", ["bframes"], crf=23, pbratio=1.0, **XB),
       # extra (not in the v2 prompt): constant-QP pair, because CRF cannot hold B QP = P QP
       _e("x264_qp24", "libx264", ["bframes_cqp"], extra=True, qp=24, **X),
       _e("x264_qp24_bf2_pb1", "libx264", ["bframes_cqp"], extra=True, qp=24, pbratio=1.0, **XB)]
    + [_e(f"nvenc_qp{q}", "h264_nvenc", ["nvenc_qp"] + (["bframes", "encoder"] if q == 28 else []), qp=q, **N)
       for q in (18, 23, 28, 33, 38, 45)]
    + [_e("nvenc_qp28_bf2", "h264_nvenc", ["bframes"], qp=28, **NB),
       _e("nvenc_qp28_bf2_bqeq", "h264_nvenc", ["bframes"], qp=28, b_qfactor=1.0, b_qoffset=0, **NB)]
    + [_e(f"mpeg4_q{q}", "mpeg4", ["mpeg4_q"] + (["encoder"] if q == 4 else []), q=q, **M) for q in (2, 4, 8, 16, 31)]
    + [_e("x264_crf23_veryslow", "libx264", ["preset"], **{**X, "crf": 23, "preset": "veryslow"})]
)


# --------------------------------------------------------------------------- bitstream slice counts
def mpeg4_packets_per_frame(path):
    """1 + number of byte-aligned MPEG-4 Part 2 resync markers (>=16 zeros then a 1, not a start code)."""
    out = []
    with av.open(path) as c:
        st = c.streams.video[0]
        for pk in c.demux(st):
            b = bytes(pk)
            if not b:
                continue
            a = np.frombuffer(b, np.uint8)
            i = np.nonzero((a[:-2] == 0) & (a[1:-1] == 0) & (a[2:] != 0) & (a[2:] != 1))[0]
            out.append(1 + int(i.size))
    return out


def slices_per_frame(path, codec):
    if codec == "mpeg4":
        return mpeg4_packets_per_frame(path)
    cnt = collections.Counter(r[0] for r in slice_ref_idc(path))
    return [cnt[k] for k in sorted(cnt)]


# --------------------------------------------------------------------------- one run
def run_one(enc, sq, refs, log):
    """refs: {ref_encode_id: {(seq, t, tau): decision}} for the flip references already run."""
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
    spf = slices_per_frame(path, enc["codec"])

    psnr = []
    with av.open(path) as c:
        for i, f in enumerate(c.decode(video=0)):
            mse = np.mean((y_plane(f, H) - sq["yref"][i]) ** 2)
            psnr.append(100.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))

    types = [f["pict_type"] for f in frames]
    prev, nxt = ref_distances(types)
    rows, frows, acc, my_dec = [], [], {}, {}
    flip_refs = {"x264ref": X264_REF, "ownref": OWN_REF[enc["codec"]]}

    def A(key):
        if key not in acc:
            a = {"frames": 0, "inframe": 0, "valid": 0, "valid_vec": 0, "zero": 0, "false_static": 0}
            for t in TAUS:
                for k in ("reuse", "static", "tp", "missed", "stale"):
                    a[f"{k}@{t}"] = 0
                for lo, hi in SPEED_BINS:
                    a[f"stale_{lo}_{hi}@{t}"] = 0
                for fr_ in flip_refs:
                    a[f"flip_{fr_}@{t}"] = 0
                    a[f"flipden_{fr_}@{t}"] = 0
            acc[key] = a
        return acc[key]

    for fr in frames:
        t, ft, qp = fr["index"], fr["pict_type"], fr["qp"]
        frec = {"frame": t, "ftype": ft, "pkt_size": fr["pkt_size"], "psnr_y": psnr[t],
                "mean_qp": float(qp.mean()) if qp is not None else np.nan,
                "n_vectors": int(fr["mvs"].size), "slices": spf[t] if t < len(spf) else np.nan}
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
        else:
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

        cmv = np.full((ncell, 2), np.nan)
        cd = np.full(ncell, np.nan)
        cdir = np.zeros(ncell, np.int8)
        overlap = 0
        for sgn in (1, -1):  # future first, past overrides
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
        static = valid & (gmag < STATIC_GT)
        frec.update(no_vector_area=int((~has).sum()) * C * C, overlap_cells=int(overlap),
                    excluded_cells=int((~valid).sum()), group=group)
        frows.append(frec)

        variants = ["naive"] if group in ("d1", "d1(assumed)") else ["naive", "scaled"]
        for vname in variants:
            key = group if group in ("d1", "d1(assumed)") else f"{group}_{vname}"
            use = cmv / cd[:, None] * np.where(cdir == 1, -1.0, 1.0)[:, None] if vname == "scaled" else cmv
            umag = np.hypot(use[:, 0], use[:, 1])
            a = A(key)
            a["frames"] += 1
            a["inframe"] += ncell
            a["valid"] += int(valid.sum())
            a["valid_vec"] += int((valid & has).sum())
            a["zero"] += int((valid & zero).sum())
            a["false_static"] += int((valid & zero & (gmag > FALSE_STATIC_GT)).sum())
            for tau in TAUS:
                reuse = has & (umag < tau)
                my_dec[(seq, t, tau, key)] = reuse
                vr = valid & reuse
                a[f"reuse@{tau}"] += int(vr.sum())
                a[f"static@{tau}"] += int(static.sum())
                a[f"tp@{tau}"] += int((vr & static).sum())
                a[f"missed@{tau}"] += int((static & ~reuse).sum())
                a[f"stale@{tau}"] += int((vr & (gmag > STALE_GT)).sum())
                for lo, hi in SPEED_BINS:
                    a[f"stale_{lo}_{hi}@{tau}"] += int((vr & (gmag > max(lo, STALE_GT)) & (gmag <= hi)).sum()) \
                        if lo == 2 else int((vr & (gmag > lo) & (gmag <= hi)).sum())
                for frn, rid in flip_refs.items():
                    rd = refs.get(rid, {}).get((seq, t, tau))
                    if rd is not None:
                        a[f"flip_{frn}@{tau}"] += int((reuse != rd).sum())
                        a[f"flipden_{frn}@{tau}"] += ncell

    blocks = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    fdf = pd.DataFrame(frows)
    for df in (blocks, fdf):
        df.insert(0, "encode_id", eid)
        df.insert(0, "sequence", seq)
    blocks.to_parquet(os.path.join(BLK_DIR, f"{eid}__{seq}.parquet"), index=False)
    fdf.to_parquet(os.path.join(BLK_DIR, f"{eid}__{seq}__frames.parquet"), index=False)

    n = len(frames)
    qmean = {tt: [f["qp"].mean() for f in frames if f["pict_type"] == tt and f["qp"] is not None] for tt in "IPB"}
    sei = rec["x264_sei"]
    enc_level = {
        "file_bytes": rec["file_bytes"], "packet_bytes": rec["packet_bytes"],
        "bitrate_kbps": rec["packet_bytes"] * 8 * FPS / n / 1000.0,
        "psnr_y": float(np.mean(psnr)),
        "mean_i_qp": float(np.mean(qmean["I"])) if qmean["I"] else np.nan,
        "mean_p_qp": float(np.mean(qmean["P"])) if qmean["P"] else np.nan,
        "mean_b_qp": float(np.mean(qmean["B"])) if qmean["B"] else np.nan,
        "n_I": types.count("I"), "n_P": types.count("P"), "n_B": types.count("B"), "n_frames": n,
        "slices_min": int(min(spf)), "slices_max": int(max(spf)),
        "sei_threads": sei_value(sei, "threads"), "sei_sliced_threads": sei_value(sei, "sliced_threads"),
        "sei_slices": sei_value(sei, "slices"),
        "overlap_cells_total": int(fdf.get("overlap_cells", pd.Series(dtype=float)).fillna(0).sum()),
    }

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
        V = a["valid"]
        base = {"encode_id": eid, "sequence": seq, "codec": enc["codec"], "factors": "|".join(enc["factors"]),
                "extra_arm": enc["extra"], "group": key, "d_known": enc["d_known"], **enc_level,
                "n_group_frames": a["frames"], "n_vectors": int(len(b)),
                "epe_median": wmedian(ek, ak), "epe_mean": float(np.average(ek, weights=ak)) if ak.sum() else np.nan,
                "epe_gt3_area_pct": 100 * float(ak[ek > EPE_BIG].sum() / ak.sum()) if ak.sum() else np.nan,
                "block_excluded_area_share": float(area[~kept].sum() / area.sum()) if area.sum() else np.nan,
                "cell_excluded_share": 1 - V / a["inframe"] if a["inframe"] else np.nan,
                "mv_coverage_pct": 100 * a["valid_vec"] / V if V else np.nan,
                "zero_mv_share": a["zero"] / V if V else np.nan,
                "false_static_of_zero": a["false_static"] / a["zero"] if a["zero"] else np.nan,
                "false_static_of_valid": a["false_static"] / V if V else np.nan,
                "zero_cells": a["zero"], "false_static_cells": a["false_static"], "valid_cells": V}
        for tau in TAUS:
            r, s, tp, st, mi = (a[f"reuse@{tau}"], a[f"static@{tau}"], a[f"tp@{tau}"], a[f"stale@{tau}"],
                                a[f"missed@{tau}"])
            row = {**base, "tau": tau,
                   "stale_of_valid": st / V if V else np.nan, "missed_of_valid": mi / V if V else np.nan,
                   "stale_of_reuse": st / r if r else np.nan,
                   "reuse_cells": r, "gt_static_cells": s, "tp_cells": tp, "missed_cells": mi, "stale_cells": st,
                   "precision": tp / r if r else np.nan, "recall": tp / s if s else np.nan}
            for lo, hi in SPEED_BINS:
                lab = f"{lo}_{'inf' if hi == np.inf else hi}"
                row[f"stale_cells_{lab}"] = a[f"stale_{lo}_{hi}@{tau}"]
                row[f"stale_of_valid_{lab}"] = a[f"stale_{lo}_{hi}@{tau}"] / V if V else np.nan
            for frn, rid in flip_refs.items():
                den = a[f"flipden_{frn}@{tau}"]
                row[f"flip_{frn}"] = (a[f"flip_{frn}@{tau}"] / den) if den else (0.0 if eid == rid else np.nan)
                row[f"flip_{frn}_id"] = rid
                row[f"flip_{frn}_cells"] = a[f"flip_{frn}@{tau}"]
                row[f"flip_{frn}_den"] = den
            out.append(row)
    log.update({"options": rec["options"], "x264_sei": sei, "encode_s": t_enc, "encoder_reported_s": rec["seconds"],
                "mv_extract_s": t_mv, "run_s": time.perf_counter() - t0, "slices_per_frame": spf,
                **{k: enc_level[k] for k in ("file_bytes", "bitrate_kbps", "psnr_y", "n_I", "n_P", "n_B",
                                             "mean_i_qp", "mean_p_qp", "mean_b_qp", "slices_min", "slices_max",
                                             "sei_threads", "sei_sliced_threads", "sei_slices",
                                             "overlap_cells_total")},
                "types": "".join(types)})
    return out, my_dec


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main():
    T0 = time.perf_counter()
    v1man = json.load(open(os.path.join(P3, "pilot", "manifest.json")))
    seqs = v1man["sequences"]
    man = {k: v1man[k] for k in ("python", "pyav", "ffmpeg", "ffmpeg_libs", "x264_build", "nvidia", "numpy",
                                 "sequences", "static_share", "pass", "gt")}
    man.update({"start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_hash": git("rev-parse", "HEAD"),
                "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "uncommitted_lib_changes": git("status", "--porcelain", "--", "p3/lib"),
                "lib_encode_diff": git("diff", "--", "p3/lib/encode.py")})
    json.dump(man, open(os.path.join(HERE, "manifest.json"), "w"), indent=1)
    os.makedirs(ENC_DIR, exist_ok=True)
    os.makedirs(BLK_DIR, exist_ok=True)
    print("loading", seqs, flush=True)
    cache = {s: load_seq(s) for s in seqs}
    print(f"loaded in {time.perf_counter() - T0:.0f}s", flush=True)

    ref_ids = set(OWN_REF.values()) | {X264_REF}
    refs = {rid: {} for rid in ref_ids}
    results, logs, failures = [], [], []
    for enc in ENCODES:
        for s in seqs:
            lg = {"encode_id": enc["id"], "sequence": s, "codec": enc["codec"], "knobs": enc["knobs"],
                  "extra_arm": enc["extra"]}
            try:
                out, dec = run_one(enc, cache[s], refs, lg)
                if enc["id"] in ref_ids:
                    for (sq_, t, tau, key), v in dec.items():
                        if key in ("d1", "d1(assumed)"):
                            refs[enc["id"]][(sq_, t, tau)] = v
                results += out
                lg["status"] = "ok"
                print(f"{enc['id']:22s} {s:10s} {lg['bitrate_kbps']:8.0f} kbps PSNR {lg['psnr_y']:5.2f} "
                      f"slices {lg['slices_min']}-{lg['slices_max']} thr {lg['sei_threads']}/{lg['sei_sliced_threads']} "
                      f"QP I/P/B {lg['mean_i_qp']:.1f}/{lg['mean_p_qp']:.1f}/{lg['mean_b_qp']:.1f} "
                      f"{lg['types'][:12]} {lg['run_s']:.1f}s", flush=True)
            except Exception as e:
                lg.update(status="FAILED", error=repr(e), traceback=traceback.format_exc())
                failures.append({"encode_id": enc["id"], "sequence": s, "error": repr(e)})
                print("FAILED", enc["id"], s, repr(e), flush=True)
            logs.append(lg)
            pd.DataFrame(results).to_csv(os.path.join(HERE, "results.csv"), index=False)
            json.dump({"logs": logs, "failures": failures}, open(os.path.join(HERE, "encode_log.json"), "w"),
                      indent=1, default=str)
    total = time.perf_counter() - T0
    man.update({"end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pilot_runtime_s": total,
                "n_failures": len(failures)})
    json.dump(man, open(os.path.join(HERE, "manifest.json"), "w"), indent=1)
    print(f"done in {total:.0f}s, failures: {len(failures)}")


if __name__ == "__main__":
    main()
