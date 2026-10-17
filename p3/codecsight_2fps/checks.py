"""CodecSight-matched 2 fps arm: the pre-registered checks, computed BEFORE any policy, gate or stale metric
(PREREGISTRATION.md cc2eaf9, §2, §3, §5). Inputs: data/<clip>/raft.npz, chain.npz, x264_crf12.npz,
encodes/<clip>/x264_crf12.mp4 (float vectors re-extracted), the sampled source frames, manifest.json.

  §5 motion count  valid cell-frames (splat_count >= 8) with |T_j| > 2 px, j = 1..n-1; counted if >= 2,000; N.
  §3 R-chain       per clip median r = |D - F| / |F| pooled over the clip's chain pixels (|F| > 2 px);
                   clip PASS if r < 0.20; overall PASS if >= ceil(0.75 N) counted clips pass.
  §3 V0-2fps       x264 CRF 12, P-frames with d_past 1; valid, |RAFT| > 2 px, a vector, texture >= q75 (texture of
                   the sampled source frame j, analyze_virat.cell_texture; q75 over all cells of frames 1..n-1 of
                   all windows, histogram as analyze_confirm); per clip median |MV - RAFT| (float vectors,
                   analyze_virat.cell_vectors); PASS < 1.5 px in >= ceil(0.75 N) counted clips. Also on
                   2 < |RAFT| <= 16 px.
  §3 FB            |f + b_w|^2 < 0.01 (|f|^2 + |b_w|^2) + 0.5 per cell (i08b); consistent share of valid and of
                   moving cells.
  §2 encodes       status, check mismatches and I-frame positions of every encode in manifest.json.
Writes checks.md, checks.csv. No policy, gate, reuse or stale quantity is computed.
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(P3, "virat"))
import collect  # noqa: E402
from analyze_virat import cell_texture, cell_vectors, TEX_BIN, TEX_MAX  # noqa: E402
from lib.mvs import extract_mvs  # noqa: E402

COVER_COUNT, STALE_GT, MOTION_MIN, N_MIN = 8, 2.0, 2000, 10
R_MAX, V0_ERR, V0_CAP, SHARE = 0.20, 1.5, 16.0, 0.75


def need(n, x):
    return math.ceil(x * n - 1e-9)


def one(clip):
    t0 = time.perf_counter()
    d = os.path.join(HERE, "data", clip)
    rz = np.load(os.path.join(d, "raft.npz"))
    sp = rz["splat_t"].astype(np.float32)
    nm1, Hc, Wc = sp.shape[:3]
    n, ncell = nm1 + 1, Hc * Wc
    sp = sp.reshape(nm1, ncell, 2)
    valid = rz["splat_count"].reshape(nm1, ncell) >= COVER_COUNT
    gmag = np.hypot(sp[..., 0], sp[..., 1])
    mov = valid & (gmag > STALE_GT)
    # FB
    b = rz["bwd_t"].astype(np.float32).reshape(nm1, ncell, 2)
    s = sp + b
    ok = (s ** 2).sum(-1) < 0.01 * ((sp ** 2).sum(-1) + (b ** 2).sum(-1)) + 0.5
    # chain
    ch = np.load(os.path.join(d, "chain.npz"))
    r, e = ch["r"], ch["e"]
    # texture of sampled source frames 1..n-1
    w = collect.windows()[clip]
    src = collect.sampled_yuv(clip, w)
    assert len(src) == n
    H = src[0].shape[0] * 2 // 3
    tex = np.zeros((n, ncell), np.float32)
    for t in range(1, n):
        tex[t] = cell_texture(src[t][:H].astype(np.float64)).ravel()
    tex_hist = np.bincount(np.minimum((tex[1:] / TEX_BIN).astype(np.int64), int(TEX_MAX / TEX_BIN) - 1).ravel(),
                           minlength=int(TEX_MAX / TEX_BIN))
    # V0 candidates: x264 CRF 12, P-frames d_past 1
    a = np.load(os.path.join(d, "x264_crf12.npz"))
    q = a["mv_q"].reshape(n, ncell)
    fr, _ = extract_mvs(os.path.join(HERE, "encodes", clip, "x264_crf12.mp4"))
    vec = cell_vectors(fr, Hc, Wc)
    del fr
    P = [t for t in range(1, n) if str(a["ftype"][t]) == "P" and a["d_past"][t] == 1]
    sel = np.zeros((n, ncell), bool)
    sel[P] = valid[np.array(P) - 1] & mov[np.array(P) - 1] & (q[P] != 255)
    tt, cc = np.nonzero(sel)
    err = np.hypot(vec[tt, cc, 0] - sp[tt - 1, cc, 0], vec[tt, cc, 1] - sp[tt - 1, cc, 1])
    return {"clip": clip, "n": n, "valid_cf": int(valid.sum()), "moving_cf": int(mov.sum()),
            "fb_cons_valid": float(ok[valid].mean()), "fb_cons_moving": float(ok[mov].mean()) if mov.any() else np.nan,
            "mov_gt16": float((gmag[mov] > 16).mean()) if mov.any() else np.nan,
            "mov_gt32": float((gmag[mov] > 32).mean()) if mov.any() else np.nan,
            "mov_gt64": float((gmag[mov] > 64).mean()) if mov.any() else np.nan,
            "chain_px": int(r.size), "chain_r": float(np.median(r)) if r.size else np.nan,
            "chain_e_px": float(np.median(e)) if e.size else np.nan,
            "v0_err": err.astype(np.float32), "v0_tex": tex[tt, cc], "v0_raft": gmag[tt - 1, cc],
            "tex_hist": tex_hist, "run_s": time.perf_counter() - t0}


def main():
    T0 = time.perf_counter()
    man = json.load(open(os.path.join(HERE, "manifest.json")))
    clips = sorted(man["clips"])
    with mp.get_context("spawn").Pool(6) as pool:
        outs = {o["clip"]: o for o in pool.map(one, clips)}
    hist = sum(outs[c]["tex_hist"] for c in clips)
    cum = np.cumsum(hist)
    q75 = (np.searchsorted(cum, 0.75 * cum[-1]) + 1) * TEX_BIN
    rows = []
    for c in clips:
        o = outs[c]
        m = o["v0_tex"] >= q75
        e_all = o["v0_err"][m]
        e_cap = o["v0_err"][m & (o["v0_raft"] <= V0_CAP)]
        rows.append({"clip": c, "unseen": man["clips"][c]["start"] >= 600, "gops": man["clips"][c]["gops"], "pairs": o["n"] - 1,
                     "valid_cf": o["valid_cf"], "moving_cf": o["moving_cf"], "counted": o["moving_cf"] >= MOTION_MIN,
                     "chain_px": o["chain_px"], "chain_r": o["chain_r"], "chain_e_px": o["chain_e_px"],
                     "chain_pass": bool(o["chain_r"] < R_MAX),
                     "v0_cells": int(m.sum()), "v0_err": float(np.nanmedian(e_all)) if e_all.size else np.nan,
                     "v0_pass": bool(e_all.size and np.nanmedian(e_all) < V0_ERR),
                     "v0le16_cells": int(e_cap.size), "v0le16_err": float(np.nanmedian(e_cap)) if e_cap.size else np.nan,
                     "v0le16_pass": bool(e_cap.size and np.nanmedian(e_cap) < V0_ERR),
                     "fb_cons_valid": o["fb_cons_valid"], "fb_cons_moving": o["fb_cons_moving"],
                     "mov_gt16": o["mov_gt16"], "mov_gt32": o["mov_gt32"], "mov_gt64": o["mov_gt64"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "checks.csv"), index=False)
    cnt = df[df.counted]
    N = len(cnt)
    k = need(N, SHARE)
    rc, v0, v16 = int(cnt.chain_pass.sum()), int(cnt.v0_pass.sum()), int(cnt.v0le16_pass.sum())
    runs = man["runs"]
    bad_i = [(r["clip"], r["arm"], r.get("iframes")) for r in runs
             if r.get("status") == "ok" and r["iframes"] != list(range(0, r["frames_decoded"], 16))]
    mism = [(r["clip"], r["arm"], r["check_mismatches"]) for r in runs if r.get("status") == "ok" and r["check_mismatches"]]
    notok = [(r["clip"], r["arm"], r.get("status")) for r in runs if r.get("status") != "ok"]
    raft_bad = [c for c in clips if man["raft"].get(c, {}).get("status") != "ok"]
    L = ["# CodecSight-matched 2 fps arm: pre-registered checks (before any policy or gate metric)", "",
         f"Prereg {collect.PREREG_COMMIT}; collection HEAD {man['head'][:7]}, uncommitted '{man['uncommitted']}', "
         f"code sha256 {man['code_sha256'][:12]}. Generated by checks.py in {time.perf_counter() - T0:.0f} s.", "",
         "## Encodes (§2)", "",
         f"- encodes in manifest: {len(runs)}; status ok: {len(runs) - len(notok)}; not ok: {notok or 'none'}",
         f"- I-frames not exactly at 0, 16, 32, ...: {bad_i or 'none'}",
         f"- encodes with any check mismatch: {mism or 'none'}",
         f"- RAFT not ok: {raft_bad or 'none'}", "",
         "## Summary", "",
         f"- §5 counted clips (moving cell-frames >= {MOTION_MIN}): N = {N}/{len(df)}"
         f"{'' if N >= N_MIN else ' -> N < 10: everything descriptive'}; not counted: "
         f"{', '.join(df.loc[~df.counted, 'clip']) or 'none'}",
         f"- R-chain: {rc}/{N} counted clips with median r < {R_MAX} (need >= {k}) -> **{'PASS' if rc >= k else 'FAIL'}**",
         f"- V0-2fps (all |RAFT| > 2 px): {v0}/{N} with median |MV - RAFT| < {V0_ERR} px (need >= {k}) -> "
         f"**{'PASS' if v0 >= k else 'FAIL'}**; texture q75 = {q75:.3f} grey levels/px",
         f"- V0-2fps (2 < |RAFT| <= 16 px): {v16}/{N} (need >= {k}) -> **{'PASS' if v16 >= k else 'FAIL'}**",
         f"- FB consistent share, median over all clips: valid {df.fb_cons_valid.median():.4f}, "
         f"moving {df.fb_cons_moving.median():.4f}",
         f"- moving cells with |T| > 16 / 32 / 64 px, median over clips: {df.mov_gt16.median():.3f} / "
         f"{df.mov_gt32.median():.3f} / {df.mov_gt64.median():.3f}", "",
         "## Per clip", "",
         df.to_string(index=False, float_format=lambda v: f"{v:.4f}")]
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, "checks.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
