"""Validation A: synthetic integer translation -> recovered codec motion must equal the shift.

Frame i (i = 0..9) is a 960x384 window of one Sintel frame whose origin moves by -(dx, dy)
per frame, so scene content moves by +(dx, dy) pixels per frame (frame i-1 -> i).
Recovered motion per block = dst - src = -motion/motion_scale (see lib/mvs.py). Only interior
blocks (dst footprint >= MARGIN px from every border) of P-frames with source == -1 count.

Primary config (pass/fail): libx264 CRF 12, bframes 0, ref=1.
Secondary (reported, not gating): libx264 default ref, h264_nvenc QP 15, mpeg4 q 2.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, P3)

from lib.flo import SintelSeq  # noqa: E402
from lib.encode import encode_sequence, sei_value  # noqa: E402
from lib.mvs import extract_mvs, motion_prev_to_cur, dst_topleft  # noqa: E402

SINTEL = r"C:\Users\akash\Documents\datasets\MPI-Sintel"
WORK = os.path.join(P3, "work", "val_A")
SHIFTS = [(4, 0), (0, 3), (-5, 2), (3, -4)]
NF, CW, CH, MARGIN = 10, 960, 384, 32

CONFIGS = {
    "x264_crf12_bf0_ref1": ("libx264", dict(crf=12, preset="medium", bframes=0, keyint=250, ref=1)),
    "x264_crf12_bf0_refdefault": ("libx264", dict(crf=12, preset="medium", bframes=0, keyint=250)),
    "nvenc_qp15_bf0_ref1": ("h264_nvenc", dict(qp=15, bframes=0, keyint=250, ref=1)),
    "mpeg4_q2_bf0": ("mpeg4", dict(q=2, bframes=0, keyint=250)),
}
PRIMARY = "x264_crf12_bf0_ref1"


def make_frames(base, dx, dy):
    H, W = base.shape[:2]
    x0, y0 = max(0, (NF - 1) * dx), max(0, (NF - 1) * dy)
    out = []
    for i in range(NF):
        x, y = x0 - i * dx, y0 - i * dy
        assert 0 <= x and x + CW <= W and 0 <= y and y + CH <= H, (dx, dy, i)
        out.append(np.ascontiguousarray(base[y:y + CH, x:x + CW]))
    return out


def analyse(frames, dx, dy):
    per_frame, all_mv, all_int = [], [], []
    types = "".join(f["pict_type"] for f in frames)
    for fr in frames:
        m = fr["mvs"]
        if fr["pict_type"] != "P" or m.size == 0:
            continue
        m = m[m["source"] == -1]
        tl = dst_topleft(m)
        interior = ((tl[:, 0] >= MARGIN) & (tl[:, 1] >= MARGIN) &
                    (tl[:, 0] + m["w"] <= CW - MARGIN) & (tl[:, 1] + m["h"] <= CH - MARGIN))
        m = m[interior]
        mv = motion_prev_to_cur(m)
        mv_int = np.stack([m["dst_x"] - m["src_x"], m["dst_y"] - m["src_y"]], 1).astype(float)
        err = np.linalg.norm(mv - [dx, dy], axis=1)
        per_frame.append(float(np.median(err)))
        all_mv.append(mv)
        all_int.append(mv_int)
    mv = np.concatenate(all_mv)
    mv_int = np.concatenate(all_int)
    err = np.linalg.norm(mv - [dx, dy], axis=1)
    err_int = np.linalg.norm(mv_int - [dx, dy], axis=1)
    err_neg = np.linalg.norm(mv + [dx, dy], axis=1)  # what a sign flip would give
    return {
        "pict_types": types,
        "n_blocks": int(mv.shape[0]),
        "median_mv": [float(np.median(mv[:, 0])), float(np.median(mv[:, 1]))],
        "median_err_px": float(np.median(err)),
        "median_err_px_int_dst_minus_src": float(np.median(err_int)),
        "median_err_if_sign_flipped": float(np.median(err_neg)),
        "frac_err_lt_0.25": float((err < 0.25).mean()),
        "frac_err_lt_0.5": float((err < 0.5).mean()),
        "frac_zero_mv": float((np.abs(mv).sum(1) == 0).mean()),
        # mv == k*shift means the block referenced frame t-k (reference index is not exported)
        "frac_mv_eq_k_times_shift": {k: float((np.linalg.norm(mv - [k * dx, k * dy], axis=1) < 0.5).mean())
                                     for k in (1, 2, 3)},
        "per_frame_median_err_range": [min(per_frame), max(per_frame)],
        "n_p_frames": len(per_frame),
    }


def main():
    os.makedirs(WORK, exist_ok=True)
    seq = SintelSeq(SINTEL, "alley_1", "final")
    base = seq.image(0)
    res = {"base_frame": seq.image_path(0), "crop": [CW, CH], "n_frames": NF, "margin": MARGIN, "configs": {}}
    for cname, (codec, kn) in CONFIGS.items():
        rows = {}
        for dx, dy in SHIFTS:
            p = os.path.join(WORK, f"{cname}_{dx}_{dy}.mp4")
            try:
                enc = encode_sequence(make_frames(base, dx, dy), p, codec, **kn)
                frs, _ = extract_mvs(p)
                r = analyse(frs, dx, dy)
                r["options"] = enc["options"]
                if codec == "libx264":
                    r["sei_ref"] = sei_value(enc["x264_sei"], "ref")
                    r["sei_bframes"] = sei_value(enc["x264_sei"], "bframes")
                r["pass"] = r["median_err_px"] < 0.5
            except Exception as e:
                r = {"error": repr(e), "pass": False}
            rows[f"({dx},{dy})"] = r
            print(cname, (dx, dy), {k: r.get(k) for k in
                  ("median_mv", "median_err_px", "median_err_if_sign_flipped", "frac_mv_eq_k_times_shift", "error")})
        res["configs"][cname] = rows
    res["primary"] = PRIMARY
    res["PASS"] = all(r["pass"] for r in res["configs"][PRIMARY].values())
    with open(os.path.join(P3, "results", "val_A.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("VALIDATION A:", "PASS" if res["PASS"] else "FAIL")


if __name__ == "__main__":
    main()
