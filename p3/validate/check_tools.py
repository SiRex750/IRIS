"""Tool checks: flo reader, encoder knobs, B-frame emission, per-MB QP export.

Writes p3/results/check_tools.json. Encodes go to p3/work/knobs/.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, P3)

from lib.flo import SintelSeq, read_flo, read_mask  # noqa: E402
from lib.encode import encode_sequence, sei_value  # noqa: E402
from lib.mvs import extract_mvs  # noqa: E402

SINTEL = r"C:\Users\akash\Documents\datasets\MPI-Sintel"
WORK = os.path.join(P3, "work", "knobs")
N = 24  # frames used for knob checks


def check_flo():
    seq = SintelSeq(SINTEL, "alley_1", "clean")
    f = read_flo(os.path.join(SINTEL, "training", "flow", "alley_1", "frame_0001.flo"))
    occ = read_mask(os.path.join(SINTEL, "training", "occlusions", "alley_1", "frame_0001.png"))
    # Photometric check of index + sign: warp image 2 back with flow 1->2 (nearest neighbour).
    i1 = seq.image(0).astype(np.float32).mean(2)
    i2 = seq.image(1).astype(np.float32).mean(2)
    H, W = i1.shape
    yy, xx = np.mgrid[0:H, 0:W]
    fl = seq.flow_into(1)
    x2 = np.rint(xx + fl[..., 0]).astype(int)
    y2 = np.rint(yy + fl[..., 1]).astype(int)
    ok = (x2 >= 0) & (x2 < W) & (y2 >= 0) & (y2 < H) & ~seq.bad_mask_into(1)
    err_warp = float(np.abs(i2[y2[ok], x2[ok]] - i1[ok]).mean())
    err_zero = float(np.abs(i2[ok] - i1[ok]).mean())
    x2n = np.rint(xx - fl[..., 0]).astype(int)
    y2n = np.rint(yy - fl[..., 1]).astype(int)
    okn = (x2n >= 0) & (x2n < W) & (y2n >= 0) & (y2n < H) & ~seq.bad_mask_into(1)
    err_neg = float(np.abs(i2[y2n[okn], x2n[okn]] - i1[okn]).mean())
    return {
        "shape": list(f.shape), "dtype": str(f.dtype),
        "flow_abs_max": float(np.abs(f).max()), "flow_mag_median": float(np.median(np.linalg.norm(f, axis=2))),
        "occ_frac": float(occ.mean()),
        "photometric_L1_warp_I2(x+flow)_vs_I1": err_warp,
        "photometric_L1_zero_flow": err_zero,
        "photometric_L1_negated_flow": err_neg,
        "pass": err_warp < err_zero and err_warp < err_neg,
    }


def pict_counts(path):
    frames, _ = extract_mvs(path)
    types = [f["pict_type"] for f in frames]
    return {t: types.count(t) for t in sorted(set(types))}, types


def monotone_decreasing(xs):
    return all(b < a for a, b in zip(xs, xs[1:]))


def check_knobs(frames):
    os.makedirs(WORK, exist_ok=True)
    out = {}

    def enc(tag, codec, **kn):
        p = os.path.join(WORK, f"{tag}.mp4")
        r = encode_sequence(frames, p, codec, **kn)
        r["path"] = p
        return r

    # rate knob -> size must fall monotonically as CRF / QP / q rise
    sweeps = {
        "libx264_crf": ("libx264", "crf", [12, 18, 24, 30, 36], dict(preset="medium", bframes=0, keyint=250)),
        "h264_nvenc_qp": ("h264_nvenc", "qp", [12, 18, 24, 30, 36], dict(bframes=0, keyint=250)),
        "mpeg4_q": ("mpeg4", "q", [2, 4, 8, 16, 31], dict(bframes=0, keyint=250)),
    }
    for name, (codec, knob, vals, base) in sweeps.items():
        rows = []
        for v in vals:
            try:
                r = enc(f"{name}_{v}", codec, **{knob: v}, **base)
                row = {knob: v, "file_bytes": r["file_bytes"], "options": r["options"]}
                if codec == "libx264":
                    row["sei_crf"] = sei_value(r["x264_sei"], "crf")
                if codec != "libx264":
                    frs, _ = extract_mvs(r["path"], want_qp=(codec == "h264_nvenc"))
                    if codec == "h264_nvenc":
                        qps = np.concatenate([f["qp"].ravel() for f in frs if f["qp"] is not None])
                        row["decoded_mb_qp_unique"] = sorted(set(int(x) for x in np.unique(qps)))[:10]
                rows.append(row)
            except Exception as e:  # report, don't hide
                rows.append({knob: v, "error": repr(e)})
        sizes = [r.get("file_bytes") for r in rows]
        out[name] = {"rows": rows,
                     "size_monotone_decreasing": None not in sizes and monotone_decreasing(sizes)}

    # B-frame knob: bframes>0 must produce B pictures; bframes=0 must produce none
    bf = {}
    for codec, rk in [("libx264", {"crf": 18, "preset": "medium"}), ("h264_nvenc", {"qp": 24}), ("mpeg4", {"q": 4})]:
        for b in (0, 2):
            try:
                r = enc(f"bf_{codec}_{b}", codec, bframes=b, keyint=250, **rk)
                counts, types = pict_counts(r["path"])
                row = {"options": r["options"], "pict_counts": counts, "types": "".join(types)}
                if codec == "libx264":
                    row["sei_bframes"] = sei_value(r["x264_sei"], "bframes")
                bf[f"{codec}_bf{b}"] = row
            except Exception as e:
                bf[f"{codec}_bf{b}"] = {"error": repr(e)}
    out["bframes"] = bf
    out["bframes_pass"] = all(
        ("error" not in bf[f"{c}_bf0"]) and ("error" not in bf[f"{c}_bf2"]) and
        bf[f"{c}_bf0"]["pict_counts"].get("B", 0) == 0 and bf[f"{c}_bf2"]["pict_counts"].get("B", 0) > 0
        for c in ("libx264", "h264_nvenc", "mpeg4"))

    # keyint knob (x264): keyint=8 over N frames -> I at 0,8,16
    r = enc("x264_keyint8", "libx264", crf=18, bframes=0, keyint=8)
    counts, types = pict_counts(r["path"])
    out["x264_keyint8"] = {"options": r["options"], "types": "".join(types),
                           "sei_keyint": sei_value(r["x264_sei"], "keyint"),
                           "pass": [i for i, t in enumerate(types) if t == "I"] == list(range(0, N, 8))}

    # ref knob (x264)
    r = enc("x264_ref1", "libx264", crf=18, bframes=0, keyint=250, ref=1)
    out["x264_ref1_sei_ref"] = sei_value(r["x264_sei"], "ref")

    # per-MB QP export (venc_params) on an x264 stream
    frs, _ = extract_mvs(os.path.join(WORK, "libx264_crf_24.mp4"), want_qp=True)
    f1 = frs[1]
    out["venc_params"] = {
        "available": f1["qp"] is not None,
        "qp_map_shape": None if f1["qp"] is None else list(f1["qp"].shape),
        "example_frame": 1, "example_pict_type": f1["pict_type"],
        "example_qp_min_median_max": None if f1["qp"] is None else
            [int(f1["qp"].min()), float(np.median(f1["qp"])), int(f1["qp"].max())],
        "example_top_left_4x4": None if f1["qp"] is None else f1["qp"][:4, :4].tolist(),
    }
    return out


def main():
    res = {"flo": check_flo()}
    seq = SintelSeq(SINTEL, "alley_1", "final")
    frames = [seq.image(i) for i in range(N)]
    res["knobs"] = check_knobs(frames)
    os.makedirs(os.path.join(P3, "results"), exist_ok=True)
    with open(os.path.join(P3, "results", "check_tools.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
