"""Validation B (real GT sanity on alley_1) and C (speed).

B: alley_1, libx264 CRF 12, bframes 0. Primary uses ref=1 so every P-block's reference is
   t-1 (validation A shows default ref=3 sends ~65% of blocks to t-2 with no way to tell from
   the exported MVs). Default ref is reported alongside. Final pass is primary; clean reported.
   "Textured" = block's mean luma gradient magnitude (frame d-1, src footprint) in the top
   quartile of kept blocks; "moving" = |mean GT flow| > 1 px.
C: frames/sec for encode (frames preloaded, PNG decode timed separately) and MV extraction.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, P3)

from lib.flo import SintelSeq  # noqa: E402
from lib.encode import encode_sequence  # noqa: E402
from lib.mvs import extract_mvs, src_topleft  # noqa: E402
from lib.compare import compare_sequence, summarize, _sat, _box_sum  # noqa: E402

SINTEL = r"C:\Users\akash\Documents\datasets\MPI-Sintel"
WORK = os.path.join(P3, "work", "val_BC")
SEQ = "alley_1"


def block_texture(frames, seq, res):
    """Mean gradient magnitude of frame d-1 luma over each compared block's src footprint."""
    tex = np.full(res["epe"].shape[0], np.nan)
    for fr in frames:
        d = fr["index"]
        rows = np.nonzero(res["frame"] == d)[0]
        if rows.size == 0:
            continue
        y = seq.image(d - 1).astype(np.float64) @ [0.299, 0.587, 0.114]
        gy, gx = np.gradient(y)
        sat = _sat(np.hypot(gx, gy))
        m = fr["mvs"][fr["mvs"]["source"] == -1]
        tl = np.rint(src_topleft(m)).astype(np.int64)
        ok = res["inside"][rows]
        r, t = rows[ok], tl[ok]
        w, h = res["w"][r], res["h"][r]
        tex[r] = _box_sum(sat, t[:, 0], t[:, 1], w, h) / (w * h)
    return tex


def run(pass_, ref, tag):
    seq = SintelSeq(SINTEL, SEQ, pass_)
    t0 = time.perf_counter()
    imgs = [seq.image(i) for i in range(seq.n_frames)]
    t_load = time.perf_counter() - t0
    kn = dict(crf=12, preset="medium", bframes=0, keyint=250)
    if ref is not None:
        kn["ref"] = ref
    p = os.path.join(WORK, f"{tag}.mp4")
    enc = encode_sequence(imgs, p, "libx264", **kn)
    frames, t_mv = extract_mvs(p)
    _, t_mvqp = extract_mvs(p, want_qp=True)
    t0 = time.perf_counter()
    res = compare_sequence(frames, seq)
    t_cmp = time.perf_counter() - t0

    tex = block_texture(frames, seq, res)
    k = res["kept"]
    thr = float(np.nanpercentile(tex[k], 75))
    textured = tex >= thr
    moving = res["gt_mag"] > 1.0

    out = {
        "pass": pass_, "ref": ref, "options": enc["options"], "file_bytes": enc["file_bytes"],
        "n_frames": len(frames), "pict_types": "".join(f["pict_type"] for f in frames),
        "all": summarize(res),
        "moving_gt>1px": summarize(res, textured=moving),
        "textured_top25%": summarize(res, textured=textured),
        "textured_and_moving": summarize(res, textured=textured & moving),
        "texture_threshold_mean_grad": thr,
        "excluded": {
            "outside_image": int((~res["inside"]).sum()),
            "valid_frac_lt_0.5": int((res["inside"] & ~k).sum()),
        },
        "speed": {
            "png_load_fps": len(imgs) / t_load,
            "encode_fps": len(imgs) / enc["seconds"],
            "mv_extract_fps": len(frames) / t_mv,
            "mv_plus_qp_extract_fps": len(frames) / t_mvqp,
            "compare_fps": len(frames) / t_cmp,
        },
    }
    np.savez_compressed(os.path.join(WORK, f"{tag}_blocks.npz"), **res, texture=tex)
    return out


def speed_other_encoders():
    seq = SintelSeq(SINTEL, SEQ, "final")
    imgs = [seq.image(i) for i in range(seq.n_frames)]
    out = {}
    for tag, codec, kn in [("h264_nvenc_qp15_bf0", "h264_nvenc", dict(qp=15, bframes=0, keyint=250, ref=1)),
                           ("mpeg4_q2_bf0", "mpeg4", dict(q=2, bframes=0, keyint=250)),
                           ("libx264_crf12_bf2", "libx264", dict(crf=12, preset="medium", bframes=2, keyint=250))]:
        p = os.path.join(WORK, f"speed_{tag}.mp4")
        enc = encode_sequence(imgs, p, codec, **kn)
        frames, t_mv = extract_mvs(p)
        out[tag] = {"encode_fps": len(imgs) / enc["seconds"], "mv_extract_fps": len(frames) / t_mv,
                    "file_bytes": enc["file_bytes"]}
    return out


def main():
    os.makedirs(WORK, exist_ok=True)
    res = {"sequence": SEQ}
    for tag, pass_, ref in [("final_ref1", "final", 1), ("final_refdefault", "final", None),
                            ("clean_ref1", "clean", 1)]:
        res[tag] = run(pass_, ref, f"{SEQ}_{tag}")
        a = res[tag]
        print(tag, "median EPE all=%.3f moving=%.3f tex&mov=%.3f zeroMV=%.3f kept=%d" % (
            a["all"]["epe_median"], a["moving_gt>1px"]["epe_median"], a["textured_and_moving"]["epe_median"],
            a["all"]["zero_mv_frac"], a["all"]["blocks_kept"]), a["speed"])
    res["speed_other_encoders"] = speed_other_encoders()
    print(res["speed_other_encoders"])
    with open(os.path.join(P3, "results", "val_BC.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
