"""CodecSight policy simulation: data inventory ONLY (written with the pre-registration, before any metric).

Lists, for every encode the simulation will use: file presence, number of frames, frame-type counts, I-frame
positions and the refresh positions they imply under PREREGISTRATION.md (refresh at every I-frame, and 16 frames
after the previous refresh, whichever comes first). Also lists GT availability (Sintel .flo files, CCTV RAFT frames).

It reads only frame types (Sintel __frames.parquet `ftype`; CCTV npz `ftype`, `meta`), file names and the RAFT
frame index `t`. It never loads motion vectors, ground-truth flow or RAFT flow values, and computes no reuse or
stale quantity. Writes inventory.txt and inventory.csv next to this file.
"""
import json
import math
from fractions import Fraction
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
SINTEL = r"C:\Users\akash\Documents\datasets\MPI-Sintel"
WINDOW = 16

SINTEL_ARMS = ([f"x264_crf{c}" for c in (12, 18, 23, 28, 33, 38, 45)]
               + [f"nvenc_qp{q}" for q in (18, 23, 28, 33, 38, 45)]
               + ["x264_qp24", "x264_qp24_bf2_pb1", "nvenc_qp28_bf2_bqeq", "mpeg4_q4"])
CCTV_ARMS = ("x264_crf12", "x264_crf45", "x264_qp24", "x264_qp24_bf2_pb1", "nvenc_qp18", "nvenc_qp23",
             "nvenc_qp45", "nvenc_qp28", "nvenc_qp28_bf2_bqeq")


def refreshes(iframes, n, window=WINDOW):
    """Refresh positions: every I-frame, and `window` frames after the previous refresh, whichever comes first."""
    out, last, iset = [], None, set(int(i) for i in iframes)
    for t in range(n):
        if t in iset or last is None or t - last >= window:
            out.append(t)
            last = t
    return out


def runs(xs):
    """Compact text for a sorted int list: arithmetic runs as a..b/step."""
    if not xs:
        return "-"
    parts, i = [], 0
    while i < len(xs):
        j = i
        step = xs[i + 1] - xs[i] if i + 1 < len(xs) else 0
        while j + 1 < len(xs) and xs[j + 1] - xs[j] == step:
            j += 1
        parts.append(str(xs[i]) if j == i else (f"{xs[i]},{xs[j]}" if j == i + 1 else f"{xs[i]}..{xs[j]}/{step}"))
        i = j + 1
    return ",".join(parts)


def sintel_rows():
    bdir = os.path.join(P3, "sweep", "blocks", "final")
    seqs = sorted({f.split("__")[1] for f in os.listdir(bdir) if f.endswith("__frames.parquet")})
    rows = []
    for seq in seqs:
        flo_dir = os.path.join(SINTEL, "training", "flow", seq)
        img_dir = os.path.join(SINTEL, "training", "final", seq)
        n_flo = len([f for f in os.listdir(flo_dir) if f.endswith(".flo")]) if os.path.isdir(flo_dir) else -1
        n_img = len([f for f in os.listdir(img_dir) if f.endswith(".png")]) if os.path.isdir(img_dir) else -1
        for arm in SINTEL_ARMS:
            fp = os.path.join(bdir, f"{arm}__{seq}__frames.parquet")
            bp = os.path.join(bdir, f"{arm}__{seq}.parquet")
            r = {"data": "Sintel final", "unit": seq, "arm": arm, "frames_file": os.path.isfile(fp),
                 "blocks_file": os.path.isfile(bp), "source_frames": n_img, "gt_frames": n_flo, "fps": "24"}
            if r["frames_file"]:
                ft = pd.read_parquet(fp, columns=["frame", "ftype"]).sort_values("frame")
                r.update(_types(ft["ftype"].tolist(), ft["frame"].tolist()))
            rows.append(r)
    return rows


def cctv_rows():
    el = pd.read_csv(os.path.join(P3, "virat_confirm", "eligibility_v0.csv"))
    el = el[el.eligible.astype(bool)].sort_values("clip")
    rows = []
    for clip, scene in zip(el["clip"], el["scene"]):
        d = os.path.join(P3, "virat_confirm", "data", clip)
        rp = os.path.join(d, "raft.npz")
        n_raft = int(np.load(rp)["t"].shape[0]) if os.path.isfile(rp) else -1
        for arm in CCTV_ARMS:
            p = os.path.join(d, f"{arm}.npz")
            r = {"data": "CCTV confirmatory", "unit": clip, "scene": scene, "arm": arm, "frames_file": os.path.isfile(p),
                 "blocks_file": os.path.isfile(p), "gt_frames": n_raft}
            if r["frames_file"]:
                z = np.load(p)
                meta = json.loads(str(z["meta"]))
                r["fps"] = meta.get("fps", "?")
                r["source_frames"] = int(z["ftype"].shape[0])
                r.update(_types([str(x) for x in z["ftype"]], list(range(z["ftype"].shape[0]))))
                fps = float(Fraction(r["fps"])) if r["fps"] != "?" else math.nan  # "30" or "24000/1001"
                k = int(round(fps / 2)) if fps == fps else -1
                n = r["n_frames"]
                samp = list(range(0, n, k)) if k > 0 else []
                # 2 FPS exploratory arm: a sampled frame refreshes if it is the first sampled frame at/after an
                # I-frame, or 16 sampled frames after the previous refresh
                ifr = r["_iframes"]
                firsts = {next((s for s in samp if s >= i), None) for i in ifr} - {None}
                sref, last = [], None
                for j, s in enumerate(samp):
                    if s in firsts or last is None or j - last >= WINDOW:
                        sref.append(s)
                        last = j
                r.update({"step_2fps": k, "n_sampled_2fps": len(samp), "refresh_2fps": runs(sref),
                          "n_refresh_2fps": len(sref)})
            rows.append(r)
    return rows


def _types(ftypes, frames):
    n = len(ftypes)
    assert frames == list(range(n)), "frame indices not 0..n-1"
    ifr = [i for i, t in enumerate(ftypes) if t == "I"]
    ref = refreshes(ifr, n)
    return {"n_frames": n, "n_I": ftypes.count("I"), "n_P": ftypes.count("P"), "n_B": ftypes.count("B"),
            "iframes": runs(ifr), "_iframes": ifr, "refresh": runs(ref), "n_refresh": len(ref),
            "n_windows_lt16": sum(1 for a, b in zip(ref, ref[1:] + [n]) if b - a < WINDOW)}


def main():
    rows = sintel_rows() + cctv_rows()
    df = pd.DataFrame(rows).drop(columns=["_iframes"])
    df.to_csv(os.path.join(HERE, "inventory.csv"), index=False)

    L = ["CodecSight policy simulation: data inventory (no metric computed)", ""]
    for data, g in df.groupby("data", sort=False):
        miss = g[~(g.frames_file & g.blocks_file)]
        L.append(f"== {data}: {g.unit.nunique()} units x {g.arm.nunique()} arms = {len(g)} encodes; "
                 f"missing files: {len(miss)}")
        for _, m in miss.iterrows():
            L.append(f"   MISSING {m.unit} {m.arm}")
        L.append(f"   fps: {dict(g.fps.value_counts())}")
        L.append(f"   frames per encode: min {int(g.n_frames.min())}, median {g.n_frames.median():g}, "
                 f"max {int(g.n_frames.max())}")
        bad = g[g.n_frames != g.source_frames]
        L.append(f"   encodes whose frame count != source frame count: {len(bad)}")
        gtbad = g[g.gt_frames != g.n_frames - 1]
        L.append(f"   encodes whose GT frame count != n_frames - 1: {len(gtbad)}"
                 + (f"  {sorted(set(zip(gtbad.unit, gtbad.gt_frames, gtbad.n_frames)))[:5]}" if len(gtbad) else ""))
        L.append("   distinct I-frame position patterns (count of encodes):")
        for (ip, rf), c in g.groupby(["iframes", "refresh"]).size().items():
            L.append(f"     I at [{ip}]  -> refresh at [{rf}]   ({c})")
        L.append("   frame types per arm (sum over units): ")
        for arm, a in g.groupby("arm", sort=False):
            L.append(f"     {arm:<22} I {int(a.n_I.sum()):>4}  P {int(a.n_P.sum()):>6}  B {int(a.n_B.sum()):>6}  "
                     f"refreshes {int(a.n_refresh.sum()):>5}  short windows (<16) {int(a.n_windows_lt16.sum())}")
        if "step_2fps" in g and g.step_2fps.notna().any():
            L.append("   2 FPS exploratory arm: sampling step by fps: "
                     f"{dict(g.groupby('fps').step_2fps.first())}; sampled frames per encode "
                     f"{dict(g.n_sampled_2fps.value_counts())}; refresh patterns:")
            for rf, c in g.groupby("refresh_2fps").size().items():
                L.append(f"     [{rf}]  ({c})")
        L.append("")
    L.append("Per-unit rows (one per sequence/clip; arms with an identical pattern collapsed):")
    for (data, unit), g in df.groupby(["data", "unit"], sort=False):
        pats = g.groupby(["n_frames", "iframes"]).arm.apply(lambda a: ",".join(a)).reset_index()
        for _, p in pats.iterrows():
            L.append(f"  {data[:6]:<6} {unit:<34} n={int(p.n_frames):>3} I=[{p.iframes}]  arms: "
                     f"{'all' if len(p.arm.split(',')) == g.arm.nunique() else p.arm}")
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, "inventory.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
