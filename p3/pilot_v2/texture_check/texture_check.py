"""Texture check on ambush_7 (report only; not committed).

Cells: 4x4 px on frame t's grid (256 x 109), P-frames t = 1..49, d = 1, past vectors.
Texture of a cell = mean luma-gradient magnitude of the SOURCE frame t over the cell
(Y from the same swscale RGB->yuv420p path the encoder input used; np.gradient, central diffs).
Quartile edges: from the final-pass source frames over valid cells (the frames the CRF 12
encode was made from), fixed for every encode, including the clean pass.
Gate as in pilot v2: reuse iff cell has a vector and |MV| < tau (tau = 1); GT-static |GT| < 0.5;
stale = reused and |GT| > 2; valid = gt_cover >= 0.5. GT from lib/gt_grid (pass-independent).
Final-pass encodes are the v2 bitstreams in ../encodes; clean-pass encodes are made here with the
identical v2 knobs.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
P3 = os.path.dirname(V2)
sys.path.insert(0, P3)
sys.path.insert(0, os.path.join(P3, "pilot"))
sys.path.insert(0, V2)

from lib.flo import SintelSeq  # noqa: E402
from lib.encode import encode_sequence, sei_value  # noqa: E402
from lib.mvs import extract_mvs, motion_prev_to_cur, dst_topleft  # noqa: E402
from run_pilot import load_seq, y_plane, paint  # noqa: E402
from run_pilot_v2 import ENCODES, slices_per_frame  # noqa: E402

SINTEL = r"C:\Users\akash\Documents\datasets\MPI-Sintel"
SEQ, TAU, C = "ambush_7", 1.0, 4
CRFS = (12, 23, 45)
KNOBS = {e["id"]: e["knobs"] for e in ENCODES}


def cell_texture(y):
    gy, gx = np.gradient(y)
    g = np.hypot(gx, gy)
    H, W = g.shape
    return g.reshape(H // C, C, W // C, C).mean(axis=(1, 3))


def cells(path, sq):
    """Per P-frame cell arrays: mv magnitude (nan = no vector), valid, |GT|."""
    frames, _ = extract_mvs(path)
    Hc, Wc = sq["Hc"], sq["Wc"]
    out = {}
    for fr in frames:
        t = fr["index"]
        if fr["pict_type"] != "P" or t == 0:
            continue
        m = fr["mvs"][fr["mvs"]["source"] == -1]
        tl = dst_topleft(m)
        x0, y0 = np.rint(tl[:, 0]).astype(np.int64), np.rint(tl[:, 1]).astype(np.int64)
        ci, bi = paint(Hc, Wc, x0, y0, m["w"].astype(np.int64), m["h"].astype(np.int64))
        mv = motion_prev_to_cur(m)
        mag = np.full(Hc * Wc, np.nan)
        mag[ci] = np.hypot(mv[bi, 0], mv[bi, 1])
        gt = sq["gt"][t]
        out[t] = {"mag": mag, "valid": gt["ccov"].ravel() >= 0.5,
                  "gmag": np.hypot(gt["cgx"].ravel(), gt["cgy"].ravel())}
    return out, "".join(f["pict_type"] for f in frames)


def table(cellmap, tex, edges):
    rows = []
    q_all, mag, valid, gmag = [], [], [], []
    for t, c in cellmap.items():
        q_all.append(np.digitize(tex[t].ravel(), edges[1:-1]))
        mag.append(c["mag"])
        valid.append(c["valid"])
        gmag.append(c["gmag"])
    q, mag, valid, gmag = map(np.concatenate, (q_all, mag, valid, gmag))
    has = ~np.isnan(mag)
    reuse = has & (np.nan_to_num(mag, nan=np.inf) < TAU)
    static = valid & (gmag < 0.5)
    for k in range(4):
        v = valid & (q == k)
        st = static & (q == k)
        nv = int(v.sum())
        rows.append({
            "quartile": f"Q{k + 1}", "valid_cells": nv, "gt_static_cells": int(st.sum()),
            "gt_static_share_of_valid": st.sum() / nv if nv else np.nan,
            "missed_of_valid": (st & ~reuse).sum() / nv if nv else np.nan,
            "stale_of_valid": (v & reuse & (gmag > 2)).sum() / nv if nv else np.nan,
            "static_with_mv_ge1": (st & has & (np.nan_to_num(mag) >= 1)).sum() / st.sum() if st.sum() else np.nan,
            "static_no_vector": (st & ~has).sum() / st.sum() if st.sum() else np.nan,
            "static_mv_0_to_1": (st & reuse).sum() / st.sum() if st.sum() else np.nan,
        })
    tot = {"quartile": "all", "valid_cells": int(valid.sum()), "gt_static_cells": int(static.sum()),
           "gt_static_share_of_valid": static.sum() / valid.sum(),
           "missed_of_valid": (static & ~reuse).sum() / valid.sum(),
           "stale_of_valid": (valid & reuse & (gmag > 2)).sum() / valid.sum(),
           "static_with_mv_ge1": (static & has & (np.nan_to_num(mag) >= 1)).sum() / static.sum(),
           "static_no_vector": (static & ~has).sum() / static.sum(),
           "static_mv_0_to_1": (static & reuse).sum() / static.sum()}
    return rows + [tot]


def main():
    sq = load_seq(SEQ)  # final pass images + GT
    clean = SintelSeq(SINTEL, SEQ, "clean")
    clean_imgs = [clean.image(i) for i in range(clean.n_frames)]
    H = sq["H"]
    tex_final = {t: cell_texture(sq["yref"][t]) for t in range(1, len(sq["imgs"]))}
    tex_clean = {t: cell_texture(y_plane(clean_imgs[t], H)) for t in range(1, len(clean_imgs))}

    # quartile edges: final-pass source, valid cells, P-frames 1..49
    vals = np.concatenate([tex_final[t].ravel()[sq["gt"][t]["ccov"].ravel() >= 0.5] for t in tex_final])
    qs = np.percentile(vals, [25, 50, 75])
    edges = np.array([-np.inf, *qs, np.inf])
    vals_c = np.concatenate([tex_clean[t].ravel()[sq["gt"][t]["ccov"].ravel() >= 0.5] for t in tex_clean])
    qs_c = np.percentile(vals_c, [25, 50, 75])
    edges_c = np.array([-np.inf, *qs_c, np.inf])

    os.makedirs(os.path.join(HERE, "encodes"), exist_ok=True)
    rows, meta = [], {"quartile_edges_final": qs.tolist(), "quartile_edges_clean_own": qs_c.tolist(), "encodes": {}}
    for crf in CRFS:
        eid = f"x264_crf{crf}"
        # final pass: v2 bitstream
        pf = os.path.join(V2, "encodes", f"{eid}__{SEQ}.mp4")
        cm, types = cells(pf, sq)
        for r in table(cm, tex_final, edges):
            rows.append({"pass": "final", "crf": crf, "edges": "final", **r})
        # clean pass: new encode, identical knobs
        pc = os.path.join(HERE, "encodes", f"{eid}__{SEQ}_clean.mp4")
        rec = encode_sequence(clean_imgs, pc, "libx264", fps=24, **KNOBS[eid])
        cmc, types_c = cells(pc, sq)
        for r in table(cmc, tex_clean, edges):
            rows.append({"pass": "clean", "crf": crf, "edges": "final (fixed)", **r})
        for r in table(cmc, tex_clean, edges_c):
            rows.append({"pass": "clean", "crf": crf, "edges": "clean (own)", **r})
        spf = slices_per_frame(pc, "libx264")
        meta["encodes"][eid] = {"final_types": types, "clean_types": types_c, "clean_options": rec["options"],
                                "clean_kbps": rec["packet_bytes"] * 8 * 24 / len(clean_imgs) / 1000,
                                "clean_sei_threads": sei_value(rec["x264_sei"], "threads"),
                                "clean_sei_sliced": sei_value(rec["x264_sei"], "sliced_threads"),
                                "clean_slices_per_frame": [min(spf), max(spf)]}
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "texture_check.csv"), index=False)
    json.dump(meta, open(os.path.join(HERE, "texture_meta.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(json.dumps(meta, indent=1))
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
