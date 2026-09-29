"""Validation D: frame-t GT grid (lib/gt_grid.py) and B-frame reference structure.

a. synthetic, must be exact: (i) constant flow (4,-3); (ii) static background + moving square
b. alley_1 / ambush_5 final, x264 CRF 12 ref 1 bf 0: per P-block |GT_grid - GT_src| and EPE
   under each (GT_src = compare.py, sampled at the src footprint in frame t-1).
   STOP rule (alley_1 only): median |diff| > 0.01 px or p90 > 0.1 px.
c. B-frame check: x264 bframes 2 b-pyramid none ref 1; NVENC bf 2 b_ref_mode disabled refs 1.
   Every B slice must have nal_ref_idc == 0 (non-reference), and x264 SEI b_pyramid=0.
Writes p3/results/val_D.json.
"""
import json
import os
import sys

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, P3)

from lib.flo import SintelSeq  # noqa: E402
from lib.encode import encode_sequence, sei_value  # noqa: E402
from lib.mvs import extract_mvs, dst_topleft  # noqa: E402
from lib.compare import compare_frame  # noqa: E402
from lib.gt_grid import GTGrid  # noqa: E402

SINTEL = r"C:\Users\akash\Documents\datasets\MPI-Sintel"
WORK = os.path.join(P3, "work", "val_D")
H, W = 436, 1024


# ---------------------------------------------------------------- a. synthetic
def synth_constant():
    flow = np.zeros((H, W, 2), np.float32)
    flow[..., 0], flow[..., 1] = 4, -3
    g = GTGrid(flow, np.zeros((H, W), bool))
    gx, gy, cov, _ = g.cell_gt()
    cnt = g.count
    col0 = np.all(cnt[:, 0] == 0)
    bottom = np.all(cnt[-1, 1:] == 4)
    rest = cnt[:-1, 1:]
    rest_ok = np.all(rest == 16) and np.all(gx[:-1, 1:] == 4) and np.all(gy[:-1, 1:] == -3)
    bottom_gt = np.all(gx[-1, 1:] == 4) and np.all(gy[-1, 1:] == -3)
    return {"col0_count0": bool(col0), "bottom_row_count4": bool(bottom),
            "other_cells_count16_gt(4,-3)": bool(rest_ok), "bottom_row_gt(4,-3)": bool(bottom_gt),
            "count_unique": sorted(set(np.unique(cnt).astype(int).tolist())),
            "pass": bool(col0 and bottom and rest_ok and bottom_gt)}


def synth_square():
    x0, y0, S, dx = 400, 160, 100, 8
    flow = np.zeros((H, W, 2), np.float32)
    flow[y0:y0 + S, x0:x0 + S, 0] = dx
    bad = np.zeros((H, W), bool)
    bad[y0:y0 + S, x0 + S:x0 + S + dx] = True  # background covered by the square in frame t
    g = GTGrid(flow, bad)
    gx, gy, cov, std = g.cell_gt()
    cnt = g.count
    c = 4
    sq = (slice(y0 // c, (y0 + S) // c), slice((x0 + dx) // c, (x0 + S + dx) // c))
    strip = (slice(y0 // c, (y0 + S) // c), slice(x0 // c, (x0 + dx) // c))
    bg = np.ones_like(cnt, bool)
    bg[sq] = False
    bg[strip] = False
    ok_sq = np.all(cnt[sq] == 16) and np.all(gx[sq] == dx) and np.all(gy[sq] == 0)
    ok_strip = np.all(cnt[strip] == 0)
    ok_bg = np.all(cnt[bg] == 16) and np.all(gx[bg] == 0) and np.all(gy[bg] == 0)
    return {"square_cells_count16_gt(8,0)": bool(ok_sq), "left_strip_count0": bool(ok_strip),
            "background_cells_count16_gt(0,0)": bool(ok_bg),
            "n_cells": {"square": int(cnt[sq].size), "strip": int(cnt[strip].size), "background": int(bg.sum())},
            "pass": bool(ok_sq and ok_strip and ok_bg)}


# ---------------------------------------------------------------- b. real GT
def real(seqname):
    seq = SintelSeq(SINTEL, seqname, "final")
    p = os.path.join(WORK, f"{seqname}_x264_crf12_ref1_bf0.mp4")
    enc = encode_sequence([seq.image(i) for i in range(seq.n_frames)], p, "libx264",
                          crf=12, preset="medium", bframes=0, keyint=250, ref=1)
    frames, _ = extract_mvs(p)
    diffs, epe_src_c, epe_grid_c, epe_src_all, epe_grid_all = [], [], [], [], []
    area_blocks = area_grid_excl = area_src_excl = 0.0
    for fr in frames:
        t = fr["index"]
        if fr["pict_type"] != "P" or fr["mvs"].size == 0:
            continue
        flow, bad = seq.flow_into(t), seq.bad_mask_into(t)
        r = compare_frame(fr["mvs"], flow, bad)
        m = fr["mvs"][fr["mvs"]["source"] == -1]
        tl = dst_topleft(m)
        gx, gy, cov, std, _ = GTGrid(flow, bad).block_gt(tl[:, 0], tl[:, 1], m["w"], m["h"])
        mv = np.stack([r["mv_x"], r["mv_y"]], 1)
        grid = np.stack([gx, gy], 1)
        src = np.stack([r["gt_x"], r["gt_y"]], 1)
        k_grid = cov >= 0.5
        k_src = r["kept"]
        both = k_grid & k_src
        a = (m["w"] * m["h"]).astype(float)
        area_blocks += a.sum()
        area_grid_excl += a[~k_grid].sum()
        area_src_excl += a[~k_src].sum()
        diffs.append(np.linalg.norm(grid[both] - src[both], axis=1))
        epe_src_c.append(np.linalg.norm(mv[both] - src[both], axis=1))
        epe_grid_c.append(np.linalg.norm(mv[both] - grid[both], axis=1))
        epe_src_all.append(np.linalg.norm(mv[k_src] - src[k_src], axis=1))
        epe_grid_all.append(np.linalg.norm(mv[k_grid] - grid[k_grid], axis=1))
    d = np.concatenate(diffs)
    cat = lambda xs: np.concatenate(xs)  # noqa: E731
    return {
        "options": enc["options"], "n_blocks_both_kept": int(d.size),
        "abs_diff_median": float(np.median(d)), "abs_diff_p90": float(np.percentile(d, 90)),
        "abs_diff_p99": float(np.percentile(d, 99)), "abs_diff_mean": float(d.mean()),
        "epe_median_src_common": float(np.median(cat(epe_src_c))),
        "epe_median_grid_common": float(np.median(cat(epe_grid_c))),
        "epe_median_src_own_kept": float(np.median(cat(epe_src_all))),
        "epe_median_grid_own_kept": float(np.median(cat(epe_grid_all))),
        "excluded_area_share_grid_cover_lt_0.5": area_grid_excl / area_blocks,
        "excluded_area_share_src_rule": area_src_excl / area_blocks,
    }


# ---------------------------------------------------------------- c. B-frame refs
def _rbsp(b):
    out, zeros = bytearray(), 0
    for x in b:
        if zeros >= 2 and x == 3:
            zeros = 0
            continue
        out.append(x)
        zeros = zeros + 1 if x == 0 else 0
    return bytes(out)


def _ue(bits, pos):
    z = 0
    while bits[pos] == 0:
        z += 1
        pos += 1
    pos += 1
    v = 0
    for _ in range(z):
        v = (v << 1) | int(bits[pos])
        pos += 1
    return (1 << z) - 1 + v, pos


def _nals(data, nal_len):
    if data[:4] == b"\x00\x00\x00\x01" or data[:3] == b"\x00\x00\x01":
        parts = data.replace(b"\x00\x00\x00\x01", b"\x00\x00\x01").split(b"\x00\x00\x01")
        return [p for p in parts if p]
    out, i = [], 0
    while i + nal_len <= len(data):
        n = int.from_bytes(data[i:i + nal_len], "big")
        out.append(data[i + nal_len:i + nal_len + n])
        i += nal_len + n
    return out


def slice_ref_idc(path):
    """Per slice NAL: (packet index, slice type letter, nal_ref_idc)."""
    rows = []
    with av.open(path) as c:
        st = c.streams.video[0]
        ex = bytes(st.codec_context.extradata or b"")
        nal_len = (ex[4] & 3) + 1 if ex[:1] == b"\x01" else 4
        for pi, pkt in enumerate(c.demux(st)):
            if pkt.size == 0:
                continue
            for nal in _nals(bytes(pkt), nal_len):
                typ, ref_idc = nal[0] & 0x1F, (nal[0] >> 5) & 3
                if typ not in (1, 5):
                    continue
                rb = _rbsp(nal[1:24])
                bits = np.unpackbits(np.frombuffer(rb, np.uint8))
                _, pos = _ue(bits, 0)          # first_mb_in_slice
                stype, _ = _ue(bits, pos)      # slice_type
                rows.append((pi, "PBI"[stype % 5] if stype % 5 < 3 else f"S{stype}", ref_idc))
    return rows


def bframe_check():
    seq = SintelSeq(SINTEL, "alley_1", "final")
    imgs = [seq.image(i) for i in range(24)]
    out = {}
    for tag, codec, kn in [("x264_bf2_bpyr_none_ref1", "libx264",
                            dict(crf=23, preset="medium", bframes=2, keyint=250, ref=1, b_pyramid="none")),
                           ("nvenc_bf2_bref_disabled_refs1", "h264_nvenc",
                            dict(qp=28, bframes=2, keyint=250, ref=1, b_ref_mode="disabled"))]:
        p = os.path.join(WORK, f"bcheck_{tag}.mp4")
        enc = encode_sequence(imgs, p, codec, **kn)
        rows = slice_ref_idc(p)
        frames, _ = extract_mvs(p)
        b = [r for r in rows if r[1] == "B"]
        nonb = [r for r in rows if r[1] != "B"]
        res = {"options": enc["options"], "display_types": "".join(f["pict_type"] for f in frames),
               "n_slices": len(rows), "n_B_slices": len(b),
               "B_slice_ref_idc_values": sorted({r[2] for r in b}),
               "nonB_slice_ref_idc_values": sorted({r[2] for r in nonb}),
               "all_B_nonref": len(b) > 0 and all(r[2] == 0 for r in b)}
        if codec == "libx264":
            res["sei_b_pyramid"] = sei_value(enc["x264_sei"], "b_pyramid")
            res["sei_bframes"] = sei_value(enc["x264_sei"], "bframes")
            res["sei_ref"] = sei_value(enc["x264_sei"], "ref")
            res["pass"] = res["all_B_nonref"] and res["sei_b_pyramid"] == "0"
        else:
            res["pass"] = res["all_B_nonref"]
        out[tag] = res
    return out


def main():
    os.makedirs(WORK, exist_ok=True)
    res = {"a_constant": synth_constant(), "a_square": synth_square()}
    print("a:", res["a_constant"]["pass"], res["a_square"]["pass"])
    res["b_alley_1"] = real("alley_1")
    res["b_ambush_5"] = real("ambush_5")
    b = res["b_alley_1"]
    res["b_stop"] = b["abs_diff_median"] > 0.01 or b["abs_diff_p90"] > 0.1
    print("b alley_1:", {k: v for k, v in b.items() if k != "options"})
    print("b ambush_5:", {k: v for k, v in res["b_ambush_5"].items() if k != "options"})
    res["c"] = bframe_check()
    for k, v in res["c"].items():
        print("c", k, {kk: vv for kk, vv in v.items() if kk != "options"})
    res["PASS"] = (res["a_constant"]["pass"] and res["a_square"]["pass"] and not res["b_stop"]
                   and all(v["pass"] for v in res["c"].values()))
    with open(os.path.join(P3, "results", "val_D.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("VALIDATION D:", "PASS" if res["PASS"] else "FAIL")


if __name__ == "__main__":
    main()
