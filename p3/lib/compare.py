"""Per-block comparison of codec motion vectors against Sintel ground-truth flow.

For P-frame d and each past-referencing block (source == -1), the reference is ASSUMED to be
d-1 (only guaranteed with one reference frame and no B-frames; see mvs.py). The block's
codec motion d-1 -> d is compared with the mean GT flow file (d-1 -> d) over the block's
footprint in frame d-1, i.e. at the src position.

Exclusions:
  - blocks whose dst footprint (frame d) or src footprint (frame d-1) is not entirely inside
    the image (Sintel is 1024x436; H.264 pads to 448 rows, so the last MB row is partial)
  - pixels that are occluded (occlusions/frame_{d:04d}) or invalid (invalid/frame_{d:04d});
    a block is kept only if at least `min_valid` of its footprint pixels survive

Gate decisions, per tau in TAUS (px):
  mv_moving[tau]  |codec motion|   > tau
  gt_moving[tau]  |mean GT flow|   > tau
  epe_ok[tau]     EPE             <= tau
"""
import numpy as np

from .mvs import motion_prev_to_cur, src_topleft, dst_topleft

TAUS = (0.5, 1.0, 2.0)


def _sat(a):
    """Summed-area table with a zero row/col prepended; a: (H, W) or (H, W, C)."""
    s = np.cumsum(np.cumsum(a, axis=0, dtype=np.float64), axis=1)
    pad = [(1, 0), (1, 0)] + [(0, 0)] * (a.ndim - 2)
    return np.pad(s, pad)


def _box_sum(sat, x0, y0, w, h):
    x1, y1 = x0 + w, y0 + h
    return sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]


def compare_frame(mvs, gt_flow, bad_mask, min_valid=0.5, taus=TAUS):
    """Compare one P-frame's MVs (structured array) with GT flow (H,W,2) and bad mask (H,W).

    Returns dict of 1-D arrays (one entry per source==-1 block), including a boolean `kept`.
    """
    H, W = gt_flow.shape[:2]
    sel = mvs["source"] == -1
    m = mvs[sel]
    n = m.shape[0]
    mv = motion_prev_to_cur(m)
    w = m["w"].astype(np.int64)
    h = m["h"].astype(np.int64)
    s0 = np.rint(src_topleft(m)).astype(np.int64)
    d0 = np.rint(dst_topleft(m)).astype(np.int64)

    inside = ((s0[:, 0] >= 0) & (s0[:, 1] >= 0) & (s0[:, 0] + w <= W) & (s0[:, 1] + h <= H) &
              (d0[:, 0] >= 0) & (d0[:, 1] >= 0) & (d0[:, 0] + w <= W) & (d0[:, 1] + h <= H))

    good = (~bad_mask).astype(np.float64)
    sat_n = _sat(good)
    sat_f = _sat(gt_flow.astype(np.float64) * good[..., None])

    gt = np.full((n, 2), np.nan)
    valid_frac = np.zeros(n)
    idx = np.nonzero(inside)[0]
    if idx.size:
        x0, y0, ww, hh = s0[idx, 0], s0[idx, 1], w[idx], h[idx]
        cnt = _box_sum(sat_n, x0, y0, ww, hh)
        fs = _box_sum(sat_f, x0, y0, ww, hh)
        valid_frac[idx] = cnt / (ww * hh)
        with np.errstate(invalid="ignore", divide="ignore"):
            gt[idx] = fs / cnt[:, None]

    kept = inside & (valid_frac >= min_valid)
    epe = np.linalg.norm(mv - gt, axis=1)
    epe[~kept] = np.nan
    mv_mag = np.linalg.norm(mv, axis=1)
    gt_mag = np.linalg.norm(gt, axis=1)

    out = {
        "w": w, "h": h, "dst_x": m["dst_x"], "dst_y": m["dst_y"],
        "mv_x": mv[:, 0], "mv_y": mv[:, 1], "gt_x": gt[:, 0], "gt_y": gt[:, 1],
        "inside": inside, "valid_frac": valid_frac, "kept": kept, "epe": epe,
        "zero_mv": (m["motion_x"] == 0) & (m["motion_y"] == 0),
        "gt_mag": gt_mag, "mv_mag": mv_mag,
    }
    for t in taus:
        out[f"mv_moving@{t}"] = mv_mag > t
        out[f"gt_moving@{t}"] = gt_mag > t
        out[f"epe_ok@{t}"] = epe <= t
    return out


def compare_sequence(frames, seq, min_valid=0.5, taus=TAUS):
    """Run compare_frame over every P-frame of a decoded Sintel sequence.

    frames: output of mvs.extract_mvs (display order, index 0 = Sintel frame_0001)
    seq:    flo.SintelSeq
    Returns concatenated per-block dict with an added `frame` column.
    """
    parts = []
    for fr in frames:
        d = fr["index"]
        if fr["pict_type"] != "P" or d == 0 or fr["mvs"].size == 0:
            continue
        r = compare_frame(fr["mvs"], seq.flow_into(d), seq.bad_mask_into(d), min_valid, taus)
        r["frame"] = np.full(r["epe"].shape[0], d)
        parts.append(r)
    if not parts:
        return {}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def summarize(res, taus=TAUS, textured=None):
    """Headline numbers over kept blocks. `textured` optional bool mask to sub-select."""
    k = res["kept"] if textured is None else res["kept"] & textured
    e = res["epe"][k]
    s = {
        "blocks_total": int(res["kept"].size),
        "blocks_kept": int(k.sum()),
        "epe_median": float(np.median(e)) if e.size else float("nan"),
        "epe_mean": float(np.mean(e)) if e.size else float("nan"),
        "epe_p90": float(np.percentile(e, 90)) if e.size else float("nan"),
        "zero_mv_frac": float(res["zero_mv"][k].mean()) if e.size else float("nan"),
    }
    for t in taus:
        mvm, gtm = res[f"mv_moving@{t}"][k], res[f"gt_moving@{t}"][k]
        s[f"gate_agree@{t}"] = float((mvm == gtm).mean()) if e.size else float("nan")
        s[f"gt_moving_frac@{t}"] = float(gtm.mean()) if e.size else float("nan")
        s[f"mv_moving_frac@{t}"] = float(mvm.mean()) if e.size else float("nan")
        s[f"epe_ok_frac@{t}"] = float(res[f"epe_ok@{t}"][k].mean()) if e.size else float("nan")
    return s
