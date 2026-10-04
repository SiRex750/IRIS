"""Review-2 item 8(b), analysis step: confirmatory C1-C4 (STALE_MOV, tau = 1) recomputed on RAFT forward-backward
CONSISTENT cells only. ROBUSTNESS ANALYSIS - the pre-registered verdicts are not changed.

Inputs: forward RAFT as collected (p3/virat_confirm/data/<clip>/raft.npz: splat_t = forward flow t-1 -> t splatted
onto frame t's 4x4 grid, the flow every gate metric uses) and the new backward RAFT (_raft_bwd/<clip>.npz: bwd_t =
mean flow t -> t-1 over each 4x4 cell of frame t; i08b_raft_backward_compute.py).

Check (standard forward-backward test, Sundaram et al. 2010 / Meister et al. 2018 constants):
    |f + b_w|^2 < 0.01 (|f|^2 + |b_w|^2) + 0.5
applied per cell of frame t with f = splat_t (forward flow of the frame t-1 pixels that land in the cell) and
b_w = bwd_t of the same cell. Because splat_t already sits at the forward landing position, b_w is the backward
flow sampled where the forward flow lands, i.e. the warped backward flow, at 4x4-cell resolution. This is a cell-level
version of the per-pixel check: the saved forward flow exists only on the cell grid, and recomputing forward flow
was out of scope. A cell with no forward landing (NaN) is not valid anyway.
Moving / stale cells are then counted with valid := valid & consistent; everything else as analyze_confirm.
Writes i08b_fb_consistency.md / .csv (+ _summary.csv) and _cache/fb_<clip>.npz (consistency mask, not committed).
"""
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from _common import HERE, eligible, cctv_base, cctv_counts, cpred_table, table, write

BWD = os.path.join(HERE, "_raft_bwd")
CACHE = os.path.join(HERE, "_cache")


def fb_mask(clip, base):
    b = np.load(os.path.join(BWD, f"{clip}.npz"))["bwd_t"].astype(np.float32)
    fw = base["splat"]
    assert b.shape == fw.shape, (b.shape, fw.shape)
    s = fw + b
    lhs = (s ** 2).sum(-1)
    rhs = 0.01 * ((fw ** 2).sum(-1) + (b ** 2).sum(-1)) + 0.5
    ok = np.zeros((base["n"], base["Hc"] * base["Wc"]), bool)
    ok[1:] = (lhs < rhs).reshape(base["n"] - 1, -1)  # NaN forward -> False
    np.savez_compressed(os.path.join(CACHE, f"fb_{clip}.npz"), ok=ok)
    return ok


def one(clip):
    base = cctv_base(clip)
    ok = fb_mask(clip, base)
    v, mov = base["valid"][1:], base["valid"][1:] & (base["gmag"][1:] > 2.0)
    stats = {"clip": clip, "valid_cells": int(v.sum()), "consistent_of_valid": float(ok[1:][v].mean()),
             "moving_cells": int(mov.sum()), "consistent_of_moving": float(ok[1:][mov].mean())}
    return cctv_counts(clip, base, {"all valid (original)": None, "FB-consistent": ok}), stats


def main():
    os.makedirs(CACHE, exist_ok=True)
    el = eligible()
    missing = [c for c in el if not os.path.exists(os.path.join(BWD, f"{c}.npz"))]
    assert not missing, f"backward flow missing for {missing}"
    with mp.get_context("spawn").Pool(4) as pool:
        out = pool.map(one, el)
    df = pd.DataFrame([r for rs, _ in out for r in rs])
    st = pd.DataFrame([s for _, s in out]).set_index("clip")
    df.to_csv(os.path.join(HERE, "i08b_fb_consistency.csv"), index=False)
    st.to_csv(os.path.join(HERE, "i08b_fb_consistency_cells.csv"))
    tab, recs = cpred_table(df, ["all valid (original)", "FB-consistent"], el)
    pd.DataFrame(recs).to_csv(os.path.join(HERE, "i08b_fb_consistency_summary.csv"), index=False)
    write("i08b_fb_consistency.md", [
        "# Item 8(b): C1-C4 on RAFT forward-backward consistent cells (robustness; not a verdict)", "",
        "Consistency |f + b_w|^2 < 0.01(|f|^2 + |b_w|^2) + 0.5, at 4x4-cell level (see script docstring). "
        "27 eligible clips, frames 300-599, tau = 1, STALE_MOV.", "",
        f"- Consistent share of valid cells: median {st.consistent_of_valid.median():.4f} "
        f"(min {st.consistent_of_valid.min():.4f} {st.consistent_of_valid.idxmin()}, max {st.consistent_of_valid.max():.4f}).",
        f"- Consistent share of moving cells (|RAFT| > 2 px): median {st.consistent_of_moving.median():.4f} "
        f"(min {st.consistent_of_moving.min():.4f} {st.consistent_of_moving.idxmin()}, "
        f"max {st.consistent_of_moving.max():.4f}).", "",
        tab, "",
        "- 'count met' applies the pre-registered count rule to this subset for orientation only.", "",
        "## Per clip consistency", "",
        table(["clip", "valid cells", "consistent / valid", "moving cells", "consistent / moving"],
              [[c, int(r.valid_cells), f"{r.consistent_of_valid:.4f}", int(r.moving_cells),
                f"{r.consistent_of_moving:.4f}"] for c, r in st.iterrows()])])


if __name__ == "__main__":
    main()
