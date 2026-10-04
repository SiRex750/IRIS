"""Review-2 item 8(a): confirmatory C1-C4 (STALE_MOV, tau = 1) recomputed on HIGH-TEXTURE cells only.
ROBUSTNESS ANALYSIS - the pre-registered verdicts (p3/virat_confirm/verdicts.csv) are not changed.

High texture = cell texture (analyze_virat.cell_texture: mean |grad Y| over the 4x4 cell of source frame t) >= the
75th percentile of cell texture over all cells of source frames 1..n-1 of all 50 analysed clips - the same
threshold V0 uses (analyze_confirm.main: histogram with bin 0.001, (searchsorted(cum, 0.75 total) + 1) x bin),
recomputed here and checked against summary.md (9.829). Source frames: 300.. as the collection (decode_source,
start from the manifest). Moving / stale cells are then counted with valid := valid & high texture, everything
else as analyze_confirm (gate_counts, d1 for bf 0 arms, B naive for B arms). The original (all valid cells) is
recomputed alongside and must equal results.csv. Texture arrays of the eligible clips are cached in
_cache/tex_<clip>.npz (not committed) for items 8(b) and 9. Writes i08a_high_texture.md / .csv.
"""
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

from _common import P3, HERE, CPREDS, eligible, V, cctv_base, cctv_counts, cpred_table, write

CACHE = os.path.join(HERE, "_cache")
TEX_BIN, TEX_MAX = 0.001, 200.0


def texture(clip):
    sys.path.insert(0, os.path.join(P3, "virat"))
    import json
    from analyze_virat import cell_texture
    from run_virat import decode_source, VIRAT
    fn = os.path.join(CACHE, f"tex_{clip}.npz")
    man = json.load(open(os.path.join(P3, "virat_confirm", "manifest.json")))
    n = np.load(os.path.join(P3, "virat_confirm", "data", clip, "raft.npz"))["splat_count"].shape[0] + 1
    if os.path.exists(fn):
        tex = np.load(fn)["tex"]
    else:
        src, _ = decode_source(os.path.join(VIRAT, clip + ".mp4"), n, start=int(man["config"]["start"]))
        H = src[0].shape[0] * 2 // 3
        tex = np.zeros((n, (H // 4) * (src[0].shape[1] // 4)), np.float32)
        for t in range(1, n):
            tex[t] = cell_texture(src[t][:H].astype(np.float64)).ravel()
        np.savez_compressed(fn, tex=tex)
    hist = np.bincount(np.minimum((tex[1:] / TEX_BIN).astype(np.int64), int(TEX_MAX / TEX_BIN) - 1).ravel(),
                       minlength=int(TEX_MAX / TEX_BIN))
    return clip, hist


def counts(args):
    clip, q75 = args
    tex = np.load(os.path.join(CACHE, f"tex_{clip}.npz"))["tex"]
    base = cctv_base(clip)
    return cctv_counts(clip, base, {"all valid (original)": None, "high texture": tex >= q75})


def main():
    os.makedirs(CACHE, exist_ok=True)
    clips = sorted(os.listdir(os.path.join(P3, "virat_confirm", "data")))
    el = eligible()
    with mp.get_context("spawn").Pool(4) as pool:
        hists = dict(pool.map(texture, clips))
    cum = np.cumsum(sum(hists.values()))
    q75 = (np.searchsorted(cum, 0.75 * cum[-1]) + 1) * TEX_BIN
    with mp.get_context("spawn").Pool(4) as pool:
        rows = [r for rs in pool.map(counts, [(c, q75) for c in el]) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "i08a_high_texture.csv"), index=False)
    o = df[df["mask"] == "all valid (original)"]
    chk = all(np.isclose(r.stale_mov, V(r.arm, r.group).get(r.clip), equal_nan=True) for r in o.itertuples())
    tab, recs = cpred_table(df, ["all valid (original)", "high texture"], el)
    pd.DataFrame(recs).to_csv(os.path.join(HERE, "i08a_high_texture_summary.csv"), index=False)
    ht = df[df["mask"] == "high texture"]
    share = (ht.groupby(["arm", "group"]).moving_cells.sum() / o.groupby(["arm", "group"]).moving_cells.sum())
    write("i08a_high_texture.md", [
        "# Item 8(a): C1-C4 on high-texture cells only (robustness; not a verdict)", "",
        f"High texture: cell texture >= {q75:.3f} grey levels/px (75th percentile over all cells of the 50 clips; "
        "V0's threshold, summary.md: 9.829). 27 eligible clips, tau = 1, STALE_MOV.", "",
        tab, "",
        f"- Original rows reproduce p3/virat_confirm/results.csv stale_mov: {'yes' if chk else 'NO'}.",
        f"- High-texture moving cells as a share of all moving cells (pooled over clips, per arm): "
        + ", ".join(f"{a}/{g} {v:.3f}" for (a, g), v in share.items()) + ".",
        "- High texture, clips with diff > 0 / diff < 0: " + "; ".join(
            f"{k.split()[0]} {int((dd > 0).sum())} / {int((dd < 0).sum())} (min {dd.min():.4f})"
            for k, dd in ((p[0], ht[ht.arm == p[1]].set_index("clip").stale_mov.reindex(el)
                           - ht[ht.arm == p[3]].set_index("clip").stale_mov.reindex(el)) for p in CPREDS)) + ".",
        "- 'count met' applies the pre-registered count rule to this subset for orientation only."])


if __name__ == "__main__":
    main()
