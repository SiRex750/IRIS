"""Review-2 item 5: share of 4x4 cells that are not valid (excluded from every gate metric). SUPPLEMENTARY.

Sintel: per sequence (GT is pass-independent; FINAL is the primary pass), over frames t = 1..n-1, a cell is valid if
gt_cover >= 0.5 (lib.gt_grid forward splat of the non-occluded, valid t-1 pixels; p3/sweep/run_sweep.py load_seq).
GT read through p3/review_fixes/_common.gt_cells (same construction; cached). Cross-check: results.csv column
cell_excluded_share of x264_crf12 / d1 (a bf 0 arm whose d1 group is every frame 1..n-1).
CCTV confirmatory: per clip, over frames 1..n-1 of p3/virat_confirm/data/<clip>/raft.npz, valid = RAFT splat count
>= 8 of 16 pixels (analyze_virat COVER_COUNT, the same rule with RAFT as the cover). Writes i05_excluded_cells.md/.csv.
"""
import os


import numpy as np
import pandas as pd

from _common import P3, S, seqs, eligible, f, table, write, HERE

import importlib.util  # noqa: E402

# p3/review_fixes/_common.py, loaded under another name (this folder has its own _common)
_rf = importlib.util.spec_from_file_location("rf_common", os.path.join(P3, "review_fixes", "_common.py"))
rf = importlib.util.module_from_spec(_rf)
_rf.loader.exec_module(rf)


def mmm(s):
    return f"min {f(s.min())} ({s.idxmin()}), median {f(s.median())}, max {f(s.max())} ({s.idxmax()})"


def main():
    rows = []
    chk = S("x264_crf12", "d1", col="cell_excluded_share")
    for q in seqs():
        gt, (Hc, Wc) = rf.gt_cells(q)
        v = np.stack([gt[t]["valid"] for t in sorted(gt)])
        nolanding = np.stack([~np.isfinite(gt[t]["gmag"]) for t in sorted(gt)])
        rows.append({"data": "Sintel", "unit": q, "frames": v.shape[0], "cells_per_frame": Hc * Wc,
                     "excluded_share": 1 - v.mean(), "no_landing_share": nolanding.mean(),
                     "partial_cover_share": (1 - v.mean()) - nolanding.mean(),
                     "results_csv_x264_crf12_d1": chk[q]})
    sd = pd.DataFrame(rows).set_index("unit")
    el = set(eligible())
    crow = []
    for clip in sorted(os.listdir(os.path.join(P3, "virat_confirm", "data"))):
        z = np.load(os.path.join(P3, "virat_confirm", "data", clip, "raft.npz"))
        cnt = z["splat_count"]
        crow.append({"data": "CCTV confirmatory", "unit": clip, "eligible": clip in el, "frames": cnt.shape[0],
                     "cells_per_frame": cnt.shape[1] * cnt.shape[2], "excluded_share": float((cnt < 8).mean()),
                     "no_landing_share": float((cnt == 0).mean()),
                     "partial_cover_share": float(((cnt > 0) & (cnt < 8)).mean())})
    cd = pd.DataFrame(crow).set_index("unit")
    pd.concat([sd.reset_index(), cd.reset_index()]).to_csv(os.path.join(HERE, "i05_excluded_cells.csv"), index=False)
    ce = cd[cd.eligible]
    write("i05_excluded_cells.md", [
        "# Item 5: excluded-cell share (cells that are not valid; supplementary)", "",
        "## Sintel (23 sequences, frames 1..n-1, valid = GT cover >= 0.5)", "",
        f"- Excluded share: {mmm(sd.excluded_share)}.",
        f"- Of which no GT pixel lands in the cell: median {f(sd.no_landing_share.median())}; "
        f"partial cover (0 < cover < 0.5): median {f(sd.partial_cover_share.median())}.",
        f"- Max |this - results.csv cell_excluded_share (x264_crf12, d1)|: "
        f"{(sd.excluded_share - sd.results_csv_x264_crf12_d1).abs().max():.2e}.", "",
        table(["sequence", "excluded", "no landing", "partial cover"],
              [[q, f(r.excluded_share), f(r.no_landing_share), f(r.partial_cover_share)] for q, r in sd.iterrows()]),
        "", "## CCTV confirmatory test (frames 1..n-1, valid = RAFT splat count >= 8 of 16)", "",
        f"- All {len(cd)} clips: {mmm(cd.excluded_share)}.",
        f"- {len(ce)} eligible clips: {mmm(ce.excluded_share)}.",
        f"- Of which no RAFT landing: median {f(cd.no_landing_share.median(), 5)} (all clips).", "",
        table(["clip", "eligible", "excluded", "no landing", "partial cover"],
              [[c, r.eligible, f(r.excluded_share, 5), f(r.no_landing_share, 5), f(r.partial_cover_share, 5)]
               for c, r in cd.iterrows()])])


if __name__ == "__main__":
    main()
