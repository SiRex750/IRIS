"""Review-2 item 4: every Sintel prediction recomputed on 19 sequences, excluding alley_1, ambush_5, ambush_7 and
bandage_2. SENSITIVITY ANALYSIS - the pre-registered verdicts (23 sequences, p3/sweep/summary.md) are not changed.

Same computations as p3/sweep/summarize_sweep.py verdicts() at tau = 1 on the FINAL pass. Count thresholds are scaled
proportionally to 19 sequences and rounded up: ceil(18 x 19 / 23) = 15, ceil(16 x 19 / 23) = 14. The median
thresholds (P2b < 0.005; P3 < 0.5 x median |CRF 33 - CRF 23|; P2c) are not counts and are unchanged in form; their
medians are taken over the 19 sequences. Writes i04_19seq_verdicts.md / .csv.
"""
import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import S, seqs, CRFS, f, table, write, HERE

DROP = ("alley_1", "ambush_5", "ambush_7", "bandage_2")
P3_ARMS = [("ref 2", "x264_crf23_ref2", "d1(assumed)", "x264_crf23", "d1"),
           ("ref 3", "x264_crf23_ref3", "d1(assumed)", "x264_crf23", "d1"),
           ("keyint 30", "x264_crf23_keyint30", "d1", "x264_crf23", "d1"),
           ("umh 32", "x264_crf23_umh32", "d1", "x264_crf23_umh16", "d1"),
           ("umh 64", "x264_crf23_umh64", "d1", "x264_crf23_umh16", "d1")]


def verdicts(keep, tau=1.0):
    N = len(keep)
    sc = lambda k: math.ceil(k * N / 23)  # noqa: E731
    s = lambda e, g: S(e, g, tau=tau).reindex(keep)  # noqa: E731
    rows = []

    def add(name, value, thr, ok):
        rows.append((name, value, thr, "PASS" if ok else "FAIL"))

    for lab, hi, lo, k in [("P1 x264 CRF 45 vs 12", "x264_crf45", "x264_crf12", 18),
                           ("P1 NVENC QP 45 vs 18", "nvenc_qp45", "nvenc_qp18", 18),
                           ("P1 mpeg4 q 31 vs 2 (reported, not a kill rule)", "mpeg4_q31", "mpeg4_q2", 16)]:
        n = int(((s(hi, "d1") - s(lo, "d1")) > 0.001).sum())
        add(lab, f"{n}/{N} with diff > 0.001", f">= {sc(k)}/{N} (= {k}/23 scaled)", n >= sc(k))
    M = pd.concat([s(f"x264_crf{c}", "d1") for c in CRFS], axis=1)
    rho = pd.Series({q: spearmanr(CRFS, M.loc[q].to_numpy()).statistic for q in keep})
    n = int((rho >= 0.8).sum())
    add("P1 Spearman(CRF, stale/valid) >= 0.8", f"{n}/{N}", f">= {sc(16)}/{N} (= 16/23 scaled)", n >= sc(16))
    bx = s("x264_qp24_bf2_pb1", "B_naive") - s("x264_qp24", "d1")
    bn = s("nvenc_qp28_bf2_bqeq", "B_naive") - s("nvenc_qp28", "d1")
    n = int((bx > 0.001).sum())
    add("P2 x264 QP 24: B naive minus bf0 d1", f"{n}/{N} with diff > 0.001", f">= {sc(18)}/{N}", n >= sc(18))
    n = int((bn > 0.001).sum())
    add("P2 NVENC QP 28 B=P: B naive minus bf0 d1", f"{n}/{N} with diff > 0.001", f">= {sc(16)}/{N}", n >= sc(16))
    for lab, eid in [("x264 QP 24 pair", "x264_qp24_bf2_pb1"), ("NVENC QP 28 B=P pair", "nvenc_qp28_bf2_bqeq")]:
        m = (s(eid, "B_scaled") - s(eid, "B_naive")).abs().median()
        add(f"P2b abs(B scaled - B naive), {lab}", f"median {f(m)}", "< 0.005", m < 0.005)
    base = (s("x264_crf33", "d1") - s("x264_crf23", "d1")).abs().median()
    p3 = {}
    for lab, a, ga, b, gb in P3_ARMS:
        p3[lab] = (s(a, ga) - s(b, gb)).abs().median()
        add(f"P3 {lab}", f"median abs diff {f(p3[lab], 5)}", f"< 0.5 x {f(base, 5)} = {f(0.5 * base, 5)}",
            p3[lab] < 0.5 * base)
    mb, mx = bx.median(), max(p3.values())
    add("P2c median x264 B effect > every P3 median", f"{f(mb, 5)}", f"> {f(mx, 5)} ({max(p3, key=p3.get)})", mb > mx)
    return rows


def main():
    all_, keep = seqs(), [q for q in seqs() if q not in DROP]
    assert len(all_) == 23 and len(keep) == 19
    v23, v19 = verdicts(all_), verdicts(keep)
    # the 23-sequence values must reproduce summary.md's thresholds (scaled thresholds equal the originals at 23)
    lines = [[a[0], a[1], a[2], a[3], b[1], b[2], b[3], "" if a[3] == b[3] else "CHANGED"] for a, b in zip(v23, v19)]
    pd.DataFrame(lines, columns=["prediction", "value_23", "threshold_23", "verdict_23", "value_19", "threshold_19",
                                 "verdict_19", "change"]).to_csv(f"{HERE}/i04_19seq_verdicts.csv", index=False)
    write("i04_19seq_verdicts.md", [
        "# Item 4: Sintel predictions on 19 sequences (sensitivity; not a verdict)", "",
        f"Excluded: {', '.join(DROP)}. tau = 1, FINAL pass, as p3/sweep/summarize_sweep.py. "
        "Count thresholds scaled to 19 sequences and rounded up: 18/23 -> 15/19, 16/23 -> 14/19. "
        "Median thresholds unchanged in form, medians over the 19 sequences.", "",
        table(["prediction", "23 seq value", "threshold", "verdict", "19 seq value", "threshold", "verdict",
               "change"], lines), "",
        f"Verdicts changed: {sum(1 for x in lines if x[-1])}/{len(lines)}."])


if __name__ == "__main__":
    main()
