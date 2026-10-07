"""Writes p3/review_fixes3/README.md from the item CSVs (run r1..r5 first) and checks the draft's numbers against
them. Every number in README.md comes from this script. Exit code 1 if any draft number fails to match.
"""
import os
import sys

import numpy as np
import pandas as pd

from _common import HERE, S, matched_arms, f, table
from r4_bootstrap_ci import CRFS

rd = lambda n: pd.read_csv(os.path.join(HERE, n))  # noqa: E731


def spearman_rhos():
    """Per-sequence Spearman rho(CRF, stale/valid) over the 7 x264 CRF levels, as in r4_bootstrap_ci.py."""
    ser = pd.DataFrame({c: S(f"x264_crf{c}", "d1") for c in CRFS})
    return ser.apply(lambda r: pd.Series(CRFS, dtype=float).corr(pd.Series(r.to_numpy()), method="spearman"),
                     axis=1).to_numpy()


def main():
    r1, r2, r3, r4, r5 = (rd(f"{p}.csv") for p in ("r1_eight_point_seven", "r2_mpeg4_p1", "r3_fold_changes",
                                                 "r4_bootstrap_ci", "r5_tau_appendix"))
    m = matched_arms()
    med = lambda eid, grp="d1": S(eid, grp).median()  # noqa: E731
    p2n = r1[(r1.encode_id == "nvenc_qp28_bf2_bqeq") & (r1.metric == "stale_of_valid")].iloc[0]
    p2d = r1[(r1.encode_id == "nvenc_qp28_bf2") & (r1.metric == "stale_of_valid")].iloc[0]
    smv = r1[(r1.encode_id == "nvenc_qp28_bf2_bqeq") & (r1.metric == "STALE_MOV")].iloc[0]

    def pc(x, nd):
        return f"{100 * x:.{nd}f}"

    # (draft text, what it is taken to mean, computed text in the draft's precision)
    checks = [
        ("0.34%", f"x264 at ~41 dB ({m['x264'][0]}) median stale/valid", pc(med(m["x264"][0]), 2) + "%"),
        ("3.3%", f"NVENC at ~41 dB ({m['NVENC'][0]}) median stale/valid", pc(med(m["NVENC"][0]), 1) + "%"),
        ("16%", f"MPEG-4 at ~41 dB ({m['MPEG-4'][0]}) median stale/valid", pc(med(m["MPEG-4"][0]), 0) + "%"),
        ("0.55→4.4%", "x264 QP 24 bf 0 -> B (pb1, B_naive) medians",
         f"{pc(med('x264_qp24'), 2)}→{pc(med('x264_qp24_bf2_pb1', 'B_naive'), 1)}%"),
        ("3.3→7.0%", "NVENC QP 28 bf 0 -> B=P (bqeq, B_naive) medians",
         f"{pc(med('nvenc_qp28'), 1)}→{pc(med('nvenc_qp28_bf2_bqeq', 'B_naive'), 1)}%"),
        ("3.3→8.7%", "NVENC QP 28 bf 0 -> bf 2 default B QP (B_naive) medians",
         f"{pc(med('nvenc_qp28'), 1)}→{pc(med('nvenc_qp28_bf2', 'B_naive'), 1)}%"),
        ("22/23", "P2 NVENC QP 28 B=P - bf 0 > 0.001 (pre-registered count)",
         f"{int(p2n['n_diff_gt_0.001'])}/{int(p2n['n'])}"),
    ]
    clines, bad = [], []
    for draft, meaning, got in checks:
        ok = draft == got
        clines.append([draft, meaning, got, "match" if ok else "**MISMATCH**"])
        if not ok:
            bad.append(draft)

    # key lines
    x = r3.set_index("contrast")
    k3 = "; ".join(f"{c}: {f(r.median_low)} -> {f(r.median_high)} ({r.ratio_of_medians:.1f}x of medians, "
                   f"{r.median_per_seq_ratio:.1f}x per-seq)" for c, r in x.iterrows())
    s4 = r4[r4.comparison.str.match(r"P[12] ") & r4.metric.str.startswith("median per-seq diff")]
    k4s = "; ".join(f"{r.comparison} {f(r.estimate)} [{f(r.ci_low)}, {f(r.ci_high)}]" for r in s4.itertuples())
    c4 = r4[r4.data != "Sintel"]
    k4c = "; ".join(f"{r.comparison.split()[0]} {f(r.estimate)} scene [{f(r.ci_low)}, {f(r.ci_high)}] / clip "
                    f"[{f(r2_.ci_low)}, {f(r2_.ci_high)}]"
                    for r, r2_ in zip(c4[c4.method.str.startswith("scene")].itertuples(),
                                      c4[c4.method.str.startswith("clip")].itertuples()))
    n_excl = int(r4[r4.excludes_0 == True].shape[0])  # noqa: E712
    n_zt = int(r4.excludes_0.notna().sum())
    t5 = r5[r5.available & (r5.tau != 1.0)]
    k5 = "; ".join(f"{'Sintel ' + r.contrast if r.data.startswith('Sintel') else 'CCTV ' + r.contrast.split()[0]} "
                   f"tau {r.tau:g}: {f(r.median_low)}->{f(r.median_high)}, rise>0.01 {int(r['n_rise_gt_0.01'])}/"
                   f"{int(r.n)}" for _, r in t5.iterrows())
    mp = r2[r2.prediction.str.contains("MPEG-4")].iloc[0]
    sp = r4[r4.comparison.str.startswith("P1 Spearman")].iloc[0]
    rho = spearman_rhos()
    n_rho1 = int(np.isclose(rho, 1.0).sum())

    L = ["# p3/review_fixes3: follow-ups to the third internal review of the Paper 3 draft", "",
         "**Reporting and exploratory analysis on saved data only. Nothing here is a new verdict.** No encoding, MV "
         "extraction or RAFT was run; the scripts only read p3/sweep/results.csv, p3/PREREGISTRATION.md, "
         "p3/sweep/independent_verdicts.txt, p3/review_fixes/sintel_stale_mov.csv and p3/virat_confirm "
         "(results.csv, eligibility_v0.csv). No existing results file is modified. Every number in this folder is "
         "printed by a script; this README is written by `make_readme.py`.", "",
         table(["item", "script", "output"], [
             ["R1. The two \"8.7%\" values; the \"22/23\" count", "`r1_eight_point_seven.py`",
              "`r1_eight_point_seven.md/.csv`"],
             ["R2. The MPEG-4 P1 row", "`r2_mpeg4_p1.py`", "`r2_mpeg4_p1.md/.csv`"],
             ["R3. Fold changes", "`r3_fold_changes.py`", "`r3_fold_changes.md/.csv`"],
             ["R4. Bootstrap 95% CIs (exploratory, not pre-registered)", "`r4_bootstrap_ci.py`",
              "`r4_bootstrap_ci.md/.csv`"],
             ["R5. tau = 0.5 / 2 appendix", "`r5_tau_appendix.py`", "`r5_tau_appendix.md/.csv`"],
             ["README + draft-number check", "`make_readme.py`", "`README.md`"]]), "",
         "Run from this folder: `python r1_eight_point_seven.py`, ..., `python r5_tau_appendix.py`, then "
         "`python make_readme.py`. `_common.py` holds the shared readers.", "",
         "## Draft numbers checked", "",
         "Each draft number is recomputed in the draft's precision (Sintel, FINAL pass, tau = 1, median over 23 "
         "sequences unless stated).", "",
         table(["draft", "taken to be", "recomputed", "result"], clines), "",
         ("All draft numbers match." if not bad else "MISMATCH in: " + ", ".join(bad) + ".")
         + f" Note on \"22/23\": it is the > 0.001 count; with a > 0.01 rule the B=P arm is "
           f"{int(p2n['n_diff_gt_0.01'])}/23, the default-B-QP arm {int(p2d['n_diff_gt_0.01'])}/23 "
           f"({int(p2d['n_diff_gt_0.001'])}/23 at > 0.001, {int(p2d['n_diff_gt_0'])}/23 at > 0), and Sintel STALE_MOV "
           f"B=P {int(smv['n_diff_gt_0.01'])}/23 ({int(smv['n_diff_gt_0'])}/23 at > 0).", "",
         "## Key numbers", "",
         f"- **R1:** stale/valid median NVENC QP 28 bf 0 = {f(r1.iloc[0]['median'])}, bf 2 default B QP (B_naive) = "
         f"{f(p2d['median'])}, B=P = {f(p2n['median'])}; Sintel STALE_MOV B=P pair = {f(r1.iloc[3]['median'])} -> "
         f"{f(smv['median'])}. The two \"8.7%\" are {f(p2d['median'])} (stale/valid, default B QP) and "
         f"{f(smv['median'])} (STALE_MOV, B=P): different numbers. Default-B-QP minus bf 0: "
         f"{int(p2d['n_diff_gt_0.01'])}/23 > 0.01, {int(p2d['n_diff_gt_0.001'])}/23 > 0.001, "
         f"{int(p2d['n_diff_gt_0'])}/23 > 0; \"22/23\" confirmed as the "
         f"pre-registered B=P count (> 0.001).",
         f"- **R2:** MPEG-4 P1 was pre-registered as reported-only, not a kill-rule row: {mp['rule']}, "
         f"threshold {mp['threshold']}, count {mp['count']}, {mp['verdict']} (matches independent_verdicts.txt: "
         f"{'yes' if mp['recount_matches_record'] else 'NO'}).",
         f"- **R3:** {k3}.",
         f"- **R4 (exploratory):** Sintel median paired diff [95% CI]: {k4s}. CCTV STALE_MOV: {k4c}. "
         f"{n_excl}/{n_zt} intervals where the zero test applies exclude 0. The P1 Spearman interval "
         f"[{f(sp.ci_low)}, {f(sp.ci_high)}] is degenerate: per-sequence rho = 1 in {n_rho1}/{len(rho)} sequences "
         f"(minimum {rho.min():.3f}), so report it as that count, not as a CI.",
         f"- **R5:** {k5}. Matched-quality codec contrasts: not available on CCTV."]
    open(os.path.join(HERE, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    if bad:
        print("\nDRAFT NUMBER MISMATCH:", bad)
        sys.exit(1)


if __name__ == "__main__":
    main()
