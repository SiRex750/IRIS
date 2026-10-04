"""Review-2 item 2: the quality-end and matched-QP B-frame comparisons at tau = 0.5 and 2 px (tau = 1 shown for
reference). SUPPLEMENTARY / ROBUSTNESS - the pre-registered verdicts are at tau = 1 and are not changed.

Sintel (FINAL pass, groups as in p3/PREREGISTRATION.md): stale/valid, d1 P-frame vectors for bf 0 arms, B_naive for
B arms. Ends: x264 CRF 12 / 45, NVENC QP 18 / 45; pairs: x264 QP 24 bf 0 vs bf 2 (B naive), NVENC QP 28 bf 0 vs
bf 2 B=P (B naive). Count: sequences (of 23) whose stale/valid rises by more than 0.001.
CCTV confirmatory (p3/virat_confirm/PREREGISTRATION.md): STALE_MOV over the 27 eligible clips, the C1-C4 pairs,
count of clips rising by more than 0.01.
Writes i02_thresholds.md / .csv.
"""
import pandas as pd

from _common import S, V, eligible, CPREDS, TAUS, f, table, write, HERE

SPAIRS = (("x264 CRF 45 vs 12", "x264_crf45", "d1", "x264_crf12", "d1"),
          ("NVENC QP 45 vs 18", "nvenc_qp45", "d1", "nvenc_qp18", "d1"),
          ("x264 QP 24 B vs bf 0", "x264_qp24_bf2_pb1", "B_naive", "x264_qp24", "d1"),
          ("NVENC QP 28 B=P vs bf 0", "nvenc_qp28_bf2_bqeq", "B_naive", "nvenc_qp28", "d1"))


def main():
    rows, L = [], ["# Item 2: thresholds 0.5 and 2 px (supplementary; verdicts stay at tau = 1)", ""]
    sl, cl = [], []
    for tau in TAUS:
        for lab, hi, gh, lo, gl in SPAIRS:
            a, b = S(lo, gl, tau=tau), S(hi, gh, tau=tau)
            d = b - a
            r = {"data": "Sintel final", "comparison": lab, "tau": tau, "n": int(d.notna().sum()),
                 "median_low": a.median(), "median_high": b.median(), "median_diff": d.median(),
                 "rise_count": int((d > 0.001).sum()), "rise_threshold": 0.001}
            rows.append(r)
            sl.append([lab, tau, f(r["median_low"]), f(r["median_high"]), f(r["median_diff"]),
                       f"{r['rise_count']}/{r['n']}"])
        el = eligible()
        for lab, hi, gh, lo, gl, _ in CPREDS:
            a, b = V(lo, gl, tau=tau).reindex(el), V(hi, gh, tau=tau).reindex(el)
            d = b - a
            r = {"data": "CCTV confirmatory", "comparison": lab, "tau": tau, "n": int(d.notna().sum()),
                 "median_low": a.median(), "median_high": b.median(), "median_diff": d.median(),
                 "rise_count": int((d > 0.01).sum()), "rise_threshold": 0.01}
            rows.append(r)
            cl.append([lab, tau, f(r["median_low"]), f(r["median_high"]), f(r["median_diff"]),
                       f"{r['rise_count']}/{r['n']}"])
    pd.DataFrame(rows).to_csv(f"{HERE}/i02_thresholds.csv", index=False)
    order = [p[0] for p in SPAIRS] + [p[0] for p in CPREDS]
    key = lambda x: (order.index(x[0]), x[1])  # noqa: E731
    L += ["## Sintel, FINAL pass, stale/valid (median over 23 sequences)", "",
          table(["comparison", "tau", "median low end / bf 0", "median high end / B", "median diff",
                 "sequences with rise > 0.001"], sorted(sl, key=key)), "",
          "## CCTV confirmatory test, STALE_MOV (median over the 27 eligible clips)", "",
          table(["comparison", "tau", "median low end / bf 0", "median high end / B", "median diff",
                 "clips with rise > 0.01"], sorted(cl, key=key)), "",
          "tau = 1 rows reproduce the pre-registered numbers (p3/sweep/summary.md, p3/virat_confirm/verdicts.csv)."]
    write("i02_thresholds.md", L)


if __name__ == "__main__":
    main()
