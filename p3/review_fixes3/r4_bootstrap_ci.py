"""Review-3 item R4: bootstrap 95% CIs. EXPLORATORY, NOT PRE-REGISTERED.

Percentile intervals (numpy.percentile, 2.5 / 97.5, linear interpolation), 10,000 resamples,
numpy.random.default_rng(20261006). Draw order from the one generator: (1) Sintel sequence indices, shape
(10000, 23); (2) CCTV scene indices, shape (10000, 7); (3) CCTV clip indices, shape (10000, 27). The same Sintel
draws are used for every Sintel comparison, and the same CCTV draws for C1-C4, so comparisons are resampled jointly.

Sintel (p3/sweep/results.csv, FINAL pass, tau = 1): the 23 sequences resampled with replacement. Every comparison
of p3/PREREGISTRATION.md, each in the metric of its verdict:
  P1 (x264, NVENC, MPEG-4 reported-only) and P2 (x264, NVENC B=P): median per-sequence paired difference of
  stale/valid; P1 Spearman: median per-sequence rho over the 7 x264 CRF levels; P2b: median per-sequence
  |B_scaled - B_naive|; P3: median per-sequence |change|, its reference median |CRF 33 - CRF 23|, and the margin
  0.5 x reference - median |change| (> 0 = the P3 condition holds), computed within each resample; P2c: median x264
  B effect minus the largest P3 median |change|, within each resample.
CCTV confirmatory test (p3/virat_confirm/results.csv, pre-registration d62753f, tau = 1, 27 eligible clips in
7 scenes): C1-C4 median per-clip paired difference of STALE_MOV. Primary: scene-cluster bootstrap (7 scenes drawn
with replacement, all eligible clips of each drawn scene kept). Beside it: a plain clip-level bootstrap.
Writes r4_bootstrap_ci.md / .csv. No verdict changes.
"""
import os

import numpy as np
import pandas as pd

from _common import HERE, S, CPREDS, V, eligible, seqs, f, table, write

SEED, B = 20261006, 10_000
CRFS = (12, 18, 23, 28, 33, 38, 45)
P3ARMS = (("ref 2", "x264_crf23_ref2", "d1(assumed)", "x264_crf23"),
          ("ref 3", "x264_crf23_ref3", "d1(assumed)", "x264_crf23"),
          ("keyint 30", "x264_crf23_keyint30", "d1", "x264_crf23"),
          ("umh 32", "x264_crf23_umh32", "d1", "x264_crf23_umh16"),
          ("umh 64", "x264_crf23_umh64", "d1", "x264_crf23_umh16"))


def ci(stats):
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return lo, hi


def boot_median(x, idx):
    """x: (n,) per-unit values; idx: (B, n) resample indices -> (B,) medians."""
    return np.median(x[idx], axis=1)


def main():
    rng = np.random.default_rng(SEED)
    sq = seqs()
    sidx = rng.integers(0, len(sq), size=(B, len(sq)))
    el = eligible()
    scenes = sorted(el.unique())
    scidx = rng.integers(0, len(scenes), size=(B, len(scenes)))
    clidx = rng.integers(0, len(el), size=(B, len(el)))

    rows = []

    def add(data, comp, metric, est, stats, ref=0.0, method="sequence bootstrap (23)", zero_test=True):
        lo, hi = ci(stats)
        rows.append({"data": data, "comparison": comp, "metric": metric, "method": method, "estimate": est,
                     "ci_low": lo, "ci_high": hi,
                     "excludes_0": bool(lo > 0 or hi < 0) if zero_test else None, "reference": ref,
                     "excludes_reference": bool(lo > ref or hi < ref)})

    sint = "Sintel"
    # P1 and P2: paired differences of stale/valid
    for comp, hi_, ghi, lo_, glo in (("P1 x264 CRF 45 - CRF 12", "x264_crf45", "d1", "x264_crf12", "d1"),
                                     ("P1 NVENC QP 45 - QP 18", "nvenc_qp45", "d1", "nvenc_qp18", "d1"),
                                     ("P1 MPEG-4 q31 - q2 (reported, not kill)", "mpeg4_q31", "d1", "mpeg4_q2", "d1"),
                                     ("P2 x264 QP 24 B - bf 0", "x264_qp24_bf2_pb1", "B_naive", "x264_qp24", "d1"),
                                     ("P2 NVENC QP 28 B=P - bf 0", "nvenc_qp28_bf2_bqeq", "B_naive", "nvenc_qp28",
                                      "d1")):
        d = (S(hi_, ghi) - S(lo_, glo)).to_numpy()
        add(sint, comp, "median per-seq diff, stale/valid", np.median(d), boot_median(d, sidx))
        if comp.startswith("P2 x264"):
            dx = d
    # P1 Spearman
    ser = pd.DataFrame({c: S(f"x264_crf{c}", "d1") for c in CRFS})
    rho = ser.apply(lambda r: pd.Series(CRFS, dtype=float).corr(pd.Series(r.to_numpy()), method="spearman"),
                    axis=1).to_numpy()
    add(sint, "P1 Spearman(CRF, stale/valid), 7 x264 CRF levels", "median per-seq rho (verdict counts rho >= 0.8)",
        np.median(rho), boot_median(rho, sidx), ref=0.8, zero_test=False)
    # P2b
    for nm, eid in (("x264 QP 24", "x264_qp24_bf2_pb1"), ("NVENC QP 28 B=P", "nvenc_qp28_bf2_bqeq")):
        d = (S(eid, "B_scaled") - S(eid, "B_naive")).abs().to_numpy()
        add(sint, f"P2b {nm} |B_scaled - B_naive|", "median per-seq |diff| (verdict: < 0.005)", np.median(d),
            boot_median(d, sidx), ref=0.005, zero_test=False)
    # P3
    ref = (S("x264_crf33", "d1") - S("x264_crf23", "d1")).abs().to_numpy()
    ref_b = boot_median(ref, sidx)
    add(sint, "P3 reference |CRF 33 - CRF 23|", "median per-seq |diff|", np.median(ref), ref_b, zero_test=False)
    p3_est, p3_b = {}, {}
    for nm, eid, grp, base in P3ARMS:
        d = (S(eid, grp) - S(base, "d1")).abs().to_numpy()
        p3_est[nm], p3_b[nm] = np.median(d), boot_median(d, sidx)
        add(sint, f"P3 {nm} |change|", "median per-seq |diff|", p3_est[nm], p3_b[nm], zero_test=False)
        add(sint, f"P3 {nm} margin: 0.5 x ref - median |change|", "difference of medians (> 0 = P3 holds)",
            0.5 * np.median(ref) - p3_est[nm], 0.5 * ref_b - p3_b[nm])
    # P2c
    mx_est = max(p3_est.values())
    mx_b = np.max(np.vstack(list(p3_b.values())), axis=0)
    add(sint, "P2c median x264 B effect - max P3 median |change|", "difference of medians (> 0 = P2c holds)",
        np.median(dx) - mx_est, boot_median(dx, sidx) - mx_b)

    # CCTV
    clips = list(el.index)
    by_scene = [np.array([clips.index(c) for c in el.index[el == s]]) for s in scenes]
    cluster_idx = [np.concatenate([by_scene[j] for j in row]) for row in scidx]
    for key, lo_, glo, hi_, ghi in CPREDS:
        d = (V(hi_, ghi).reindex(clips) - V(lo_, glo).reindex(clips)).to_numpy()
        assert not np.isnan(d).any()
        est = np.median(d)
        add("CCTV confirmatory", key, "median per-clip diff, STALE_MOV", est,
            np.array([np.median(d[ix]) for ix in cluster_idx]),
            method=f"scene-cluster bootstrap ({len(scenes)} scenes, {len(clips)} clips)")
        add("CCTV confirmatory", key, "median per-clip diff, STALE_MOV", est, boot_median(d, clidx),
            method=f"clip-level bootstrap ({len(clips)} clips)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "r4_bootstrap_ci.csv"), index=False)
    yn = lambda b: "n/a" if b is None else ("yes" if b else "no")  # noqa: E731
    mk = lambda d: [[r.data, r.comparison, r.metric, r.method, f(r.estimate, 5), f(r.ci_low, 5), f(r.ci_high, 5),  # noqa
                     yn(r.excludes_0), f"{r.reference:g}: {'yes' if r.excludes_reference else 'no'}" if r.reference else "–"]
                    for r in d.itertuples()]
    hdr = ["data", "comparison", "metric", "method", "estimate", "CI low", "CI high", "CI excludes 0",
           "CI excludes reference (value: yes/no)"]
    scn = el.value_counts().sort_index()
    L = ["# R4: bootstrap 95% confidence intervals", "",
         "**EXPLORATORY, NOT PRE-REGISTERED.** None of these intervals was in p3/PREREGISTRATION.md or the CCTV "
         "pre-registration (d62753f); the pre-registered verdicts are counts and are unchanged.", "",
         f"Generated by `r4_bootstrap_ci.py`. Percentile intervals, {B:,} resamples, "
         f"`numpy.random.default_rng({SEED})`; draw order and resampling units are in the script docstring. "
         "Absolute-value metrics (P2b, P3 |change|) are non-negative by construction, so \"excludes 0\" is "
         "uninformative for them (shown as n/a, as is P1 Spearman, whose verdict is against 0.8); the column \"excludes reference\" compares with the verdict threshold where one "
         "exists (P1 Spearman 0.8, P2b 0.005). For P3 and P2c the margin rows carry the test (> 0 = condition holds).",
         f"Per-sequence Spearman rho is 1 in {int(np.isclose(rho, 1).sum())}/{len(rho)} sequences (minimum "
         f"{rho.min():.3f}), so the bootstrap distribution of its median is degenerate at 1.", "",
         "Sintel: 23 sequences, FINAL pass, tau = 1 px. CCTV: confirmatory test, tau = 1 px, eligible clips per "
         "scene: " + ", ".join(f"{s} {n}" for s, n in scn.items()) + ".", "",
         table(hdr, mk(df))]
    write("r4_bootstrap_ci.md", L)


if __name__ == "__main__":
    main()
