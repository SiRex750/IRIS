"""CodecSight policy simulation: metrics, S1x-K4 verdicts, KILL-A, per-arm medians and bootstrap CIs, exactly as
fixed by p3/codecsight_sim/PREREGISTRATION.md (commit 9f5b5a3) §4 and §6. Reads per_unit.csv (simulate.py) only.

Verdicts are taken at the primary configuration (derived footprint, tau 0.5 px, native rate); the same rules at
tau 1 px are secondary. Counting rule (§8): "x% of N" = at least ceil(x N); a missing value does not meet the
condition and is listed. Metric per dataset: Sintel STALE, CCTV STALE_MOV, over all scored frames.
Bootstrap (§4, as p3/review_fixes3/r4_bootstrap_ci.py): percentile 2.5 / 97.5 (numpy.percentile, linear), 10,000
resamples, one numpy.random.default_rng(20261006) drawn in the order Sintel (10000, 23), CCTV scenes (10000, 7),
CCTV clips (10000, 27); the same draws for every comparison and both configurations within a dataset.

Usage: python analyze.py [--input per_unit.csv] [--outdir .]
Writes verdicts.md, verdicts.csv, arm_medians.csv, bootstrap_ci.csv.
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from simulate import provenance, SINTEL_ARMS, CCTV_ARMS  # noqa: E402

SEED, B = 20261006, 10_000
CRFS = (12, 18, 23, 28, 33, 38, 45)
# key, data, high arm, low arm, margin, required count (of N)
PAIRS = (("S1x", "Sintel", "x264_crf45", "x264_crf12", 0.001, 18),
         ("S1n", "Sintel", "nvenc_qp45", "nvenc_qp18", 0.001, 18),
         ("S2x", "Sintel", "x264_qp24_bf2_pb1", "x264_qp24", 0.001, 18),
         ("S2n", "Sintel", "nvenc_qp28_bf2_bqeq", "nvenc_qp28", 0.001, 16),
         ("K1", "CCTV", "x264_crf45", "x264_crf12", 0.01, math.ceil(0.80 * 27)),
         ("K2", "CCTV", "nvenc_qp45", "nvenc_qp18", 0.01, math.ceil(0.75 * 27)),
         ("K3", "CCTV", "x264_qp24_bf2_pb1", "x264_qp24", 0.01, math.ceil(0.80 * 27)),
         ("K4", "CCTV", "nvenc_qp28_bf2_bqeq", "nvenc_qp28", 0.01, math.ceil(0.75 * 27)))
S1R = ("S1r", "Sintel", 0.8, 16)
METRIC = {"Sintel": "STALE", "CCTV": "STALE_MOV"}
VERDICT_CONFIGS = (("primary", "verdict"), ("tau1", "secondary"))
KILLA_STALE, KILLA_REUSE = 0.01, 0.20


def f(v, n=5):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{n}f}"


def table(hdr, rows):
    L = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    L += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(L)


def wide(df, data, cfg, col):
    """unit x arm table of one column."""
    d = df[(df["data"] == data) & (df["config"] == cfg)]
    return d.pivot(index="unit", columns="arm", values=col).sort_index()


def spearman_rows(W):
    x = pd.Series(CRFS, dtype=float)
    return W[[f"x264_crf{c}" for c in CRFS]].apply(
        lambda r: x.corr(pd.Series(r.to_numpy(dtype=float)), method="spearman"), axis=1)


def verdicts(df, cfg):
    out = []
    for key, data, hi, lo, margin, req in PAIRS:
        W = wide(df, data, cfg, METRIC[data])
        d = W[hi] - W[lo]
        meets = (d > margin).fillna(False)
        out.append({"config": cfg, "key": key, "data": data, "metric": METRIC[data],
                    "comparison": f"{METRIC[data]}({hi}) - {METRIC[data]}({lo})", "margin": f"> {margin:g}",
                    "count": int(meets.sum()), "N": len(d), "required": req, "verdict": "PASS" if meets.sum() >= req
                    else "FAIL", "missing": ";".join(d.index[d.isna()]), "_d": d})
    key, data, thr, req = S1R
    rho = spearman_rows(wide(df, data, cfg, "STALE"))
    meets = (rho >= thr).fillna(False)
    out.insert(2, {"config": cfg, "key": key, "data": data, "metric": "STALE",
                   "comparison": "Spearman(CRF, STALE), 7 x264 CRF levels", "margin": f">= {thr:g}",
                   "count": int(meets.sum()), "N": len(rho), "required": req,
                   "verdict": "PASS" if meets.sum() >= req else "FAIL", "missing": ";".join(rho.index[rho.isna()]),
                   "_d": rho})
    return out


def kill_rules(V):
    p = {r["key"]: r["verdict"] == "PASS" for r in V}
    return [("S1x and S1n both fail", not p["S1x"] and not p["S1n"]),
            ("exactly one of S1x, S1n fails", p["S1x"] != p["S1n"]),
            ("S2x fails", not p["S2x"]), ("S2n fails", not p["S2n"]),
            ("K1 and K2 both fail", not p["K1"] and not p["K2"]),
            ("K3 fails", not p["K3"]), ("K4 fails", not p["K4"])]


def arm_medians(df):
    rows = []
    cols = ["REUSE", "STALE", "STALE_MOV", "B_REUSE", "B_STALE", "B_STALE_MOV", "g_REUSE", "g_STALE", "g_STALE_MOV",
            "active_share_pre_refresh"]
    for (data, cfg, arm), g in df.groupby(["data", "config", "arm"], sort=False):
        r = {"data": data, "config": cfg, "arm": arm, "n_units": len(g)}
        for c in cols:
            if c in g:
                v = g[c].astype(float)
                r[f"median_{c}"] = float(v.median()) if v.notna().any() else np.nan
                r[f"missing_{c}"] = int(v.isna().sum())
        V, D = g["valid_tf"].sum(), g["dropped_finite_tf"].sum()
        r["dropped_finite_share"] = D / (V + D) if V + D else np.nan
        r["valid_tf_total"], r["moved_tf_total"] = int(V), int(g["moved_tf"].sum())
        rows.append(r)
    return pd.DataFrame(rows)


def kill_a(M, cfg):
    rows = []
    for data, arms in (("Sintel", SINTEL_ARMS), ("CCTV", CCTV_ARMS)):
        for arm in arms:
            r = M[(M["data"] == data) & (M["config"] == cfg) & (M["arm"] == arm)].iloc[0]
            sm, ru = r["median_STALE_MOV"], r["median_REUSE"]
            ok_s = bool(sm < KILLA_STALE) if not np.isnan(sm) else False
            ok_r = bool(ru > KILLA_REUSE) if not np.isnan(ru) else False
            rows.append({"config": cfg, "data": data, "arm": arm, "median_REUSE": ru, "median_STALE_MOV": sm,
                         "missing_STALE_MOV": r["missing_STALE_MOV"], "g_median_REUSE": r["median_g_REUSE"],
                         "g_median_STALE_MOV": r["median_g_STALE_MOV"], "stale_mov_lt_0.01": ok_s,
                         "reuse_gt_0.20": ok_r, "absorbs": ok_s and ok_r,
                         "note": "" if ok_r else "policy reuses little"})
    return pd.DataFrame(rows)


def bootstrap(df, Vs):
    rng = np.random.default_rng(SEED)
    seqs = sorted(df.loc[df["data"] == "Sintel", "unit"].unique())
    sidx = rng.integers(0, len(seqs), size=(B, len(seqs)))
    el = df[df["data"] == "CCTV"].drop_duplicates("unit").set_index("unit")["scene"].sort_index()
    scenes = sorted(el.unique())
    scidx = rng.integers(0, len(scenes), size=(B, len(scenes)))
    clips = list(el.index)
    clidx = rng.integers(0, len(clips), size=(B, len(clips)))
    by_scene = [np.array([clips.index(c) for c in el.index[el == s]]) for s in scenes]
    cluster_idx = [np.concatenate([by_scene[j] for j in row]) for row in scidx]
    rows = []
    for cfg, V in Vs.items():
        for v in V:
            d = v["_d"]
            if v["data"] == "Sintel":
                x = d.reindex(seqs).to_numpy(dtype=float)
                meths = (("sequence bootstrap (23)", x[sidx]),)
            else:
                x = d.reindex(clips).to_numpy(dtype=float)
                meths = ((f"scene-cluster bootstrap ({len(scenes)} scenes, {len(clips)} clips)",
                          [x[ix] for ix in cluster_idx]),
                         (f"clip-level bootstrap ({len(clips)} clips)", x[clidx]))
            nmiss = int(np.isnan(x).sum())
            med = np.nanmedian if nmiss else np.median
            est = float(med(x))
            for meth, samples in meths:
                stats = (med(samples, axis=1) if isinstance(samples, np.ndarray)
                         else np.array([med(s) for s in samples]))
                lo, hi = np.percentile(stats, [2.5, 97.5])
                r = {"config": cfg, "key": v["key"], "data": v["data"], "method": meth,
                     "statistic": "median per-seq rho" if v["key"] == "S1r" else
                     f"median per-unit paired diff, {v['metric']}", "estimate": est, "ci_low": float(lo),
                     "ci_high": float(hi), "n_missing": nmiss, "degenerate": bool(lo == hi), "n_rho_eq_1": ""}
                if v["key"] == "S1r":
                    r["n_rho_eq_1"] = int(np.isclose(x, 1).sum())
                rows.append(r)
    return pd.DataFrame(rows)


def per_scene(df, cfg):
    el = df[df["data"] == "CCTV"].drop_duplicates("unit").set_index("unit")["scene"]
    rows = []
    for key, data, hi, lo, margin, req in PAIRS:
        if data != "CCTV":
            continue
        W = wide(df, data, cfg, "STALE_MOV")
        d = W[hi] - W[lo]
        for sc in sorted(el.unique()):
            ds = d[el.reindex(d.index) == sc]
            rows.append([key, sc, len(ds), int((ds > margin).sum()), f(ds.median())])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(HERE, "per_unit.csv"))
    ap.add_argument("--outdir", default=HERE)
    a = ap.parse_args()
    df = pd.read_csv(a.input)
    pv = provenance()
    data_hash = sorted(df["code_sha256"].unique())
    if data_hash != [pv["code_sha256"]]:
        print(f"WARNING: per_unit.csv code hash {data_hash} != current code {pv['code_sha256']}")
    stamp = (f"prereg {pv['prereg_commit']} (file unchanged: {pv['prereg_unchanged']}); "
             f"code sha256 {pv['code_sha256']}; per_unit.csv code sha256 {', '.join(data_hash)}")

    Vs = {cfg: verdicts(df, cfg) for cfg, _ in VERDICT_CONFIGS}
    M = arm_medians(df)
    KA = {cfg: kill_a(M, cfg) for cfg, _ in VERDICT_CONFIGS}
    BS = bootstrap(df, Vs)

    vc = pd.DataFrame([{k: v for k, v in r.items() if k != "_d"} for V in Vs.values() for r in V])
    vc["role"] = vc["config"].map(dict(VERDICT_CONFIGS))
    ka_rows = []
    for cfg, role in VERDICT_CONFIGS:
        fired = bool(KA[cfg]["absorbs"].all())
        ka_rows.append({"config": cfg, "key": "KILL-A", "data": "both", "metric": "median STALE_MOV, median REUSE",
                        "comparison": "every arm: median STALE_MOV < 0.01 and median REUSE > 0.20",
                        "margin": "see §6", "count": int(KA[cfg]["absorbs"].sum()), "N": len(KA[cfg]),
                        "required": len(KA[cfg]), "verdict": "FIRES" if fired else "DOES NOT FIRE",
                        "missing": "", "role": role})
    vc = pd.concat([vc, pd.DataFrame(ka_rows)], ignore_index=True)
    for k, v in pv.items():
        vc[k] = v
    vc.to_csv(os.path.join(a.outdir, "verdicts.csv"), index=False)
    for k, v in pv.items():
        M[k] = v
        BS[k] = v
    M.to_csv(os.path.join(a.outdir, "arm_medians.csv"), index=False)
    BS.to_csv(os.path.join(a.outdir, "bootstrap_ci.csv"), index=False)

    # ---------------------------------------------------------------- verdicts.md
    n_rows = df.groupby(["data", "config"]).size()
    L = ["# CodecSight policy simulation: verdicts", "",
         f"Provenance: {stamp}. Generated by `analyze.py` from `per_unit.csv` ({len(df)} rows).",
         "Pre-registration: `PREREGISTRATION.md` (9f5b5a3). Verdicts at the primary configuration (derived token "
         "footprint, tau 0.5 px, native rate); tau 1 px is secondary. Metric: Sintel STALE, CCTV STALE_MOV, all scored "
         "frames. A missing value does not meet the condition.", "",
         "Rows per data x configuration: " + ", ".join(f"{d} {c} {n}" for (d, c), n in n_rows.items()), ""]
    for cfg, role in VERDICT_CONFIGS:
        V = Vs[cfg]
        L += [f"## Verdicts, {cfg} ({role})", "",
              table(["key", "comparison", "margin", "count", "required", "result", "missing units"],
                    [[r["key"], r["comparison"], r["margin"], f"{r['count']}/{r['N']}", f">= {r['required']}",
                      r["verdict"], r["missing"] or "none"] for r in V]), "",
              "Kill rules (mechanical): " + "; ".join(f"{nm}: {'yes' if t else 'no'}" for nm, t in kill_rules(V)), ""]
        ka = KA[cfg]
        L += [f"### KILL-A, {cfg} ({role}): {'FIRES' if ka['absorbs'].all() else 'DOES NOT FIRE'} "
              f"({int(ka['absorbs'].sum())}/{len(ka)} arm-medians meet both conditions)", "",
              table(["data", "arm", "policy median REUSE", "policy median STALE_MOV", "STALE_MOV missing units",
                     "bare gate median REUSE", "bare gate median STALE_MOV", "STALE_MOV < 0.01", "REUSE > 0.20",
                     "note"],
                    [[r.data, r.arm, f(r.median_REUSE), f(r.median_STALE_MOV), r.missing_STALE_MOV,
                      f(r.g_median_REUSE), f(r.g_median_STALE_MOV), "yes" if r["stale_mov_lt_0.01"] else "no",
                      "yes" if r["reuse_gt_0.20"] else "no", r.note] for _, r in ka.iterrows()]), ""]
        bs = BS[BS["config"] == cfg]
        L += [f"### Paired medians and 95% bootstrap CIs, {cfg} (descriptive)", "",
              table(["key", "method", "statistic", "estimate", "CI low", "CI high", "note"],
                    [[r.key, r.method, r.statistic, f(r.estimate), f(r.ci_low), f(r.ci_high),
                      (f"CI degenerate; rho = 1 in {r.n_rho_eq_1}/23 sequences" if r.key == "S1r" and r.degenerate
                       else "") + (f" {r.n_missing} missing units excluded" if r.n_missing else "")]
                     for r in bs.itertuples()]), ""]
        L += [f"### CCTV per scene, {cfg} (descriptive): clips meeting the margin, median paired diff", "",
              table(["key", "scene", "clips", "meeting > 0.01", "median diff"], per_scene(df, cfg)), ""]
    # per-arm medians, all configurations
    L += ["## Per-arm medians over units, all configurations", "",
          "Policy REUSE / STALE / STALE_MOV; bare per-cell gate (native configurations only); B-frames only (bf 2 "
          "arms, descriptive); share of tokens active at the frame before each refresh; share of token-frames lost to "
          "the finite-terms rule (dropped / (valid + dropped), pooled).", ""]
    hdr = ["data", "config", "arm", "REUSE", "STALE", "STALE_MOV", "gate REUSE", "gate STALE", "gate STALE_MOV",
           "B REUSE", "B STALE", "B STALE_MOV", "active pre-refresh", "finite-rule dropped", "STALE_MOV missing"]
    order = {c: i for i, c in enumerate(["primary", "tau1", "tau0.25", "sq16", "sq32", "sq64", "2fps_a", "2fps_b"])}
    Ms = M.assign(_o=M["config"].map(order)).sort_values(["data", "_o"], ascending=[False, True], kind="stable")
    L.append(table(hdr, [[r.data, r.config, r.arm, f(r.median_REUSE), f(r.median_STALE), f(r.median_STALE_MOV),
                          f(r.median_g_REUSE), f(r.median_g_STALE), f(r.median_g_STALE_MOV),
                          f(r.median_B_REUSE), f(r.median_B_STALE), f(r.median_B_STALE_MOV),
                          f(r.median_active_share_pre_refresh), f(r.dropped_finite_share), r.missing_STALE_MOV]
                         for r in Ms.itertuples()]))
    open(os.path.join(a.outdir, "verdicts.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
