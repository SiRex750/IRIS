"""POST HOC (not pre-registered; changes no verdict). Matched-reuse tables and figure from per_unit_posthoc.csv
(PLAN.md §5, §8). Run via `python posthoc.py --analyze`."""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BANNER = "POST HOC (not pre-registered; changes no verdict). Plan: p3/codecsight_sim/posthoc_matched/PLAN.md (e311c83)."
FPS = ("derived", "sq16", "sq32", "sq64")
TAUS = (0.25, 0.5, 1.0)
FP_LAB = {"derived": "d", "sq16": "16", "sq32": "32", "sq64": "64", "cell": "cell"}
FIG_ARMS = ("x264_crf12", "x264_crf45", "nvenc_qp18", "nvenc_qp45")


def interp_at(x0, xs, ys):
    """Piecewise-linear STALE at REUSE x0 through (xs, ys); equal xs merged by mean ys; no extrapolation -> NaN."""
    d = pd.DataFrame({"x": xs, "y": ys}).dropna().groupby("x", as_index=False)["y"].mean().sort_values("x")
    if not len(d) or np.isnan(x0) or x0 < d.x.iloc[0] or x0 > d.x.iloc[-1]:
        return np.nan
    return float(np.interp(x0, d.x.to_numpy(), d.y.to_numpy()))


def matched(df):
    rows = []
    for (data, arm, unit), u in df.groupby(["data", "arm", "unit"]):
        g1 = u[u.method == "G1"]
        for fp in FPS:
            for tau in TAUS:
                p = u[(u.method == "P") & (u.footprint == fp) & (u.tau == tau)].iloc[0]
                g2 = u[(u.method == "G2") & (u.tau == tau)]
                rows.append({"data": data, "arm": arm, "unit": unit, "footprint": fp, "tau": tau,
                             "P_REUSE": p.REUSE_c, "P_STALE": p.STALE_c,
                             "G2_REUSE_min": g2.REUSE_c.min(), "G2_REUSE_max": g2.REUSE_c.max(),
                             "G1_REUSE_min": g1.REUSE_c.min(), "G1_REUSE_max": g1.REUSE_c.max(),
                             "D_G2": p.STALE_c - interp_at(p.REUSE_c, g2.REUSE_c, g2.STALE_c),
                             "D_G1": p.STALE_c - interp_at(p.REUSE_c, g1.REUSE_c, g1.STALE_c)})
    return pd.DataFrame(rows)


def summarise(m):
    out = []
    for (data, arm, fp, tau), g in m.groupby(["data", "arm", "footprint", "tau"], sort=False):
        r = {"data": data, "arm": arm, "P config": f"{FP_LAB[fp]} / {tau:g}", "units": len(g),
             "P med REUSE_c": g.P_REUSE.median(), "P med STALE_c": g.P_STALE.median()}
        for k in ("G2", "G1"):
            D = g[f"D_{k}"].dropna()
            r.update({f"{k} in range": len(D), f"{k} med D": D.median() if len(D) else np.nan,
                      f"{k} P lower": int((D < 0).sum()), f"{k} D=0": int((D == 0).sum()),
                      f"{k} out of range": len(g) - len(D)})
        out.append(r)
    return pd.DataFrame(out)


def md_table(d, fmt="{:.5f}"):
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if np.isnan(v) else fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def arm_order(df, data):
    import simulate as S
    return list(S.SINTEL_ARMS if data == "Sintel" else S.CCTV_ARMS)


def figure(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"G1": "#2a78d6", "G2": "#eb6834", "P": "#1baf7a"}
    mk = {0.25: "o", 0.5: "s", 1.0: "^"}
    c = df[df.data == "CCTV"]
    med = c.groupby(["arm", "method", "footprint", "tau"], as_index=False)[["REUSE_c", "STALE_c"]].median()
    fig, axs = plt.subplots(2, 2, figsize=(12, 9.5))
    for ax, arm in zip(axs.ravel(), FIG_ARMS):
        a = med[med.arm == arm]
        for meth in ("G1", "G2", "P"):
            groups = [("cell", a[a.method == "G1"].sort_values("tau"))] if meth == "G1" else \
                [(tau, a[(a.method == meth) & (a.tau == tau)].set_index("footprint").loc[list(FPS)].reset_index())
                 for tau in TAUS]
            for key, g in groups:
                ax.plot(g.REUSE_c, g.STALE_c, "-", color=col[meth], lw=1, alpha=0.5, zorder=1)
                for _, r in g.iterrows():
                    ax.scatter(r.REUSE_c, r.STALE_c, s=46, marker=mk[r.tau], color=col[meth],
                               edgecolor="white", linewidth=1.2, zorder=3)
                    lab = f"τ{r.tau:g}" if meth == "G1" else f"{FP_LAB[r.footprint]}/{r.tau:g}"
                    ax.annotate(lab, (r.REUSE_c, r.STALE_c), xytext=(4, 3), textcoords="offset points",
                                fontsize=7, color="#444444")
        ax.set_title(arm, fontsize=11)
        ax.grid(True, color="#e6e6e6", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axs.ravel():
        ax.set_xlabel("REUSE_c (median over 27 clips)")
    for ax in axs.ravel():
        ax.set_ylabel("STALE_c (median over 27 clips)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="-", color=col[m], label=l) for m, l in
         (("G1", "G1 per-cell gate"), ("G2", "G2 token gate"), ("P", "P policy (union since refresh)"))]
    h += [Line2D([], [], marker=mk[t], ls="", color="#666666", label=f"τ = {t:g} px") for t in TAUS]
    fig.legend(handles=h, loc="upper center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("POST HOC (not pre-registered) — CCTV: STALE_c vs REUSE_c, common truth; labels = token size / τ "
                 "(d = derived)", fontsize=11, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)


def main():
    df = pd.read_csv(os.path.join(HERE, "per_unit_posthoc.csv"), comment="#")
    m = matched(df)
    s = summarise(m)
    L = [f"# {BANNER}", "", "# Matched-reuse comparison: P vs G2 and G1, common truth (cell level)", "",
         "D = STALE_c(P) − STALE_c(gate) interpolated at P's REUSE_c, per unit. 'P lower' = units with D < 0. "
         "Out of range = P's REUSE_c outside the gate's REUSE_c range for that unit (no extrapolation). "
         "G2 is interpolated over token sizes at P's τ; G1 over τ 0.25 / 0.5 / 1.", ""]
    keep = ["arm", "units", "P med REUSE_c", "P med STALE_c", "G2 in range", "G2 med D", "G2 P lower", "G2 D=0",
            "G2 out of range", "G1 in range", "G1 med D", "G1 P lower", "G1 D=0", "G1 out of range"]
    for data in ("CCTV", "Sintel"):
        sd = s[s.data == data].copy()
        sd["o"] = sd.arm.map({a: i for i, a in enumerate(arm_order(df, data))})
        prim = sd[sd["P config"] == "d / 0.5"].sort_values("o")
        L += [f"## {data}, P at derived / τ 0.5", "", md_table(prim[keep]), ""]
        g1n = int(prim["G1 in range"].sum())
        L += [f"G1 comparison: P's REUSE_c is inside G1's range for {g1n} of {int(prim.units.sum())} "
              f"{data} encodes at this configuration.", ""]
    L += ["## All 12 P configurations (token size / τ)", ""]
    for data in ("CCTV", "Sintel"):
        sd = s[s.data == data].copy()
        sd["o"] = sd.arm.map({a: i for i, a in enumerate(arm_order(df, data))})
        sd = sd.sort_values(["o", "P config"])
        L += [f"### {data}", "", md_table(sd[["arm", "P config"] + keep[1:]]), ""]
        tot = sd.groupby("P config")[["G2 in range", "G2 P lower", "G2 D=0", "G1 in range", "G1 P lower"]].sum()
        L += [f"#### {data}, summed over arms", "", md_table(tot.reset_index(), "{:.0f}"), ""]
    L += ["## Per-arm medians over units (REUSE_c / STALE_c), every method and configuration", ""]
    med = df.groupby(["data", "arm", "method", "footprint", "tau"], as_index=False)[["REUSE_c", "STALE_c"]].median()
    for data in ("CCTV", "Sintel"):
        arms = arm_order(df, data)
        for met in ("REUSE_c", "STALE_c"):
            d = med[med.data == data].copy()
            d["config"] = d.method + " " + d.footprint.map(FP_LAB) + " / " + d.tau.map("{:g}".format)
            w = d.pivot(index="config", columns="arm", values=met)[arms].reset_index()
            L += [f"### {data}, median {met}", "", md_table(w, "{:.4f}"), ""]
    open(os.path.join(HERE, "matched.md"), "w", encoding="utf-8").write("\n".join(L))
    with open(os.path.join(HERE, "matched_per_unit.csv"), "w", encoding="utf-8", newline="") as f:
        f.write(f"# {BANNER}\n")
        m.to_csv(f, index=False)
    figure(df, os.path.join(HERE, "fig_matched.png"))
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
