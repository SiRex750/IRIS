"""Build p3/pilot/summary.md and plots from results.csv / encode_log.json / manifest.json.

Narrative text is read from summary_narrative.md (hand-written after looking at the numbers)
so re-running this script never overwrites it.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
TAU = 1.0

# reference palette (dataviz skill, light mode), first three slots: validated all-pairs
COL = {"libx264": "#2a78d6", "h264_nvenc": "#eb6834", "mpeg4": "#1baf7a"}
NAME = {"libx264": "x264", "h264_nvenc": "NVENC", "mpeg4": "mpeg4"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

man = json.load(open(os.path.join(HERE, "manifest.json")))
SEQS = man["sequences"]
df = pd.read_csv(os.path.join(HERE, "results.csv"))
logs = json.load(open(os.path.join(HERE, "encode_log.json")))
HEAD = {"d1", "d1(assumed)"}

METRICS = [  # (column, label, fmt, tau-specific)
    ("psnr_y", "PSNR-Y dB", "{:.2f}", False),
    ("bitrate_kbps", "bitrate kb/s", "{:.0f}", False),
    ("mean_p_qp", "mean P QP", "{:.1f}", False),
    ("epe_median", "EPE median px", "{:.3f}", False),
    ("epe_gt3_area_pct", "EPE>3px % area", "{:.2f}", False),
    ("mv_coverage_pct", "MV coverage %", "{:.2f}", False),
    ("zero_mv_share", "zero-MV share", "{:.4f}", False),
    ("false_static_of_zero", "false-static / zero-MV", "{:.3f}", False),
    ("false_static_of_valid", "false-static / valid", "{:.4f}", False),
    ("flip_rate", f"flip rate @{TAU:g}", "{:.4f}", True),
    ("precision", f"reuse precision @{TAU:g}", "{:.3f}", True),
    ("recall", f"reuse recall @{TAU:g}", "{:.3f}", True),
    ("stale_of_reuse", f"stale / reused @{TAU:g}", "{:.4f}", True),
    ("stale_of_valid", f"stale / valid @{TAU:g}", "{:.5f}", True),
]


def val(eid, group, col, seq, tau=TAU):
    r = df[(df.encode_id == eid) & (df.group == group) & (df.sequence == seq) & (df.tau == tau)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def fmt(v, f):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def headline_group(eid):
    g = set(df[df.encode_id == eid].group)
    for k in ("d1", "d1(assumed)"):
        if k in g:
            return k
    return None


def factor_table(title, arms, metrics=METRICS, note=None):
    """arms: list of (label, encode_id, group)."""
    out = [f"### {title}", ""]
    if note:
        out += [note, ""]
    hdr = "| arm | metric | " + " | ".join(SEQS) + " | mean of 4 |"
    out += [hdr, "|" + "---|" * (len(SEQS) + 3)]
    for label, eid, grp in arms:
        if grp is None:
            continue
        for col, mlabel, f, _ in metrics:
            vs = [val(eid, grp, col, s) for s in SEQS]
            mean = np.nanmean(vs) if not all(np.isnan(vs)) else np.nan
            out.append(f"| {label} | {mlabel} | " + " | ".join(fmt(v, f) for v in vs) + f" | **{fmt(mean, f)}** |")
    return out + [""]


def mean4(eid, group, col, tau=TAU):
    vs = [val(eid, group, col, s, tau) for s in SEQS]
    return np.nanmean(vs) if not all(np.isnan(vs)) else np.nan


def ids(prefix, vals, fmtid):
    return [(fmtid.format(v), prefix.format(v)) for v in vals]


# ------------------------------------------------------------------ arms per factor
CRF = [12, 18, 23, 28, 33, 38, 45]
NQP = [18, 23, 28, 33, 38, 45]
MQ = [2, 4, 8, 16, 31]
enc_ok = set(df.encode_id)


def arm(label, eid, grp=None):
    if eid not in enc_ok:
        return (label + " (FAILED/missing)", eid, None)
    return (label, eid, grp or headline_group(eid))


FACTORS = {
    "CRF (x264 medium, ref 1, bf 0)": [arm(f"CRF {c}", f"x264_crf{c}") for c in CRF],
    "NVENC QP (constqp, p4, refs 1, bf 0)": [arm(f"QP {q}", f"nvenc_qp{q}") for q in NQP],
    "mpeg4 q (qmin=qmax=q, bf 0)": [arm(f"q {q}", f"mpeg4_q{q}") for q in MQ],
    "ref (x264 CRF 23 medium, bf 0; ref 2/3 read as d=1)": [
        arm("ref 1", "x264_crf23"), arm("ref 2 [d assumed 1]", "x264_crf23_ref2"),
        arm("ref 3 [d assumed 1]", "x264_crf23_ref3")],
    "preset (x264 CRF 23, ref 1, bf 0)": [
        arm("ultrafast", "x264_crf23_ultrafast"), arm("medium", "x264_crf23"),
        arm("veryslow", "x264_crf23_veryslow")],
    "keyint (x264 CRF 23 medium)": [arm("keyint 250", "x264_crf23"), arm("keyint 30", "x264_crf23_keyint30")],
}
BF_ARMS = []
for base, bf, lab in [("x264_crf23", "x264_crf23_bf2", "x264 CRF 23"), ("nvenc_qp28", "nvenc_qp28_bf2", "NVENC QP 28")]:
    BF_ARMS.append(arm(f"{lab} bf 0 — P d=1", base))
    for g, gl in [("dgt1_naive", "P d>1 naive"), ("dgt1_scaled", "P d>1 scaled"),
                  ("B_naive", "B naive"), ("B_scaled", "B scaled"), ("d1", "P d=1")]:
        if bf in enc_ok and g in set(df[df.encode_id == bf].group):
            BF_ARMS.append((f"{lab} bf 2 — {gl}", bf, g))

# encoder factor: x264 CRF 23 vs the NVENC QP / mpeg4 q arm closest in mean PSNR
x_psnr = mean4("x264_crf23", "d1", "psnr_y")


def closest(prefix, vals):
    c = [(abs(mean4(prefix.format(v), "d1", "psnr_y") - x_psnr), prefix.format(v)) for v in vals
         if prefix.format(v) in enc_ok]
    return min(c)[1] if c else None


ENC_ARMS = [arm("x264 CRF 23", "x264_crf23")]
for pre, vals, lab in [("nvenc_qp{}", NQP, "NVENC"), ("mpeg4_q{}", MQ, "mpeg4")]:
    e = closest(pre, vals)
    if e:
        ENC_ARMS.append(arm(f"{lab} {e.split('_')[1]}", e))
FACTORS["encoder (closest mean PSNR to x264 CRF 23)"] = ENC_ARMS


# ------------------------------------------------------------------ factor ranges
def factor_ranges():
    rows = []
    cols = [("epe_median", "EPE med"), ("false_static_of_zero", "false-static/zero"),
            ("stale_of_reuse", f"stale/reused@{TAU:g}"), ("precision", f"prec@{TAU:g}"),
            ("recall", f"recall@{TAU:g}"), ("flip_rate", f"flip@{TAU:g}")]
    for fname, arms in list(FACTORS.items()) + [("bframes: B naive vs bf0 d1 (x264 and NVENC arms pooled; see narrative for per-encoder)", None),
                                                ("bframes: B scaled vs bf0 d1 (x264 and NVENC arms pooled)", None)]:
        if arms is None:
            g = "B_naive" if "naive" in fname else "B_scaled"
            arms = [a for a in BF_ARMS if a[2] in ("d1", g) and ("bf 0" in a[0] or g in a[2])]
        r = {"factor": fname}
        for c, cl in cols:
            vs = [mean4(e, g, c) for _, e, g in arms if g]
            vs = [v for v in vs if not np.isnan(v)]
            r[cl] = (max(vs) - min(vs)) if len(vs) >= 2 else np.nan
        rows.append(r)
    return pd.DataFrame(rows), [cl for _, cl in cols]


def shape(series):
    s = [v for v in series if not np.isnan(v)]
    if len(s) < 3:
        return "n/a"
    d = np.sign(np.diff(s))
    if np.all(d >= 0):
        return "monotone ↑"
    if np.all(d <= 0):
        return "monotone ↓"
    i = int(np.argmax(s))
    j = int(np.argmin(s))
    if 0 < i < len(s) - 1 and np.all(d[:i] >= 0) and np.all(d[i:] <= 0):
        return f"inverted-U (peak at arm {i + 1})"
    if 0 < j < len(s) - 1 and np.all(d[:j] <= 0) and np.all(d[j:] >= 0):
        return f"U (min at arm {j + 1})"
    return "non-monotone: " + "".join("+" if x > 0 else "−" if x < 0 else "0" for x in d)


# ------------------------------------------------------------------ plots
def plots():
    os.makedirs(PLOTS, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "figure.facecolor": SURF, "axes.facecolor": SURF})
    ys = [("epe_median", "MV EPE median (px), d=1 vectors"),
          ("false_static_of_zero", "false-static ÷ zero-MV area"),
          ("stale_of_reuse", f"stale-reuse ÷ reused cells (τ={TAU:g})"),
          ("flip_rate", f"flip rate vs x264 CRF 12 (τ={TAU:g})")]
    series = [("libx264", [f"x264_crf{c}" for c in CRF]), ("h264_nvenc", [f"nvenc_qp{q}" for q in NQP]),
              ("mpeg4", [f"mpeg4_q{q}" for q in MQ])]
    arms = [("x264 CRF23 ref 2 (read as d=1)", "x264_crf23_ref2", "d1(assumed)", "D", "libx264"),
            ("x264 CRF23 ref 3 (read as d=1)", "x264_crf23_ref3", "d1(assumed)", "d", "libx264"),
            ("x264 CRF23 ultrafast", "x264_crf23_ultrafast", "d1", "s", "libx264"),
            ("x264 CRF23 veryslow", "x264_crf23_veryslow", "d1", "p", "libx264"),
            ("x264 CRF23 keyint 30", "x264_crf23_keyint30", "d1", "^", "libx264"),
            ("x264 CRF23 bf2, B vectors naive", "x264_crf23_bf2", "B_naive", "v", "libx264"),
            ("x264 CRF23 bf2, B vectors scaled", "x264_crf23_bf2", "B_scaled", "<", "libx264"),
            ("NVENC QP28 bf2, B vectors naive", "nvenc_qp28_bf2", "B_naive", "v", "h264_nvenc"),
            ("NVENC QP28 bf2, B vectors scaled", "nvenc_qp28_bf2", "B_scaled", "<", "h264_nvenc")]
    paths = []
    for xcol, xlab, logx in [("psnr_y", "PSNR-Y (dB), mean of 4 sequences", False),
                             ("bitrate_kbps", "bitrate (kb/s, log), mean of 4 sequences", True)]:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8.4))
        for ax, (ycol, ylab) in zip(axs.ravel(), ys):
            for codec, eids in series:
                pts = [(mean4(e, "d1", xcol), mean4(e, "d1", ycol), e) for e in eids if e in enc_ok]
                pts = [p for p in pts if not np.isnan(p[0]) and not np.isnan(p[1])]
                if not pts:
                    continue
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=COL[codec], lw=2, ms=6,
                        mec=SURF, mew=1.5, label=f"{NAME[codec]} rate series", zorder=3)
            for lab, e, g, mk, codec in arms:
                if e not in enc_ok:
                    continue
                x, y = mean4(e, g, xcol), mean4(e, g, ycol)
                if np.isnan(x) or np.isnan(y):
                    continue
                ax.plot([x], [y], mk, color=COL[codec], mfc=SURF, mew=1.8, ms=8, zorder=4, label=lab)
            if logx:
                ax.set_xscale("log")
            ax.set_title(ylab, fontsize=9.5, color=INK, loc="left")
            ax.set_xlabel(xlab)
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=8)
        fig.text(0.5, 0.995, "Each point = mean of the 4 sequences. Lines: rate series (d=1 vectors). "
                 "Hollow markers: single arms. Flip rate of x264 CRF 12 is 0 by construction.",
                 ha="center", va="top", fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.13, 1, 0.975))
        p = os.path.join(PLOTS, f"metrics_vs_{xcol}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(os.path.relpath(p, HERE).replace("\\", "/"))
    return paths


# ------------------------------------------------------------------ assemble
def main():
    L = ["# Paper 3 pilot: codec MVs vs Sintel GT (exploratory)", "",
         "Exploratory pilot. No claims; the numbers are reported as they came out, with no tuning or reruns.", ""]
    L += [f"- Sequences (final pass), with pixel static share: " +
          ", ".join(f"{s} ({man['static_share'][s]:.4f})" for s in SEQS),
          f"- Git {man['git_hash'][:10]} on `{man['git_branch']}`; PyAV {man['pyav']}, FFmpeg {man['ffmpeg']}, "
          f"{man['x264_build']} (the SEI carries no revision), NVIDIA {man['nvidia']}",
          f"- Start {man['start_time']}; total pilot runtime **{man.get('pilot_runtime_s', float('nan')) / 60:.1f} min**; "
          f"failures: **{man.get('n_failures', '?')}**",
          "- GT: `lib/gt_grid.py` (forward splat onto frame t's 4×4 grid); blocks and cells with gt_cover < 0.5 "
          "are excluded. All rates are area-weighted. The per-sequence gate and headline groups are defined in "
          "`run_pilot.py`'s docstring.",
          "- Tables show τ = 1 px for the gate metrics. `results.csv` has τ ∈ {0.5, 1, 2} and every vector group.", ""]
    nar = os.path.join(HERE, "summary_narrative.md")
    if os.path.exists(nar):
        L += [open(nar, encoding="utf-8").read().strip(), ""]

    L += ["## Plots", ""]
    for p in plots():
        L += [f"![{p}]({p})", ""]

    fr, cols = factor_ranges()
    L += ["## Factor effect: range (max − min) across each factor's arms, mean of the 4 sequences", "",
          "| factor | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for _, r in fr.iterrows():
        L.append(f"| {r['factor']} | " + " | ".join(fmt(r[c], "{:.4f}") for c in cols) + " |")
    L += [""]

    L += ["## Shape along the rate series (mean of 4, headline d=1 vectors, arms ordered low → high QP)", "",
          "| series | EPE median | EPE>3 % | false-static/zero | stale/reused@1 | precision@1 | recall@1 |",
          "|---|---|---|---|---|---|---|"]
    for lab, eids in [("x264 CRF", [f"x264_crf{c}" for c in CRF]), ("NVENC QP", [f"nvenc_qp{q}" for q in NQP]),
                      ("mpeg4 q", [f"mpeg4_q{q}" for q in MQ])]:
        cells = []
        for c in ("epe_median", "epe_gt3_area_pct", "false_static_of_zero", "stale_of_reuse", "precision", "recall"):
            s = [mean4(e, "d1", c) for e in eids if e in enc_ok]
            cells.append(shape(s) + " (" + ", ".join(fmt(v, "{:.3g}") for v in s) + ")")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L += ["", "Per sequence (mean-of-4 mixes static-rich and fully-moving sequences, so the shape can differ):", "",
          "| series | sequence | EPE median | false-static/zero | stale/reused@1 | precision@1 | recall@1 |",
          "|---|---|---|---|---|---|---|"]
    for lab, eids in [("x264 CRF", [f"x264_crf{c}" for c in CRF]), ("NVENC QP", [f"nvenc_qp{q}" for q in NQP]),
                      ("mpeg4 q", [f"mpeg4_q{q}" for q in MQ])]:
        for sq in SEQS:
            cells = []
            for c in ("epe_median", "false_static_of_zero", "stale_of_reuse", "precision", "recall"):
                s = [val(e, "d1", c, sq) for e in eids if e in enc_ok]
                cells.append(shape(s) + " (" + ", ".join(fmt(v, "{:.3g}") for v in s) + ")")
            L.append(f"| {lab} | {sq} | " + " | ".join(cells) + " |")
    L += [""]

    L += ["## Gate across τ: x264 CRF series, d=1, mean of 4", "",
          "| CRF | τ | flip | reused cells | precision | recall | stale/reused | stale/valid |", "|---|---|---|---|---|---|---|---|"]
    for c in CRF:
        e = f"x264_crf{c}"
        for t in (0.5, 1.0, 2.0):
            rc = sum(val(e, "d1", "reuse_cells", s, t) for s in SEQS)
            L.append(f"| {c} | {t:g} | {fmt(mean4(e, 'd1', 'flip_rate', t), '{:.4f}')} | {rc:.0f} | "
                     f"{fmt(mean4(e, 'd1', 'precision', t), '{:.3f}')} | {fmt(mean4(e, 'd1', 'recall', t), '{:.3f}')} | "
                     f"{fmt(mean4(e, 'd1', 'stale_of_reuse', t), '{:.4f}')} | {fmt(mean4(e, 'd1', 'stale_of_valid', t), '{:.5f}')} |")
    L += [""]

    L += ["## Tables per factor (τ = 1; each sequence's value and the mean of the 4)", ""]
    for title, arms in FACTORS.items():
        L += factor_table(title, arms)
    L += factor_table("bframes (x264 CRF 23 medium b-pyramid none; NVENC QP 28 b_ref_mode disabled; ref 1)",
                      BF_ARMS, note="With ref 1 and bf 2, P-frames point back to the previous I/P (d = 2–3), "
                      "and B vectors have d = 1–2 (future vectors point forward). Naive = raw vector vs 1-step GT. "
                      "Scaled = vector ÷ d, with the sign flipped for future references.")

    L += ["## Gate counts (τ = 1, headline group, summed over the 4 sequences)", "",
          "| encode | group | valid cells | reused | GT-static | TP | stale | zero-MV | false-static |",
          "|---|---|---|---|---|---|---|---|---|"]
    for e in df.encode_id.unique():
        for g in df[df.encode_id == e].group.unique():
            sub = df[(df.encode_id == e) & (df.group == g) & (df.tau == TAU)]
            L.append(f"| {e} | {g} | {sub.valid_cells.sum():.0f} | {sub.reuse_cells.sum():.0f} | "
                     f"{sub.gt_static_cells.sum():.0f} | {sub.tp_cells.sum():.0f} | {sub.stale_cells.sum():.0f} | "
                     f"{sub.zero_cells.sum():.0f} | {sub.false_static_cells.sum():.0f} |")
    L += [""]

    L += ["## Encodes: exact options, frame types, size", "",
          "| encode × sequence | status | options | types (I/P/B) | file bytes | kb/s | PSNR-Y | run s |",
          "|---|---|---|---|---|---|---|---|"]
    for lg in logs["logs"]:
        if lg.get("status") == "ok":
            L.append(f"| {lg['encode_id']} × {lg['sequence']} | ok | `{json.dumps(lg['options'])}` | "
                     f"{lg['n_I']}/{lg['n_P']}/{lg['n_B']} | {lg['file_bytes']} | {lg['bitrate_kbps']:.0f} | "
                     f"{lg['psnr_y']:.2f} | {lg['run_s']:.1f} |")
        else:
            L.append(f"| {lg['encode_id']} × {lg['sequence']} | **FAILED** | {lg.get('error')} | | | | | |")
    L += [""]
    with open(os.path.join(HERE, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("wrote summary.md")


if __name__ == "__main__":
    main()
