"""Build p3/pilot_v2/summary.md + plots from results.csv / encode_log.json / manifest.json.

No averaging across sequences anywhere. Sequences are shown per column, grouped static-heavy
(ambush_7, bandage_2) then moving (alley_1, ambush_5). Vector groups with < MIN_FRAMES frames are
kept in results.csv but left out of every table and plot. The narrative is read from
summary_narrative.md so re-running never overwrites it.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
PLOTS = os.path.join(HERE, "plots")
TAU = 1.0
MIN_FRAMES = 5
SEQS = ["ambush_7", "bandage_2", "alley_1", "ambush_5"]
SGROUP = {"ambush_7": "S", "bandage_2": "S", "alley_1": "M", "ambush_5": "M"}
COLHDR = " | ".join(f"{SGROUP[s]}: {s}" for s in SEQS)

COL = {"libx264": "#2a78d6", "h264_nvenc": "#eb6834", "mpeg4": "#1baf7a"}
NAME = {"libx264": "x264", "h264_nvenc": "NVENC", "mpeg4": "mpeg4"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

man = json.load(open(os.path.join(HERE, "manifest.json")))
df = pd.read_csv(os.path.join(HERE, "results.csv"))
df = df[df.n_group_frames >= MIN_FRAMES].copy()
logs = json.load(open(os.path.join(HERE, "encode_log.json")))
v1 = pd.read_csv(os.path.join(P3, "pilot", "results.csv"))
enc_ok = set(df.encode_id)

METRICS = [
    ("n_group_frames", "frames in group", "{:.0f}"),
    ("psnr_y", "PSNR-Y dB", "{:.2f}"),
    ("bitrate_kbps", "bitrate kb/s", "{:.0f}"),
    ("mean_p_qp", "decoded mean P QP", "{:.1f}"),
    ("mean_b_qp", "decoded mean B QP", "{:.1f}"),
    ("epe_median", "EPE median px", "{:.3f}"),
    ("epe_gt3_area_pct", "EPE>3px % area", "{:.2f}"),
    ("mv_coverage_pct", "MV coverage %", "{:.2f}"),
    ("zero_mv_share", "zero-MV share", "{:.4f}"),
    ("false_static_of_zero", "false-static / zero-MV", "{:.3f}"),
    ("stale_of_valid", f"**stale / valid @{TAU:g}**", "{:.5f}"),
    ("missed_of_valid", f"**missed reuse / valid @{TAU:g}**", "{:.5f}"),
    ("stale_of_reuse", f"stale / reused @{TAU:g} (secondary)", "{:.4f}"),
    ("flip_x264ref", f"flip vs x264 CRF 12 @{TAU:g}", "{:.4f}"),
    ("flip_ownref", f"flip vs own best @{TAU:g}", "{:.4f}"),
]


def val(eid, group, col, seq, tau=TAU, d=df):
    r = d[(d.encode_id == eid) & (d.group == group) & (d.sequence == seq) & (d.tau == tau)]
    return float(r[col].iloc[0]) if len(r) and pd.notna(r[col].iloc[0]) else np.nan


def fmt(v, f):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)


def headline(eid):
    g = set(df[df.encode_id == eid].group)
    return next((k for k in ("d1", "d1(assumed)") if k in g), None)


def arm(label, eid, grp=None):
    if eid not in enc_ok:
        return (label + " (missing)", eid, None)
    return (label, eid, grp or headline(eid))


def b_arms(label, eid):
    out = []
    for g, gl in [("B_naive", "B naive"), ("B_scaled", "B scaled"), ("dgt1_naive", "P d>1 naive"),
                  ("dgt1_scaled", "P d>1 scaled"), ("d1", "P d=1")]:
        if eid in enc_ok and g in set(df[df.encode_id == eid].group):
            out.append((f"{label} — {gl}", eid, g))
    return out


def table(title, arms, metrics=METRICS, note=None):
    L = [f"### {title}", ""]
    if note:
        L += [note, ""]
    L += [f"| arm | metric | {COLHDR} |", "|---|---|" + "---|" * len(SEQS)]
    for label, eid, grp in arms:
        if grp is None:
            continue
        for col, ml, f in metrics:
            vs = [val(eid, grp, col, s) for s in SEQS]
            if col == "mean_b_qp" and all(np.isnan(vs)):
                continue
            L.append(f"| {label} | {ml} | " + " | ".join(fmt(v, f) for v in vs) + " |")
    return L + [""]


CRF = [12, 18, 23, 28, 33, 38, 45]
NQP = [18, 23, 28, 33, 38, 45]
MQ = [2, 4, 8, 16, 31]
FACTORS = {
    "CRF (x264 medium, ref 1, bf 0, 1 thread)": [arm(f"CRF {c}", f"x264_crf{c}") for c in CRF],
    "NVENC QP (constqp, p4, refs 1, bf 0)": [arm(f"QP {q}", f"nvenc_qp{q}") for q in NQP],
    "mpeg4 q (qmin=qmax=q, bf 0, 1 thread)": [arm(f"q {q}", f"mpeg4_q{q}") for q in MQ],
    "ref (x264 CRF 23; ref 2/3 read as d=1)": [arm("ref 1", "x264_crf23"), arm("ref 2 [d assumed 1]", "x264_crf23_ref2"),
                                              arm("ref 3 [d assumed 1]", "x264_crf23_ref3")],
    "preset (x264 CRF 23)": [arm("ultrafast", "x264_crf23_ultrafast"), arm("medium", "x264_crf23"),
                             arm("veryslow", "x264_crf23_veryslow")],
    "keyint (x264 CRF 23)": [arm("keyint 250", "x264_crf23"), arm("keyint 30", "x264_crf23_keyint30")],
    "search range (x264 CRF 23 medium)": [arm("me=hex merange 16 (medium default)", "x264_crf23"),
                                          arm("me=umh merange 16", "x264_crf23_umh16"),
                                          arm("me=umh merange 32", "x264_crf23_umh32"),
                                          arm("me=umh merange 64", "x264_crf23_umh64")],
    "encoder (x264 CRF 23 / NVENC QP 28 / mpeg4 q 4 — compare PSNR row)": [
        arm("x264 CRF 23", "x264_crf23"), arm("NVENC QP 28", "nvenc_qp28"), arm("mpeg4 q 4", "mpeg4_q4")],
}
BF_X264 = [arm("bf 0", "x264_crf23")] + b_arms("bf 2 default B QP", "x264_crf23_bf2") + \
    b_arms("bf 2 pbratio 1.0", "x264_crf23_bf2_pb1")
BF_NV = [arm("bf 0", "nvenc_qp28")] + b_arms("bf 2 default B QP", "nvenc_qp28_bf2") + \
    b_arms("bf 2 B QP = P QP (b_qfactor 1, b_qoffset 0)", "nvenc_qp28_bf2_bqeq")
BF_CQP = [arm("x264 --qp 24 bf 0", "x264_qp24")] + b_arms("x264 --qp 24 bf 2 pbratio 1.0", "x264_qp24_bf2_pb1")


def factor_effect():
    cols = [("stale_of_valid", "stale/valid"), ("missed_of_valid", "missed/valid"),
            ("flip_x264ref", "flip vs CRF12"), ("epe_median", "EPE med")]
    groups = list(FACTORS.items()) + [
        ("bframes x264 CRF 23: bf0 d1 vs bf2 B naive (default / pb1)",
         [a for a in BF_X264 if a[2] in ("d1", "B_naive")]),
        ("bframes x264 CRF 23: bf0 d1 vs bf2 B scaled", [a for a in BF_X264 if a[2] in ("d1", "B_scaled")]),
        ("bframes NVENC QP 28: bf0 d1 vs bf2 B naive (default / B=P)", [a for a in BF_NV if a[2] in ("d1", "B_naive")]),
        ("bframes NVENC QP 28: bf0 d1 vs bf2 B scaled", [a for a in BF_NV if a[2] in ("d1", "B_scaled")]),
        ("[extra] x264 --qp 24: bf0 d1 vs bf2 B naive (B QP = P QP)", [a for a in BF_CQP if a[2] in ("d1", "B_naive")]),
    ]
    L = [f"## Factor effect per sequence: range (max − min) across the factor's arms, τ = {TAU:g}", "",
         "Each cell shows stale/valid · missed/valid · flip vs CRF 12 · EPE median.", "",
         f"| factor | {COLHDR} |", "|---|" + "---|" * len(SEQS)]
    for name, arms in groups:
        cells = []
        for s in SEQS:
            parts = []
            for c, _ in cols:
                vs = [val(e, g, c, s) for _, e, g in arms if g]
                vs = [v for v in vs if not np.isnan(v)]
                parts.append(fmt(max(vs) - min(vs), "{:.4f}") if len(vs) >= 2 else "–")
            cells.append(" · ".join(parts))
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    return L + [""]


def shape(series):
    s = [v for v in series if not np.isnan(v)]
    if len(s) < 3:
        return "n/a"
    d = np.sign(np.diff(s))
    if np.all(d >= 0):
        return "↑"
    if np.all(d <= 0):
        return "↓"
    i, j = int(np.argmax(s)), int(np.argmin(s))
    if 0 < i < len(s) - 1 and np.all(d[:i] >= 0) and np.all(d[i:] <= 0):
        return f"∩ (peak arm {i + 1})"
    if 0 < j < len(s) - 1 and np.all(d[:j] <= 0) and np.all(d[j:] >= 0):
        return f"∪ (min arm {j + 1})"
    return "".join("+" if x > 0 else "−" if x < 0 else "0" for x in d)


def shapes():
    L = ["## Shape along the rate series (d=1, per sequence, arms low → high QP)", "",
         "| series | metric | " + COLHDR + " |", "|---|---|" + "---|" * len(SEQS)]
    for lab, eids in [("x264 CRF", [f"x264_crf{c}" for c in CRF]), ("NVENC QP", [f"nvenc_qp{q}" for q in NQP]),
                      ("mpeg4 q", [f"mpeg4_q{q}" for q in MQ])]:
        for c, cl in [("epe_median", "EPE median"), ("false_static_of_zero", "false-static/zero"),
                      ("stale_of_valid", "stale/valid@1"), ("missed_of_valid", "missed/valid@1"),
                      ("stale_of_reuse", "stale/reused@1")]:
            cells = []
            for s in SEQS:
                v = [val(e, "d1", c, s) for e in eids]
                cells.append(shape(v) + " (" + ", ".join(fmt(x, "{:.3g}") for x in v) + ")")
            L.append(f"| {lab} | {cl} | " + " | ".join(cells) + " |")
    return L + [""]


def gate_tau():
    L = ["## Gate across τ: x264 CRF series, d=1 (stale/valid · missed/valid)", "",
         f"| CRF | τ | {COLHDR} |", "|---|---|" + "---|" * len(SEQS)]
    for c in CRF:
        for t in (0.5, 1.0, 2.0):
            L.append(f"| {c} | {t:g} | " + " | ".join(
                fmt(val(f"x264_crf{c}", "d1", "stale_of_valid", s, t), "{:.5f}") + " · " +
                fmt(val(f"x264_crf{c}", "d1", "missed_of_valid", s, t), "{:.5f}") for s in SEQS) + " |")
    return L + [""]


def speed_bins():
    bins = ["2_8", "8_16", "16_32", "32_inf"]
    L = [f"## Stale cells by GT speed (τ = {TAU:g}): counts in 2–8 / 8–16 / 16–32 / >32 px, and total stale ÷ valid",
         "", f"| encode | group | {COLHDR} |", "|---|---|" + "---|" * len(SEQS)]
    for e in df.encode_id.unique():
        for g in df[df.encode_id == e].group.unique():
            cells = []
            for s in SEQS:
                ns = [val(e, g, f"stale_cells_{b}", s) for b in bins]
                tot = val(e, g, "stale_of_valid", s)
                cells.append(" / ".join(fmt(x, "{:.0f}") for x in ns) + f" ({fmt(tot, '{:.4f}')})")
            L.append(f"| {e} | {g} | " + " | ".join(cells) + " |")
    return L + [""]


def v1v2():
    ids = [f"x264_crf{c}" for c in CRF] + ["x264_crf23_ref2", "x264_crf23_ref3", "x264_crf23_ultrafast",
                                           "x264_crf23_keyint30", "x264_crf23_veryslow"]
    cols = [("bitrate_kbps", "kb/s", "{:.0f}"), ("psnr_y", "PSNR", "{:.2f}"), ("epe_median", "EPE med", "{:.3f}"),
            ("stale_of_valid", "stale/valid", "{:.5f}"), ("stale_of_reuse", "stale/reused", "{:.4f}"),
            ("zero_mv_share", "zero-MV", "{:.3f}"), ("flip_rate", "flip vs CRF12", "{:.4f}")]
    L = ["## v1 → v2 for the x264 arms (v1: 7 slices per frame, sliced threads; v2: 1 thread, 1 slice)", "",
         "Each cell shows v1 → v2, τ = 1, headline group. The v1 bf2 arm used b-adapt=1, so it isn't comparable "
         "and is left out. In v2, v1's flip_rate corresponds to `flip_x264ref`.", "",
         f"| arm | metric | {COLHDR} |", "|---|---|" + "---|" * len(SEQS)]
    for e in ids:
        g1 = "d1(assumed)" if "ref2" in e or "ref3" in e else "d1"
        for c, cl, f in cols:
            c2 = "flip_x264ref" if c == "flip_rate" else c
            cells = [f"{fmt(val(e, g1, c, s, d=v1), f)} → {fmt(val(e, g1, c2, s), f)}" for s in SEQS]
            L.append(f"| {e} | {cl} | " + " | ".join(cells) + " |")
    return L + [""]


def plots():
    os.makedirs(PLOTS, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
    ys = [("epe_median", "EPE median px"), ("false_static_of_zero", "false-static ÷ zero-MV"),
          ("stale_of_valid", f"stale ÷ valid (τ={TAU:g})"), ("missed_of_valid", f"missed reuse ÷ valid (τ={TAU:g})"),
          ("flip_x264ref", f"flip vs x264 CRF 12 (τ={TAU:g})")]
    series = [("libx264", [f"x264_crf{c}" for c in CRF]), ("h264_nvenc", [f"nvenc_qp{q}" for q in NQP]),
              ("mpeg4", [f"mpeg4_q{q}" for q in MQ])]
    arms = [("x264 CRF23 ref 3 (read d=1)", "x264_crf23_ref3", "d1(assumed)", "d", "libx264"),
            ("x264 CRF23 ultrafast", "x264_crf23_ultrafast", "d1", "s", "libx264"),
            ("x264 CRF23 veryslow", "x264_crf23_veryslow", "d1", "p", "libx264"),
            ("x264 CRF23 umh merange 64", "x264_crf23_umh64", "d1", "h", "libx264"),
            ("x264 CRF23 bf2, B naive", "x264_crf23_bf2", "B_naive", "v", "libx264"),
            ("x264 CRF23 bf2 pbratio 1, B naive", "x264_crf23_bf2_pb1", "B_naive", "<", "libx264"),
            ("x264 --qp 24 bf0 [extra]", "x264_qp24", "d1", "X", "libx264"),
            ("x264 --qp 24 bf2 pb1, B naive [extra]", "x264_qp24_bf2_pb1", "B_naive", ">", "libx264"),
            ("NVENC QP28 bf2, B naive", "nvenc_qp28_bf2", "B_naive", "v", "h264_nvenc"),
            ("NVENC QP28 bf2 B=P QP, B naive", "nvenc_qp28_bf2_bqeq", "B_naive", "<", "h264_nvenc")]
    paths = []
    for xcol, xlab, logx in [("psnr_y", "PSNR-Y dB", False), ("bitrate_kbps", "kb/s (log)", True)]:
        fig, axs = plt.subplots(len(ys), len(SEQS), figsize=(14, 15.5), squeeze=False)
        for ci, s in enumerate(SEQS):
            for ri, (ycol, ylab) in enumerate(ys):
                ax = axs[ri, ci]
                for codec, eids in series:
                    pts = sorted((val(e, "d1", xcol, s), val(e, "d1", ycol, s)) for e in eids if e in enc_ok)
                    pts = [p for p in pts if not (np.isnan(p[0]) or np.isnan(p[1]))]
                    if pts:
                        ax.plot(*zip(*pts), "-o", color=COL[codec], lw=2, ms=4.5, mec=SURF, mew=1,
                                label=f"{NAME[codec]} rate series (d=1)", zorder=3)
                for lab, e, g, mk, codec in arms:
                    x, y = val(e, g, xcol, s), val(e, g, ycol, s)
                    if not (np.isnan(x) or np.isnan(y)):
                        ax.plot([x], [y], mk, color=COL[codec], mfc=SURF, mew=1.5, ms=7, zorder=4, label=lab)
                if logx:
                    ax.set_xscale("log")
                if ri == 0:
                    ax.set_title(f"{s}  ({'static-heavy' if SGROUP[s] == 'S' else 'moving'})", fontsize=9.5,
                                 color=INK, loc="left")
                if ci == 0:
                    ax.set_ylabel(ylab)
                if ri == len(ys) - 1:
                    ax.set_xlabel(xlab)
                ax.grid(True, color=GRID, lw=0.7)
                ax.set_axisbelow(True)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
        h, l = axs[0, 0].get_legend_handles_labels()
        seen = {}
        for hh, ll in zip(h, l):
            seen.setdefault(ll, hh)
        fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=4, frameon=False, fontsize=8)
        fig.text(0.5, 0.997, "Per sequence; no averaging. Each panel has its own y-scale. "
                 "Flip rate of x264 CRF 12 is 0 by construction.", ha="center", va="top", fontsize=8.5, color=INK2)
        fig.tight_layout(rect=(0, 0.07, 1, 0.985))
        p = os.path.join(PLOTS, f"metrics_vs_{xcol}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(os.path.relpath(p, HERE).replace("\\", "/"))
    return paths


def checks():
    L = ["## Encoder checks: threading and slices (from the bitstream) and decoded QP", "",
         "| encode | x264 SEI threads / sliced_threads (all 4 sequences) | slices per frame min–max | "
         + " | ".join(f"I/P/B QP {s}" for s in SEQS) + " |", "|---|---|---|" + "---|" * len(SEQS)]
    lg = pd.DataFrame([x for x in logs["logs"] if x.get("status") == "ok"])
    for e in lg.encode_id.unique():
        sub = lg[lg.encode_id == e].set_index("sequence")
        thr = sorted({f"{a}/{b}" for a, b in zip(sub.sei_threads, sub.sei_sliced_threads)})
        qp = []
        for s in SEQS:
            if s in sub.index:
                r = sub.loc[s]
                qp.append("/".join(fmt(float(r[k]) if r[k] is not None else np.nan, "{:.1f}")
                                   for k in ("mean_i_qp", "mean_p_qp", "mean_b_qp")))
            else:
                qp.append("FAILED")
        L.append(f"| {e} | {', '.join(thr)} | {sub.slices_min.min()}–{sub.slices_max.max()} | " + " | ".join(qp) + " |")
    return L + [""]


def main():
    L = ["# Paper 3 pilot v2: codec MVs vs Sintel GT (exploratory)", "",
         "Exploratory pilot. No claims; the numbers are reported as they came out, with no tuning or reruns. "
         "This supersedes v1 (`p3/pilot/`, where x264 ran 7 slices per frame and mpeg4 ran 16 packets per frame).",
         "",
         "- Sequences (final pass), with pixel static share: static-heavy ambush_7 (0.794), bandage_2 (0.555); "
         "moving alley_1 (0.0015), ambush_5 (0.0059)",
         f"- Git {man['git_hash'][:10]} on `{man['git_branch']}`, plus **uncommitted additive knobs in "
         f"`p3/lib/encode.py`** (the diff is in manifest.json). PyAV {man['pyav']}, FFmpeg {man['ffmpeg']}, "
         f"{man['x264_build']}, NVIDIA {man['nvidia']}",
         f"- Start {man['start_time']}; pilot runtime **{man.get('pilot_runtime_s', float('nan')) / 60:.1f} min**; "
         f"failures: **{man.get('n_failures', '?')}**",
         f"- Vector groups with fewer than {MIN_FRAMES} frames are in results.csv and blocks/ but not in these tables. "
         "The gate headlines are stale ÷ valid and missed reuse ÷ valid; stale ÷ reused is secondary. "
         f"Tables use τ = {TAU:g}; results.csv has τ ∈ {{0.5, 1, 2}}.", ""]
    nar = os.path.join(HERE, "summary_narrative.md")
    if os.path.exists(nar):
        L += [open(nar, encoding="utf-8").read().strip(), ""]
    L += checks()
    L += v1v2()
    L += ["## Plots", ""] + [f"![{p}]({p})\n" for p in plots()]
    L += factor_effect()
    L += shapes()
    L += gate_tau()
    L += ["## Tables per factor (τ = 1)", ""]
    for t, a in FACTORS.items():
        L += table(t, a)
    note = ("ref 1, b-pyramid none, fixed pattern. P-frames point back d = 3 to the previous I/P; "
            "B vectors have d = 1–2. Naive = raw vector vs 1-step GT. Scaled = vector ÷ d, with the sign "
            "flipped for future references.")
    L += table("bframes, x264 CRF 23 (b-adapt 0)", BF_X264, note=note)
    L += table("bframes, NVENC QP 28 (b_ref_mode disabled)", BF_NV, note=note)
    L += table("[extra arm, not in the v2 prompt] bframes at matched QP, x264 constant QP 24 (pbratio 1.0)",
               BF_CQP, note="Added because x264 CRF could not hold B QP = P QP, even with pbratio 1.0 (see QP checks).")
    L += speed_bins()
    L += ["## Encodes: exact options", "", "| encode × sequence | status | options | types I/P/B | kb/s | PSNR | run s |",
          "|---|---|---|---|---|---|---|"]
    for lg in logs["logs"]:
        if lg.get("status") == "ok":
            L.append(f"| {lg['encode_id']} × {lg['sequence']} | ok | `{json.dumps(lg['options'])}` | "
                     f"{lg['n_I']}/{lg['n_P']}/{lg['n_B']} | {lg['bitrate_kbps']:.0f} | {lg['psnr_y']:.2f} | {lg['run_s']:.1f} |")
        else:
            L.append(f"| {lg['encode_id']} × {lg['sequence']} | **FAILED** | {lg.get('error')} | | | | |")
    open(os.path.join(HERE, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote summary.md")


if __name__ == "__main__":
    main()
