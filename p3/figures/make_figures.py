"""Paper 3 figures B1-B3, from saved data only. Outputs PDF + PNG in p3/figures/.

No encoding, MV extraction, RAFT or new metric is run here. Plotted values come from
  - p3/sweep/results.csv          (Sintel, FINAL pass, tau = 1 px, stale/valid)       -> B1
  - p3/virat_confirm/results.csv  (CCTV confirmatory test, tau = 1 px, STALE_MOV)     -> B2
B3 is illustrative: it redraws one frame's cells from the saved blocks of the main sweep
(p3/sweep/blocks/final/<encode>__<seq>.parquet) and the Sintel GT, with the cell rules of
p3/pilot_v2/run_pilot_v2.run_one, and first checks that the rebuilt per-sequence counts equal results.csv.

Before plotting, the script asserts that the medians it plots equal the values the paper quotes.

B3 frame rule (fixed before looking at any picture):
  sequence = the sequence whose x264 CRF 45 stale/valid (FINAL, d1, tau 1) is the median of the 23
             (23 is odd, so this is one sequence);
  frame    = among that sequence's d1 P-frames at x264 CRF 45, the frame whose stale-cell count is the
             median (lower median if the count of frames is even; ties -> earliest frame).
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, P3)
sys.path.insert(0, os.path.join(P3, "pilot"))

TAU = 1.0
STATIC_GT, STALE_GT, COVER_MIN, C = 0.5, 2.0, 0.5, 4  # as in p3/pilot_v2/run_pilot_v2.py

# ---------------------------------------------------------------- style
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dc"
SCENE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]  # validated, light
SCENE_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
STALE_C, REUSE_C = "#eb6834", "#2a78d6"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 9, "legend.frameon": False,
                     "pdf.fonttype": 42})


def pct(v):
    """Percent as the paper prints it: 2 decimals below 1%, else 1 decimal."""
    return f"{100 * v:.2f}%" if v < 0.01 else f"{100 * v:.1f}%"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- data
R = pd.read_csv(os.path.join(P3, "sweep", "results.csv"))
VR = pd.read_csv(os.path.join(P3, "virat_confirm", "results.csv"))
SEQS = sorted(R.sequence.unique())
assert len(SEQS) == 23, SEQS


def S(eid, group, col="stale_of_valid"):
    d = R[(R["pass"] == "final") & (R.encode_id == eid) & (R.group == group) & np.isclose(R.tau, TAU)]
    s = d.set_index("sequence")[col].reindex(SEQS)
    assert s.notna().all() and len(d) == 23, (eid, group)
    return s


EL = VR[VR.eligible.astype(bool)][["clip", "scene"]].drop_duplicates().set_index("clip")["scene"].sort_index()
assert len(EL) == 27 and EL.nunique() == 7, (len(EL), EL.nunique())


def V(arm):
    d = VR[(VR.arm == arm) & (VR.group == "d1") & np.isclose(VR.tau, TAU)].set_index("clip")["stale_mov"]
    v = d.reindex(EL.index)
    assert v.notna().all(), arm
    return v


# ---------------------------------------------------------------- assertions against the paper's quoted values
B1_PANELS = [("x264, QP 24", ("x264_qp24", "d1", "bf 0"), ("x264_qp24_bf2_pb1", "B_naive", "bf 2 (B)")),
             ("NVENC, QP 28", ("nvenc_qp28", "d1", "bf 0"), ("nvenc_qp28_bf2_bqeq", "B_naive", "B = P"))]
B2_PANELS = [("x264", [("x264_crf12", "CRF 12"), ("x264_crf45", "CRF 45")]),
             ("NVENC", [("nvenc_qp18", "QP 18"), ("nvenc_qp23", "QP 23"), ("nvenc_qp45", "QP 45")])]
QUOTED = [  # (label, series, quoted % as printed in the paper)
    ("Sintel x264 QP 24 bf 0", S("x264_qp24", "d1"), "0.55"),
    ("Sintel x264 QP 24 B (naive)", S("x264_qp24_bf2_pb1", "B_naive"), "4.4"),
    ("Sintel NVENC QP 28 bf 0", S("nvenc_qp28", "d1"), "3.3"),
    ("Sintel NVENC QP 28 B=P (naive)", S("nvenc_qp28_bf2_bqeq", "B_naive"), "7.0"),
    ("CCTV x264 CRF 12", V("x264_crf12"), "5.5"),
    ("CCTV x264 CRF 45", V("x264_crf45"), "24.9"),
    ("CCTV NVENC QP 18", V("nvenc_qp18"), "4.9"),
    ("CCTV NVENC QP 45", V("nvenc_qp45"), "10.1"),
]
print("Assertions: plotted medians vs the paper's quoted values")
for lab, s, q in QUOTED:
    nd = len(q.split(".")[1])
    got = f"{100 * s.median():.{nd}f}"
    ok = got == q
    print(f"  {'OK  ' if ok else 'FAIL'} {lab:32s} n={len(s):2d}  median {100 * s.median():8.4f}%  -> {got}%  "
          f"(paper {q}%)")
    assert ok, (lab, got, q)
print("  all 8 quoted medians match")

# ---------------------------------------------------------------- B1
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
allv = np.concatenate([S(e, g).values for _, *ps in B1_PANELS for e, g, _ in ps])
use_log = allv.max() / allv[allv > 0].min() > 50
for ax, (title, lo, hi) in zip(axes, B1_PANELS):
    a, b = S(lo[0], lo[1]), S(hi[0], hi[1])
    for seq in SEQS:
        ax.plot([0, 1], [a[seq], b[seq]], color="#a8a79f", lw=0.8, marker="o", ms=2.5, zorder=1)
    ma, mb = a.median(), b.median()
    ax.plot([0, 1], [ma, mb], color=INK, lw=2, marker="o", ms=6, zorder=3,
            markeredgecolor="white", markeredgewidth=1)
    ax.annotate(f"median {pct(ma)}", (0, ma), xytext=(-8, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7.5)
    ax.annotate(f"median {pct(mb)}", (1, mb), xytext=(8, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7.5)
    up = int((b > a).sum())
    ax.set_title(f"{title}  ({up}/23 sequences up)")
    ax.set_xticks([0, 1], [lo[2], hi[2]])
    ax.set_xlim(-0.95, 1.85)
    if use_log:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("stale cells / valid cells (tau = 1 px)")
fig.tight_layout()
save(fig, "fig_b1_bframes_matched_qp")

# ---------------------------------------------------------------- B2
scenes = sorted(EL.unique())
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), sharey=True, gridspec_kw={"width_ratios": [2, 3]})
for ax, (title, arms) in zip(axes, B2_PANELS):
    vals = {a: V(a) for a, _ in arms}
    xs = np.arange(len(arms))
    for i, sc in enumerate(scenes):
        clips = EL.index[EL == sc]
        y = [vals[a].loc[clips].median() for a, _ in arms]
        ax.plot(xs, y, color=SCENE_COLORS[i], marker=SCENE_MARKERS[i], ms=4.5, lw=1.4,
                markeredgecolor="white", markeredgewidth=0.6, label=f"{sc.replace('VIRAT_S_', '')} (n={len(clips)})")
    yall = [vals[a].median() for a, _ in arms]
    ax.plot(xs, yall, color=INK, lw=2.4, ls="--", marker="o", ms=5, zorder=5, label="all 27 clips")
    ax.annotate(f"{100 * yall[0]:.1f}%", (xs[0], yall[0]), xytext=(-6, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7.5, fontweight="bold")
    ax.annotate(f"{100 * yall[-1]:.1f}%", (xs[-1], yall[-1]), xytext=(6, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7.5, fontweight="bold")
    ax.set_xticks(xs, [l for _, l in arms])
    ax.set_xlim(-0.6, len(arms) - 0.4)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("STALE_MOV (tau = 1 px), median per scene")
axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, title="scene (eligible clips)",
               title_fontsize=7)
fig.tight_layout()
save(fig, "fig_b2_cctv_quality")

# ---------------------------------------------------------------- B3 (illustrative)
from lib.flo import SintelSeq  # noqa: E402
from lib.gt_grid import GTGrid  # noqa: E402
from run_pilot import SINTEL, paint  # noqa: E402

s45 = S("x264_crf45", "d1")
SEQ = s45.sort_values().index[len(s45) // 2]
assert s45[SEQ] == s45.median()
sq = SintelSeq(SINTEL, SEQ, "final")
H, W = sq.image(0).shape[:2]
Hc, Wc = -(-H // C), -(-W // C)


def cells(eid):
    """Per d1 P-frame cell maps at tau 1, rebuilt from the saved blocks as in run_pilot_v2.run_one."""
    blk = pd.read_parquet(os.path.join(P3, "sweep", "blocks", "final", f"{eid}__{SEQ}.parquet"))
    blk = blk[blk.group == "d1"]
    out = {}
    for t, b in blk.groupby("frame"):
        cmv = np.full((Hc * Wc, 2), np.nan)
        cdir = np.zeros(Hc * Wc, np.int8)
        for sgn, dname in ((1, "future"), (-1, "past")):  # future first, past overrides
            bb = b[b.dir == dname]
            if len(bb) == 0:
                continue
            ci, bi = paint(Hc, Wc, bb.x.values, bb.y.values, bb.w.values, bb.h.values)
            cmv[ci] = bb[["mv_x", "mv_y"]].values[bi]
            cdir[ci] = sgn
        out[t] = (cmv, cdir != 0)
    return out


GT = {}


def gt(t):
    if t not in GT:
        cgx, cgy, ccov, _ = GTGrid(sq.flow_into(t), sq.bad_mask_into(t)).cell_gt()
        GT[t] = (np.hypot(cgx, cgy).ravel(), ccov.ravel() >= COVER_MIN)
    return GT[t]


def classify(cmv, has, t):
    gmag, valid = gt(t)
    reuse = has & (np.hypot(cmv[:, 0], cmv[:, 1]) < TAU)
    vr = valid & reuse
    return vr & (gmag > STALE_GT), vr & (gmag < STATIC_GT), vr, valid


maps = {}
print(f"\nB3: sequence {SEQ} (x264 CRF 45 stale/valid {s45[SEQ]:.4f} = median of 23)")
for eid in ("x264_crf12", "x264_crf45"):
    cm = cells(eid)
    per = {t: classify(*cm[t], t) for t in sorted(cm)}
    tot = {k: sum(int(p[i].sum()) for p in per.values()) for i, k in enumerate(("stale", "tp", "reuse", "valid"))}
    row = R[(R["pass"] == "final") & (R.encode_id == eid) & (R.sequence == SEQ) & (R.group == "d1")
            & np.isclose(R.tau, TAU)].iloc[0]
    want = {"stale": row.stale_cells, "tp": row.tp_cells, "reuse": row.reuse_cells, "valid": row.valid_cells}
    print(f"  rebuild check {eid}: {tot} vs results.csv {dict((k, int(v)) for k, v in want.items())}")
    assert tot == {k: int(v) for k, v in want.items()}, (eid, tot, want)
    maps[eid] = per
counts = sorted((int(p[0].sum()), t) for t, p in maps["x264_crf45"].items())
n_st, FRAME = counts[(len(counts) - 1) // 2]
FRAME = min(t for c, t in counts if c == n_st)
print(f"  frame rule: {len(counts)} d1 P-frames; median stale count {n_st} -> display frame {FRAME} "
      f"(Sintel frame_{FRAME + 1:04d}.png)")

img = sq.image(FRAME).astype(float) / 255
grey = img.mean(axis=2, keepdims=True) * 0.75 + 0.25
base = np.repeat(grey, 3, axis=2)


def overlay(stale, tp):
    rgba = np.zeros((Hc * Wc, 4))
    for m, col in ((tp, REUSE_C), (stale, STALE_C)):
        rgba[m] = (*matplotlib.colors.to_rgb(col), 0.6)
    return np.kron(rgba.reshape(Hc, Wc, 4), np.ones((C, C, 1)))[:H, :W]


fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.75), gridspec_kw={"wspace": 0.04})
for ax, eid, lab in zip(axes, ("x264_crf12", "x264_crf45"), ("x264 CRF 12", "x264 CRF 45")):
    stale, tp, vr, valid = maps[eid][FRAME]
    ax.imshow(base, interpolation="nearest")
    ax.imshow(overlay(stale, tp), interpolation="nearest")
    ax.set_title(f"{lab}\n{int(stale.sum())} stale, {int(tp.sum())} correctly reused cells", fontsize=8)
    ax.set_axis_off()
    print(f"  {eid} frame {FRAME}: stale {int(stale.sum())}, correctly reused {int(tp.sum())}, "
          f"reused {int(vr.sum())}, valid {int(valid.sum())}")
handles = [matplotlib.patches.Patch(color=STALE_C, alpha=0.7, label="stale: reused, GT |flow| > 2 px"),
           matplotlib.patches.Patch(color=REUSE_C, alpha=0.7, label="correctly reused: reused, GT |flow| < 0.5 px")]
fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), fontsize=7.5)
fig.suptitle(f"Illustrative: Sintel {SEQ} (FINAL), frame {FRAME + 1}, 4x4-px cells, tau = 1 px", fontsize=8.5,
             y=1.1)
save(fig, "fig_b3_example_frame")
print("\nwrote fig_b1_bframes_matched_qp, fig_b2_cctv_quality, fig_b3_example_frame (.pdf, .png)")
