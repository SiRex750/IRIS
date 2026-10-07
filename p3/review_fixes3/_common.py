"""Shared read-only helpers for p3/review_fixes3 (third internal review). Nothing here writes to a results file.

Sintel: p3/sweep/results.csv (as collected; every arm/group a prediction uses is identical in p3/sweep_rerun, see
p3/sweep_rerun/COMPARISON.md). CCTV: p3/virat_confirm/results.csv (confirmatory test, pre-registration d62753f).
No encoding, MV extraction or RAFT is run anywhere in this folder.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
TAUS = (0.5, 1.0, 2.0)

_R = None
_V = None


def sintel():
    global _R
    if _R is None:
        _R = pd.read_csv(os.path.join(P3, "sweep", "results.csv"))
    return _R


def seqs():
    return sorted(sintel().sequence.unique())


def S(eid, group, tau=1.0, col="stale_of_valid", pass_="final"):
    """Per-sequence Series (index = the 23 sequences) of one column for one arm/group."""
    R = sintel()
    d = R[(R["pass"] == pass_) & (R["encode_id"] == eid) & (R["group"] == group) & np.isclose(R["tau"], tau)]
    return d.set_index("sequence")[col].reindex(seqs())


def virat():
    global _V
    if _V is None:
        _V = pd.read_csv(os.path.join(P3, "virat_confirm", "results.csv"))
    return _V


def eligible():
    """Eligible clips (index) -> scene, from the confirmatory test's eligibility file."""
    el = pd.read_csv(os.path.join(P3, "virat_confirm", "eligibility_v0.csv"))
    el = el[el.eligible.astype(bool)].sort_values("clip")
    return el.set_index("clip")["scene"]


def V(arm, group, tau=1.0, col="stale_mov"):
    r = virat()
    r = r[(r.arm == arm) & (r.group == group) & np.isclose(r.tau, tau)]
    return r.set_index("clip")[col]


# Sintel contrasts of review item R3 (label, low arm, low group, high arm, high group)
SCONTRASTS = (("x264 CRF 12 -> 45", "x264_crf12", "d1", "x264_crf45", "d1"),
              ("NVENC QP 18 -> 45", "nvenc_qp18", "d1", "nvenc_qp45", "d1"),
              ("x264 QP 24: bf 0 -> B (pb1, B_naive)", "x264_qp24", "d1", "x264_qp24_bf2_pb1", "B_naive"),
              ("NVENC QP 28: bf 0 -> B=P (bqeq, B_naive)", "nvenc_qp28", "d1", "nvenc_qp28_bf2_bqeq", "B_naive"))

# confirmatory predictions C1-C4: (key, low arm, low group, high arm, high group)
CPREDS = (("C1 x264 CRF 12 -> 45", "x264_crf12", "d1", "x264_crf45", "d1"),
          ("C2 NVENC QP 18 -> 45", "nvenc_qp18", "d1", "nvenc_qp45", "d1"),
          ("C3 x264 QP 24: bf 0 -> B", "x264_qp24", "d1", "x264_qp24_bf2_pb1", "B"),
          ("C4 NVENC QP 28: bf 0 -> B=P", "nvenc_qp28", "d1", "nvenc_qp28_bf2_bqeq", "B"))

# matched-quality series: the codec's main quality series (bf 0, d1)
SERIES = {"x264": [f"x264_crf{c}" for c in (12, 18, 23, 28, 33, 38, 45)],
          "NVENC": [f"nvenc_qp{q}" for q in (18, 23, 28, 33, 38, 45)],
          "MPEG-4": [f"mpeg4_q{q}" for q in (2, 4, 8, 16, 31)]}
TARGET_DB = 41.0


def matched_arms():
    """Per codec, the arm of its main series whose median (over 23 sequences) PSNR-Y is closest to 41 dB.
    Returns {codec: (encode_id, median psnr)}."""
    out = {}
    for codec, arms in SERIES.items():
        ps = {a: S(a, "d1", col="psnr_y").median() for a in arms}
        best = min(ps, key=lambda a: abs(ps[a] - TARGET_DB))
        out[codec] = (best, ps[best])
    return out


def f(x, nd=4):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def pct(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f}%"


def table(header, lines):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in ln) + " |" for ln in lines]
    return "\n".join(out)


def write(name, lines):
    open(os.path.join(HERE, name), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
