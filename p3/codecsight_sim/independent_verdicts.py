"""Independent recomputation of every S1x-K4 verdict and KILL-A (primary and tau 1 px) from per_unit.csv alone.

Does not import analyze.py or simulate.py, and does not use pandas. Ratios are recomputed from the integer counts
(STALE = stale_tf / valid_tf, STALE_MOV = stale_tf / moved_tf, REUSE = reused_tf / valid_tf; empty denominator =
missing), not read from the ratio columns. Spearman rho = Pearson correlation of average ranks. Thresholds, arms and
required counts are re-typed from PREREGISTRATION.md §6 (commit 9f5b5a3). Medians: middle value / mean of the two
middle values over non-missing units.

Then reads analyze.py's verdicts.csv (an output file, not code) and reports agreement on every verdict, count and
KILL-A. Writes independent_verdicts.txt.
Usage: python independent_verdicts.py [dir]   (default dir: this folder) Exit 1 on any disagreement.
"""
import csv
import hashlib
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PREREG = "9f5b5a3"
CODE_FILES = ("simulate.py", "analyze.py", "independent_verdicts.py", "test_policy.py")

RULES = [  # key, data, metric, high arm, low arm, margin, required
    ("S1x", "Sintel", "STALE", "x264_crf45", "x264_crf12", 0.001, 18),
    ("S1n", "Sintel", "STALE", "nvenc_qp45", "nvenc_qp18", 0.001, 18),
    ("S2x", "Sintel", "STALE", "x264_qp24_bf2_pb1", "x264_qp24", 0.001, 18),
    ("S2n", "Sintel", "STALE", "nvenc_qp28_bf2_bqeq", "nvenc_qp28", 0.001, 16),
    ("K1", "CCTV", "STALE_MOV", "x264_crf45", "x264_crf12", 0.01, 22),
    ("K2", "CCTV", "STALE_MOV", "nvenc_qp45", "nvenc_qp18", 0.01, 21),
    ("K3", "CCTV", "STALE_MOV", "x264_qp24_bf2_pb1", "x264_qp24", 0.01, 22),
    ("K4", "CCTV", "STALE_MOV", "nvenc_qp28_bf2_bqeq", "nvenc_qp28", 0.01, 21),
]
CRF_LEVELS = [12, 18, 23, 28, 33, 38, 45]
N_SINTEL, N_CCTV = 23, 27
N_ARMS = {"Sintel": 17, "CCTV": 9}


def sha_code():
    h = hashlib.sha256()
    for name in CODE_FILES:
        p = os.path.join(HERE, name)
        body = open(p, "rb").read() if os.path.isfile(p) else b""
        h.update(name.encode() + b"\0" + body + b"\0")
    return h.hexdigest()


def prereg_same():
    return subprocess.run(["git", "diff", "--quiet", PREREG, "--", "p3/codecsight_sim/PREREGISTRATION.md"],
                          cwd=REPO).returncode == 0


def ratio(num, den):
    return None if den == 0 else num / den


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    if any(v is None for v in y):
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def load(path):
    T = {}  # (config, data, arm, unit) -> dict of metrics
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            V, R, Mv, S = (int(float(r[k])) for k in ("valid_tf", "reused_tf", "moved_tf", "stale_tf"))
            key = (r["config"], r["data"], r["arm"], r["unit"])
            assert key not in T, ("duplicate row", key)
            T[key] = {"REUSE": ratio(R, V), "STALE": ratio(S, V), "STALE_MOV": ratio(S, Mv), "hash": r["code_sha256"]}
    return T


def main():
    D = sys.argv[1] if len(sys.argv) > 1 else HERE  # directory of per_unit.csv / verdicts.csv (default: here)
    T = load(os.path.join(D, "per_unit.csv"))
    hashes = sorted({v["hash"] for v in T.values()})
    me = sha_code()
    out = ["# Independent recomputation of the verdicts (independent_verdicts.py)",
           f"prereg {PREREG} (file unchanged: {prereg_same()}); code sha256 {me}; per_unit.csv code sha256 "
           f"{', '.join(hashes)}", ""]
    mine = {}
    for cfg in ("primary", "tau1"):
        units = {d: sorted({k[3] for k in T if k[0] == cfg and k[1] == d}) for d in ("Sintel", "CCTV")}
        arms = {d: sorted({k[2] for k in T if k[0] == cfg and k[1] == d}) for d in ("Sintel", "CCTV")}
        assert len(units["Sintel"]) == N_SINTEL and len(units["CCTV"]) == N_CCTV, (cfg, {d: len(u) for d, u in units.items()})
        assert len(arms["Sintel"]) == N_ARMS["Sintel"] and len(arms["CCTV"]) == N_ARMS["CCTV"]
        out.append(f"## {cfg}")
        for key, data, met, hi, lo, margin, req in RULES:
            cnt, miss = 0, []
            for u in units[data]:
                a, b = T[(cfg, data, hi, u)][met], T[(cfg, data, lo, u)][met]
                if a is None or b is None:
                    miss.append(u)
                elif a - b > margin:
                    cnt += 1
            v = "PASS" if cnt >= req else "FAIL"
            mine[(cfg, key)] = (v, cnt)
            out.append(f"{key:6s} {met}({hi}) - {met}({lo}) > {margin:g}: {cnt}/{len(units[data])}, need {req} -> {v}"
                       + (f"; missing {miss}" if miss else ""))
        cnt, miss = 0, []
        for u in units["Sintel"]:
            rho = spearman(CRF_LEVELS, [T[(cfg, "Sintel", f"x264_crf{c}", u)]["STALE"] for c in CRF_LEVELS])
            if rho is None:
                miss.append(u)
            elif rho >= 0.8:
                cnt += 1
        v = "PASS" if cnt >= 16 else "FAIL"
        mine[(cfg, "S1r")] = (v, cnt)
        out.append(f"S1r    Spearman(CRF, STALE) >= 0.8: {cnt}/23, need 16 -> {v}" + (f"; missing {miss}" if miss else ""))
        ok_arms = 0
        out.append("KILL-A arm medians (REUSE, STALE_MOV, units with STALE_MOV missing):")
        for data in ("Sintel", "CCTV"):
            for arm in arms[data]:
                re = median([T[(cfg, data, arm, u)]["REUSE"] for u in units[data]])
                sm_all = [T[(cfg, data, arm, u)]["STALE_MOV"] for u in units[data]]
                sm = median(sm_all)
                good = sm is not None and re is not None and sm < 0.01 and re > 0.20
                ok_arms += good
                mine[(cfg, "KILLA", data, arm)] = (re, sm)
                out.append(f"  {data:6s} {arm:20s} REUSE {re:.6f}  STALE_MOV {sm:.6f}  missing {sm_all.count(None)}"
                           f"  {'meets both' if good else 'does not meet'}")
        n_all = len(arms["Sintel"]) + len(arms["CCTV"])
        v = "FIRES" if ok_arms == n_all else "DOES NOT FIRE"
        mine[(cfg, "KILL-A")] = (v, ok_arms)
        out.append(f"KILL-A: {ok_arms}/{n_all} arm-medians meet both -> {v}")
        out.append("")

    # agreement with analyze.py's output
    agree, lines = True, []
    vp = os.path.join(D, "verdicts.csv")
    if not os.path.isfile(vp):
        agree, lines = False, ["verdicts.csv not found: no comparison"]
    else:
        with open(vp, newline="", encoding="utf-8") as fh:
            theirs = {(r["config"], r["key"]): (r["verdict"], int(r["count"])) for r in csv.DictReader(fh)}
        for k, v in mine.items():
            if len(k) != 2:
                continue
            t = theirs.get(k)
            same = t == v
            agree &= same
            lines.append(f"{k[0]:8s} {k[1]:7s} independent {v[0]:14s} {v[1]:3d} | analyze.py "
                         f"{(t or ('missing', -1))[0]:14s} {(t or ('missing', -1))[1]:3d} | {'agree' if same else 'DISAGREE'}")
        extra = set(theirs) - {k for k in mine if len(k) == 2}
        if extra:
            agree = False
            lines.append(f"keys only in verdicts.csv: {sorted(extra)}")
        am = os.path.join(D, "arm_medians.csv")
        maxdiff = 0.0
        with open(am, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                k = (r["config"], "KILLA", r["data"], r["arm"])
                if k in mine:
                    re, sm = mine[k]
                    maxdiff = max(maxdiff, abs(re - float(r["median_REUSE"])), abs(sm - float(r["median_STALE_MOV"])))
        ok_med = maxdiff < 1e-12
        agree &= ok_med
        lines.append(f"KILL-A arm medians vs arm_medians.csv: max |diff| {maxdiff:.3e} -> {'agree' if ok_med else 'DISAGREE'}")
    out += ["## Agreement with analyze.py (verdicts.csv, arm_medians.csv)"] + lines + ["", f"ALL AGREE: {agree}"]
    txt = "\n".join(out) + "\n"
    open(os.path.join(D, "independent_verdicts.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    sys.exit(0 if agree else 1)


if __name__ == "__main__":
    main()
