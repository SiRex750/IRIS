"""Independent recomputation of F1-F4, KILL-A2 and the decision rule (primary and tau 1 px; counted clips and the
unseen-window subgroup) from per_unit.csv, checks.csv and manifest.json alone.

Does not import analyze.py, simulate.py or pandas. Ratios are recomputed from the integer counts
(STALE_MOV = stale_tf / moved_tf, REUSE = reused_tf / valid_tf; empty denominator = missing), not read from the ratio
columns. Thresholds, arms and required shares are re-typed from PREREGISTRATION.md §5-§6 (cc2eaf9). Medians: middle
value / mean of the two middle values over non-missing units. R-chain overall verdict recomputed from checks.csv
(chain_r < 0.20 per counted clip, >= ceil(0.75 N)).
Then reads analyze.py's verdicts.csv (an output file, not code) and reports agreement on every count, result,
KILL-A2 and decision. Writes independent_verdicts.txt. Exit 1 on any disagreement.
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RULES = [("F1", "x264_crf45", "x264_crf12", 0.80), ("F2", "nvenc_qp45", "nvenc_qp18", 0.75),
         ("F3", "x264_qp24_bf2_pb1", "x264_qp24", 0.80), ("F4", "nvenc_qp28_bf2_bqeq", "nvenc_qp28", 0.75)]
ARMS = ["x264_crf12", "x264_crf45", "x264_qp24", "x264_qp24_bf2_pb1", "nvenc_qp18", "nvenc_qp23", "nvenc_qp45",
        "nvenc_qp28", "nvenc_qp28_bf2_bqeq"]


def rd(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ratio(a, b):
    a, b = int(float(a)), int(float(b))
    return a / b if b else None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def need(n, x):
    return math.ceil(x * n - 1e-9)


def main():
    ck = rd(os.path.join(HERE, "checks.csv"))
    counted = sorted(r["clip"] for r in ck if r["counted"] == "True")
    N = len(counted)
    rpass = sum(1 for r in ck if r["counted"] == "True" and r["chain_r"] not in ("", "nan") and float(r["chain_r"]) < 0.20)
    man = json.load(open(os.path.join(HERE, "manifest.json")))
    unseen = [c for c in counted if man["clips"][c]["start"] >= 600]
    pu = [r for r in rd(os.path.join(HERE, "per_unit.csv")) if r["truth"] == "raft"]
    mine = {}
    lines = [f"R-chain: {rpass}/{N} (need {need(N, 0.75)}) -> {'PASS' if rpass >= need(N, 0.75) else 'FAIL'}"]
    for cfg in ("primary", "tau1"):
        sm, ru = {}, {}
        for r in pu:
            if r["config"] == cfg:
                sm[(r["unit"], r["arm"])] = ratio(r["stale_tf"], r["moved_tf"])
                ru[(r["unit"], r["arm"])] = ratio(r["reused_tf"], r["valid_tf"])
        for sub, units in (("counted", counted), ("unseen", unseen)):
            for key, hi, lo, share in RULES:
                cnt = 0
                for u in units:
                    a, b = sm.get((u, hi)), sm.get((u, lo))
                    if a is not None and b is not None and a - b > 0.01:
                        cnt += 1
                req = need(len(units), share)
                mine[(cfg, sub, key)] = (cnt, "PASS" if cnt >= req else "FAIL")
            meds = {a: (median([sm.get((u, a)) for u in units]), median([ru.get((u, a)) for u in units])) for a in ARMS}
            ok = [a for a in ARMS if meds[a][0] is not None and meds[a][0] < 0.01 and meds[a][1] > 0.20]
            leak = [a for a in ARMS if meds[a][0] is not None and meds[a][0] > 0.01 and meds[a][1] > 0.20]
            little = [a for a in ARMS if not meds[a][1] > 0.20]
            if len(ok) == 9:
                dec = "fresh (KILL-A2 fires)"
            elif leak:
                dec = "leaks: " + ", ".join(leak)
            else:
                dec = "neither; reuses little: " + (", ".join(little) or "none")
            if sub == "counted":
                mine[(cfg, sub, "KILL-A2")] = (len(ok), "FIRES" if len(ok) == 9 else "DOES NOT FIRE")
            mine[(cfg, sub, "decision")] = (None, dec)
            lines.append(f"{cfg} {sub}: " + "; ".join(f"{a} REUSE {meds[a][1]:.5f} STALE_MOV {meds[a][0]:.5f}"
                                                     for a in ARMS))
    theirs = {}
    for r in rd(os.path.join(HERE, "verdicts.csv")):
        if r["subset"] in ("counted", "unseen") and r["config"] in ("primary", "tau1"):
            c = int(float(r["count"])) if r.get("count") not in (None, "") else None
            theirs[(r["config"], r["subset"], r["key"])] = (c, r["result"])
    bad = 0
    for k, (c, res) in sorted(mine.items()):
        t = theirs.get(k)
        same = t is not None and t[1] == res and (c is None or t[0] == c)
        bad += not same
        lines.append(f"{'OK  ' if same else 'DIFF'} {k}: independent {c} {res} | analyze {t}")
    lines.append(f"ALL AGREE: {bad == 0} ({len(mine)} items)")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(HERE, "independent_verdicts.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
