"""Review-2 item 3: missed reuse per quality level. DESCRIPTIVE (pre-registered as secondary, no prediction).

Sintel, CLEAN pass, tau = 1: missed/valid = GT-static cells (|GT| < 0.5 px) that the gate recomputes, divided by
valid cells, d1 P-frame group; median (and min / max) over 23 sequences per x264 CRF level and NVENC QP level.
CCTV confirmatory (p3/virat_confirm/results.csv, column `missed`): the same quantity with RAFT in place of GT
(|RAFT| < 0.5 px), d1 group, median over all 50 clips and over the 27 eligible clips. Writes i03_missed_reuse.md/.csv.
"""
import pandas as pd

from _common import S, V, eligible, CRFS, NQPS, f, table, write, HERE

CCTV_ARMS = (("x264 CRF 12", "x264_crf12"), ("x264 CRF 45", "x264_crf45"), ("x264 QP 24", "x264_qp24"),
             ("NVENC QP 18", "nvenc_qp18"), ("NVENC QP 23", "nvenc_qp23"), ("NVENC QP 28", "nvenc_qp28"),
             ("NVENC QP 45", "nvenc_qp45"))


def main():
    rows, sl = [], []
    for lab, eid in [(f"x264 CRF {c}", f"x264_crf{c}") for c in CRFS] + [(f"NVENC QP {q}", f"nvenc_qp{q}") for q in NQPS]:
        for pass_ in ("clean", "final"):
            s = S(eid, "d1", pass_=pass_, col="missed_of_valid")
            rows.append({"data": f"Sintel {pass_}", "arm": lab, "n": int(s.notna().sum()), "median": s.median(),
                         "min": s.min(), "max": s.max()})
        c = rows[-2]
        sl.append([lab, f(c["median"]), f(c["min"]), f(c["max"]), f(rows[-1]["median"])])
    el = eligible()
    cl = []
    for lab, arm in CCTV_ARMS:
        s = V(arm, "d1", col="missed")
        se = s.reindex(el)
        rows.append({"data": "CCTV confirmatory (all clips)", "arm": lab, "n": int(s.notna().sum()),
                     "median": s.median(), "min": s.min(), "max": s.max()})
        rows.append({"data": "CCTV confirmatory (eligible)", "arm": lab, "n": int(se.notna().sum()),
                     "median": se.median(), "min": se.min(), "max": se.max()})
        cl.append([lab, f(s.median()), f(s.min()), f(s.max()), f(se.median())])
    pd.DataFrame(rows).to_csv(f"{HERE}/i03_missed_reuse.csv", index=False)
    write("i03_missed_reuse.md", [
        "# Item 3: missed reuse / valid, tau = 1 (descriptive; no prediction)", "",
        "## Sintel, CLEAN pass, d1 P-frame vectors (23 sequences)", "",
        table(["arm", "median", "min", "max", "median, FINAL pass (for reference)"], sl), "",
        "## CCTV confirmatory test, d1 P-frame vectors, RAFT-static (|RAFT| < 0.5 px) cells recomputed / valid", "",
        table(["arm", "median (50 clips)", "min", "max", "median (27 eligible)"], cl), "",
        "CCTV static/valid uses RAFT, not ground truth; values are descriptive."])


if __name__ == "__main__":
    main()
