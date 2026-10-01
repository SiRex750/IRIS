"""Sintel sweep: original run (p3/sweep, frame-threaded MV export) vs rerun (p3/sweep_rerun, single-threaded MV
export; p3/sweep/DEVIATIONS.md 1). Writes p3/sweep_rerun/COMPARISON.md; also writes the rerun's own summary.md
(summarize_sweep.py with SWEEP_OUT=p3/sweep_rerun) and independent_verdicts.txt.

  1. verdict tables side by side: summarize_sweep.verdicts() and independent_verdicts.py on each results.csv
  2. did any verdict change
  3. per prediction, the largest per-sequence change in stale/valid of any arm/group it uses, and of the
     per-sequence quantity it counts (difference, rho, |scaled - naive|)
  4. results.csv row by row: which (pass, arm, group) rows differ in any column
  5. encodes byte-identical (sha256) and block records (blocks/*.parquet) frame by frame
"""
import glob
import hashlib
import importlib.util
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.environ.get("RERUN_DIR", os.path.dirname(os.path.abspath(__file__)))  # override: smoke tests only
P3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG = os.path.join(P3, "sweep")
RUNS = {"original": ORIG, "rerun": HERE}
KEY = ["pass", "encode_id", "group", "sequence", "tau"]


def load_summarizer(d, tag):
    """summarize_sweep.py as a fresh module reading d/results.csv (it reads SWEEP_OUT at import)."""
    os.environ["SWEEP_OUT"] = d
    spec = importlib.util.spec_from_file_location(f"summ_{tag}", os.path.join(ORIG, "summarize_sweep.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def independent(d):
    """independent_verdicts.py (chat auditor) run in d; returns (output, {prediction: (value, threshold, verdict)},
    error). Its verdict table is printed before its detail sections, so it is parsed even if those fail."""
    out = subprocess.run([sys.executable, os.path.join(ORIG, "independent_verdicts.py")], cwd=d,
                         capture_output=True, text=True)
    rows = {}
    for ln in out.stdout.splitlines():
        if ln.rstrip().endswith(("PASS", "FAIL")) and (ln.startswith("P1") or ln.startswith("P2") or ln.startswith("P3")):
            rows[ln[:45].strip()] = (ln[46:74].strip(), ln[75:89].strip(), ln.split()[-1])
    if out.returncode == 0:
        return out.stdout, rows, ""
    err = f"exit code {out.returncode}: {(out.stderr.strip().splitlines() or ['?'])[-1]}"
    return out.stdout + "\n[stderr]\n" + out.stderr, rows, err


def components(m):
    """{prediction: ([(eid, group)], per-sequence counted quantity)} for the rows of summarize_sweep.verdicts()."""
    S = m.S
    c = {"P1 x264 CRF 45 vs 12": ([("x264_crf45", "d1"), ("x264_crf12", "d1")],
                                  S("x264_crf45", "d1") - S("x264_crf12", "d1")),
         "P1 NVENC QP 45 vs 18": ([("nvenc_qp45", "d1"), ("nvenc_qp18", "d1")],
                                  S("nvenc_qp45", "d1") - S("nvenc_qp18", "d1")),
         "P1 mpeg4 q 31 vs 2 (reported, not a kill rule)": ([("mpeg4_q31", "d1"), ("mpeg4_q2", "d1")],
                                                            S("mpeg4_q31", "d1") - S("mpeg4_q2", "d1"))}
    rows, notes, rho, bx, bn = m.verdicts()
    c["P1 Spearman(CRF, stale/valid), 7 x264 CRF levels"] = ([(f"x264_crf{q}", "d1") for q in m.CRFS], rho)
    c["P2 x264 QP 24: B naive minus bf0 d1"] = ([("x264_qp24_bf2_pb1", "B_naive"), ("x264_qp24", "d1")], bx)
    c["P2 NVENC QP 28 B=P: B naive minus bf0 d1"] = ([("nvenc_qp28_bf2_bqeq", "B_naive"), ("nvenc_qp28", "d1")], bn)
    for lab, eid in [("x264 QP 24 pair", "x264_qp24_bf2_pb1"), ("NVENC QP 28 B=P pair", "nvenc_qp28_bf2_bqeq")]:
        c[f"P2b abs(B scaled - B naive), {lab}"] = ([(eid, "B_scaled"), (eid, "B_naive")],
                                                    (S(eid, "B_scaled") - S(eid, "B_naive")).abs())
    for lab, a, ga, b, gb in m.P3_ARMS:
        c[f"P3 {lab} vs {'umh 16' if 'umh' in lab else 'CRF 23 medium'}"] = ([(a, ga), (b, gb)],
                                                                              (S(a, ga) - S(b, gb)).abs())
    c["P2c median x264 B effect > median abs change of every P3 arm"] = (
        [("x264_qp24_bf2_pb1", "B_naive"), ("x264_qp24", "d1"), ("x264_crf23", "d1"), ("x264_crf33", "d1")]
        + [(a, ga) for _, a, ga, _, _ in m.P3_ARMS] + [(b, gb) for _, _, _, b, gb in m.P3_ARMS], bx)
    return rows, notes, c


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmp_blocks(rel):
    """Frame numbers whose block records differ between the two runs (rows compared as sorted sets)."""
    a = pd.read_parquet(os.path.join(ORIG, "blocks", rel))
    b = pd.read_parquet(os.path.join(HERE, "blocks", rel))
    if list(a.columns) != list(b.columns):
        return rel, "columns differ"
    if a.equals(b):
        return rel, ""
    fcol = "frame" if "frame" in a.columns else None
    if fcol is None:
        return rel, "differs"
    diff = []
    ga, gb = dict(tuple(a.groupby(fcol))), dict(tuple(b.groupby(fcol)))
    for t in sorted(set(ga) | set(gb)):
        x, y = ga.get(t), gb.get(t)
        if x is None or y is None or len(x) != len(y):
            diff.append(int(t))
            continue
        cols = list(a.columns)
        x = x.sort_values(cols).reset_index(drop=True)
        y = y.sort_values(cols).reset_index(drop=True)
        if not x.equals(y):
            diff.append(int(t))
    return rel, " ".join(map(str, diff))


def table(header, lines):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in ln) + " |" for ln in lines]
    return "\n".join(out)


def f(x, nd=5):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    T0 = time.perf_counter()
    L = ["# Sintel sweep: original vs rerun with single-threaded MV extraction", "",
         "Original: p3/sweep (bb8122c; frame-threaded MV export). Rerun: p3/sweep_rerun (same arms, sequences, "
         "settings and code except lib.mvs.extract_mvs single-threaded; p3/sweep/DEVIATIONS.md 1). "
         "Pre-registration 0bf4cac. Written by p3/sweep_rerun/compare.py.", ""]
    missing = [k for k, d in RUNS.items() if not os.path.exists(os.path.join(d, "results.csv"))]
    if missing:
        raise SystemExit(f"results.csv missing for {missing}")
    man = {k: __import__("json").load(open(os.path.join(d, "manifest.json"))) for k, d in RUNS.items()}
    L += [f"- Rerun: {man['rerun']['start_time']} → {man['rerun'].get('end_time', '(not finished)')}, HEAD "
          f"`{man['rerun']['git_hash']}`, code sha256 `{man['rerun']['code_sha256'][:16]}…` (original "
          f"`{man['original']['code_sha256'][:16]}…`), runs {man['rerun'].get('n_runs')}, failures "
          f"{man['rerun'].get('n_failures')}, check mismatches {man['rerun'].get('n_mismatches')} (original: "
          f"{man['original'].get('n_runs')}, {man['original'].get('n_failures')}, {man['original'].get('n_mismatches')}).",
          f"- Code files that differ (sha256): "
          + (", ".join(k for k in sorted(set(man['rerun']['code_files_sha256']) | set(man['original']['code_files_sha256']))
                       if man['rerun']['code_files_sha256'].get(k) != man['original']['code_files_sha256'].get(k))
             or "none"), ""]

    # rerun summary.md (mechanical, same script)
    env = {**os.environ, "SWEEP_OUT": HERE}
    s = subprocess.run([sys.executable, os.path.join(ORIG, "summarize_sweep.py")], env=env, capture_output=True, text=True)
    L.append(f"- Rerun summary.md written by summarize_sweep.py: {'ok' if s.returncode == 0 else 'FAILED: ' + s.stderr[-500:]}")

    # 1-2 verdicts
    summ = {k: components(load_summarizer(d, k)) for k, d in RUNS.items()}
    ind = {}
    for k, d in RUNS.items():
        txt, rows, err = independent(d)
        ind[k] = rows
        if err:
            L.append(f"- independent_verdicts.py on the {k} run: {err} (verdict table parsed from the output printed "
                     f"before the error: {len(rows)} rows)")
        if k == "rerun":
            open(os.path.join(HERE, "independent_verdicts.txt"), "w", encoding="utf-8").write(txt)
    L.append("- Rerun independent_verdicts.txt written by p3/sweep/independent_verdicts.py.")
    L += ["", "## 1. Verdicts, original vs rerun (τ = 1, FINAL pass)", "", "### summarize_sweep.py", ""]
    lines, changed = [], []
    o_rows = {r[0]: r for r in summ["original"][0]}
    for r in summ["rerun"][0]:
        o = o_rows.get(r[0])
        same = o is not None and o[3] == r[3]
        if not same:
            changed.append(f"summarize_sweep: {r[0]}")
        lines.append([r[0], o[1] if o else "–", o[3] if o else "–", r[1], r[3], "" if same else "CHANGED",
                      "same" if o is not None and o[1] == r[1] else "differs"])
    L += [table(["prediction", "original value", "original", "rerun value", "rerun", "verdict change", "value"], lines), ""]
    L += ["### independent_verdicts.py (chat auditor)", ""]
    lines = []
    for k in ind["rerun"]:
        o, r = ind["original"].get(k), ind["rerun"][k]
        same = o is not None and o[2] == r[2]
        if not same:
            changed.append(f"independent: {k}")
        lines.append([k, o[0] if o else "–", o[2] if o else "–", r[0], r[2], "" if same else "CHANGED",
                      "same" if o is not None and o[0] == r[0] else "differs"])
    L += [table(["prediction", "original value", "original", "rerun value", "rerun", "verdict change", "value"], lines), ""]
    L += [f"**Any verdict changed: {'YES — ' + '; '.join(changed) if changed else 'NO'}** "
          f"(summarize_sweep: {len(summ['rerun'][0])} rows; independent: {len(ind['rerun'])} rows).", ""]
    if summ["rerun"][1]:
        L += ["Rerun notes (missing values):"] + [f"- {n}" for n in summ["rerun"][1]] + [""]

    # 3 per-prediction largest per-sequence change
    R = {k: pd.read_csv(os.path.join(d, "results.csv")) for k, d in RUNS.items()}
    ser = {}
    for k in RUNS:
        r = R[k][(R[k]["pass"] == "final") & np.isclose(R[k]["tau"], 1.0)]
        ser[k] = r.set_index(["encode_id", "group", "sequence"])["stale_of_valid"]
    lines = []
    for name, (comps, q_r) in summ["rerun"][2].items():
        q_o = summ["original"][2][name][1]
        best = (0.0, "–")
        for eid, g in comps:
            for seq in q_r.index:
                a, b = ser["original"].get((eid, g, seq), np.nan), ser["rerun"].get((eid, g, seq), np.nan)
                dlt = abs(b - a) if not (np.isnan(a) and np.isnan(b)) else 0.0
                if np.isnan(dlt):
                    dlt = np.inf
                if dlt > best[0]:
                    best = (dlt, f"{eid} {g} {seq}")
        dq = (q_r - q_o).abs()
        nq = int((dq > 0).sum())
        lines.append([name, f(best[0], 6) if best[0] else "0", best[1] if best[0] else "–",
                      f(dq.max(), 6) if nq else "0", dq.idxmax() if nq else "–", f"{nq}/{len(dq)}"])
    L += ["## 2. Largest per-sequence change (FINAL, τ = 1)", "",
          "Per prediction: the largest |rerun − original| stale/valid over every arm/group the prediction uses and "
          "every sequence; and the largest change in the per-sequence quantity it counts.", "",
          table(["prediction", "max abs Δ stale/valid", "where", "max abs Δ counted quantity", "sequence",
                 "sequences changed"], lines), ""]

    # 4 results.csv row by row
    a, b = R["original"], R["rerun"]
    mg = a.merge(b, on=KEY, how="outer", suffixes=("_o", "_r"), indicator=True)
    only = mg[mg["_merge"] != "both"]
    both = mg[mg["_merge"] == "both"]
    cols = [c for c in a.columns if c not in KEY]
    dif = np.zeros(len(both), bool)
    dcols = {}
    for c in cols:
        x, y = both[c + "_o"], both[c + "_r"]
        if x.dtype.kind in "fi" and y.dtype.kind in "fi":
            d = ~((x == y) | (x.isna() & y.isna()))
        else:
            d = ~((x.astype(str) == y.astype(str)))
        d = d.to_numpy()
        if d.any():
            dcols[c] = int(d.sum())
        dif |= d
    drows = both[dif]
    grp = drows.groupby(["pass", "encode_id", "group"]).size().reset_index(name="rows (seq × τ)")
    L += ["## 3. results.csv, row by row (every column, exact)", "",
          f"- Rows: original {len(a)}, rerun {len(b)}; only in one run: {len(only)}.",
          f"- Rows with any differing value: {int(dif.sum())}/{len(both)}; columns involved: "
          + (", ".join(f"{c} ({n})" for c, n in sorted(dcols.items())) or "none") + ".", ""]
    if len(grp):
        L += [table(list(grp.columns), grp.values.tolist()), ""]

    # 5 encodes and blocks
    enc = []
    for p in sorted(glob.glob(os.path.join(ORIG, "encodes", "*", "*.mp4"))):
        rel = os.path.relpath(p, os.path.join(ORIG, "encodes"))
        q = os.path.join(HERE, "encodes", rel)
        enc.append({"file": rel.replace("\\", "/"), "status": "missing in rerun" if not os.path.exists(q)
                    else ("identical" if sha(p) == sha(q) else "DIFFERENT")})
    enc = pd.DataFrame(enc)
    extra = len(glob.glob(os.path.join(HERE, "encodes", "*", "*.mp4"))) - int((enc.status != "missing in rerun").sum())
    L += ["## 4. Encodes and block records", "",
          f"- Encodes byte-identical (sha256): {int((enc.status == 'identical').sum())}/{len(enc)}; different: "
          f"{int((enc.status == 'DIFFERENT').sum())}; missing in rerun: {int((enc.status == 'missing in rerun').sum())}; "
          f"only in rerun: {extra}."]
    bad = enc[enc.status != "identical"]
    if len(bad):
        L += [f"  - {r.file}: {r.status}" for r in bad.itertuples()][:200]
    rels = sorted(os.path.relpath(p, os.path.join(ORIG, "blocks")) for p in glob.glob(os.path.join(ORIG, "blocks", "*", "*.parquet"))
                  if os.path.exists(os.path.join(HERE, "blocks", os.path.relpath(p, os.path.join(ORIG, "blocks")))))
    with mp.get_context("spawn").Pool(6) as pool:
        bl = dict(pool.map(cmp_blocks, rels, chunksize=8))
    bdf = pd.DataFrame([{"file": k.replace("\\", "/"), "differing_frames": v} for k, v in bl.items()])
    bdf.to_csv(os.path.join(HERE, "blocks_comparison.csv"), index=False)
    bd = bdf[bdf.differing_frames != ""]
    L += [f"- Block record files compared: {len(bdf)} (block + frame-type files present in both runs); differing: "
          f"{len(bd)} (blocks_comparison.csv).",
          "- Expected from p3/sweep/DEVIATIONS.md 1: only the last frame of B-frame arms; up to 112 block files "
          "(the threaded export can vary between runs, so the 26 matching B files may also differ)."]
    if len(bd):
        kinds = bd.assign(arm=bd.file.str.split("/").str[1].str.split("__").str[0])
        L += ["", table(["arm", "files differing", "differing frames (distinct)"],
                        [[a_, len(g), " ".join(sorted(set(" ".join(g.differing_frames).split()), key=lambda x: (len(x), x)))]
                         for a_, g in kinds.groupby("arm")])]
    L += ["", f"Runtime of compare.py: {time.perf_counter() - T0:.0f}s."]
    open(os.path.join(HERE, "COMPARISON.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
