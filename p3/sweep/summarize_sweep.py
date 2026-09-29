"""Mechanical summary of the Sintel sweep: p3/sweep/results.csv -> p3/sweep/summary.md.

Every verdict is computed from results.csv with the thresholds of p3/PREREGISTRATION.md.
Metric: stale_of_valid (stale/valid), tau = 1, FINAL pass, unless stated. Counts are out of 23;
a sequence with a missing value counts as not meeting the condition (and is listed).
"""
import json
import os

import av
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.environ.get("SWEEP_OUT", os.path.dirname(os.path.abspath(__file__)))
TAU = 1.0
CRFS = (12, 18, 23, 28, 33, 38, 45)
NQPS = (18, 23, 28, 33, 38, 45)
MQS = (2, 4, 8, 16, 31)
P3_ARMS = [("ref 2", "x264_crf23_ref2", "d1(assumed)", "x264_crf23", "d1"),
           ("ref 3", "x264_crf23_ref3", "d1(assumed)", "x264_crf23", "d1"),
           ("keyint 30", "x264_crf23_keyint30", "d1", "x264_crf23", "d1"),
           ("umh 32", "x264_crf23_umh32", "d1", "x264_crf23_umh16", "d1"),
           ("umh 64", "x264_crf23_umh64", "d1", "x264_crf23_umh16", "d1")]

R = pd.read_csv(os.path.join(HERE, "results.csv"))
MAN = json.load(open(os.path.join(HERE, "manifest.json")))
LOG = json.load(open(os.path.join(HERE, "encode_log.json")))
SEQS = sorted(MAN["sequences"], key=lambda s: -MAN["static_share"][s])


def S(eid, group, pass_="final", tau=TAU, col="stale_of_valid"):
    """Per-sequence Series (index = SEQS) of one metric for one arm/group."""
    d = R[(R["pass"] == pass_) & (R["encode_id"] == eid) & (R["group"] == group) & np.isclose(R["tau"], tau)]
    return d.set_index("sequence")[col].reindex(SEQS)


def count(diff, thr):
    return int((diff > thr).sum()), [s for s in SEQS if pd.isna(diff[s])]


def f(x, nd=4):
    return "–" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def verdicts(tau=TAU):
    rows, notes = [], []

    def add(name, value, thr, passed, missing=()):
        rows.append((name, value, thr, "PASS" if passed else "FAIL"))
        if missing:
            notes.append(f"{name}: missing for {', '.join(missing)} (counted as not meeting the condition)")

    # P1
    for lab, hi, lo, need in [("P1 x264 CRF 45 vs 12", "x264_crf45", "x264_crf12", 18),
                              ("P1 NVENC QP 45 vs 18", "nvenc_qp45", "nvenc_qp18", 18),
                              ("P1 mpeg4 q 31 vs 2 (reported, not a kill rule)", "mpeg4_q31", "mpeg4_q2", 16)]:
        n, miss = count(S(hi, "d1", tau=tau) - S(lo, "d1", tau=tau), 0.001)
        add(lab, f"{n}/23 with diff > 0.001", f">= {need}/23", n >= need, miss)
    # P1 Spearman (quality level = CRF value; higher CRF = lower quality)
    M = pd.concat([S(f"x264_crf{c}", "d1", tau=tau) for c in CRFS], axis=1)
    rho = pd.Series({s: (spearmanr(CRFS, M.loc[s].to_numpy()).statistic if M.loc[s].notna().all() else np.nan)
                     for s in SEQS}, dtype=float)
    n = int((rho >= 0.8).sum())
    add("P1 Spearman(CRF, stale/valid), 7 x264 CRF levels", f"{n}/23 with rho >= 0.8", ">= 16/23", n >= 16,
        [s for s in SEQS if pd.isna(rho[s])])
    # P2
    bx = S("x264_qp24_bf2_pb1", "B_naive", tau=tau) - S("x264_qp24", "d1", tau=tau)
    bn = S("nvenc_qp28_bf2_bqeq", "B_naive", tau=tau) - S("nvenc_qp28", "d1", tau=tau)
    n, miss = count(bx, 0.001)
    add("P2 x264 QP 24: B naive minus bf0 d1", f"{n}/23 with diff > 0.001", ">= 18/23", n >= 18, miss)
    n, miss = count(bn, 0.001)
    add("P2 NVENC QP 28 B=P: B naive minus bf0 d1", f"{n}/23 with diff > 0.001", ">= 16/23", n >= 16, miss)
    # P2b (the pre-registration does not name the pair; both reported)
    for lab, eid in [("x264 QP 24 pair", "x264_qp24_bf2_pb1"), ("NVENC QP 28 B=P pair", "nvenc_qp28_bf2_bqeq")]:
        dd = (S(eid, "B_scaled", tau=tau) - S(eid, "B_naive", tau=tau)).abs()
        m = dd.median()
        add(f"P2b abs(B scaled - B naive), {lab}", f"median {f(m)}", "< 0.005", m < 0.005,
            [s for s in SEQS if pd.isna(dd[s])])
    # P3
    base = (S("x264_crf33", "d1", tau=tau) - S("x264_crf23", "d1", tau=tau)).abs().median()
    p3med = {}
    for lab, a, ga, b, gb in P3_ARMS:
        dd = (S(a, ga, tau=tau) - S(b, gb, tau=tau)).abs()
        p3med[lab] = dd.median()
        add(f"P3 {lab} vs {'umh 16' if 'umh' in lab else 'CRF 23 medium'}", f"median abs diff {f(p3med[lab], 5)}",
            f"< 0.5 x {f(base, 5)} = {f(0.5 * base, 5)}", p3med[lab] < 0.5 * base,
            [s for s in SEQS if pd.isna(dd[s])])
    # P2c
    mb = bx.median()
    mx = max(p3med.values())
    add("P2c median x264 B effect > median abs change of every P3 arm", f"median B effect {f(mb, 5)}",
        f"> max P3 median {f(mx, 5)} ({max(p3med, key=p3med.get)})", mb > mx)
    return rows, notes, rho, bx, bn


def table(header, lines):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in ln) + " |" for ln in lines]
    return "\n".join(out)


def seq_table(cols, pass_="final", col="stale_of_valid", extra=None):
    """cols: [(label, eid, group)]. extra: {label: per-sequence Series}."""
    ser = {lab: S(e, g, pass_=pass_, col=col) for lab, e, g in cols}
    hdr = ["sequence", "static"] + [c[0] for c in cols] + list(extra or {})
    lines = []
    for s in SEQS:
        lines.append([s, f"{MAN['static_share'][s]:.3f}"] + [f(ser[c[0]][s]) for c in cols]
                     + [f(v[s], 2 if "ρ" in k else 4) for k, v in (extra or {}).items()])
    return table(hdr, lines)


def mpeg4_marker_check(pass_, eid, seq, q):
    """Parse every resync-marker-shaped match in an mpeg4 encode as a video-packet header.
    Genuine iff quant_scale == q and macroblock_number < 64*28 (11-bit field for 1024x448)."""
    path = os.path.join(HERE, "encodes", pass_, f"{eid}__{seq}.mp4")
    res = []
    with av.open(path) as c:
        for k, pk in enumerate(c.demux(c.streams.video[0])):
            b = bytes(pk)
            if not b:
                continue
            a = np.frombuffer(b, np.uint8)
            for i in np.nonzero((a[:-2] == 0) & (a[1:-1] == 0) & (a[2:] != 0) & (a[2:] != 1))[0]:
                bits = "".join(f"{x:08b}" for x in b[i:i + 8])
                z = len(bits) - len(bits.lstrip("0"))
                rest = bits[z + 1:]
                mb, qs = int(rest[:11], 2), int(rest[11:16], 2)
                res.append({"packet": k, "rel": i / len(b), "zeros": z, "mb": mb, "quant": qs,
                            "genuine": bool(qs == q and mb < 64 * 28)})
    return res


def main():
    L = []
    rows, notes, rho, bx, bn = verdicts()
    L += ["# Paper 3 — Sintel sweep summary", "",
          f"Pre-registration commit `{MAN['prereg_commit']}`; code frozen at `{MAN['code_frozen_commit']}`; "
          f"run at HEAD `{MAN['git_hash']}`; code sha256 `{MAN['code_sha256'][:16]}…`.",
          f"Run {MAN['start_time']} → {MAN.get('end_time', '(not finished)')}. "
          "Computed mechanically from results.csv by summarize_sweep.py. stale/valid, τ = 1, FINAL pass unless stated.",
          "", "## 1. Prediction verdicts (τ = 1, primary)", "",
          table(["prediction", "value", "threshold", "verdict"], rows), ""]
    if notes:
        L += ["Notes:"] + [f"- {n}" for n in notes] + [""]
    v = {r[0]: r[3] for r in rows}
    p1x, p1n = v["P1 x264 CRF 45 vs 12"], v["P1 NVENC QP 45 vs 18"]
    kill = []
    if p1x == "FAIL" and p1n == "FAIL":
        kill.append("P1 fails for both x264 and NVENC → drop the quantiser claim (robustness paper).")
    elif "FAIL" in (p1x, p1n):
        kill.append(f"P1 fails for {'x264' if p1x == 'FAIL' else 'NVENC'} only → quantiser claim is encoder-specific.")
    for enc, k in [("x264", "P2 x264 QP 24: B naive minus bf0 d1"),
                   ("NVENC", "P2 NVENC QP 28 B=P: B naive minus bf0 d1")]:
        if v[k] == "FAIL":
            kill.append(f"P2 fails for {enc} → drop the B-frame claim for {enc}; report as a null.")
    for r in rows:
        if r[0].startswith("P3") and r[3] == "FAIL":
            kill.append(f"{r[0]} fails → that factor is reported as mattering; "
                        "drop \"only quantiser and B-frames matter\".")
    L += ["Kill rules triggered (mechanical reading of the pre-registration):"] + \
         [f"- {k}" for k in kill or ["none"]] + [""]
    L += ["Reading choices where the pre-registration is not explicit:",
          "- P1 Spearman: quality level = CRF value (higher = worse), so the prediction is ρ ≥ 0.8 with CRF.",
          "- P2b does not name which matched pair it applies to; both pairs are reported separately and not combined.",
          "- P2c uses the signed per-sequence B effect (as defined in P2), median over sequences.",
          "- P3 ref 2/3 use the `d1(assumed)` group (all P vectors read as d = 1, as in the pilots).", ""]

    L += ["### Secondary: the same computations at τ = 0.5 and τ = 2 (reported, not verdicts)", ""]
    alt = {tau: verdicts(tau)[0] for tau in (0.5, 2.0)}
    L += [table(["prediction", "τ = 0.5", "τ = 2"],
                [[r[0], f"{alt[0.5][i][1]} ({alt[0.5][i][3]})", f"{alt[2.0][i][1]} ({alt[2.0][i][3]})"]
                 for i, r in enumerate(rows)]), ""]

    L += ["## 2. Per-sequence tables", "",
          "Sequences are ordered by static share (share of valid GT pixels with |GT| < 0.5 px, from "
          "p3/pilot/static_scan.csv, highest first). Values are stale/valid at τ = 1 for the d = 1 P-frame group; "
          "B columns use B-frame vectors.", "",
          "### 2a. x264 CRF series (final)", "",
          seq_table([(f"CRF {c}", f"x264_crf{c}", "d1") for c in CRFS],
                    extra={"45−12": S("x264_crf45", "d1") - S("x264_crf12", "d1"), "Spearman ρ": rho}), "",
          "### 2b. NVENC QP series and mpeg4 q series (final)", "",
          seq_table([(f"NV {q}", f"nvenc_qp{q}", "d1") for q in NQPS]
                    + [(f"m4 q{q}", f"mpeg4_q{q}", "d1") for q in MQS],
                    extra={"NV 45−18": S("nvenc_qp45", "d1") - S("nvenc_qp18", "d1"),
                           "m4 31−2": S("mpeg4_q31", "d1") - S("mpeg4_q2", "d1")}), "",
          "### 2c. x264 CRF 23 variants (final)", "",
          seq_table([("CRF 23", "x264_crf23", "d1"), ("ref 2*", "x264_crf23_ref2", "d1(assumed)"),
                     ("ref 3*", "x264_crf23_ref3", "d1(assumed)"), ("keyint 30", "x264_crf23_keyint30", "d1"),
                     ("ultrafast", "x264_crf23_ultrafast", "d1"), ("veryslow", "x264_crf23_veryslow", "d1"),
                     ("umh 16", "x264_crf23_umh16", "d1"), ("umh 32", "x264_crf23_umh32", "d1"),
                     ("umh 64", "x264_crf23_umh64", "d1"), ("bf2 B naive", "x264_crf23_bf2", "B_naive")]),
          "", "\\* ref 2/3 vectors read as d = 1 (the decoder does not export the reference picture).", "",
          "### 2d. B-frames at matched QP (final)", "",
          seq_table([("x264 QP24 bf0", "x264_qp24", "d1"), ("x264 B naive", "x264_qp24_bf2_pb1", "B_naive"),
                     ("x264 B scaled", "x264_qp24_bf2_pb1", "B_scaled"), ("NV QP28 bf0", "nvenc_qp28", "d1"),
                     ("NV B=P naive", "nvenc_qp28_bf2_bqeq", "B_naive"),
                     ("NV B=P scaled", "nvenc_qp28_bf2_bqeq", "B_scaled"),
                     ("NV default-B naive", "nvenc_qp28_bf2", "B_naive")],
                    extra={"x264 B−bf0": bx, "NV B−bf0": bn}), "",
          "### 2e. Missed reuse / valid (CLEAN pass, τ = 1; descriptive only)", "",
          seq_table([(f"CRF {c}", f"x264_crf{c}", "d1") for c in CRFS]
                    + [(f"NV {q}", f"nvenc_qp{q}", "d1") for q in NQPS]
                    + [("x264 QP24 bf0", "x264_qp24", "d1"), ("x264 B naive", "x264_qp24_bf2_pb1", "B_naive"),
                       ("NV QP28 bf0", "nvenc_qp28", "d1"), ("NV B=P naive", "nvenc_qp28_bf2_bqeq", "B_naive")],
                    pass_="clean", col="missed_of_valid"), ""]

    # 3. failures / oddities / runtime
    logs, fails, mism = LOG["logs"], LOG["failures"], LOG["mismatches"]
    n_exp = len(MAN["sequences"]) * sum(len(st["arms"]) for st in MAN["stages"])
    L += ["## 3. Failures, oddities, runtime", "",
          f"- Runs: {len(logs)} encode × sequence × pass ({sum(lg['status'] == 'ok' for lg in logs)} ok); "
          f"expected {n_exp}.",
          f"- Failures: {len(fails)}" + ("" if not fails else ":")]
    L += [f"  - {x['pass']} {x['encode_id']} {x['sequence']}: {x['error']}" for x in fails]
    L += [f"- Check mismatches: {len(mism)}" + ("" if not mism else ":")]
    m4 = []
    for x in mism:
        L.append(f"  - {x['pass']} {x['encode_id']} {x['sequence']}: {x['mismatch']}")
        if x["encode_id"].startswith("mpeg4") and "slices" in x["mismatch"]:
            q = int(x["encode_id"].split("_q")[1])
            for h in mpeg4_marker_check(x["pass"], x["encode_id"], x["sequence"], q):
                m4.append({**h, "encode_id": x["encode_id"], "sequence": x["sequence"], "q": q})
    if m4:
        ng = sum(h["genuine"] for h in m4)
        L += [f"- mpeg4 extra-packet flags, each match parsed as a video-packet header: {len(m4)} matches, "
              f"{ng} consistent with a genuine resync marker (quant_scale = q and MB index in range), "
              f"{len(m4) - ng} not (chance `00 00 xx` in coded data; the packet counter is a byte-pattern heuristic):"]
        L += [f"  - {h['encode_id']} {h['sequence']} packet {h['packet']} at {h['rel']:.0%}: zeros {h['zeros']}, "
              f"MB {h['mb']}, quant {h['quant']} (q = {h['q']}) → {'GENUINE?' if h['genuine'] else 'false positive'}"
              for h in m4]
    ok = [lg for lg in logs if lg["status"] == "ok"]
    x264 = [lg for lg in ok if lg["codec"] == "libx264"]
    L += [f"- x264 SEI: threads=1 in {sum(lg.get('sei_threads') == '1' for lg in x264)}/{len(x264)}, "
          f"sliced_threads=0 in {sum(lg.get('sei_sliced_threads') == '0' for lg in x264)}/{len(x264)}, "
          f"no `slices` field in {sum(lg.get('sei_slices') is None for lg in x264)}/{len(x264)}. "
          f"Bitstream slices (h264) / video packets (mpeg4) per frame = 1 in "
          f"{sum(lg['slices_min'] == lg['slices_max'] == 1 for lg in ok)}/{len(ok)} encodes.",
          "- x264 SEI reports keyint_min=126 although min-keyint=250 is passed (x264 clamps it to keyint/2+1); "
          "irrelevant with scenecut=0 and no I-frame after frame 0 (same as v2)."]
    for eid in ("x264_qp24", "x264_qp24_bf2_pb1", "nvenc_qp28_bf2_bqeq", "nvenc_qp28_bf2", "x264_crf23_bf2"):
        qs = {}
        for lg in ok:
            if lg["encode_id"] == eid:
                for t, vals in lg.get("decoded_qp_values", {}).items():
                    qs.setdefault(t, set()).update(vals)
        L.append(f"- decoded per-MB QP range, {eid} (all sequences and passes run): "
                 + ", ".join(f"{t} {min(v)}–{max(v)}" for t, v in sorted(qs.items())))
    d1b = R[(R["group"] == "d1") & R["encode_id"].str.contains("bf2") & (R["tau"] == 1.0)]
    L += [f"- B arms: a trailing d = 1 P-frame group exists in {len(d1b)} arm × sequence × pass rows "
          f"(group sizes in frames: {sorted(d1b['n_group_frames'].unique().tolist())}); "
          "in results.csv, not used by any prediction."]
    per = pd.DataFrame(ok)
    tot = MAN.get("runtime_s")
    L += [f"- Runtime: {tot / 3600:.2f} h total ({tot:.0f} s), including sequence loading" if tot
          else "- Runtime: (manifest has no end time)",
          "  - per stage (sum of per-run times, excluding loading): "
          + ", ".join(f"{p}/{st} {g['run_s'].sum() / 60:.1f} min" for (p, st), g in per.groupby(["pass", "stage"])),
          f"  - slowest arm: {per.groupby('encode_id')['run_s'].sum().idxmax()} "
          f"({per.groupby('encode_id')['run_s'].sum().max() / 60:.1f} min over all sequences)", ""]
    open(os.path.join(HERE, "summary.md"), "w", encoding="utf-8").write("\n".join(L))
    print("wrote", os.path.join(HERE, "summary.md"))


if __name__ == "__main__":
    main()
