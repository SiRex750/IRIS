"""Paper 3: confirmatory CCTV analysis, as fixed by p3/virat_confirm/PREREGISTRATION.md (commit d62753f).
Inputs: the collection run by p3/virat/run_virat.py into p3/virat_confirm/ (--start 300, 9 arms, single-threaded MV
extraction). Outputs in p3/virat_confirm/: results.csv (clip x arm x group x tau), eligibility_v0.csv, verdicts.csv,
summary.md.

The metric core is p3/virat/analyze_virat.py unchanged (frame_groups, gate_counts, finish, ref_decisions,
cell_texture, cell_vectors; selftest() re-checks gate_counts against the Sintel sweep). Added here:
  - STALE_MOV = stale_cells / moving_cells: valid cells reused (|MV| < tau) with |RAFT| > 2 px, divided by valid
    cells with |RAFT| > 2 px, per arm and frame group (d1 for bf 0 arms, B = naive B-frame vectors for B arms).
  - FLIP vs NVENC QP 18 (flip_nv18): share of in-frame cells of the arm's d1 frames (that are also d1 frames of
    nvenc_qp18) whose gate decision differs from nvenc_qp18. FLIP vs x264 CRF 12 (flip) is kept, descriptive.
  - source frames are frames start..start+n-1 of the clip (start from the collection manifest).
Implementation choices the pre-registration does not fix (decided before any confirmatory metric was computed):
  - moving cell-frames for eligibility: valid cells with |RAFT| > 2 px over all frame pairs t = 1..n-1 of the clip
    (RAFT only, before any gate metric), as the eligibility count of 45f3e98.
  - V0 as in 45f3e98: x264 CRF 12, d1 frames, valid cells with |RAFT| > 2 px, a vector present, texture >= the 75th
    percentile of cell texture over all cells of source frames 1..n-1 of all analysed clips; per clip median
    |MV - RAFT|; PASS if < 1.5 px in >= ceil(0.75 x eligible) clips. A clip with no V0 cell does not pass. If V0
    fails, C1-C4 carry no verdict (45f3e98: RAFT-based results unreliable); C5 (no RAFT) stands.
  - "x% of N clips" = at least ceil(x N) clips. A missing value counts as not meeting the condition (listed).
  - "fewer than 10 eligible -> all predictions descriptive only" includes C5.
  - tau = 1 px for all verdicts; 0.5 and 2 px recomputed, descriptive.
"""
import argparse
import csv
import glob
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
REPO = os.path.dirname(P3)
sys.path.insert(0, os.path.join(P3, "virat"))

import analyze_virat as av_  # noqa: E402
from analyze_virat import (frame_groups, gate_counts, finish, ref_decisions, cell_texture, cell_vectors,  # noqa: E402
                           md_table, scene_of, TAUS, STALE_GT, COVER_COUNT, TEX_BIN, TEX_MAX, V0_ERR, PIX)
from lib.mvs import extract_mvs  # noqa: E402
from run_virat import cell_arrays, decode_source, VIRAT  # noqa: E402

PREREG = "d62753f71986167271d7b9aab400cc589db4e5aa"
TAU = 1.0
ELIG_MIN_CELLFRAMES = 2000
MIN_ELIGIBLE = 10
DELTA = 0.01
REF_X, REF_NV = "x264_crf12", "nvenc_qp18"
ARMS = ("x264_crf12", "x264_crf45", "x264_qp24", "x264_qp24_bf2_pb1", "nvenc_qp18", "nvenc_qp23", "nvenc_qp45",
        "nvenc_qp28", "nvenc_qp28_bf2_bqeq")
# (key, label, high arm, high group, low arm, low group, share of eligible clips)
STALE_PREDS = (("C1", "C1 x264: STALE_MOV(CRF 45) - STALE_MOV(CRF 12) > 0.01", "x264_crf45", "d1", "x264_crf12", "d1", .80),
               ("C2", "C2 NVENC: STALE_MOV(QP 45) - STALE_MOV(QP 18) > 0.01", "nvenc_qp45", "d1", "nvenc_qp18", "d1", .75),
               ("C3", "C3 x264 B-frames: STALE_MOV(QP 24 B) - STALE_MOV(QP 24 bf 0) > 0.01", "x264_qp24_bf2_pb1", "B",
                "x264_qp24", "d1", .80),
               ("C4", "C4 NVENC B-frames: STALE_MOV(QP 28 B=P) - STALE_MOV(QP 28 bf 0) > 0.01", "nvenc_qp28_bf2_bqeq",
                "B", "nvenc_qp28", "d1", .75))
C5_LABEL = "C5 NVENC flip vs QP 18 (all clips): FLIP(QP 45) > FLIP(QP 23)"

ROOT = HERE
START = 0


def md(df, floatfmt="{:.4f}"):
    """analyze_virat.md_table with "|" escaped inside string cells."""
    return md_table(df.map(lambda x: x.replace("|", "\\|") if isinstance(x, str) else x), floatfmt)


def need(n, frac):
    return math.ceil(frac * n - 1e-9)


def load_arm(clip, arm):
    z = np.load(os.path.join(ROOT, "data", clip, f"{arm}.npz"))
    n = z["mv_q"].shape[0]
    return {"mv_q": z["mv_q"].reshape(n, -1), "ftype": [str(x) for x in z["ftype"]], "d_past": z["d_past"],
            "d_known": bool(z["d_known"]), "meta": json.loads(str(z["meta"]))}


def do_clip(clip):
    t0 = time.perf_counter()
    notes = []
    data = os.path.join(ROOT, "data", clip)
    arms = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(data, "*.npz"))
                  if os.path.basename(p) not in ("source.npz", "raft.npz"))
    missing = sorted(set(ARMS) - set(arms))
    if missing:
        notes.append(f"arms missing: {missing}")
    rz = np.load(os.path.join(data, "raft.npz"))
    sp = rz["splat_t"].astype(np.float32)
    nm1, Hc, Wc = sp.shape[:3]
    n, ncell = nm1 + 1, Hc * Wc
    valid = np.zeros((n, ncell), bool)
    gmag = np.full((n, ncell), np.nan, np.float32)
    valid[1:] = rz["splat_count"].reshape(nm1, ncell) >= COVER_COUNT
    gmag[1:] = np.hypot(sp[..., 0], sp[..., 1]).reshape(nm1, ncell)
    del sp
    n_valid = int(valid[1:].sum())
    n_mov = int((valid[1:] & (gmag[1:] > STALE_GT)).sum())

    # source frames start..start+n-1: texture of frame t, exact mean |Y_t - Y_(t-1)|
    src, _ = decode_source(os.path.join(VIRAT, clip + ".mp4"), n, start=START)
    H = src[0].shape[0] * 2 // 3
    tex = np.zeros((n, ncell), np.float32)
    absd = np.zeros((n, ncell), np.float32)
    yp = src[0][:H].astype(np.float64)
    for t in range(1, n):
        y = src[t][:H].astype(np.float64)
        tex[t] = cell_texture(y).ravel()
        absd[t] = np.abs(y - yp).reshape(H // av_.C, av_.C, -1, av_.C).mean(axis=(1, 3)).ravel()
        yp = y
    del src
    tex_hist = np.bincount(np.minimum((tex[1:] / TEX_BIN).astype(np.int64), int(TEX_MAX / TEX_BIN) - 1).ravel(),
                           minlength=int(TEX_MAX / TEX_BIN))
    stored = np.load(os.path.join(data, "source.npz"))["absdiff"].reshape(nm1, ncell)
    ad_mismatch = int((np.minimum(np.floor(absd[1:] + 0.5), 255) != stored).sum())
    pix = absd > PIX

    refs = {}
    for rid in (REF_X, REF_NV):
        if rid in arms:
            ra = load_arm(clip, rid)
            refs[rid] = ra, ref_decisions(ra["mv_q"], frame_groups(ra["ftype"], ra["d_past"], ra["d_known"]))
    rows = []
    for arm in arms:
        a = load_arm(clip, arm)
        if a["mv_q"].shape != (n, ncell):
            notes.append(f"{arm}: mv_q shape {a['mv_q'].shape} != {(n, ncell)}")
            continue
        grp = frame_groups(a["ftype"], a["d_past"], a["d_known"])
        rx = gate_counts(a["mv_q"], grp, valid, gmag, ref=refs[REF_X][1] if REF_X in refs else None, pix=pix)
        rn = gate_counts(a["mv_q"], grp, valid, gmag, ref=refs[REF_NV][1] if REF_NV in refs else None)
        for r, r2 in zip(rx, rn):
            assert (r["group"], r["tau"]) == (r2["group"], r2["tau"])
            r = finish(dict(r))
            r["stale_mov"] = r["stale_cells"] / r["moving_cells"] if r["moving_cells"] else np.nan
            r["flip_nv18_cells"], r["flip_nv18_den"] = r2.get("flip_cells"), r2.get("flip_den")
            r["flip_nv18"] = r2["flip_cells"] / r2["flip_den"] if r2.get("flip_den") else np.nan
            rows.append({"clip": clip, "arm": arm, "codec": a["meta"].get("codec"), "d_known": a["d_known"],
                         "n_source_frames": n, **r})

    # V0 cells: x264 CRF 12, d1 frames, valid, |RAFT| > 2, has a vector (re-extracted, single-threaded)
    v0_tex, v0_err, q_mismatch, v0_nan = np.zeros(0, np.float32), np.zeros(0, np.float32), None, None
    if REF_X in refs:
        ref_arm = refs[REF_X][0]
        fr, _ = extract_mvs(os.path.join(ROOT, "encodes", clip, f"{REF_X}.mp4"))
        vec = cell_vectors(fr, Hc, Wc)
        q_mismatch = int((cell_arrays(fr, Hc, Wc)["mv_q"].reshape(len(fr), -1) != ref_arm["mv_q"]).sum())
        del fr
        d1 = frame_groups(ref_arm["ftype"], ref_arm["d_past"], ref_arm["d_known"]).get("d1", np.array([], int))
        sel = np.zeros((n, ncell), bool)
        sel[d1] = valid[d1] & (gmag[d1] > STALE_GT) & (ref_arm["mv_q"][d1] != 255)
        flow = rz["splat_t"].reshape(nm1, ncell, 2)
        tt, cc = np.nonzero(sel)
        v0_err = np.hypot(vec[tt, cc, 0] - flow[tt - 1, cc, 0].astype(np.float32),
                          vec[tt, cc, 1] - flow[tt - 1, cc, 1].astype(np.float32)).astype(np.float32)
        v0_tex = tex[tt, cc]
        v0_nan = int(np.isnan(v0_err).sum())
    return {"clip": clip, "rows": rows, "n_frames": n, "valid_cellframes": n_valid, "moving_cellframes": n_mov,
            "tex_hist": tex_hist, "v0_tex": v0_tex, "v0_err": v0_err,
            "checks": {"absdiff_rounding_mismatch_cells": ad_mismatch, "crf12_q_repaint_mismatch_cells": q_mismatch,
                       "v0_nan_err": v0_nan, "n_arms": len(arms)},
            "notes": notes, "run_s": time.perf_counter() - t0}


def do_clip_safe(args):
    global ROOT, START
    clip, ROOT, START = args
    try:
        return do_clip(clip)
    except Exception as e:
        return {"clip": clip, "error": repr(e), "traceback": traceback.format_exc()}


# --------------------------------------------------------------------------- verdicts
def series(res, arm, group, col, tau=TAU):
    r = res[(res.tau == tau) & (res.arm == arm) & (res.group == group)]
    return r.set_index("clip")[col]


def verdicts(res, clips_all, elig, v0med, tau=TAU):
    """Mechanical verdicts. Returns (table, per-clip criterion frame)."""
    out, crit = [], pd.DataFrame(index=clips_all)
    e_ok = len(elig) >= MIN_ELIGIBLE
    v0_pass_clips = int((v0med.reindex(elig) < V0_ERR).sum())
    v0_pass = len(elig) > 0 and v0_pass_clips >= need(len(elig), 0.75)
    out.append({"key": "V0", "prediction": "V0 RAFT validity (45f3e98): median |MV - RAFT| < 1.5 px, x264 CRF 12, "
                                           "|RAFT| > 2 px, top-quartile texture", "n": len(elig),
                "value": f"{v0_pass_clips}/{len(elig)} eligible clips", "threshold": f">= 75% ({need(len(elig), .75)})",
                "verdict": "PASS" if v0_pass else "FAIL", "missing": ""})
    crit["V0 ok"] = v0med.reindex(clips_all) < V0_ERR

    def word(ok, raft, n):
        tag = "" if n == 0 else (" [would PASS]" if ok else " [would FAIL]")
        if not e_ok:
            return f"descriptive only (< {MIN_ELIGIBLE} eligible){tag}"
        if raft and not v0_pass:
            return "no verdict (V0 failed)" + tag
        return "PASS" if ok else "FAIL"

    for key, lab, hi, ghi, lo, glo, frac in STALE_PREDS:
        d = (series(res, hi, ghi, "stale_mov", tau) - series(res, lo, glo, "stale_mov", tau)).reindex(clips_all)
        crit[f"{key} diff"] = d
        crit[key] = d > DELTA
        de = d.reindex(elig)
        k = int((de > DELTA).sum())
        miss = [c for c in elig if pd.isna(de[c])]
        out.append({"key": key, "prediction": lab, "n": len(elig),
                    "value": f"{k}/{len(elig)} eligible clips (median diff {de.median():.4f})",
                    "threshold": f">= {int(frac * 100)}% ({need(len(elig), frac)})",
                    "verdict": word(k >= need(len(elig), frac), True, len(elig)), "missing": ", ".join(miss)})
    f45 = series(res, "nvenc_qp45", "d1", "flip_nv18", tau).reindex(clips_all)
    f23 = series(res, "nvenc_qp23", "d1", "flip_nv18", tau).reindex(clips_all)
    crit["C5 diff"] = f45 - f23
    crit["C5"] = f45 > f23
    k = int(crit["C5"].sum())
    N = len(clips_all)
    out.append({"key": "C5", "prediction": C5_LABEL, "n": N,
                "value": f"{k}/{N} clips (median FLIP QP 23 {f23.median():.4f}, QP 45 {f45.median():.4f})",
                "threshold": f">= 75% ({need(N, .75)})", "verdict": word(k >= need(N, .75), False, N),
                "missing": ", ".join(c for c in clips_all if pd.isna(f45[c]) or pd.isna(f23[c]))})
    return pd.DataFrame(out), crit


def recount(results_csv, clips_all, elig):
    """Second code path (csv module, no pandas): the C1-C5 clip counts from results.csv as written."""
    I = {}
    with open(results_csv, newline="") as fh:
        for x in csv.DictReader(fh):
            if float(x["tau"]) == TAU:
                I[(x["clip"], x["arm"], x["group"])] = x

    def g(c, a, grp, k):
        x = I.get((c, a, grp))
        if x is None or x[k] in ("", "nan"):
            return None
        return float(x[k])

    def sm(c, a, grp):
        x = I.get((c, a, grp))
        if x is None or float(x["moving_cells"]) == 0:
            return None
        return float(x["stale_cells"]) / float(x["moving_cells"])

    out = {}
    for key, _, hi, ghi, lo, glo, _ in STALE_PREDS:
        out[key] = sum(1 for c in elig if sm(c, hi, ghi) is not None and sm(c, lo, glo) is not None
                       and sm(c, hi, ghi) - sm(c, lo, glo) > DELTA)
    out["C5"] = sum(1 for c in clips_all if g(c, "nvenc_qp45", "d1", "flip_nv18") is not None
                    and g(c, "nvenc_qp23", "d1", "flip_nv18") is not None
                    and g(c, "nvenc_qp45", "d1", "flip_nv18") > g(c, "nvenc_qp23", "d1", "flip_nv18"))
    return out


def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main():
    global ROOT, START
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=HERE, help="collection directory (data/, encodes/, manifest.json); outputs go here")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--no-selftest", action="store_true", help="skip the Sintel gate_counts self-test (smoke tests)")
    a = ap.parse_args()
    ROOT = a.root
    T0 = time.perf_counter()
    st = None
    if not a.no_selftest:
        print("self-test on Sintel (alley_1) ...", flush=True)
        st = av_.selftest()
        print(f"self-test: {st['values_checked']} counters checked, {len(st['mismatches'])} unexplained mismatches",
              flush=True)
    t_self = time.perf_counter() - T0

    man = json.load(open(os.path.join(ROOT, "manifest.json")))
    START = int(man["config"].get("start", 0))
    clips = sorted(man["clips"])
    raft_ok = {c for c, r in man["raft"]["clips"].items() if r.get("status") in ("ok", "skipped_exists")}
    outs, fails = {}, []
    for c in clips:
        if c not in raft_ok:
            fails.append({"clip": c, "error": f"no RAFT (status {man['raft']['clips'].get(c, {}).get('status')})"})
    todo = [c for c in clips if c in raft_ok]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for o in pool.imap_unordered(do_clip_safe, [(c, ROOT, START) for c in todo]):
            if "error" in o:
                fails.append(o)
                print(f"FAILED {o['clip']}: {o['error']}", flush=True)
            else:
                outs[o["clip"]] = o
                print(f"{o['clip']} done in {o['run_s']:.0f}s", flush=True)
    ok_clips = [c for c in clips if c in outs]

    el = pd.DataFrame({"clip": ok_clips, "scene": [scene_of(c) for c in ok_clips],
                       "n_frames": [outs[c]["n_frames"] for c in ok_clips],
                       "valid_cellframes": [outs[c]["valid_cellframes"] for c in ok_clips],
                       "moving_cellframes": [outs[c]["moving_cellframes"] for c in ok_clips]})
    el["eligible"] = el["moving_cellframes"] >= ELIG_MIN_CELLFRAMES
    elig = el.loc[el.eligible, "clip"].tolist()

    hist = sum(outs[c]["tex_hist"] for c in ok_clips)
    cum = np.cumsum(hist)
    q75 = (np.searchsorted(cum, 0.75 * cum[-1]) + 1) * TEX_BIN
    v0 = []
    for c in ok_clips:
        m = outs[c]["v0_tex"] >= q75
        e = outs[c]["v0_err"][m]
        v0.append({"clip": c, "v0_candidate_cells": int(outs[c]["v0_err"].size), "v0_cells": int(m.sum()),
                   "v0_median_err_px": float(np.nanmedian(e)) if e.size else np.nan})
    v0 = pd.DataFrame(v0)
    v0med = v0.set_index("clip")["v0_median_err_px"]
    el.merge(v0, on="clip").to_csv(os.path.join(ROOT, "eligibility_v0.csv"), index=False)

    res = pd.DataFrame([r for c in ok_clips for r in outs[c]["rows"]])
    res.insert(1, "scene", res["clip"].map(scene_of))
    res.insert(2, "eligible", res["clip"].isin(elig))
    res_p = os.path.join(ROOT, "results.csv")
    res.to_csv(res_p, index=False)
    vt, crit = verdicts(res, ok_clips, elig, v0med)
    vt.to_csv(os.path.join(ROOT, "verdicts.csv"), index=False)
    rc = recount(res_p, ok_clips, elig)
    rc_ok = all(int(vt.set_index("key").loc[k, "value"].split("/")[0]) == v for k, v in rc.items())

    # ---------------- summary.md
    L = [f"# Paper 3 — confirmatory CCTV test (VIRAT frames {START}-{START + 299}, or to the end of the clip)\n",
         f"Pre-registration `{PREREG}` (p3/virat_confirm/PREREGISTRATION.md). Data: runner commit "
         f"`{man['runner_commit']}`{' (uncommitted: ' + man['runner_uncommitted_changes'] + ')' if man.get('runner_uncommitted_changes') else ''}, "
         f"collected {man['start_time']} → {man.get('end_time')}, source frames from {START}. Analysis run at "
         f"`{git('rev-parse', 'HEAD')}`"
         f"{' (uncommitted: ' + git('status', '--porcelain', '--', 'p3/virat_confirm/analyze_confirm.py', 'p3/virat/analyze_virat.py') + ')' if git('status', '--porcelain', '--', 'p3/virat_confirm/analyze_confirm.py', 'p3/virat/analyze_virat.py') else ''}.\n",
         "Computed mechanically by analyze_confirm.py. All flow quantities are RAFT-estimated (torchvision raft_large), "
         "not ground truth. tau = 1 px. STALE_MOV = stale cells / valid cells with |RAFT| > 2 px. Values are per clip; "
         "no pooled means.\n",
         "## 1. Verdicts (C1-C5, tau = 1 px)\n",
         md(vt.drop(columns=["key"])), "",
         f"Independent recount of the clip counts from results.csv (csv module): {rc} → "
         f"{'agrees with the table' if rc_ok else 'DISAGREES with the table'}.\n"]
    v = vt.set_index("key")["verdict"]
    kill = []
    if v["C1"] == "FAIL" and v["C2"] == "FAIL":
        kill.append("C1 and C2 both fail → the quality effect does not carry to CCTV.")
    if v["C3"] == "FAIL":
        kill.append("C3 fails → the B-frame claim is Sintel-only for x264.")
    if v["C4"] == "FAIL":
        kill.append("C4 fails → the B-frame claim is Sintel-only for NVENC.")
    if v["C5"] == "FAIL":
        kill.append("C5 fails → NVENC decisions do not track quality on CCTV.")
    nov = [k for k in ("C1", "C2", "C3", "C4", "C5") if v[k] not in ("PASS", "FAIL")]
    L += ["Kill rules triggered (mechanical):"] + [f"- {k}" for k in kill or ["none"]]
    if nov:
        L.append(f"- Without a verdict (kill rules not applicable): {', '.join(nov)}.")
    L += ["", "## 2. Clips, eligibility, V0\n",
          f"- Analysed clips: {len(ok_clips)}. Excluded before collection: "
          + ("; ".join(f"{c} ({r})" for c, r in sorted(man.get("excluded_clips", {}).items())) or "none") + ".",
          f"- Clip failures: {len(fails)}" + (": " + "; ".join(f"{f['clip']} {f['error']}" for f in fails) if fails else "") + ".",
          f"- Eligible (>= {ELIG_MIN_CELLFRAMES} moving cell-frames): **{len(elig)}/{len(ok_clips)}**"
          f" ({'>= ' + str(MIN_ELIGIBLE) + ': predictions carry verdicts' if len(elig) >= MIN_ELIGIBLE else '< ' + str(MIN_ELIGIBLE) + ': all predictions descriptive only'}).",
          f"- Ineligible: {', '.join(c for c in ok_clips if c not in elig) or 'none'}.",
          f"- V0 texture threshold (75th percentile over all cells of source frames 1..n-1 of the {len(ok_clips)} "
          f"clips): {q75:.3f} grey levels/px. V0: {vt.set_index('key').loc['V0', 'value']} → **{v['V0']}**.", ""]

    # per clip
    pc = pd.DataFrame(index=ok_clips)
    pc["scene"] = [scene_of(c) for c in ok_clips]
    pc["frames"] = el.set_index("clip")["n_frames"].reindex(ok_clips)
    pc["moving cf"] = el.set_index("clip")["moving_cellframes"].reindex(ok_clips)
    pc["elig"] = pc.index.isin(elig)
    pc["V0 err"] = v0med.reindex(ok_clips)
    for lab, arm, g in (("CRF12", "x264_crf12", "d1"), ("CRF45", "x264_crf45", "d1"), ("NV18", "nvenc_qp18", "d1"),
                        ("NV45", "nvenc_qp45", "d1"), ("QP24 bf0", "x264_qp24", "d1"),
                        ("QP24 B", "x264_qp24_bf2_pb1", "B"), ("NV28 bf0", "nvenc_qp28", "d1"),
                        ("NV28 B=P", "nvenc_qp28_bf2_bqeq", "B")):
        pc[lab] = series(res, arm, g, "stale_mov").reindex(ok_clips)
    for k in ("C1", "C2", "C3", "C4"):
        pc[f"{k} diff"] = crit[f"{k} diff"]
    L += ["## 3. Per-clip results (primary)\n", "### STALE_MOV per clip (tau = 1 px)\n",
          md(pc.reset_index().rename(columns={"index": "clip"}), "{:.4f}"), ""]
    fc = pd.DataFrame(index=ok_clips)
    fc["scene"] = pc["scene"]
    for q in (23, 28, 45):
        fc[f"FLIP NV{q} vs NV18"] = series(res, f"nvenc_qp{q}", "d1", "flip_nv18").reindex(ok_clips)
    fc["C5 met"] = crit["C5"]
    fc["FLIP CRF45 vs CRF12"] = series(res, "x264_crf45", "d1", "flip").reindex(ok_clips)
    L += ["### FLIP per clip (d1 frames, tau = 1 px)\n", md(fc.reset_index().rename(columns={"index": "clip"})), ""]

    # per scene
    rows = []
    for s, d in crit.assign(scene=[scene_of(c) for c in ok_clips], elig=pc["elig"]).groupby("scene"):
        e = d[d.elig]
        row = {"scene": s, "clips": len(d), "eligible": len(e)}
        for k in ("V0 ok", "C1", "C2", "C3", "C4"):
            row[k] = f"{int(e[k].sum())}/{len(e)}"
        row["C5"] = f"{int(d['C5'].sum())}/{len(d)}"
        for k in ("C1", "C2", "C3", "C4"):
            row[f"med {k} diff (elig)"] = e[f"{k} diff"].median() if len(e) else np.nan
        row["med C5 diff"] = d["C5 diff"].median()
        rows.append(row)
    L += ["## 4. Per-scene results (primary; C1-C4 over the scene's eligible clips, C5 over all its clips)\n",
          md(pd.DataFrame(rows), "{:.4f}"), ""]

    # descriptive
    L += ["## 5. Descriptive (no verdicts)\n", "### The verdict quantities at tau = 0.5 and 2 px\n"]
    for tau in (0.5, 2.0):
        vx, _ = verdicts(res, ok_clips, elig, v0med, tau=tau)
        L += [f"tau = {tau}:\n", md(vx.iloc[1:].drop(columns=["key", "verdict", "missing"])), ""]
    r1 = res[res.tau == TAU]
    med = []
    for arm in ARMS:
        for g, s in r1[r1.arm == arm].groupby("group"):
            se = s[s["clip"].isin(elig)]
            med.append({"arm": arm, "group": g, "clips": len(s), "med STALE_MOV (elig)": se["stale_mov"].median(),
                        "med stale/valid": s["stale"].median(), "med FLIP vs CRF12": s["flip"].median(),
                        "med FLIP vs NV18": s["flip_nv18"].median(), "med missed": s["missed"].median(),
                        "med reuse share": s["reuse_share_inframe"].median()})
    L += ["### Medians over clips, every arm and frame group (tau = 1 px)\n", md(pd.DataFrame(med), "{:.4f}"), ""]

    chk = pd.DataFrame([{"clip": c, **outs[c]["checks"]} for c in ok_clips])
    L += ["## 6. Checks and runtime\n"]
    if st is not None:
        L.append(f"- Metric-core self-test (analyze_virat.selftest: gate_counts on Sintel alley_1 sweep encodes vs "
                 f"p3/sweep/results.csv): {st['values_checked']} counters, {len(st['mismatches'])} unexplained "
                 f"mismatches{': ' + str(st['mismatches'][:5]) if st['mismatches'] else ''}; {len(st['expected'])} "
                 "explained (" + "; ".join(sorted({f'{m[0]} {m[1]}: {m[-1]}' for m in st['expected']})) + ").")
    else:
        L.append("- Metric-core self-test: skipped (--no-selftest).")
    L += [f"- Arms per clip: {sorted(chk.n_arms.unique().tolist())}.",
          f"- x264 CRF 12 vectors re-extracted for V0 vs stored q, mismatching cells: "
          f"{int(chk.crf12_q_repaint_mismatch_cells.fillna(0).sum())}.",
          f"- Source mean |ΔY| recomputed vs stored source.npz (checks the frame offset), mismatching cells: "
          f"{int(chk.absdiff_rounding_mismatch_cells.sum())}.",
          f"- V0 cells with NaN error: {int(chk.v0_nan_err.fillna(0).sum())}; clips with no V0 cell: "
          f"{', '.join(v0.loc[v0.v0_cells == 0, 'clip']) or 'none'}.",
          "- Groups per arm: " + "; ".join(f"{arm}: {', '.join(sorted(r1[r1.arm == arm].group.unique()))}"
                                          for arm in ARMS if arm in set(r1.arm)) + ".",
          f"- Notes: {'; '.join(f'{c}: {n}' for c in ok_clips for n in outs[c]['notes']) or 'none'}.",
          f"- Runtime: self-test {t_self:.0f}s; total {time.perf_counter() - T0:.0f}s with {a.workers} workers."]
    with open(os.path.join(ROOT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(vt.drop(columns=["missing"]).to_string())
    print(f"recount {rc} agrees: {rc_ok}")
    print(f"done in {time.perf_counter() - T0:.0f}s")


if __name__ == "__main__":
    main()
