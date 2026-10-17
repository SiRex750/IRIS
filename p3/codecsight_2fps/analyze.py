"""CodecSight-matched 2 fps arm: analysis as fixed by PREREGISTRATION.md (cc2eaf9). Run only after checks.py.

Reuses, unchanged:
  p3/codecsight_sim/simulate.py   simulate_unit (native-rate path: token_matrix, native_refresh, cell_truth,
                                  policy_counts, bare_gate, ratios, gate_ratios); the 2 fps stream is "native" here
  p3/codecsight_sim/posthoc_matched/posthoc.py   owner, policy_tokens, cell_counts (common truth, cell level)
Only the two CCTV loaders are replaced (simulate.cctv_truth / cctv_encode read p3/virat_confirm/data): the
replacements have the same bodies with this arm's data folder. truth="fb" sets every FB-inconsistent cell-frame
(i08b test on splat_t and bwd_t) to invalid: g = NaN and cover False, so summed truth drops it (finite-terms rule).

Configurations (derived token grid, all scored frames 1..n-1): primary tau 0.5 (verdict), tau1 (secondary),
tau0.25 (descriptive); FB truth for primary and tau1 (descriptive).
Units for verdicts and medians: the clips counted in checks.csv (§5). R-chain overall verdict from checks.csv;
if it is FAIL, every output is labelled descriptive and no verdict is given (§3).
Writes per_unit.csv, common_truth.csv, verdicts.csv, verdicts.md, arm_medians.csv, bootstrap_ci.csv.
"""
import hashlib
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
SIM = os.path.join(P3, "codecsight_sim")
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(SIM, "posthoc_matched"))
import simulate as S  # noqa: E402
import posthoc as PH  # noqa: E402

PREREG_COMMIT = "cc2eaf9"
DATA = os.path.join(HERE, "data")
ARMS = S.CCTV_ARMS
CONFIGS = (("primary", "derived", 0.5, "native"), ("tau1", "derived", 1.0, "native"),
           ("tau0.25", "derived", 0.25, "native"))
FB_CONFIGS = CONFIGS[:2]
RULES = [("F1", "x264_crf45", "x264_crf12", 0.80), ("F2", "nvenc_qp45", "nvenc_qp18", 0.75),
         ("F3", "x264_qp24_bf2_pb1", "x264_qp24", 0.80), ("F4", "nvenc_qp28_bf2_bqeq", "nvenc_qp28", 0.75)]
MARGIN, KA_STALE, KA_REUSE, N_MIN = 0.01, 0.01, 0.20, 10
CODE_FILES = ("collect.py", "checks.py", "analyze.py", "independent_verdicts.py")
_FB = False


# --------------------------------------------------------------------------- loaders (simulate.py bodies, this data)
def fb_ok(rz):
    f = rz["splat_t"].astype(np.float32)
    b = rz["bwd_t"].astype(np.float32)
    s = f + b
    return (s ** 2).sum(-1) < 0.01 * ((f ** 2).sum(-1) + (b ** 2).sum(-1)) + 0.5


def cctv_truth(clip):
    rz = np.load(os.path.join(DATA, clip, "raft.npz"))
    sp = rz["splat_t"].astype(np.float32)
    nm1, Hc, Wc = sp.shape[:3]
    n, ncell = nm1 + 1, Hc * Wc
    cover_ok = np.zeros((n, ncell), bool)
    cover_ok[1:] = rz["splat_count"].reshape(nm1, ncell) >= S.COVER_COUNT
    gmag = np.full((n, ncell), np.nan, np.float32)
    gmag[1:] = np.hypot(sp[..., 0], sp[..., 1]).reshape(nm1, ncell)
    g = np.full((n, ncell, 2), np.nan)
    g[1:] = sp.reshape(nm1, ncell, 2)
    if _FB:
        bad = np.zeros((n, ncell), bool)
        bad[1:] = ~fb_ok(rz).reshape(nm1, ncell)
        g[bad] = np.nan
        cover_ok &= ~bad
        gmag[bad] = np.nan
    return g, cover_ok, gmag, Hc, Wc


def cctv_encode(clip, arm):
    z = np.load(os.path.join(DATA, clip, f"{arm}.npz"))
    q = z["mv_q"]
    n = q.shape[0]
    meta = json.loads(str(z["meta"]))
    return (q.reshape(n, -1), [str(x) for x in z["ftype"]], z["d_past"], bool(z["d_known"]), meta)


S.cctv_truth = cctv_truth
S.cctv_encode = cctv_encode


# --------------------------------------------------------------------------- per clip
def common_truth(clip, scene):
    """P, G1, G2 (derived) at tau 0.5 and 1 under one truth (posthoc_matched PLAN §2-4); MOVE_c per encode."""
    g, cover_ok, gmag, Hc, Wc = cctv_truth(clip)
    H, W = Hc * S.C, Wc * S.C
    n = cover_ok.shape[0]
    M = S.token_matrix(Hc, Wc, H, W, "derived")
    own = PH.owner(Hc, Wc, H, W, "derived")
    mv_num = int((cover_ok[1:] & (gmag[1:] > S.STALE_GT)).sum())
    mv_den = int(cover_ok[1:].sum())
    rows = []
    for arm in ARMS:
        q, ftype, _, _, _ = cctv_encode(clip, arm)
        ref = S.native_refresh([t for t, f in enumerate(ftype) if f == "I"], n)
        for tau in (0.5, 1.0):
            dyn = q >= 4 * tau
            td = S.tok_any(dyn, M)
            base = {"clip": clip, "scene": scene, "arm": arm, "tau": tau, "move_num": mv_num, "move_den": mv_den}
            rows.append({**base, "method": "G1", **PH.cell_counts(ref[:, None] | dyn, g, cover_ok)})
            rows.append({**base, "method": "G2", **PH.cell_counts((ref[:, None] | td)[:, own], g, cover_ok)})
            rows.append({**base, "method": "P", **PH.cell_counts(PH.policy_tokens(td, ref)[:, own], g, cover_ok)})
    return rows


def task(args):
    global _FB
    clip, scene = args
    t0 = time.perf_counter()
    try:
        _FB = False
        rows = [{**r, "truth": "raft"} for r in S.simulate_unit("CCTV", clip, scene, ARMS, CONFIGS)]
        ct = common_truth(clip, scene)
        _FB = True
        rows += [{**r, "truth": "fb"} for r in S.simulate_unit("CCTV", clip, scene, ARMS, FB_CONFIGS)]
        _FB = False
        return clip, rows, ct, time.perf_counter() - t0, None
    except Exception:
        return clip, [], [], time.perf_counter() - t0, traceback.format_exc()


# --------------------------------------------------------------------------- summaries
def need(n, x):
    return math.ceil(x * n - 1e-9)


def wide(df, metric):
    return df.pivot(index="unit", columns="arm", values=metric)


def f_rules(df, units):
    """F1-F4 on STALE_MOV over the given units."""
    w = wide(df, "STALE_MOV").reindex(units)
    out = []
    for key, hi, lo, share in RULES:
        d = w[hi] - w[lo]
        miss = d.index[d.isna()].tolist()
        cnt = int((d > MARGIN).sum())
        req = need(len(units), share)
        out.append({"key": key, "comparison": f"STALE_MOV({hi}) - STALE_MOV({lo})", "margin": f"> {MARGIN}",
                    "count": cnt, "units": len(units), "required": req, "share": share,
                    "result": "PASS" if cnt >= req else "FAIL", "median_diff": float(d.median()),
                    "missing": ";".join(miss) or "none"})
    return out


def arm_medians(df, units):
    rows = []
    for arm in ARMS:
        s = df[(df.arm == arm) & df.unit.isin(units)]
        r = {"arm": arm, "units": len(s), "REUSE": s.REUSE.median(), "STALE": s.STALE.median(),
             "STALE_MOV": s.STALE_MOV.median(), "STALE_MOV_missing": int(s.STALE_MOV.isna().sum())}
        for c in ("g_REUSE", "g_STALE", "g_STALE_MOV", "B_REUSE", "B_STALE", "B_STALE_MOV", "active_share_pre_refresh"):
            r[c] = s[c].median() if c in s and s[c].notna().any() else np.nan
        dv = s.dropped_finite_tf.sum() + s.valid_tf.sum()
        r["finite_dropped"] = s.dropped_finite_tf.sum() / dv if dv else np.nan
        r["stale_lt"] = bool(r["STALE_MOV"] < KA_STALE)
        r["reuse_gt"] = bool(r["REUSE"] > KA_REUSE)
        rows.append(r)
    return pd.DataFrame(rows)


def decision(am):
    fires = bool((am.stale_lt & am.reuse_gt).all())
    leak = am[(am.STALE_MOV > KA_STALE) & am.reuse_gt]
    little = am[~am.reuse_gt]
    if fires:
        out = "fresh (KILL-A2 fires)"
    elif len(leak):
        out = "leaks: " + ", ".join(leak.arm)
    else:
        out = "neither; reuses little: " + (", ".join(little.arm) or "none")
    return fires, out


def boot(df, units, scene_of, cfg):
    """Median per-unit paired diff with 95% percentile CIs; rng 20261006; scene draws, then clip draws."""
    rng = np.random.default_rng(20261006)
    scenes = sorted({scene_of[u] for u in units})
    sd = rng.integers(0, len(scenes), (10000, len(scenes)))
    cd = rng.integers(0, len(units), (10000, len(units)))
    w = wide(df, "STALE_MOV").reindex(units)
    rows = []
    for key, hi, lo, _ in RULES:
        d = (w[hi] - w[lo]).to_numpy()
        by_scene = [np.array([i for i, u in enumerate(units) if scene_of[u] == s]) for s in scenes]
        est = float(np.nanmedian(d))
        sc = np.array([np.nanmedian(d[np.concatenate([by_scene[j] for j in row])]) for row in sd])
        cl = np.array([np.nanmedian(d[row]) for row in cd])
        for meth, b in ((f"scene-cluster bootstrap ({len(scenes)} scenes, {len(units)} clips)", sc),
                        (f"clip-level bootstrap ({len(units)} clips)", cl)):
            lo_, hi_ = np.percentile(b, [2.5, 97.5])
            rows.append({"config": cfg, "key": key, "method": meth, "estimate": est, "ci_low": lo_, "ci_high": hi_})
    return rows


def md(df, fmt="{:.5f}"):
    cols = list(df.columns)
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        L.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v)
                                    else ("–" if isinstance(v, float) and pd.isna(v) else str(v)) for v in r) + " |")
    return L


def code_hash():
    h = hashlib.sha256()
    for f in CODE_FILES:
        p = os.path.join(HERE, f)
        h.update(f.encode() + b"\0" + (open(p, "rb").read() if os.path.isfile(p) else b"") + b"\0")
    for p in (os.path.join(SIM, "simulate.py"), os.path.join(SIM, "posthoc_matched", "posthoc.py")):
        h.update(open(p, "rb").read())
    return h.hexdigest()


def main():
    T0 = time.perf_counter()
    prereg_same = subprocess.run(["git", "diff", "--quiet", PREREG_COMMIT, "--", "p3/codecsight_2fps/PREREGISTRATION.md"],
                                 cwd=REPO).returncode == 0
    ck = pd.read_csv(os.path.join(HERE, "checks.csv"))
    man = json.load(open(os.path.join(HERE, "manifest.json")))
    clips = sorted(man["clips"])
    scene_of = {c: man["clips"][c]["scene"] for c in clips}
    units = sorted(ck.loc[ck.counted, "clip"])
    N = len(units)
    rc_pass = int(ck.loc[ck.counted, "chain_pass"].sum())
    rchain = "PASS" if rc_pass >= need(N, 0.75) else "FAIL"
    verdict_ok = rchain == "PASS" and N >= N_MIN
    label = "verdict" if verdict_ok else "descriptive (no verdict)"
    unseen = sorted(c for c in units if man["clips"][c]["start"] >= 600)

    rows, ct, errs = [], [], []
    with mp.get_context("spawn").Pool(6) as pool:
        for clip, r, c, dt, err in pool.imap_unordered(task, [(c, scene_of[c]) for c in clips]):
            print(f"{clip} {len(r)} rows {dt:.0f}s" + (f" ERROR\n{err}" if err else ""), flush=True)
            rows += r
            ct += c
            if err:
                errs.append(clip)
    if errs:
        sys.exit(f"errors: {errs}")
    pu = pd.DataFrame(rows).sort_values(["truth", "config", "arm", "unit"]).reset_index(drop=True)
    pu["prereg_commit"], pu["code_sha256"] = PREREG_COMMIT, code_hash()
    pu.to_csv(os.path.join(HERE, "per_unit.csv"), index=False)
    ctd = pd.DataFrame(ct)
    ctd["REUSE_c"] = ctd.reused_c / ctd.valid_c
    ctd["STALE_c"] = ctd.stale_c / ctd.valid_c
    ctd["MOVE_c"] = ctd.move_num / ctd.move_den
    ctd["STALE_c_per_MOVE_c"] = ctd.STALE_c / ctd.MOVE_c
    ctd.to_csv(os.path.join(HERE, "common_truth.csv"), index=False)

    def sel(cfg, truth="raft"):
        return pu[(pu.config == cfg) & (pu.truth == truth)]

    vrows, amrows, brows = [], [], []
    L = ["# CodecSight-matched 2 fps arm: verdicts", "",
         f"Prereg {PREREG_COMMIT} (file unchanged: {prereg_same}); code sha256 {code_hash()[:16]}; collection HEAD "
         f"{man['head'][:7]}. Generated by analyze.py from per_unit.csv ({len(pu)} rows).", "",
         f"- Counted clips (§5): N = {N}. R-chain: {rc_pass}/{N} (need {need(N, 0.75)}) -> **{rchain}**. "
         f"Status of everything below: **{label}**.",
         f"- Verdict metric: policy STALE_MOV, derived token grid, all scored frames. Medians over the {N} counted "
         f"clips. A missing value does not meet a condition.", ""]
    for cfg, lab in (("primary", label), ("tau1", "secondary" if verdict_ok else label)):
        d = sel(cfg)
        fr = f_rules(d, units)
        for r in fr:
            vrows.append({"config": cfg, "subset": "counted", **r})
        am = arm_medians(d, units)
        fires, out = decision(am)
        amrows += [{"config": cfg, "truth": "raft", "subset": "counted", **r} for r in am.to_dict("records")]
        vrows.append({"config": cfg, "subset": "counted", "key": "KILL-A2", "count": int((am.stale_lt & am.reuse_gt).sum()),
                      "units": 9, "required": 9, "result": "FIRES" if fires else "DOES NOT FIRE"})
        vrows.append({"config": cfg, "subset": "counted", "key": "decision", "result": out})
        brows += boot(d, units, scene_of, cfg)
        fx = {r["key"]: r["result"] for r in fr}
        L += [f"## F1-F4, {cfg} ({lab})", ""] + md(pd.DataFrame(fr)[["key", "comparison", "margin", "count", "required",
                                                                    "result", "median_diff", "missing"]]) + [
            "", f"Kill rules (mechanical): F1 and F2 both fail: {'yes' if fx['F1'] == fx['F2'] == 'FAIL' else 'no'}; "
                f"exactly one of F1, F2 fails: {'yes' if (fx['F1'] == 'FAIL') != (fx['F2'] == 'FAIL') else 'no'}; "
                f"F3 fails: {'yes' if fx['F3'] == 'FAIL' else 'no'}; F4 fails: {'yes' if fx['F4'] == 'FAIL' else 'no'}", "",
            f"### KILL-A2 and decision rule, {cfg} ({lab}): KILL-A2 {'FIRES' if fires else 'DOES NOT FIRE'} "
            f"({int((am.stale_lt & am.reuse_gt).sum())}/9); decision: **{out}**", ""]
        L += md(am[["arm", "units", "REUSE", "STALE_MOV", "STALE_MOV_missing", "g_REUSE", "g_STALE_MOV", "stale_lt",
                    "reuse_gt"]]) + [""]
        L += ["### Paired medians and 95% bootstrap CIs (descriptive)", ""] + md(
            pd.DataFrame([b for b in brows if b["config"] == cfg])[["key", "method", "estimate", "ci_low", "ci_high"]]) + [""]
    # unseen subgroup
    L += [f"## Unseen-window subgroup (descriptive): {len(unseen)} counted clips with start 600", ""]
    for cfg in ("primary", "tau1"):
        d = sel(cfg)
        fr = f_rules(d, unseen)
        am = arm_medians(d, unseen)
        fires, out = decision(am)
        for r in fr:
            vrows.append({"config": cfg, "subset": "unseen", **r})
        vrows.append({"config": cfg, "subset": "unseen", "key": "decision", "result": out})
        amrows += [{"config": cfg, "truth": "raft", "subset": "unseen", **r} for r in am.to_dict("records")]
        L += [f"### {cfg}", ""] + md(pd.DataFrame(fr)[["key", "count", "units", "required", "result", "median_diff"]])
        L += ["", f"KILL-A2 {'fires' if fires else 'does not fire'}; decision rule applied to this subgroup: {out}", ""]
        L += md(am[["arm", "units", "REUSE", "STALE_MOV"]]) + [""]
    # per scene
    L += ["## F1-F4 per scene, primary (descriptive)", ""]
    w = wide(sel("primary"), "STALE_MOV").reindex(units)
    ps = []
    for key, hi, lo, _ in RULES:
        d = w[hi] - w[lo]
        for sc in sorted({scene_of[u] for u in units}):
            dd = d[[u for u in units if scene_of[u] == sc]]
            ps.append({"key": key, "scene": sc, "clips": len(dd), "meeting > 0.01": int((dd > MARGIN).sum()),
                       "median diff": float(dd.median())})
    L += md(pd.DataFrame(ps)) + [""]
    # FB and tau 0.25
    for cfg, truth in (("primary", "fb"), ("tau1", "fb"), ("tau0.25", "raft")):
        d = sel(cfg, truth)
        fr = f_rules(d, units)
        am = arm_medians(d, units)
        fires, out = decision(am)
        for r in fr:
            vrows.append({"config": cfg, "subset": f"counted, {truth}", **r})
        vrows.append({"config": cfg, "subset": f"counted, {truth}", "key": "decision", "result": out})
        amrows += [{"config": cfg, "truth": truth, "subset": "counted", **r} for r in am.to_dict("records")]
        L += [f"## {cfg}, truth {truth} (descriptive)", ""] + md(pd.DataFrame(fr)[["key", "count", "required", "result",
                                                                                 "median_diff"]])
        L += ["", f"KILL-A2 {'fires' if fires else 'does not fire'}; decision rule: {out}", ""]
    amd = pd.DataFrame(amrows)
    amd.to_csv(os.path.join(HERE, "arm_medians.csv"), index=False)
    pd.DataFrame(vrows).to_csv(os.path.join(HERE, "verdicts.csv"), index=False)
    pd.DataFrame(brows).to_csv(os.path.join(HERE, "bootstrap_ci.csv"), index=False)
    # per-arm descriptive medians
    L += ["## Per-arm medians, counted clips (descriptive)", "",
          "Policy REUSE / STALE / STALE_MOV; bare per-cell gate; B-frames only (bf 2 arms); share of tokens active "
          "at the frame before each refresh; token-frames lost to the finite-terms rule (pooled).", ""]
    L += md(amd[amd.subset == "counted"][["config", "truth", "arm", "REUSE", "STALE", "STALE_MOV", "g_REUSE", "g_STALE",
                                         "g_STALE_MOV", "B_REUSE", "B_STALE", "B_STALE_MOV", "active_share_pre_refresh",
                                         "finite_dropped"]]) + [""]
    ctm = (ctd[ctd["clip"].isin(units)].groupby(["method", "tau", "arm"])[["REUSE_c", "STALE_c", "MOVE_c",
                                                                       "STALE_c_per_MOVE_c"]].median().reset_index())
    L += ["## Common-truth metrics, medians over counted clips (descriptive)", "",
          "MOVE_c = valid cell-frames with |T_j| > 2 px / valid cell-frames (same for every method on an encode). "
          "STALE_c / MOVE_c is not a share.", ""] + md(ctm) + [""]
    br = pd.DataFrame([{"arm": r["arm"], "clip": r["clip"], "kbps": r["bitrate_kbps"], "psnr": r["psnr_y_mean"],
                        "P_qp": r.get("mean_p_qp")} for r in man["runs"] if r.get("status") == "ok"])
    L += ["## Bitrate and PSNR per arm, medians over all clips (descriptive)", ""] + md(
        br.groupby("arm")[["kbps", "psnr", "P_qp"]].median().reset_index(), "{:.2f}") + [""]
    L.append(f"Runtime {time.perf_counter() - T0:.0f} s.")
    open(os.path.join(HERE, "verdicts.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
