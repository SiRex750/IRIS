"""Native-rate encode, 2 fps sampling (reading B): analysis as fixed by PREREGISTRATION.md (amended, ed606f4).

Pre-registered order, each stage written to disk before the next starts:
  1. Counting rule (§6, RAFT only): valid cell-frames (cover >= 8, finite) with |T_i| > 2 px over native pairs
     1..L-1; counted if >= 2,000.                                                      -> gates.csv (count columns)
  2. V0 (§5): x264 CRF 12, native d1 frames, valid, |RAFT| > 2 px, a vector, texture >= the 75th percentile of
     cell texture pooled over all 27 windows; per-clip median |MV - RAFT| < 1.5 px in >= ceil(0.75 N) counted clips.
  3. Drift gate (§3): false-move rate <= 0.01 in >= ceil(0.75 N) counted clips.         -> gates.csv, gates.md
     Also the descriptive drift-by-lag report (§7).                                     -> drift_by_lag.csv
  4. Policy (§5), variants (a) and (b), tau 0.5 / 1 / 0.25, then F1-F4, KILL-A, decision per variant (§7).
     Verdict wording only if V0 PASS, drift gate PASS and N >= 10; otherwise every number is descriptive under the
     heading(s) "RAFT unreliable" / "truth drift too large".   -> per_unit.csv, common_truth.csv, verdicts.csv,
                                                                  verdicts.md, arm_medians.csv, bootstrap_ci.csv
Reuses, unchanged: p3/codecsight_sim/simulate.py simulate_unit with its 2 FPS path (fps2_schedule; variant a:
contrib = sampled frames, variant b: every native frame; truth summed over every native frame; sampled frames
1..n-1 scored), and posthoc_matched/posthoc.py (owner, accumulate) for the common-truth metrics. Only the CCTV loaders
are replaced (same bodies, this arm's data folder), as in p3/codecsight_2fps/analyze.py.
Descriptive bare per-cell gate and common truth: on sampled frames only; per-frame truth for the bare gate is the
native pair s_j - 1 -> s_j (the pair the vectors of s_j describe).

Usage: python analyze.py [--workers N] [--stage gates|all]
"""
import argparse
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
sys.path.insert(0, HERE)
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(SIM, "posthoc_matched"))
import simulate as S  # noqa: E402
import posthoc as PH  # noqa: E402
import gates as G  # noqa: E402

PREREG_COMMIT = "ed606f4"
DATA = os.path.join(HERE, "data")
ARMS = S.CCTV_ARMS
GOP = 16
CONFIGS = {"a": (("primary", "derived", 0.5, "2fps_a"), ("tau1", "derived", 1.0, "2fps_a"),
                 ("tau0.25", "derived", 0.25, "2fps_a")),
           "b": (("primary", "derived", 0.5, "2fps_b"), ("tau1", "derived", 1.0, "2fps_b"),
                 ("tau0.25", "derived", 0.25, "2fps_b"))}
RULES = [("F1", "x264_crf45", "x264_crf12", 0.80), ("F2", "nvenc_qp45", "nvenc_qp18", 0.75),
         ("F3", "x264_qp24_bf2_pb1", "x264_qp24", 0.80), ("F4", "nvenc_qp28_bf2_bqeq", "nvenc_qp28", 0.75)]
MARGIN, KA_STALE, KA_REUSE, N_MIN, COUNT_MIN = 0.01, 0.01, 0.20, 10, 2000
CODE_FILES = ("collect.py", "gates.py", "analyze.py")


# --------------------------------------------------------------------------- loaders (simulate.py bodies, this data)
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
    return g, cover_ok, gmag, Hc, Wc


def cctv_encode(clip, arm):
    z = np.load(os.path.join(DATA, clip, f"{arm}.npz"))
    q = z["mv_q"]
    n = q.shape[0]
    meta = json.loads(str(z["meta"]))
    return (q.reshape(n, -1), [str(x) for x in z["ftype"]], z["d_past"], bool(z["d_known"]), meta)


S.cctv_truth = cctv_truth
S.cctv_encode = cctv_encode


def windows():
    inv = pd.read_csv(os.path.join(HERE, "inventory.csv"))
    return {r["clip"]: {"scene": r["scene"], "k": int(r["k"]), "keyint": int(r["keyint"]), "start": int(r["win_first"]),
                        "L": int(r["native_frames"]), "gops": int(r["gops"]), "unseen": r["rule"] == "unseen"}
            for _, r in inv.iterrows()}


# --------------------------------------------------------------------------- stage 1-3: RAFT-only gates, per clip
def gate_task(args):
    clip, w = args
    t0 = time.perf_counter()
    try:
        T, ok, Hc, Wc = G.load_truth(clip, HERE)
        assert T.shape[0] == w["L"] - 1, (clip, T.shape, w["L"])
        mag = np.hypot(T[..., 0], T[..., 1])
        row = {"clip": clip, "scene": w["scene"], "k": w["k"], "keyint": w["keyint"], "gops": w["gops"],
               "unseen": w["unseen"], "native_pairs": w["L"] - 1,
               "valid_cellframes": int(ok.sum()), "moving_cellframes": int((ok & (mag > S.STALE_GT)).sum())}
        q, f = G.drift_false_moves(T, ok, w["k"], w["keyint"])
        row.update(quiet_cellframes=q, false_move_cellframes=f)
        lag = {"clip": clip, "keyint": w["keyint"], **G.drift_by_lag(T, ok, w["keyint"])}
        del T, ok, mag
        v = G.v0_clip(clip, HERE, w["start"], w["L"])
        row.update(v0_cells=int(v["v0_err"].size), v0_nan_err=int(np.isnan(v["v0_err"]).sum()),
                   crf12_q_repaint_mismatch=v["q_mismatch"])
        return clip, row, lag, v, time.perf_counter() - t0, None
    except Exception:
        return clip, None, None, None, time.perf_counter() - t0, traceback.format_exc()


def need(n, x):
    return math.ceil(x * n - 1e-9)


def run_gates(clips, W_, workers):
    rows, lags, v0 = [], [], {}
    with mp.get_context("spawn").Pool(workers) as pool:
        for clip, r, lg, v, dt, err in pool.imap_unordered(gate_task, [(c, W_[c]) for c in clips]):
            print(f"gates {clip} {dt:.0f}s" + (f" ERROR\n{err}" if err else ""), flush=True)
            if err:
                sys.exit(f"gate stage failed on {clip}")
            rows.append(r)
            lags.append(lg)
            v0[clip] = v
    gt = pd.DataFrame(rows).sort_values("clip").reset_index(drop=True)
    # 1. counting rule
    gt["counted"] = gt.moving_cellframes >= COUNT_MIN
    units = gt.loc[gt.counted, "clip"].tolist()
    N = len(units)
    # 2. V0: texture threshold pooled over all 27 windows, pass over the counted clips
    q75 = G.tex_q75([v0[c]["tex_hist"] for c in clips])
    gt["v0_median_err"] = [G.v0_median(v0[c]["v0_tex"], v0[c]["v0_err"], q75) for c in gt["clip"]]
    gt["v0_cells_textured"] = [int((v0[c]["v0_tex"] >= q75).sum()) for c in gt["clip"]]
    gt["v0_pass"] = gt.v0_median_err < G.V0_ERR  # NaN (no V0 cell) does not pass
    # 3. drift gate
    gt["false_move_rate"] = gt.false_move_cellframes / gt.quiet_cellframes.where(gt.quiet_cellframes > 0)
    gt["drift_pass"] = gt.false_move_rate <= G.DRIFT_RATE  # NaN does not pass
    c = gt[gt.counted]
    req = need(N, 0.75)
    v0n, drn = int(c.v0_pass.sum()), int(c.drift_pass.sum())
    res = {"N": N, "required": req, "q75": float(q75), "v0_pass_clips": v0n, "drift_pass_clips": drn,
           "V0": "PASS" if v0n >= req else "FAIL", "drift": "PASS" if drn >= req else "FAIL",
           "not_counted": gt.loc[~gt.counted, "clip"].tolist(),
           "v0_missing": c.loc[c.v0_median_err.isna(), "clip"].tolist(),
           "drift_missing": c.loc[c.false_move_rate.isna(), "clip"].tolist()}
    gt.to_csv(os.path.join(HERE, "gates.csv"), index=False)
    pd.DataFrame(lags).sort_values("clip").to_csv(os.path.join(HERE, "drift_by_lag.csv"), index=False)
    json.dump(res, open(os.path.join(HERE, "gates.json"), "w"), indent=1)
    L = ["# Reading B: counting rule, V0, drift gate", "",
         f"Prereg {PREREG_COMMIT}; generated by analyze.py stage 1-3 before any policy metric.", "",
         f"- Counting rule (§6): {N}/{len(clips)} clips counted (moving valid cell-frames >= {COUNT_MIN}). "
         f"Not counted: {', '.join(res['not_counted']) or 'none'}.",
         f"- V0 (§5): texture threshold {q75:.3f} grey levels/px (75th percentile, all {len(clips)} windows); "
         f"{v0n}/{N} counted clips < {G.V0_ERR} px (need {req}) -> **{res['V0']}**. "
         f"No V0 cell: {', '.join(res['v0_missing']) or 'none'}.",
         f"- Drift gate (§3): {drn}/{N} counted clips with false-move rate <= {G.DRIFT_RATE} (need {req}) -> "
         f"**{res['drift']}**. No quiet cell-frame: {', '.join(res['drift_missing']) or 'none'}.", ""]
    L += md(gt[["clip", "k", "gops", "unseen", "moving_cellframes", "counted", "v0_cells_textured", "v0_median_err",
                "v0_pass", "quiet_cellframes", "false_move_cellframes", "false_move_rate", "drift_pass"]])
    open(os.path.join(HERE, "gates.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:8]), flush=True)
    return gt, res


# --------------------------------------------------------------------------- stage 4: policy, per clip
def sampled_common(clip, scene, w):
    """Bare per-cell gate and common-truth metrics (P, G1, G2; derived; tau 0.5 and 1) on the sampled frames.
    Nothing is computed at skipped frames, so the common truth keeps summing through them."""
    g, cover_ok, gmag, Hc, Wc = cctv_truth(clip)
    H, W = Hc * S.C, Wc * S.C
    n = cover_ok.shape[0]
    M = S.token_matrix(Hc, Wc, H, W, "derived")
    own = PH.owner(Hc, Wc, H, W, "derived")
    smask = np.zeros(n, bool)
    smask[::w["k"]] = True
    sc = np.nonzero(smask & (np.arange(n) >= 1))[0]
    mv_num = int((cover_ok[sc] & (gmag[sc] > S.STALE_GT)).sum())
    mv_den = int(cover_ok[sc].sum())
    rows, gate = [], []
    for arm in ARMS:
        q, ftype, _, _, _ = cctv_encode(clip, arm)
        iframes = [t for t, f in enumerate(ftype) if f == "I"]
        _, ref = S.fps2_schedule(iframes, n, w["k"])
        for tau in (0.5, 1.0):
            dyn = q >= 4 * tau
            gate.append({"unit": clip, "arm": arm, "tau": tau, **S.bare_gate(~dyn, cover_ok, gmag, sc)})
            td = S.tok_any(dyn, M)
            base = {"clip": clip, "scene": scene, "arm": arm, "tau": tau, "move_num": mv_num, "move_den": mv_den}
            for meth, comp in (("G1", ref[:, None] | dyn), ("G2", (ref[:, None] | td)[:, own])):
                comp = comp & smask[:, None]
                rows.append({**base, "method": meth, **cell_counts_sampled(comp, g, cover_ok, sc)})
            for var, contrib in (("a", smask), ("b", np.ones(n, bool))):
                tdv = td & (contrib & ~ref)[:, None]
                comp = PH.policy_tokens(tdv, ref)[:, own] & smask[:, None]
                rows.append({**base, "method": f"P_{var}", **cell_counts_sampled(comp, g, cover_ok, sc)})
    return rows, gate


def cell_counts_sampled(comp, g, cover_ok, sc):
    valid, moved = PH.accumulate(g, cover_ok, comp, comp)
    reu = ~comp[sc]
    v = valid[sc]
    return {"valid_c": int(v.sum()), "reused_c": int((reu & v).sum()), "stale_c": int((reu & moved[sc]).sum())}


def policy_task(args):
    clip, scene, w = args
    t0 = time.perf_counter()
    try:
        n = w["L"]
        for arm in ARMS:  # schedule assertions (§4, §5)
            q, ftype, _, _, meta = cctv_encode(clip, arm)
            iframes = [t for t, f in enumerate(ftype) if f == "I"]
            assert q.shape[0] == n, (clip, arm, q.shape)
            assert S.fps_step(meta["fps"]) == w["k"], (clip, arm, meta["fps"])
            assert iframes == list(range(0, n, w["keyint"])), (clip, arm, iframes)
            assert all(t % w["k"] == 0 for t in iframes), (clip, arm, "I-frame at a skipped frame")
            smask, ref = S.fps2_schedule(iframes, n, w["k"])
            assert np.nonzero(ref)[0].tolist() == list(range(0, n, w["keyint"])), (clip, arm, "refresh schedule")
        rows = []
        for var, cfgs in CONFIGS.items():
            rows += [{**r, "variant": var} for r in S.simulate_unit("CCTV", clip, scene, ARMS, cfgs)]
        ct, gate = sampled_common(clip, scene, w)
        gk = {(r["arm"], r["tau"]): r for r in gate}
        for r in rows:
            gr = gk.get((r["arm"], r["tau"]))
            if gr:
                r.update({k: v for k, v in gr.items() if k.startswith("g_")})
                r.update(S.gate_ratios(gr))
        return clip, rows, ct, time.perf_counter() - t0, None
    except Exception:
        return clip, [], [], time.perf_counter() - t0, traceback.format_exc()


# --------------------------------------------------------------------------- summaries
def wide(df, metric):
    return df.pivot(index="unit", columns="arm", values=metric)


def f_rules(df, units):
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
        # a missing median does not meet a condition (§6)
        r["stale_lt"] = bool(r["STALE_MOV"] < KA_STALE)
        r["reuse_gt"] = bool(r["REUSE"] > KA_REUSE)
        rows.append(r)
    return pd.DataFrame(rows)


def decision(am):
    fires = bool((am.stale_lt & am.reuse_gt).all())
    leak = am[(am.STALE_MOV > KA_STALE) & am.reuse_gt]
    little = am[~am.reuse_gt]
    if fires:
        out = "the policy stays fresh (KILL-A fires)"
    elif len(leak):
        out = "it leaks: " + ", ".join(leak.arm)
    else:
        out = "neither; policy reuses little: " + (", ".join(little.arm) or "none")
    return fires, out


def boot(df, units, scene_of, tag):
    """Median per-unit paired diff with 95% percentile CIs; rng 20261006; scene draws, then clip draws."""
    rng = np.random.default_rng(20261006)
    scenes = sorted({scene_of[u] for u in units})
    sd = rng.integers(0, len(scenes), (10000, len(scenes)))
    cd = rng.integers(0, len(units), (10000, len(units)))
    w = wide(df, "STALE_MOV").reindex(units)
    by_scene = [np.array([i for i, u in enumerate(units) if scene_of[u] == s]) for s in scenes]
    rows = []
    for key, hi, lo, _ in RULES:
        d = (w[hi] - w[lo]).to_numpy()
        est = float(np.nanmedian(d))
        sc = np.array([np.nanmedian(d[np.concatenate([by_scene[j] for j in row])]) for row in sd])
        cl = np.array([np.nanmedian(d[row]) for row in cd])
        for meth, b in ((f"scene-cluster bootstrap ({len(scenes)} scenes, {len(units)} clips)", sc),
                        (f"clip-level bootstrap ({len(units)} clips)", cl)):
            lo_, hi_ = np.percentile(b, [2.5, 97.5])
            rows.append({**tag, "key": key, "method": meth, "estimate": est, "ci_low": lo_, "ci_high": hi_})
    return rows


def md(df, fmt="{:.5f}"):
    cols = list(df.columns)
    L = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage", choices=("gates", "all"), default="all")
    a = ap.parse_args()
    T0 = time.perf_counter()
    prereg_same = subprocess.run(["git", "diff", "--quiet", PREREG_COMMIT, "--",
                                  "p3/codecsight_native2fps/PREREGISTRATION.md"], cwd=REPO).returncode == 0
    man = json.load(open(os.path.join(HERE, "manifest.json")))
    W_ = windows()
    clips = sorted(man["clips"])
    assert set(clips) == set(W_) and len(clips) == 27, len(clips)
    bad = [(r["clip"], r["arm"]) for r in man["runs"] if r["status"] != "ok" or r.get("check_mismatches")]
    assert not bad and all(man["raft"][c]["status"] == "ok" for c in clips), bad
    scene_of = {c: W_[c]["scene"] for c in clips}

    # ---- stages 1-3
    gt, gres = run_gates(clips, W_, a.workers)
    if a.stage == "gates":
        return
    units = gt.loc[gt.counted, "clip"].tolist()
    N = len(units)
    heads = []
    if gres["V0"] == "FAIL":
        heads.append("RAFT unreliable")
    if gres["drift"] == "FAIL":
        heads.append("truth drift too large")
    if N < N_MIN:
        heads.append(f"N = {N} < {N_MIN}")
    verdict_ok = not heads
    label = "verdict" if verdict_ok else "descriptive (no verdict): " + "; ".join(heads)
    unseen = [c for c in units if W_[c]["unseen"]]

    # ---- stage 4
    rows, ct, errs = [], [], []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for clip, r, c, dt, err in pool.imap_unordered(policy_task, [(c, scene_of[c], W_[c]) for c in clips]):
            print(f"policy {clip} {len(r)} rows {dt:.0f}s" + (f" ERROR\n{err}" if err else ""), flush=True)
            rows += r
            ct += c
            if err:
                errs.append(clip)
    if errs:
        sys.exit(f"errors: {errs}")
    pu = pd.DataFrame(rows).sort_values(["variant", "config", "arm", "unit"]).reset_index(drop=True)
    pu["prereg_commit"], pu["code_sha256"] = PREREG_COMMIT, code_hash()
    pu.to_csv(os.path.join(HERE, "per_unit.csv"), index=False)
    ctd = pd.DataFrame(ct)
    ctd["REUSE_c"] = ctd.reused_c / ctd.valid_c
    ctd["STALE_c"] = ctd.stale_c / ctd.valid_c
    ctd["MOVE_c"] = ctd.move_num / ctd.move_den
    ctd["STALE_c_per_MOVE_c"] = ctd.STALE_c / ctd.MOVE_c
    ctd.to_csv(os.path.join(HERE, "common_truth.csv"), index=False)

    def sel(var, cfg):
        return pu[(pu.variant == var) & (pu.config == cfg)]

    vrows, amrows, brows = [], [], []
    L = ["# Reading B (native-rate encode, 8 s GOP, 2 FPS sampling): verdicts", "",
         f"Prereg {PREREG_COMMIT} (file unchanged: {prereg_same}); code sha256 {code_hash()[:16]}; collection HEAD "
         f"{man['head'][:7]}. Generated by analyze.py from per_unit.csv ({len(pu)} rows) and gates.csv.", ""]
    if heads:
        L += [f"## {' / '.join(h for h in heads)}", ""]
    L += [f"- Counted clips (§6): N = {N}. V0: {gres['v0_pass_clips']}/{N} -> **{gres['V0']}**. Drift gate: "
          f"{gres['drift_pass_clips']}/{N} -> **{gres['drift']}** (need {gres['required']} each). "
          f"Status of everything below: **{label}**.",
          f"- Verdict metric: policy STALE_MOV, derived token grid, sampled frames 1..n-1. Medians over the {N} counted "
          f"clips. A missing value does not meet a condition. Variants (a) and (b) each have their own verdicts.", ""]
    decisions = {}
    for var in ("a", "b"):
        for cfg, lab in (("primary", label), ("tau1", "secondary" if verdict_ok else label)):
            d = sel(var, cfg)
            fr = f_rules(d, units)
            am = arm_medians(d, units)
            fires, out = decision(am)
            if cfg == "primary":
                decisions[var] = out
            for r in fr:
                vrows.append({"variant": var, "config": cfg, "subset": "counted", **r})
            vrows.append({"variant": var, "config": cfg, "subset": "counted", "key": "KILL-A",
                          "count": int((am.stale_lt & am.reuse_gt).sum()), "units": 9, "required": 9,
                          "result": "FIRES" if fires else "DOES NOT FIRE"})
            vrows.append({"variant": var, "config": cfg, "subset": "counted", "key": "decision", "result": out})
            amrows += [{"variant": var, "config": cfg, "subset": "counted", **r} for r in am.to_dict("records")]
            brows += boot(d, units, scene_of, {"variant": var, "config": cfg})
            fx = {r["key"]: r["result"] for r in fr}
            L += [f"## Variant ({var}), {cfg} ({lab})", ""] + md(pd.DataFrame(fr)[
                ["key", "comparison", "margin", "count", "required", "result", "median_diff", "missing"]]) + [
                "", f"Kill rules (mechanical): F1 and F2 both fail: {'yes' if fx['F1'] == fx['F2'] == 'FAIL' else 'no'}; "
                    f"exactly one of F1, F2 fails: {'yes' if (fx['F1'] == 'FAIL') != (fx['F2'] == 'FAIL') else 'no'}; "
                    f"F3 fails: {'yes' if fx['F3'] == 'FAIL' else 'no'}; F4 fails: {'yes' if fx['F4'] == 'FAIL' else 'no'}",
                "", f"### KILL-A and decision rule, variant ({var}), {cfg} ({lab}): KILL-A "
                    f"{'FIRES' if fires else 'DOES NOT FIRE'} ({int((am.stale_lt & am.reuse_gt).sum())}/9); "
                    f"decision: **{out}**", ""]
            L += md(am[["arm", "units", "REUSE", "STALE_MOV", "STALE_MOV_missing", "g_REUSE", "g_STALE_MOV",
                        "stale_lt", "reuse_gt"]]) + [""]
            L += ["### Paired medians and 95% bootstrap CIs (descriptive)", ""] + md(pd.DataFrame(
                [b for b in brows if b["variant"] == var and b["config"] == cfg])[
                ["key", "method", "estimate", "ci_low", "ci_high"]]) + [""]
    if decisions["a"] != decisions["b"]:
        L += ["**The primary decisions of variants (a) and (b) differ: the result depends on whether the vectors of "
              "skipped frames are read (§7).**", ""]
    # descriptive: unseen subgroup, tau 0.25, per scene
    L += [f"## Unseen-window subgroup (descriptive): {len(unseen)} counted clips", ""]
    for var in ("a", "b"):
        for cfg in ("primary", "tau1"):
            d = sel(var, cfg)
            fr = f_rules(d, unseen)
            am = arm_medians(d, unseen)
            fires, out = decision(am)
            vrows += [{"variant": var, "config": cfg, "subset": "unseen", **r} for r in fr]
            vrows.append({"variant": var, "config": cfg, "subset": "unseen", "key": "decision", "result": out})
            amrows += [{"variant": var, "config": cfg, "subset": "unseen", **r} for r in am.to_dict("records")]
            L += [f"### ({var}) {cfg}", ""] + md(pd.DataFrame(fr)[["key", "count", "units", "required", "result",
                                                                   "median_diff"]])
            L += ["", f"KILL-A {'fires' if fires else 'does not fire'}; decision rule on this subgroup: {out}", ""]
            L += md(am[["arm", "units", "REUSE", "STALE_MOV"]]) + [""]
    for var in ("a", "b"):
        d = sel(var, "tau0.25")
        fr = f_rules(d, units)
        am = arm_medians(d, units)
        fires, out = decision(am)
        vrows += [{"variant": var, "config": "tau0.25", "subset": "counted", **r} for r in fr]
        vrows.append({"variant": var, "config": "tau0.25", "subset": "counted", "key": "decision", "result": out})
        amrows += [{"variant": var, "config": "tau0.25", "subset": "counted", **r} for r in am.to_dict("records")]
        L += [f"## ({var}) tau 0.25 (descriptive)", ""] + md(pd.DataFrame(fr)[["key", "count", "required", "result",
                                                                               "median_diff"]])
        L += ["", f"KILL-A {'fires' if fires else 'does not fire'}; decision rule: {out}", ""]
    L += ["## F1-F4 per scene, primary (descriptive)", ""]
    ps = []
    for var in ("a", "b"):
        w = wide(sel(var, "primary"), "STALE_MOV").reindex(units)
        for key, hi, lo, _ in RULES:
            d = w[hi] - w[lo]
            for sc in sorted({scene_of[u] for u in units}):
                dd = d[[u for u in units if scene_of[u] == sc]]
                ps.append({"variant": var, "key": key, "scene": sc, "clips": len(dd),
                           "meeting > 0.01": int((dd > MARGIN).sum()), "median diff": float(dd.median())})
    L += md(pd.DataFrame(ps)) + [""]
    amd = pd.DataFrame(amrows)
    amd.to_csv(os.path.join(HERE, "arm_medians.csv"), index=False)
    pd.DataFrame(vrows).to_csv(os.path.join(HERE, "verdicts.csv"), index=False)
    pd.DataFrame(brows).to_csv(os.path.join(HERE, "bootstrap_ci.csv"), index=False)
    L += ["## Per-arm medians, counted clips (descriptive)", "",
          "Policy REUSE / STALE / STALE_MOV; bare per-cell gate on sampled frames; B-frames only (bf 2 arms); share of "
          "tokens active at the sampled frame before each refresh; token-frames lost to the finite-terms rule.", ""]
    L += md(amd[amd.subset == "counted"][["variant", "config", "arm", "REUSE", "STALE", "STALE_MOV", "g_REUSE",
                                         "g_STALE", "g_STALE_MOV", "B_REUSE", "B_STALE", "B_STALE_MOV",
                                         "active_share_pre_refresh", "finite_dropped"]]) + [""]
    ctm = (ctd[ctd["clip"].isin(units)].groupby(["method", "tau", "arm"])[["REUSE_c", "STALE_c", "MOVE_c",
                                                                       "STALE_c_per_MOVE_c"]].median().reset_index())
    L += ["## Common-truth metrics on sampled frames, medians over counted clips (descriptive)", "",
          "MOVE_c = valid sampled cell-frames with |T| > 2 px (native pair s_j - 1 -> s_j) / valid sampled "
          "cell-frames. STALE_c / MOVE_c is not a share.", ""] + md(ctm) + [""]
    # reading B against reading A, same sampled source frames, medians only
    pa = os.path.join(P3, "codecsight_2fps", "per_unit.csv")
    if os.path.isfile(pa):
        A = pd.read_csv(pa)
        A = A[(A.config == "primary") & (A.truth == "raft") & A.unit.isin(units)]
        cmp_ = [{"arm": arm, "A_REUSE": A[A.arm == arm].REUSE.median(), "A_STALE_MOV": A[A.arm == arm].STALE_MOV.median(),
                 **{f"B{v}_{m}": sel(v, "primary").query("arm == @arm and unit in @units")[m].median()
                    for v in ("a", "b") for m in ("REUSE", "STALE_MOV")}} for arm in ARMS]
        L += ["## Reading B against reading A, primary, same counted clips (descriptive; reading A has no verdict)",
              ""] + md(pd.DataFrame(cmp_)) + [""]
    dl = pd.read_csv(os.path.join(HERE, "drift_by_lag.csv"))
    L += ["## Drift report: |S| over valid cells by native lag since refresh, per clip (descriptive)", ""] + md(
        dl, "{:.3f}") + [""]
    br = pd.DataFrame([{"arm": r["arm"], "clip": r["clip"], "kbps": r["bitrate_kbps"], "psnr": r["psnr_y_mean"],
                        "P_qp": r.get("mean_p_qp")} for r in man["runs"] if r.get("status") == "ok"])
    L += ["## Bitrate and PSNR per arm, medians over all clips (descriptive)", ""] + md(
        br.groupby("arm")[["kbps", "psnr", "P_qp"]].median().reset_index(), "{:.2f}") + [""]
    L.append(f"Runtime {time.perf_counter() - T0:.0f} s.")
    open(os.path.join(HERE, "verdicts.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
