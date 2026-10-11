"""POST HOC (not pre-registered; changes no verdict). Matched-reuse comparison of the CodecSight policy (P) against
gates without accumulation (G1 per-cell, G2 per-token), every method scored against one truth. Plan: PLAN.md
(committed alone, e311c83). Loaders and token geometry are imported from p3/codecsight_sim/simulate.py.

Truth (PLAN §3), per cell: last computed L = last frame <= t at which the cell's unit was computed (refresh counts);
S = sum g_k, k = L+1..t; valid = cover at t and every g_k in the sum finite (computed cell: cover only);
stale = reused & valid & |S| > 2 px. Metrics at cell level, frames 1..n-1.

Usage: python posthoc.py --validate     (PLAN §7; exit 1 on any mismatch)
       python posthoc.py [--workers N]  (full run -> per_unit_posthoc.csv)
       python posthoc.py --analyze      (matched.md, fig_matched.png)
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, SIM)
import simulate as S  # noqa: E402

TAUS = (0.25, 0.5, 1.0)
FPS = ("derived", "sq16", "sq32", "sq64")
BANNER = "POST HOC (not pre-registered; changes no verdict). Plan: p3/codecsight_sim/posthoc_matched/PLAN.md (e311c83)."


# --------------------------------------------------------------------------- truth
def accumulate(g, cover_ok, reset, cvalid):
    """g (n, ncell, 2); cover_ok, reset, cvalid (n, ncell) bool (or (n, 1) broadcastable).
    At t: S += g_t (finite terms), F &= finite; valid = cover & (cvalid | F); moved = valid & ~cvalid & |S| > 2;
    then the window restarts where reset[t]. Returns valid, moved (n, ncell)."""
    n, ncell = cover_ok.shape
    Sx = np.zeros(ncell)
    Sy = np.zeros(ncell)
    F = np.ones(ncell, bool)
    valid = np.zeros((n, ncell), bool)
    moved = np.zeros((n, ncell), bool)
    for t in range(n):
        gx, gy = g[t, :, 0], g[t, :, 1]
        fin = np.isfinite(gx) & np.isfinite(gy)
        F &= fin
        Sx += np.where(fin, gx, 0.0)
        Sy += np.where(fin, gy, 0.0)
        cv = np.broadcast_to(cvalid[t], (ncell,))
        valid[t] = cover_ok[t] & (cv | F)
        moved[t] = valid[t] & ~cv & (np.hypot(Sx, Sy) > S.STALE_GT)
        r = np.broadcast_to(reset[t], (ncell,))
        Sx[r] = 0.0
        Sy[r] = 0.0
        F[r] = True
    return valid, moved


# --------------------------------------------------------------------------- methods
def owner(Hc, Wc, H, W, fp):
    """Cell -> index of the token containing the cell centre (column order of simulate.token_matrix)."""
    cy, cx = np.arange(Hc), np.arange(Wc)
    if fp == "derived":
        oy, ox, ntx = (4 * cy + 2) * 16 // H, (4 * cx + 2) * 16 // W, 16
    else:
        s = int(fp[2:])
        oy, ox, ntx = (4 * cy + 2) // s, (4 * cx + 2) // s, -(-W // s)
    return (oy[:, None] * ntx + ox[None, :]).ravel()


def policy_tokens(tokdyn, refresh):
    """P: token computed at t = refresh, or in the union of dynamic tokens of frames after the last refresh up to t."""
    n = tokdyn.shape[0]
    comp = np.ones_like(tokdyn)
    rf = np.nonzero(refresh)[0].tolist() + [n]
    for a, b in zip(rf[:-1], rf[1:]):
        if b > a + 1:
            comp[a + 1:b] = np.logical_or.accumulate(tokdyn[a + 1:b], axis=0)
    return comp


def cell_counts(comp, g, cover_ok):
    """Cell-level counts under the common truth for a cell computed-mask comp (n, ncell), frames 1..n-1."""
    valid, moved = accumulate(g, cover_ok, comp, comp)
    reu = ~comp[1:]
    v = valid[1:]
    return {"valid_c": int(v.sum()), "reused_c": int((reu & v).sum()), "stale_c": int((reu & moved[1:]).sum())}


# --------------------------------------------------------------------------- loaders
def load_unit(data, unit):
    if data == "Sintel":
        g, cov, H, W = S.sintel_gt(unit)
        cover_ok = cov >= S.COVER_MIN
    else:
        g, cover_ok, _, Hc, Wc = S.cctv_truth(unit)
        H, W = Hc * S.C, Wc * S.C
    n, ncell = cover_ok.shape
    Hc, Wc = -(-H // S.C), -(-W // S.C)
    assert Hc * Wc == ncell
    return g, cover_ok, H, W, Hc, Wc, n


def load_encode(data, unit, arm, n, Hc, Wc):
    """(dyn_of(tau) -> (n, ncell) bool, iframes)."""
    if data == "Sintel":
        has, mag, ftype, _ = S.sintel_encode(arm, unit, n, Hc, Wc)
        dyn_of = lambda tau: ~has | (mag >= tau)  # noqa: E731
    else:
        q, ftype, _, _, _ = S.cctv_encode(unit, arm)
        assert q.shape == (n, Hc * Wc)
        dyn_of = lambda tau: q >= 4 * tau  # noqa: E731
    return dyn_of, [t for t, f in enumerate(ftype) if f == "I"]


# --------------------------------------------------------------------------- full run
def run_unit(data, unit, scene, arms):
    g, cover_ok, H, W, Hc, Wc, n = load_unit(data, unit)
    Ms = {}
    for fp in FPS:
        M = S.token_matrix(Hc, Wc, H, W, fp)
        Ms[fp] = (M, owner(Hc, Wc, H, W, fp))
    rows = []
    for arm in arms:
        dyn_of, ifr = load_encode(data, unit, arm, n, Hc, Wc)
        ref = S.native_refresh(ifr, n)
        base = {"data": data, "unit": unit, "scene": scene, "arm": arm, "n_frames": n}
        for tau in TAUS:
            dyn = dyn_of(tau)
            rows.append({**base, "method": "G1", "footprint": "cell", "tau": tau,
                         **cell_counts(ref[:, None] | dyn, g, cover_ok)})
            for fp in FPS:
                M, own = Ms[fp]
                td = S.tok_any(dyn, M)
                g2 = ref[:, None] | td
                rows.append({**base, "method": "G2", "footprint": fp, "tau": tau,
                             **cell_counts(g2[:, own], g, cover_ok)})
                pc = policy_tokens(td, ref)
                rows.append({**base, "method": "P", "footprint": fp, "tau": tau,
                             **cell_counts(pc[:, own], g, cover_ok)})
    return rows


def _task(args):
    t0 = time.perf_counter()
    try:
        return args[1], run_unit(*args), time.perf_counter() - t0, None
    except Exception:
        return args[1], [], time.perf_counter() - t0, traceback.format_exc()


def units():
    return [(d, u, sc, arms) for d, u, sc, arms, _ in S.units()]


# --------------------------------------------------------------------------- validation (PLAN §7)
def validate():
    pu = pd.read_csv(os.path.join(SIM, "per_unit.csv"))
    U = units()
    pick = ([u for u in U if u[0] == "CCTV"][:3]) + ([u for u in U if u[0] == "Sintel"][:3])
    la, lb = [], []
    for data, unit, scene, arms in pick:
        g, cover_ok, H, W, Hc, Wc, n = load_unit(data, unit)
        gmag = np.hypot(g[..., 0], g[..., 1])
        M = S.token_matrix(Hc, Wc, H, W, "derived")
        nct = np.asarray(M.sum(axis=0)).ravel()
        for arm in arms:
            dyn_of, ifr = load_encode(data, unit, arm, n, Hc, Wc)
            iref = np.zeros(n, bool)
            iref[ifr] = True
            # (a) G1, per-frame truth, I-frame refresh only
            for tau in TAUS:
                dyn = dyn_of(tau)
                comp = iref[:, None] | dyn
                valid, moved = accumulate(g, cover_ok, np.ones((n, 1), bool), comp)
                v, r = valid[1:], ~comp[1:]
                mine = {"g_valid": int(v.sum()), "g_reused": int((r & v).sum()),
                        "g_moved": int((v & (gmag[1:] > S.STALE_GT)).sum()), "g_stale": int((r & moved[1:]).sum())}
                cfg = {0.25: "tau0.25", 0.5: "primary", 1.0: "tau1"}[tau]
                ref_row = pu[(pu["unit"] == unit) & (pu["arm"] == arm) & (pu["config"] == cfg)].iloc[0]
                for k, val in mine.items():
                    la.append((data, unit, arm, tau, k, val, int(ref_row[k]), val == int(ref_row[k])))
            # (b) P derived / tau 0.5, token level, pre-registered truth (summed since refresh)
            ref = S.native_refresh(ifr, n)
            td = S.tok_any(dyn_of(0.5), M)
            pc = policy_tokens(td, ref)
            valid, moved = accumulate(g, cover_ok, ref[:, None], ref[:, None])
            vt = S.tok_count(valid, M) >= nct / 2.0
            mt = vt & S.tok_any(moved, M)
            reu = ~pc
            mine = {"valid_tf": int(vt[1:].sum()), "reused_tf": int((reu & vt)[1:].sum()),
                    "stale_tf": int((reu & mt)[1:].sum())}
            ref_row = pu[(pu["unit"] == unit) & (pu["arm"] == arm) & (pu["config"] == "primary")].iloc[0]
            for k, val in mine.items():
                lb.append((data, unit, arm, k, val, int(ref_row[k]), val == int(ref_row[k])))
        print(f"validated {data} {unit}", flush=True)
    A = pd.DataFrame(la, columns=["data", "unit", "arm", "tau", "count", "posthoc", "per_unit_csv", "match"])
    B = pd.DataFrame(lb, columns=["data", "unit", "arm", "count", "posthoc", "per_unit_csv", "match"])
    oka, okb = bool(A.match.all()), bool(B.match.all())
    summ = lambda D, keys: D.groupby(keys).agg(checks=("match", "size"), matches=("match", "sum")).reset_index()  # noqa
    txt = (f"# {BANNER}\n# Validation (PLAN §7)\n\n"
           f"## (a) G1, per-frame truth, I-frame refresh: vs per_unit.csv g_ columns\n"
           f"{summ(A, ['data', 'unit', 'count']).to_string(index=False)}\n\nALL MATCH (a): {oka} "
           f"({int(A.match.sum())}/{len(A)})\n\n"
           f"## (b) P derived / tau 0.5, token level, pre-registered truth: vs per_unit.csv primary\n"
           f"{summ(B, ['data', 'unit', 'count']).to_string(index=False)}\n\nALL MATCH (b): {okb} "
           f"({int(B.match.sum())}/{len(B)})\n")
    if not (oka and okb):
        txt += "\n## Mismatches\n" + A[~A.match].to_string(index=False) + "\n" + B[~B.match].to_string(index=False) + "\n"
    open(os.path.join(HERE, "validation.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    return oka and okb


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.validate:
        t0 = time.perf_counter()
        ok = validate()
        print(f"validation runtime {time.perf_counter() - t0:.1f} s")
        sys.exit(0 if ok else 1)
    if a.analyze:
        import analyze_posthoc
        analyze_posthoc.main()
        return
    U = units()
    log = open(os.path.join(HERE, "posthoc.log"), "a", encoding="utf-8")

    def L(s):
        s = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()
    L(f"POST HOC run start: {len(U)} units, workers {a.workers}")
    t0 = time.perf_counter()
    rows, errs = [], []
    with mp.Pool(a.workers) as pool:
        for i, (unit, r, dt, err) in enumerate(pool.imap_unordered(_task, U), 1):
            if err:
                errs.append(unit)
                L(f"[{i}/{len(U)}] {unit} ERROR\n{err}")
            else:
                rows += r
                L(f"[{i}/{len(U)}] {unit} {len(r)} rows {dt:.1f} s")
    df = pd.DataFrame(rows)
    df["REUSE_c"] = df.reused_c / df.valid_c
    df["STALE_c"] = df.stale_c / df.valid_c
    df = df.sort_values(["data", "arm", "unit", "method", "footprint", "tau"]).reset_index(drop=True)
    with open(os.path.join(HERE, "per_unit_posthoc.csv"), "w", encoding="utf-8", newline="") as f:
        f.write(f"# {BANNER}\n")
        df.to_csv(f, index=False)
    L(f"done: {len(df)} rows; errors {len(errs)}; runtime {time.perf_counter() - t0:.1f} s")
    if errs:
        sys.exit(1)


if __name__ == "__main__":
    main()
