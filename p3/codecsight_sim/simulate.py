"""CodecSight policy simulation: policy, bare per-cell gate and truth, as fixed by p3/codecsight_sim/PREREGISTRATION.md
(commit 9f5b5a3). Saved data only: no encode, no MV extraction, no RAFT. CPU only.

Inputs
  Sintel: p3/sweep/blocks/final/<arm>__<seq>.parquet (codec blocks: frame, dir, x, y, w, h, mv_x, mv_y) and
          <arm>__<seq>__frames.parquet (ftype, group); per-cell GT rebuilt from the Sintel .flo files with
          lib/gt_grid.GTGrid(flow_into(t), bad_mask_into(t)).cell_gt(), exactly as the main sweep (cached under cache/).
  CCTV:   p3/virat_confirm/data/<clip>/<arm>.npz (mv_q = floor(4|MV|) of the painted cell vector, 255 = no vector;
          ftype, d_past, meta) and raft.npz (splat_t, splat_count); the 27 eligible clips of eligibility_v0.csv.

Policy (pre-registration §2), per encode, frames in display order:
  cell dynamic at t   : no vector, or |MV| >= tau        (Sintel: painted float vector; CCTV: q >= 4 tau, q = 255 incl.)
  token dynamic at t  : any of its cells dynamic (token cells = cells whose 4x4 footprint overlaps the token)
  active set A        : cleared at every refresh frame; otherwise A |= dynamic tokens of t (union since refresh)
  reused at t         : t is not a refresh frame and the token is not in A (A includes t's own dynamic tokens)
  refresh (native)    : every I-frame, and 16 frames after the previous refresh (counter restarts at each refresh)
Truth (§3): S_t(c) = sum of per-frame GT g_k(c), k = r+1..t, at the fixed cell; a cell is valid at t if its
  per-frame cover rule holds at t and every g_k(c) in the sum is finite; token valid if >= half its cells are valid;
  moved if valid and some valid cell has |S| > 2 px; stale = reused and moved. Frames 1..n-1 are scored.
Bare gate (§4): reuse = has vector & |MV| < tau, per-frame truth |g_t| > 2 px, same frames and validity.
2 FPS (§7, CCTV): sampled frames j*k (k = 12 at 23.97 fps, 15 at 30 fps); refresh at the first sampled frame at or
  after an I-frame and 16 sampled frames after the previous refresh; truth summed over every native frame; only
  sampled frames scored. Variant a: union over sampled frames' dynamic tokens; variant b: also every skipped native
  frame since the previous sampled frame (= every native frame since the refresh).

Usage: python simulate.py --repro      (validation 2b; exit 1 on any mismatch)
       python simulate.py [--workers N] (full run -> per_unit.csv, simulate.log)
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from fractions import Fraction

import numpy as np
import pandas as pd
import scipy.sparse as sps

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
REPO = os.path.dirname(P3)
sys.path.insert(0, P3)

SINTEL = r"C:\Users\akash\Documents\datasets\MPI-Sintel"
PREREG_COMMIT = "9f5b5a3"
C, COVER_MIN, COVER_COUNT, STALE_GT, WINDOW = 4, 0.5, 8, 2.0, 16
CODE_FILES = ("simulate.py", "analyze.py", "independent_verdicts.py", "test_policy.py")

SINTEL_ARMS = ([f"x264_crf{c}" for c in (12, 18, 23, 28, 33, 38, 45)]
               + [f"nvenc_qp{q}" for q in (18, 23, 28, 33, 38, 45)]
               + ["x264_qp24", "x264_qp24_bf2_pb1", "nvenc_qp28_bf2_bqeq", "mpeg4_q4"])
CCTV_ARMS = ("x264_crf12", "x264_crf45", "x264_qp24", "x264_qp24_bf2_pb1", "nvenc_qp18", "nvenc_qp23",
             "nvenc_qp45", "nvenc_qp28", "nvenc_qp28_bf2_bqeq")
# (config, footprint, tau, rate); 2 FPS rows are CCTV only
CONFIGS = (("primary", "derived", 0.5, "native"), ("tau1", "derived", 1.0, "native"),
           ("tau0.25", "derived", 0.25, "native"), ("sq16", "sq16", 0.5, "native"), ("sq32", "sq32", 0.5, "native"),
           ("sq64", "sq64", 0.5, "native"), ("2fps_a", "derived", 0.5, "2fps_a"), ("2fps_b", "derived", 0.5, "2fps_b"))


# --------------------------------------------------------------------------- provenance
def code_hash():
    """sha256 over the analysis code files (fixed order); the files are not committed, so this stands in for a
    git diff hash. Missing files hash as empty."""
    h = hashlib.sha256()
    for f in CODE_FILES:
        p = os.path.join(HERE, f)
        h.update(f.encode() + b"\0" + (open(p, "rb").read() if os.path.isfile(p) else b"") + b"\0")
    return h.hexdigest()


def prereg_check():
    """True if PREREGISTRATION.md in the working tree is identical to commit 9f5b5a3."""
    r = subprocess.run(["git", "diff", "--quiet", PREREG_COMMIT, "--", "p3/codecsight_sim/PREREGISTRATION.md"],
                       cwd=REPO)
    return r.returncode == 0


def provenance():
    return {"prereg_commit": PREREG_COMMIT, "prereg_unchanged": prereg_check(), "code_sha256": code_hash()}


# --------------------------------------------------------------------------- geometry
def token_matrix(Hc, Wc, H, W, footprint):
    """Sparse (ncell, ntok) 0/1 matrix: cell c belongs to token j if the cell's 4x4 footprint overlaps the token
    rectangle with positive area. 'derived' = 16 x 16 grid of W/16 x H/16 px; 'sqN' = N x N px squares from (0, 0)."""
    def axis(nc, L):
        if footprint == "derived":
            # token i spans [i L/16, (i+1) L/16); cell spans [4c, 4c+4): overlap iff 64c < (i+1)L and 64c+64 > iL
            return [(c, i) for c in range(nc) for i in range(16) if 64 * c < (i + 1) * L and 64 * c + 64 > i * L], 16
        s = int(footprint[2:])
        return [(c, (C * c) // s) for c in range(nc)], -(-L // s)
    ys, nty = axis(Hc, H)
    xs, ntx = axis(Wc, W)
    rows, cols = [], []
    for cy, ty in ys:
        for cx, tx in xs:
            rows.append(cy * Wc + cx)
            cols.append(ty * ntx + tx)
    M = sps.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(Hc * Wc, nty * ntx))
    return M


def tok_any(X, M):
    """X (n, ncell) bool -> (n, ntok) bool: any member cell true."""
    return np.asarray((M.T @ X.T.astype(np.float32)).T) > 0


def tok_count(X, M):
    return np.asarray((M.T @ X.T.astype(np.float32)).T)


# --------------------------------------------------------------------------- schedules
def native_refresh(iframes, n, window=WINDOW):
    ref, last, iset = np.zeros(n, bool), None, set(int(i) for i in iframes)
    for t in range(n):
        if t in iset or last is None or t - last >= window:
            ref[t] = True
            last = t
    return ref


def fps2_schedule(iframes, n, k, window=WINDOW):
    """(sampled mask, refresh mask) for 2 FPS sampling every k-th frame."""
    samp = list(range(0, n, k))
    firsts = {next((s for s in samp if s >= i), None) for i in iframes} - {None}
    smask, ref, last = np.zeros(n, bool), np.zeros(n, bool), None
    smask[samp] = True
    for j, s in enumerate(samp):
        if s in firsts or last is None or j - last >= window:
            ref[s] = True
            last = j
    return smask, ref


def fps_step(fps_str):
    return int(round(float(Fraction(fps_str)) / 2))


# --------------------------------------------------------------------------- truth
def cell_truth(g, cover_ok, refresh):
    """g (n, ncell, 2) float with NaN; cover_ok (n, ncell) bool; refresh (n,) bool.
    Returns valid (n, ncell), moved (n, ncell): summed truth since the last refresh, finite-terms rule."""
    n, ncell = cover_ok.shape
    S = np.zeros((ncell, 2), np.float64)
    F = np.ones(ncell, bool)
    valid = np.zeros((n, ncell), bool)
    moved = np.zeros((n, ncell), bool)
    for t in range(n):
        if refresh[t]:
            S[:] = 0.0
            F[:] = True
        else:
            gt = g[t]
            fin = np.isfinite(gt[:, 0]) & np.isfinite(gt[:, 1])
            F &= fin
            S += np.where(fin[:, None], gt, 0.0)
        valid[t] = cover_ok[t] & F
        moved[t] = valid[t] & (np.hypot(S[:, 0], S[:, 1]) > STALE_GT)
    return valid, moved


# --------------------------------------------------------------------------- policy
def policy_counts(dyn, M, ncell_tok, refresh, scored, contrib, valid, lenient, moved, bmask=None, debug=False):
    """dyn (n, ncell) bool; refresh / scored / contrib (n,) bool; valid / lenient / moved (n, ncell) bool cell truth.
    Returns dict of counts over scored frames (and over scored B-frames if bmask is given)."""
    n = dyn.shape[0]
    ntok = M.shape[1]
    td = np.zeros((n, ntok), bool)
    ci = np.nonzero(contrib & ~refresh)[0]
    if ci.size:
        td[ci] = tok_any(dyn[ci], M)
    si = np.nonzero(scored)[0]
    half = ncell_tok / 2.0
    vt = np.zeros((n, ntok), bool)
    lt = np.zeros((n, ntok), bool)
    mt = np.zeros((n, ntok), bool)
    vt[si] = tok_count(valid[si], M) >= half
    lt[si] = tok_count(lenient[si], M) >= half
    mt[si] = vt[si] & tok_any(moved[si], M)
    A = np.zeros(ntok, bool)
    reused = np.zeros((n, ntok), bool)
    pre_ref, last_share = [], None
    for t in range(n):
        if refresh[t]:
            if last_share is not None:
                pre_ref.append(last_share)
            A[:] = False
        else:
            if contrib[t]:
                A |= td[t]
            if scored[t]:
                reused[t] = ~A
                last_share = A.mean()
    out = {"ntok": ntok, "active_share_pre_refresh": float(np.mean(pre_ref)) if pre_ref else np.nan}
    for lab, fm in (("", scored), ("B_", None if bmask is None else scored & bmask)):
        if fm is None:
            continue
        r_, v_, m_ = reused[fm], vt[fm], mt[fm]
        out.update({f"{lab}n_scored": int(fm.sum()), f"{lab}valid_tf": int(v_.sum()),
                    f"{lab}reused_tf": int((r_ & v_).sum()), f"{lab}moved_tf": int(m_.sum()),
                    f"{lab}stale_tf": int((r_ & m_).sum()), f"{lab}dropped_finite_tf": int((lt[fm] & ~v_).sum())})
    if debug:
        out["_reused"], out["_valid_tok"], out["_moved_tok"] = reused, vt, mt
    return out


def bare_gate(reuse, cover_ok, gmag, frames):
    """Per-cell gate counts over the given frame indices."""
    v = cover_ok[frames]
    mv = v & (gmag[frames] > STALE_GT)
    r = reuse[frames]
    return {"g_valid": int(v.sum()), "g_reused": int((r & v).sum()), "g_moved": int(mv.sum()),
            "g_stale": int((r & mv).sum())}


def ratios(d, pre=""):
    V, R, Mv, S = (d.get(f"{pre}{k}") for k in ("valid_tf", "reused_tf", "moved_tf", "stale_tf"))
    if V is None:
        return {}
    return {f"{pre}REUSE": R / V if V else np.nan, f"{pre}STALE": S / V if V else np.nan,
            f"{pre}STALE_MOV": S / Mv if Mv else np.nan}


def gate_ratios(d):
    V, R, Mv, S = d["g_valid"], d["g_reused"], d["g_moved"], d["g_stale"]
    return {"g_REUSE": R / V if V else np.nan, "g_STALE": S / V if V else np.nan, "g_STALE_MOV": S / Mv if Mv else np.nan}


# --------------------------------------------------------------------------- Sintel loaders
def sintel_gt(seq):
    """(g (n, ncell, 2) float64 NaN where no GT pixel, cover (n, ncell) float64, H, W); row 0 = NaN / 0."""
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    cp = os.path.join(HERE, "cache", f"sintel_gt_{seq}.npz")
    if os.path.isfile(cp):
        z = np.load(cp)
        return z["g"], z["cover"], int(z["H"]), int(z["W"])
    from lib.flo import SintelSeq
    from lib.gt_grid import GTGrid
    s = SintelSeq(SINTEL, seq, "final")
    n = s.n_frames
    gs, cs = [], []
    for t in range(1, n):
        flow, bad = s.flow_into(t), s.bad_mask_into(t)
        gg = GTGrid(flow, bad)
        gg._build_sat()
        cgx, cgy, ccov, _ = gg.cell_gt()
        gs.append(np.stack([cgx, cgy], -1))
        cs.append(ccov)
        H, W = flow.shape[:2]
    Hc, Wc = cgx.shape
    g = np.full((n, Hc * Wc, 2), np.nan)
    cov = np.zeros((n, Hc * Wc))
    g[1:] = np.stack(gs).reshape(n - 1, Hc * Wc, 2)
    cov[1:] = np.stack(cs).reshape(n - 1, Hc * Wc)
    np.savez(cp, g=g, cover=cov, H=H, W=W)
    return g, cov, H, W


def paint(Hc, Wc, x0, y0, w, h):
    """As pilot/run_pilot.py:paint: flat cell indices + owning block index (blocks on the 4-px grid, <= 16 px)."""
    assert np.all(w <= 16) and np.all(h <= 16)
    o = np.arange(4)
    cx = (x0 // C)[:, None, None] + o[None, None, :]
    cy = (y0 // C)[:, None, None] + o[None, :, None]
    msk = ((o[None, None, :] < (w // C)[:, None, None]) & (o[None, :, None] < (h // C)[:, None, None])
           & (cx < Wc) & (cy < Hc))
    bi = np.broadcast_to(np.arange(x0.size)[:, None, None], msk.shape)
    return (cy * Wc + cx)[msk], bi[msk]


def sintel_encode(arm, seq, n, Hc, Wc):
    """(has (n, ncell) bool, mag (n, ncell) float64, ftype list, group list) from the saved blocks;
    painting as the main sweep: future-referencing blocks first, past-referencing blocks override."""
    bdir = os.path.join(P3, "sweep", "blocks", "final")
    fr = pd.read_parquet(os.path.join(bdir, f"{arm}__{seq}__frames.parquet"), columns=["frame", "ftype", "group"])
    fr = fr.sort_values("frame")
    assert fr.frame.tolist() == list(range(n)), (arm, seq, "frames")
    b = pd.read_parquet(os.path.join(bdir, f"{arm}__{seq}.parquet"),
                        columns=["frame", "dir", "x", "y", "w", "h", "mv_x", "mv_y"])
    ncell = Hc * Wc
    has = np.zeros((n, ncell), bool)
    mag = np.full((n, ncell), np.nan)
    for t, bt in b.groupby("frame", sort=True):
        cmv = np.full((ncell, 2), np.nan)
        cdir = np.zeros(ncell, np.int8)
        for sgn, dn in ((1, "future"), (-1, "past")):
            s = bt[bt.dir == dn]
            if not len(s):
                continue
            ci, bi = paint(Hc, Wc, s.x.to_numpy(), s.y.to_numpy(), s.w.to_numpy(), s.h.to_numpy())
            cmv[ci] = s[["mv_x", "mv_y"]].to_numpy()[bi]
            cdir[ci] = sgn
        has[t] = cdir != 0
        mag[t] = np.hypot(cmv[:, 0], cmv[:, 1])
    return has, mag, fr.ftype.tolist(), fr.group.tolist()


# --------------------------------------------------------------------------- CCTV loaders
def cctv_truth(clip):
    """(g (n, ncell, 2) float64 from float32(splat_t), cover_ok (n, ncell), gmag32 (n, ncell) float32 as analyze_confirm)."""
    rz = np.load(os.path.join(P3, "virat_confirm", "data", clip, "raft.npz"))
    sp = rz["splat_t"].astype(np.float32)
    nm1, Hc, Wc = sp.shape[:3]
    n, ncell = nm1 + 1, Hc * Wc
    cover_ok = np.zeros((n, ncell), bool)
    cover_ok[1:] = rz["splat_count"].reshape(nm1, ncell) >= COVER_COUNT
    gmag = np.full((n, ncell), np.nan, np.float32)
    gmag[1:] = np.hypot(sp[..., 0], sp[..., 1]).reshape(nm1, ncell)
    g = np.full((n, ncell, 2), np.nan)
    g[1:] = sp.reshape(nm1, ncell, 2)
    return g, cover_ok, gmag, Hc, Wc


def cctv_encode(clip, arm):
    z = np.load(os.path.join(P3, "virat_confirm", "data", clip, f"{arm}.npz"))
    q = z["mv_q"]
    n = q.shape[0]
    meta = json.loads(str(z["meta"]))
    return (q.reshape(n, -1), [str(x) for x in z["ftype"]], z["d_past"], bool(z["d_known"]), meta)


# --------------------------------------------------------------------------- per-unit simulation
def simulate_unit(data, unit, scene, arms, configs):
    """All arms x configs for one sequence / clip. Returns list of row dicts."""
    rows = []
    if data == "Sintel":
        g, cov, H, W = sintel_gt(unit)
        cover_ok = cov >= COVER_MIN
        gmag = np.hypot(g[..., 0], g[..., 1])
    else:
        g, cover_ok, gmag, Hc, Wc = cctv_truth(unit)
        H, W = Hc * C, Wc * C
    n, ncell = cover_ok.shape
    Hc, Wc = -(-H // C), -(-W // C)
    assert Hc * Wc == ncell
    Ms = {}
    truth_cache = {}
    lenient = cover_ok  # finite-terms rule off: per-frame cover only
    for arm in arms:
        if data == "Sintel":
            has, mag, ftype, _ = sintel_encode(arm, unit, n, Hc, Wc)
            fps = "24"

            def dyn_of(tau, has=has, mag=mag):
                return ~has | (mag >= tau)
        else:
            q, ftype, _, _, meta = cctv_encode(unit, arm)
            assert q.shape == (n, ncell), (unit, arm, q.shape)
            fps = meta["fps"]

            def dyn_of(tau, q=q):
                return q >= 4 * tau
        iframes = [t for t, f in enumerate(ftype) if f == "I"]
        bmask = np.array([f == "B" for f in ftype]) if "B" in ftype else None
        nat_ref = native_refresh(iframes, n)
        nat_scored = np.arange(n) >= 1
        dyn_cache = {}
        gate_cache = {}
        for cfg, fp, tau, rate in configs:
            if rate != "native" and data == "Sintel":
                continue
            if tau not in dyn_cache:
                dyn_cache[tau] = dyn_of(tau)
            dyn = dyn_cache[tau]
            if rate == "native":
                ref, scored, contrib = nat_ref, nat_scored, np.ones(n, bool)
            else:
                smask, ref = fps2_schedule(iframes, n, fps_step(fps))
                scored = smask & (np.arange(n) >= 1)
                contrib = smask if rate == "2fps_a" else np.ones(n, bool)
            key = ref.tobytes()
            if key not in truth_cache:
                truth_cache[key] = cell_truth(g, cover_ok, ref)
            valid, moved = truth_cache[key]
            if fp not in Ms:
                M = token_matrix(Hc, Wc, H, W, fp)
                Ms[fp] = (M, np.asarray(M.sum(axis=0)).ravel())
            M, nct = Ms[fp]
            d = policy_counts(dyn, M, nct, ref, scored, contrib, valid, lenient, moved, bmask)
            row = {"data": data, "unit": unit, "scene": scene, "arm": arm, "config": cfg, "footprint": fp,
                   "tau": tau, "rate": rate, "fps": fps, "n_frames": n, "n_refresh": int((ref & scored).sum()) +
                   int(ref[0]), **d, **ratios(d), **ratios(d, "B_")}
            if rate == "native":
                if tau not in gate_cache:
                    gate_cache[tau] = bare_gate(~dyn, cover_ok, gmag, np.arange(1, n))
                row.update(gate_cache[tau])
                row.update(gate_ratios(gate_cache[tau]))
            rows.append(row)
    return rows


def _task(args):
    t0 = time.perf_counter()
    try:
        rows = simulate_unit(*args)
        return args[1], rows, time.perf_counter() - t0, None
    except Exception:
        return args[1], [], time.perf_counter() - t0, traceback.format_exc()


def units():
    bdir = os.path.join(P3, "sweep", "blocks", "final")
    seqs = sorted({f.split("__")[1] for f in os.listdir(bdir) if f.endswith("__frames.parquet")})
    el = pd.read_csv(os.path.join(P3, "virat_confirm", "eligibility_v0.csv"))
    el = el[el.eligible.astype(bool)].sort_values("clip")
    return ([("Sintel", s, s, SINTEL_ARMS, CONFIGS) for s in seqs]
            + [("CCTV", c, s, CCTV_ARMS, CONFIGS) for c, s in zip(el["clip"], el["scene"])])


# --------------------------------------------------------------------------- validation 2b
def repro():
    """Main-sweep bare gate (tau = 1, its frame groups and validity) vs p3/sweep/results.csv and
    p3/virat_confirm/results.csv. Exit code 1 on any difference at 6 decimals."""
    sr = pd.read_csv(os.path.join(P3, "sweep", "results.csv"))
    cr = pd.read_csv(os.path.join(P3, "virat_confirm", "results.csv"))
    lines, ok = [], True
    cases = [(a, s, "d1") for a in ("x264_crf12", "x264_crf45", "nvenc_qp45") for s in ("alley_1", "cave_2", "temple_3")]
    cases.append(("x264_qp24_bf2_pb1", "market_5", "B_naive"))
    for arm, seq, grp in cases:
        g, cov, H, W = sintel_gt(seq)
        n = cov.shape[0]
        Hc, Wc = -(-H // C), -(-W // C)
        has, mag, ftype, fgroup = sintel_encode(arm, seq, n, Hc, Wc)
        want = "d1" if grp == "d1" else "B"
        T = np.array([t for t in range(1, n) if fgroup[t] == want])
        reuse = has & (mag < 1.0)
        valid = cov >= COVER_MIN
        gm = np.hypot(g[..., 0], g[..., 1])
        V = int(valid[T].sum())
        S = int((valid[T] & reuse[T] & (gm[T] > STALE_GT)).sum())
        mine = S / V
        ref = sr[(sr["pass"] == "final") & (sr.encode_id == arm) & (sr.sequence == seq) & (sr.group == grp)
                 & np.isclose(sr.tau, 1.0)]["stale_of_valid"]
        theirs = float(ref.iloc[0]) if len(ref) == 1 else np.nan
        same = round(mine, 6) == round(theirs, 6)
        ok &= same
        lines.append(("Sintel", seq, arm, grp, "stale/valid", mine, theirs, same))
    el = pd.read_csv(os.path.join(P3, "virat_confirm", "eligibility_v0.csv"))
    clips = sorted(el[el.eligible.astype(bool)]["clip"])[:2]
    for clip in clips:
        g, cover_ok, gmag, Hc, Wc = cctv_truth(clip)
        q, ftype, d_past, d_known, _ = cctv_encode(clip, "x264_crf45")
        T = np.array([t for t in range(1, len(ftype)) if ftype[t] == "P" and d_past[t] == 1])
        reuse = q < 4 * 1.0
        mov = cover_ok[T] & (gmag[T] > STALE_GT)
        mine = int((reuse[T] & mov).sum()) / int(mov.sum())
        ref = cr[(cr["clip"] == clip) & (cr.arm == "x264_crf45") & (cr.group == "d1") & np.isclose(cr.tau, 1.0)]["stale_mov"]
        theirs = float(ref.iloc[0]) if len(ref) == 1 else np.nan
        same = round(mine, 6) == round(theirs, 6)
        ok &= same
        lines.append(("CCTV", clip, "x264_crf45", "d1", "STALE_MOV", mine, theirs, same))
    df = pd.DataFrame(lines, columns=["data", "unit", "arm", "group", "metric", "this_code", "saved_results", "match_6dp"])
    pv = provenance()
    txt = (f"# Validation 2b: reproduction of the main bare gate (tau = 1 px)\n"
           f"prereg {pv['prereg_commit']} (file unchanged: {pv['prereg_unchanged']}); code sha256 {pv['code_sha256']}\n\n"
           + df.to_string(index=False, float_format=lambda v: f"{v:.8f}") + f"\n\nALL MATCH: {ok}\n")
    open(os.path.join(HERE, "repro.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    return ok


PHASES = (("primary", ("primary",)), ("tau1", ("tau1",)),
          ("descriptive", ("tau0.25", "sq16", "sq32", "sq64", "2fps_a", "2fps_b")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="", help="comma list of units (debug)")
    a = ap.parse_args()
    if a.repro:
        sys.exit(0 if repro() else 1)
    pv = provenance()
    if not pv["prereg_unchanged"]:
        sys.exit("PREREGISTRATION.md differs from 9f5b5a3; refusing to run")
    U = units()
    if a.only:
        U = [u for u in U if u[1] in a.only.split(",")]
    log = open(os.path.join(HERE, "simulate.log"), "a", encoding="utf-8")

    def L(s):
        s = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()
    L(f"start: {len(U)} units, workers {a.workers}, prereg {pv['prereg_commit']}, code {pv['code_sha256']}")
    t0 = time.perf_counter()
    # Sintel GT cache first (serial, avoids two workers writing one file)
    for u in U:
        if u[0] == "Sintel":
            t1 = time.perf_counter()
            sintel_gt(u[1])
            L(f"sintel gt cache {u[1]} {time.perf_counter() - t1:.1f} s")
    rows, errs = [], []
    cfg_by_name = {c[0]: c for c in CONFIGS}
    # phases in pre-registered order: primary, then tau 1, then the descriptive configurations
    for ph, names in PHASES:
        tp = time.perf_counter()
        PU = [(d, u, sc, arms, tuple(cfg_by_name[nm] for nm in names)) for d, u, sc, arms, _ in U]
        L(f"phase {ph}: configs {', '.join(names)}")
        with mp.Pool(a.workers) as pool:
            for i, (unit, r, dt, err) in enumerate(pool.imap_unordered(_task, PU), 1):
                if err:
                    errs.append((ph, unit, err))
                    L(f"[{ph} {i}/{len(PU)}] {unit} ERROR\n{err}")
                else:
                    rows += r
                    L(f"[{ph} {i}/{len(PU)}] {unit} {len(r)} rows {dt:.1f} s")
        L(f"phase {ph} done: {time.perf_counter() - tp:.1f} s")
    df = pd.DataFrame(rows)
    df["prereg_commit"] = pv["prereg_commit"]
    df["code_sha256"] = pv["code_sha256"]
    df = df.sort_values(["data", "config", "arm", "unit"]).reset_index(drop=True)
    out = "per_unit.csv" if not a.only else "per_unit_debug.csv"
    df.to_csv(os.path.join(HERE, out), index=False)
    L(f"done: {len(df)} rows -> {out}; errors {len(errs)}; runtime {time.perf_counter() - t0:.1f} s")
    if errs:
        sys.exit(1)


if __name__ == "__main__":
    main()
