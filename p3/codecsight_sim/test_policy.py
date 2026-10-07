"""Validation 2a: synthetic unit tests of simulate.py (policy, refresh schedule, truth, token geometry).
Run: python test_policy.py   (writes test_results.txt; exit 1 on any failure)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simulate as S  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))


def scene(n=20, Hc=8, Wc=8, fp="sq8"):
    """n frames of an Hc x Wc cell grid (4 px cells), tokens of 8 x 8 px (2 x 2 cells), I-frame at 0 only."""
    M = S.token_matrix(Hc, Wc, Hc * 4, Wc * 4, fp)
    nct = np.asarray(M.sum(axis=0)).ravel()
    ncell = Hc * Wc
    ref = S.native_refresh([0], n)
    scored = np.arange(n) >= 1
    cover = np.ones((n, ncell), bool)
    cover[0] = False
    g = np.zeros((n, ncell, 2))
    g[0] = np.nan
    return M, nct, ncell, ref, scored, cover, g


def run(dyn, M, nct, ref, scored, cover, g):
    valid, moved = S.cell_truth(g, cover, ref)
    return S.policy_counts(dyn, M, nct, ref, scored, np.ones(len(ref), bool), valid, cover, moved, debug=True)


def t_static():
    M, nct, ncell, ref, scored, cover, g = scene()
    dyn = np.zeros((20, ncell), bool)
    dyn[0] = True  # I-frame: no vectors
    d = run(dyn, M, nct, ref, scored, cover, g)
    R = d["_reused"]
    nonref = scored & ~ref
    check("static: every token reused at every non-refresh frame", R[nonref].all(),
          f"reused token-frames {int(R[nonref].sum())} of {int(nonref.sum()) * M.shape[1]}")
    check("static: nothing reused at refresh frames 0 and 16", not R[ref].any())
    check("static: nothing stale", d["stale_tf"] == 0 and d["moved_tf"] == 0, f"stale {d['stale_tf']}")
    check("static: REUSE = 18/19", abs(S.ratios(d)["REUSE"] - 18 / 19) < 1e-12, f"REUSE {S.ratios(d)['REUSE']:.6f}")


def t_one_cell():
    M, nct, ncell, ref, scored, cover, g = scene(n=40)
    dyn = np.zeros((40, ncell), bool)
    dyn[0] = True
    c = 2 * 8 + 5            # cell (row 2, col 5) -> token (row 1, col 2) = token 1 * 4 + 2 = 6
    dyn[3, c] = True         # |MV| >= tau at frame 3 only
    d = run(dyn, M, nct, ref, scored, cover, g)
    R = d["_reused"]
    tok = 6
    active = [t for t in range(40) if not R[t, tok] and not ref[t]]
    check("one cell: token active from frame 3 to 15 (the frame before the next refresh)", active == list(range(3, 16)),
          f"active frames {active}")
    check("one cell: token reused again after refresh 16 (frames 17-31)", R[17:32, tok].all())
    others = np.delete(np.arange(M.shape[1]), tok)
    check("one cell: no other token ever active", R[np.ix_(scored & ~ref, others)].all())


def t_no_vector():
    # Sintel form: dyn = ~has | mag >= tau ; CCTV form: q >= 4 tau with q = 255 for no vector
    has = np.ones((5, 4), bool)
    mag = np.zeros((5, 4))
    has[2, 1] = False
    mag[2, 1] = np.nan
    dyn_s = ~has | (mag >= 0.5)
    q = np.zeros((5, 4), np.uint8)
    q[2, 1] = 255
    dyn_c = q >= 4 * 0.5
    check("no-vector cell is dynamic (Sintel form)", dyn_s[2, 1] and dyn_s.sum() == 1)
    check("no-vector cell is dynamic (CCTV form, q = 255)", dyn_c[2, 1] and dyn_c.sum() == 1)
    M, nct, ncell, ref, scored, cover, g = scene()
    dyn = np.zeros((20, ncell), bool)
    dyn[0] = True
    dyn[5, 0] = True  # the no-vector cell, as the policy sees it
    d = run(dyn, M, nct, ref, scored, cover, g)
    check("no-vector cell: its token not reused at frame 5", not d["_reused"][5, 0] and d["_reused"][4, 0])


def t_refresh():
    r = np.nonzero(S.native_refresh([0, 5, 30], 50))[0].tolist()
    check("refresh at every I-frame and 16 frames after the previous refresh (counter restarts)",
          r == [0, 5, 21, 30, 46], f"refresh {r}")
    r2 = np.nonzero(S.native_refresh([0], 50))[0].tolist()
    check("refresh with I-frame at 0 only", r2 == [0, 16, 32, 48], f"refresh {r2}")
    r3 = np.nonzero(S.native_refresh([0, 250], 300))[0].tolist()
    check("CCTV schedule I at 0, 250", r3 == list(range(0, 241, 16)) + [250, 266, 282, 298], f"refresh {r3}")
    sm, rf = S.fps2_schedule([0, 250], 300, 12)
    check("2 FPS schedule (step 12, I at 0 and 250)", np.nonzero(rf)[0].tolist() == [0, 192, 252],
          f"refresh {np.nonzero(rf)[0].tolist()}, sampled {int(sm.sum())}")


def t_truth():
    n, ncell = 20, 4
    cover = np.ones((n, ncell), bool)
    cover[0] = False
    g = np.zeros((n, ncell, 2))
    g[0] = np.nan
    g[1:, 0, 0] = 0.3                      # cell 0 moves 0.3 px / frame in x
    ref = S.native_refresh([0], n)
    valid, moved = S.cell_truth(g, cover, ref)
    first = int(np.argmax(moved[:, 0]))
    sums = np.cumsum(np.r_[0.0, np.full(n - 1, 0.3)])
    expect = int(np.argmax(sums > 2.0))
    check("truth: 0.3 px/frame cell first 'truly moved' where the sum first exceeds 2 px", first == expect == 7,
          f"first moved frame {first}, expected {expect} (sum {sums[expect]:.1f})")
    check("truth: stays moved until refresh 16, cleared at 16", moved[7:16, 0].all() and not moved[16, 0]
          and moved[16 + 7, 0] if n > 23 else moved[7:16, 0].all() and not moved[16, 0])
    g2 = g.copy()
    g2[4, 1] = np.nan                      # one missing GT term for cell 1 at frame 4
    v2, _ = S.cell_truth(g2, cover, ref)
    check("finite rule: a missing term invalidates the cell until the next refresh",
          v2[1:4, 1].all() and not v2[4:16, 1].any() and v2[16:, 1].all())


def t_boundary():
    M = S.token_matrix(109, 256, 436, 1024, "derived").tocsr()
    toks = lambda cy, cx=0: sorted(M[cy * 256 + cx].indices.tolist())  # noqa: E731
    check("27.25-px boundary: cell row 6 (y 24-28 px) is in token rows 0 and 1", toks(6) == [0, 16], f"{toks(6)}")
    check("cell rows 5 and 7 are in one token row each", toks(5) == [0] and toks(7) == [16], f"{toks(5)} {toks(7)}")
    check("cell row 13 (y 52-56, boundary 54.5) shared by token rows 1 and 2", toks(13) == [16, 32], f"{toks(13)}")
    check("64-px columns: cell col 16 only in token col 1", toks(0, 16) == [1], f"{toks(0, 16)}")
    nct = np.asarray(M.sum(axis=0)).ravel()
    check("every cell in >= 1 token; 256 tokens", (np.asarray(M.sum(axis=1)).ravel() >= 1).all() and M.shape[1] == 256,
          f"cells per token min {nct.min():g} max {nct.max():g}")
    Mc = S.token_matrix(180, 320, 720, 1280, "derived")
    nc = np.asarray(Mc.sum(axis=0)).ravel()
    # 45 px = 11.25 cells; a token starting at offset 0-3 px inside a cell always overlaps 12 cell rows
    check("CCTV derived: 80 x 45 px tokens -> 20 cols x 12 rows of cells (240) for every token", set(nc) == {240.0},
          f"{set(nc)}")


def main():
    for f in (t_static, t_one_cell, t_no_vector, t_refresh, t_truth, t_boundary):
        try:
            f()
        except Exception as e:  # a crash is a failure
            RESULTS.append((f.__name__ + " CRASHED", False, repr(e)))
    pv = S.provenance()
    L = ["# Validation 2a: synthetic unit tests",
         f"prereg {pv['prereg_commit']} (file unchanged: {pv['prereg_unchanged']}); code sha256 {pv['code_sha256']}", ""]
    L += [f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   [{det}]" if det else "") for name, ok, det in RESULTS]
    nf = sum(not ok for _, ok, _ in RESULTS)
    L += ["", f"{len(RESULTS) - nf}/{len(RESULTS)} passed"]
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, "test_results.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()
