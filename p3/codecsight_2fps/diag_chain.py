"""Post-results diagnostic of the R-chain (DEVIATIONS.md entry 4). Changes no gate, threshold or result.

On one real VIRAT frame (the first sampled frame of the first inventory clip), with collect.py's RAFT (load_raft:
raft_large DEFAULT weights, 12 updates, float32, batch 1, weights.transforms()) and collect.py's chain composition
(F <- F + f_i(x + F), bilinear grid_sample, align_corners, border padding, a pixel whose path leaves the frame is
dropped; D = direct RAFT frame 0 -> frame 12; r = |D - F| / |F| over in-frame pixels with |F| > 2 px):
  (a) global shift: 13 frames, frame t = frame 0 shifted t px right (integer shift, edge padding). Truth: 12 px.
      Stats over interior pixels (32 px margin on every side, and x >= 12 + 32 so no edge-padded column).
  (b) small object: a 24 x 48 px (w x h) patch cut from the same frame is pasted onto the static frame 0 and moved
      1 px right per frame, 12 steps. Stats over the patch's pixels at t = 0. Truth: 12 px.
Pass rule for (a), fixed before running: |median D_x - 12| < 0.5 px and |median F_x - 12| < 0.5 px.

Usage: python diag_chain.py   (writes diag_chain.txt next to this file)
"""
import os
import sys

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import collect  # noqa: E402

N = 12  # steps; 13 frames
MARGIN = 32
PW, PH = 24, 48  # patch width, height


def load_frame():
    w = collect.windows()
    clip = sorted(w)[0]
    s0 = collect.sampled_index(w[clip])[0]
    with av.open(os.path.join(collect.VIRAT, clip + ".mp4")) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i == s0:
                return clip, s0, fr.to_ndarray(format="rgb24")


def shift_right(img, t):
    return np.ascontiguousarray(np.pad(img, ((0, 0), (t, 0), (0, 0)), mode="edge")[:, :img.shape[1]])


def run_chain(frames, model, tf, torch):
    """collect.raft_clip's flow() and chain composition, over frames[0] -> ... -> frames[-1]."""
    import torch.nn.functional as Fn
    H, W = frames[0].shape[:2]

    def flow(a, b):
        ta = torch.from_numpy(a).permute(2, 0, 1)[None]
        tb = torch.from_numpy(b).permute(2, 0, 1)[None]
        ta, tb = tf(ta, tb)
        return model(ta.cuda(), tb.cuda())[-1][0].float()

    with torch.inference_mode():
        ys, xs = torch.meshgrid(torch.arange(H, device="cuda", dtype=torch.float32),
                                torch.arange(W, device="cuda", dtype=torch.float32), indexing="ij")
        F = torch.zeros((2, H, W), device="cuda")
        inside = torch.ones((H, W), dtype=torch.bool, device="cuda")
        for i in range(1, len(frames)):
            f = flow(frames[i - 1], frames[i])
            px, py = xs + F[0], ys + F[1]
            inside &= (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
            grid = torch.stack([2 * px / (W - 1) - 1, 2 * py / (H - 1) - 1], -1)[None]
            F = F + Fn.grid_sample(f[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
        D = flow(frames[0], frames[-1])
        px, py = xs + F[0], ys + F[1]
        ok = inside & (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
        fm = torch.hypot(F[0], F[1])
        e = torch.hypot(D[0] - F[0], D[1] - F[1])
        r = e / fm.clamp_min(1e-9)
    return D.cpu().numpy(), F.cpu().numpy(), ok.cpu().numpy(), (fm > 2.0).cpu().numpy(), e.cpu().numpy(), r.cpu().numpy()


def summarize(name, sel, D, F, ok, mov, e, r):
    m = sel & ok & mov
    med = lambda a: float(np.median(a)) if a.size else float("nan")  # noqa: E731
    return [f"{name}",
            f"  pixels selected {int(sel.sum())}, in frame {int((sel & ok).sum())}, |F| > 2 px (r computed) {int(m.sum())}",
            f"  median D: x {med(D[0][sel]):+.3f}  y {med(D[1][sel]):+.3f}  |D| {med(np.hypot(D[0], D[1])[sel]):.3f} px",
            f"  median F: x {med(F[0][sel & ok]):+.3f}  y {med(F[1][sel & ok]):+.3f}  "
            f"|F| {med(np.hypot(F[0], F[1])[sel & ok]):.3f} px",
            f"  median e = |D - F| {med(e[m]):.3f} px, median r = |D - F| / |F| {med(r[m]):.4f}"], med(D[0][sel]), med(F[0][sel & ok])


def main():
    torch, model, tf, info = collect.load_raft()
    clip, s0, img = load_frame()
    H, W = img.shape[:2]
    out = [f"diag_chain.py: post-results R-chain diagnostic (DEVIATIONS.md entry 4); changes no gate or threshold",
           f"frame: {clip} source frame {s0}, {W}x{H}; RAFT {info['weights']}, 12 updates, float32",
           f"truth: {N} px right in both tests (13 frames, {N} composed native steps; collect.py's chain uses k = 15)", ""]

    # (a) global shift
    frames = [shift_right(img, t) for t in range(N + 1)]
    D, F, ok, mov, e, r = run_chain(frames, model, tf, torch)
    sel = np.zeros((H, W), bool)
    sel[MARGIN:H - MARGIN, N + MARGIN:W - MARGIN] = True
    lines, dx, fx = summarize(f"(a) global shift, interior pixels (margin {MARGIN} px, x >= {N + MARGIN})",
                              sel, D, F, ok, mov, e, r)
    pass_a = abs(dx - N) < 0.5 and abs(fx - N) < 0.5
    out += lines + [f"  pass rule |median D_x - 12| < 0.5 and |median F_x - 12| < 0.5: {'PASS' if pass_a else 'FAIL'}", ""]

    # (b) small object: the most textured 24x48 patch (max grey std on a 24 px grid, away from the paste site)
    grey = img.astype(np.float64).mean(2)
    y0, x0 = H // 2 - PH // 2, W // 2 - PW // 2 - N // 2  # paste site at t = 0 (centre of frame)
    best = None
    for yy in range(MARGIN, H - MARGIN - PH, 24):
        for xx in range(MARGIN, W - MARGIN - PW, 24):
            if abs(yy - y0) < 2 * PH and abs(xx - x0) < 4 * PW:
                continue
            s = grey[yy:yy + PH, xx:xx + PW].std()
            if best is None or s > best[0]:
                best = (s, yy, xx)
    _, py0, px0 = best
    patch = img[py0:py0 + PH, px0:px0 + PW].copy()
    frames = []
    for t in range(N + 1):
        fr = img.copy()
        fr[y0:y0 + PH, x0 + t:x0 + t + PW] = patch
        frames.append(fr)
    D, F, ok, mov, e, r = run_chain(frames, model, tf, torch)
    sel = np.zeros((H, W), bool)
    sel[y0:y0 + PH, x0:x0 + PW] = True
    lines, _, _ = summarize(f"(b) small object, {PW}x{PH} px patch from (x {px0}, y {py0}), pasted at (x {x0}, y {y0}) "
                            f"at t = 0, 1 px/frame right; patch pixels at t = 0", sel, D, F, ok, mov, e, r)
    bg = np.zeros((H, W), bool)
    bg[MARGIN:H - MARGIN, MARGIN:W - MARGIN] = True
    bg[y0 - MARGIN:y0 + PH + MARGIN, x0 - MARGIN:x0 + N + PW + MARGIN] = False
    out += lines + [f"  background (static, >= {MARGIN} px from the patch path): median |D| "
                    f"{np.median(np.hypot(D[0], D[1])[bg]):.3f} px, median |F| {np.median(np.hypot(F[0], F[1])[bg & ok]):.3f} px",
                    ""]
    out.append(f"(a) {'PASS' if pass_a else 'FAIL'}")
    txt = "\n".join(out) + "\n"
    print(txt, end="")
    open(os.path.join(HERE, "diag_chain.txt"), "w").write(txt)


if __name__ == "__main__":
    main()
