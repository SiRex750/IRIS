"""Review-2 item 8(b), compute step: RAFT BACKWARD flow (t -> t-1) on the confirmatory test's source frames.
ROBUSTNESS ANALYSIS ONLY - no new verdict. This is the only new compute in p3/review_fixes2/.

Same model, weights, preprocessing and frames as the forward RAFT of the confirmatory collection
(p3/virat/run_virat.py raft_all, run into p3/virat_confirm/): torchvision raft_large DEFAULT weights
(C_T_SKHT_V2), 12 flow updates, float32, batch 1, decoded source frames 300..599 (or to the end of the clip)
-> rgb24 -> weights.transforms(). The only change is the input order: model(frame t, frame t-1), giving flow
t -> t-1 on frame t's pixel grid.

Stored per clip (eligible clips only, from p3/virat_confirm/eligibility_v0.csv), in _raft_bwd/<clip>.npz
(not committed, *.npz is git-ignored):
  bwd_t (n-1, Hc, Wc, 2) f16  mean backward flow over each 4x4 cell of frame t; row i is t = i + 1
Run detached: python i08b_raft_backward_compute.py  (resumes: clips with an existing file are skipped).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, P3)
sys.path.insert(0, os.path.join(P3, "virat"))
sys.path.insert(0, os.path.join(P3, "pilot_v2"))
sys.path.insert(0, os.path.join(P3, "pilot"))

from run_virat import decode_source, VIRAT  # noqa: E402

CONF = os.path.join(P3, "virat_confirm")
OUT = os.path.join(HERE, "_raft_bwd")
C = 4


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(HERE, "i08b_raft_backward_compute.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    import torch
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    os.makedirs(OUT, exist_ok=True)
    man = json.load(open(os.path.join(CONF, "manifest.json")))
    start, min_n, nfr = int(man["config"]["start"]), int(man["config"]["min_frames"]), int(man["config"]["raft_frames"])
    el = pd.read_csv(os.path.join(CONF, "eligibility_v0.csv"))
    clips = sorted(el.loc[el.eligible.astype(bool), "clip"])
    w = Raft_Large_Weights.DEFAULT
    assert str(w) == man["raft"]["weights"], (str(w), man["raft"]["weights"])
    model = raft_large(weights=w, progress=False).eval().cuda()
    tf = w.transforms()
    log(f"start: {len(clips)} eligible clips, weights {w}, frames from {start}")
    for clip in clips:
        out = os.path.join(OUT, f"{clip}.npz")
        if os.path.exists(out):
            log(f"{clip} exists, skipped")
            continue
        t0 = time.perf_counter()
        rgb, _ = decode_source(os.path.join(VIRAT, clip + ".mp4"), nfr, fmt="rgb24", start=start, min_n=min_n)
        n = len(rgb)
        H, W = rgb[0].shape[:2]
        Hc, Wc = -(-H // C), -(-W // C)
        fz = np.load(os.path.join(CONF, "data", clip, "raft.npz"))
        assert fz["splat_t"].shape[:3] == (n - 1, Hc, Wc), (fz["splat_t"].shape, n, Hc, Wc)
        cnt = np.zeros((Hc * C, Wc * C))
        cnt[:H, :W] = 1
        cnt = cnt.reshape(Hc, C, Wc, C).sum((1, 3))
        pad = np.zeros((Hc * C, Wc * C, 2))
        bwd = np.empty((n - 1, Hc, Wc, 2), np.float16)
        with torch.inference_mode():
            for t in range(1, n):
                a = torch.from_numpy(rgb[t]).permute(2, 0, 1)[None]
                b = torch.from_numpy(rgb[t - 1]).permute(2, 0, 1)[None]
                a, b = tf(a, b)
                fl = model(a.cuda(), b.cuda())[-1][0].float().cpu().numpy().transpose(1, 2, 0)
                pad[:H, :W] = fl
                bwd[t - 1] = (pad.reshape(Hc, C, Wc, C, 2).sum((1, 3)) / cnt[..., None]).astype(np.float16)
        del rgb
        np.savez_compressed(out, bwd_t=bwd, meta=np.array(json.dumps(
            {"clip": clip, "start_frame": start, "n": n, "H": H, "W": W, "Hc": Hc, "Wc": Wc, "weights": str(w),
             "bwd_t": "RAFT flow t -> t-1 (u, v) px, mean over each 4x4 cell of frame t's grid; row i is t = i+1"})))
        torch.cuda.empty_cache()
        log(f"{clip} {n - 1} pairs {time.perf_counter() - t0:.0f}s")
    log("done")


if __name__ == "__main__":
    main()
