"""Truth pilot (PLAN.md, e7ac3b0): run one candidate (DOT or CoTracker3 online) on the 3 fixed clips and save the
candidate displacement D(c, t) of the 3,600 query cells, per refresh period. Computes no gate; see gates.py.

Out: <out>/data/<clip>/<cand>.npz  disp (L, 3600, 2) float32 native px, row t = window-relative native frame t,
     displacement from the period start r(t) = 240 * (t // 240); rows r are 0.  <out>/<cand>_manifest.json.
Code and weights of the candidates live outside the repository (EXT below) and are never committed.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

import av
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
EXT = os.path.join(os.path.dirname(REPO), "iris_ext")
CKPT = os.path.join(EXT, "ckpt")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "virat"))
from run_virat import VIRAT  # noqa: E402

PLAN_COMMIT = "e7ac3b0"
CLIPS = ["VIRAT_S_000206_07_001501_001600", "VIRAT_S_000206_08_001618_001712", "VIRAT_S_000207_00_000000_000045"]
KEYINT, C, HN, WN = 240, 4, 720, 1280
SEED = 0


def windows():
    import csv
    inv = {r["clip"]: r for r in csv.DictReader(open(os.path.join(os.path.dirname(HERE), "codecsight_2fps", "inventory.csv")))}
    out = {}
    for c in CLIPS:
        r = inv[c]
        assert r["rule"] == "unseen" and int(r["win_first"]) == 600 and int(r["k"]) == 15, c
        out[c] = {"start": int(r["win_first"]), "L": int(r["win_last"]) - int(r["win_first"]) + 1}
    return out


def query_points():
    """3,600 query cells (cy, cx multiples of 4 on the 180x320 grid); native px (x, y) = cell centre."""
    cy, cx = np.meshgrid(np.arange(0, 180, 4), np.arange(0, 320, 4), indexing="ij")
    cells = (cy * 320 + cx).ravel()
    xy = np.stack([C * cx.ravel() + 1.5, C * cy.ravel() + 1.5], -1).astype(np.float32)
    return cells, xy


def periods(L):
    return [(r, min(r + KEYINT - 1, L - 1)) for r in range(0, L, KEYINT)]


def decode_periods(clip, w):
    """Yield (r, end, frames uint8 (n, 720, 1280, 3)) per period; rgb24, display order (as collect.raft_clip)."""
    P = periods(w["L"])
    pi, buf = 0, []
    with av.open(os.path.join(VIRAT, clip + ".mp4")) as c:
        for i, fr in enumerate(c.decode(c.streams.video[0])):
            t = i - w["start"]
            if t < 0:
                continue
            buf.append(fr.to_ndarray(format="rgb24"))
            if t == P[pi][1]:
                yield P[pi][0], P[pi][1], np.stack(buf)
                buf, pi = [], pi + 1
                if pi == len(P):
                    return
    raise RuntimeError(f"{clip}: decode ended early")


# --------------------------------------------------------------------------- DOT
class DOT:
    name = "dot"

    def __init__(self, height, width):
        sys.path.insert(0, os.path.join(EXT, "dot"))
        import argparse as ap
        from dot.models import create_model
        from dot.utils.options.demo_options import DemoOptions
        from dot.utils.torch import get_grid
        self.get_grid = get_grid
        p = ap.ArgumentParser()
        DemoOptions().initialize(p)
        cfg = os.path.join(EXT, "dot", "configs")
        self.args = p.parse_args([
            "--height", str(height), "--width", str(width),
            "--inference_mode", "tracks_from_first_to_every_other_frame",
            # pytorch3d is not installed: DOT's documented pure-torch interpolation path
            "--interpolation_version", "torch",
            "--estimator_config", os.path.join(cfg, "raft_patch_8.json"),
            "--estimator_path", os.path.join(CKPT, "cvo_raft_patch_8.pth"),
            "--refiner_config", os.path.join(cfg, "raft_patch_4_alpha.json"),
            "--refiner_path", os.path.join(CKPT, "movi_f_raft_patch_4_alpha.pth"),
            "--tracker_config", os.path.join(cfg, "cotracker2_patch_4_wind_8.json"),
            "--tracker_path", os.path.join(CKPT, "movi_f_cotracker2_patch_4_wind_8.pth")])
        self.H, self.W = height, width
        self.model = create_model(self.args).cuda().eval()
        self.config = {k: v for k, v in vars(self.args).items()}

    @torch.no_grad()
    def run(self, frames, xy):
        """Mirrors DenseOpticalTracker.get_tracks_from_first_to_every_other_frame step by step (same calls, same
        order); the only change is that each target step's dense flow is sampled at the query points and moved to
        the CPU instead of being stacked on the GPU. Returns disp (n, N, 2) native px."""
        H, W = self.H, self.W
        T = frames.shape[0]
        video = torch.empty(T, 3, H, W, device="cuda")
        for a in range(0, T, 16):  # read_video: /255 after bilinear resize of the uint8->float frames
            ch = torch.from_numpy(frames[a:a + 16]).cuda().permute(0, 3, 1, 2).float()
            video[a:a + 16] = F.interpolate(ch, size=(H, W), mode="bilinear") / 255.
        video = video[None]
        kw = vars(self.args)
        init = self.model.point_tracker({"video": video}, mode="tracks_at_motion_boundaries", **kw)["tracks"]
        init = torch.stack([init[..., 0] / (W - 1), init[..., 1] / (H - 1), init[..., 2]], dim=-1)
        src_points, src_frame = init[:, 0], video[:, 0]
        # query points at inference resolution (PLAN §2), then grid_sample coordinates (align_corners=True)
        qx = (xy[:, 0] + 0.5) * W / WN - 0.5
        qy = (xy[:, 1] + 0.5) * H / HN - 0.5
        g = torch.from_numpy(np.stack([qx / (W - 1) * 2 - 1, qy / (H - 1) * 2 - 1], -1)).float().cuda()[None, None]
        scale = np.array([WN / W, HN / H], np.float32)
        out = np.zeros((T, xy.shape[0], 2), np.float32)
        for tgt in range(1, T):
            data = {"src_frame": src_frame, "tgt_frame": video[:, tgt], "src_points": src_points,
                    "tgt_points": init[:, tgt]}
            flow = self.model.optical_flow_refiner(data, mode="flow_with_tracks_init", **kw)["flow"]  # (1,H,W,2)
            s = F.grid_sample(flow.permute(0, 3, 1, 2), g, mode="bilinear", align_corners=True)[0, :, 0].T
            out[tgt] = s.cpu().numpy() * scale
        return out


# --------------------------------------------------------------------------- CoTracker3 online
class CoTracker3Online:
    name = "cotracker3"

    def __init__(self, qbatch):
        sys.path.insert(0, os.path.join(EXT, "co-tracker"))
        from cotracker.predictor import CoTrackerOnlinePredictor
        self.model = CoTrackerOnlinePredictor(checkpoint=os.path.join(CKPT, "scaled_online.pth")).cuda()
        self.qbatch = qbatch
        self.config = {"checkpoint": "scaled_online.pth", "window_len": 16, "step": self.model.step,
                       "interp_shape": list(self.model.interp_shape), "qbatch": qbatch}

    @torch.no_grad()
    def _track(self, frames, q):
        """online_demo.py loop: chunks of 2*step frames every step frames, then the final partial chunk."""
        m, step = self.model, self.model.step
        queries = torch.from_numpy(np.concatenate([np.zeros((len(q), 1), np.float32), q], 1)).cuda()[None]

        def proc(idx, first):
            ch = torch.from_numpy(frames[idx]).cuda().float().permute(0, 3, 1, 2)[None]
            return m(ch, is_first_step=first, queries=queries if first else None)

        first, n = True, frames.shape[0]
        tr = None
        for i in range(n):
            if i % step == 0 and i != 0:
                tr, _ = proc(np.arange(max(0, i - 2 * step), i), first)
                first = False
        tr, _ = proc(np.arange(n - (n - 1) % step - step - 1, n), first)
        tr = tr[0].cpu().numpy()
        assert tr.shape[0] == n, (tr.shape, n)
        return tr

    def run(self, frames, xy):
        out = np.zeros((frames.shape[0], xy.shape[0], 2), np.float32)
        for a in range(0, xy.shape[0], self.qbatch):
            tr = self._track(frames, xy[a:a + self.qbatch])
            out[:, a:a + self.qbatch] = tr - tr[:1]
        return out


# ---------------------------------------------------------------------------
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a, cwd=REPO):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["dot", "cotracker3"])
    ap.add_argument("--dot_res", default="480x856")
    ap.add_argument("--qbatch", type=int, default=3600)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    T0 = time.perf_counter()
    if a.cand == "dot":
        h, w_ = map(int, a.dot_res.split("x"))
        cand = DOT(h, w_)
        ext = os.path.join(EXT, "dot")
        ck = ["cvo_raft_patch_8.pth", "movi_f_raft_patch_4_alpha.pth", "movi_f_cotracker2_patch_4_wind_8.pth"]
    else:
        cand = CoTracker3Online(a.qbatch)
        ext = os.path.join(EXT, "co-tracker")
        ck = ["scaled_online.pth"]
    name = cand.name + a.tag
    W_ = windows()
    cells, xy = query_points()
    man = {"plan_commit": PLAN_COMMIT, "head": git("rev-parse", "HEAD"),
           "uncommitted": git("status", "--porcelain", "--", "p3/truth_pilot"),
           "candidate": name, "repo_commit": git("rev-parse", "HEAD", cwd=ext),
           "checkpoints": {c: sha(os.path.join(CKPT, c)) for c in ck}, "config": cand.config, "seed": SEED,
           "argv": sys.argv, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "clips": {}}
    mp = os.path.join(a.out, f"{name}_manifest.json")
    os.makedirs(a.out, exist_ok=True)
    for clip in CLIPS:
        w = W_[clip]
        rec = man["clips"][clip] = {"L": w["L"], "periods": []}
        try:
            disp = np.zeros((w["L"], len(cells), 2), np.float32)
            torch.cuda.reset_peak_memory_stats()
            for r, end, frames in decode_periods(clip, w):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                disp[r:end + 1] = cand.run(frames, xy)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                rec["periods"].append({"r": r, "end": end, "frames": int(frames.shape[0]), "infer_s": dt})
                print(f"{time.strftime('%H:%M:%S')} {name} {clip} period {r}-{end}: {dt:.0f}s", flush=True)
                torch.cuda.empty_cache()
            rec.update(status="ok", infer_s=sum(p["infer_s"] for p in rec["periods"]),
                       peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30,
                       finite=bool(np.isfinite(disp).all()))
            d = os.path.join(a.out, "data", clip)
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, f"{name}.npz"), disp=disp, cells=cells, xy=xy)
        except Exception as e:
            rec.update(status="FAILED", error=repr(e), traceback=traceback.format_exc(),
                       peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30)
            print(f"{name} {clip} FAILED {e!r}", flush=True)
        json.dump(man, open(mp, "w"), indent=1)
    man.update(end_time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), wall_s=time.perf_counter() - T0)
    json.dump(man, open(mp, "w"), indent=1)
    print(f"done in {man['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
