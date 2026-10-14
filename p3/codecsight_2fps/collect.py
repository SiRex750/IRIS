"""CodecSight-matched 2 fps arm: data collection ONLY, as fixed by p3/codecsight_2fps/PREREGISTRATION.md (cc2eaf9).
No policy, gate, stale or motion-count metric is computed here.

Per clip (windows from inventory.csv): source frames s_j = win_first + j k, j = 0..16 G - 1.
  Encodes  the sampled frames as decoded yuv420p, one 2 fps stream per arm (keyint 16, the 9 confirmatory arms,
           NVENC no-scenecut 1). Reuses p3/virat/run_virat.py: write_arm_npz (same npz layout, so
           codecsight_sim/simulate.py's loaders read it), check (run_sweep.check_encode), qp_stats; slices via
           run_pilot_v2.slices_per_frame; vectors via lib.mvs.extract_mvs with PyAV's default threading.
           Added checks: I-frames exactly at 0, 16, 32, ...; bf 0 arms: every non-I frame is P with d_past = 1;
           no avcodec option left unconsumed after the encoder opened.
  RAFT     raft_large (DEFAULT weights), 12 updates, float32, batch 1, decoded source -> rgb24 -> weights.transforms(),
           as run_virat.raft_all. One sequential decode pass per clip:
           - forward s_(j-1) -> s_j: flow_tm1, splat_t, splat_count (run_virat layout; row i is j = i + 1)
           - backward s_j -> s_(j-1): bwd_t = mean over each 4x4 cell of frame j (i08b_raft_backward_compute layout)
           - R-chain: for j = 16 g + 1..16 g + 4, native RAFT over every source pair between s_(j-1) and s_j,
             composed along each pixel's path (F <- F + f_i(x + F), bilinear, align_corners grid); a pixel whose path
             leaves the frame is dropped. Saved per chain pair: pixels with |F| > 2 px, r = |D - F| / |F| and
             e = |D - F| (D = the direct forward flow of that pair), and their exact medians.
Outputs (under --out): data/<clip>/<arm>.npz, raft.npz, chain.npz; encodes/<clip>/<arm>.mp4; manifest.json.

Usage: python collect.py [--clips a,b] [--out DIR] [--workers N] [--no-raft] [--overwrite]
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback

import av
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
REPO = os.path.dirname(P3)
sys.path.insert(0, os.path.join(P3, "virat"))
import run_virat as rv  # noqa: E402  (sets up the p3 / sweep / pilot paths)
from run_virat import VIRAT, C, write_arm_npz, check, qp_stats, save_npz  # noqa: E402
from run_pilot_v2 import slices_per_frame  # noqa: E402
from run_sweep import CRF_SERIES, X264_QP_PAIR, NVENC_SERIES, NVENC_B  # noqa: E402
from lib.encode import build_options, read_x264_sei, sei_value  # noqa: E402
from lib.mvs import extract_mvs  # noqa: E402
from lib.gt_grid import GTGrid  # noqa: E402

PREREG_COMMIT = "cc2eaf9"
FPS = 2
GOP = 16
CHAIN_PER_GOP = 4
ARM_IDS = ("x264_crf12", "x264_crf45", "x264_qp24", "x264_qp24_bf2_pb1", "nvenc_qp18", "nvenc_qp23",
           "nvenc_qp45", "nvenc_qp28", "nvenc_qp28_bf2_bqeq")


def arms():
    """The 9 confirmatory arms with keyint 16; NVENC also no_scenecut 1 (PREREGISTRATION.md §2)."""
    src = {e["id"]: e for e in CRF_SERIES + X264_QP_PAIR + NVENC_SERIES + NVENC_B}
    out = []
    for a in ARM_IDS:
        e = dict(src[a])
        kn = {**e["knobs"], "keyint": GOP}
        if e["codec"] == "h264_nvenc":
            kn["no_scenecut"] = 1
        e["knobs"] = kn
        out.append(e)
    return out


def windows():
    inv = pd.read_csv(os.path.join(HERE, "inventory.csv"))
    inv = inv[inv["gops"] > 0]
    return {r["clip"]: {"scene": r["scene"], "k": int(r["k"]), "start": int(r["win_first"]), "gops": int(r["gops"]),
                        "n": int(r["sampled_frames"]), "fps_src": str(r["avg_fps"])} for _, r in inv.iterrows()}


def sampled_index(w):
    return [w["start"] + j * w["k"] for j in range(w["n"])]


# --------------------------------------------------------------------------- source
_CACHE = {}


def sampled_yuv(clip, w):
    """Sampled source frames as yuv420p ndarrays (formatting as run_virat.decode_source); cached per process."""
    if clip in _CACHE:
        return _CACHE[clip]
    want = sampled_index(w)
    pos = {s: j for j, s in enumerate(want)}
    out = [None] * len(want)
    with av.open(os.path.join(VIRAT, clip + ".mp4")) as c:
        st = c.streams.video[0]
        for i, f in enumerate(c.decode(st)):
            if i in pos:
                if f.format.name != "yuv420p":
                    f = f.reformat(format="yuv420p")
                out[pos[i]] = np.ascontiguousarray(f.to_ndarray())
            if i >= want[-1]:
                break
    assert all(x is not None for x in out), (clip, "missing sampled frames")
    _CACHE.clear()
    _CACHE[clip] = out
    return out


# --------------------------------------------------------------------------- encode / extract
def encode_yuv(frames, out_path, codec, fps, **knobs):
    """run_virat.encode_yuv, plus the avcodec options the encoder did not consume (must be empty)."""
    opts = build_options(codec, **knobs)
    H, W = frames[0].shape[0] * 2 // 3, frames[0].shape[1]
    t0 = time.perf_counter()
    packet_bytes = n = 0
    with av.open(out_path, mode="w") as out:
        st = out.add_stream(codec, rate=fps, options=dict(opts))
        st.width, st.height, st.pix_fmt = W, H, "yuv420p"
        for a in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(a, format="yuv420p")):
                packet_bytes += pkt.size
                out.mux(pkt)
            n += 1
        unused = dict(st.codec_context.options)
        for pkt in st.encode(None):
            packet_bytes += pkt.size
            out.mux(pkt)
    return {"options": opts, "n_frames": n, "file_bytes": os.path.getsize(out_path), "packet_bytes": packet_bytes,
            "seconds": time.perf_counter() - t0, "unused_options": unused,
            "x264_sei": read_x264_sei(out_path) if codec == "libx264" else None}


def run_arm(task):
    clip, w, enc, out_dir, overwrite = task
    d_dir = os.path.join(out_dir, "data", clip)
    e_dir = os.path.join(out_dir, "encodes", clip)
    os.makedirs(d_dir, exist_ok=True)
    os.makedirs(e_dir, exist_ok=True)
    npz = os.path.join(d_dir, f"{enc['id']}.npz")
    lg = {"clip": clip, "arm": enc["id"], "codec": enc["codec"], "knobs": enc["knobs"]}
    if os.path.exists(npz) and not overwrite:
        lg["status"] = "skipped_exists"
        return lg
    t0 = time.perf_counter()
    try:
        src = sampled_yuv(clip, w)
        H, W = src[0].shape[0] * 2 // 3, src[0].shape[1]
        Hc, Wc = -(-H // C), -(-W // C)
        path = os.path.join(e_dir, f"{enc['id']}.mp4")
        rec = encode_yuv(src, path, enc["codec"], FPS, **enc["knobs"])
        t_enc = time.perf_counter() - t0
        frames, t_mv = extract_mvs(path, want_qp=True)
        if len(frames) != len(src):
            raise RuntimeError(f"decoded {len(frames)} frames, sampled {len(src)}")
        spf = slices_per_frame(path, enc["codec"])
        psnr = []
        with av.open(path) as c:
            for i, f in enumerate(c.decode(video=0)):
                if f.format.name != "yuv420p":
                    f = f.reformat(format="yuv420p")
                mse = np.mean((f.to_ndarray()[:H].astype(np.float64) - src[i][:H]) ** 2)
                psnr.append(100.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
        sidx = sampled_index(w)
        types, n_qp = write_arm_npz(npz, clip, enc["id"],
                                    {"codec": enc["codec"], "options": rec["options"], "knobs": enc["knobs"],
                                     "fps": str(FPS), "x264_sei": rec["x264_sei"], "source_frames": sidx,
                                     "k": w["k"], "gops": w["gops"], "source_fps": w["fps_src"]},
                                    frames, Hc, Wc, enc.get("d_known", True))
        qs, qvals = qp_stats(frames)
        n = len(frames)
        z = np.load(npz)
        d_past = z["d_past"]
        iframes = [t for t, ft in enumerate(types) if ft == "I"]
        lg.update({"status": "ok", "options": rec["options"], "x264_sei": rec["x264_sei"],
                   "unused_options": rec["unused_options"], "frames_encoded": rec["n_frames"], "frames_decoded": n,
                   "file_bytes": rec["file_bytes"], "bitrate_kbps": rec["packet_bytes"] * 8 * FPS / n / 1000.0,
                   "psnr_y_mean": float(np.mean(psnr)), "psnr_y_min": float(np.min(psnr)),
                   "slices_min": int(min(spf)), "slices_max": int(max(spf)),
                   "sei_keyint": sei_value(rec["x264_sei"], "keyint"),
                   "sei_keyint_min": sei_value(rec["x264_sei"], "keyint_min"),
                   "sei_scenecut": sei_value(rec["x264_sei"], "scenecut"),
                   "n_I": types.count("I"), "n_P": types.count("P"), "n_B": types.count("B"),
                   "iframes": iframes, "ftypes": "".join(types), "qp_frames_available": n_qp, **qs,
                   "encode_s": t_enc, "mv_extract_s": t_mv, "run_s": time.perf_counter() - t0})
        bad = check(enc, lg, spf, qvals)
        want_i = list(range(0, n, GOP))
        if iframes != want_i:
            bad.append(f"I-frames at {iframes} (want {want_i})")
        if rec["unused_options"]:
            bad.append(f"unconsumed avcodec options {rec['unused_options']}")
        if enc["knobs"].get("bframes", 0) == 0:
            nonI = [t for t in range(n) if types[t] != "I"]
            if any(types[t] != "P" or d_past[t] != 1 for t in nonI):
                bad.append("bf 0 arm: a non-I frame is not P with d_past 1")
        lg["check_mismatches"] = bad
    except Exception as e:
        lg.update(status="FAILED", error=repr(e), traceback=traceback.format_exc(), run_s=time.perf_counter() - t0)
    return lg


# --------------------------------------------------------------------------- RAFT
def raft_clip(clip, w, out_dir, model, tf, torch):
    """Forward + backward 2 fps RAFT and the R-chain, in one sequential decode pass."""
    import torch.nn.functional as Fn
    sidx = sampled_index(w)
    n = len(sidx)
    pos = {s: j for j, s in enumerate(sidx)}
    chain_j = sorted(j for g in range(w["gops"]) for j in range(GOP * g + 1, GOP * g + 1 + CHAIN_PER_GOP) if j < n)
    native_t = {t for j in chain_j for t in range(sidx[j - 1] + 1, sidx[j] + 1)}

    def flow(a, b):
        ta = torch.from_numpy(a).permute(2, 0, 1)[None]
        tb = torch.from_numpy(b).permute(2, 0, 1)[None]
        ta, tb = tf(ta, tb)
        return model(ta.cuda(), tb.cuda())[-1][0].float()  # (2, H, W) on GPU

    t0 = time.perf_counter()
    H = W = None
    flow_tm1 = splat = scount = bwd = None
    chain_stats, chain_r, chain_e, chain_pair = [], [], [], []
    prev_native = prev_sampled = None
    F = inside = None
    n_native = 0
    with av.open(os.path.join(VIRAT, clip + ".mp4")) as c, torch.inference_mode():
        st = c.streams.video[0]
        for i, fr in enumerate(c.decode(st)):
            if i < sidx[0]:
                continue
            rgb = fr.to_ndarray(format="rgb24")
            if H is None:
                H, W = rgb.shape[:2]
                Hc, Wc = -(-H // C), -(-W // C)
                flow_tm1 = np.empty((n - 1, Hc, Wc, 2), np.float16)
                splat = np.empty((n - 1, Hc, Wc, 2), np.float16)
                scount = np.empty((n - 1, Hc, Wc), np.uint8)
                bwd = np.empty((n - 1, Hc, Wc, 2), np.float16)
                cnt = np.zeros((Hc * C, Wc * C))
                cnt[:H, :W] = 1
                cnt = cnt.reshape(Hc, C, Wc, C).sum((1, 3))
                pad = np.zeros((Hc * C, Wc * C, 2))
                nobad = np.zeros((H, W), bool)
                ys, xs = torch.meshgrid(torch.arange(H, device="cuda", dtype=torch.float32),
                                        torch.arange(W, device="cuda", dtype=torch.float32), indexing="ij")

                def cellmean(fl):
                    pad[:H, :W] = fl
                    return (pad.reshape(Hc, C, Wc, C, 2).sum((1, 3)) / cnt[..., None]).astype(np.float16)
            if i in native_t:  # R-chain step i-1 -> i, composed along the path
                f = flow(prev_native, rgb)
                px, py = xs + F[0], ys + F[1]
                inside &= (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
                grid = torch.stack([2 * px / (W - 1) - 1, 2 * py / (H - 1) - 1], -1)[None]
                F = F + Fn.grid_sample(f[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
                n_native += 1
            if i in pos:
                j = pos[i]
                if j >= 1:
                    D = flow(prev_sampled, rgb)
                    fl = D.cpu().numpy().transpose(1, 2, 0)
                    flow_tm1[j - 1] = cellmean(fl)
                    g = GTGrid(fl, nobad)
                    gx, gy, _, _ = g.cell_gt()
                    splat[j - 1] = np.stack([gx, gy], -1).astype(np.float16)
                    scount[j - 1] = np.minimum(g.count, 255).astype(np.uint8)
                    bwd[j - 1] = cellmean(flow(rgb, prev_sampled).cpu().numpy().transpose(1, 2, 0))
                    if j in chain_j:
                        px, py = xs + F[0], ys + F[1]
                        ok = inside & (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)
                        fm = torch.hypot(F[0], F[1])
                        m = ok & (fm > 2.0)
                        e = torch.hypot(D[0] - F[0], D[1] - F[1])[m]
                        r = e / fm[m]
                        chain_stats.append({"j": j, "src_from": sidx[j - 1], "src_to": sidx[j],
                                            "pixels_in_frame": int(ok.sum()), "pixels_moving": int(m.sum()),
                                            "median_r": float(r.median()) if r.numel() else np.nan,
                                            "median_e_px": float(e.median()) if e.numel() else np.nan})
                        chain_r.append(r.cpu().numpy().astype(np.float32))
                        chain_e.append(e.cpu().numpy().astype(np.float32))
                        chain_pair.append(np.full(r.numel(), j, np.int16))
                F = torch.zeros((2, H, W), device="cuda")
                inside = torch.ones((H, W), dtype=torch.bool, device="cuda")
                prev_sampled = rgb
            prev_native = rgb
            if i >= sidx[-1]:
                break
    d_dir = os.path.join(out_dir, "data", clip)
    os.makedirs(d_dir, exist_ok=True)
    meta = {"clip": clip, "source_frames": sidx, "k": w["k"], "gops": w["gops"], "H": H, "W": W, "Hc": Hc, "Wc": Wc,
            "cell": C, "row": "row i is 2 fps frame j = i + 1 (pair s_(j-1) -> s_j)",
            "flow_tm1": "RAFT s_(j-1) -> s_j, mean over each 4x4 cell of frame j-1's grid",
            "splat_t": "lib.gt_grid.GTGrid(flow, no bad mask).cell_gt(): forward splat onto frame j's grid",
            "splat_count": "pixels landing in the cell, clipped at 255",
            "bwd_t": "RAFT s_j -> s_(j-1), mean over each 4x4 cell of frame j's grid"}
    save_npz(os.path.join(d_dir, "raft.npz"), flow_tm1=flow_tm1, splat_t=splat, splat_count=scount, bwd_t=bwd,
             t=np.arange(1, n, dtype=np.int16), meta=np.array(json.dumps(meta)))
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt)  # noqa: E731
    save_npz(os.path.join(d_dir, "chain.npz"), r=cat(chain_r, np.float32), e=cat(chain_e, np.float32),
             pair=cat(chain_pair, np.int16), stats=np.array(json.dumps(chain_stats)),
             meta=np.array(json.dumps({"clip": clip, "chain_pairs": chain_j, "native_pairs": n_native,
                                       "r": "|D - F| / |F|, pixels in frame with |F| > 2 px",
                                       "e": "|D - F| px, same pixels",
                                       "F": "native RAFT composed along the path, bilinear, align_corners"})))
    return {"clip": clip, "status": "ok", "pairs_2fps": n - 1, "native_pairs": n_native,
            "chain_pairs": len(chain_j), "run_s": time.perf_counter() - t0,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2 ** 30}


def load_raft():
    import torch
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    w = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=w, progress=False).eval().cuda()
    ck = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(w.url))
    info = {"weights": str(w), "url": w.url, "checkpoint_sha256": hashlib.sha256(open(ck, "rb").read()).hexdigest(),
            "num_flow_updates": 12, "dtype": "float32", "batch": 1}
    return torch, model, w.transforms(), info


# --------------------------------------------------------------------------- manifest
def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def code_hash():
    files = [os.path.abspath(__file__), rv.__file__] + sorted(glob.glob(os.path.join(P3, "lib", "*.py")))
    h = hashlib.sha256()
    for f in files:
        h.update(open(f, "rb").read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="")
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-raft", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    T0 = time.perf_counter()
    W_ = windows()
    clips = sorted(W_) if not a.clips else a.clips.split(",")
    assert set(clips) <= set(W_), set(clips) - set(W_)
    os.makedirs(a.out, exist_ok=True)
    import torch
    import torchvision
    man = {"purpose": "CodecSight-matched 2 fps arm, data collection only; no metric computed",
           "prereg_commit": PREREG_COMMIT, "code_sha256": code_hash(), "head": git("rev-parse", "HEAD"),
           "uncommitted": git("status", "--porcelain", "--", "p3/codecsight_2fps/collect.py", "p3/lib",
                              "p3/virat/run_virat.py"),
           "argv": sys.argv, "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "versions": {"python": sys.version.split()[0], "pyav": av.__version__, "numpy": np.__version__,
                        "torch": torch.__version__, "torchvision": torchvision.__version__},
           "arms": {e["id"]: {"codec": e["codec"], "knobs": e["knobs"],
                              "options": build_options(e["codec"], **e["knobs"])} for e in arms()},
           "clips": {c: W_[c] for c in clips}, "runs": [], "raft": {}}
    mp_path = os.path.join(a.out, "manifest.json")

    def save():
        json.dump(man, open(mp_path, "w"), indent=1, default=str)

    def log(s):
        print(f"{time.strftime('%H:%M:%S')} {s}", flush=True)

    save()
    cpu = [(c, W_[c], e, a.out, a.overwrite) for c in clips for e in arms() if e["codec"] == "libx264"]
    gpu = [(c, W_[c], e, a.out, a.overwrite) for c in clips for e in arms() if e["codec"] == "h264_nvenc"]
    log(f"{len(clips)} clips, {len(cpu)} x264 + {len(gpu)} NVENC encodes, out {a.out}")
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map_async(run_arm, cpu)
        for t in gpu:
            r = run_arm(t)
            man["runs"].append(r)
            log(f"{r['clip']} {r['arm']:20s} {r['status']} {r.get('check_mismatches', r.get('error'))}")
        for r in res.get():
            man["runs"].append(r)
            log(f"{r['clip']} {r['arm']:20s} {r['status']} {r.get('check_mismatches', r.get('error'))}")
    save()
    if not a.no_raft:
        torch, model, tf, info = load_raft()
        man["raft"]["model"] = info
        for c in clips:
            out = os.path.join(a.out, "data", c, "raft.npz")
            if os.path.exists(out) and not a.overwrite:
                man["raft"][c] = {"status": "skipped_exists"}
                continue
            try:
                man["raft"][c] = r = raft_clip(c, W_[c], a.out, model, tf, torch)
                log(f"RAFT {c}: {r['pairs_2fps']} pairs x2, {r['native_pairs']} native chain pairs, "
                    f"{r['run_s']:.0f}s, peak {r['peak_vram_gb']:.2f} GB")
            except Exception as e:
                man["raft"][c] = {"status": "FAILED", "error": repr(e), "traceback": traceback.format_exc()}
                log(f"RAFT {c} FAILED {e!r}")
            torch.cuda.empty_cache()
            save()
    man.update(end_time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), runtime_s=time.perf_counter() - T0)
    save()
    log(f"done in {time.perf_counter() - T0:.0f}s")


if __name__ == "__main__":
    main()
