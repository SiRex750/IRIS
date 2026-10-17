"""Native-rate encode, 2 fps sampling (reading B): data collection ONLY, as fixed by
p3/codecsight_native2fps/PREREGISTRATION.md (amended; amendment commit ed606f4). No policy, gate, stale or
motion-count metric is computed here. Adapted from p3/codecsight_2fps/collect.py (cc2eaf9).

Per clip (windows from inventory.csv): native frames win_first .. win_last (L of them), all encoded and all given truth.
  Encodes  the native window as decoded yuv420p at the source frame rate, one stream per arm (keyint = min-keyint
           = 16 k = 8 s, scenecut off, the 9 confirmatory arms, NVENC no-scenecut 1). Reuses run_virat: write_arm_npz
           (same npz layout), check (run_sweep.check_encode), qp_stats; slices via run_pilot_v2.slices_per_frame;
           vectors via lib.mvs.extract_mvs with PyAV's default threading. Added checks: I-frames exactly at
           0, K, 2K, ... (hence at sampled positions 0, 16, 32, ...); bf 0 arms: every non-I frame is P with
           d_past = 1; no avcodec option left unconsumed after the encoder opened.
  RAFT     raft_large (DEFAULT weights), 12 updates, float32, batch 1, decoded source -> rgb24 ->
           weights.transforms(), forward on EVERY native pair i-1 -> i, run_virat.raft_all layout: flow_tm1,
           splat_t, splat_count (row i-1 is native frame i, window-relative). No backward pass, no chain.
Outputs (under --out): data/<clip>/<arm>.npz, raft.npz; encodes/<clip>/<arm>.mp4; manifest.json.

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

PREREG_COMMIT = "ed606f4"
GOP = 16  # sampled frames per refresh period
ARM_IDS = ("x264_crf12", "x264_crf45", "x264_qp24", "x264_qp24_bf2_pb1", "nvenc_qp18", "nvenc_qp23",
           "nvenc_qp45", "nvenc_qp28", "nvenc_qp28_bf2_bqeq")


def arms(keyint=None):
    """The 9 confirmatory arms; keyint is set per clip (16 k); NVENC also no_scenecut 1 (PREREGISTRATION.md §4)."""
    src = {e["id"]: e for e in CRF_SERIES + X264_QP_PAIR + NVENC_SERIES + NVENC_B}
    out = []
    for a in ARM_IDS:
        e = dict(src[a])
        kn = {**e["knobs"], "keyint": keyint if keyint else GOP}
        if e["codec"] == "h264_nvenc":
            kn["no_scenecut"] = 1
        e["knobs"] = kn
        out.append(e)
    return out


def windows():
    inv = pd.read_csv(os.path.join(HERE, "inventory.csv"))
    return {r["clip"]: {"scene": r["scene"], "k": int(r["k"]), "keyint": int(r["keyint"]), "start": int(r["win_first"]),
                        "last": int(r["win_last"]), "gops": int(r["gops"]), "L": int(r["native_frames"]),
                        "n_sampled": int(r["sampled_frames"]), "fps_src": str(r["avg_fps"])}
            for _, r in inv.iterrows()}


# --------------------------------------------------------------------------- source
_CACHE = {}


def native_yuv(clip, w):
    """Native window frames as yuv420p ndarrays (run_virat.decode_source), and the source frame rate; cached per
    process (one clip)."""
    if clip not in _CACHE:
        _CACHE.clear()
        _CACHE[clip] = rv.decode_source(os.path.join(VIRAT, clip + ".mp4"), w["L"], start=w["start"])
    return _CACHE[clip]


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
        src, fps = native_yuv(clip, w)
        H, W = src[0].shape[0] * 2 // 3, src[0].shape[1]
        Hc, Wc = -(-H // C), -(-W // C)
        path = os.path.join(e_dir, f"{enc['id']}.mp4")
        rec = encode_yuv(src, path, enc["codec"], fps, **enc["knobs"])
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
        sidx = list(range(w["start"], w["last"] + 1))
        types, n_qp = write_arm_npz(npz, clip, enc["id"],
                                    {"codec": enc["codec"], "options": rec["options"], "knobs": enc["knobs"],
                                     "fps": str(fps), "x264_sei": rec["x264_sei"], "source_frames": [sidx[0], sidx[-1]],
                                     "k": w["k"], "keyint": w["keyint"], "gops": w["gops"], "source_fps": w["fps_src"]},
                                    frames, Hc, Wc, enc.get("d_known", True))
        qs, qvals = qp_stats(frames)
        n = len(frames)
        z = np.load(npz)
        d_past = z["d_past"]
        iframes = [t for t, ft in enumerate(types) if ft == "I"]
        lg.update({"status": "ok", "options": rec["options"], "x264_sei": rec["x264_sei"],
                   "unused_options": rec["unused_options"], "frames_encoded": rec["n_frames"], "frames_decoded": n,
                   "file_bytes": rec["file_bytes"], "bitrate_kbps": rec["packet_bytes"] * 8 * float(fps) / n / 1000.0,
                   "psnr_y_mean": float(np.mean(psnr)), "psnr_y_min": float(np.min(psnr)),
                   "slices_min": int(min(spf)), "slices_max": int(max(spf)),
                   "sei_keyint": sei_value(rec["x264_sei"], "keyint"),
                   "sei_keyint_min": sei_value(rec["x264_sei"], "keyint_min"),
                   "sei_scenecut": sei_value(rec["x264_sei"], "scenecut"),
                   "n_I": types.count("I"), "n_P": types.count("P"), "n_B": types.count("B"),
                   "iframes": iframes, "ftypes": "".join(types), "qp_frames_available": n_qp, **qs,
                   "encode_s": t_enc, "mv_extract_s": t_mv, "run_s": time.perf_counter() - t0})
        bad = check(enc, lg, spf, qvals)
        want_i = list(range(0, n, w["keyint"]))
        if iframes != want_i:
            bad.append(f"I-frames at {iframes} (want {want_i})")
        if any(t % w["k"] for t in iframes):
            bad.append("an I-frame is not at a sampled position")
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
    """Forward RAFT on every native pair of the window, one sequential decode pass (run_virat.raft_all layout)."""
    L = w["L"]
    t0 = time.perf_counter()
    H = W = None
    prev = None
    with av.open(os.path.join(VIRAT, clip + ".mp4")) as c, torch.inference_mode():
        st = c.streams.video[0]
        for i, fr in enumerate(c.decode(st)):
            if i < w["start"]:
                continue
            rgb = fr.to_ndarray(format="rgb24")
            t = i - w["start"]
            if H is None:
                H, W = rgb.shape[:2]
                Hc, Wc = -(-H // C), -(-W // C)
                flow_tm1 = np.empty((L - 1, Hc, Wc, 2), np.float16)
                splat = np.empty((L - 1, Hc, Wc, 2), np.float16)
                scount = np.empty((L - 1, Hc, Wc), np.uint8)
                cnt = np.zeros((Hc * C, Wc * C))
                cnt[:H, :W] = 1
                cnt = cnt.reshape(Hc, C, Wc, C).sum((1, 3))
                pad = np.zeros((Hc * C, Wc * C, 2))
                nobad = np.zeros((H, W), bool)
                torch.cuda.reset_peak_memory_stats()
            if t >= 1:
                a = torch.from_numpy(prev).permute(2, 0, 1)[None]
                b = torch.from_numpy(rgb).permute(2, 0, 1)[None]
                a, b = tf(a, b)
                fl = model(a.cuda(), b.cuda())[-1][0].float().cpu().numpy().transpose(1, 2, 0)
                pad[:H, :W] = fl
                flow_tm1[t - 1] = (pad.reshape(Hc, C, Wc, C, 2).sum((1, 3)) / cnt[..., None]).astype(np.float16)
                g = GTGrid(fl, nobad)
                gx, gy, _, _ = g.cell_gt()
                splat[t - 1] = np.stack([gx, gy], -1).astype(np.float16)
                scount[t - 1] = np.minimum(g.count, 255).astype(np.uint8)
            prev = rgb
            if t >= L - 1:
                break
    assert t == L - 1, (clip, t, L)
    d_dir = os.path.join(out_dir, "data", clip)
    os.makedirs(d_dir, exist_ok=True)
    meta = {"clip": clip, "start_frame": w["start"], "L": L, "k": w["k"], "keyint": w["keyint"], "H": H, "W": W,
            "Hc": Hc, "Wc": Wc, "cell": C,
            "flow_tm1": "RAFT native pair t-1 -> t (u, v) px, mean over each 4x4 cell of frame t-1's grid; row i is "
                        "window-relative native frame t = i + 1",
            "splat_t": "lib.gt_grid.GTGrid(flow, no bad mask).cell_gt(): forward splat onto frame t's grid "
                       "(NaN = no landing)",
            "splat_count": "pixels landing in the cell, clipped at 255"}
    save_npz(os.path.join(d_dir, "raft.npz"), flow_tm1=flow_tm1, splat_t=splat, splat_count=scount,
             t=np.arange(1, L, dtype=np.int16), meta=np.array(json.dumps(meta)))
    return {"clip": clip, "status": "ok", "native_pairs": L - 1, "run_s": time.perf_counter() - t0,
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
    ap.add_argument("--workers", type=int, default=4)
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
    man = {"purpose": "native-rate encode, 2 fps sampling (reading B), data collection only; no metric computed",
           "prereg_commit": PREREG_COMMIT, "code_sha256": code_hash(), "head": git("rev-parse", "HEAD"),
           "uncommitted": git("status", "--porcelain", "--", "p3/codecsight_native2fps/collect.py", "p3/codecsight_native2fps/gates.py", "p3/lib",
                              "p3/virat/run_virat.py"),
           "argv": sys.argv, "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "versions": {"python": sys.version.split()[0], "pyav": av.__version__, "numpy": np.__version__,
                        "torch": torch.__version__, "torchvision": torchvision.__version__},
           "arms": {e["id"]: {"codec": e["codec"], "knobs": e["knobs"],
                              "options": build_options(e["codec"], **e["knobs"])} for e in arms()},  # keyint per clip: 16 k, see clips
           "clips": {c: W_[c] for c in clips}, "runs": [], "raft": {}}
    mp_path = os.path.join(a.out, "manifest.json")

    def save():
        json.dump(man, open(mp_path, "w"), indent=1, default=str)

    def log(s):
        print(f"{time.strftime('%H:%M:%S')} {s}", flush=True)

    save()
    cpu = [(c, W_[c], e, a.out, a.overwrite) for c in clips for e in arms(W_[c]["keyint"]) if e["codec"] == "libx264"]
    gpu = [(c, W_[c], e, a.out, a.overwrite) for c in clips for e in arms(W_[c]["keyint"]) if e["codec"] == "h264_nvenc"]
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
                log(f"RAFT {c}: {r['native_pairs']} native pairs, "
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
