"""Paper 3: VIRAT data collection ONLY. No gate metric is computed here (analysis plan is pre-registered later).

Reuses the Sintel sweep unchanged where it can:
  - arm knobs: the arm constructors and knob sets of p3/sweep/run_sweep.py (-> run_pilot_v2 X, XB, N, NB, M)
    and lib.encode.build_options, i.e. the exact avcodec option dicts of the sweep
  - MV extraction: lib.mvs.extract_mvs (display order, FFmpeg export_mvs, per-MB QP via venc_params)
  - 4x4 cell mapping: run_pilot_v2.run_one's painting (run_pilot.paint, block top-left = rint(dst_topleft),
    future-pointing vectors painted first, past-pointing override), cell grid Hc = ceil(H/4), Wc = ceil(W/4)
  - slice counting: run_pilot_v2.slices_per_frame; encode checks as run_sweep.check_encode (QP_REQ)
Differences forced by the source (VIRAT is video, Sintel is PNG):
  - input frames are the source's decoded yuv420p planes, handed to the encoder as yuv420p (no RGB round
    trip); encoded at the source's average frame rate. lib.encode.encode_sequence only takes RGB, so
    encode_yuv() below is that function with a yuv420p input frame.

Outputs (under --out, default p3/virat/):
  data/<clip>/source.npz   absdiff (n-1, Hc, Wc) uint8: per 4x4 cell mean |Y_t - Y_{t-1}|, t = 1..n-1, rounded
  data/<clip>/<arm>.npz    per frame: index, ftype, pts, pkt_size, qp (n, mb_rows, mb_cols) uint8,
                           d_past / d_fut (int16, -1 = none); per frame x cell: mv_q, mv_fut_q (uint8), cell_dir
  data/<clip>/raft.npz     torchvision raft_large (DEFAULT weights), source t-1 -> t, t = 1..n-1:
                           flow_tm1 (n-1, Hc, Wc, 2) f16 = cell mean of the flow field on frame t-1's grid;
                           splat_t (n-1, Hc, Wc, 2) f16 + splat_count uint8 = lib.gt_grid forward splat onto
                           frame t's grid (the Sintel GT construction; bad mask empty)
  encodes/<clip>/<arm>.mp4
  manifest.json (health only), stdout = run log

|MV| quantisation: q = floor(4 |MV|) clipped to 254, so q = k means k/4 <= |MV| < (k+1)/4 and
q = 254 means |MV| >= 63.5 px; 255 = no vector (intra block, I-frame, frame 0). For any tau that is a
multiple of 0.25 px and < 63.5, |MV| < tau  <=>  q < 4 tau exactly. |MV| is the raw (unscaled) vector
length; mv_q holds the vector the Sintel runner uses for the cell (past-pointing if present, else
future-pointing; cell_dir -1 / +1 / 0 says which), mv_fut_q the future-pointing vector (B-frames).
"""
import argparse
import ctypes
import glob
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from fractions import Fraction

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
REPO = os.path.dirname(P3)
for p in (P3, os.path.join(P3, "sweep"), os.path.join(P3, "pilot_v2"), os.path.join(P3, "pilot"),
          os.path.join(P3, "validate")):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.encode import build_options, read_x264_sei, sei_value  # noqa: E402
from lib.mvs import extract_mvs, dst_topleft  # noqa: E402
from lib.gt_grid import GTGrid  # noqa: E402
from run_pilot import paint, ref_distances  # noqa: E402
from run_pilot_v2 import slices_per_frame  # noqa: E402
from run_sweep import CRF_SERIES, X264_QP_PAIR, NVENC_SERIES, NVENC_B, MPEG4, QP_REQ  # noqa: E402
from val_D_gt_grid import _nals, _rbsp, _ue  # noqa: E402

VIRAT = r"C:\Users\akash\Documents\datasets\VIRAT\CCTV 01"
OUTLIER = "VIRAT_S_000002"
C = 4
N_FRAMES = 300
DISK_MIN = 15e9
CPU_ARMS = CRF_SERIES + X264_QP_PAIR + MPEG4
GPU_ARMS = NVENC_SERIES + NVENC_B
NATIVE = "native"
ALL_ARM_IDS = [NATIVE] + [e["id"] for e in CPU_ARMS + GPU_ARMS]
QUANT_DOC = ("q = floor(4*|MV|) clipped to 254 (254 = |MV| >= 63.5 px), 255 = no vector / intra; "
             "|MV| < tau <=> q < 4*tau for tau a multiple of 0.25 px")

_Q = None
_CFG = None


# --------------------------------------------------------------------------- plumbing
def _init(q, cfg):
    global _Q, _CFG
    _Q, _CFG = q, cfg


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    if _Q is not None:
        _Q.put(("log", line))
    else:
        print(line, flush=True)


def emit(kind, payload):
    _Q.put((kind, payload))


def disk_free():
    return shutil.disk_usage(REPO).free


def disk_ok():
    if os.path.exists(_CFG["stop_file"]):
        return False
    if disk_free() < DISK_MIN:
        with open(_CFG["stop_file"], "w") as f:
            f.write(f"free {disk_free() / 1e9:.1f} GB < {DISK_MIN / 1e9:.0f} GB at {time.ctime()}\n")
        log(f"DISK GUARD: free {disk_free() / 1e9:.1f} GB < {DISK_MIN / 1e9:.0f} GB, stopping")
        return False
    return True


def save_npz(path, **arrays):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def clip_dirs(clip):
    d = os.path.join(_CFG["out"], "data", clip)
    e = os.path.join(_CFG["out"], "encodes", clip)
    os.makedirs(d, exist_ok=True)
    os.makedirs(e, exist_ok=True)
    return d, e


# --------------------------------------------------------------------------- source
def decode_source(path, n, fmt="yuv420p"):
    """First n frames in display order as ndarrays (yuv420p: (1.5H, W) planes; rgb24: (H, W, 3))."""
    out = []
    with av.open(path) as c:
        st = c.streams.video[0]
        # thread_type left at PyAV's default (no frame threading), as lib.mvs.extract_mvs (p3/sweep/DEVIATIONS.md 1)
        fps = st.average_rate
        for f in c.decode(st):
            if fmt == "yuv420p":
                if f.format.name != "yuv420p":
                    f = f.reformat(format="yuv420p")
                out.append(np.ascontiguousarray(f.to_ndarray()))
            else:
                out.append(f.to_ndarray(format=fmt))
            if len(out) == n:
                break
    if len(out) < n:
        raise RuntimeError(f"source has {len(out)} frames < {n}")
    return out, fps


def _se(bits, pos):
    k, pos = _ue(bits, pos)
    return ((k + 1) // 2 if k % 2 else -(k // 2)), pos


def h264_max_num_ref_frames(path):
    """max_num_ref_frames from the first SPS (avcC extradata or Annex B), None if not parsable."""
    try:
        with av.open(path) as c:
            ex = bytes(c.streams.video[0].codec_context.extradata or b"")
        if ex[:1] == b"\x01":
            sps = ex[8:8 + int.from_bytes(ex[6:8], "big")]
        else:
            sps = next(n for n in _nals(ex, 4) if n and n[0] & 0x1F == 7)
        rb = _rbsp(sps[1:])
        bits = np.unpackbits(np.frombuffer(rb, np.uint8))
        profile, pos = rb[0], 24
        _, pos = _ue(bits, pos)
        if profile in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
            cf, pos = _ue(bits, pos)
            if cf == 3:
                pos += 1
            _, pos = _ue(bits, pos)
            _, pos = _ue(bits, pos)
            pos += 1
            scm = bits[pos]
            pos += 1
            if scm:
                for i in range(8 if cf != 3 else 12):
                    present = bits[pos]
                    pos += 1
                    if present:
                        last = nxt = 8
                        for _ in range(16 if i < 6 else 64):
                            if nxt != 0:
                                dl, pos = _se(bits, pos)
                                nxt = (last + dl + 256) % 256
                            last = nxt if nxt != 0 else last
        _, pos = _ue(bits, pos)
        poc, pos = _ue(bits, pos)
        if poc == 0:
            _, pos = _ue(bits, pos)
        elif poc == 1:
            pos += 1
            _, pos = _se(bits, pos)
            _, pos = _se(bits, pos)
            k, pos = _ue(bits, pos)
            for _ in range(k):
                _, pos = _se(bits, pos)
        refs, pos = _ue(bits, pos)
        return int(refs)
    except Exception:
        return None


def source_meta(path):
    clip = os.path.splitext(os.path.basename(path))[0]
    with av.open(path) as c:
        st = c.streams.video[0]
        cc = st.codec_context
        m = {"file": os.path.basename(path), "codec": cc.name, "profile": cc.profile, "width": cc.width,
             "height": cc.height, "pix_fmt": cc.pix_fmt, "avg_fps": str(st.average_rate),
             "stream_frames": st.frames, "stream_bit_rate": st.bit_rate, "container_bit_rate": c.bit_rate,
             "file_bytes": os.path.getsize(path)}
    m["max_num_ref_frames"] = h264_max_num_ref_frames(path) if m["codec"] == "h264" else None
    m["group"] = "000002" if clip == OUTLIER else {"h264": "h264_~1Mbps", "mpeg4": "mpeg4_~8.4Mbps"}.get(
        m["codec"], m["codec"])
    m["outlier"] = clip == OUTLIER
    if m["outlier"]:
        m["outlier_note"] = "only 1080p clip; h264 High ~17 Mbps with B-frames; kept, flagged"
    return m


def frames_sha256(frames):
    h = hashlib.sha256()
    for f in frames:
        h.update(f.tobytes())
    return h.hexdigest()


def source_absdiff(frames, H, W, Hc, Wc):
    out = np.empty((len(frames) - 1, Hc, Wc), np.uint8)
    cnt = np.zeros((Hc * C, Wc * C), np.float64)
    cnt[:H, :W] = 1
    cnt = cnt.reshape(Hc, C, Wc, C).sum((1, 3))
    pad = np.zeros((Hc * C, Wc * C), np.float64)
    for t in range(1, len(frames)):
        pad[:H, :W] = np.abs(frames[t][:H].astype(np.int16) - frames[t - 1][:H].astype(np.int16))
        s = pad.reshape(Hc, C, Wc, C).sum((1, 3))
        out[t - 1] = np.minimum(np.floor(s / cnt + 0.5), 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- encode / extract
def encode_yuv(frames, out_path, codec, fps, **knobs):
    """lib.encode.encode_sequence with yuv420p ndarray input frames."""
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
        for pkt in st.encode(None):
            packet_bytes += pkt.size
            out.mux(pkt)
    return {"options": opts, "n_frames": n, "file_bytes": os.path.getsize(out_path), "packet_bytes": packet_bytes,
            "seconds": time.perf_counter() - t0,
            "x264_sei": read_x264_sei(out_path) if codec == "libx264" else None}


def qmag(m):
    mx = m["motion_x"].astype(np.float64)
    my = m["motion_y"].astype(np.float64)
    s = m["motion_scale"].astype(np.float64)
    s[s == 0] = 1.0
    return np.minimum(np.floor(np.sqrt(mx * mx + my * my) * (4.0 / s)), 254).astype(np.uint8)


def cell_arrays(frames, Hc, Wc):
    """Per frame x cell arrays with run_pilot_v2.run_one's cell mapping (see module docstring)."""
    n, ncell = len(frames), Hc * Wc
    types = [f["pict_type"] for f in frames]
    prev, nxt = ref_distances(types)
    mv_q = np.full((n, ncell), 255, np.uint8)
    mv_fut_q = np.full((n, ncell), 255, np.uint8)
    cell_dir = np.zeros((n, ncell), np.int8)
    d_past = np.full(n, -1, np.int16)
    d_fut = np.full(n, -1, np.int16)
    for fr in frames:
        t, ft, m = fr["index"], fr["pict_type"], fr["mvs"]
        if ft == "I" or t == 0:
            continue
        if prev[t] is not None:
            d_past[t] = t - prev[t]
        if ft != "P" and nxt[t] is not None:
            d_fut[t] = nxt[t] - t
        if m.size == 0:
            continue
        tl = dst_topleft(m)
        x0 = np.rint(tl[:, 0]).astype(np.int64)
        y0 = np.rint(tl[:, 1]).astype(np.int64)
        w, h = m["w"].astype(np.int64), m["h"].astype(np.int64)
        src = m["source"].astype(np.int64)
        q = qmag(m)
        for sgn in (1, -1):  # future first, past overrides
            sel = np.nonzero(src == sgn)[0]
            if sel.size == 0:
                continue
            ci, bi = paint(Hc, Wc, x0[sel], y0[sel], w[sel], h[sel])
            qs = q[sel][bi]
            mv_q[t, ci] = qs
            cell_dir[t, ci] = sgn
            if sgn == 1:
                mv_fut_q[t, ci] = qs
    shp = (n, Hc, Wc)
    return {"mv_q": mv_q.reshape(shp), "mv_fut_q": mv_fut_q.reshape(shp), "cell_dir": cell_dir.reshape(shp),
            "d_past": d_past, "d_fut": d_fut, "types": types}


def frame_arrays(frames):
    qps = [f["qp"] for f in frames]
    have = [q for q in qps if q is not None]
    if have:
        shp = have[0].shape
        qp = np.full((len(frames),) + shp, 255, np.uint8)
        for i, q in enumerate(qps):
            if q is not None and q.shape == shp:
                qp[i] = np.clip(q, 0, 254)
    else:
        qp = np.zeros((len(frames), 0, 0), np.uint8)
    return {"index": np.array([f["index"] for f in frames], np.int16),
            "ftype": np.array([f["pict_type"] for f in frames], "<U2"),
            "pts": np.array([f["pts"] if f["pts"] is not None else -1 for f in frames], np.int64),
            "pkt_size": np.array([f["pkt_size"] if f["pkt_size"] is not None else -1 for f in frames], np.int32),
            "qp": qp}, len(have)


def qp_stats(frames):
    qm = {t: [float(f["qp"].mean()) for f in frames if f["pict_type"] == t and f["qp"] is not None] for t in "IPB"}
    vals = {}
    for f in frames:
        if f["qp"] is not None:
            vals.setdefault(f["pict_type"], set()).update(np.unique(f["qp"]).tolist())
    return {f"mean_{t.lower()}_qp": (float(np.mean(v)) if v else None) for t, v in qm.items()}, vals


def write_arm_npz(path, clip, arm_id, meta, frames, Hc, Wc, d_known):
    ca = cell_arrays(frames, Hc, Wc)
    fa, n_qp = frame_arrays(frames)
    meta = {**meta, "clip": clip, "arm": arm_id, "d_known": d_known, "Hc": Hc, "Wc": Wc, "cell": C,
            "quantisation": QUANT_DOC, "qp_frames_available": n_qp,
            "d": "d_past = t - previous I/P display index, d_fut = next I/P - t (B only); -1 = none. "
                 "Meaningful only if d_known (one reference picture)",
            "cell_mapping": "run_pilot_v2.run_one: rint(dst_topleft), run_pilot.paint, future painted first, "
                            "past overrides; I-frames and frame 0 = 255"}
    save_npz(path, meta=np.array(json.dumps(meta, default=str)), d_known=np.array(bool(d_known)),
             **{k: v for k, v in fa.items()}, mv_q=ca["mv_q"], mv_fut_q=ca["mv_fut_q"], cell_dir=ca["cell_dir"],
             d_past=ca["d_past"], d_fut=ca["d_fut"])
    return ca["types"], n_qp


def check(enc, lg, spf, qvals):
    """run_sweep.check_encode on already-extracted frames."""
    bad = []
    if min(spf) != 1 or max(spf) != 1:
        bad.append(f"slices per frame {min(spf)}-{max(spf)} (want 1)")
    kn = enc["knobs"]
    if enc["codec"] == "libx264":
        sei = lg["x264_sei"]
        want = {"threads": "1", "sliced_threads": "0", "scenecut": "0", "keyint": str(kn["keyint"]),
                "ref": str(kn["ref"]), "bframes": str(kn["bframes"])}
        if kn.get("bframes"):
            want.update(b_adapt=str(kn["b_adapt"]), b_pyramid="0")
        for k, v in want.items():
            got = sei_value(sei, k)
            if got != v:
                bad.append(f"SEI {k}={got} (want {v})")
        if sei_value(sei, "slices") is not None:
            bad.append(f"SEI slices={sei_value(sei, 'slices')} (want absent)")
    if kn.get("bframes", 0) == 0 and lg["n_B"]:
        bad.append(f"{lg['n_B']} B-frames with bframes 0")
    for ft, q in QP_REQ.get(enc["id"], {}).items():
        got = qvals.get(ft)
        if got != {q}:
            bad.append(f"decoded {ft} QP values {sorted(got) if got else got} (want {{{q}}})")
    return bad


def run_arm(enc, clip, src, fps, H, W, Hc, Wc):
    d_dir, e_dir = clip_dirs(clip)
    npz = os.path.join(d_dir, f"{enc['id']}.npz")
    lg = {"clip": clip, "arm": enc["id"], "codec": enc["codec"], "knobs": enc["knobs"]}
    if os.path.exists(npz) and not _CFG["overwrite"]:
        lg["status"] = "skipped_exists"
        log(f"{clip} {enc['id']:22s} exists, skipped")
        return lg
    if not disk_ok():
        lg["status"] = "skipped_disk"
        return lg
    t0 = time.perf_counter()
    try:
        path = os.path.join(e_dir, f"{enc['id']}.mp4")
        rec = encode_yuv(src, path, enc["codec"], fps, **enc["knobs"])
        t_enc = time.perf_counter() - t0
        frames, t_mv = extract_mvs(path, want_qp=True)
        if len(frames) != len(src):
            raise RuntimeError(f"decoded {len(frames)} frames, source has {len(src)}")
        spf = slices_per_frame(path, enc["codec"])
        psnr = []
        with av.open(path) as c:
            for i, f in enumerate(c.decode(video=0)):
                if f.format.name != "yuv420p":
                    f = f.reformat(format="yuv420p")
                mse = np.mean((f.to_ndarray()[:H].astype(np.float64) - src[i][:H]) ** 2)
                psnr.append(100.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
        types, n_qp = write_arm_npz(npz, clip, enc["id"],
                                    {"codec": enc["codec"], "options": rec["options"], "knobs": enc["knobs"],
                                     "fps": str(fps), "x264_sei": rec["x264_sei"]},
                                    frames, Hc, Wc, enc.get("d_known", True))
        qs, qvals = qp_stats(frames)
        sei = rec["x264_sei"]
        n = len(frames)
        lg.update({"status": "ok", "options": rec["options"], "x264_sei": sei, "frames_encoded": rec["n_frames"],
                   "frames_decoded": n, "file_bytes": rec["file_bytes"],
                   "bitrate_kbps": rec["packet_bytes"] * 8 * float(fps) / n / 1000.0,
                   "psnr_y_mean": float(np.mean(psnr)), "psnr_y_min": float(np.min(psnr)),
                   "slices_min": int(min(spf)), "slices_max": int(max(spf)),
                   "sei_threads": sei_value(sei, "threads"), "sei_sliced_threads": sei_value(sei, "sliced_threads"),
                   "n_I": types.count("I"), "n_P": types.count("P"), "n_B": types.count("B"),
                   "qp_frames_available": n_qp, **qs,
                   "encode_s": t_enc, "mv_extract_s": t_mv, "run_s": time.perf_counter() - t0})
        lg["check_mismatches"] = check(enc, lg, spf, qvals)
        lg["notes"] = [f"{lg['n_I']} I-frames (keyint {enc['knobs'].get('keyint')})"] \
            if lg["n_I"] > -(-n // enc["knobs"].get("keyint", 250)) else []
        f2 = lambda v: "  -" if v is None else f"{v:4.1f}"  # noqa: E731
        log(f"{clip} {enc['id']:22s} {lg['bitrate_kbps']:8.0f} kbps PSNR {lg['psnr_y_mean']:5.2f} "
            f"sl {lg['slices_min']}-{lg['slices_max']} thr {lg['sei_threads']}/{lg['sei_sliced_threads']} "
            f"QP I/P/B {f2(qs['mean_i_qp'])}/{f2(qs['mean_p_qp'])}/{f2(qs['mean_b_qp'])} "
            f"I/P/B {lg['n_I']}/{lg['n_P']}/{lg['n_B']} {lg['run_s']:.1f}s"
            + (f"  MISMATCH {lg['check_mismatches']}" if lg["check_mismatches"] else "")
            + (f"  NOTE {lg['notes']}" if lg["notes"] else ""))
    except Exception as e:
        lg.update(status="FAILED", error=repr(e), traceback=traceback.format_exc(), run_s=time.perf_counter() - t0)
        log(f"FAILED {clip} {enc['id']} {e!r}")
    return lg


def run_native(clip, path, meta, n, Hc, Wc):
    d_dir, _ = clip_dirs(clip)
    npz = os.path.join(d_dir, f"{NATIVE}.npz")
    lg = {"clip": clip, "arm": NATIVE, "codec": meta["codec"]}
    if os.path.exists(npz) and not _CFG["overwrite"]:
        lg["status"] = "skipped_exists"
        return lg
    if not disk_ok():
        lg["status"] = "skipped_disk"
        return lg
    t0 = time.perf_counter()
    try:
        frames, t_mv = extract_mvs(path, want_qp=True, max_frames=n)
        frames = frames[:n]
        if len(frames) != n:
            raise RuntimeError(f"extracted {len(frames)} frames < {n}")
        types = [f["pict_type"] for f in frames]
        if meta["codec"] == "mpeg4":
            d_known, d_note = True, "MPEG-4 Part 2: P references the previous I/P, B the surrounding I/P"
        elif meta["codec"] == "h264":
            d_known = meta["max_num_ref_frames"] == 1 and "B" not in types
            d_note = f"h264 max_num_ref_frames={meta['max_num_ref_frames']}, B in first {n}: {'B' in types}"
        else:
            d_known, d_note = False, "unknown codec"
        try:
            spf = slices_per_frame(path, meta["codec"])[:n]
        except Exception as e:
            spf, lg["slices_error"] = [], repr(e)
        types, n_qp = write_arm_npz(npz, clip, NATIVE, {"codec": meta["codec"], "source": meta["file"],
                                                        "d_note": d_note}, frames, Hc, Wc, d_known)
        qs, _ = qp_stats(frames)
        pk = [f["pkt_size"] for f in frames if f["pkt_size"] is not None]
        fps = float(Fraction(meta["avg_fps"]))
        lg.update({"status": "ok", "frames_decoded": len(frames), "d_known": d_known, "d_note": d_note,
                   "slices_min": int(min(spf)) if spf else None, "slices_max": int(max(spf)) if spf else None,
                   "n_I": types.count("I"), "n_P": types.count("P"), "n_B": types.count("B"),
                   "qp_frames_available": n_qp, **qs,
                   "bitrate_kbps_first_n": sum(pk) * 8 * fps / len(pk) / 1000.0 if pk else None,
                   "stream_bit_rate": meta["stream_bit_rate"], "mv_extract_s": t_mv,
                   "run_s": time.perf_counter() - t0})
        f2 = lambda v: "  -" if v is None else f"{v:4.1f}"  # noqa: E731
        log(f"{clip} {NATIVE:22s} {lg['bitrate_kbps_first_n'] or 0:8.0f} kbps (src {meta['codec']}) "
            f"sl {lg['slices_min']}-{lg['slices_max']} QP I/P/B {f2(qs['mean_i_qp'])}/{f2(qs['mean_p_qp'])}/"
            f"{f2(qs['mean_b_qp'])} I/P/B {lg['n_I']}/{lg['n_P']}/{lg['n_B']} d_known {d_known} "
            f"{lg['run_s']:.1f}s")
    except Exception as e:
        lg.update(status="FAILED", error=repr(e), traceback=traceback.format_exc(), run_s=time.perf_counter() - t0)
        log(f"FAILED {clip} {NATIVE} {e!r}")
    return lg


def _load_clip(path, n):
    clip = os.path.splitext(os.path.basename(path))[0]
    t0 = time.perf_counter()
    src, fps = decode_source(path, n)
    H, W = src[0].shape[0] * 2 // 3, src[0].shape[1]
    return clip, src, fps, H, W, -(-H // C), -(-W // C), time.perf_counter() - t0


def cpu_clip(path):
    """Worker task: source.npz, native MVs, then the x264 and mpeg4 arms of one clip."""
    clip = os.path.splitext(os.path.basename(path))[0]
    arms = [e for e in CPU_ARMS if e["id"] in _CFG["arms"]]
    try:
        meta = source_meta(path)
        clip, src, fps, H, W, Hc, Wc, t_dec = _load_clip(path, _CFG["frames"])
        emit("clip", (clip, {**meta, "decoded_frames": len(src), "fps_used": str(fps), "H": H, "W": W,
                             "Hc": Hc, "Wc": Wc, "decoded_yuv_sha256_cpu": frames_sha256(src)}))
        log(f"{clip} loaded {len(src)} frames {W}x{H} @ {fps} ({meta['group']}) in {t_dec:.1f}s")
        d_dir, _ = clip_dirs(clip)
        sp = os.path.join(d_dir, "source.npz")
        if (_CFG["overwrite"] or not os.path.exists(sp)) and disk_ok():
            save_npz(sp, absdiff=source_absdiff(src, H, W, Hc, Wc), t=np.arange(1, len(src), dtype=np.int16),
                     meta=np.array(json.dumps({**meta, "clip": clip, "Hc": Hc, "Wc": Wc, "cell": C,
                                               "absdiff": "per 4x4 cell mean |Y_t - Y_(t-1)| of decoded source, "
                                                          "rounded half up, uint8; row i is t = i + 1"},
                                              default=str)))
        if NATIVE in _CFG["arms"]:
            emit("rec", run_native(clip, path, meta, _CFG["frames"], Hc, Wc))
        for enc in arms:
            emit("rec", run_arm(enc, clip, src, fps, H, W, Hc, Wc))
    except Exception as e:
        log(f"FAILED {clip} cpu task {e!r}")
        for enc in [{"id": NATIVE, "codec": "?"}] + arms:
            emit("rec", {"clip": clip, "arm": enc["id"], "codec": enc["codec"], "status": "FAILED",
                         "error": "clip task: " + repr(e), "traceback": traceback.format_exc()})
    return clip


def gpu_worker(paths, q, cfg):
    """NVENC arms, one encode at a time, clip after clip."""
    _init(q, cfg)
    arms = [e for e in GPU_ARMS if e["id"] in cfg["arms"]]
    for path in paths:
        clip = os.path.splitext(os.path.basename(path))[0]
        if not disk_ok():
            break
        try:
            clip, src, fps, H, W, Hc, Wc, t_dec = _load_clip(path, cfg["frames"])
            emit("clip", (clip, {"decoded_yuv_sha256_nvenc": frames_sha256(src)}))
            for enc in arms:
                emit("rec", run_arm(enc, clip, src, fps, H, W, Hc, Wc))
            del src
        except Exception as e:
            log(f"FAILED {clip} nvenc task {e!r}")
            for enc in arms:
                emit("rec", {"clip": clip, "arm": enc["id"], "codec": enc["codec"], "status": "FAILED",
                             "error": "clip task: " + repr(e), "traceback": traceback.format_exc()})
    emit("gpu_done", None)


# --------------------------------------------------------------------------- RAFT (main process, last)
def raft_all(paths, man, lock):
    info = man["raft"]
    try:
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        w = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=w, progress=False).eval().cuda()
        tf = w.transforms()
        ck = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(w.url))
        info.update({"weights": str(w), "url": w.url, "checkpoint": ck,
                     "checkpoint_sha256": hashlib.sha256(open(ck, "rb").read()).hexdigest(),
                     "num_flow_updates": 12, "dtype": "float32", "batch": 1,
                     "input": "decoded source frames -> rgb24 (PyAV/swscale) -> weights.transforms()"})
    except Exception as e:
        info.update(status="FAILED_LOAD", error=repr(e))
        log(f"RAFT: model/weights failed, skipping RAFT: {e!r}")
        return
    info["status"] = "running"
    n = _CFG["raft_frames"]
    for path in paths:
        clip = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(_CFG["out"], "data", clip, "raft.npz")
        rec = {"clip": clip}
        if os.path.exists(out) and not _CFG["overwrite"]:
            rec["status"] = "skipped_exists"
            info["clips"][clip] = rec
            continue
        if not disk_ok():
            info["status"] = "stopped_disk"
            return
        t0 = time.perf_counter()
        try:
            rgb, _ = decode_source(path, n, fmt="rgb24")
            H, W = rgb[0].shape[:2]
            Hc, Wc = -(-H // C), -(-W // C)
            flow_tm1 = np.empty((n - 1, Hc, Wc, 2), np.float16)
            splat = np.empty((n - 1, Hc, Wc, 2), np.float16)
            scount = np.empty((n - 1, Hc, Wc), np.uint8)
            cnt = np.zeros((Hc * C, Wc * C))
            cnt[:H, :W] = 1
            cnt = cnt.reshape(Hc, C, Wc, C).sum((1, 3))
            pad = np.zeros((Hc * C, Wc * C, 2))
            nobad = np.zeros((H, W), bool)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for t in range(1, n):
                    a = torch.from_numpy(rgb[t - 1]).permute(2, 0, 1)[None]
                    b = torch.from_numpy(rgb[t]).permute(2, 0, 1)[None]
                    a, b = tf(a, b)
                    fl = model(a.cuda(), b.cuda())[-1][0].float().cpu().numpy().transpose(1, 2, 0)
                    pad[:H, :W] = fl
                    flow_tm1[t - 1] = (pad.reshape(Hc, C, Wc, C, 2).sum((1, 3)) / cnt[..., None]).astype(np.float16)
                    g = GTGrid(fl, nobad)
                    gx, gy, _, _ = g.cell_gt()
                    splat[t - 1] = np.stack([gx, gy], -1).astype(np.float16)
                    scount[t - 1] = np.minimum(g.count, 255).astype(np.uint8)
            del rgb
            save_npz(out, flow_tm1=flow_tm1, splat_t=splat, splat_count=scount, t=np.arange(1, n, dtype=np.int16),
                     meta=np.array(json.dumps({"clip": clip, "H": H, "W": W, "Hc": Hc, "Wc": Wc, "cell": C,
                                               "weights": info["weights"],
                                               "flow_tm1": "flow t-1 -> t (u, v) px, mean over each 4x4 cell of "
                                                           "frame t-1's grid; row i is t = i + 1",
                                               "splat_t": "lib.gt_grid.GTGrid(flow, no bad mask).cell_gt(): "
                                                          "forward splat onto frame t's grid (NaN = no landing)",
                                               "splat_count": "pixels landing in the cell, clipped at 255"})))
            rec.update(status="ok", pairs=n - 1, run_s=time.perf_counter() - t0,
                       peak_vram_gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            log(f"RAFT {clip} {n - 1} pairs {rec['run_s']:.0f}s peak VRAM {rec['peak_vram_gb']:.2f} GB")
        except torch.cuda.OutOfMemoryError as e:
            rec.update(status="SKIPPED_OOM", error=repr(e)[:300], run_s=time.perf_counter() - t0)
            log(f"RAFT {clip} out of VRAM, skipped")
        except Exception as e:
            rec.update(status="FAILED", error=repr(e), traceback=traceback.format_exc(), run_s=time.perf_counter() - t0)
            log(f"RAFT {clip} FAILED {e!r}")
        finally:
            torch.cuda.empty_cache()
        with lock:
            info["clips"][clip] = rec
            write_manifest(man)
    info["status"] = "done"


# --------------------------------------------------------------------------- manifest
def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


def code_hash():
    files = [os.path.abspath(__file__), os.path.join(P3, "sweep", "run_sweep.py"),
             os.path.join(P3, "pilot_v2", "run_pilot_v2.py"), os.path.join(P3, "pilot", "run_pilot.py"),
             os.path.join(P3, "validate", "val_D_gt_grid.py")] + sorted(glob.glob(os.path.join(P3, "lib", "*.py")))
    h, per = hashlib.sha256(), {}
    for f in files:
        b = open(f, "rb").read()
        h.update(b)
        per[os.path.relpath(f, REPO).replace("\\", "/")] = hashlib.sha256(b).hexdigest()
    return h.hexdigest(), per


def dir_bytes(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) if os.path.isdir(d) else 0


def write_manifest(man):
    p = os.path.join(_CFG["out"], "manifest.json")
    with open(p + ".tmp", "w") as f:
        json.dump(man, f, indent=1, default=str)
    os.replace(p + ".tmp", p)


def summary(man):
    runs = man["runs"]
    ok = [r for r in runs if r.get("status") == "ok"]
    fails = [r for r in runs if r.get("status") == "FAILED"]
    mm = [r for r in ok if r.get("check_mismatches")]
    return {"records": len(runs), "ok": len(ok), "failed": len(fails), "with_mismatches": len(mm),
            "skipped": len(runs) - len(ok) - len(fails)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--clips", default="", help="comma list of clip stems (default: all)")
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--raft-frames", type=int, default=None)
    ap.add_argument("--arms", default="", help="comma list of arm ids (default: all)")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--no-raft", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    T0 = time.perf_counter()
    try:  # keep the machine awake while running (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(VIRAT, "*.mp4")))
    if a.clips:
        want = a.clips.split(",")
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in want]
    paths.sort(key=lambda p: os.path.splitext(os.path.basename(p))[0] != OUTLIER)  # biggest clip first
    arms = a.arms.split(",") if a.arms else ALL_ARM_IDS
    unknown = set(arms) - set(ALL_ARM_IDS)
    assert not unknown, unknown
    cfg = {"out": a.out, "frames": a.frames, "raft_frames": a.raft_frames or a.frames, "arms": arms,
           "overwrite": a.overwrite, "stop_file": os.path.join(a.out, "STOP_DISK")}
    q = mp.get_context("spawn").Queue()
    _init(q, cfg)
    if os.path.exists(cfg["stop_file"]):
        os.remove(cfg["stop_file"])

    from step0 import x264_build
    import torch
    import torchvision
    ch, per = code_hash()
    free0 = disk_free()
    man = {"purpose": "Paper 3 VIRAT data collection only; no gate metric computed (analysis plan to be "
                      "pre-registered before any metric is computed). Health checks only.",
           "status": "running", "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code_sha256": ch,
           "code_files_sha256": per, "runner_commit": git("rev-parse", "HEAD"),
           "runner_uncommitted_changes": git("status", "--porcelain", "--", "p3/virat/run_virat.py", "p3/lib",
                                             "p3/sweep/run_sweep.py", "p3/pilot_v2/run_pilot_v2.py",
                                             "p3/pilot/run_pilot.py", "p3/validate/val_D_gt_grid.py"),
           "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"), "argv": sys.argv,
           "versions": {"python": sys.version.split()[0], "pyav": av.__version__,
                        "ffmpeg_libs": {k: ".".join(map(str, v)) for k, v in av.library_versions.items()},
                        "numpy": np.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__,
                        "cuda": torch.version.cuda, "x264_build": x264_build(),
                        "nvidia": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                                                  "--format=csv,noheader"], capture_output=True,
                                                 text=True).stdout.strip()},
           "config": {**{k: v for k, v in cfg.items() if k != "stop_file"}, "workers": a.workers, "cell": C,
                      "quantisation": QUANT_DOC, "disk_min_gb": DISK_MIN / 1e9, "source_dir": VIRAT,
                      "arm_knobs": {e["id"]: {"codec": e["codec"], "knobs": e["knobs"],
                                              "options": build_options(e["codec"], **e["knobs"])}
                                    for e in CPU_ARMS + GPU_ARMS if e["id"] in arms}},
           "clips": {}, "runs": [], "raft": {"status": "pending", "clips": {}},
           "disk": {"free_start_gb": free0 / 1e9}, "health_summary": {}}
    write_manifest(man)
    log(f"VIRAT run: {len(paths)} clips, {len(arms)} arms, {a.frames} frames, {a.workers} CPU workers, "
        f"free disk {free0 / 1e9:.1f} GB, runner commit {man['runner_commit'][:7]}"
        + (f"  UNCOMMITTED: {man['runner_uncommitted_changes']}" if man["runner_uncommitted_changes"] else ""))

    lock = threading.Lock()
    gpu_done = threading.Event()

    def listener():
        while True:
            kind, pl = q.get()
            if kind == "stop":
                return
            with lock:
                if kind == "log":
                    print(pl, flush=True)
                    continue
                if kind == "rec":
                    man["runs"].append(pl)
                elif kind == "clip":
                    man["clips"].setdefault(pl[0], {}).update(pl[1])
                elif kind == "gpu_done":
                    gpu_done.set()
                man["health_summary"] = summary(man)
                write_manifest(man)

    th = threading.Thread(target=listener, daemon=True)
    th.start()
    ctx = mp.get_context("spawn")
    gp = None
    if any(e["id"] in arms for e in GPU_ARMS):
        gp = ctx.Process(target=gpu_worker, args=(paths, q, cfg))
        gp.start()
    else:
        gpu_done.set()
    if any(x in arms for x in [NATIVE] + [e["id"] for e in CPU_ARMS]):
        with ctx.Pool(a.workers, initializer=_init, initargs=(q, cfg)) as pool:
            for clip in pool.imap_unordered(cpu_clip, paths):
                log(f"{clip} CPU arms done ({time.perf_counter() - T0:.0f}s elapsed)")
    log(f"CPU arms finished at {time.perf_counter() - T0:.0f}s; waiting for NVENC")
    if gp is not None:
        gp.join()
        gpu_done.wait(60)
        if gp.exitcode:
            log(f"NVENC process exit code {gp.exitcode}")
    log(f"NVENC finished at {time.perf_counter() - T0:.0f}s")
    time.sleep(1)
    with lock:
        man["cpu_gpu_encode_phase_s"] = time.perf_counter() - T0
    if a.no_raft:
        man["raft"]["status"] = "not_run (--no-raft)"
    elif os.path.exists(cfg["stop_file"]):
        man["raft"]["status"] = "not_run (disk guard)"
    else:
        raft_all(paths, man, lock)
    q.put(("stop", None))
    th.join()
    global _Q
    _Q = None  # listener is gone: log straight to stdout
    stopped = os.path.exists(cfg["stop_file"])
    man.update({"status": "stopped_disk_guard" if stopped else "done",
                "end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runtime_s": time.perf_counter() - T0,
                "health_summary": summary(man)})
    man["disk"].update({"free_end_gb": disk_free() / 1e9,
                        "data_bytes": dir_bytes(os.path.join(a.out, "data")),
                        "encodes_bytes": dir_bytes(os.path.join(a.out, "encodes"))})
    write_manifest(man)
    hs = man["health_summary"]
    log(f"DONE status {man['status']} in {man['runtime_s']:.0f}s: records {hs['records']}, ok {hs['ok']}, "
        f"failed {hs['failed']}, with mismatches {hs['with_mismatches']}, skipped {hs['skipped']}; RAFT "
        f"{man['raft']['status']} ({sum(r.get('status') == 'ok' for r in man['raft']['clips'].values())} clips ok); "
        f"data {man['disk']['data_bytes'] / 1e9:.1f} GB, encodes {man['disk']['encodes_bytes'] / 1e9:.1f} GB, "
        f"free {man['disk']['free_end_gb']:.1f} GB")


if __name__ == "__main__":
    main()
