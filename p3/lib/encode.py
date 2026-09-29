"""Encode an RGB frame sequence with PyAV under explicit, recorded rate-control knobs.

Supported encoders and knobs:
  libx264     crf, preset, bframes, keyint, ref, b_pyramid   (all but crf/preset go through x264-params)
  h264_nvenc  qp (constant QP), bframes, keyint, preset, ref, b_ref_mode
  mpeg4       q  (fixed quantiser via qmin=qmax=q), bframes, keyint

encode_sequence() returns a record with the exact option dict handed to avcodec, the
output file size, and the summed packet payload bytes. For libx264 it also returns the
option string x264 embeds in its SEI, which is what the encoder actually used.
"""
import os
import re
import time

import av
import numpy as np
from PIL import Image



def _as_array(f):
    if isinstance(f, np.ndarray):
        return f
    return np.asarray(Image.open(f).convert("RGB"))


def build_options(codec, *, crf=None, qp=None, q=None, preset=None, bframes=0, keyint=250, ref=None,
                  b_pyramid=None, b_ref_mode=None):
    """Return the avcodec option dict for one configuration (all values as strings)."""
    if codec == "libx264":
        if crf is None:
            raise ValueError("libx264 needs crf")
        xp = [f"bframes={bframes}", f"keyint={keyint}", f"min-keyint={keyint}", "scenecut=0"]
        if ref is not None:
            xp.append(f"ref={ref}")
        if b_pyramid is not None:
            xp.append(f"b-pyramid={b_pyramid}")
        opts = {"crf": str(crf), "preset": preset or "medium", "x264-params": ":".join(xp)}
    elif codec == "h264_nvenc":
        if qp is None:
            raise ValueError("h264_nvenc needs qp")
        opts = {"rc": "constqp", "qp": str(qp), "bf": str(bframes), "g": str(keyint),
                "preset": preset or "p4"}
        if ref is not None:
            opts["refs"] = str(ref)
        if b_ref_mode is not None:
            opts["b_ref_mode"] = str(b_ref_mode)
    elif codec == "mpeg4":
        if q is None:
            raise ValueError("mpeg4 needs q")
        # flags=+qscale / global_quality is NOT used: mpegvideo's fixed-qscale path takes lambda
        # from each input AVFrame.quality, which PyAV cannot set (it stays 0 -> clamped to q=2
        # for every q; verified, sizes identical). Pinning qmin=qmax=q forces the quantiser.
        opts = {"qmin": str(q), "qmax": str(q), "bf": str(bframes), "g": str(keyint)}
    else:
        raise ValueError(f"unsupported codec {codec}")
    return opts


def encode_sequence(frames, out_path, codec, fps=24, **knobs):
    """Encode frames (HxWx3 uint8 arrays or image paths) to out_path.

    Returns dict(codec, options, knobs, n_frames, file_bytes, packet_bytes, seconds, x264_sei).
    """
    opts = build_options(codec, **knobs)
    first = _as_array(frames[0])
    h, w = first.shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    t0 = time.perf_counter()
    packet_bytes = 0
    n = 0
    with av.open(out_path, mode="w") as out:
        st = out.add_stream(codec, rate=fps, options=dict(opts))
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        for f in frames:
            arr = _as_array(f)
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
            for pkt in st.encode(vf):
                packet_bytes += pkt.size
                out.mux(pkt)
            n += 1
        for pkt in st.encode(None):
            packet_bytes += pkt.size
            out.mux(pkt)
    secs = time.perf_counter() - t0

    rec = {
        "codec": codec,
        "options": opts,
        "knobs": knobs,
        "n_frames": n,
        "width": w,
        "height": h,
        "fps": fps,
        "file_bytes": os.path.getsize(out_path),
        "packet_bytes": packet_bytes,
        "seconds": secs,
        "x264_sei": None,
    }
    if codec == "libx264":
        rec["x264_sei"] = read_x264_sei(out_path)
    return rec


def read_x264_sei(path):
    """Return the 'options:' string x264 writes into its user-data SEI, or None."""
    with open(path, "rb") as f:
        blob = f.read(1 << 20)
    m = re.search(rb"options: ([ -~]+)", blob)
    return m.group(1).decode("ascii") if m else None


def sei_value(sei, key):
    """Pull one key=value out of an x264 SEI options string."""
    if not sei:
        return None
    m = re.search(rf"(?:^| ){re.escape(key)}=(\S+)", sei)
    return m.group(1) if m else None
