"""Decode a video with FFmpeg's exported motion vectors, as numpy arrays.

FFmpeg AVMotionVector semantics (libavcodec/mpegutils.c add_mb):
  dst_x, dst_y   block centre in the CURRENT frame (top-left = dst - (w/2, h/2))
  src_x, src_y   = dst + motion / motion_scale (C integer division, i.e. truncated)
                 -> block position in the REFERENCE frame
  motion_x/y     reference displacement in 1/motion_scale pel (scale=4 -> quarter-pel for H.264)
  source         -1: reference in the past (list 0), +1: future (list 1)

So for a P-frame t whose reference is t-1, the block's motion t-1 -> t is
    dst - src  =  -motion / motion_scale        (sub-pel exact form used below)
The decoder does NOT export which reference picture a block used. It is only guaranteed
to be t-1 when the encoder is limited to one reference frame (x264 ref=1) and no B-frames.
"""
import time

import av
import numpy as np

MV_FIELDS = ("source", "w", "h", "src_x", "src_y", "dst_x", "dst_y",
             "motion_x", "motion_y", "motion_scale")

PICT_TYPE_NAMES = {0: "?", 1: "I", 2: "P", 3: "B", 4: "S", 5: "SI", 6: "SP", 7: "BI"}


def _pict_name(pt):
    try:
        return PICT_TYPE_NAMES.get(int(pt), str(pt))
    except (TypeError, ValueError):
        return getattr(pt, "name", str(pt))


def extract_mvs(path, want_qp=False, thread_type="AUTO", max_frames=None):
    """Decode `path` and return (frames, seconds).

    frames: list (display order) of dicts
      index      0-based display index
      pts        presentation timestamp
      pict_type  'I' / 'P' / 'B'
      pkt_size   bytes of the packet that carried this frame
      mvs        structured ndarray with MV_FIELDS (empty array if the frame has none)
      qp         (mb_rows, mb_cols) int array of per-MB QP if want_qp and available, else None
    """
    opts = {"flags2": "+export_mvs"}
    if want_qp:
        opts["export_side_data"] = "+venc_params"

    empty = np.zeros(0, dtype=[(k, "i4") for k in MV_FIELDS])
    frames = []
    pkt_size_by_pts = {}
    t0 = time.perf_counter()
    with av.open(path) as c:
        st = c.streams.video[0]
        st.codec_context.options = opts
        if thread_type:
            st.thread_type = thread_type

        def take(f):
            mv = f.side_data.get("MOTION_VECTORS")
            if mv is not None:
                a = mv.to_ndarray()
                out = np.empty(a.shape[0], dtype=empty.dtype)
                for k in MV_FIELDS:
                    out[k] = a[k]
            else:
                out = empty
            qp = None
            if want_qp:
                vp = f.side_data.get("VIDEO_ENC_PARAMS")
                if vp is not None:
                    qp = np.asarray(vp.qp_map()).copy()
            frames.append({
                "index": len(frames),
                "pts": f.pts,
                "pict_type": _pict_name(f.pict_type),
                "pkt_size": pkt_size_by_pts.get(f.pts),
                "mvs": out,
                "qp": qp,
            })

        for pkt in c.demux(st):
            if pkt.pts is not None:
                pkt_size_by_pts[pkt.pts] = pkt.size
            for f in pkt.decode():  # pkt.dts None + size 0 is the flush packet
                take(f)
            if max_frames and len(frames) >= max_frames:
                break
    secs = time.perf_counter() - t0

    pts = [f["pts"] for f in frames]
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise RuntimeError("decoder output not in increasing pts order; display index would be wrong")
    return frames, secs


def motion_prev_to_cur(mvs):
    """(N,2) float motion from the reference to the current frame, i.e. dst - src, sub-pel exact."""
    s = mvs["motion_scale"].astype(np.float64)
    s[s == 0] = 1.0
    return np.stack([-mvs["motion_x"] / s, -mvs["motion_y"] / s], axis=1)


def src_topleft(mvs):
    """(N,2) float top-left of the block in the REFERENCE frame, sub-pel exact."""
    m = motion_prev_to_cur(mvs)
    x0 = mvs["dst_x"] - mvs["w"] / 2.0 - m[:, 0]
    y0 = mvs["dst_y"] - mvs["h"] / 2.0 - m[:, 1]
    return np.stack([x0, y0], axis=1)


def dst_topleft(mvs):
    return np.stack([mvs["dst_x"] - mvs["w"] / 2.0, mvs["dst_y"] - mvs["h"] / 2.0], axis=1)
