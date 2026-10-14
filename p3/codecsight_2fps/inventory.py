"""CodecSight-matched 2 fps arm: data inventory ONLY (written with the pre-registration, before any encode, RAFT run
or metric).

For each of the 27 eligible confirmatory clips (p3/virat_confirm/eligibility_v0.csv) it reads the raw VIRAT file and
lists: container metadata (codec, size, average frame rate, stream frame count, duration), the exact number of
decoded frames (full decode, display order, as run_virat.decode_source counts them), the sampling step k (12 at
23.97 fps, 15 at 30 fps), and the source-frame window that PREREGISTRATION.md §2 fixes, with its overlap with the
frames analysed before at native rate (0-299: exploratory p3/virat, all 54 clips; 300-599: confirmatory
p3/virat_confirm, start 300). It also lists the 2 fps stream that the window implies (frames, I-frame and refresh
positions, scored frames, RAFT pairs) and the work counts used for the runtime estimate.

It decodes frames only to count them. It reads no pixel value, motion vector or flow value, and computes no reuse,
stale or motion quantity. Writes inventory.txt and inventory.csv next to this file.

Usage: python inventory.py [--workers N]
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from fractions import Fraction

import av
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
VIRAT = r"C:\Users\akash\Documents\datasets\VIRAT\CCTV 01"
GOP = 16                 # 2 fps stream: keyint 16 = I-frame every 8 s
MIN_GOPS, MAX_GOPS = 2, 5  # at least 2 full GOPs (32 sampled frames, 16 s); at most 5 (80 frames, CodecSight's 40 s window)
UNSEEN = 600             # first source frame never analysed (0-299 exploratory, 300-599 confirmatory)
SEEN = (("exploratory p3/virat", 0, 299), ("confirmatory p3/virat_confirm", 300, 599))
ARMS = ("x264_crf12", "x264_crf45", "x264_qp24", "x264_qp24_bf2_pb1", "nvenc_qp18", "nvenc_qp23",
        "nvenc_qp45", "nvenc_qp28", "nvenc_qp28_bf2_bqeq")
CHAIN_PAIRS_PER_GOP = 4  # R-chain check: native-rate RAFT under the first 4 sampled pairs after each I-frame


def step(fps):
    """Sampling step: every 12th frame at 23.97 fps, every 15th at 30 fps (round(fps / 2))."""
    return int(round(float(fps) / 2))


def window(n_frames, k):
    """PREREGISTRATION.md §2. Returns (start, gops, rule) or (None, 0, reason).
    span(G) = (16 G - 1) k + 1 source frames.
    1. Unseen: start 600, G = the largest of 2..5 that fits in frames 600..n-1.
    2. Otherwise end-aligned, 2 GOPs: start = n - span(2), the latest start (least overlap with analysed frames).
    3. n < span(2): the clip cannot give 2 full GOPs at 2 fps and is excluded."""
    span = lambda g: (GOP * g - 1) * k + 1  # noqa: E731
    fit = [g for g in range(MIN_GOPS, MAX_GOPS + 1) if UNSEEN + span(g) <= n_frames]
    if fit:
        return UNSEEN, max(fit), "unseen"
    if span(MIN_GOPS) <= n_frames:
        return n_frames - span(MIN_GOPS), MIN_GOPS, "end-aligned (overlaps analysed frames)"
    return None, 0, f"cannot: {n_frames} frames < {span(MIN_GOPS)} needed for 2 GOPs"


def overlap(a, b):
    out = []
    for name, lo, hi in SEEN:
        x, y = max(a, lo), min(b, hi)
        if x <= y:
            out.append((name, x, y))
    return out


def probe(clip):
    path = os.path.join(VIRAT, clip + ".mp4")
    t0 = time.perf_counter()
    with av.open(path) as c:
        st = c.streams.video[0]
        cc = st.codec_context
        meta = {"codec": cc.name, "width": cc.width, "height": cc.height, "avg_fps": str(st.average_rate),
                "stream_frames": st.frames,
                "duration_s": float(st.duration * st.time_base) if st.duration else None,
                "file_bytes": os.path.getsize(path)}
        n = 0
        for _ in c.decode(st):  # thread_type left at PyAV's default, as run_virat.decode_source
            n += 1
    meta["decoded_frames"] = n
    meta["count_s"] = time.perf_counter() - t0
    return clip, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.perf_counter()
    el = pd.read_csv(os.path.join(P3, "virat_confirm", "eligibility_v0.csv"))
    el = el[el["eligible"].astype(bool)].sort_values("clip")
    man = json.load(open(os.path.join(P3, "virat_confirm", "manifest.json")))
    with mp.Pool(a.workers) as pool:
        metas = dict(pool.map(probe, el["clip"].tolist()))
    rows = []
    for clip, scene in zip(el["clip"], el["scene"]):
        m = metas[clip]
        conf = man["clips"].get(clip, {})
        assert str(Fraction(conf.get("fps_used", m["avg_fps"]))) == str(Fraction(m["avg_fps"])), clip
        k = step(Fraction(m["avg_fps"]))
        n = m["decoded_frames"]
        start, gops, rule = window(n, k)
        r = {"clip": clip, "scene": scene, **{x: m[x] for x in ("codec", "width", "height", "avg_fps", "stream_frames",
                                                                "duration_s", "decoded_frames", "file_bytes")},
             "k": k, "sample_interval_s": k / float(Fraction(m["avg_fps"])),
             "confirm_start": conf.get("start_frame"), "confirm_frames": conf.get("decoded_frames"),
             "frames_after_599": max(n - UNSEEN, 0),
             "max_gops_from_600": max([g for g in range(1, 50) if UNSEEN + (GOP * g - 1) * k + 1 <= n], default=0),
             "max_gops_whole_clip": max([g for g in range(1, 200) if (GOP * g - 1) * k + 1 <= n], default=0),
             "rule": rule, "gops": gops}
        if start is not None:
            ns = GOP * gops
            last = start + (ns - 1) * k
            ov = overlap(start, last)
            samp = [start + j * k for j in range(ns)]
            r.update({"win_first": start, "win_last": last, "win_source_frames": last - start + 1,
                      "sampled_frames": ns, "seconds": ns * k / float(Fraction(m["avg_fps"])),
                      "sampled_first5": ",".join(map(str, samp[:5])),
                      "iframes_2fps": ",".join(str(g * GOP) for g in range(gops)),
                      "refresh_2fps": ",".join(str(g * GOP) for g in range(gops)),
                      "scored_2fps": ns - 1, "scored_refresh": gops - 1,
                      "raft_pairs_2fps": ns - 1,
                      "chain_native_pairs": gops * CHAIN_PAIRS_PER_GOP * k,
                      "overlap_analysed": "; ".join(f"{nm} {x}-{y} ({y - x + 1} src fr, "
                                                    f"{sum(x <= s <= y for s in samp)} sampled)" for nm, x, y in ov)
                      or "none",
                      "sampled_in_analysed": sum(any(lo <= s <= hi for _, lo, hi in SEEN) for s in samp),
                      "decode_through": last + 1})
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "inventory.csv"), index=False)

    ok = df[df["gops"] > 0]
    L = ["# CodecSight-matched 2 fps arm: data inventory (no pixel, vector or flow value read; no metric)",
         f"Generated by inventory.py in {time.perf_counter() - t0:.0f} s. Source: {VIRAT}. "
         f"Clips: the {len(df)} eligible clips of p3/virat_confirm/eligibility_v0.csv.", "",
         "## Clip lengths (full decode, display order)", ""]
    cols = ["clip", "codec", "avg_fps", "stream_frames", "decoded_frames", "duration_s", "k", "frames_after_599",
            "max_gops_from_600", "max_gops_whole_clip"]
    L.append(df[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    L += ["", f"## Windows (PREREGISTRATION.md §2: start 600 with the most full GOPs in 2..{MAX_GOPS}; "
              f"else end-aligned 2 GOPs; else excluded)", ""]
    cols = ["clip", "k", "rule", "gops", "win_first", "win_last", "sampled_frames", "seconds", "sampled_first5",
            "overlap_analysed"]
    L.append(df[[c for c in cols if c in df]].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    L += ["", "## Summary", "",
          f"- clips usable: {len(ok)}/{len(df)}; excluded: "
          f"{', '.join(df.loc[df['gops'] == 0, 'clip']) or 'none'}",
          f"- by rule: {ok['rule'].value_counts().to_dict()}",
          f"- by GOP count: {ok['gops'].value_counts().sort_index().to_dict()}",
          f"- by k: {df['k'].value_counts().sort_index().to_dict()}",
          f"- clips whose window overlaps frames analysed before at native rate: "
          f"{int((ok['overlap_analysed'] != 'none').sum())}",
          f"- sampled frames in total: {int(ok['sampled_frames'].sum())}; of them in previously analysed ranges: "
          f"{int(ok['sampled_in_analysed'].sum())}",
          f"- 2 fps RAFT pairs (forward): {int(ok['raft_pairs_2fps'].sum())}; with backward: "
          f"{2 * int(ok['raft_pairs_2fps'].sum())}",
          f"- R-chain native RAFT pairs ({CHAIN_PAIRS_PER_GOP} sampled pairs after each I-frame): "
          f"{int(ok['chain_native_pairs'].sum())}",
          f"- encodes: {len(ok)} clips x {len(ARMS)} arms = {len(ok) * len(ARMS)} "
          f"({len(ok) * sum(x.startswith('x264') for x in ARMS)} x264 on CPU, "
          f"{len(ok) * sum(x.startswith('nvenc') for x in ARMS)} NVENC on the GPU), frames encoded "
          f"{int(ok['sampled_frames'].sum()) * len(ARMS)}",
          f"- source frames decoded to reach the windows (decode from frame 0): {int(ok['decode_through'].sum())}",
          f"- frame counting decode time (this run, {a.workers} workers): "
          f"{sum(metas[c]['count_s'] for c in df['clip']):.0f} s CPU in total"]
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, "inventory.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
