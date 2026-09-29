"""Data inventory: MPI-Sintel training structure and VIRAT clip properties. Read-only.

usage: python inventory.py [virat_root ...]   (default: the datasets VIRAT folder)
Writes p3/results/inventory.json.
"""
import json
import os
import sys

import av
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.mvs import _pict_name  # noqa: E402  (PyAV 17 gives pict_type as int)

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
SINTEL = r"C:\Users\akash\Documents\datasets\MPI-Sintel"
VIRAT = r"C:\Users\akash\Documents\datasets\VIRAT"
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".ts")


def sintel():
    tr = os.path.join(SINTEL, "training")
    subdirs = sorted(d for d in os.listdir(tr) if os.path.isdir(os.path.join(tr, d)))
    seqs = sorted(os.listdir(os.path.join(tr, "final")))
    rows, problems = {}, []
    for s in seqs:
        r = {}
        for sub, ext in [("final", ".png"), ("clean", ".png"), ("flow", ".flo"),
                         ("occlusions", ".png"), ("invalid", ".png")]:
            d = os.path.join(tr, sub, s)
            r[sub] = len([f for f in os.listdir(d) if f.endswith(ext)]) if os.path.isdir(d) else 0
        if not (r["final"] == r["clean"] == r["invalid"] and r["flow"] == r["occlusions"] == r["final"] - 1):
            problems.append(s)
        r["size"] = Image.open(os.path.join(tr, "final", s, "frame_0001.png")).size
        rows[s] = r
    test = os.path.join(SINTEL, "test", "final")
    return {
        "root": SINTEL, "training_subdirs": subdirs, "n_sequences": len(seqs),
        "n_frames_final": sum(r["final"] for r in rows.values()),
        "n_flow": sum(r["flow"] for r in rows.values()),
        "sizes": sorted({tuple(r["size"]) for r in rows.values()}),
        "count_rule_violations": problems,
        "per_sequence": rows,
        "test_sequences": len(os.listdir(test)) if os.path.isdir(test) else 0,
    }


def clip_info(path, n_probe=300):
    with av.open(path) as c:
        st = c.streams.video[0]
        cc = st.codec_context
        dur = float(st.duration * st.time_base) if st.duration else (
            c.duration / 1e6 if c.duration else None)
        info = {
            "file_bytes": os.path.getsize(path),
            "codec": cc.name, "profile": cc.profile, "pix_fmt": cc.pix_fmt,
            "width": cc.width, "height": cc.height,
            "fps_avg": float(st.average_rate) if st.average_rate else None,
            "duration_s": dur, "n_frames_header": st.frames,
            "bitrate_stream_bps": st.bit_rate or cc.bit_rate,
            "bitrate_file_bps": (os.path.getsize(path) * 8 / dur) if dur else None,
            "has_b_frames_header": cc.has_b_frames if hasattr(cc, "has_b_frames") else None,
        }
        types = []
        for f in c.decode(st):
            types.append(_pict_name(f.pict_type))
            if len(types) >= n_probe:
                break
        info["probe_frames"] = len(types)
        info["pict_counts_first300"] = {t: types.count(t) for t in sorted(set(types))}
        info["b_frames_present"] = types.count("B") > 0
    return info


def virat(root):
    files = []
    for dp, _, fs in os.walk(root):
        files += [os.path.join(dp, f) for f in fs if f.lower().endswith(VIDEO_EXT)]
    files.sort()
    clips = {}
    for p in files:
        try:
            clips[os.path.relpath(p, root)] = clip_info(p)
        except Exception as e:
            clips[os.path.relpath(p, root)] = {"error": repr(e)}
    n_any = sum(len(fs) for _, _, fs in os.walk(root)) if os.path.isdir(root) else None
    return {"root": root, "exists": os.path.isdir(root), "n_files_any_type": n_any,
            "n_clips": len(clips), "clips": clips}


def main():
    roots = sys.argv[1:] or [VIRAT]
    res = {"sintel": sintel(), "virat": [virat(r) for r in roots]}
    with open(os.path.join(P3, "results", "inventory.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    s = res["sintel"]
    print("Sintel:", s["n_sequences"], "seqs,", s["n_frames_final"], "frames,", s["n_flow"], "flo, sizes", s["sizes"],
          "violations", s["count_rule_violations"])
    for v in res["virat"]:
        print("VIRAT root", v["root"], "exists", v["exists"], "files", v["n_files_any_type"], "clips", v["n_clips"])
        for k, c in v["clips"].items():
            print(" ", k, c)


if __name__ == "__main__":
    main()
