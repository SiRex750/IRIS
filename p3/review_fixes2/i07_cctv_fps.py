"""Review-2 item 7: frame rate of the VIRAT encodes, from the logs and the encoded files. SUPPLEMENTARY.
For both CCTV collections (p3/virat exploratory, p3/virat_confirm confirmatory): source avg_fps and fps_used per clip
(manifest.json), the fps recorded in every arm's npz meta, the 'loaded ... @ fps' lines of run.log, and the stream
average_rate / time_base of every encoded .mp4 (PyAV). Writes i07_cctv_fps.md and i07_cctv_fps.csv.
"""
import collections
import glob
import json
import os
import re

import av
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)


def one(d):
    root = os.path.join(P3, d)
    man = json.load(open(os.path.join(root, "manifest.json")))
    logfps = {}
    for ln in open(os.path.join(root, "run.log"), encoding="utf-8", errors="replace"):
        m = re.search(r"(VIRAT_\S+) loaded .* @ (\S+) ", ln)
        if m:
            logfps[m.group(1)] = m.group(2)
    rows = []
    for clip, c in sorted(man["clips"].items()):
        npz_fps = set()
        for p in glob.glob(os.path.join(root, "data", clip, "*.npz")):
            if os.path.basename(p) in ("source.npz", "raft.npz"):
                continue
            z = np.load(p)
            if "meta" in z.files:
                f = json.loads(str(z["meta"])).get("fps")
                if f is not None:
                    npz_fps.add(f)
        mp4 = collections.Counter()
        for p in glob.glob(os.path.join(root, "encodes", clip, "*.mp4")):
            with av.open(p) as cn:
                st = cn.streams.video[0]
                mp4[str(st.average_rate)] += 1
        rows.append({"collection": d, "clip": clip, "source_avg_fps": c.get("avg_fps"), "fps_used": c.get("fps_used"),
                     "run_log_fps": logfps.get(clip), "arm_npz_fps": "|".join(sorted(npz_fps)),
                     "encoded_mp4_avg_rate": "|".join(f"{k} x{v}" for k, v in sorted(mp4.items())),
                     "start_frame": c.get("start_frame")})
    return pd.DataFrame(rows)


def main():
    df = pd.concat([one("virat"), one("virat_confirm")], ignore_index=True)
    df.to_csv(os.path.join(HERE, "i07_cctv_fps.csv"), index=False)
    L = ["# Item 7: CCTV encode frame rate (supplementary)", "",
         "Each clip is encoded at its own source average frame rate (p3/virat/run_virat.py: `encode_yuv(src, path, "
         "codec, fps, ...)` with `fps = stream.average_rate` of the source, `add_stream(codec, rate=fps)`). "
         "Per clip: source avg_fps (manifest), fps_used (manifest), the `loaded ... @ fps` run.log line, the fps in "
         "every arm's npz meta, and the average_rate of every encoded .mp4 (PyAV).", ""]
    for d, g in df.groupby("collection"):
        agree = (g.source_avg_fps == g.fps_used) & (g.fps_used == g.run_log_fps) & (g.arm_npz_fps == g.fps_used)
        mp4_ok = g.apply(lambda r: all(x.split(" x")[0] == r.fps_used for x in r.encoded_mp4_avg_rate.split("|")
                                       if x), axis=1)
        L += [f"## p3/{d}", "",
              f"- Clips: {len(g)}. fps used: " + ", ".join(f"{k} in {v} clips" for k, v in
                                                          g.fps_used.value_counts().items()) + ".",
              f"- Source avg_fps = fps_used = run.log = arm npz fps in {int(agree.sum())}/{len(g)} clips; "
              f"encoded .mp4 average_rate = fps_used in {int(mp4_ok.sum())}/{len(g)} clips "
              f"({sum(int(x.split(' x')[1]) for s in g.encoded_mp4_avg_rate for x in s.split('|') if x)} files).",
              "- Clips at 30 fps: " + (", ".join(r.clip for r in g.itertuples() if r.fps_used == "30") or "none")
              + "; every other clip is listed with its rate in i07_cctv_fps.csv.", ""]
        if d == "virat_confirm":
            el = pd.read_csv(os.path.join(P3, d, "eligibility_v0.csv"))
            e = g[g["clip"].isin(el.loc[el.eligible.astype(bool), "clip"])]
            L.insert(len(L) - 1, f"- Eligible clips (27): " + ", ".join(f"{k} in {v}" for k, v in
                                                                     e.fps_used.value_counts().items()) + ".")
    open(os.path.join(HERE, "i07_cctv_fps.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
