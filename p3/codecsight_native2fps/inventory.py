"""Native-rate encode, 2 fps sampling arm (reading B): window inventory and runtime estimate ONLY.
Written with the pre-registration, before any encode, RAFT run or metric.

Reads p3/codecsight_2fps/inventory.csv (cc2eaf9's windows: container metadata, decoded frame counts, sampled-frame
windows). It decodes nothing and reads no pixel value, motion vector or flow value. For every clip it derives, from the
frame counts alone: the native-frame window (first sampled frame .. last sampled frame), the native encode length,
the I-frame / refresh positions of a keyint = 8 s encode, the sampled frames, and the native RAFT pair count. It then
applies the unit costs below to estimate the runtime.

Writes inventory.txt and inventory.csv next to this file.   Usage: python inventory.py
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "codecsight_2fps", "inventory.csv")
GOP = 16                 # sampled frames per refresh period
N_ARMS, N_X264, N_NVENC = 9, 4, 5
RAFT_S_PER_PAIR = 0.61   # RTX 4050, 1280x720, 12 updates (p3/virat_confirm/manifest.json, as cc2eaf9 §7)
MV_S_PER_FRAME = 0.0075  # lib.mvs.extract_mvs, cc2eaf9 §7
NVENC_S_PER_FRAME = 0.003
X264_S_PER_FRAME = 0.134 / 4  # cc2eaf9 §7: 0.134 s per sampled frame over the 4 x264 arms; native motion is smaller
DECODE_FPS = 300.0


def main():
    df = pd.read_csv(SRC)
    df = df[df["gops"] > 0].sort_values("clip").reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        k, G = int(r["k"]), int(r["gops"])
        first, last = int(r["win_first"]), int(r["win_last"])
        L = last - first + 1                       # native frames encoded
        keyint = GOP * k                           # 192 at k = 12, 240 at k = 15
        ifr = list(range(0, L, keyint))
        assert L == (GOP * G - 1) * k + 1 and int(r["win_source_frames"]) == L, r["clip"]
        assert ifr == [g * keyint for g in range(G)], r["clip"]
        assert last < int(r["decoded_frames"]), r["clip"]
        ns = GOP * G
        rows.append({"clip": r["clip"], "scene": r["scene"], "avg_fps": r["avg_fps"], "k": k, "keyint": keyint,
                     "rule": r["rule"], "gops": G, "win_first": first, "win_last": last, "native_frames": L,
                     "sampled_frames": ns, "scored_sampled": ns - 1, "native_pairs": L - 1,
                     "skipped_native_frames": L - ns, "iframes_native": ",".join(map(str, ifr)),
                     "refresh_sampled": ",".join(str(g * GOP) for g in range(G)),
                     "overlap_analysed": r["overlap_analysed"], "sampled_in_analysed": int(r["sampled_in_analysed"]),
                     "decode_through": int(r["decode_through"])})
    o = pd.DataFrame(rows)
    o.to_csv(os.path.join(HERE, "inventory.csv"), index=False)

    nat, pairs = int(o["native_frames"].sum()), int(o["native_pairs"].sum())
    unseen = o[o["rule"] == "unseen"]
    est = {
        "decode (CPU path + RGB for RAFT, from frame 0)": (3 * int(o["decode_through"].sum()) / DECODE_FPS / 60, "CPU"),
        "x264 encodes (108)": (N_X264 * nat * X264_S_PER_FRAME / 60, "CPU"),
        "NVENC encodes (135)": (N_NVENC * nat * NVENC_S_PER_FRAME / 60, "GPU"),
        "MV extraction (243)": (N_ARMS * nat * MV_S_PER_FRAME / 60, "CPU"),
        "RAFT forward, every native pair": (pairs * RAFT_S_PER_PAIR / 60, "GPU"),
    }
    raft_min = est["RAFT forward, every native pair"][0]
    L = ["# Native-rate encode, 2 fps sampling (reading B): inventory and runtime (no pixel, vector or flow value read)",
         f"Source: {os.path.relpath(SRC, os.path.dirname(HERE))} (cc2eaf9 windows). Clips: {len(o)}.", "",
         "## Windows (same source-frame windows as cc2eaf9; native encode = every frame first..last)", ""]
    cols = ["clip", "k", "keyint", "gops", "win_first", "win_last", "native_frames", "sampled_frames", "native_pairs",
            "iframes_native"]
    L.append(o[cols].to_string(index=False))
    L += ["", "## Summary", "",
          f"- clips: {len(o)}; by k: {o['k'].value_counts().sort_index().to_dict()}; "
          f"by GOP count: {o['gops'].value_counts().sort_index().to_dict()}",
          f"- unseen-window clips (start 600): {len(unseen)}; end-aligned (overlap analysed frames): {len(o) - len(unseen)}",
          f"- native frames per encode, total over clips: {nat}; per arm x clip encoded frames: {N_ARMS * nat}",
          f"- sampled frames (policy-scored positions): {int(o['sampled_frames'].sum())}, "
          f"scored j=1..n-1: {int(o['scored_sampled'].sum())}",
          f"- skipped native frames (not sampled): {int(o['skipped_native_frames'].sum())}",
          f"- **native RAFT pairs (forward, every native pair of every window): {pairs}**",
          f"  - of them in the 11 unseen-window clips: {int(unseen['native_pairs'].sum())}",
          f"  - cc2eaf9 for comparison: 1,301 pairs at 2 fps (x2 with backward) + 4,308 chain pairs",
          f"- sampled frames in ranges analysed before (native rate, any encode): "
          f"{int(o['sampled_in_analysed'].sum())} / {int(o['sampled_frames'].sum())}",
          f"- encodes: {len(o)} x {N_ARMS} = {len(o) * N_ARMS} ({len(o) * N_X264} x264, {len(o) * N_NVENC} NVENC)", "",
          "## Runtime estimate", "", "| step | estimate (min) | resource |", "|---|---|---|"]
    for name, (m, res) in est.items():
        L.append(f"| {name} | {m:.0f} | {res} |")
    L += ["", f"RAFT: {pairs} pairs x {RAFT_S_PER_PAIR} s = {pairs * RAFT_S_PER_PAIR:.0f} s = {raft_min:.0f} min "
              f"= {raft_min / 60:.2f} h on the RTX 4050.",
          "Encode + decode + MV overlap with RAFT only partly (as cc2eaf9, encodes ran first); analysis adds < 15 min CPU.",
          f"Wall-clock total, serial worst case: {sum(m for m, _ in est.values()) + 15:.0f} min "
          f"({(sum(m for m, _ in est.values()) + 15) / 60:.1f} h); with CPU encodes overlapped: about "
          f"{raft_min + 15 + 10:.0f}-{raft_min + 15 + 25:.0f} min.",
          "Peak VRAM: RAFT at 1280x720 peaked at 2.6 GB (confirmatory run). Disk: native encodes are small; the per-clip "
          "raft.npz stores float16 cell grids (flow, splat, count: about 0.5 MB per pair), so about "
          f"{pairs * 0.5 / 1024:.1f} GB for {pairs} native pairs (not committed)."]
    txt = "\n".join(L) + "\n"
    open(os.path.join(HERE, "inventory.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
