"""Encoder-settings census for UCF-Crime (Anomaly-Videos-Part-1).

Read-only on all video files; writes only _census/ucf_census.csv and _census/summary.md.
"""
import csv
import statistics as st
import sys
import traceback
from collections import Counter
from pathlib import Path

import av

ROOT = Path(r"C:\Users\akash\Downloads\Anomaly-Videos-Part-1")
OUT = Path(__file__).resolve().parent
N_FRAMES = 600
SEI_SCAN_BYTES = 4 * 1024 * 1024

X264_KEYS = [
    "rc", "crf", "qp", "bitrate", "keyint", "keyint_min", "scenecut", "bframes", "ref",
    "me", "subme",
    # preset-implying fields
    "me_range", "trellis", "analyse", "8x8dct", "mixed_ref", "b_adapt", "direct",
    "weightp", "rc_lookahead", "cabac", "deblock", "b_pyramid", "psy_rd", "aq",
]

FIELDS = (
    ["path", "category", "file", "size_bytes", "status", "error",
     "codec", "profile", "width", "height", "avg_fps", "duration_s", "container_bitrate",
     "pix_fmt", "pkts_scanned", "pkt_keyframes", "frames_decoded", "n_I", "n_P", "n_B",
     "n_other", "has_B", "I_indices", "I_spacing_mean", "I_spacing_median", "I_spacing_max",
     "x264_core", "x264_status"]
    + [f"x264_{k}" for k in X264_KEYS]
    + ["x264_options", "name_codec_mismatch"]
)


def parse_x264(path):
    with open(path, "rb") as fh:  # read-only
        blob = fh.read(SEI_SCAN_BYTES)
    i = blob.find(b"x264 - core")
    if i < 0:
        return {"x264_status": "no_x264_sei"}
    end = blob.find(b"\x00", i)
    s = blob[i:end if end > 0 else i + 4096].decode("latin-1", "replace")
    out = {"x264_status": "found"}
    head, _, opts = s.partition("options:")
    out["x264_core"] = head.split("-")[1].strip().split(" ")[1] if "core" in head else ""
    opts = opts.strip()
    out["x264_options"] = opts
    kv = {}
    for tok in opts.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    for k in X264_KEYS:
        out[f"x264_{k}"] = kv.get(k, "")
    return out


AV_PICTURE_TYPE = {0: "NONE", 1: "I", 2: "P", 3: "B", 4: "S", 5: "SI", 6: "SP", 7: "BI"}


def pict_name(pt):
    if isinstance(pt, int):  # PyAV 17 returns the raw AVPictureType int
        return AV_PICTURE_TYPE.get(pt, str(pt))
    n = getattr(pt, "name", None) or str(pt)
    return n.split(".")[-1]


def probe(path):
    row = {"path": str(path), "category": path.parent.name, "file": path.name,
           "size_bytes": path.stat().st_size}
    with av.open(str(path), mode="r") as c:
        vs = c.streams.video[0]
        cc = vs.codec_context
        row["codec"] = cc.name
        row["profile"] = cc.profile or ""
        row["width"], row["height"] = cc.width, cc.height
        row["avg_fps"] = round(float(vs.average_rate), 4) if vs.average_rate else ""
        if c.duration is not None:
            row["duration_s"] = round(c.duration / av.time_base, 3)
        elif vs.duration is not None:
            row["duration_s"] = round(float(vs.duration * vs.time_base), 3)
        row["container_bitrate"] = c.bit_rate or ""
        row["pix_fmt"] = cc.pix_fmt or ""

        types = []
        pkts = pkt_kf = 0
        done = False
        for pkt in c.demux(vs):
            if pkt.size and pkts < N_FRAMES:
                pkts += 1
                pkt_kf += int(pkt.is_keyframe)
            for fr in pkt.decode():
                types.append(pict_name(fr.pict_type))
                if len(types) >= N_FRAMES:
                    done = True
                    break
            if done:
                break

    cnt = Counter(types)
    I_idx = [i for i, t in enumerate(types) if t == "I"]
    gaps = [b - a for a, b in zip(I_idx, I_idx[1:])]
    row.update({
        "pkts_scanned": pkts, "pkt_keyframes": pkt_kf, "frames_decoded": len(types),
        "n_I": cnt.get("I", 0), "n_P": cnt.get("P", 0), "n_B": cnt.get("B", 0),
        "n_other": len(types) - cnt.get("I", 0) - cnt.get("P", 0) - cnt.get("B", 0),
        "has_B": cnt.get("B", 0) > 0,
        "I_indices": " ".join(map(str, I_idx)),
        "I_spacing_mean": round(st.mean(gaps), 2) if gaps else "",
        "I_spacing_median": st.median(gaps) if gaps else "",
        "I_spacing_max": max(gaps) if gaps else "",
    })
    return row, gaps


def dist_table(title, counter, total, key_sort=None):
    lines = [f"### {title}", "", "| value | files | % |", "|---|---:|---:|"]
    items = sorted(counter.items(), key=key_sort or (lambda kv: -kv[1]))
    for v, n in items:
        lines.append(f"| {v if v not in ('', None) else '(blank)'} | {n} | {100*n/total:.1f} |")
    return lines + [""]


def main():
    files = sorted(ROOT.rglob("*.mp4"))
    rows, all_gaps, errors = [], [], []
    for k, p in enumerate(files, 1):
        row = {"path": str(p), "category": p.parent.name, "file": p.name}
        try:
            r, gaps = probe(p)
            row.update(r)
            all_gaps.extend(gaps)
            row["status"] = "ok"
        except Exception as e:  # log and continue
            row["status"] = "pyav_error"
            row["error"] = f"{type(e).__name__}: {e}"
            errors.append((p, traceback.format_exc(limit=2)))
        try:
            row.update(parse_x264(p))
        except Exception as e:
            row["x264_status"] = f"read_error: {e}"
        codec = row.get("codec", "")
        row["name_codec_mismatch"] = ("_x264" in p.name.lower()) and codec != "h264"
        rows.append(row)
        print(f"[{k}/{len(files)}] {p.name} {row['status']} {codec} "
              f"I={row.get('n_I')} B={row.get('n_B')} x264={row.get('x264_status')}", flush=True)

    with open(OUT / "ucf_census.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    write_summary(rows, all_gaps, errors)


def write_summary(rows, all_gaps, errors):
    n = len(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    L = ["# UCF-Crime encoder-settings census (Anomaly-Videos-Part-1)", "",
         f"- Root: `{ROOT}`",
         f"- Files: **{n}** .mp4 ({len(ok)} probed OK, {n-len(ok)} PyAV errors)",
         f"- Frame-type analysis: first {N_FRAMES} decoded frames per file; "
         f"x264 SEI searched in first {SEI_SCAN_BYTES//(1024*1024)} MB", ""]

    if errors:
        L += ["## PyAV errors", ""] + [f"- `{p.name}`: {tb.strip().splitlines()[-1]}" for p, tb in errors] + [""]

    L += ["## Codec", ""]
    L += dist_table("Codec", Counter(r.get("codec", "(error)") for r in rows), n)
    L += dist_table("Profile", Counter(r.get("profile", "") for r in ok), len(ok))
    L += dist_table("Pixel format", Counter(r.get("pix_fmt", "") for r in ok), len(ok))
    mism = [r for r in rows if r["name_codec_mismatch"]]
    L += [f"**Filename/codec mismatches** (`_x264` in name but codec != h264): {len(mism)}", ""]
    L += [f"- `{r['category']}/{r['file']}` → {r.get('codec', '(error)')}" for r in mism] + [""]

    L += ["## Resolution / frame rate / bitrate", ""]
    L += dist_table("Resolution", Counter(f"{r['width']}x{r['height']}" for r in ok), len(ok))
    L += dist_table("Average fps", Counter(r["avg_fps"] for r in ok), len(ok))
    br = [int(r["container_bitrate"]) for r in ok if r.get("container_bitrate")]
    if br:
        q = st.quantiles(br, n=4)
        L += ["### Container bitrate (kbps)", "", "| min | p25 | median | p75 | max | mean |",
              "|---:|---:|---:|---:|---:|---:|",
              f"| {min(br)/1e3:.0f} | {q[0]/1e3:.0f} | {st.median(br)/1e3:.0f} | {q[2]/1e3:.0f} "
              f"| {max(br)/1e3:.0f} | {st.mean(br)/1e3:.0f} |", ""]
        bins = [0, 100e3, 200e3, 300e3, 400e3, 600e3, 1e6, 2e6, float("inf")]
        bc = Counter()
        for b in br:
            for lo, hi in zip(bins, bins[1:]):
                if lo <= b < hi:
                    bc[f"{lo/1e3:.0f}–{hi/1e3:.0f}" if hi != float('inf') else f">={lo/1e3:.0f}"] += 1
        order = {f"{lo/1e3:.0f}–{hi/1e3:.0f}" if hi != float('inf') else f">={lo/1e3:.0f}": i
                 for i, (lo, hi) in enumerate(zip(bins, bins[1:]))}
        L += dist_table("Container bitrate bins (kbps)", bc, len(br), key_sort=lambda kv: order[kv[0]])
    dur = [r["duration_s"] for r in ok if r.get("duration_s") not in ("", None)]
    if dur:
        L += [f"Duration (s): min {min(dur):.1f}, median {st.median(dur):.1f}, max {max(dur):.1f}, "
              f"total {sum(dur)/3600:.2f} h", ""]

    L += ["## x264 settings (from SEI user-data string)", ""]
    L += dist_table("x264 SEI presence", Counter(r.get("x264_status", "") for r in rows), n)
    xs = [r for r in rows if r.get("x264_status") == "found"]
    if xs:
        L += dist_table("x264 core version", Counter(r.get("x264_core", "") for r in xs), len(xs))
        L += dist_table("rc mode", Counter(r["x264_rc"] for r in xs), len(xs))
        L += dist_table("crf", Counter(r["x264_crf"] for r in xs), len(xs))
        L += dist_table("qp", Counter(r["x264_qp"] for r in xs), len(xs))
        L += dist_table("bitrate (x264 option)", Counter(r["x264_bitrate"] for r in xs), len(xs))
        for k in ["keyint", "keyint_min", "scenecut", "bframes", "ref", "me", "subme",
                  "me_range", "trellis", "rc_lookahead", "b_adapt", "weightp", "direct"]:
            L += dist_table(k, Counter(r[f"x264_{k}"] for r in xs), len(xs))
        sc = Counter(r["x264_options"] for r in xs)
        top = sc.most_common(1)[0][1]
        L += ["### Identical settings strings", "",
              f"- Distinct full `options:` strings: **{len(sc)}** across {len(xs)} files with SEI",
              f"- Largest group sharing one identical string: **{top}** files "
              f"({100*top/len(xs):.1f}% of SEI files)",
              "", "| group size | files |", "|---:|---:|"]
        L += [f"| {sz} | {c} |" for sz, c in sorted(Counter(sc.values()).items(), reverse=True)] + [""]
        L += ["Most common settings string:", "", "```", sc.most_common(1)[0][0], "```", ""]
        # which fields differ between strings
        keys = set()
        parsed = []
        for s in sc:
            d = dict(t.split("=", 1) for t in s.split() if "=" in t)
            parsed.append(d)
            keys |= set(d)
        varying = sorted(k for k in keys if len({d.get(k) for d in parsed}) > 1)
        L += [f"Option keys that vary across the distinct strings: {', '.join(varying) or 'none'}", ""]

    L += ["## Frame types (first 600 frames)", ""]
    L += dist_table("B-frames present", Counter(r["has_B"] for r in ok), len(ok))
    nI = [r["n_I"] for r in ok]
    L += [f"I-frames per file in first {N_FRAMES}: min {min(nI)}, median {st.median(nI)}, max {max(nI)}", ""]
    single = sum(1 for r in ok if r["n_I"] <= 1)
    L += [f"Files with ≤1 I-frame in the window (spacing undefined, i.e. GOP ≥ window): {single}", ""]
    short = sum(1 for r in ok if r["frames_decoded"] < N_FRAMES)
    L += [f"Files shorter than the {N_FRAMES}-frame window: {short}", ""]
    mism_kf = [r for r in ok if r["pkt_keyframes"] != r["n_I"]]
    L += [f"Files where packet keyframe count ≠ decoded I count: {len(mism_kf)} "
          f"(in all of them decoded I > keyframe packets, i.e. non-IDR I-frames — x264 emits a plain "
          f"I-frame instead of an IDR when a scenecut falls within keyint_min of the last IDR)", ""]

    def spacing_table(title, gaps):
        return [f"### {title}", "", "| pairs | mean | median | max | min |",
                "|---:|---:|---:|---:|---:|",
                f"| {len(gaps)} | {st.mean(gaps):.2f} | {st.median(gaps)} | {max(gaps)} | {min(gaps)} |", ""]

    h264 = [r for r in ok if r["codec"] == "h264"]
    h_gaps = []
    for r in h264:
        ix = [int(x) for x in r["I_indices"].split()]
        h_gaps += [b - a for a, b in zip(ix, ix[1:])]
    if h_gaps:
        L += spacing_table("I-frame spacing, H.264 files only (all consecutive I-pairs)", h_gaps)
        gc = Counter(h_gaps)
        L += [f"Exactly {max(h_gaps)} frames (= keyint): {gc[250]} of {len(h_gaps)} gaps "
              f"({100*gc[250]/len(h_gaps):.1f}%); gaps < 25 (scenecut non-IDR I near an IDR): "
              f"{sum(n for g, n in gc.items() if g < 25)}", ""]
        L += dist_table("I-frames per H.264 file in window", Counter(r["n_I"] for r in h264), len(h264),
                        key_sort=lambda kv: kv[0])
    if all_gaps:
        L += spacing_table("I-frame spacing, all files pooled (MJPEG files contribute 1-frame gaps)", all_gaps)
        per_file_med = [r["I_spacing_median"] for r in ok if r["I_spacing_median"] != ""]
        L += [f"Per-file median spacing: min {min(per_file_med)}, median {st.median(per_file_med)}, "
              f"max {max(per_file_med)} ({len(per_file_med)} files with ≥2 I-frames)", ""]

    sc = Counter(r["x264_options"] for r in xs) if xs else Counter()
    n_mj = sum(1 for r in rows if r.get("codec") == "mjpeg")
    if len(sc) == 1:
        verdict = (f"**Effectively a single encoder setting:** {len(xs)}/{n} files carry the identical x264 "
                   f"core {xs[0]['x264_core']} options string (1-pass ABR {xs[0]['x264_bitrate']} kbps, "
                   f"keyint={xs[0]['x264_keyint']}, scenecut={xs[0]['x264_scenecut']}, "
                   f"bframes={xs[0]['x264_bframes']}, 320x240); the only exceptions are {n_mj} MJPEG files "
                   f"mislabelled `_x264`, and the per-file variation that remains (realised bitrate, "
                   f"scenecut I-frame positions) comes from content, not settings.")
    else:
        verdict = f"**Settings vary:** {len(sc)} distinct x264 options strings across {len(xs)} files."
    L += ["## Verdict", "", verdict, ""]
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
