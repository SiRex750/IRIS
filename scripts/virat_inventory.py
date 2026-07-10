"""Discovery-only inventory of a VIRAT/DIVA annotation directory.

Walks the root, finds video clips and KWIVER/DIVA-style per-clip yml
annotation files (*.activities.yml, *.geom.yml, *.regions.yml,
*.types.yml), and prints raw structural info: codec/fps/frames/duration
for video; total line count + first 5 raw lines verbatim for each yml
kind. Pairs videos to clip stems (a clip's yml files share the stem
before the first of these suffixes).

No assumption about which yml fields carry frame/time bounds, no
parsing beyond a raw top-level-key sniff per line, no bounds-checking.
Discovery only.
"""
import json
import os
import re
import subprocess
import sys

VIDEO_EXT = {".mp4"}

YML_KIND_RE = re.compile(
    r"\.(activities|geom|regions|types)\.yml$", re.IGNORECASE
)
TOPKEY_RE = re.compile(r"\{\s*([A-Za-z0-9_]+)\s*:")

try:
    import av
    HAVE_PYAV = True
except ImportError:
    HAVE_PYAV = False


def classify_annotation(name):
    m = YML_KIND_RE.search(name)
    return m.group(1).lower() if m else None


def video_stem(name):
    return name[:-4] if name.lower().endswith(".mp4") else name


def annotation_stem(name):
    return YML_KIND_RE.sub("", name)


def walk_and_categorize(root):
    videos = []
    yml_by_kind = {"activities": [], "geom": [], "regions": [], "types": []}
    other = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            ext = os.path.splitext(name)[1].lower()
            kind = classify_annotation(name)
            if ext in VIDEO_EXT:
                videos.append(path)
            elif kind in yml_by_kind:
                yml_by_kind[kind].append(path)
            else:
                other.append(path)
    for kind in yml_by_kind:
        yml_by_kind[kind].sort()
    return sorted(videos), yml_by_kind, sorted(other)


def probe_video_pyav(path):
    try:
        container = av.open(path)
    except Exception as e:
        return {"error": f"PyAV open failed: {e}"}
    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            return {"error": "no video stream found"}
        codec = stream.codec_context.name if stream.codec_context else None
        fps = stream.average_rate
        fps_val = float(fps) if fps is not None else None
        frame_count = stream.frames or None
        duration = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration) / 1_000_000.0
        if frame_count is None and fps_val and duration:
            frame_count = fps_val * duration
        return {
            "codec": codec,
            "fps": fps_val,
            "frame_count": frame_count,
            "duration_seconds": duration,
        }
    finally:
        container.close()


def probe_video_ffprobe(path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,avg_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception as e:
        return {"error": f"ffprobe failed: {e}"}
    if result.returncode != 0:
        return {"error": f"ffprobe error: {result.stderr.strip()}"}
    try:
        data = json.loads(result.stdout)
    except Exception as e:
        return {"error": f"ffprobe json parse failed: {e}"}
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    codec = stream.get("codec_name")
    avg_rate = stream.get("avg_frame_rate")
    fps_val = None
    if avg_rate and avg_rate != "0/0":
        num, _, den = avg_rate.partition("/")
        try:
            fps_val = float(num) / float(den) if den else float(num)
        except (ValueError, ZeroDivisionError):
            fps_val = None
    duration = stream.get("duration") or fmt.get("duration")
    duration = float(duration) if duration else None
    nb_frames = stream.get("nb_frames")
    frame_count = float(nb_frames) if nb_frames else None
    if frame_count is None and fps_val and duration:
        frame_count = fps_val * duration
    return {
        "codec": codec,
        "fps": fps_val,
        "frame_count": frame_count,
        "duration_seconds": duration,
    }


def print_video_table(videos):
    print("\n=== VIDEO FILES ===")
    if not videos:
        print("(none found)")
        return
    header = f"{'path':<70} {'codec':<10} {'fps':>10} {'frame_count':>14} {'duration_s':>12}"
    print(header)
    print("-" * len(header))
    for path in videos:
        if HAVE_PYAV:
            info = probe_video_pyav(path)
        else:
            info = probe_video_ffprobe(path)
        if "error" in info:
            print(f"{path:<70} ERROR: {info['error']}")
            continue
        codec = info.get("codec") or "?"
        fps = info.get("fps")
        frame_count = info.get("frame_count")
        duration = info.get("duration_seconds")
        fps_s = f"{fps:.3f}" if fps is not None else "?"
        fc_s = f"{frame_count:.0f}" if frame_count is not None else "?"
        dur_s = f"{duration:.3f}" if duration is not None else "?"
        print(f"{path:<70} {codec:<10} {fps_s:>10} {fc_s:>14} {dur_s:>12}")


def print_yml_dump(kind, paths):
    print(f"\n=== *.{kind}.yml FILES (raw, unparsed) ===")
    if not paths:
        print("(none found)")
        return
    for path in paths:
        print(f"\n--- {path} ---")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"  ERROR reading file: {e}")
            continue
        print(f"  total line count: {len(lines)}")
        print("  first 5 raw lines (verbatim):")
        for i, line in enumerate(lines[:5]):
            raw = line.rstrip("\n")
            print(f"    [{i}] {raw!r}")
            m = TOPKEY_RE.search(raw)
            topkey = m.group(1) if m else "?"
            print(f"        top-level key sniffed: {topkey}")


def print_pairing(videos, yml_by_kind):
    print("\n=== VIDEO <-> ANNOTATION CLIP PAIRING (by filename stem) ===")
    video_stems = {video_stem(os.path.basename(p)): p for p in videos}

    all_yml_stems = set()
    stems_by_kind = {}
    for kind, paths in yml_by_kind.items():
        stems = {annotation_stem(os.path.basename(p)): p for p in paths}
        stems_by_kind[kind] = stems
        all_yml_stems |= set(stems)

    paired = sorted(set(video_stems) & all_yml_stems)
    videos_without_yml = sorted(set(video_stems) - all_yml_stems)
    yml_without_video = sorted(all_yml_stems - set(video_stems))

    print(f"\nClips WITH at least one matching yml file ({len(paired)}):")
    if not paired:
        print("  (none)")
    for stem in paired:
        kinds_present = [k for k in stems_by_kind if stem in stems_by_kind[k]]
        print(f"  {stem}  ->  {video_stems[stem]}  <->  yml kinds: {kinds_present}")

    print(f"\nClips WITHOUT any matching yml file ({len(videos_without_yml)}):")
    if not videos_without_yml:
        print("  (none)")
    for stem in videos_without_yml:
        print(f"  {stem}  ->  {video_stems[stem]}")

    print(f"\nAnnotation stems WITHOUT a matching video ({len(yml_without_video)}):")
    if not yml_without_video:
        print("  (none)")
    for stem in yml_without_video:
        kinds_present = [k for k in stems_by_kind if stem in stems_by_kind[k]]
        print(f"  {stem}  ->  yml kinds: {kinds_present}")


def main():
    if len(sys.argv) < 2:
        print("usage: python virat_inventory.py <VIRAT_ROOT>")
        sys.exit(1)
    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"ERROR: {root} is not a directory")
        sys.exit(1)

    print(f"PyAV available: {HAVE_PYAV}")
    if not HAVE_PYAV:
        print("Falling back to ffprobe subprocess for video probing.")

    videos, yml_by_kind, other = walk_and_categorize(root)
    counts = ", ".join(f"{len(yml_by_kind[k])} *.{k}.yml" for k in
                        ("activities", "geom", "regions", "types"))
    print(f"\nFound {len(videos)} video file(s), {counts}, {len(other)} other file(s).")

    print_video_table(videos)
    for kind in ("activities", "geom", "regions", "types"):
        print_yml_dump(kind, yml_by_kind[kind])
    print_pairing(videos, yml_by_kind)

    print("\nYML FIELD MEANING NOT ASSUMED — identify frame/time-bound fields "
          "(e.g. ts0/ts1/keyframe) from raw lines above; that mapping is the next task.")


if __name__ == "__main__":
    main()
