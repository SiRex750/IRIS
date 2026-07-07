"""Pillar-1 measurement: codec ingest-gate event-recall scorer, all clips.

Loops over every matched (clip.mp4, events.txt) pair under
eval/data/virat/{videos,annotations}/ (matched by VIRAT_S stem), scores
each at matched budget N=survivor_count against uniform/random
baselines (seed=42), and reports per-clip rows plus aggregates.

events.txt format (verified: 10 space-separated columns, one row per
frame-bbox): col1=event_id col2=event_type col3=duration_frames
col4=start_frame col5=end_frame col6=current_frame col7-10=bbox(x,y,w,h)
An event's frame interval is [min(col6), max(col6)] over all rows
sharing col1.
"""
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris import charon_v

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "eval" / "data" / "virat"

STEM_RE = re.compile(r"^(VIRAT_S_[0-9A-Za-z_]+?)(?:\.viratdata\.events)?$")


def clip_stem(name):
    return name[:-4] if name.lower().endswith(".mp4") else name


def events_stem(name):
    if name.lower().endswith(".viratdata.events.txt"):
        return name[: -len(".viratdata.events.txt")]
    if name.lower().endswith(".events.txt"):
        return name[: -len(".events.txt")]
    return name


def find_matched_pairs(root):
    videos_dir = root / "videos"
    ann_dir = root / "annotations"

    videos = {clip_stem(p.name): p for p in videos_dir.glob("*.mp4")} if videos_dir.is_dir() else {}
    ann_candidates = list(ann_dir.glob("*.viratdata.events.txt")) if ann_dir.is_dir() else []
    if not ann_candidates and ann_dir.is_dir():
        ann_candidates = list(ann_dir.glob("*events.txt"))
    events = {events_stem(p.name): p for p in ann_candidates}

    matched_stems = sorted(set(videos) & set(events))
    video_only = sorted(set(videos) - set(events))
    events_only = sorted(set(events) - set(videos))

    pairs = [(stem, videos[stem], events[stem]) for stem in matched_stems]
    return pairs, video_only, events_only


def parse_events_txt(path):
    events = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split()
            if len(cols) < 10:
                continue
            event_id = cols[0]
            event_type = cols[1]
            current_frame = int(cols[5])
            ev = events.setdefault(event_id, {
                "event_id": event_id,
                "event_type": event_type,
                "frames": [],
            })
            ev["frames"].append(current_frame)

    out = []
    for ev in events.values():
        out.append({
            "event_id": ev["event_id"],
            "event_type": ev["event_type"],
            "start": min(ev["frames"]),
            "end": max(ev["frames"]),
        })
    return out


def get_survivor_frame_indices(clip_path):
    output_frames, stats = charon_v.parse_video(str(clip_path), return_stats=True)
    survivors = {f["frame_idx"] for f in output_frames}
    total_frames = stats["total"]
    return survivors, total_frames


def event_recall(events, sampled):
    if not events:
        return None
    hit = sum(1 for ev in events if any(ev["start"] <= idx <= ev["end"] for idx in sampled))
    return hit / len(events)


def frame_recall(events, sampled):
    event_frames = set()
    for ev in events:
        event_frames.update(range(ev["start"], ev["end"] + 1))
    if not event_frames:
        return None
    survived_event_frames = event_frames & sampled
    return len(survived_event_frames) / len(event_frames)


def survivor_precision(events, sampled):
    if not sampled:
        return None
    event_frames = set()
    for ev in events:
        event_frames.update(range(ev["start"], ev["end"] + 1))
    inside = sum(1 for idx in sampled if idx in event_frames)
    return inside / len(sampled)


def uniform_sample(n, total_frames):
    if n <= 0 or total_frames <= 0:
        return set()
    step = total_frames / n
    return {min(total_frames - 1, int(i * step)) for i in range(n)}


def random_sample(n, total_frames, seed=42):
    if n <= 0 or total_frames <= 0:
        return set()
    rng = random.Random(seed)
    n = min(n, total_frames)
    return set(rng.sample(range(total_frames), n))


def score_clip(stem, clip_path, events_path):
    events = parse_events_txt(events_path)
    num_events = len(events)
    survivors, total_frames = get_survivor_frame_indices(clip_path)
    n = len(survivors)

    arms = {
        "gate": survivors,
        "uniform": uniform_sample(n, total_frames),
        "random": random_sample(n, total_frames, seed=42),
    }

    rows = []
    for arm_name, sampled in arms.items():
        rows.append({
            "stem": stem,
            "arm": arm_name,
            "event_recall": event_recall(events, sampled),
            "frame_recall": frame_recall(events, sampled),
            "survivor_precision": survivor_precision(events, sampled),
            "num_events": num_events,
            "N": len(sampled),
            "total_frames": total_frames,
        })
    return rows


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "n/a"


def print_per_clip_table(all_rows):
    print("\n=== PER-CLIP x ARM TABLE ===")
    header = (f"{'stem':<35} {'arm':<8} {'event_recall':>13} {'frame_recall':>13} "
              f"{'survivor_prec':>14} {'num_events':>11} {'N':>6} {'total_frames':>13}")
    print(header)
    print("-" * len(header))
    for r in all_rows:
        print(f"{r['stem']:<35} {r['arm']:<8} {fmt(r['event_recall']):>13} "
              f"{fmt(r['frame_recall']):>13} {fmt(r['survivor_precision']):>14} "
              f"{r['num_events']:>11} {r['N']:>6} {r['total_frames']:>13}")


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def weighted_mean(values, weights):
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
    if not pairs:
        return None
    total_w = sum(w for _, w in pairs)
    if total_w == 0:
        return None
    return sum(v * w for v, w in pairs) / total_w


def print_aggregate_table(clip_scores_by_arm):
    print("\n=== AGGREGATE TABLE (per arm) ===")
    header = (f"{'arm':<8} {'event_recall(mean)':>20} {'event_recall(wmean)':>21} "
              f"{'frame_recall(mean)':>20} {'frame_recall(wmean)':>21} "
              f"{'survivor_prec(mean)':>21} {'survivor_prec(wmean)':>22}")
    print(header)
    print("-" * len(header))
    for arm_name, rows in clip_scores_by_arm.items():
        weights = [r["num_events"] for r in rows]
        er_mean = mean([r["event_recall"] for r in rows])
        er_wmean = weighted_mean([r["event_recall"] for r in rows], weights)
        fr_mean = mean([r["frame_recall"] for r in rows])
        fr_wmean = weighted_mean([r["frame_recall"] for r in rows], weights)
        sp_mean = mean([r["survivor_precision"] for r in rows])
        sp_wmean = weighted_mean([r["survivor_precision"] for r in rows], weights)
        print(f"{arm_name:<8} {fmt(er_mean):>20} {fmt(er_wmean):>21} "
              f"{fmt(fr_mean):>20} {fmt(fr_wmean):>21} "
              f"{fmt(sp_mean):>21} {fmt(sp_wmean):>22}")


def print_pooled_event_recall(clip_scores_by_arm, min_events=3):
    print(f"\n=== POOLED event_recall, clips with num_events >= {min_events} ===")
    header = f"{'arm':<8} {'event_recall(mean over qualifying clips)':>42} {'num_qualifying_clips':>22}"
    print(header)
    print("-" * len(header))
    for arm_name, rows in clip_scores_by_arm.items():
        qualifying = [r for r in rows if r["num_events"] >= min_events]
        er_mean = mean([r["event_recall"] for r in qualifying])
        print(f"{arm_name:<8} {fmt(er_mean):>42} {len(qualifying):>22}")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT

    pairs, video_only, events_only = find_matched_pairs(root)

    print(f"Root: {root}")
    print(f"\nMatched stems ({len(pairs)}):")
    for stem, _, _ in pairs:
        print(f"  {stem}")
    print(f"\nVideo-only stems, no matching events.txt ({len(video_only)}):")
    for stem in video_only:
        print(f"  {stem}")
    print(f"\nEvents-only stems, no matching video ({len(events_only)}):")
    for stem in events_only:
        print(f"  {stem}")

    all_rows = []
    clip_scores_by_arm = {"gate": [], "uniform": [], "random": []}
    total_events = 0
    clips_scored = 0
    clips_errored = []

    for stem, clip_path, events_path in pairs:
        try:
            rows = score_clip(stem, clip_path, events_path)
        except Exception as e:
            print(f"\nERROR scoring {stem}: {e!r} — skipping this clip, continuing with the rest.")
            clips_errored.append(stem)
            continue
        all_rows.extend(rows)
        for r in rows:
            clip_scores_by_arm[r["arm"]].append(r)
        total_events += rows[0]["num_events"]
        clips_scored += 1

    print_per_clip_table(all_rows)
    print_aggregate_table(clip_scores_by_arm)
    print_pooled_event_recall(clip_scores_by_arm, min_events=3)

    print(f"\nTotal clips scored: {clips_scored}")
    print(f"Total events (summed across clips): {total_events}")
    print(f"Clips errored during scoring: {len(clips_errored)} {clips_errored}")


if __name__ == "__main__":
    main()
