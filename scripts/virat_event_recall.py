"""Pillar-1 measurement: codec ingest-gate event-recall scorer.

Given a VIRAT clip and its matching .viratdata.events.txt, computes
event_recall and frame_recall for the IRIS codec-gate survivor set,
and compares against uniform/random baselines at the same sampling
budget N.

events.txt format (verified: 10 space-separated columns, one row per frame-bbox):
  col1=event_id col2=event_type col3=duration_frames col4=start_frame
  col5=end_frame col6=current_frame col7-10=bbox(x,y,w,h)
An event's frame interval is [min(col6), max(col6)] over all rows
sharing col1.
"""
import inspect
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris import charon_v


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
    result = charon_v.parse_video(clip_path, return_stats=True)
    print(f"parse_video return signature: {inspect.signature(charon_v.parse_video)}")

    if not isinstance(result, tuple) or len(result) != 2:
        print(f"UNEXPECTED return shape from parse_video(return_stats=True): {type(result)!r}")
        sys.exit(1)

    output_frames, stats = result
    print(f"output_frames: list of {len(output_frames)} dict(s)")
    if output_frames:
        print(f"keys of output_frames[0]: {sorted(output_frames[0].keys())}")
    else:
        print("output_frames is empty — cannot confirm frame_idx key.")
        sys.exit(1)

    if "frame_idx" not in output_frames[0]:
        print("frame_idx NOT found in output_frames[0] — STOPPING per spec, not guessing.")
        sys.exit(1)

    print("frame_idx confirmed present in output_frames dicts.")
    print(f"stats keys: {sorted(stats.keys())}")

    survivors = {f["frame_idx"] for f in output_frames}
    total_frames = stats.get("total")
    if total_frames is None:
        print("stats['total'] NOT found — STOPPING per spec, not guessing total_frames.")
        sys.exit(1)

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


def main():
    if len(sys.argv) < 3:
        print("usage: python virat_event_recall.py <CLIP.mp4> <CLIP.viratdata.events.txt>")
        sys.exit(1)

    clip_path = sys.argv[1]
    events_path = sys.argv[2]

    events = parse_events_txt(events_path)
    num_events = len(events)
    print(f"Parsed {num_events} event(s) from {events_path}")

    survivors, total_frames = get_survivor_frame_indices(clip_path)
    n = len(survivors)
    print(f"\nN (survivor count) = {n}, total_frames = {total_frames}, num_events = {num_events}")

    arms = {
        "gate": survivors,
        "uniform": uniform_sample(n, total_frames),
        "random": random_sample(n, total_frames, seed=42),
    }

    print(f"\n{'arm':<10} {'event_recall':>14} {'frame_recall':>14} {'N':>8}")
    print("-" * 50)
    for arm_name, sampled in arms.items():
        er = event_recall(events, sampled)
        fr = frame_recall(events, sampled)
        er_s = f"{er:.4f}" if er is not None else "n/a"
        fr_s = f"{fr:.4f}" if fr is not None else "n/a"
        print(f"{arm_name:<10} {er_s:>14} {fr_s:>14} {len(sampled):>8}")

    print(f"\nN={n}, total_frames={total_frames}, num_events={num_events}")


if __name__ == "__main__":
    main()
