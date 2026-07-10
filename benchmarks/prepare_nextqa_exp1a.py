"""
Prepare NExT-QA subset for Benchmark 1A (Budget-Matched Frame Selection).

Since NExT-QA does not include per-question temporal grounding (no gsub/start-end
timestamps in standard val.csv), this script uses each question as an "event query"
and represents the "event span" as a synthetic interval based on question type:
  - Temporal type (T*): 3-second window at one of three temporal thirds
  - Causal type (C*): 5-second window in the middle quarter
  - Descriptive type (D*): full video span

Ground-truth spans are synthetic; the benchmark is marked SMOKE TEST.

Usage:
    python -m benchmarks.prepare_nextqa_exp1a \
        --video_dir /path/to/nextqa/videos \
        --val_csv   /path/to/nextqa/val.csv \
        --out_dir   data/nextqa_exp1a \
        [--max_videos N] \
        [--max_questions_per_video N]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def synthetic_span(question_type: str, duration: float, qid: int) -> tuple:
    """
    Derive a synthetic ground-truth event span from question type + duration.

    Temporal (T*): 3-second window placed at qid%3 third of the video.
    Causal  (C*): 5-second window in the middle half of the video.
    Descriptive (D*): full duration.
    """
    if duration <= 0:
        return (0.0, max(1.0, duration))

    qtype = (question_type or "D").strip().upper()

    if qtype.startswith("T"):
        third = duration / 3.0
        anchor = third * (qid % 3) + third * 0.25
        start = max(0.0, anchor - 1.5)
        end = min(duration, start + 3.0)
        return (round(start, 3), round(end, 3))

    elif qtype.startswith("C"):
        quarter = duration / 4.0
        start = max(0.0, quarter)
        end = min(duration, start + 5.0)
        return (round(start, 3), round(end, 3))

    else:
        return (0.0, round(duration, 3))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare NExT-QA subset CSV for Benchmark 1A."
    )
    parser.add_argument(
        "--video_dir",
        default="/home/ccbd/Documents/Public/DSP/CCBD-CAUSE/data/nextqa/videos",
        help="Directory containing NExT-QA .mp4 files (flat layout)",
    )
    parser.add_argument(
        "--val_csv",
        default="/home/ccbd/Documents/Public/DSP/CCBD-CAUSE/data/nextqa/val.csv",
        help="Path to NExT-QA val.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="data/nextqa_exp1a",
        help="Output directory for the prepared CSV",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=0,
        help="Max number of videos to include (0 = all available)",
    )
    parser.add_argument(
        "--max_questions_per_video",
        type=int,
        default=3,
        help="Max questions (events) per video",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"[ERROR] video_dir does not exist: {video_dir}", file=sys.stderr)
        sys.exit(1)

    val_csv_path = Path(args.val_csv)
    if not val_csv_path.exists():
        print(f"[ERROR] val_csv does not exist: {val_csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "nextqa_exp1a_subset.csv"

    # Load val.csv
    df_val = pd.read_csv(val_csv_path)
    df_val["video"] = df_val["video"].astype(str)
    print(f"Loaded {len(df_val)} rows from {val_csv_path}")
    print(f"Unique videos in val.csv: {df_val['video'].nunique()}")

    # Resolve local video files
    all_videos = sorted({
        f.stem for f in video_dir.iterdir() if f.suffix == ".mp4"
    })
    print(f"MP4 files found in {video_dir}: {len(all_videos)}")

    available_ids = set(all_videos)
    val_ids = set(df_val["video"].unique())
    matched_ids = sorted(val_ids & available_ids)
    print(f"Videos in both val.csv and local dir: {len(matched_ids)}")

    if not matched_ids:
        print("[ERROR] No matching videos found between val.csv and video_dir.", file=sys.stderr)
        sys.exit(1)

    # Build output rows
    records = []
    videos_used = []

    for vid_id in matched_ids:
        if args.max_videos > 0 and len(videos_used) >= args.max_videos:
            break

        video_path = str(video_dir / f"{vid_id}.mp4")
        sub = df_val[df_val["video"] == vid_id].head(args.max_questions_per_video)

        if sub.empty:
            continue

        # Get video duration
        duration = 0.0
        frame_count = int(sub.iloc[0].get("frame_count", 0)) if "frame_count" in sub.columns else 0

        try:
            import av
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                if container.duration:
                    duration = float(container.duration) / 1_000_000.0
                elif stream.duration and stream.time_base:
                    duration = float(stream.duration) * float(stream.time_base)
                elif frame_count > 0:
                    fps = float(stream.average_rate) if stream.average_rate else 25.0
                    duration = frame_count / fps
        except Exception:
            if frame_count > 0:
                duration = frame_count / 25.0

        if duration <= 0.5:
            print(f"  Skipping {vid_id}: duration {duration:.2f}s too short")
            continue

        videos_used.append(vid_id)

        for _, row in sub.iterrows():
            qid = int(row["qid"])
            question = str(row["question"])
            qtype = str(row.get("type", "D"))
            answer_idx = int(row.get("answer", 0))
            answer_key = f"a{answer_idx}"
            answer_text = str(row.get(answer_key, ""))

            start_t, end_t = synthetic_span(qtype, duration, qid)

            records.append({
                "video_id": vid_id,
                "path": video_path,
                "query": f"Find the event: {question}",
                "start_time": start_t,
                "end_time": end_t,
                "event_label": question,
                "split": "val",
                "duration": round(duration, 3),
                "qid": qid,
                "question_type": qtype,
                "answer_text": answer_text,
                "span_is_synthetic": True,
            })

    df_out = pd.DataFrame(records)
    df_out.to_csv(out_csv, index=False)

    print()
    print(f"Prepared {len(videos_used)} videos, {len(df_out)} event rows")
    print(f"Output CSV: {out_csv}")
    print()
    print("NOTE: start_time/end_time are SYNTHETIC (approximated from question type).")
    print("This benchmark run should be labelled SMOKE TEST ONLY.")


if __name__ == "__main__":
    main()
