"""
Pre-download subset selection for MLVU.

Reuses scripts/mlvu_eval.py's task-file discovery, question loading, and
deterministic sampling logic (same MC_TASKS, seed=42, n=25/task, <=600s cap,
dedup to unique videos) but resolves duration from the annotation record's
own `duration` field instead of probing the actual video file -- so it can
run BEFORE any video is downloaded.

Verified against the real MLVU/MVLU (HF) schema on 2026-08-15: every task
JSON already carries top-level `question`, `candidates`, `answer`, `video`,
`duration` keys -- an exact match for mlvu_eval.py's QUESTION_KEYS/
CANDIDATE_KEYS/ANSWER_KEYS/VIDEO_KEYS aliases, and `duration` is a plain
float number of seconds. If a re-run of this script hits a schema mismatch
(missing `duration`, or load_task_questions raising HarnessError), it will
fail loud below rather than silently guessing.

Video paths in the HF repo are `MLVU/video/<task_json_stem>/<video>`, e.g.
`MLVU/video/6_anomaly_reco/surveil_20.mp4`. That per-task directory prefix
is what this script emits, so the output list can be passed directly as
`allow_patterns` to snapshot_download.

Usage:
  python scripts/mlvu_select_videos.py --anno-dir ./mlvu/MLVU/json \
      --out eval_results/mlvu_video_allowlist.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.mlvu_eval import (  # noqa: E402
    MC_TASKS, TASK_NAMES, HarnessError, find_task_file, deterministic_sample,
)


def load_task_records_with_duration(anno_path: Path, task_code: str) -> list[dict]:
    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if not isinstance(data, list):
        raise HarnessError(f"{anno_path}: expected a JSON list, got {type(data)}.")

    out = []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise HarnessError(f"{anno_path}[{i}]: expected dict record, got {type(rec)}.")
        if "duration" not in rec:
            raise HarnessError(
                f"{anno_path}[{i}]: no 'duration' field present. Schema has changed from "
                f"the verified 2026-08-15 MLVU/MVLU layout. Record keys: {sorted(rec.keys())}"
            )
        for req in ("question", "candidates", "answer", "video"):
            if req not in rec:
                raise HarnessError(
                    f"{anno_path}[{i}]: missing required field '{req}'. "
                    f"Record keys present: {sorted(rec.keys())}"
                )
        out.append({
            "task": task_code,
            "question_id": rec.get("question_id", f"{task_code}_{i}"),
            "question": str(rec["question"]).strip(),
            "video": str(rec["video"]),
            "duration": float(rec["duration"]),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anno-dir", type=Path, required=True)
    ap.add_argument("--n-per-task", type=int, default=25)
    ap.add_argument("--max-duration", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "eval_results" / "mlvu_video_allowlist.json")
    args = ap.parse_args()

    per_task_pool: dict[str, list[dict]] = {}
    skipped_by_length: dict[str, int] = {}

    for task in MC_TASKS:
        anno_path = find_task_file(args.anno_dir, task)
        records = load_task_records_with_duration(anno_path, task)
        kept = [r for r in records if r["duration"] <= args.max_duration]
        skipped_by_length[task] = len(records) - len(kept)
        print(f"[filter] {task}: {len(records)} loaded -> {len(kept)} <= {args.max_duration}s "
              f"(skipped_by_length={skipped_by_length[task]})")

        sampled = deterministic_sample(kept, args.n_per_task, args.seed)
        print(f"[sample] {task}: sampled {len(sampled)}/{len(kept)} "
              f"(n_per_task={args.n_per_task}, seed={args.seed})")
        per_task_pool[task] = sampled

    # dedup to unique videos, with per-task directory prefix (task_json_stem)
    video_patterns: set[str] = set()
    per_task_videos: dict[str, list[str]] = {}
    task_json_stem = {t: find_task_file(args.anno_dir, t).stem for t in MC_TASKS}
    for task in MC_TASKS:
        vids = sorted({r["video"] for r in per_task_pool[task]})
        per_task_videos[task] = vids
        for v in vids:
            video_patterns.add(f"MLVU/video/{task_json_stem[task]}/{v}")

    total_questions = sum(len(v) for v in per_task_pool.values())
    print(f"\n[select] {total_questions} questions sampled across {len(MC_TASKS)} tasks "
          f"-> {len(video_patterns)} unique video files needed")

    out = {
        "params": {"n_per_task": args.n_per_task, "max_duration_s": args.max_duration, "seed": args.seed},
        "per_task_n_sampled": {t: len(per_task_pool[t]) for t in MC_TASKS},
        "per_task_skipped_by_length": skipped_by_length,
        "unique_video_count": len(video_patterns),
        "allow_patterns": sorted(video_patterns),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
