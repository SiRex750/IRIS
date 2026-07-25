"""Deterministic stratified pilot-video selection for QVHighlights external_dev.

Selects a fixed-seed sample of unique external_dev videos, stratified across
duration, query length, relevant-window count, and query-count-per-video.
This script performs SELECTION ONLY: it reads the existing external_dev
manifest, computes strata, and writes the pilot selection files. It does
not download, validate, or touch any video file.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEV_MANIFEST = REPO / "external_data" / "qvhighlights" / "manifests" / "external_dev.jsonl"
OUT_DIR = REPO / "tuning" / "qvhighlights_encoder_screening_v1"
SEED = 42
TARGET_N = 200


def load_dev_rows() -> list[dict]:
    rows = []
    with open(DEV_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_by_video(rows: list[dict]) -> dict[str, dict]:
    by_vid: dict[str, dict] = {}
    for r in rows:
        vid = r["vid"]
        entry = by_vid.setdefault(vid, {
            "vid": vid,
            "split": r["split"],
            "duration": r["duration"],
            "qids": [],
            "query_lengths": [],
            "window_counts": [],
        })
        entry["qids"].append(r["qid"])
        entry["query_lengths"].append(len(r["query"].split()))
        entry["window_counts"].append(len(r["relevant_windows"]))
    return by_vid


def duration_bucket(duration: float) -> str:
    # QVHighlights durations are heavily skewed (1511/1519 unique dev videos
    # are exactly 150s -- these are fixed ~2.5-minute clips by dataset
    # design). A quantile/tercile split on such a skewed distribution
    # degenerates to a single bucket, so fixed thresholds are used instead
    # to keep "short/medium/long" meaningful.
    if duration < 140:
        return "short"
    if duration < 150:
        return "medium"
    return "long"


def query_length_bucket(median_len: float, rep_len: float) -> str:
    return "short_query" if rep_len <= median_len else "long_query"


def window_count_bucket(max_windows: int) -> str:
    if max_windows <= 1:
        return "single_window"
    if max_windows <= 3:
        return "few_windows"
    return "many_windows"


def query_count_bucket(n_queries: int) -> str:
    return "single_query_video" if n_queries == 1 else "multi_query_video"


def compute_strata(by_vid: dict[str, dict]) -> dict[str, str]:
    all_rep_lengths = sorted(
        (sum(e["query_lengths"]) / len(e["query_lengths"])) for e in by_vid.values()
    )
    median_len = all_rep_lengths[len(all_rep_lengths) // 2]

    strata = {}
    for vid, e in by_vid.items():
        rep_len = sum(e["query_lengths"]) / len(e["query_lengths"])
        max_windows = max(e["window_counts"])
        n_queries = len(e["qids"])
        bucket = "|".join([
            f"dur={duration_bucket(e['duration'])}",
            f"qlen={query_length_bucket(median_len, rep_len)}",
            f"win={window_count_bucket(max_windows)}",
            f"qcount={query_count_bucket(n_queries)}",
        ])
        strata[vid] = bucket
    return strata


def stratified_sample(by_vid: dict[str, dict], strata: dict[str, str], *, target_n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for vid, bucket in strata.items():
        groups[bucket].append(vid)
    for bucket in groups:
        groups[bucket].sort()  # deterministic order before shuffling
        rng.shuffle(groups[bucket])

    total = sum(len(v) for v in groups.values())
    selected: list[str] = []
    quotas = {}
    for bucket, members in groups.items():
        quota = max(1, round(target_n * len(members) / total))
        quotas[bucket] = min(quota, len(members))

    for bucket, quota in quotas.items():
        selected.extend(groups[bucket][:quota])

    # Trim/fill to exactly target_n deterministically.
    selected = sorted(set(selected))
    rng.shuffle(selected)
    if len(selected) > target_n:
        selected = sorted(selected[:target_n])
    elif len(selected) < target_n:
        remaining = sorted(set(by_vid.keys()) - set(selected))
        rng.shuffle(remaining)
        selected = sorted(selected + remaining[: target_n - len(selected)])

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-n", type=int, default=TARGET_N)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    rows = load_dev_rows()
    by_vid = aggregate_by_video(rows)
    strata = compute_strata(by_vid)
    selected_vids = stratified_sample(by_vid, strata, target_n=args.target_n, seed=args.seed)

    video_root = REPO / "external_data" / "qvhighlights" / "videos"
    records = []
    for vid in selected_vids:
        e = by_vid[vid]
        records.append({
            "video_id": vid,
            "source_split": "external_dev",
            "duration": e["duration"],
            "query_ids": e["qids"],
            "relevant_window_count_per_query": e["window_counts"],
            "relevant_window_count_total": sum(e["window_counts"]),
            "selection_bucket": strata[vid],
            "download_status": "not_attempted",
            "local_path": str(video_root / f"{vid}.mp4"),
            "file_size_bytes": None,
            "sha256": None,
            "decoding_validation_status": "not_attempted",
            "video_available": False,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "pilot_selection.json"
    csv_path = args.out_dir / "pilot_selection.csv"

    bucket_counts = defaultdict(int)
    for r in records:
        bucket_counts[r["selection_bucket"]] += 1

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "target_n": args.target_n,
            "n_selected": len(records),
            "source_manifest": str(DEV_MANIFEST.relative_to(REPO)),
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "download_attempted": False,
            "download_blocker": (
                "QVHighlights raw videos are officially distributed only as a "
                "single 133.8 GiB archive with no documented per-video download "
                "endpoint; individual acquisition was not attempted (see "
                "tuning/qvhighlights_encoder_screening_v1/pilot_storage_report.md)."
            ),
            "records": records,
        }, f, indent=2)

    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            row = dict(r)
            row["query_ids"] = json.dumps(row["query_ids"])
            row["relevant_window_count_per_query"] = json.dumps(row["relevant_window_count_per_query"])
            w.writerow(row)

    print(f"[select] {len(records)} videos selected -> {json_path}")
    print(f"[select] bucket distribution: {dict(sorted(bucket_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
