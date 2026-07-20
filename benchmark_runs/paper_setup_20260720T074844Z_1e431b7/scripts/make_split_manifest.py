"""Deterministic, leakage-safe, video-level split generator for the (placeholder) NExT-GQA subset.

Splits are computed by hashing each video_id with a fixed seed -- no question ever crosses
a partition because partitioning happens on video_id, and every question inherits its video's
partition. This is a config/manifest-generation script only: it reads CSV metadata and writes
JSON, it does not decode video or run any model.
"""
import csv
import hashlib
import json
import sys

SEED = "iris-nextgqa-paper-setup-2026-07-20"
VAL_TUNE_FRACTION = 0.80

SUBSET_CSV = "data/nextqa_exp1a/nextqa_exp1a_subset.csv"
OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "split_manifest.json"


def stable_unit(video_id: str) -> float:
    h = hashlib.sha256(f"{SEED}:{video_id}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def main():
    rows_by_video = {}
    with open(SUBSET_CSV, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows_by_video.setdefault(row["video_id"], []).append(row)

    video_ids = sorted(rows_by_video.keys())
    scored = sorted(video_ids, key=stable_unit)
    n_val_tune = round(len(scored) * VAL_TUNE_FRACTION)
    val_tune_videos = sorted(scored[:n_val_tune])
    val_confirm_videos = sorted(scored[n_val_tune:])

    def qids(videos):
        out = []
        for v in videos:
            for i, row in enumerate(rows_by_video[v]):
                out.append(f"{v}::{i}")
        return out

    manifest = {
        "generated_by": "scripts/make_split_manifest.py",
        "seed": SEED,
        "algorithm": "sha256(seed + ':' + video_id) -> uniform[0,1] via top 16 hex digits; sort videos "
                     "by this score; first floor(N*0.80) videos (by ID, tie-broken by sorted video_id "
                     "order after scoring) become val_tune, remainder become val_confirm. Partitioning "
                     "key is video_id only -- every question/event row inherits its video's partition, "
                     "so no video ever appears in more than one partition.",
        "source_note": "PLACEHOLDER SUBSET ONLY (see dataset_manifest.json non_official_working_subset). "
                        "official_test partition is NOT populated here because no official test annotation "
                        "file could be verified (see setup_failures.jsonl) -- this field is a stub with "
                        "video_ids: [] and must be filled in once the true official test split is acquired.",
        "val_tune_fraction": VAL_TUNE_FRACTION,
        "partitions": {
            "val_tune": {
                "video_count": len(val_tune_videos),
                "video_ids": val_tune_videos,
                "question_count": len(qids(val_tune_videos)),
                "question_ids": qids(val_tune_videos),
            },
            "val_confirm": {
                "video_count": len(val_confirm_videos),
                "video_ids": val_confirm_videos,
                "question_count": len(qids(val_confirm_videos)),
                "question_ids": qids(val_confirm_videos),
            },
            "official_test": {
                "video_count": 0,
                "video_ids": [],
                "question_count": 0,
                "question_ids": [],
                "status": "UNPOPULATED -- official test annotations not available locally, see setup_failures.jsonl",
            },
        },
    }

    overlap = set(val_tune_videos) & set(val_confirm_videos)
    assert not overlap, f"leakage: videos in both partitions: {overlap}"

    with open(OUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {OUT_PATH}: val_tune={len(val_tune_videos)} videos, val_confirm={len(val_confirm_videos)} videos, no overlap confirmed")


if __name__ == "__main__":
    main()
