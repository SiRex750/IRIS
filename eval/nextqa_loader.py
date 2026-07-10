"""NExT-QA MC loader, partition helper, and video resolver.

Schema (verified against CSVs + lmms-lab/NExTQA HF dataset):
  video       : str  — YouTube video ID (string, even though HF exposed as int)
  frame_count : int
  width/height: int
  question    : str
  answer      : int  — 0-indexed into a0..a4
  qid         : int
  type        : str  — CW, CH, TN, TC, TP, DL, DC, DB, DO, ...
  a0..a4      : str  — five MC answer candidates
  family      : str  — C, T, D (derived; see FAMILY_MAP)

Family mapping: first letter of type → C/T/D.
Any unlisted code is mapped by first letter; if first letter is also unlisted,
family is set to '?' and reported.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

FAMILY_MAP: dict[str, str] = {
    "C": "C",  # CW, CH
    "T": "T",  # TN, TC, TP
    "D": "D",  # DL, DC, DB, DO, ...
}

KNOWN_TYPES = {"CW", "CH", "TN", "TC", "TP", "DL", "DC", "DB", "DO"}


def _family(type_code: str) -> str:
    first = type_code[0].upper() if type_code else "?"
    return FAMILY_MAP.get(first, "?")


def load_split(path: str | Path) -> list[dict]:
    """Read a NExT-QA MC CSV and return list of dicts.

    Casts answer, qid, frame_count to int. Attaches:
      - family ∈ {C, T, D, ?}
      - answer_text : correct answer string
    Leaves video as str. Reports unknown type codes to stdout.
    """
    path = Path(path)
    rows: list[dict] = []
    unknown_types: set[str] = set()

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            type_code = raw["type"]
            if type_code not in KNOWN_TYPES:
                unknown_types.add(type_code)
            fam = _family(type_code)
            row: dict[str, Any] = {
                "video":       str(raw["video"]),
                "frame_count": int(raw["frame_count"]),
                "width":       int(raw["width"]),
                "height":      int(raw["height"]),
                "question":    raw["question"],
                "answer":      int(raw["answer"]),
                "qid":         int(raw["qid"]),
                "type":        type_code,
                "family":      fam,
                "a0": raw["a0"], "a1": raw["a1"], "a2": raw["a2"],
                "a3": raw["a3"], "a4": raw["a4"],
                "answer_text": raw[f"a{raw['answer']}"],
            }
            rows.append(row)

    if unknown_types:
        print(f"[load_split] UNMAPPED type codes in {path.name}: {sorted(unknown_types)}")

    return rows


def partition(
    val_rows: list[dict],
    dev_n: int = 100,
    report_n: int = 500,
    seed: int = 20260629,
) -> tuple[list[dict], list[dict]]:
    """Stratified partition of val_rows into (dev, report), disjoint by qid.

    Stratified by family so dev and report each preserve C/T/D proportions of val.
    Prints realized family counts for dev and report.
    """
    assert dev_n + report_n <= len(val_rows), (
        f"dev_n+report_n={dev_n+report_n} exceeds val size {len(val_rows)}"
    )

    # Group by family
    by_family: dict[str, list[dict]] = {}
    for r in val_rows:
        by_family.setdefault(r["family"], []).append(r)

    families = sorted(by_family.keys())
    total = len(val_rows)
    rng = random.Random(seed)

    # Shuffle each bucket deterministically
    for fam in families:
        rng.shuffle(by_family[fam])

    # Compute per-family allocation (round; adjust largest bucket for remainder)
    def allocate(n: int) -> dict[str, int]:
        alloc = {fam: round(n * len(by_family[fam]) / total) for fam in families}
        diff = n - sum(alloc.values())
        if diff != 0:
            # Adjust the largest family
            largest = max(families, key=lambda f: len(by_family[f]))
            alloc[largest] += diff
        return alloc

    dev_alloc = allocate(dev_n)
    report_alloc = allocate(report_n)

    dev: list[dict] = []
    report: list[dict] = []

    for fam in families:
        bucket = by_family[fam]
        d = dev_alloc[fam]
        r = report_alloc[fam]
        assert d + r <= len(bucket), (
            f"family {fam}: need {d+r} but only {len(bucket)} available"
        )
        dev.extend(bucket[:d])
        report.extend(bucket[d:d + r])

    # Verify disjoint — qid is per-video (only ~8 unique values); use (video, qid) as key
    dev_keys = {(r["video"], r["qid"]) for r in dev}
    rep_keys = {(r["video"], r["qid"]) for r in report}
    assert not dev_keys & rep_keys, "dev/report (video,qid) overlap detected"

    # Print stratification
    val_counts  = Counter(r["family"] for r in val_rows)
    dev_counts  = Counter(r["family"] for r in dev)
    rep_counts  = Counter(r["family"] for r in report)

    print(f"{'family':>8} | {'val N':>6} | {'val %':>6} | {'dev N':>6} | {'dev %':>6} | {'rep N':>6} | {'rep %':>6}")
    print("-" * 60)
    for fam in sorted(set(val_counts) | set(dev_counts) | set(rep_counts)):
        vn = val_counts.get(fam, 0)
        dn = dev_counts.get(fam, 0)
        rn = rep_counts.get(fam, 0)
        print(
            f"{fam:>8} | {vn:>6} | {vn/total*100:>5.1f}% | "
            f"{dn:>6} | {dn/dev_n*100:>5.1f}% | "
            f"{rn:>6} | {rn/report_n*100:>5.1f}%"
        )
    print(f"{'TOTAL':>8} | {total:>6} | {'100.0%':>6} | {len(dev):>6} | {'100.0%':>6} | {len(report):>6} | {'100.0%':>6}")

    return dev, report


def resolve_video(
    video_id: str,
    video_root: str | Path,
    flat_layout: bool = True,
    map_path: str | Path | None = None,
) -> tuple[str, bool]:
    """Build the expected file path for a video and return (path_str, exists: bool).

    flat_layout=True (default): eval/data/nextqa/NExTVideo_flat/{id}.mp4
      — no map needed; used for the VLM2Vec mirror.
    flat_layout=False: bucket-path layout {bucket}/{id}.mp4 via map_vid_vidorID.json
      — requires map_path; used for the original Vidor directory structure.
    """
    video_root = Path(video_root)

    if flat_layout:
        candidate = video_root / f"{video_id}.mp4"
        return (str(candidate), candidate.exists())

    # Bucket-path layout (original Vidor)
    if map_path is None:
        raise ValueError("map_path required when flat_layout=False")
    with open(Path(map_path), encoding="utf-8") as f:
        vidmap: dict = json.load(f)
    vidor_rel = vidmap.get(str(video_id))
    if vidor_rel is None:
        return (f"<not in map: {video_id}>", False)
    candidate = video_root / f"{vidor_rel}.mp4"
    return (str(candidate), candidate.exists())
