"""Phase 6 — NExT-QA dev/report partition + retrievability pre-check.

Read-only diagnostic: no ingest, no query, no video loading.
Outputs:
  1. Family stratification: val baseline vs dev(100) vs report(500)
  2. Dev video resolution: map hits, file presence (re-runnable as download completes)
  3. Retrievability ceiling pre-check (metadata heuristic — NOT a measured ceiling)
  4. Writes eval/data/nextqa/dev_100.jsonl and report_500.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval.nextqa_loader import load_split, partition, resolve_video

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR   = REPO / "eval" / "data" / "nextqa"
VAL_CSV    = DATA_DIR / "val.csv"
MAP_JSON   = DATA_DIR / "map_vid_vidorID.json"
VIDEO_ROOT = DATA_DIR / "NExTVideo"

DEV_OUT    = DATA_DIR / "dev_100.jsonl"
REP_OUT    = DATA_DIR / "report_500.jsonl"

# ── Confirm paths ─────────────────────────────────────────────────────────────

print("=== PATH CHECK ===")
for p in [VAL_CSV, MAP_JSON, VIDEO_ROOT]:
    print(f"  {'OK' if p.exists() else 'MISSING':7} {p}")
print()

# ── Load + partition ──────────────────────────────────────────────────────────

print("=== LOADING val.csv ===")
val_rows = load_split(VAL_CSV)
print(f"  Loaded {len(val_rows)} rows from {VAL_CSV.name}")
print()

print("=== PARTITION: dev=100 / report=500 (seed=20260629, stratified by family) ===")
dev, report = partition(val_rows, dev_n=100, report_n=500, seed=20260629)
print()

# ── Dev video resolution ──────────────────────────────────────────────────────

print("=== DEV VIDEO RESOLUTION ===")

with open(MAP_JSON, encoding="utf-8") as f:
    vidmap: dict = json.load(f)

dev_video_ids = sorted({r["video"] for r in dev})
n_distinct = len(dev_video_ids)

in_map, present, absent = [], [], []
for vid in dev_video_ids:
    vidor_rel = vidmap.get(vid)
    if vidor_rel is None:
        absent.append((vid, "<not in map>"))
    else:
        path = VIDEO_ROOT / f"{vidor_rel}.mp4"
        if path.exists():
            present.append((vid, str(path)))
        else:
            absent.append((vid, str(path)))
        in_map.append(vid)

print(f"  Distinct dev videos : {n_distinct}")
print(f"  Found in map        : {len(in_map)} / {n_distinct}")
print(f"  File present        : {len(present)} / {len(in_map)}")
print(f"  File absent         : {len(absent)}")
if absent:
    print("  Absent video IDs (first 20):")
    for vid, path in absent[:20]:
        print(f"    {vid}  ->  {path}")
    if len(absent) > 20:
        print(f"    ... and {len(absent) - 20} more")
print()

# ── Retrievability ceiling pre-check (metadata heuristic) ────────────────────

print("=== RETRIEVABILITY PRE-CHECK (metadata heuristic — NOT a measured ceiling) ===")
print("  NOTE: frame budgets are estimated from IRIS measured survival band (11%–15%).")
print("  This is a coarse proxy. The real ceiling requires ingest (6.3-b-ii).")
print()

SURVIVAL_LOW  = 0.11
SURVIVAL_HIGH = 0.15
THRESHOLDS = [1000, 1500, 2000]

for fam in sorted({r["family"] for r in dev}):
    fam_rows = [r for r in dev if r["family"] == fam]
    fcs = sorted(r["frame_count"] for r in fam_rows)
    n = len(fcs)
    median_fc = fcs[n // 2] if n % 2 == 1 else (fcs[n // 2 - 1] + fcs[n // 2]) / 2

    print(f"  Family {fam}  (n={n})")
    print(f"    median frame_count : {median_fc:.0f}")
    print(f"    estimated keyframe budget @ 11%: {median_fc * SURVIVAL_LOW:.0f}  "
          f"@ 15%: {median_fc * SURVIVAL_HIGH:.0f}")
    for thr in THRESHOLDS:
        frac = sum(1 for fc in fcs if fc > thr) / n
        print(f"    frame_count > {thr:4d} : {frac:.1%}  ({sum(1 for fc in fcs if fc > thr)}/{n})")
    print()

# All families combined
all_fcs = sorted(r["frame_count"] for r in dev)
n = len(all_fcs)
median_all = all_fcs[n // 2] if n % 2 == 1 else (all_fcs[n // 2 - 1] + all_fcs[n // 2]) / 2
print(f"  ALL dev  (n={n})")
print(f"    median frame_count : {median_all:.0f}")
print(f"    estimated keyframe budget @ 11%: {median_all * SURVIVAL_LOW:.0f}  "
      f"@ 15%: {median_all * SURVIVAL_HIGH:.0f}")
for thr in THRESHOLDS:
    frac = sum(1 for fc in all_fcs if fc > thr) / n
    print(f"    frame_count > {thr:4d} : {frac:.1%}  ({sum(1 for fc in all_fcs if fc > thr)}/{n})")
print()

# ── Write frozen splits ───────────────────────────────────────────────────────

KEEP_KEYS = ["qid", "video", "question", "a0", "a1", "a2", "a3", "a4",
             "answer", "type", "family"]

def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in KEEP_KEYS}) + "\n")

write_jsonl(dev, DEV_OUT)
write_jsonl(report, REP_OUT)

print(f"=== WRITTEN ===")
print(f"  dev_100.jsonl   -> {DEV_OUT}  ({len(dev)} rows)")
print(f"  report_500.jsonl -> {REP_OUT}  ({len(report)} rows)")
