"""NExT-GQA grounding-coverage probe on dev_100.

Read-only diagnostic: no model calls, no tuning, no thresholds.
Fails loudly if the NExT-GQA grounding file is absent.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO      = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO / "eval" / "data" / "nextqa"
DEV_JSONL = DATA_DIR / "dev_100.jsonl"
CACHE_DIR = DATA_DIR / "index_cache"

# Canonical search order for the NExT-GQA grounding file.
_GQA_CANDIDATES = [
    REPO / "NExT-GQA" / "gsub_val.json",
    REPO / "NExT-GQA-sparse" / "gsub_val.json",
    DATA_DIR / "gsub_val.json",
]

_SCHEMA_HINT = """\
Expected schema (gsub_val.json):
{
  "<video_id>": {                          // e.g. "3719793245"
    "duration": <int>,                     // video duration in seconds (ignored)
    "fps":      <float>,                   // container fps (ignored)
    "location": {                          // grounding annotations
      "<qid_str>": [                       // qid as STRING, e.g. "2"
        [<start_sec>, <end_sec>],          // one or more temporal spans in SECONDS
        ...
      ]
    }
  }
}
"""


# ── 1. Locate grounding file ──────────────────────────────────────────────────

gqa_path: Path | None = None
for _cand in _GQA_CANDIDATES:
    if _cand.exists():
        gqa_path = _cand
        break

if gqa_path is None:
    print("FATAL: NExT-GQA grounding file not found.", file=sys.stderr)
    print("Searched:", file=sys.stderr)
    for _cand in _GQA_CANDIDATES:
        print(f"  {_cand}", file=sys.stderr)
    print(file=sys.stderr)
    print(_SCHEMA_HINT, file=sys.stderr)
    print(
        "Obtain gsub_val.json from the NExT-GQA release\n"
        "(https://github.com/doc-doc/NExT-GQA) and place it at:\n"
        f"  {_GQA_CANDIDATES[0]}",
        file=sys.stderr,
    )
    sys.exit(1)


# ── 2. Load files ─────────────────────────────────────────────────────────────

with open(DEV_JSONL) as fh:
    dev_rows = [json.loads(line) for line in fh]

with open(gqa_path) as fh:
    gqa_raw = json.load(fh)

# Normalise GQA index: {(video_str, qid_str) -> list of [start, end] spans}
# Real schema: gqa_raw[vid]["location"] is a dict {qid_str -> [[s,e], ...]}.
# "duration" and "fps" are siblings of "location" and are skipped.
gqa_index: dict[tuple[str, str], list[list[float]]] = {}
for vid, vdata in gqa_raw.items():
    vid_str = str(vid)
    loc_dict = vdata["location"]  # {qid_str: [[start_sec, end_sec], ...]}
    for qid_str, spans in loc_dict.items():
        key = (vid_str, str(qid_str))
        if key in gqa_index:
            raise AssertionError(
                f"Collision in grounding file: video={vid_str} qid={qid_str} "
                "appears more than once — grounding file is malformed."
            )
        # Normalise: [[s,e], ...] or bare [s, e]
        if spans and not isinstance(spans[0], (list, tuple)):
            spans = [spans]
        gqa_index[key] = [[float(s), float(e)] for s, e in spans]

KEY_SCHEME = "(video as str, qid as str)"
print(f"Key scheme used for join: {KEY_SCHEME}")


# ── 3. Join dev_100 onto grounding ────────────────────────────────────────────

# Assert no (video, qid) collision in dev set itself.
dev_keys: set[tuple[str, str]] = set()
for row in dev_rows:
    k = (str(row["video"]), str(row["qid"]))
    assert k not in dev_keys, (
        f"Collision in dev_100: video={row['video']} qid={row['qid']} "
        "appears more than once — dev file is malformed."
    )
    dev_keys.add(k)

cached_vids = {p.stem for p in CACHE_DIR.glob("*.npz")}

# Per-row join results.
grounded_rows: list[dict] = []
grounded_cached_rows: list[dict] = []

for row in dev_rows:
    vid_str = str(row["video"])
    qid_str = str(row["qid"])
    key = (vid_str, qid_str)
    spans = gqa_index.get(key)
    if spans:
        entry = {**row, "_spans": spans, "_cached": vid_str in cached_vids}
        grounded_rows.append(entry)
        if entry["_cached"]:
            grounded_cached_rows.append(entry)


# ── 4. Collect all gold spans for distribution stats ─────────────────────────

all_widths: list[float] = []
for entry in grounded_rows:
    for span in entry["_spans"]:
        w = span[1] - span[0]
        all_widths.append(w)

def _pct(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    data_s = sorted(data)
    idx = (len(data_s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(data_s) - 1)
    return data_s[lo] + (data_s[hi] - data_s[lo]) * (idx - lo)


# ── 5. Frames-per-gold-span via PyAV ─────────────────────────────────────────

try:
    import av  # type: ignore
    _av_available = True
except ImportError:
    _av_available = False

VIDEO_ROOT = DATA_DIR / "NExTVideo_flat"

frames_per_span: list[float] = []
fps_missing: list[str] = []

if _av_available:
    fps_cache: dict[str, float | None] = {}
    for entry in grounded_rows:
        vid = str(entry["video"])
        if vid not in fps_cache:
            vid_path = VIDEO_ROOT / f"{vid}.mp4"
            if vid_path.exists():
                try:
                    with av.open(str(vid_path)) as container:
                        stream = container.streams.video[0]
                        r = stream.average_rate
                        fps_cache[vid] = float(r) if r else None
                except Exception:
                    fps_cache[vid] = None
            else:
                fps_cache[vid] = None
        fps = fps_cache[vid]
        if fps is None:
            fps_missing.append(vid)
            continue
        for span in entry["_spans"]:
            w = span[1] - span[0]
            frames_per_span.append(w * fps)
else:
    fps_missing = ["(PyAV not installed)"]


# ── 6. Build per-family counts ────────────────────────────────────────────────

families = ("C", "T", "D")

n_total:    dict[str, int] = {f: 0 for f in families}
n_grounded: dict[str, int] = {f: 0 for f in families}
n_gr_cached: dict[str, int] = {f: 0 for f in families}

for row in dev_rows:
    fam = row["family"]
    n_total[fam] = n_total.get(fam, 0) + 1

for entry in grounded_rows:
    fam = entry["family"]
    n_grounded[fam] = n_grounded.get(fam, 0) + 1

for entry in grounded_cached_rows:
    fam = entry["family"]
    n_gr_cached[fam] = n_gr_cached.get(fam, 0) + 1


# ── 7. Sanity check: D must be zero ──────────────────────────────────────────

if n_gr_cached.get("D", 0) != 0:
    print(
        f"\nERROR: n_grounded_AND_cached[D] = {n_gr_cached['D']} (expected 0).\n"
        "NExT-GQA grounds only C/T questions. A non-zero D count means\n"
        "the (video, qid) join is colliding across question families.\n"
        "Inspect the grounding file key scheme before proceeding.",
        file=sys.stderr,
    )
    sys.exit(2)


# ── 8. Print table ────────────────────────────────────────────────────────────

print()
print(f"{'Family':<8} {'n_dev':>6} {'n_grounded':>12} {'n_gr_AND_cached':>16}")
print("-" * 46)
for fam in families:
    print(
        f"{fam:<8} {n_total.get(fam, 0):>6} "
        f"{n_grounded.get(fam, 0):>12} "
        f"{n_gr_cached.get(fam, 0):>16}"
    )
print("-" * 46)
print(
    f"{'TOTAL':<8} {sum(n_total.values()):>6} "
    f"{sum(n_grounded.values()):>12} "
    f"{sum(n_gr_cached.values()):>16}"
)

print()
print("Gold-span width distribution (seconds):")
if all_widths:
    print(
        f"  n_spans={len(all_widths)}"
        f"  min={min(all_widths):.2f}"
        f"  median={statistics.median(all_widths):.2f}"
        f"  p90={_pct(all_widths, 90):.2f}"
        f"  max={max(all_widths):.2f}"
    )
else:
    print("  (no spans — join produced zero matches)")

print()
print("Frames per gold-span at true container fps (PyAV):")
if not _av_available:
    print("  SKIP: PyAV not installed (pip install av)")
elif frames_per_span:
    print(
        f"  n_spans={len(frames_per_span)}"
        f"  min={min(frames_per_span):.1f}"
        f"  median={statistics.median(frames_per_span):.1f}"
        f"  max={max(frames_per_span):.1f}"
    )
    if fps_missing:
        uniq_missing = sorted(set(fps_missing))
        print(f"  (fps unavailable for {len(uniq_missing)} video(s): {uniq_missing[:5]}{'...' if len(uniq_missing) > 5 else ''})")
else:
    print("  (no fps data available — video files may be missing)")
