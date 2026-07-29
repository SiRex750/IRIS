"""Step 1 verification for UCF-Crime VAD Experiment 1.

Full-population (not sampled) container-open + packet-level readability check
over every .mp4 under eval/data/ucf/videos, using PyAV (no standalone ffprobe
binary on this box -- same substitution already used in eval_results/ucf_inventory.md).

Also searches the checkout for the official test-video annotation file
(Temporal_Anomaly_Annotation_ForTestVideos.txt) and reports its absence.

Writes tuning/ucfcrime_vad_exp1/ucfcrime_dataset_validation.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import av

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = REPO_ROOT / "eval" / "data" / "ucf" / "videos"
OUT_PATH = REPO_ROOT / "tuning" / "ucfcrime_vad_exp1" / "ucfcrime_dataset_validation.json"

ANNOTATION_FILENAME = "Temporal_Anomaly_Annotation_ForTestVideos.txt"


def probe_one(path: Path) -> dict:
    rec = {
        "path": str(path.relative_to(REPO_ROOT)),
        "size_bytes": path.stat().st_size,
        "open_ok": False,
        "codec_name": None,
        "width": None,
        "height": None,
        "fps": None,
        "nb_frames_meta": None,
        "duration_s": None,
        "packets_readable": False,
        "n_packets_checked": 0,
        "packet_size_field_present": False,
        "packet_flags_field_present": False,
        "is_h264_or_hevc": False,
        "error": None,
    }
    try:
        container = av.open(str(path))
        try:
            stream = container.streams.video[0]
            rec["open_ok"] = True
            rec["codec_name"] = stream.codec_context.name
            rec["width"] = stream.codec_context.width
            rec["height"] = stream.codec_context.height
            rec["fps"] = float(stream.average_rate) if stream.average_rate else None
            rec["nb_frames_meta"] = stream.frames if stream.frames else None
            rec["duration_s"] = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None
                else None
            )
            rec["is_h264_or_hevc"] = rec["codec_name"] in ("h264", "hevc")

            # Packet-level readability check (ffprobe -show_packets equivalent):
            # demux up to 50 packets from the video stream, confirm .size and
            # .is_keyframe (PyAV's flags equivalent) are readable.
            n = 0
            container.seek(0)
            for packet in container.demux(stream):
                if packet.dts is None and packet.size == 0:
                    continue
                _ = int(packet.size)
                _ = bool(packet.is_keyframe)
                rec["packet_size_field_present"] = True
                rec["packet_flags_field_present"] = True
                n += 1
                if n >= 50:
                    break
            rec["n_packets_checked"] = n
            rec["packets_readable"] = n > 0
        finally:
            container.close()
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main() -> None:
    if not VIDEO_ROOT.exists():
        print(f"BLOCKER: {VIDEO_ROOT} does not exist.", file=sys.stderr)
        sys.exit(1)

    all_mp4 = sorted(VIDEO_ROOT.rglob("*.mp4"))
    all_files = sorted(p for p in VIDEO_ROOT.rglob("*") if p.is_file())
    non_mp4 = [p for p in all_files if p.suffix.lower() != ".mp4"]
    image_frame_dirs = [
        p for p in VIDEO_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    t0 = time.time()
    records = []
    for i, p in enumerate(all_mp4):
        records.append(probe_one(p))
        if (i + 1) % 50 == 0:
            print(f"  probed {i + 1}/{len(all_mp4)} ...", file=sys.stderr)
    elapsed = time.time() - t0

    n_open_ok = sum(1 for r in records if r["open_ok"])
    n_packets_readable = sum(1 for r in records if r["packets_readable"])
    n_h264_hevc = sum(1 for r in records if r["is_h264_or_hevc"])
    codec_counts: dict = {}
    for r in records:
        c = r["codec_name"] or "ERROR"
        codec_counts[c] = codec_counts.get(c, 0) + 1

    # Category breakdown by directory name.
    category_counts: dict = {}
    for p in all_mp4:
        cat = p.parent.name
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Search whole repo checkout (this worktree) for the official annotation file.
    annotation_hits = list(REPO_ROOT.rglob(ANNOTATION_FILENAME))

    total_bytes = sum(r["size_bytes"] for r in records)

    out = {
        "video_root": str(VIDEO_ROOT),
        "video_root_exists": True,
        "total_mp4_files": len(all_mp4),
        "total_non_mp4_files_in_tree": len(non_mp4),
        "non_mp4_file_examples": [str(p.relative_to(REPO_ROOT)) for p in non_mp4[:20]],
        "image_frame_files_found": len(image_frame_dirs),
        "total_size_bytes": total_bytes,
        "probe_elapsed_s": elapsed,
        "category_counts_by_directory": category_counts,
        "codec_distribution": codec_counts,
        "n_open_ok": n_open_ok,
        "n_open_failed": len(records) - n_open_ok,
        "n_packets_readable": n_packets_readable,
        "n_h264_or_hevc": n_h264_hevc,
        "open_failures": [
            {"path": r["path"], "error": r["error"]} for r in records if not r["open_ok"]
        ],
        "non_h264_hevc_files": [
            {"path": r["path"], "codec_name": r["codec_name"]}
            for r in records
            if r["open_ok"] and not r["is_h264_or_hevc"]
        ],
        "annotation_file_search": {
            "filename_searched": ANNOTATION_FILENAME,
            "found_paths": [str(p) for p in annotation_hits],
            "found": len(annotation_hits) > 0,
        },
        "official_290_test_video_set_present": False,
        "official_290_test_video_set_note": (
            "Cannot determine which of the 350 local .mp4 files (if any) belong to "
            "the official 290-video annotated test split: no train/test split list "
            "and no Temporal_Anomaly_Annotation_ForTestVideos.txt is present on this "
            "box, in this git worktree, or in any branch checked on either the "
            "SiRex750/IRIS or swarapotd-rgb/IRIS remotes. The local set also only "
            "covers 4 of the 13 official anomaly categories (Abuse, Arrest, Arson, "
            "Assault) plus the Testing_Normal_Videos_Anomaly folder (150 videos, "
            "consistent with the official 150-normal-test-video count) -- Burglary, "
            "Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, "
            "Stealing, and Vandalism are entirely absent."
        ),
        "per_file_records": records,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH}")
    print(f"open_ok={n_open_ok}/{len(records)} packets_readable={n_packets_readable}/{len(records)}")
    print(f"codec_distribution={codec_counts}")
    print(f"annotation file found: {out['annotation_file_search']['found']}")


if __name__ == "__main__":
    main()
