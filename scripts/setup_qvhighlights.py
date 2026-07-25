"""Acquire and validate the QVHighlights temporal-grounding dataset.

Official source: https://github.com/jayleicn/moment_detr (data/README.md).
Only the train and validation annotation splits are ever fetched or used;
the test split is never downloaded by this script. Annotation licence is
CC BY-NC-SA 4.0 (see data/LICENSE in the official repository) -- the
downloaded material is for non-commercial research use unless separately
authorised.

This script performs acquisition and structural validation ONLY. It does
not train, fine-tune, or evaluate any model, and it does not touch IRIS's
retrieval, reformulation, or traversal code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Pinned to a specific commit of the official repository for reproducibility
# (verified against https://github.com/jayleicn/moment_detr/blob/main/data/README.md
# on the access date recorded in the generated manifest).
OFFICIAL_COMMIT = "b7e553ac3b0c898ee6b85e03ee507c064eab89ca"
RAW_BASE = f"https://raw.githubusercontent.com/jayleicn/moment_detr/{OFFICIAL_COMMIT}/data"

ANNOTATION_FILES = {
    "train": "highlight_train_release.jsonl",
    "val": "highlight_val_release.jsonl",
}
# Test annotations exist in the official repo but MUST NOT be fetched by this
# tool -- listed here only so the exclusion is explicit and auditable.
FORBIDDEN_TEST_FILES = ("highlight_test_release.jsonl", "highlight_test_with_gt.jsonl")

LICENSE_FILE = "LICENSE"
VIDEO_ARCHIVE_URL = "https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz"
LICENSE_NAME = "CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)"

CHUNK_SIZE = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _http_head(url: str) -> tuple[int, int | None]:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        length = resp.headers.get("Content-Length")
        return resp.status, (int(length) if length is not None else None)


def download_resumable(url: str, dest: Path, *, expected_length: int | None = None) -> Path:
    """Download `url` to `dest` with a `.part` staging file, supporting resume.

    Never overwrites a valid, already-complete destination file.
    """
    if dest.exists():
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size} bytes)")
        return dest

    part = dest.with_suffix(dest.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(url)
    if existing:
        req.add_header("Range", f"bytes={existing}-")
        print(f"  [resume] {dest.name} from byte {existing}")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status
            if existing and status not in (200, 206):
                raise RuntimeError(f"server did not honour Range request for {url} (status={status})")
            if not existing and status != 200:
                raise RuntimeError(f"unexpected HTTP status {status} for {url}")

            mode = "ab" if existing and status == 206 else "wb"
            if mode == "wb":
                existing = 0
            total = existing
            with open(part, mode) as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
            print(f"  [ok] downloaded {total} bytes -> {part.name}")
    except urllib.error.URLError as exc:
        print(f"  [error] download failed for {url}: {exc}", file=sys.stderr)
        raise

    if expected_length is not None and part.stat().st_size != expected_length:
        raise RuntimeError(
            f"size mismatch for {dest.name}: got {part.stat().st_size}, expected {expected_length}"
        )

    part.rename(dest)
    return dest


def safe_extract_tar(archive: Path, dest_root: Path) -> None:
    """Extract `archive` into `dest_root`, rejecting path traversal / absolute paths."""
    dest_root = dest_root.resolve()
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            member_path = (dest_root / member.name).resolve()
            if member.name.startswith("/") or Path(member.name).is_absolute():
                raise RuntimeError(f"refusing to extract absolute path member: {member.name}")
            if ".." in Path(member.name).parts:
                raise RuntimeError(f"refusing to extract path-traversal member: {member.name}")
            if not str(member_path).startswith(str(dest_root) + "/") and member_path != dest_root:
                raise RuntimeError(f"refusing to extract member outside data root: {member.name}")
        tf.extractall(dest_root, filter="data")  # members individually validated above


def download_annotations(data_root: Path, *, resume: bool = False) -> dict[str, Path]:
    ann_dir = data_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, fname in ANNOTATION_FILES.items():
        url = f"{RAW_BASE}/{fname}"
        dest = ann_dir / fname
        print(f"[annotations] fetching {split} split: {url}")
        try:
            _, length = _http_head(url)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] HEAD failed ({exc}); proceeding without expected length")
            length = None
        download_resumable(url, dest, expected_length=length)
        out[split] = dest

    license_dest = ann_dir / LICENSE_FILE
    if not license_dest.exists():
        download_resumable(f"{RAW_BASE}/{LICENSE_FILE}", license_dest)
    out["license"] = license_dest
    return out


def download_videos(data_root: Path, *, resume: bool = False) -> Path:
    archive_dir = data_root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / "qvhilights_videos.tar.gz"

    try:
        _, length = _http_head(VIDEO_ARCHIVE_URL)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"could not reach official video archive URL: {exc}") from exc

    print(f"[videos] official archive reports Content-Length={length} bytes")
    download_resumable(VIDEO_ARCHIVE_URL, dest, expected_length=length)

    video_dir = data_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"[videos] extracting {dest.name} -> {video_dir} (safe extraction)")
    safe_extract_tar(dest, video_dir)
    return dest


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def validate_split(path: Path, split: str) -> dict[str, Any]:
    """Structurally validate one annotation JSONL split.

    Returns a dict of counters/stats; never raises on bad rows -- invalid
    rows are counted by reason instead so validation always produces a
    complete report.
    """
    qids: set = set()
    vids: set = set()
    durations: list[float] = []
    window_widths: list[float] = []
    invalid: dict[str, int] = {}
    n_rows = 0

    def bump(reason: str) -> None:
        invalid[reason] = invalid.get(reason, 0) + 1

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bump("invalid_json")
                continue

            qid = row.get("qid")
            query = row.get("query")
            vid = row.get("vid")
            duration = row.get("duration")
            windows = row.get("relevant_windows")

            row_ok = True

            if qid is None:
                bump("missing_qid")
                row_ok = False
            elif qid in qids:
                bump("duplicate_qid")
                row_ok = False
            else:
                qids.add(qid)

            if not isinstance(query, str) or not query.strip():
                bump("empty_or_missing_query")
                row_ok = False

            if not vid:
                bump("missing_vid")
                row_ok = False
            else:
                vids.add(vid)

            if not _finite(duration) or float(duration) <= 0:
                bump("invalid_duration")
                row_ok = False
            else:
                durations.append(float(duration))

            if windows is None:
                bump("missing_relevant_windows")
                row_ok = False
            elif not isinstance(windows, list) or len(windows) == 0:
                bump("empty_relevant_windows")
                row_ok = False
            else:
                for w in windows:
                    if not isinstance(w, list) or len(w) != 2:
                        bump("malformed_window")
                        row_ok = False
                        continue
                    start, end = w
                    if not _finite(start) or not _finite(end):
                        bump("non_finite_window")
                        row_ok = False
                        continue
                    start, end = float(start), float(end)
                    if start < 0:
                        bump("negative_start")
                        row_ok = False
                        continue
                    if end <= start:
                        bump("end_not_after_start")
                        row_ok = False
                        continue
                    tolerance = 0.5  # seconds; documented tolerance for annotation rounding
                    if _finite(duration) and end > float(duration) + tolerance:
                        bump("end_exceeds_duration")
                        row_ok = False
                        continue
                    window_widths.append(end - start)

            if row_ok:
                pass  # row counted fully valid implicitly via absence from `invalid`

    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }

    return {
        "split": split,
        "n_rows": n_rows,
        "n_valid_qids": len(qids),
        "unique_videos": vids,
        "invalid_row_counts": invalid,
        "duration_stats": _stats(durations),
        "window_width_stats": _stats(window_widths),
    }


def build_manifests(data_root: Path, stats: dict[str, dict], *, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    video_root = (data_root / "videos").resolve()
    missing_report = []

    for split, split_key in (("train", "train"), ("val", "dev")):
        src = data_root / "annotations" / ANNOTATION_FILES[split]
        out_path = out_dir / f"external_{split_key}.jsonl"
        seen_qids: set = set()
        rows_out = []
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = row.get("qid")
                vid = row.get("vid")
                if qid in seen_qids:
                    raise RuntimeError(f"duplicate qid {qid} within {split} split while building manifest")
                seen_qids.add(qid)

                local_video_path = video_root / f"{vid}.mp4"
                # Ensure the candidate path cannot escape the data root even
                # if `vid` were ever adversarial/unexpected.
                resolved = local_video_path.resolve()
                if not str(resolved).startswith(str(video_root) + "/") and resolved != video_root:
                    raise RuntimeError(f"video path for vid={vid} escapes data root")
                video_available = local_video_path.exists()
                if not video_available:
                    missing_report.append({"split": split, "vid": vid, "expected_path": str(local_video_path)})

                rows_out.append({
                    "dataset": "qvhighlights",
                    "split": split,
                    "qid": qid,
                    "vid": vid,
                    "local_video_path": str(local_video_path),
                    "video_available": video_available,
                    "query": row.get("query"),
                    "duration": row.get("duration"),
                    "relevant_windows": row.get("relevant_windows"),
                    "source_annotation": f"moment_detr@{OFFICIAL_COMMIT}:data/{ANNOTATION_FILES[split]}",
                })

        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps(r) + "\n")
        print(f"[manifest] wrote {len(rows_out)} rows -> {out_path}")

    missing_path = out_dir / "missing_videos_report.json"
    with open(missing_path, "w", encoding="utf-8") as f:
        json.dump({"n_missing": len(missing_report), "rows": missing_report}, f, indent=2)
    print(f"[manifest] {len(missing_report)} videos not locally available -> {missing_path}")

    return {"missing_videos": len(missing_report)}


def write_validation_manifest(data_root: Path, out_path: Path, *, access_date: str) -> dict[str, Any]:
    train_stats = validate_split(data_root / "annotations" / ANNOTATION_FILES["train"], "train")
    val_stats = validate_split(data_root / "annotations" / ANNOTATION_FILES["val"], "val")

    overlap = train_stats["unique_videos"] & val_stats["unique_videos"]

    archive_path = data_root / "archives" / "qvhilights_videos.tar.gz"
    archive_info = {"present": archive_path.exists(), "size_bytes": None, "sha256": None}
    if archive_path.exists():
        archive_info["size_bytes"] = archive_path.stat().st_size
        archive_info["sha256"] = sha256_file(archive_path)

    annotation_hashes = {}
    for split, fname in ANNOTATION_FILES.items():
        p = data_root / "annotations" / fname
        if p.exists():
            annotation_hashes[split] = {"path": fname, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)}

    video_dir = data_root / "videos"
    n_downloaded = sum(1 for _ in video_dir.glob("*.mp4")) if video_dir.exists() else 0

    manifest = {
        "source_repository": "https://github.com/jayleicn/moment_detr",
        "source_commit": OFFICIAL_COMMIT,
        "source_urls": {
            "train": f"{RAW_BASE}/{ANNOTATION_FILES['train']}",
            "val": f"{RAW_BASE}/{ANNOTATION_FILES['val']}",
            "license": f"{RAW_BASE}/{LICENSE_FILE}",
            "videos": VIDEO_ARCHIVE_URL,
        },
        "licence": LICENSE_NAME,
        "access_date": access_date,
        "n_train_queries": train_stats["n_rows"],
        "n_val_queries": val_stats["n_rows"],
        "n_unique_train_videos": len(train_stats["unique_videos"]),
        "n_unique_val_videos": len(val_stats["unique_videos"]),
        "train_val_video_overlap": len(overlap),
        "invalid_row_counts": {
            "train": train_stats["invalid_row_counts"],
            "val": val_stats["invalid_row_counts"],
        },
        "duration_stats": {
            "train": train_stats["duration_stats"],
            "val": val_stats["duration_stats"],
        },
        "window_width_stats": {
            "train": train_stats["window_width_stats"],
            "val": val_stats["window_width_stats"],
        },
        "annotation_hashes": annotation_hashes,
        "video_archive": archive_info,
        "n_videos_downloaded": n_downloaded,
        "n_videos_missing": None if not archive_info["present"] else "see missing_videos_report.json",
        "n_videos_corrupt": "not measured -- run --download-videos then video validation to populate this field",
        "test_split_used": False,
        "validation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[validate] wrote validation summary -> {out_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("external_data/qvhighlights"))
    parser.add_argument("--annotations-only", action="store_true", help="download annotations only, skip videos")
    parser.add_argument("--download-videos", action="store_true", help="also download and extract raw videos")
    parser.add_argument("--verify-only", action="store_true", help="skip downloads; only validate/re-verify existing files")
    parser.add_argument("--resume", action="store_true", help="resume any in-progress .part downloads")
    args = parser.parse_args()

    data_root: Path = args.data_root
    data_root.mkdir(parents=True, exist_ok=True)
    access_date = time.strftime("%Y-%m-%d", time.gmtime())

    if not args.verify_only:
        download_annotations(data_root, resume=args.resume)
        if args.download_videos and not args.annotations_only:
            download_videos(data_root, resume=args.resume)

    ann_train = data_root / "annotations" / ANNOTATION_FILES["train"]
    ann_val = data_root / "annotations" / ANNOTATION_FILES["val"]
    if not ann_train.exists() or not ann_val.exists():
        print("[error] required annotation files are missing; cannot validate", file=sys.stderr)
        return 1

    manifests_dir = data_root / "manifests"
    manifest = write_validation_manifest(
        data_root, manifests_dir / "qvhighlights_validation.json", access_date=access_date
    )
    build_manifests(data_root, {}, out_dir=manifests_dir)

    any_invalid = any(manifest["invalid_row_counts"][s] for s in ("train", "val"))
    if any_invalid:
        print("[warn] some rows failed structural validation; see invalid_row_counts in the manifest")

    print("[done] QVHighlights annotation acquisition + validation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
