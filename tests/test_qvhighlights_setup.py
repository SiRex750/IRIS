"""Tests for scripts/setup_qvhighlights.py using small synthetic fixtures only.

No real QVHighlights annotations or videos are used here -- everything is
constructed in-memory or under pytest's tmp_path.
"""
import json
import tarfile
from pathlib import Path

import pytest

from scripts.setup_qvhighlights import (
    build_manifests,
    safe_extract_tar,
    validate_split,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(**overrides):
    base = {
        "qid": 1,
        "query": "a person walks across the room",
        "vid": "abc123_0.0_150.0",
        "duration": 150,
        "relevant_windows": [[10, 20]],
    }
    base.update(overrides)
    return base


# --- annotation validation -------------------------------------------------

def test_valid_row_produces_no_invalid_counts(tmp_path):
    p = tmp_path / "split.jsonl"
    _write_jsonl(p, [_row()])
    stats = validate_split(p, "train")
    assert stats["invalid_row_counts"] == {}
    assert stats["n_rows"] == 1
    assert stats["duration_stats"]["count"] == 1
    assert stats["window_width_stats"]["count"] == 1


def test_duplicate_qid_detected(tmp_path):
    p = tmp_path / "split.jsonl"
    _write_jsonl(p, [_row(qid=1, vid="a_0_1"), _row(qid=1, vid="b_0_1")])
    stats = validate_split(p, "train")
    assert stats["invalid_row_counts"].get("duplicate_qid") == 1


def test_invalid_temporal_window_end_before_start(tmp_path):
    p = tmp_path / "split.jsonl"
    _write_jsonl(p, [_row(relevant_windows=[[20, 10]])])
    stats = validate_split(p, "train")
    assert stats["invalid_row_counts"].get("end_not_after_start") == 1


def test_negative_start_rejected(tmp_path):
    p = tmp_path / "split.jsonl"
    _write_jsonl(p, [_row(relevant_windows=[[-5, 10]])])
    stats = validate_split(p, "train")
    assert stats["invalid_row_counts"].get("negative_start") == 1


def test_end_exceeds_duration_rejected(tmp_path):
    p = tmp_path / "split.jsonl"
    _write_jsonl(p, [_row(duration=100, relevant_windows=[[10, 200]])])
    stats = validate_split(p, "train")
    assert stats["invalid_row_counts"].get("end_exceeds_duration") == 1


def test_end_within_documented_tolerance_accepted(tmp_path):
    p = tmp_path / "split.jsonl"
    # 0.3s over duration is within the documented 0.5s tolerance
    _write_jsonl(p, [_row(duration=100, relevant_windows=[[10, 100.3]])])
    stats = validate_split(p, "train")
    assert stats["invalid_row_counts"] == {}


def test_malformed_window_shape_rejected(tmp_path):
    p = tmp_path / "split.jsonl"
    _write_jsonl(p, [_row(relevant_windows=[[10, 20, 30]])])
    stats = validate_split(p, "train")
    assert stats["invalid_row_counts"].get("malformed_window") == 1


def test_non_finite_duration_rejected(tmp_path):
    p = tmp_path / "split.jsonl"
    _write_jsonl(p, [_row(duration=float("nan"))])
    stats = validate_split(p, "train")
    assert stats["invalid_row_counts"].get("invalid_duration") == 1


def test_empty_query_rejected(tmp_path):
    p = tmp_path / "split.jsonl"
    _write_jsonl(p, [_row(query="  ")])
    stats = validate_split(p, "train")
    assert stats["invalid_row_counts"].get("empty_or_missing_query") == 1


# --- train/val video overlap -----------------------------------------------

def test_train_val_video_overlap_detected(tmp_path):
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write_jsonl(train_p, [_row(qid=1, vid="shared_vid_0_150")])
    _write_jsonl(val_p, [_row(qid=2, vid="shared_vid_0_150")])
    train_stats = validate_split(train_p, "train")
    val_stats = validate_split(val_p, "val")
    overlap = train_stats["unique_videos"] & val_stats["unique_videos"]
    assert overlap == {"shared_vid_0_150"}


def test_no_overlap_when_videos_distinct(tmp_path):
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write_jsonl(train_p, [_row(qid=1, vid="train_vid_0_150")])
    _write_jsonl(val_p, [_row(qid=2, vid="val_vid_0_150")])
    train_stats = validate_split(train_p, "train")
    val_stats = validate_split(val_p, "val")
    assert (train_stats["unique_videos"] & val_stats["unique_videos"]) == set()


# --- safe archive extraction -------------------------------------------------

def _make_tar(tmp_path: Path, members: dict[str, bytes]) -> Path:
    archive = tmp_path / "test.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name, content in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            import io

            tf.addfile(info, io.BytesIO(content))
    return archive


def test_safe_extract_accepts_well_formed_archive(tmp_path):
    archive = _make_tar(tmp_path, {"videos/ok.mp4": b"data"})
    dest = tmp_path / "extracted"
    dest.mkdir()
    safe_extract_tar(archive, dest)
    assert (dest / "videos" / "ok.mp4").read_bytes() == b"data"


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = _make_tar(tmp_path, {"../../etc/evil": b"pwned"})
    dest = tmp_path / "extracted"
    dest.mkdir()
    with pytest.raises(RuntimeError, match="traversal"):
        safe_extract_tar(archive, dest)
    assert not (tmp_path / "etc").exists()


def test_safe_extract_rejects_absolute_path(tmp_path):
    archive = _make_tar(tmp_path, {"/etc/evil": b"pwned"})
    dest = tmp_path / "extracted"
    dest.mkdir()
    with pytest.raises(RuntimeError, match="absolute"):
        safe_extract_tar(archive, dest)


# --- manifest generation: dedup, path containment, missing-video report ----

def _setup_data_root(tmp_path: Path, *, with_one_video: bool = False) -> Path:
    root = tmp_path / "data_root"
    (root / "annotations").mkdir(parents=True)
    (root / "videos").mkdir(parents=True)
    _write_jsonl(
        root / "annotations" / "highlight_train_release.jsonl",
        [_row(qid=1, vid="vidA_0_150"), _row(qid=2, vid="vidB_0_150")],
    )
    _write_jsonl(
        root / "annotations" / "highlight_val_release.jsonl",
        [_row(qid=101, vid="vidC_0_150")],
    )
    if with_one_video:
        (root / "videos" / "vidA_0_150.mp4").write_bytes(b"fake")
    return root


def test_build_manifests_marks_missing_videos(tmp_path, monkeypatch):
    root = _setup_data_root(tmp_path, with_one_video=True)
    out_dir = tmp_path / "manifests"
    build_manifests(root, {}, out_dir=out_dir)

    train_rows = [json.loads(l) for l in open(out_dir / "external_train.jsonl")]
    by_vid = {r["vid"]: r for r in train_rows}
    assert by_vid["vidA_0_150"]["video_available"] is True
    assert by_vid["vidB_0_150"]["video_available"] is False

    missing = json.load(open(out_dir / "missing_videos_report.json"))
    assert missing["n_missing"] == 2  # vidB (train) + vidC (val)


def test_build_manifests_no_duplicate_qids_within_split(tmp_path):
    root = tmp_path / "data_root"
    (root / "annotations").mkdir(parents=True)
    (root / "videos").mkdir(parents=True)
    _write_jsonl(
        root / "annotations" / "highlight_train_release.jsonl",
        [_row(qid=1, vid="vidA_0_150"), _row(qid=1, vid="vidB_0_150")],
    )
    _write_jsonl(root / "annotations" / "highlight_val_release.jsonl", [_row(qid=101, vid="vidC_0_150")])
    with pytest.raises(RuntimeError, match="duplicate qid"):
        build_manifests(root, {}, out_dir=tmp_path / "manifests")


def test_build_manifests_paths_stay_under_data_root(tmp_path):
    root = _setup_data_root(tmp_path)
    out_dir = tmp_path / "manifests"
    build_manifests(root, {}, out_dir=out_dir)
    video_root = str((root / "videos").resolve())
    for fname in ("external_train.jsonl", "external_dev.jsonl"):
        for line in open(out_dir / fname):
            row = json.loads(line)
            assert row["local_video_path"].startswith(video_root)


def test_build_manifests_idempotent(tmp_path):
    root = _setup_data_root(tmp_path, with_one_video=True)
    out_dir = tmp_path / "manifests"
    build_manifests(root, {}, out_dir=out_dir)
    first = (out_dir / "external_train.jsonl").read_text()
    build_manifests(root, {}, out_dir=out_dir)
    second = (out_dir / "external_train.jsonl").read_text()
    assert first == second


def test_no_test_split_file_referenced_anywhere():
    import scripts.setup_qvhighlights as mod

    assert "test" not in mod.ANNOTATION_FILES
    assert set(mod.FORBIDDEN_TEST_FILES) == {
        "highlight_test_release.jsonl",
        "highlight_test_with_gt.jsonl",
    }
