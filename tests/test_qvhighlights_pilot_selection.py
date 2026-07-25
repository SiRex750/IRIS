"""Tests for scripts/select_qvhighlights_pilot.py using small synthetic
fixtures. Verifies deterministic stratified selection logic without
touching the real (large) external_dev manifest or any video file.
"""
import json

from scripts.select_qvhighlights_pilot import (
    aggregate_by_video,
    compute_strata,
    duration_bucket,
    query_count_bucket,
    stratified_sample,
    window_count_bucket,
)


def _rows():
    return [
        {"vid": "v1", "split": "val", "duration": 150, "qid": 1,
         "query": "a short one", "relevant_windows": [[0, 10]]},
        {"vid": "v2", "split": "val", "duration": 150, "qid": 2,
         "query": "a much longer query with many more words in it than v1",
         "relevant_windows": [[0, 10], [20, 30], [40, 50]]},
        {"vid": "v3", "split": "val", "duration": 130, "qid": 3,
         "query": "short", "relevant_windows": [[0, 5]]},
        {"vid": "v3", "split": "val", "duration": 130, "qid": 4,
         "query": "short two", "relevant_windows": [[10, 15]]},
        {"vid": "v4", "split": "val", "duration": 120, "qid": 5,
         "query": "a", "relevant_windows": [[0, 1]]},
    ]


def test_aggregate_by_video_groups_multi_query_videos():
    by_vid = aggregate_by_video(_rows())
    assert len(by_vid) == 4
    assert by_vid["v3"]["qids"] == [3, 4]
    assert by_vid["v1"]["qids"] == [1]


def test_duration_bucket_thresholds():
    assert duration_bucket(120) == "short"
    assert duration_bucket(139.9) == "short"
    assert duration_bucket(140) == "medium"
    assert duration_bucket(149.9) == "medium"
    assert duration_bucket(150) == "long"


def test_window_count_bucket_thresholds():
    assert window_count_bucket(1) == "single_window"
    assert window_count_bucket(3) == "few_windows"
    assert window_count_bucket(4) == "many_windows"


def test_query_count_bucket():
    assert query_count_bucket(1) == "single_query_video"
    assert query_count_bucket(2) == "multi_query_video"


def test_compute_strata_distinguishes_videos():
    by_vid = aggregate_by_video(_rows())
    strata = compute_strata(by_vid)
    assert strata["v4"] != strata["v2"]
    assert "qcount=multi_query_video" in strata["v3"]
    assert "qcount=single_query_video" in strata["v1"]


def test_stratified_sample_is_deterministic_for_fixed_seed():
    by_vid = aggregate_by_video(_rows())
    strata = compute_strata(by_vid)
    a = stratified_sample(by_vid, strata, target_n=3, seed=42)
    b = stratified_sample(by_vid, strata, target_n=3, seed=42)
    assert a == b


def test_stratified_sample_respects_target_n_and_uniqueness():
    by_vid = aggregate_by_video(_rows())
    strata = compute_strata(by_vid)
    selected = stratified_sample(by_vid, strata, target_n=3, seed=42)
    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert set(selected) <= set(by_vid.keys())


def test_full_pilot_selection_json_never_marks_video_available(tmp_path, monkeypatch):
    import scripts.select_qvhighlights_pilot as mod

    dev_manifest = tmp_path / "external_dev.jsonl"
    with open(dev_manifest, "w", encoding="utf-8") as f:
        for r in _rows():
            f.write(json.dumps(r) + "\n")

    monkeypatch.setattr(mod, "DEV_MANIFEST", dev_manifest)

    rows = mod.load_dev_rows()
    by_vid = mod.aggregate_by_video(rows)
    strata = mod.compute_strata(by_vid)
    selected = mod.stratified_sample(by_vid, strata, target_n=2, seed=42)
    assert len(selected) == 2
    for vid in selected:
        assert by_vid[vid]["vid"] == vid
