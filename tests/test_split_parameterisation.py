"""Tests for the --split parameterisation of scripts/val_confirm_e2e_eval.py.

The contract under test: --split changes EXACTLY three inputs (video id list,
questions CSV, gold-span JSON) and nothing else, and --split=val_confirm is a
no-op against every pre-existing call site.

These are offline structural tests -- no model call, no ingest, no server. The
empirical reproduction (actually re-running val_confirm and diffing the
per-question CSV) is a separate, expensive check reported in
tuning/official_test/runner_report.md; it cannot live here because it needs a
GPU, a served answerer, and ~30 minutes.
"""
from __future__ import annotations

import csv
import inspect
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import val_confirm_e2e_eval as ev  # noqa: E402

COMMITTED_VAL_CONFIRM_CSV = REPO / "tuning" / "val_confirm_e2e_per_question.csv"

# Columns that are wall-clock measurements and therefore cannot be
# byte-identical between two runs of the same deterministic pipeline.
TIMING_COLUMNS = {"retrieval_span_ms", "caption_answer_ms"}


# --- the three things a split is allowed to control -------------------------

def test_split_specs_cover_exactly_the_two_splits():
    assert set(ev.SPLIT_SPECS) == {"val_confirm", "official_test"}


def test_val_confirm_spec_points_at_val_inputs():
    spec = ev.SPLIT_SPECS["val_confirm"]
    assert spec["questions_csv"].name == "val.csv"
    assert spec["gold_json"].name == "gsub_val.json"


def test_official_test_spec_points_at_test_inputs():
    spec = ev.SPLIT_SPECS["official_test"]
    assert spec["questions_csv"].name == "test.csv"
    assert spec["gold_json"].name == "gsub_test.json"


def test_official_test_ids_come_from_benchmark_manifest_not_repo_root():
    """The repo-root split_manifest.json has no official_test partition at all;
    only the benchmark_runs copy does. Reading the wrong one is a silent
    catastrophe, so pin it."""
    assert ev.BENCHMARK_MANIFEST.is_file()
    root = json.loads((REPO / "split_manifest.json").read_text())
    assert "partitions" not in root, (
        "repo-root manifest unexpectedly grew a partitions key -- "
        "re-check which manifest official_test should read"
    )
    ids = ev.split_video_ids("official_test")
    assert len(ids) == 990


def test_val_confirm_ids_come_from_repo_root_manifest():
    root = json.loads((REPO / "split_manifest.json").read_text())
    assert ev.split_video_ids("val_confirm") == set(root["confirm_videos"])


def test_splits_are_disjoint():
    """No test video may appear in val_tune or val_confirm."""
    root = json.loads((REPO / "split_manifest.json").read_text())
    tuning = set(root["tune_videos"]) | set(root["confirm_videos"])
    assert not (ev.split_video_ids("official_test") & tuning)


# --- backward compatibility --------------------------------------------------

def test_default_split_is_val_confirm():
    args = ev.build_arg_parser().parse_args([])
    assert args.split == "val_confirm"
    assert args.resume is False
    assert args.limit_videos is None


def test_output_paths_unchanged_for_historical_two_arg_call():
    """Every pre---split call site passes two positional args. That must still
    resolve to the historical filenames, byte-for-byte."""
    csv_path, report_path = ev.output_paths_for_mode("none", "none")
    assert csv_path == REPO / "tuning" / "val_confirm_e2e_per_question.csv"
    assert report_path == REPO / "tuning" / "val_confirm_e2e_report.md"


def test_output_paths_for_reformulation_arms_unchanged():
    csv_path, report_path = ev.output_paths_for_mode("structured_v2", "directional")
    assert csv_path.name == "val_confirm_e2e_per_question_structured_v2_directional.csv"
    assert report_path.name == "val_confirm_e2e_report_structured_v2_directional.md"


def test_official_test_writes_to_its_own_directory():
    """A test-split run must never be able to land on a val_confirm artifact."""
    csv_path, report_path = ev.output_paths_for_mode("none", "none", "official_test")
    assert csv_path.parent.name == "official_test"
    assert csv_path != REPO / "tuning" / "val_confirm_e2e_per_question.csv"
    assert report_path.parent.name == "official_test"


def test_index_caches_are_disjoint_per_split():
    assert ev.index_cache_dir_for("val_confirm") != ev.index_cache_dir_for("official_test")


def test_directory_resolution_honours_patched_module_globals(monkeypatch, tmp_path):
    """The existing suite redirects a run into tmp_path by patching TUNING_DIR /
    INDEX_CACHE_DIR / PER_QUESTION_CSV. Resolution must read those globals at
    call time -- a dict frozen at import would silently write into the real
    repo during tests."""
    monkeypatch.setattr(ev, "TUNING_DIR", tmp_path)
    monkeypatch.setattr(ev, "INDEX_CACHE_DIR", tmp_path / "index_cache_val_confirm_e2e")
    monkeypatch.setattr(ev, "PER_QUESTION_CSV", tmp_path / "val_confirm_e2e_per_question.csv")
    monkeypatch.setattr(ev, "REPORT_PATH", tmp_path / "val_confirm_e2e_report.md")

    assert ev.out_dir_for("val_confirm") == tmp_path
    assert ev.out_dir_for("official_test") == tmp_path / "official_test"
    assert ev.index_cache_dir_for("val_confirm") == tmp_path / "index_cache_val_confirm_e2e"
    csv_path, report_path = ev.output_paths_for_mode("none", "none")
    assert csv_path == tmp_path / "val_confirm_e2e_per_question.csv"
    assert report_path == tmp_path / "val_confirm_e2e_report.md"


def test_val_confirm_uses_historical_internal_seams():
    """main() must keep calling the symbols the pre---split test suite patches:
    load_val_confirm_questions, ensure_indexes_e2e with two positional args,
    and smoke_test_backend. Guard against a future refactor silently bypassing
    them again."""
    src = inspect.getsource(ev.main)
    assert "load_val_confirm_questions()" in src
    assert "ensure_indexes_e2e(video_ids, cfg)" in src
    assert "smoke_test_backend(cfg)" in src


def test_load_val_confirm_questions_alias_still_exists():
    assert callable(ev.load_val_confirm_questions)
    sig = inspect.signature(ev.load_val_confirm_questions)
    assert len(sig.parameters) == 0


def test_ensure_indexes_signature_is_backward_compatible():
    """cache_dir must be optional so pre---split callers keep working."""
    sig = inspect.signature(ev.ensure_indexes_e2e)
    assert sig.parameters["cache_dir"].default is None
    assert sig.parameters["n_workers"].default == 8


# --- the loader produces the same questions as before ------------------------

def test_val_confirm_question_set_matches_committed_csv():
    """The strongest offline reproduction check available: the generalised
    loader must yield exactly the (video, qid) pairs, in exactly the order,
    that the committed val_confirm run scored."""
    if not COMMITTED_VAL_CONFIRM_CSV.exists():
        pytest.skip("committed val_confirm per-question CSV not present")
    with open(COMMITTED_VAL_CONFIRM_CSV, newline="") as f:
        committed = [(r["video"], r["qid"]) for r in csv.DictReader(f)]
    loaded = [(q["video"], q["qid"]) for q in ev.load_split_questions("val_confirm")]
    assert loaded == committed


def test_val_confirm_gold_matches_committed_csv():
    """Gold spans and gold answer indices must be identical too -- not just the
    question identities."""
    if not COMMITTED_VAL_CONFIRM_CSV.exists():
        pytest.skip("committed val_confirm per-question CSV not present")
    with open(COMMITTED_VAL_CONFIRM_CSV, newline="") as f:
        committed = {(r["video"], r["qid"]): r for r in csv.DictReader(f)}
    for q in ev.load_split_questions("val_confirm"):
        row = committed[(q["video"], q["qid"])]
        assert json.loads(row["gold_spans"]) == q["gold_spans"]
        assert int(row["gold_answer_idx"]) == q["gold_answer_idx"]


def test_official_test_loads_990_videos_and_5553_questions():
    questions = ev.load_split_questions("official_test")
    assert len({q["video"] for q in questions}) == 990
    assert len(questions) == 5553


def test_no_split_leakage_in_loaded_questions():
    test_vids = {q["video"] for q in ev.load_split_questions("official_test")}
    confirm_vids = {q["video"] for q in ev.load_split_questions("val_confirm")}
    assert not (test_vids & confirm_vids)


# --- serving contract constants ---------------------------------------------

def test_required_sampler_contract_is_the_determinism_contract():
    assert ev.REQUIRED_SAMPLER == {
        "temperature": 0, "top_k": 1, "top_p": 1.0, "seed": 42, "cache_prompt": False,
    }
    assert ev.REQUIRED_ALIAS == "granite4:micro"


def test_disk_thresholds_present():
    assert ev.DISK_ABORT_PREFLIGHT_GB == 5.0
    assert ev.DISK_ABORT_MIDRUN_GB == 2.0
    assert ev.DISK_RECHECK_EVERY == 50


def test_free_gb_returns_a_positive_number():
    assert ev.free_gb(REPO) > 0


# --- resume ledger -----------------------------------------------------------

def test_load_completed_qids_on_missing_file_is_empty():
    assert ev.load_completed_qids(REPO / "does_not_exist_xyz.csv") == set()


def test_load_completed_qids_reads_video_qid_pairs(tmp_path):
    p = tmp_path / "partial.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ev.PER_Q_FIELDNAMES)
        w.writeheader()
        for vid, qid in [("111", "0"), ("111", "3"), ("222", "1")]:
            w.writerow({k: "" for k in ev.PER_Q_FIELDNAMES} | {"video": vid, "qid": qid})
    assert ev.load_completed_qids(p) == {("111", "0"), ("111", "3"), ("222", "1")}


def test_load_completed_qids_tolerates_truncated_final_line(tmp_path):
    """A kill mid-write can leave a partial last row. Resume must still read
    every complete row rather than throwing."""
    p = tmp_path / "partial.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ev.PER_Q_FIELDNAMES)
        w.writeheader()
        w.writerow({k: "" for k in ev.PER_Q_FIELDNAMES} | {"video": "111", "qid": "0"})
    with open(p, "a") as f:
        f.write("222,1,TN,truncated")
    got = ev.load_completed_qids(p)
    assert ("111", "0") in got
