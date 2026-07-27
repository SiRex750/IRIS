"""C2 regression test: iris.aria.get_minicpm_truncation_stats() (previously
never called by any eval script -- the truncation rate was invisible in
layer3_outputs.csv and every downstream report) must be wired into
scripts/val_confirm_e2e_eval.py::main()'s final metrics output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import val_confirm_e2e_eval as vce  # noqa: E402
import iris.aria as aria  # noqa: E402


class FakeFrameRecord:
    def __init__(self, frame_idx, timestamp):
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.timestamp_sec = timestamp
        self.caption = None


class FakeIndex:
    def __init__(self, frames):
        self.frames = frames
        self.video_path = "fake.mp4"


class FakePlan:
    relation = None
    relation_source = None
    fallback_reason = None


FROZEN = {
    "retrieval_strategy": "hybrid", "ppr_lambda": 0.5, "ppr_damping": 0.5,
    "l2_retrieve_top_k": 4, "span_method": "D", "span_method_half_width_s": 2.2,
    "peak_distance": 5, "peak_prominence": 0.05, "packet_size_weight": 0.8,
    "motion_weight": 0.1, "luma_entropy_weight": 0.1, "persistence_threshold": 0.4,
    "max_prominence": 0.5,
}

QUESTIONS = [
    {
        "video": "vidA", "qid": "1", "question": "what happens?", "type": "TN",
        "choices": ["a", "b", "c", "d", "e"],
        "gold_answer_idx": 1, "gold_spans": [[0.0, 3.0]], "duration": 10.0,
    },
]


def test_final_metrics_include_minicpm_truncation_stats(tmp_path, monkeypatch, capsys):
    aria.reset_minicpm_truncation_stats()
    # Simulate two captioner calls, one of which truncated after retry --
    # exactly the signal get_minicpm_truncation_stats() reports.
    aria._MINICPM_TRUNCATION_STATS["total_calls"] = 2
    aria._MINICPM_TRUNCATION_STATS["truncated_first_attempt"] = 1
    aria._MINICPM_TRUNCATION_STATS["truncated_after_retry"] = 1

    monkeypatch.setattr(vce, "TUNING_DIR", tmp_path)
    monkeypatch.setattr(vce, "INDEX_CACHE_DIR", tmp_path / "index_cache_val_confirm_e2e")
    monkeypatch.setattr(vce, "PER_QUESTION_CSV", tmp_path / "val_confirm_e2e_per_question.csv")
    monkeypatch.setattr(vce, "REPORT_PATH", tmp_path / "val_confirm_e2e_report.md")
    monkeypatch.setattr(vce, "load_frozen_state", lambda: {"frozen": FROZEN})
    monkeypatch.setattr(vce, "load_val_confirm_questions", lambda: [dict(q) for q in QUESTIONS])
    monkeypatch.setattr(vce, "ensure_indexes_e2e", lambda video_ids, cfg: ({v: f"{v}.npz" for v in video_ids}, 0, len(video_ids)))
    monkeypatch.setattr(vce, "smoke_test_backend", lambda cfg: "ANSWER: A\nREASON: ok.")
    monkeypatch.setattr(vce, "capture_answerer_provenance", lambda **kwargs: {"note": "mocked"})

    fake_index = FakeIndex([FakeFrameRecord(0, 1.0)])
    monkeypatch.setattr(vce.iris_ingest, "load_index", lambda path: fake_index)

    def fake_retrieve(question, index, cfg, type_code=None, family=None):
        retrieved = [{"frame_idx": fr.frame_idx, "timestamp": fr.timestamp} for fr in index.frames]
        return retrieved, FakePlan(), {"num_ppr_calls": 1}

    monkeypatch.setattr(vce, "retrieve_for_question", fake_retrieve)
    monkeypatch.setattr(vce.iris_query, "_call_embed_query", lambda question, cfg: ([1.0, 0.0, 0.0], None))

    def _fake_ensure_captions(index, retrieved_frames, *a, **kw):
        for f in retrieved_frames:
            f["caption"] = {"semantic_caption": "a caption"}
        return 0

    monkeypatch.setattr(vce.iris_query, "_ensure_captions", _fake_ensure_captions)
    monkeypatch.setattr(vce.aria, "generate", MagicMock(return_value="ANSWER: A\nREASON: ok."))

    vce.main([])

    captured = capsys.readouterr()
    metrics_line = [l for l in captured.out.splitlines() if l.startswith("VAL_CONFIRM_E2E_METRICS_JSON=")][-1]
    metrics = json.loads(metrics_line.split("=", 1)[1])

    assert "minicpm_truncation_stats" in metrics
    stats = metrics["minicpm_truncation_stats"]
    assert stats["total_calls"] == 2
    assert stats["truncated_after_retry"] == 1
    assert stats["truncation_rate_after_retry"] == 0.5
