"""C0 + H2 integration tests for scripts/val_confirm_e2e_eval.py::main().

C0: --caption-dump / --caption-load must round-trip captions and must not
change default (neither-flag) behaviour. A miss under --caption-load must
raise loudly rather than silently falling through to live captioning.

H2: --repeat N must run the answer stage N times over frozen retrieval/spans/
captions (via --caption-load), report Acc@QA/Acc@GQA as mean/min/max/flip
counts, and report grounding metrics (mIoP/mIoU/IoP@0.5/IoU@0.5) once, not N
times. Default (--repeat 1, omitted) must be unaffected.

Every external dependency (video decode, real captioner, real answerer,
index ingest) is mocked -- these tests exercise the harness's own wiring
logic, not iris/'s production captioning/retrieval/answering code (covered
elsewhere).
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
    {
        "video": "vidA", "qid": "2", "question": "what happens next?", "type": "TC",
        "choices": ["a", "b", "c", "d", "e"],
        "gold_answer_idx": 2, "gold_spans": [[1.0, 4.0]], "duration": 10.0,
    },
]


@pytest.fixture
def patched(tmp_path, monkeypatch):
    monkeypatch.setattr(vce, "TUNING_DIR", tmp_path)
    monkeypatch.setattr(vce, "INDEX_CACHE_DIR", tmp_path / "index_cache_val_confirm_e2e")
    monkeypatch.setattr(vce, "PER_QUESTION_CSV", tmp_path / "val_confirm_e2e_per_question.csv")
    monkeypatch.setattr(vce, "REPORT_PATH", tmp_path / "val_confirm_e2e_report.md")

    monkeypatch.setattr(vce, "load_frozen_state", lambda: {"frozen": FROZEN})
    monkeypatch.setattr(vce, "load_val_confirm_questions", lambda: [dict(q) for q in QUESTIONS])
    monkeypatch.setattr(vce, "ensure_indexes_e2e", lambda video_ids, cfg: ({v: f"{v}.npz" for v in video_ids}, 0, len(video_ids)))
    monkeypatch.setattr(vce, "smoke_test_backend", lambda cfg: "ANSWER: A\nREASON: ok.")
    monkeypatch.setattr(
        vce, "capture_answerer_provenance",
        lambda **kwargs: {"endpoint": kwargs["endpoint"], "note": "mocked"},
    )

    frames_by_video = {
        "vidA": FakeIndex([FakeFrameRecord(0, 1.0), FakeFrameRecord(1, 2.5)]),
    }
    monkeypatch.setattr(vce.iris_ingest, "load_index", lambda path: frames_by_video["vidA"])

    def fake_retrieve(question, index, cfg, type_code=None, family=None):
        retrieved = [{"frame_idx": fr.frame_idx, "timestamp": fr.timestamp} for fr in index.frames]
        return retrieved, FakePlan(), {"num_ppr_calls": 1}

    monkeypatch.setattr(vce, "retrieve_for_question", fake_retrieve)
    monkeypatch.setattr(vce.iris_query, "_call_embed_query", lambda question, cfg: ([1.0, 0.0, 0.0], None))

    def _fake_ensure_captions(index, retrieved_frames, *a, **kw):
        frame_map = {fr.frame_idx: fr for fr in index.frames}
        for f in retrieved_frames:
            fr = frame_map[f["frame_idx"]]
            fr.caption = {"semantic_caption": f"live caption {f['frame_idx']}"}
            f["caption"] = fr.caption
        return 0

    ensure_captions_mock = MagicMock(side_effect=_fake_ensure_captions)
    monkeypatch.setattr(vce.iris_query, "_ensure_captions", ensure_captions_mock)

    answer_mock = MagicMock(return_value="ANSWER: B\nREASON: because.")
    monkeypatch.setattr(vce.aria, "generate", answer_mock)

    return {"ensure_captions_mock": ensure_captions_mock, "answer_mock": answer_mock, "tmp_path": tmp_path}


def test_default_behaviour_unchanged_calls_live_captioning(patched):
    """(a) With neither --caption-dump nor --caption-load, live captioning
    must still run exactly as before."""
    vce.main([])
    assert patched["ensure_captions_mock"].call_count == 2  # once per question
    out_csv = patched["tmp_path"] / "val_confirm_e2e_per_question.csv"
    assert out_csv.exists()
    import csv as csv_mod
    with open(out_csv, newline="") as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) == 2
    assert set(rows[0].keys()) == set(vce.PER_Q_FIELDNAMES)


def test_caption_dump_then_load_round_trips_identical_context(patched, tmp_path):
    """(b) dump-then-load round-trips to identical context text."""
    dump_path = tmp_path / "captions.json"
    vce.main(["--caption-dump", str(dump_path)])
    assert patched["ensure_captions_mock"].call_count == 2
    assert dump_path.exists()

    dump_contents = json.loads(dump_path.read_text())
    assert dump_contents == {
        "vidA": {
            "1": {"0": "live caption 0", "1": "live caption 1"},
            "2": {"0": "live caption 0", "1": "live caption 1"},
        }
    }

    # Reset caption state and re-run against --caption-load: no live
    # captioning should occur, and the answerer must see the same context.
    for fr in vce.iris_ingest.load_index("vidA.npz").frames:
        fr.caption = None
    patched["ensure_captions_mock"].reset_mock()
    patched["answer_mock"].reset_mock()
    vce.PER_QUESTION_CSV.unlink()
    import shutil
    shutil.rmtree(vce.INDEX_CACHE_DIR)

    vce.main(["--caption-load", str(dump_path), "--query-mode", "none"])
    assert patched["ensure_captions_mock"].call_count == 0  # captioning skipped entirely
    contexts = [call.kwargs["context"] for call in patched["answer_mock"].call_args_list]
    assert all("live caption 0" in c and "live caption 1" in c for c in contexts)


def test_caption_load_miss_raises_loudly(patched, tmp_path):
    """(c) a deliberate miss under --caption-load raises rather than falling
    through to live captioning."""
    incomplete_dump = tmp_path / "incomplete_captions.json"
    incomplete_dump.write_text(json.dumps({"vidA": {"1": {"0": "only frame 0 captioned"}}}))

    with pytest.raises(KeyError):
        vce.main(["--caption-load", str(incomplete_dump)])
    # Live captioning must NOT have been used as a silent fallback.
    assert patched["ensure_captions_mock"].call_count == 0


def test_repeat_requires_caption_load(patched):
    with pytest.raises(SystemExit):
        vce.main(["--repeat", "3"])


def test_repeat_reports_mean_min_max_and_flip_counts_and_grounding_once(patched, tmp_path, capsys):
    dump_path = tmp_path / "captions.json"
    vce.main(["--caption-dump", str(dump_path)])

    answers = iter([
        "ANSWER: A\nREASON: r1.", "ANSWER: B\nREASON: r2.",   # repeat 1
        "ANSWER: A\nREASON: r1.", "ANSWER: A\nREASON: r2.",   # repeat 2 (qid 2 flips)
        "ANSWER: A\nREASON: r1.", "ANSWER: A\nREASON: r2.",   # repeat 3
    ])
    patched["answer_mock"].side_effect = lambda *a, **kw: next(answers)
    patched["ensure_captions_mock"].reset_mock()
    vce.PER_QUESTION_CSV.unlink()
    import shutil
    shutil.rmtree(vce.INDEX_CACHE_DIR)

    vce.main(["--caption-load", str(dump_path), "--repeat", "3"])

    assert patched["ensure_captions_mock"].call_count == 0

    for rep in (1, 2, 3):
        rep_csv = tmp_path / f"val_confirm_e2e_per_question_repeat{rep}.csv"
        assert rep_csv.exists()

    summary_path = tmp_path / "val_confirm_e2e_repeat_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["n_repeats"] == 3
    assert set(summary["Acc@QA"].keys()) >= {"mean", "min", "max", "per_repeat", "n_flipped_questions"}
    assert "grounding_metrics_reported_once" in summary
    assert set(summary["grounding_metrics_reported_once"].keys()) == {"mIoP", "mIoU", "IoP@0.5", "IoU@0.5"}

    captured = capsys.readouterr()
    metrics_line = [l for l in captured.out.splitlines() if l.startswith("VAL_CONFIRM_E2E_METRICS_JSON=")][-1]
    metrics = json.loads(metrics_line.split("=", 1)[1])
    assert metrics.get("mIoP") is not None
    assert "Acc@QA" not in metrics  # point-estimate key must not appear alongside repeat_summary
    # mIoP must not appear once per repeat -- exactly one value under repeat_summary too.
    assert metrics["repeat_summary"]["grounding_metrics_reported_once"]["mIoP"] == metrics["mIoP"]
