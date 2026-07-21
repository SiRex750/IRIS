"""Tests for the QA debug trace mode (iris.debug_trace + its wiring in
iris.query.query()). Two properties are load-bearing and tested explicitly:

1. debug_trace=False (default) must produce IDENTICAL answers/outputs to
   before this feature existed, and must not even construct a DebugTrace.
2. debug_trace=True must produce the SAME answer as debug_trace=False for
   the same inputs -- instrumentation must never change pipeline behavior.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

import iris.query as q
from iris.iris_config import IRISConfig
from iris.types import IRISIndex, FrameRecord


# Local copy of tests/test_query.py's hermetic fixture (cross-file test
# imports aren't reliable across this repo's pytest configuration).
class _FakeNode:
    def __init__(self, i):
        self.frame_idx = i
        self.timestamp = float(i)
        self.luma_diff_energy = 0.1 * i
        self.action_score = 0.5
        self.persistence_value = 0.2
        self.pagerank_score = 0.3
        self.last_retrieval_score = 0.0
        self.retrieval_contributions = {}


class _FakeGraph:
    def retrieve(self, emb, query_action_score, top_k):
        return [_FakeNode(0), _FakeNode(1)]

    def retrieve_ppr(self, emb, top_k, damping=0.5, lambda_=0.5):
        return [_FakeNode(0), _FakeNode(1)]


def _index_with_fake_graph():
    frames = [FrameRecord(
        frame_idx=i, timestamp=float(i), luma_diff_energy=0.1 * i,
        luma_entropy=0.0, motion_magnitude=0.0, action_score=0.5,
        persistence_value=0.2, is_peak=(i == 0),
        caption={"semantic_caption": f"scene {i}"},
        clip_embedding=np.ones(512, dtype=np.float32) / np.sqrt(512), pagerank_score=0.3,
        scene_id=0,
    ) for i in range(2)]
    idx = IRISIndex(
        video_path="v.mp4", frames=frames, index_action_score=0.5,
        stats={"total": 10, "skipped": 6}, frames_processed=4, peak_count=1,
        skipped_frames_ratio=0.6, storage_reduction_factor=2.5,
        config_snapshot={"graph_mode": "scene_sparse"},
    )
    idx._scene_centroids = {0: np.ones(512, dtype=np.float32) / np.sqrt(512)}
    idx._graph = _FakeGraph()
    return idx


def _patched_query(monkeypatch, answer_text="A figure moves across the frame slowly."):
    monkeypatch.setattr(q, "_embed_query", lambda question, config: np.ones(512, dtype=np.float32) / np.sqrt(512))
    monkeypatch.setattr(q.aria, "generate", lambda prompt, context, *args, **kwargs: answer_text)
    monkeypatch.setattr(q, "wrapper_cerberus_gate",
                        lambda claims, cache, score, config: (True, list(claims), [], [], False))


# ── 1. zero overhead / no behavior change when disabled ─────────────────────

def test_disabled_does_not_construct_debug_trace(monkeypatch, tmp_path):
    """When debug_trace=False (default), no debug_traces directory is created
    and the module-level `if getattr(config, "debug_trace", False):` guard in
    iris.query.query() means iris.debug_trace.DebugTrace.start() is never
    reached at all -- proven directly by patching it to raise if called."""
    _patched_query(monkeypatch)

    import iris.debug_trace as dt

    def _must_not_be_called(*a, **kw):
        raise AssertionError("DebugTrace.start() must not be called when debug_trace=False")

    monkeypatch.setattr(dt.DebugTrace, "start", staticmethod(_must_not_be_called))

    idx = _index_with_fake_graph()
    cfg = IRISConfig(cerberus_mode="legacy")
    assert cfg.debug_trace is False
    cfg.debug_trace_dir = str(tmp_path / "debug_traces")
    res = q.query("what happens?", idx, config=cfg)  # would raise AssertionError if the guard were missing
    assert res["answer"]  # normal response produced
    assert not (tmp_path / "debug_traces").exists()


def test_default_config_has_debug_trace_off():
    assert IRISConfig().debug_trace is False


# ── 2. identical answers on/off ──────────────────────────────────────────────

def test_answers_identical_with_trace_enabled(monkeypatch, tmp_path):
    _patched_query(monkeypatch)

    idx_off = _index_with_fake_graph()
    cfg_off = IRISConfig(cerberus_mode="legacy")
    cfg_off.debug_trace = False
    res_off = q.query("what happens?", idx_off, config=cfg_off)

    idx_on = _index_with_fake_graph()
    cfg_on = IRISConfig(cerberus_mode="legacy")
    cfg_on.debug_trace = True
    cfg_on.debug_trace_dir = str(tmp_path / "debug_traces")
    res_on = q.query("what happens?", idx_on, config=cfg_on,
                      debug_context={"video_id": "v", "question_id": "0"})

    # Compare every field except "timings" (wall-clock, expected to differ run-to-run).
    for key in res_off:
        if key == "timings":
            continue
        assert res_on[key] == res_off[key], f"field {key!r} differs with debug_trace enabled"


def test_debug_trace_failure_does_not_break_query(monkeypatch, tmp_path):
    """If DebugTrace construction/finalization raises, the real query() result
    must still be returned -- diagnostic tooling must never break the pipeline
    it observes."""
    _patched_query(monkeypatch)

    import iris.debug_trace as dt
    monkeypatch.setattr(dt.DebugTrace, "finalize", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    idx = _index_with_fake_graph()
    cfg = IRISConfig(cerberus_mode="legacy")
    cfg.debug_trace = True
    cfg.debug_trace_dir = str(tmp_path / "debug_traces")
    res = q.query("what happens?", idx, config=cfg, debug_context={"video_id": "v", "question_id": "0"})
    assert res["answer"]  # did not raise, real result returned


# ── 3. trace files are written with the expected structure ─────────────────

def test_debug_trace_writes_expected_files(monkeypatch, tmp_path):
    _patched_query(monkeypatch, answer_text="A figure moves across the frame slowly.")

    idx = _index_with_fake_graph()
    cfg = IRISConfig(cerberus_mode="legacy")
    cfg.debug_trace = True
    cfg.debug_trace_dir = str(tmp_path / "debug_traces")
    q.query("what happens in this clip?", idx, config=cfg, debug_context={
        "video_id": "myvid", "question_id": "3", "question_type": "TN", "split": "val",
        "ground_truth_answer": "A figure moves across the frame slowly.",
        "ground_truth_options": ["a", "b", "A figure moves across the frame slowly.", "d"],
    })

    out_dirs = list((tmp_path / "debug_traces").iterdir())
    assert len(out_dirs) == 1
    out_dir = out_dirs[0]
    assert out_dir.name == "myvid__q3"

    assert (out_dir / "trace.json").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "granite_prompt.txt").exists()
    assert (out_dir / "frames").is_dir()

    trace = json.loads((out_dir / "trace.json").read_text(encoding="utf-8"))
    for top_key in ["query", "ground_truth", "retrieval", "frames", "captions",
                     "captioner", "granite", "verification", "final_answer", "timings",
                     "pipeline_status", "root_cause"]:
        assert top_key in trace, f"missing top-level key {top_key!r}"

    assert trace["query"]["video_id"] == "myvid"
    assert trace["query"]["question_id"] == "3"
    assert trace["query"]["question_type"] == "TN"
    assert trace["query"]["split"] == "val"
    assert trace["ground_truth"]["answer"] == "A figure moves across the frame slowly."
    assert trace["final_answer"]["match"] is True
    assert trace["final_answer"]["evaluation"]["correct"] is True
    assert trace["final_answer"]["evaluation"]["comparison_method"] == "exact_match"

    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    for section in ["# Query", "# Ground Truth", "# Retrieved Scenes", "# Retrieved Frames",
                     "# Generated Captions", "# Granite Prompt", "# Granite Raw Output",
                     "# Cerberus Verification", "# Final Prediction", "# Pipeline Timings",
                     "# Root Cause Summary"]:
        assert section in summary, f"missing section {section!r} in summary.md"

    prompt_txt = (out_dir / "granite_prompt.txt").read_text(encoding="utf-8")
    assert "SYSTEM PROMPT" in prompt_txt
    assert "USER PROMPT" in prompt_txt


def test_debug_trace_no_ground_truth_leaves_match_null(monkeypatch, tmp_path):
    _patched_query(monkeypatch)
    idx = _index_with_fake_graph()
    cfg = IRISConfig(cerberus_mode="legacy")
    cfg.debug_trace = True
    cfg.debug_trace_dir = str(tmp_path / "debug_traces")
    q.query("manual ad-hoc question", idx, config=cfg)  # no debug_context at all

    out_dirs = list((tmp_path / "debug_traces").iterdir())
    trace = json.loads((out_dirs[0] / "trace.json").read_text(encoding="utf-8"))
    assert trace["ground_truth"]["answer"] is None
    assert trace["final_answer"]["match"] is None
    assert trace["final_answer"]["evaluation"]["correct"] is None


# ── 4. make_query_id ─────────────────────────────────────────────────────────

def test_make_query_id_deterministic_with_question_id():
    from iris.debug_trace import make_query_id
    assert make_query_id("12345", "7") == make_query_id("12345", "7")
    assert make_query_id("12345", "7") == "12345__q7"


def test_make_query_id_sanitizes_unsafe_chars():
    from iris.debug_trace import make_query_id
    qid = make_query_id("http://x/12345.mp4", "a b/c")
    assert "/" not in qid and ":" not in qid


# ── 5. root-cause heuristic ──────────────────────────────────────────────────

def _blank_trace(**overrides):
    from iris.debug_trace import DebugTrace
    t = DebugTrace(query_id="x", out_dir=Path("."), video_id="v", question="q")
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


def test_diagnose_correct_answer_no_investigation_needed():
    from iris.debug_trace import diagnose
    t = _blank_trace(final_answer={"match": True, "verified_answer": "x", "ground_truth_answer": "x"})
    status, paragraph = diagnose(t)
    assert "no root-cause investigation needed" in paragraph.lower()


def test_diagnose_flags_empty_retrieval_as_earliest_suspect():
    from iris.debug_trace import diagnose
    t = _blank_trace(
        retrieval={"retrieved_frame_idxs": [], "margin": None, "tau": None},
        final_answer={"match": False, "verified_answer": "x", "ground_truth_answer": "y"},
    )
    status, paragraph = diagnose(t)
    assert status["Retrieval"] == "Suspect"
    assert "Retrieval" in paragraph


def test_diagnose_flags_empty_caption_as_suspect():
    from iris.debug_trace import diagnose
    t = _blank_trace(
        retrieval={"retrieved_frame_idxs": [1, 2], "margin": 0.5, "tau": 0.05},
        frames=[{"frame_id": 1, "timestamp": 0.1, "scene_id": 0, "image_path": "frames/frame_001.jpg"},
                {"frame_id": 2, "timestamp": 0.2, "scene_id": 0, "image_path": "frames/frame_002.jpg"}],
        captions=[{"frame_idx": 1, "timestamp": 0.1, "caption": "", "caption_length": 0, "caption_generation_time": 0.1},
                  {"frame_idx": 2, "timestamp": 0.2, "caption": "a scene", "caption_length": 7, "caption_generation_time": 0.1}],
        final_answer={"match": False, "verified_answer": "x", "ground_truth_answer": "y"},
    )
    status, paragraph = diagnose(t)
    assert status["Caption Quality"] == "Suspect"


def test_diagnose_flags_rejected_claims_as_cerberus_suspect():
    from iris.debug_trace import diagnose
    t = _blank_trace(
        retrieval={"retrieved_frame_idxs": [1], "margin": 0.5, "tau": 0.05},
        granite={"raw_generation": "The answer is X.", "finish_reason": "stop"},
        verification={
            "verification_response": {"verified_claims": [], "rejected_claims": ["The answer is X."], "unverifiable_claims": []},
            "verification_decision": "not_verified", "verified": False,
            "reason_for_rejection": "1 claim(s) contradicted by retrieved captions",
        },
        final_answer={"match": False, "verified_answer": "Insufficient verified evidence to answer this question.", "ground_truth_answer": "X"},
    )
    status, paragraph = diagnose(t)
    assert status["Cerberus Verification"] == "Suspect"


def test_diagnose_no_ground_truth_reports_unscored():
    from iris.debug_trace import diagnose
    t = _blank_trace(final_answer={"match": None, "verified_answer": "x", "ground_truth_answer": None})
    status, paragraph = diagnose(t)
    assert "no ground truth" in paragraph.lower()


def test_diagnose_insufficient_evidence_when_all_correct_but_answer_wrong():
    from iris.debug_trace import diagnose
    t = _blank_trace(
        retrieval={"retrieved_frame_idxs": [1], "margin": 0.9, "tau": 0.05},
        frames=[{"frame_id": 1, "timestamp": 0.1, "scene_id": 0, "image_path": "frames/frame_001.jpg"}],
        captions=[{"frame_idx": 1, "timestamp": 0.1, "caption": "a detailed real caption here", "caption_length": 30, "caption_generation_time": 0.1}],
        granite={"raw_generation": "The answer is X.", "finish_reason": "stop", "user_prompt": "q", "retrieved_captions_context": "a detailed real caption here"},
        verification={
            "verification_response": {"verified_claims": ["The answer is X."], "rejected_claims": [], "unverifiable_claims": []},
            "verification_decision": "verified", "verified": True,
        },
        final_answer={"match": False, "verified_answer": "The answer is X.", "ground_truth_answer": "Y"},
    )
    status, paragraph = diagnose(t)
    assert status["Final Prediction"] == "Incorrect"
    assert "reasoning error" in paragraph.lower() or "manual inspection" in paragraph.lower()
