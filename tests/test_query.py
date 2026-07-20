import inspect

import numpy as np
import pytest

import iris.pipeline as p
import iris.query as q
from iris.types import IRISIndex, FrameRecord


# 1. Lifted wrappers are byte-for-byte verbatim copies from pipeline.py
def test_wrappers_verbatim():
    for name in ["wrapper_init_l1_cache", "wrapper_populate_cache", "wrapper_cerberus_gate"]:
        assert inspect.getsource(getattr(q, name)) == inspect.getsource(getattr(p, name)), name


# 2. Claim split is deterministic on a fixed input (pins the regex)
def test_split_claims_deterministic():
    raw = ("**Summary:** A figure walks across the room. "
           "It then sits down near a table.\n- short\n"
           "A second person enters from the left side.")
    claims = q._split_claims(raw)
    assert claims == [
        "A figure walks across the room.",
        "It then sits down near a table.",
        "short A second person enters from the left side.",
    ]


# 3. query() wiring is hermetic: fake graph + patched aria + patched cerberus
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


def test_query_wiring(monkeypatch):
    monkeypatch.setattr(q, "_embed_query", lambda question, config: np.ones(512, dtype=np.float32) / np.sqrt(512))
    monkeypatch.setattr(q.aria, "generate", lambda prompt, context, *args, **kwargs: "A figure moves across the frame slowly.")
    # Patch cerberus gate to a fixed verdict so no DeBERTa load
    monkeypatch.setattr(q, "wrapper_cerberus_gate",
                        lambda claims, cache, score, config: (True, list(claims), [], [], False))

    idx = _index_with_fake_graph()
    res = q.query("what happens?", idx)

    assert res["frames_processed"] == 4
    assert res["peak_count"] == 1
    assert abs(res["skipped_frames_ratio"] - 0.6) < 1e-9
    assert res["retrieved_frame_idxs"] == [0, 1]
    assert res["verified"] is True
    assert res["answer"] == "A figure moves across the frame slowly."
    assert set(res.keys()) >= {
        "answer", "raw_answer", "verified", "nli_mocked", "verified_claims",
        "rejected_claims", "unverifiable_claims", "frames_processed",
        "peak_count", "compression_ratio", "skipped_frames_ratio",
        "storage_reduction_factor", "timings",
    }
    assert res["predicted_choice_idx"] is None  # no choices passed


def test_query_wiring_with_choices(monkeypatch):
    """Item 4: choices reach the answerer prompt, and the model's free-text
    pick is mapped back to predicted_choice_idx -- gold_answer_idx is never
    accepted as a query() parameter at all (only choices, the options)."""
    captured_prompt = {}

    def fake_generate(prompt, context, *args, **kwargs):
        captured_prompt["prompt"] = prompt
        return "The woman is holding a spoon of ice cream, matching option B. blue shirt."

    monkeypatch.setattr(q, "_embed_query", lambda question, config: np.ones(512, dtype=np.float32) / np.sqrt(512))
    monkeypatch.setattr(q.aria, "generate", fake_generate)
    monkeypatch.setattr(q, "wrapper_cerberus_gate",
                        lambda claims, cache, score, config: (True, list(claims), [], [], False))

    idx = _index_with_fake_graph()
    choices = ["red shirt", "blue shirt", "green shirt"]
    res = q.query("what color is her shirt?", idx, choices=choices)

    assert "A. red shirt" in captured_prompt["prompt"]
    assert "B. blue shirt" in captured_prompt["prompt"]
    assert "C. green shirt" in captured_prompt["prompt"]
    assert res["predicted_choice_idx"] == 1  # matched "blue shirt" substring


# 4. PARITY GATE — old run_pipeline vs new ingest+query on the real clip.
#    aria.generate is patched to a FIXED answer in BOTH runs to remove LLM
#    nondeterminism; real Cerberus runs in both, so identical verdicts prove
#    L1 fact pools are identical. Skips if mov_bbb.mp4 is absent.
import os
VIDEO = os.environ.get("IRIS_TEST_VIDEO", "mov_bbb.mp4")


@pytest.mark.skipif(not os.path.exists(VIDEO), reason="test video not on disk")
def test_parity_old_vs_new(monkeypatch):
    from iris.ingest import ingest
    from iris import pipeline

    FIXED = "A figure moves across the frame. The scene appears to show motion near the center."
    monkeypatch.setattr(pipeline.aria, "generate", lambda prompt, context, *args, **kwargs: FIXED)
    monkeypatch.setattr(q.aria, "generate", lambda prompt, context, *args, **kwargs: FIXED)

    question = "What is happening in this video?"

    old = pipeline.run_pipeline(VIDEO, question)
    index = ingest(VIDEO)
    new = q.query(question, index)

    # deterministic structural parity
    assert new["frames_processed"] == old["frames_processed"]
    assert new["peak_count"] == old["peak_count"]
    assert abs(new["skipped_frames_ratio"] - old["skipped_frames_ratio"]) < 1e-9
    assert abs(new["storage_reduction_factor"] - old["storage_reduction_factor"]) < 1e-9
    # verification + answer parity (identical claims -> identical real-Cerberus verdict)
    assert new["verified"] == old["verified"]
    assert sorted(new["verified_claims"]) == sorted(old["verified_claims"])
    assert sorted(new["rejected_claims"]) == sorted(old["rejected_claims"])
    assert sorted(new["unverifiable_claims"]) == sorted(old["unverifiable_claims"])
    assert new["answer"] == old["answer"]


# 5. _build_focus_hint: pure helper, exact string contract
def test_build_focus_hint_no_question():
    assert q._build_focus_hint(None) is None
    assert q._build_focus_hint("") is None


def test_build_focus_hint_question_only():
    hint = q._build_focus_hint("why did the boy in red run?")
    assert hint == "Pay attention to: why did the boy in red run?"


def test_build_focus_hint_question_and_choices():
    hint = q._build_focus_hint("what color is the shirt?", ["red", "blue", "green"])
    assert hint == (
        "Pay attention to: what color is the shirt? "
        "Consider distinguishing details relevant to: red; blue; green."
    )


# 6. _ensure_captions threads the question through to the captioner's request
#    payload (item 3: captioner was previously blind to the question). Mocks
#    av.open's decode path and asserts get_semantic_and_clip_caption receives
#    the built focus_hint.
class _FakeStream:
    average_rate = 25.0
    time_base = 1.0 / 25.0


class _FakeVideoFrame:
    def __init__(self, pts):
        self.pts = pts

    def to_image(self):
        return "PIL_IMAGE_STUB"


class _FakeStreams:
    def __init__(self, stream):
        self.video = [stream]


class _FakeContainer:
    def __init__(self, stream):
        self.streams = _FakeStreams(stream)
        self._stream = stream

    def seek(self, target_pts, stream=None):
        pass

    def decode(self, stream):
        return [_FakeVideoFrame(pts=0)]

    def close(self):
        pass


def test_ensure_captions_threads_question_into_focus_hint(monkeypatch):
    import av
    from iris import _clip

    fr = FrameRecord(
        frame_idx=0, timestamp=0.0, luma_diff_energy=0.0, luma_entropy=0.0,
        motion_magnitude=0.0, action_score=0.5, persistence_value=0.2, is_peak=True,
        caption=None, clip_embedding=np.ones(512, dtype=np.float32) / np.sqrt(512),
    )
    idx = IRISIndex(
        video_path="v.mp4", frames=[fr], index_action_score=0.5,
        stats={"total": 1, "skipped": 0}, frames_processed=1, peak_count=1,
        skipped_frames_ratio=0.0, storage_reduction_factor=1.0,
        config_snapshot={"graph_mode": "flat"},
    )

    monkeypatch.setattr(av, "open", lambda path: _FakeContainer(_FakeStream()))

    captured = {}

    def fake_get_semantic_and_clip_caption(pil_img, frame, clip_emb, device, config=None, focus_hint=None):
        captured["focus_hint"] = focus_hint
        return {"clip_label": "stub", "semantic_caption": "stub caption"}

    monkeypatch.setattr(_clip, "get_semantic_and_clip_caption", fake_get_semantic_and_clip_caption)

    retrieved_frames = [{"frame_idx": 0, "timestamp": 0.0}]
    q._ensure_captions(idx, retrieved_frames, None,
                        question="why did the boy in red run?", choices=["fast", "slow"])

    assert captured["focus_hint"] == (
        "Pay attention to: why did the boy in red run? "
        "Consider distinguishing details relevant to: fast; slow."
    )


def test_ensure_captions_no_question_yields_no_focus_hint(monkeypatch):
    import av
    from iris import _clip

    fr = FrameRecord(
        frame_idx=0, timestamp=0.0, luma_diff_energy=0.0, luma_entropy=0.0,
        motion_magnitude=0.0, action_score=0.5, persistence_value=0.2, is_peak=True,
        caption=None, clip_embedding=np.ones(512, dtype=np.float32) / np.sqrt(512),
    )
    idx = IRISIndex(
        video_path="v.mp4", frames=[fr], index_action_score=0.5,
        stats={"total": 1, "skipped": 0}, frames_processed=1, peak_count=1,
        skipped_frames_ratio=0.0, storage_reduction_factor=1.0,
        config_snapshot={"graph_mode": "flat"},
    )
    monkeypatch.setattr(av, "open", lambda path: _FakeContainer(_FakeStream()))

    captured = {}

    def fake_get_semantic_and_clip_caption(pil_img, frame, clip_emb, device, config=None, focus_hint=None):
        captured["focus_hint"] = focus_hint
        return {"clip_label": "stub", "semantic_caption": "stub caption"}

    monkeypatch.setattr(_clip, "get_semantic_and_clip_caption", fake_get_semantic_and_clip_caption)

    q._ensure_captions(idx, [{"frame_idx": 0, "timestamp": 0.0}], None)  # no question/choices
    assert captured["focus_hint"] is None


class _CountingSeekContainer:
    """Like _FakeContainer, but counts seek() calls and serves a fixed
    synthetic frame sequence regardless of the seek target (real seek
    behavior isn't under test here -- only the *number* of seek calls and
    the resulting decode-loop frame count, which is)."""

    def __init__(self, stream, frame_pts_sequence):
        self.streams = _FakeStreams(stream)
        self._stream = stream
        self.seek_calls = 0
        self._frame_pts_sequence = frame_pts_sequence

    def seek(self, target_pts, stream=None):
        self.seek_calls += 1

    def decode(self, stream):
        return [_FakeVideoFrame(pts=p) for p in self._frame_pts_sequence]

    def close(self):
        pass


def _make_index_with_frames(timestamps: list[float]) -> IRISIndex:
    frames = [
        FrameRecord(
            frame_idx=i, timestamp=t, luma_diff_energy=0.0, luma_entropy=0.0,
            motion_magnitude=0.0, action_score=0.5, persistence_value=0.2, is_peak=False,
            caption=None, clip_embedding=np.ones(512, dtype=np.float32) / np.sqrt(512),
        )
        for i, t in enumerate(timestamps)
    ]
    return IRISIndex(
        video_path="v.mp4", frames=frames, index_action_score=0.5,
        stats={"total": len(frames), "skipped": 0}, frames_processed=len(frames), peak_count=1,
        skipped_frames_ratio=0.0, storage_reduction_factor=1.0,
        config_snapshot={"graph_mode": "flat"},
    )


def test_ensure_captions_single_seek_for_multiple_gop_targets(monkeypatch):
    """Item 7: multiple missing-caption targets sharing one decode pass must
    trigger exactly ONE container.seek() call, not one per target (the
    confirmed cause of a real smoke-test anomaly: 5 targets in one GOP cost
    574 decoded frames, matching the sum of independently re-decoding the
    same GOP prefix for each target)."""
    import av
    from iris import _clip

    # 3 survivor frames needing captions, timestamps 1.0/2.0/3.0s (fps=1 -> pts==seconds)
    idx = _make_index_with_frames([1.0, 2.0, 3.0])

    class _FpsOneStream:
        average_rate = 1.0
        time_base = 1.0

    fake_container = _CountingSeekContainer(_FpsOneStream(), frame_pts_sequence=list(range(0, 11)))
    monkeypatch.setattr(av, "open", lambda path: fake_container)
    monkeypatch.setattr(
        _clip, "get_semantic_and_clip_caption",
        lambda pil_img, frame, clip_emb, device, config=None, focus_hint=None:
            {"clip_label": "stub", "semantic_caption": "stub"},
    )

    retrieved_frames = [{"frame_idx": i, "timestamp": t} for i, t in enumerate([1.0, 2.0, 3.0])]
    frames_decoded = q._ensure_captions(idx, retrieved_frames, None)

    assert fake_container.seek_calls == 1, "expected exactly one seek() for the whole batch, not one per target"
    # Single continuous pass reaching pts=3 (the last target) decodes pts 0..3 = 4 frames,
    # not 1+2+3=6 (the sum an independent-per-target-reseek approach would produce).
    assert frames_decoded == 4
    assert all(fr.caption is not None for fr in idx.frames)


def test_ensure_captions_warns_on_high_decode_overhead(monkeypatch):
    """frames_decoded_for_captions > 3x survivor_count must emit a
    QUERY-CAPTION-001 RuntimeWarning (previously this overhead was invisible)."""
    import av
    from iris import _clip

    idx = _make_index_with_frames([9.0])  # 1 survivor, target far into a long synthetic GOP

    class _FpsOneStream:
        average_rate = 1.0
        time_base = 1.0

    fake_container = _CountingSeekContainer(_FpsOneStream(), frame_pts_sequence=list(range(0, 10)))
    monkeypatch.setattr(av, "open", lambda path: fake_container)
    monkeypatch.setattr(
        _clip, "get_semantic_and_clip_caption",
        lambda pil_img, frame, clip_emb, device, config=None, focus_hint=None:
            {"clip_label": "stub", "semantic_caption": "stub"},
    )

    retrieved_frames = [{"frame_idx": 0, "timestamp": 9.0}]
    with pytest.warns(RuntimeWarning, match="QUERY-CAPTION-001"):
        frames_decoded = q._ensure_captions(idx, retrieved_frames, None)

    assert frames_decoded == 10  # decodes pts 0..9 to reach the single target at t=9.0
    assert frames_decoded > 3 * len(idx.frames)


# 7. _build_answer_prompt / _match_predicted_choice: pure helpers
def test_build_answer_prompt_no_choices_returns_question_unchanged():
    assert q._build_answer_prompt("what happens?") == "what happens?"
    assert q._build_answer_prompt("what happens?", None) == "what happens?"
    assert q._build_answer_prompt("what happens?", []) == "what happens?"


def test_build_answer_prompt_with_choices_is_lettered_and_grounded():
    prompt = q._build_answer_prompt("what color is the shirt?", ["red", "blue", "green"])
    assert "what color is the shirt?" in prompt
    assert "A. red" in prompt
    assert "B. blue" in prompt
    assert "C. green" in prompt
    assert "justify" in prompt.lower()  # still requires claim-level grounding, not a bare pick


def test_match_predicted_choice_single_substring_match():
    assert q._match_predicted_choice("The shirt is blue.", ["red", "blue", "green"]) == 1


def test_match_predicted_choice_no_choices_or_no_answer():
    assert q._match_predicted_choice("The shirt is blue.", None) is None
    assert q._match_predicted_choice("The shirt is blue.", []) is None
    assert q._match_predicted_choice(None, ["red", "blue"]) is None
    assert q._match_predicted_choice("", ["red", "blue"]) is None


def test_match_predicted_choice_ambiguous_returns_none():
    # both "red" and "reddish" style substrings could match multiple options -> None, not a guess
    assert q._match_predicted_choice("It could be red or blue.", ["red", "blue", "green"]) is None


def test_match_predicted_choice_falls_back_to_letter():
    # no choice text appears verbatim in the answer -- only the letter distinguishes the pick
    assert q._match_predicted_choice(
        "Answer: B) confirmed by the visual evidence in frame 3",
        ["on the floor", "on the table", "on the chair"],
    ) == 1


def test_match_predicted_choice_out_of_range_letter_ignored():
    assert q._match_predicted_choice("Answer: Z) unclear", ["red", "blue"]) is None
