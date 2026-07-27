"""X1 regression test: as_context_text() must emit frame blocks in ascending
timestamp order, not retrieval-rank (insertion) order.

self._frames is populated by iris/query.py::wrapper_populate_cache walking
retrieved_frames in retrieval-rank order, and dict insertion order is
preserved in Python -- so before this fix, admitting frames out of
chronological order made as_context_text() emit them out of order too. This
matters for temporal questions ("what happened after X"), where the answerer
needs the frames presented in time order regardless of which one CLIP ranked
highest.

set_facts (the NLI fact pool) is intentionally unordered (a dict keyed by
caption text) and must NOT be touched by this fix.
"""
from __future__ import annotations

from iris.cached_frame import CachedFrame
from iris.frame_motion_descriptor import FrameMotionDescriptor
from iris.iris_config import IRISConfig
from iris.l1_elysium import L1ElysiumCache


def make_frame(frame_idx: int, timestamp_sec: float, action_score: float = 0.5) -> CachedFrame:
    motion = FrameMotionDescriptor(
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        motion_entropy=0.0,
        hessian_max_eigenvalue=0.0,
    )
    return CachedFrame(
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        action_score=action_score,
        persistence_value=0.5,
        is_peak=action_score >= 0.5,
        motion=motion,
    )


def test_as_context_text_orders_frames_by_timestamp_ascending():
    cache = L1ElysiumCache(IRISConfig())
    # Admit in deliberately non-chronological (retrieval-rank) order:
    # frame_idx=5 (t=50s) ranked highest by retrieval, then frame_idx=1 (t=10s),
    # then frame_idx=3 (t=30s).
    cache.admit(make_frame(5, timestamp_sec=50.0))
    cache.admit(make_frame(1, timestamp_sec=10.0))
    cache.admit(make_frame(3, timestamp_sec=30.0))

    context = cache.as_context_text()
    blocks = context.split("\n\n---\n\n")
    assert len(blocks) == 3

    # Chronological order by timestamp: frame 1 (10s), frame 3 (30s), frame 5 (50s).
    assert "Frame 1" in blocks[0] and "Timestamp: 10.0s" in blocks[0]
    assert "Frame 3" in blocks[1] and "Timestamp: 30.0s" in blocks[1]
    assert "Frame 5" in blocks[2] and "Timestamp: 50.0s" in blocks[2]


def test_as_context_text_uses_frame_idx_as_stable_tiebreak():
    cache = L1ElysiumCache(IRISConfig())
    # Two frames with the identical timestamp: admission (retrieval-rank) order
    # puts frame_idx=9 first, but frame_idx=4 must still sort first on tie.
    cache.admit(make_frame(9, timestamp_sec=20.0))
    cache.admit(make_frame(4, timestamp_sec=20.0))

    context = cache.as_context_text()
    blocks = context.split("\n\n---\n\n")
    assert "Frame 4" in blocks[0]
    assert "Frame 9" in blocks[1]


def test_set_facts_ordering_is_unchanged_by_this_fix():
    """set_facts must remain a dict keyed by caption text -- X1 only touches
    as_context_text(), not set_facts."""
    cache = L1ElysiumCache(IRISConfig())
    f1 = make_frame(5, timestamp_sec=50.0)
    f1.caption = {"semantic_caption": "a caption for the late frame"}
    f2 = make_frame(1, timestamp_sec=10.0)
    f2.caption = {"semantic_caption": "a caption for the early frame"}
    cache.admit(f1)
    cache.admit(f2)

    facts = cache.set_facts
    assert set(facts.keys()) == {
        "Frame 5 at 50.00s: a caption for the late frame.",
        "Frame 1 at 10.00s: a caption for the early frame.",
    }
