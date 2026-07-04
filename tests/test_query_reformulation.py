import numpy as np

from iris.query_reformulation import (
    expand_temporal_neighbors,
    fuse_ranked_results,
    parse_mc_answer,
    reformulate_query,
)
from iris.types import FrameRecord, IRISIndex


def test_parse_mc_answer_requires_explicit_marker():
    assert parse_mc_answer("A boy is stretching his arm. I choose B.") is None
    assert parse_mc_answer("ANSWER: B\nREASON: visible evidence") == 1
    assert parse_mc_answer("final answer: E") == 4
    assert parse_mc_answer("ANSWER: option D") == 3
    assert parse_mc_answer("ANSWER: (C)") == 2


def test_reformulate_query_is_visual_and_temporal_safe():
    plan = reformulate_query(
        "why did the boy stretch his arm in the middle of the video",
        family="C",
    )

    assert plan.original_query.startswith("why did")
    assert plan.needs_temporal_expansion is True
    assert plan.temporal_relation == "middle"
    assert len(plan.retrieval_queries) >= 3
    assert any("boy stretch" in q or "boy stretching" in q for q in plan.retrieval_queries)

    joined = " ".join(plan.retrieval_queries).lower()
    assert "to catch" not in joined
    assert "to grab" not in joined


def test_fuse_ranked_results_dedupes_and_rewards_repeated_frames():
    a = [
        {"frame_idx": 10, "retrieval_contributions": {"seed": "a"}},
        {"frame_idx": 20},
    ]
    b = [
        {"frame_idx": 20, "retrieval_contributions": {"seed": "b"}},
        {"frame_idx": 30},
    ]

    fused = fuse_ranked_results([a, b], top_k=3, rrf_k=0)

    assert [frame["frame_idx"] for frame in fused] == [20, 10, 30]
    assert fused[0]["retrieval_contributions"]["query_reformulation_hits"] == 2


def test_expand_temporal_neighbors_uses_indexed_frame_radius():
    idx = _fake_index(5)
    retrieved = [{
        "frame_idx": 20,
        "timestamp": 2.0,
        "luma_diff_energy": 0.2,
        "action_score": 0.5,
        "persistence_value": 0.1,
        "is_peak": True,
        "clip_embedding": np.zeros(512, dtype=np.float32),
        "luma_entropy": 0.0,
        "caption": None,
        "pagerank_score": 0.0,
        "last_retrieval_score": 0.9,
        "retrieval_contributions": {"retrieved": True},
    }]

    expanded = expand_temporal_neighbors(idx, retrieved, radius=1)

    assert [frame["frame_idx"] for frame in expanded] == [10, 20, 30]
    assert expanded[1]["retrieval_contributions"]["temporal_expansion"] is False
    assert expanded[0]["retrieval_contributions"]["temporal_expansion"] is True


def _fake_index(n: int) -> IRISIndex:
    frames = [
        FrameRecord(
            frame_idx=i * 10,
            timestamp=float(i),
            luma_diff_energy=0.1 * i,
            luma_entropy=0.0,
            motion_magnitude=0.0,
            action_score=0.2,
            persistence_value=0.1,
            is_peak=i == 2,
            clip_embedding=np.zeros(512, dtype=np.float32),
            pagerank_score=0.0,
        )
        for i in range(n)
    ]
    return IRISIndex(
        video_path="v.mp4",
        frames=frames,
        index_action_score=0.2,
        stats={},
        frames_processed=n,
        peak_count=1,
        skipped_frames_ratio=0.0,
        storage_reduction_factor=1.0,
        config_snapshot={},
    )
