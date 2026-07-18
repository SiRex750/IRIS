"""Production-path tests for eval/span.py.

Per the P0-03 lesson: a shared constructor is only a fix if every call site
actually imports it, rather than each holding its own copy. These tests
import predict_span through each call site's own module reference and assert
identity + agreement, which is exactly what would have caught the four
divergent min->max copies (DECISIONS.md 2026-07-17 §3 / 2026-07-17-later §A6).
"""
import pytest

import eval.span as span_mod
import eval.grounding_scorer as grounding_scorer_mod
import scripts.pillar2_grounded_qa as pillar2_mod
import scripts.eval_grounding_arms as eval_arms_mod

_FRAMES = [
    {"timestamp": 1.0, "last_retrieval_score": 0.2},
    {"timestamp": 5.0, "last_retrieval_score": 0.9},
    {"timestamp": 9.0, "last_retrieval_score": 0.1},
]


def test_every_call_site_imports_the_same_function():
    assert grounding_scorer_mod.predict_span is span_mod.predict_span
    assert pillar2_mod.predict_span is span_mod.predict_span
    assert eval_arms_mod.predict_span is span_mod.predict_span


def test_call_sites_agree_on_identical_input_ppr_peak():
    kwargs = dict(mode="ppr_peak", half_width=2.0, duration=10.0)
    r1 = grounding_scorer_mod.predict_span(_FRAMES, **kwargs)
    r2 = pillar2_mod.predict_span(_FRAMES, **kwargs)
    r3 = eval_arms_mod.predict_span(_FRAMES, **kwargs)
    assert r1 == r2 == r3 == (3.0, 7.0)


def test_call_sites_agree_on_identical_input_minmax():
    r1 = grounding_scorer_mod.predict_span(_FRAMES, mode="minmax")
    r2 = pillar2_mod.predict_span(_FRAMES, mode="minmax")
    r3 = eval_arms_mod.predict_span(_FRAMES, mode="minmax")
    assert r1 == r2 == r3 == (1.0, 9.0)


def test_empty_input_returns_none():
    assert span_mod.predict_span([], mode="ppr_peak", half_width=2.0) is None
    assert span_mod.predict_span([], mode="minmax") is None


def test_ppr_peak_without_half_width_raises():
    with pytest.raises(ValueError):
        span_mod.predict_span(_FRAMES, mode="ppr_peak")


def test_minmax_reproduces_legacy_min_max_exactly():
    frames = [
        {"timestamp": 4.0},
        {"timestamp": 1.0},
        {"timestamp": 7.5},
        {"timestamp": 2.0},
    ]
    ts_list = [f["timestamp"] for f in frames]
    assert span_mod.predict_span(frames, mode="minmax") == (min(ts_list), max(ts_list))


def test_ppr_peak_picks_highest_scoring_frame_and_clips_to_duration():
    frames = [
        {"timestamp": 9.5, "last_retrieval_score": 0.99},
        {"timestamp": 1.0, "last_retrieval_score": 0.1},
    ]
    span = span_mod.predict_span(frames, mode="ppr_peak", half_width=2.0, duration=10.0)
    assert span == (7.5, 10.0)


def test_ppr_peak_clips_lower_bound_to_zero():
    frames = [{"timestamp": 0.5, "last_retrieval_score": 1.0}]
    span = span_mod.predict_span(frames, mode="ppr_peak", half_width=2.0, duration=10.0)
    assert span == (0.0, 2.5)
