"""Tests for the adaptive span-construction sweep (Methods E/F) and the
guard rails the sweep harness relies on: (a) Methods A-D unchanged, (b)
Method E's tau=0/no-bounds degenerate case, (c) spans never exceed [0, D],
(d) the val_confirm exclusion assertion fires."""
import numpy as np
import pytest

from eval.metrics import (
    predicted_span_from_frames,
    predicted_span_from_frames_clustered,
    predicted_span_from_frames_scene,
    predicted_span_from_frames_peak,
    predicted_span_from_frames_profile,
    predicted_span_from_frames_weighted_spread,
    compute_similarity_profile,
    span_from_similarity_profile,
    weighted_std_topk,
    is_structurally_barred,
    assert_no_confirm_videos,
)


def _unit_embedding(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ── (a) Methods A-D byte-identical to current behaviour ────────────────────

def test_method_a_unchanged():
    assert predicted_span_from_frames([1.0, 3.0, 2.0]) == (1.0, 3.0)
    assert predicted_span_from_frames([]) == (0.0, 0.0)


def test_method_b_unchanged_reduces_to_a():
    frames = [
        {"timestamp": 0.0, "pagerank_score": 0.9},
        {"timestamp": 1.0, "pagerank_score": 0.8},
        {"timestamp": 4.0, "pagerank_score": 0.85},
    ]
    method_a = predicted_span_from_frames([f["timestamp"] for f in frames])
    method_b = predicted_span_from_frames_clustered(frames, gap_threshold_s=3.0, tail_trim_pct=0)
    assert method_b == method_a == (0.0, 4.0)


def test_method_c_unchanged_fallback():
    frames = [{"timestamp": 1.0, "frame_idx": 0, "scene_id": -1}]
    span, fallback = predicted_span_from_frames_scene(frames, {}, query_embedding=None)
    assert fallback is True
    assert span == (1.0, 1.0)


def test_method_d_unchanged():
    q = _unit_embedding(1)
    frames = [
        {"frame_idx": 0, "timestamp": 5.0, "clip_embedding": _unit_embedding(2)},
        {"frame_idx": 1, "timestamp": 10.0, "clip_embedding": q},  # exact match -> peak
    ]
    span, used = predicted_span_from_frames_peak(frames, q, half_width_s=2.2, duration_s=100.0)
    assert used is True
    assert span == (7.8, 12.2)


# ── (b) Method E degenerate case: tau=0, unrestricted bounds ───────────────

def test_method_e_tau_zero_includes_everything_up_to_bounds():
    """With tau=0, every survivor frame clears the threshold (z is always
    in [0,1] after min-max normalization), so expansion should run all the
    way to the first/last survivor -- the widest span the profile allows,
    only capped by width bounds (w_max) or [0, D] clamping."""
    frames = []
    for i in range(10):
        frames.append({
            "frame_idx": i, "timestamp": float(i),
            "clip_embedding": _unit_embedding(i + 100),
        })
    q = frames[5]["clip_embedding"]  # exact match at t=5 -> anchor
    retrieved = [frames[5]]
    span, used, tel = predicted_span_from_frames_profile(
        frames, retrieved, q, duration_s=9.0, tau=0.0, w_min=0.0, w_max=100.0, smooth_s=0.0,
    )
    assert used is True
    assert tel["degenerate_profile_fallback"] is False
    # tau=0 -> every z_i >= 0 clears the threshold -> full survivor extent
    assert span == (0.0, 9.0)


def test_method_e_degenerate_profile_falls_back_and_reports_it():
    """Fewer than 2 scored survivors -> degenerate profile -> falls back to
    a Method-D-style span (never a bare unrelated (0,0))."""
    frames = [{"frame_idx": 0, "timestamp": 5.0, "clip_embedding": _unit_embedding(1)}]
    q = _unit_embedding(2)
    retrieved = frames
    span, used, tel = predicted_span_from_frames_profile(
        frames, retrieved, q, duration_s=20.0, tau=0.5, w_min=1.0,
    )
    assert tel["degenerate_profile_fallback"] is True
    assert used is False  # forced no-anchor fallback discipline
    assert span == (4.5, 5.5)  # retrieved_frames[0] +/- w_min/2


# ── (c) spans never exceed [0, D] ───────────────────────────────────────────

@pytest.mark.parametrize("anchor_t,duration", [(0.5, 3.0), (2.9, 3.0), (0.0, 3.0)])
def test_method_e_never_exceeds_duration_bounds(anchor_t, duration):
    frames = []
    for i in range(20):
        t = i * (duration / 19)
        frames.append({"frame_idx": i, "timestamp": t, "clip_embedding": _unit_embedding(i + 200)})
    # anchor frame closest to anchor_t
    anchor_frame = min(frames, key=lambda f: abs(f["timestamp"] - anchor_t))
    q = anchor_frame["clip_embedding"]
    span, used, tel = predicted_span_from_frames_profile(
        frames, [anchor_frame], q, duration_s=duration, tau=0.0, w_min=10.0, w_max=50.0,
    )
    assert 0.0 <= span[0] <= span[1] <= duration


def test_method_f_never_exceeds_duration_bounds():
    frames = [
        {"frame_idx": 0, "timestamp": 0.1, "clip_embedding": _unit_embedding(1)},
        {"frame_idx": 1, "timestamp": 2.9, "clip_embedding": _unit_embedding(2)},
    ]
    q = frames[0]["clip_embedding"]
    span, used, tel = predicted_span_from_frames_weighted_spread(
        frames, q, duration_s=3.0, alpha=10.0, w_min=0.5,
    )
    assert 0.0 <= span[0] <= span[1] <= 3.0


def test_method_f_width_floor_prevents_zero_width():
    """Method F's whole purpose is fixing Method B's zero-width failure mode
    -- a single retrieved frame (std=0) must still get width >= w_min."""
    frames = [{"frame_idx": 0, "timestamp": 10.0, "clip_embedding": _unit_embedding(1)}]
    q = frames[0]["clip_embedding"]
    span, used, tel = predicted_span_from_frames_weighted_spread(
        frames, q, duration_s=100.0, alpha=1.0, w_min=1.4,
    )
    assert span[1] - span[0] == pytest.approx(1.4)


# ── (d) val_confirm exclusion assertion fires ───────────────────────────────

def test_confirm_exclusion_assertion_fires():
    confirm = {"111", "222"}
    with pytest.raises(AssertionError, match="val_confirm"):
        assert_no_confirm_videos(["333", "111"], confirm)


def test_confirm_exclusion_assertion_passes_clean():
    confirm = {"111", "222"}
    assert_no_confirm_videos(["333", "444"], confirm) is None  # no raise


# ── structurally-barred helper ──────────────────────────────────────────────

def test_structurally_barred():
    # max gold length 1.0 < 0.5 * 4.0 = 2.0 -> barred
    assert is_structurally_barred([[0.0, 1.0]], pred_width=4.0) is True
    # max gold length 3.0 >= 0.5 * 4.0 = 2.0 -> not barred
    assert is_structurally_barred([[0.0, 3.0]], pred_width=4.0) is False
    # zero-width prediction is never "barred" by this definition
    assert is_structurally_barred([[0.0, 0.5]], pred_width=0.0) is False


# ── two-stage profile computation matches the single-call wrapper ─────────

def test_two_stage_matches_single_call_wrapper():
    frames = []
    for i in range(30):
        frames.append({"frame_idx": i, "timestamp": i * 0.7, "clip_embedding": _unit_embedding(i + 300)})
    q = _unit_embedding(999)
    retrieved = frames[10:14]

    profile = compute_similarity_profile(frames, q, smooth_s=0.5)
    span1, used1, tel1 = span_from_similarity_profile(
        profile, retrieved, duration_s=25.0, tau=0.5, w_min=1.0,
    )
    span2, used2, tel2 = predicted_span_from_frames_profile(
        frames, retrieved, q, duration_s=25.0, tau=0.5, w_min=1.0, smooth_s=0.5,
    )
    assert span1 == span2
    assert used1 == used2
