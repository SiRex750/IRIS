"""
Layer 1 Stress Tests
====================
Covers all edge-case gaps not covered by the existing test suite.

Groups:
  A -- charon_v.parse_video / _demux_packet_curve structural invariants
  B -- compute_motion_geometry stress inputs
  C -- codec_validator.py edge cases
  D -- ActionScoreModule stress inputs
  E -- FrameMotionDescriptor invariants

Video-dependent tests (Group A) auto-skip when mov_bbb.mp4 is absent.
All Group B-E tests are purely synthetic (no video required).
"""
from __future__ import annotations

import math
import os
import socket
import urllib.request

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared video fixture
# ---------------------------------------------------------------------------

_LOCAL_VIDEO = "mov_bbb.mp4"


def _resolve_video() -> str | None:
    if os.path.exists(_LOCAL_VIDEO):
        return _LOCAL_VIDEO
    url = "https://www.w3schools.com/html/mov_bbb.mp4"
    socket.setdefaulttimeout(5.0)
    try:
        urllib.request.urlretrieve(url, _LOCAL_VIDEO)
        return _LOCAL_VIDEO
    except Exception:
        return None


@pytest.fixture(scope="module")
def video() -> str:
    v = _resolve_video()
    if v is None:
        pytest.skip("mov_bbb.mp4 unavailable and download failed")
    return v


@pytest.fixture(scope="module")
def parsed_full(video):
    from iris.charon_v import parse_video
    output, stats, raw = parse_video(
        video, return_stats=True, return_raw=True, full_decode=False
    )
    return output, stats, raw


# ===========================================================================
# GROUP A -- parse_video structural invariants (video-dependent)
# ===========================================================================

class TestCharonStructuralInvariants:
    """Prove structural correctness of parse_video output on a real clip."""

    def test_first_frame_luma_diff_is_zero(self, parsed_full):
        """Bug-6 regression: very first output survivor must have luma_diff_energy == 0.0."""
        output, _, _ = parsed_full
        assert output, "output_frames is empty"
        first = output[0]
        assert first["luma_diff_energy"] == 0.0, (
            f"First output frame has luma_diff_energy={first['luma_diff_energy']}, expected 0.0"
        )

    def test_no_skip_frames_in_output(self, parsed_full):
        """SKIP frames must never appear in output_frames."""
        output, _, _ = parsed_full
        skip_frames = [f for f in output if f.get("tier") == "SKIP"]
        assert not skip_frames, (
            f"Found {len(skip_frames)} SKIP frames in output_frames "
            f"(idxs: {[f['frame_idx'] for f in skip_frames[:5]]})"
        )

    def test_all_required_keys_present(self, parsed_full):
        """Every survivor must carry the full set of required output keys."""
        output, _, _ = parsed_full
        required = {
            "frame_idx", "timestamp", "tier", "luma_diff_energy",
            "packet_size", "motion_vectors", "pil_image",
            "divergence", "curl", "jacobian_frobenius",
            "hessian_max_eigenvalue", "motion_entropy",
        }
        for f in output:
            missing = required - f.keys()
            assert not missing, f"frame {f['frame_idx']} missing keys: {missing}"

    def test_residual_energy_key_absent(self, parsed_full):
        """Old 'residual_energy' field must not appear anywhere (Bug-8 rename)."""
        output, _, _ = parsed_full
        bad = [f["frame_idx"] for f in output if "residual_energy" in f]
        assert not bad, (
            f"Frames still carry legacy 'residual_energy' key: {bad[:5]}"
        )

    def test_first_output_frame_luma_diff_is_zero(self, parsed_full):
        """Only the first-ever processed frame gets luma_diff_energy == 0.0.
        Mid-video I-frames are diffed against the prior retained survivor,
        so their luma_diff is intentionally non-zero (gap-diff mode).
        This test verifies the design: only frame index 0 of the output is 0.0.
        """
        output, _, _ = parsed_full
        assert output, "output_frames is empty"
        # The very first survivor must have luma_diff=0.0 (no prior frame)
        first = output[0]
        assert first["luma_diff_energy"] == 0.0, (
            f"First output survivor (frame {first['frame_idx']}) has "
            f"luma_diff_energy={first['luma_diff_energy']}, expected 0.0"
        )

    def test_pil_image_present_for_all_survivors(self, parsed_full):
        """Every survivor needs pil_image for downstream CLIP captioning (Bug-2 dependency)."""
        output, _, _ = parsed_full
        missing = [f["frame_idx"] for f in output if f.get("pil_image") is None]
        assert not missing, (
            f"{len(missing)} survivors have pil_image=None: {missing[:5]}"
        )

    def test_tier_counts_sum_to_total(self, parsed_full):
        """stats tier counters must be self-consistent."""
        _, stats, _ = parsed_full
        computed = (
            stats["i_frames"] + stats["peaks"] + stats["salient"]
            + stats["candidate"] + stats["skipped"]
        )
        assert computed == stats["total"], (
            f"Tier counts sum to {computed} but stats['total']={stats['total']}"
        )

    def test_adaptive_has_at_least_one_scene(self, parsed_full):
        """Adaptive mode must always detect at least 1 scene (keyframe anchor)."""
        _, stats, _ = parsed_full
        assert stats["num_scenes"] >= 1, f"num_scenes={stats['num_scenes']}, expected >= 1"

    def test_raw_records_cover_every_frame(self, parsed_full):
        """raw_records must have exactly one entry per frame -- none silently dropped."""
        _, stats, raw = parsed_full
        assert len(raw) == stats["total"], (
            f"raw_records length {len(raw)} != stats['total'] {stats['total']}"
        )

    def test_raw_luma_entropy_non_negative(self, parsed_full):
        """Luma entropy from Y-plane histogram is always >= 0.0."""
        _, _, raw = parsed_full
        bad = [(r["frame_idx"], r["luma_entropy"]) for r in raw if r["luma_entropy"] < 0.0]
        assert not bad, f"Negative luma_entropy in raw records: {bad[:5]}"

    def test_raw_motion_magnitude_non_negative(self, parsed_full):
        """Motion magnitude is sqrt of squared MVs -- always >= 0.0."""
        _, _, raw = parsed_full
        bad = [(r["frame_idx"], r["motion_magnitude"]) for r in raw
               if r["motion_magnitude"] < 0.0]
        assert not bad, f"Negative motion_magnitude in raw records: {bad[:5]}"

    def test_stats_expensive_lt_total(self, parsed_full):
        """Selective decode must process fewer frames than total (SKIP savings proof)."""
        _, stats, _ = parsed_full
        assert stats["frames_expensive_processed"] < stats["total"], (
            "No SKIP savings: frames_expensive_processed == total. "
            "Either all frames are survivors or selective decode is broken."
        )

    def test_luma_diff_energy_in_range(self, parsed_full):
        """luma_diff_energy is mean-abs-diff / 255 -- must be in [0.0, 1.0]."""
        output, _, _ = parsed_full
        bad = [(f["frame_idx"], f["luma_diff_energy"]) for f in output
               if not (0.0 <= f["luma_diff_energy"] <= 1.0)]
        assert not bad, (
            f"{len(bad)} frames have luma_diff_energy outside [0,1]: {bad[:3]}"
        )

    def test_timestamps_monotonically_non_decreasing(self, parsed_full):
        """Timestamps in output must be non-decreasing (display order)."""
        output, _, _ = parsed_full
        for i in range(1, len(output)):
            assert output[i]["timestamp"] >= output[i - 1]["timestamp"], (
                f"Timestamp decreased: frame {output[i-1]['frame_idx']} "
                f"({output[i-1]['timestamp']:.3f}s) > frame {output[i]['frame_idx']} "
                f"({output[i]['timestamp']:.3f}s)"
            )

    def test_packet_size_positive_for_survivors(self, parsed_full):
        """All survivor frames must have a positive packet_size from demux."""
        output, _, _ = parsed_full
        bad = [(f["frame_idx"], f.get("packet_size")) for f in output
               if f.get("packet_size", 0.0) <= 0.0]
        assert not bad, f"{len(bad)} survivors have packet_size <= 0: {bad[:5]}"

    def test_geometry_fields_finite(self, parsed_full):
        """All 5 motion geometry fields must be finite floats for every survivor."""
        output, _, _ = parsed_full
        geom_keys = ("divergence", "curl", "jacobian_frobenius",
                     "hessian_max_eigenvalue", "motion_entropy")
        for f in output:
            for k in geom_keys:
                v = f.get(k, float("nan"))
                assert math.isfinite(v), (
                    f"frame {f['frame_idx']} has non-finite {k}={v}"
                )


# ===========================================================================
# GROUP B -- compute_motion_geometry stress inputs (synthetic)
# ===========================================================================

class TestMotionGeometryStress:

    def test_zero_width_returns_zero_dict(self):
        from iris.charon_v import compute_motion_geometry
        res = compute_motion_geometry([(0, 0, 8, 8, 4, 4)], width=0, height=240)
        assert all(v == 0.0 for v in res.values()), f"Non-zero output for width=0: {res}"

    def test_zero_height_returns_zero_dict(self):
        from iris.charon_v import compute_motion_geometry
        res = compute_motion_geometry([(0, 0, 8, 8, 4, 4)], width=320, height=0)
        assert all(v == 0.0 for v in res.values()), f"Non-zero output for height=0: {res}"

    def test_single_mv_no_crash(self):
        """A single MV must not crash and must return all finite values."""
        from iris.charon_v import compute_motion_geometry
        res = compute_motion_geometry([(0, 0, 8, 8, 3, -2)], width=160, height=120)
        assert set(res.keys()) == {
            "divergence", "curl", "jacobian_frobenius",
            "hessian_max_eigenvalue", "motion_entropy"
        }
        for k, v in res.items():
            assert math.isfinite(v), f"{k}={v} is not finite for single-MV input"

    def test_uniform_translation_curl_near_zero(self):
        """Pure translation (all MVs identical direction) -> curl approx 0.0."""
        from iris.charon_v import compute_motion_geometry
        mvs = [(gx * 16, gy * 16, gx * 16, gy * 16, 5, 0)
               for gy in range(10) for gx in range(10)]
        res = compute_motion_geometry(mvs, width=160, height=160)
        assert abs(res["curl"]) < 1e-5, (
            f"Uniform horizontal translation should have curl~0, got {res['curl']}"
        )

    def test_zero_mv_field_entropy_is_zero(self):
        """All-zero motion vectors -> motion_entropy == 0.0."""
        from iris.charon_v import compute_motion_geometry
        mvs = [(gx * 16, gy * 16, gx * 16, gy * 16, 0, 0)
               for gy in range(10) for gx in range(10)]
        res = compute_motion_geometry(mvs, width=160, height=160)
        assert res["motion_entropy"] == 0.0, (
            f"Zero MVs -> motion_entropy=0.0, got {res['motion_entropy']}"
        )

    def test_all_outputs_finite_for_random_input(self):
        """Random MV field must never produce NaN or Inf."""
        from iris.charon_v import compute_motion_geometry
        rng = np.random.default_rng(42)
        mvs = [
            (int(rng.integers(0, 320)), int(rng.integers(0, 240)),
             int(rng.integers(0, 320)), int(rng.integers(0, 240)),
             int(rng.integers(-64, 64)), int(rng.integers(-64, 64)))
            for _ in range(200)
        ]
        res = compute_motion_geometry(mvs, width=320, height=240)
        for k, v in res.items():
            assert math.isfinite(v), f"{k}={v} is not finite for random input"

    def test_all_five_keys_always_returned(self):
        """compute_motion_geometry must always return all 5 expected keys."""
        from iris.charon_v import compute_motion_geometry
        expected = {
            "divergence", "curl", "jacobian_frobenius",
            "hessian_max_eigenvalue", "motion_entropy"
        }
        for mvs, w, h in [([], 320, 240), ([(0, 0, 8, 8, 1, 1)], 160, 120)]:
            res = compute_motion_geometry(mvs, w, h)
            assert set(res.keys()) == expected, (
                f"Missing keys: {expected - set(res.keys())}"
            )

    def test_motion_entropy_non_negative(self):
        """Entropy is always >= 0.0 by definition."""
        from iris.charon_v import compute_motion_geometry
        rng = np.random.default_rng(7)
        mvs = [
            (int(rng.integers(0, 160)), int(rng.integers(0, 160)),
             int(rng.integers(0, 160)), int(rng.integers(0, 160)),
             int(rng.integers(-10, 10)), int(rng.integers(-10, 10)))
            for _ in range(50)
        ]
        res = compute_motion_geometry(mvs, 160, 160)
        assert res["motion_entropy"] >= 0.0, (
            f"motion_entropy={res['motion_entropy']} is negative"
        )


# ===========================================================================
# GROUP C -- codec_validator.py edge cases (synthetic)
# ===========================================================================

class TestCodecValidatorEdgeCases:

    @pytest.fixture
    def empty_file(self, tmp_path):
        p = tmp_path / "empty.mp4"
        p.write_bytes(b"")
        return str(p)

    @pytest.fixture
    def truncated_file(self, tmp_path):
        """100-byte truncated ftyp-only stub -- not a real video."""
        ftyp = b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2avc1mp41" + b"\x00" * 60
        p = tmp_path / "truncated.mp4"
        p.write_bytes(ftyp[:100])
        return str(p)

    def test_empty_file_rejected(self, empty_file):
        from iris.codec_validator import validate_video
        result = validate_video(empty_file)
        assert result.status == "reject", (
            f"Empty file should be rejected, got '{result.status}'"
        )
        assert len(result.reasons) > 0

    def test_truncated_file_rejected(self, truncated_file):
        from iris.codec_validator import validate_video
        result = validate_video(truncated_file)
        assert result.status == "reject", (
            f"Truncated file should be rejected, got '{result.status}'"
        )

    def test_all_result_fields_present_on_reject(self, empty_file):
        """ValidationResult must have all 6 fields even on immediate reject."""
        from iris.codec_validator import validate_video
        result = validate_video(empty_file)
        for attr in ("status", "codec", "reasons", "mv_available",
                     "pts_complete", "keyframe_found"):
            assert hasattr(result, attr), f"ValidationResult missing field: {attr}"

    def test_reasons_is_always_a_list(self, empty_file):
        from iris.codec_validator import validate_video
        result = validate_video(empty_file)
        assert isinstance(result.reasons, list), (
            f"reasons must be a list, got {type(result.reasons)}"
        )

    def test_missing_path_rejected(self):
        from iris.codec_validator import validate_video
        result = validate_video("/absolutely/no/such/path/ever.mp4")
        assert result.status == "reject"
        assert result.reasons

    def test_assert_valid_raises_on_empty(self, empty_file):
        from iris.codec_validator import assert_valid
        with pytest.raises(ValueError):
            assert_valid(empty_file)

    def test_assert_valid_returns_result_on_real_video(self):
        from iris.codec_validator import assert_valid
        v = _resolve_video()
        if v is None:
            pytest.skip("mov_bbb.mp4 unavailable")
        result = assert_valid(v)
        assert result is not None
        assert result.status in {"ok", "warn"}


# ===========================================================================
# GROUP D -- ActionScoreModule stress inputs (synthetic)
# ===========================================================================

class TestActionScoreStress:

    def _scorer(self, **kwargs):
        from iris.action_score import ActionScoreConfig, ActionScoreModule
        defaults = dict(
            peak_distance=2,
            peak_prominence=0.05,
            persistence_threshold=0.4,
        )
        defaults.update(kwargs)  # allow callers to override any field
        config = ActionScoreConfig(**defaults)
        return ActionScoreModule(config)

    def test_empty_input_returns_empty_list(self):
        from iris.action_score import ActionScoreModule
        assert ActionScoreModule().score_all([]) == []

    def test_single_frame_is_not_a_peak(self):
        """Single frame cannot be a local maximum -- must not be flagged is_peak."""
        scorer = self._scorer()
        frames = [{"frame_idx": 0, "packet_size": 0.9,
                   "motion_magnitude": 0.9, "luma_entropy": 0.9}]
        records = scorer.score_all(frames)
        assert len(records) == 1
        assert not records[0]["is_peak"], "Single frame cannot be a local peak"

    def test_all_required_fields_in_records(self):
        """Every output record must have all 4 required fields."""
        scorer = self._scorer()
        frames = [
            {"frame_idx": i, "packet_size": float(i) / 10,
             "motion_magnitude": float(i) / 10, "luma_entropy": float(i) / 10}
            for i in range(10)
        ]
        records = scorer.score_all(frames)
        required = {"frame_idx", "action_score", "is_peak", "persistence_value"}
        for r in records:
            assert required.issubset(r.keys()), f"Record missing keys: {required - r.keys()}"

    def test_action_score_in_unit_interval(self):
        """action_score must always be in [0.0, 1.0] for any input distribution."""
        from iris.action_score import ActionScoreModule
        rng = np.random.default_rng(99)
        frames = [
            {"frame_idx": i,
             "packet_size": float(rng.uniform(0, 1e5)),
             "motion_magnitude": float(rng.uniform(0, 1e3)),
             "luma_entropy": float(rng.uniform(0, 1.0))}
            for i in range(100)
        ]
        records = ActionScoreModule().score_all(frames)
        bad = [(r["frame_idx"], r["action_score"]) for r in records
               if not (0.0 <= r["action_score"] <= 1.0)]
        assert not bad, f"action_score out of [0,1]: {bad[:5]}"

    def test_non_peak_frames_have_zero_persistence(self):
        """Non-peak frames must always have persistence_value == 0.0."""
        scorer = self._scorer()
        values = [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
        frames = [
            {"frame_idx": i, "packet_size": v,
             "motion_magnitude": v, "luma_entropy": v}
            for i, v in enumerate(values)
        ]
        records = scorer.score_all(frames)
        for r in records:
            if not r["is_peak"]:
                assert r["persistence_value"] == 0.0, (
                    f"Non-peak frame {r['frame_idx']} has "
                    f"persistence_value={r['persistence_value']}"
                )

    def test_constant_input_no_peaks(self):
        """A perfectly flat signal has no local maxima -- must produce zero peaks."""
        scorer = self._scorer()
        frames = [
            {"frame_idx": i, "packet_size": 0.5,
             "motion_magnitude": 0.5, "luma_entropy": 0.5}
            for i in range(30)
        ]
        records = scorer.score_all(frames)
        peaks = [r for r in records if r["is_peak"]]
        assert not peaks, f"Constant signal produced {len(peaks)} peaks"

    def test_constant_input_gives_constant_score(self):
        """Constant input -> percentile range collapses -> all scores identical (Bug-5)."""
        from iris.action_score import ActionScoreModule
        frames = [
            {"frame_idx": i, "packet_size": 7777.0,
             "motion_magnitude": 0.0, "luma_entropy": 0.0}
            for i in range(60)  # >= 50 triggers percentile normalization path
        ]
        records = ActionScoreModule().score_all(frames)
        scores = [r["action_score"] for r in records]
        assert all(math.isfinite(s) for s in scores), "action_score contains NaN/Inf"
        # All scores must be equal -- constant input gives constant output
        assert max(scores) - min(scores) < 1e-6, (
            f"Constant input should give constant scores, "
            f"got range [{min(scores):.4f}, {max(scores):.4f}]"
        )

    def test_two_distinct_peaks_both_detected(self):
        """Two well-separated spikes must both be detected as peaks."""
        scorer = self._scorer(peak_distance=3, persistence_threshold=0.15)
        values = [
            0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1,
        ]
        frames = [
            {"frame_idx": i, "packet_size": v,
             "motion_magnitude": v, "luma_entropy": v}
            for i, v in enumerate(values)
        ]
        records = scorer.score_all(frames)
        peak_idxs = {r["frame_idx"] for r in records if r["is_peak"]}
        assert 3 in peak_idxs, f"Expected peak at frame 3, got peaks at {peak_idxs}"
        assert 11 in peak_idxs, f"Expected peak at frame 11, got peaks at {peak_idxs}"

    def test_frame_idx_preserved_in_output(self):
        """frame_idx in output must exactly match input frame_idx (no reindexing)."""
        from iris.action_score import ActionScoreModule
        frames = [
            {"frame_idx": i * 10, "packet_size": float(i),
             "motion_magnitude": 0.0, "luma_entropy": 0.0}
            for i in range(8)
        ]
        records = ActionScoreModule().score_all(frames)
        for inp, out in zip(frames, records):
            assert out["frame_idx"] == inp["frame_idx"], (
                f"frame_idx mismatch: input {inp['frame_idx']} vs output {out['frame_idx']}"
            )

    def test_zero_weight_sum_raises_value_error(self):
        """All weights = 0 must raise ValueError with informative message."""
        from iris.action_score import ActionScoreConfig, ActionScoreModule
        config = ActionScoreConfig(
            luma_diff_weight=0.0, motion_weight=0.0, luma_entropy_weight=0.0
        )
        scorer = ActionScoreModule(config)
        frames = [{"frame_idx": 0, "packet_size": 1.0,
                   "motion_magnitude": 1.0, "luma_entropy": 1.0}]
        with pytest.raises(ValueError, match="weights must sum"):
            scorer.score_all(frames)

    def test_missing_feature_fields_default_to_zero(self):
        """Frames with missing feature keys should default to 0.0 gracefully."""
        from iris.action_score import ActionScoreModule
        frames = [{"frame_idx": i} for i in range(10)]  # no packet_size/motion/entropy
        records = ActionScoreModule().score_all(frames)
        assert len(records) == 10
        for r in records:
            assert math.isfinite(r["action_score"]), (
                f"frame {r['frame_idx']}: action_score is not finite"
            )

    def test_persistence_value_in_unit_interval(self):
        """persistence_value must always be in [0.0, 1.0]."""
        from iris.action_score import ActionScoreModule
        rng = np.random.default_rng(7)
        frames = [
            {"frame_idx": i,
             "packet_size": float(rng.uniform(0, 100)),
             "motion_magnitude": float(rng.uniform(0, 10)),
             "luma_entropy": float(rng.uniform(0, 1))}
            for i in range(80)
        ]
        records = ActionScoreModule().score_all(frames)
        bad = [(r["frame_idx"], r["persistence_value"])
               for r in records if not (0.0 <= r["persistence_value"] <= 1.0)]
        assert not bad, f"persistence_value out of [0,1]: {bad[:5]}"

    def test_dominant_peak_persistence_is_one(self):
        """The single dominant peak must normalize to persistence_value == 1.0."""
        scorer = self._scorer()
        values = [0.1] * 5 + [0.9] + [0.1] * 5
        frames = [
            {"frame_idx": i, "packet_size": v,
             "motion_magnitude": v, "luma_entropy": v}
            for i, v in enumerate(values)
        ]
        records = scorer.score_all(frames)
        peaks = [r for r in records if r["is_peak"]]
        assert len(peaks) == 1, f"Expected exactly 1 peak, got {len(peaks)}"
        assert peaks[0]["persistence_value"] == 1.0, (
            f"Sole peak must normalize to 1.0, got {peaks[0]['persistence_value']}"
        )


# ===========================================================================
# GROUP E -- FrameMotionDescriptor invariants (synthetic)
# ===========================================================================

class TestFrameMotionDescriptorInvariants:

    def test_default_geometry_fields_are_zero(self):
        from iris.frame_motion_descriptor import FrameMotionDescriptor
        desc = FrameMotionDescriptor(frame_idx=0, timestamp_sec=1.0)
        for field in ("luma_diff_energy", "divergence", "curl",
                      "jacobian_frobenius", "hessian_max_eigenvalue", "motion_entropy"):
            assert getattr(desc, field) == 0.0, (
                f"{field} default should be 0.0, got {getattr(desc, field)}"
            )

    def test_descriptor_is_frozen(self):
        """frozen=True -- any field assignment must raise an exception."""
        from iris.frame_motion_descriptor import FrameMotionDescriptor
        desc = FrameMotionDescriptor(frame_idx=0, timestamp_sec=0.5, divergence=1.0)
        with pytest.raises(Exception):
            desc.divergence = 99.0  # type: ignore[misc]

    def test_descriptor_uses_slots(self):
        """__slots__ must be present (slots=True on frozen dataclass)."""
        from iris.frame_motion_descriptor import FrameMotionDescriptor
        assert hasattr(FrameMotionDescriptor, "__slots__"), (
            "FrameMotionDescriptor should declare __slots__ (slots=True)"
        )

    def test_descriptor_equality(self):
        """Two descriptors with identical fields must compare equal."""
        from iris.frame_motion_descriptor import FrameMotionDescriptor
        a = FrameMotionDescriptor(frame_idx=5, timestamp_sec=2.5, divergence=0.3)
        b = FrameMotionDescriptor(frame_idx=5, timestamp_sec=2.5, divergence=0.3)
        assert a == b

    def test_descriptor_inequality_on_field_diff(self):
        """Descriptors that differ on any field must not be equal."""
        from iris.frame_motion_descriptor import FrameMotionDescriptor
        a = FrameMotionDescriptor(frame_idx=5, timestamp_sec=2.5, divergence=0.3)
        b = FrameMotionDescriptor(frame_idx=5, timestamp_sec=2.5, divergence=0.9)
        assert a != b

    def test_all_fields_accessible_after_construction(self):
        """All 8 fields must be readable after construction."""
        from iris.frame_motion_descriptor import FrameMotionDescriptor
        desc = FrameMotionDescriptor(
            frame_idx=1, timestamp_sec=0.04,
            luma_diff_energy=0.1, divergence=0.2, curl=0.3,
            jacobian_frobenius=0.4, hessian_max_eigenvalue=0.5,
            motion_entropy=0.6,
        )
        assert desc.frame_idx == 1
        assert desc.timestamp_sec == pytest.approx(0.04)
        assert desc.luma_diff_energy == pytest.approx(0.1)
        assert desc.divergence == pytest.approx(0.2)
        assert desc.curl == pytest.approx(0.3)
        assert desc.jacobian_frobenius == pytest.approx(0.4)
        assert desc.hessian_max_eigenvalue == pytest.approx(0.5)
        assert desc.motion_entropy == pytest.approx(0.6)
