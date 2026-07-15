"""Regression tests for P1-02: full-decode raw records must never
receive stale or undefined motion geometry.

These tests exercise iris/charon_v.py without requiring a real video
file.  We mock the PyAV objects that parse_video consumes so the tests
run in pure Python with no I/O and no av dependency at import time.

Scenarios covered
-----------------
1. SKIP-first  : the very first decoded frame yields tier SKIP under
   full_decode=True, return_raw=True.  Must not raise NameError.
   The SKIP raw record must carry the correct all-zero geometry.
2. Stale-carry  : non-SKIP frame (non-zero MVs) immediately followed by
   a SKIP frame (empty MVs).  The SKIP raw record must NOT equal the
   preceding frame's geometry.
3. Survivor-unchanged: a non-SKIP frame appended to output_frames must
   carry exactly the same geometry it would have produced before the fix
   (i.e. the pre-existing code path for non-SKIP survivors is unchanged).
"""
from __future__ import annotations

import types
import sys
import unittest.mock as mock
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from iris.charon_v import compute_motion_geometry


# ---------------------------------------------------------------------------
# Minimal PyAV mock infrastructure
# ---------------------------------------------------------------------------


GEOM_ZERO = compute_motion_geometry([], 320, 240)
_GEOM_KEYS = set(GEOM_ZERO.keys())


class _FakeSideDataEntry:
    """Single motion vector side-data entry."""
    def __init__(self, src_x, src_y, dst_x, dst_y, motion_x, motion_y):
        self.src_x = src_x; self.src_y = src_y
        self.dst_x = dst_x; self.dst_y = dst_y
        # charon_v divides by 4 (quarter-pixel units)
        self.motion_x = motion_x * 4
        self.motion_y = motion_y * 4


class _FakeSideData:
    """Iterable side-data block carrying motion vectors."""
    def __init__(self, mvs):
        self._mvs = [_FakeSideDataEntry(*mv) for mv in mvs]
        # charon_v checks sd.type.name == "MOTION_VECTORS"
        self.type = types.SimpleNamespace(name="MOTION_VECTORS")

    def __iter__(self):
        return iter(self._mvs)


def _make_frame(frame_idx, tier, mvs, width=320, height=240):
    """
    Build a mock PyAV VideoFrame whose attributes satisfy parse_video's
    access patterns.
    
    tier is used to set key_frame and pts so that the tier classification
    logic in parse_video lands on the right branch.
    """
    frame = MagicMock()
    frame.pts = frame_idx        # unique PTS per frame
    frame.time = float(frame_idx)
    frame.width = width
    frame.height = height
    frame.key_frame = (tier == "I_FRAME")

    # Y-plane: small constant array so luma computations work
    Y = np.full((height, width), 128, dtype=np.uint8)
    frame.to_ndarray.return_value = Y
    frame.to_image.return_value = MagicMock()  # PIL image stub

    # pict_type attribute
    frame.pict_type = types.SimpleNamespace(name="I" if tier == "I_FRAME" else "P")

    # side_data: list containing one side-data block with the given MVs
    if mvs and tier != "I_FRAME":
        frame.side_data = [_FakeSideData(mvs)]
    else:
        frame.side_data = []

    return frame


def _make_stream(fps=25.0):
    stream = MagicMock()
    stream.average_rate = fps
    stream.time_base = 1 / fps
    stream.codec_context = MagicMock()
    stream.codec_context.options = {}
    return stream


def _run_parse_video_mocked(
    frames_spec,
    full_decode=True,
    return_raw=True,
    adaptive=False,
    candidate_thresh=0.0,
    salient_thresh=0.0,
):
    """
    Drive parse_video with a fully-mocked PyAV container.

    frames_spec: list of (tier, mvs) tuples where
      tier is "SKIP", "CANDIDATE", "SALIENT", "PEAK", or "I_FRAME"
      mvs  is a list of (src_x, src_y, dst_x, dst_y, motion_x, motion_y)
            tuples (in integer-pixel units; the mock multiplies by 4).

    Returns (output_frames, raw_records, stats).
    """
    import iris.charon_v as charon_mod

    n = len(frames_spec)

    # Build mock frames
    mock_frames = [
        _make_frame(i, tier, mvs) for i, (tier, mvs) in enumerate(frames_spec)
    ]

    # pts_to_packet: maps pts -> (packet_size, is_keyframe)
    # We set packet sizes so that tier classification matches frames_spec.
    # Under adaptive=False, salient_thresh=0 and candidate_thresh=0, all
    # non-I frames would normally be CANDIDATE (ps > 0 >= 0).  We force
    # SKIP by setting ps = -1 (below 0 candidate threshold) when needed.
    pts_to_packet = {}
    for i, (tier, mvs) in enumerate(frames_spec):
        is_kf = (tier == "I_FRAME")
        if tier == "SKIP":
            ps = -1  # below candidate_thresh=0.0 → SKIP
        elif tier == "I_FRAME":
            ps = 10000
        else:
            ps = 1    # above candidate_thresh=0.0 → CANDIDATE
        pts_to_packet[i] = (ps, is_kf)

    # Mock _demux_packet_curve to return our synthetic data
    all_frame_energies = [(i, abs(pts_to_packet[i][0])) for i in range(n)]
    iframe_indices = [i for i, (t, _) in enumerate(frames_spec) if t == "I_FRAME"]
    if not iframe_indices:
        iframe_indices = [0]
    energies = [abs(pts_to_packet[i][0]) for i in range(n)]

    # Mock container and stream
    mock_stream = _make_stream()
    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]
    mock_container.decode.return_value = iter(mock_frames)

    with (
        patch.object(charon_mod, "_demux_packet_curve",
                     return_value=(all_frame_energies, iframe_indices,
                                   energies, pts_to_packet)),
        patch("iris.charon_v.av.open", return_value=mock_container),
    ):
        result = charon_mod.parse_video(
            "fake.mp4",
            full_decode=full_decode,
            return_raw=return_raw,
            return_stats=True,
            adaptive=adaptive,
            candidate_thresh=candidate_thresh,
            salient_thresh=salient_thresh,
        )

    # parse_video with return_stats=True, return_raw=True returns
    # (output_frames, stats, raw_records)
    if return_raw:
        output_frames, stats, raw_records = result
        return output_frames, raw_records, stats
    else:
        output_frames, stats = result
        return output_frames, [], stats


# ---------------------------------------------------------------------------
# Test 1: SKIP-first — no NameError, geometry is all-zero
# ---------------------------------------------------------------------------


class TestSkipFirstFrame:
    """
    The very first decoded frame is SKIP under full_decode=True.
    Before the fix this raised NameError because geom was never assigned.
    """

    def test_no_name_error(self):
        """full_decode SKIP-first must not raise NameError."""
        # I_FRAME at index 0 so tier classification works, then a real SKIP.
        # We put the I_FRAME first (key_frame=True always wins tier),
        # then a SKIP frame (ps < 0).
        # To exercise "SKIP is the first *raw-record-producing* frame" we
        # use a single SKIP frame with an I_FRAME that already has geometry.
        # Real scenario: full_decode visits every frame, first one is SKIP.
        # We achieve this by making frame 0 a SKIP (ps=-1, not a key_frame).
        # We need at least one I_FRAME anchor for detect_peaks; put it at 1.
        frames_spec = [
            ("SKIP", []),
            ("I_FRAME", []),
        ]
        # Should not raise
        output_frames, raw_records, stats = _run_parse_video_mocked(
            frames_spec, full_decode=True, return_raw=True,
        )
        # SKIP frame should have produced a raw record
        assert len(raw_records) >= 1

    def test_skip_first_raw_record_geometry_is_zero(self):
        """SKIP raw record when MVs are empty must have all-zero geometry."""
        frames_spec = [
            ("SKIP", []),
            ("I_FRAME", []),
        ]
        output_frames, raw_records, stats = _run_parse_video_mocked(
            frames_spec, full_decode=True, return_raw=True,
        )
        skip_recs = [r for r in raw_records if r["frame_idx"] == 0]
        assert skip_recs, "Expected at least one raw record for the SKIP frame at idx 0"
        skip_rec = skip_recs[0]
        # All geometry fields must be present and equal to all-zero
        for key, expected in GEOM_ZERO.items():
            assert key in skip_rec, f"Geometry key {key!r} missing from SKIP raw record"
            assert skip_rec[key] == pytest.approx(expected), (
                f"SKIP raw record {key}={skip_rec[key]} != expected {expected} (all-zero)"
            )


# ---------------------------------------------------------------------------
# Test 2: Stale carry-over — SKIP after non-SKIP must not inherit geometry
# ---------------------------------------------------------------------------


class TestNoStaleCarryover:
    """
    A non-SKIP frame with non-zero MVs followed by a SKIP frame with empty MVs.
    The SKIP raw record geometry must NOT equal the non-SKIP frame's geometry.
    """

    # A single non-zero MV: (src_x, src_y, dst_x, dst_y, motion_x, motion_y)
    # Using large values to ensure the geometry dict comes out non-zero.
    _STRONG_MVS = [(0, 0, 16, 16, 8, 8), (16, 0, 32, 16, -4, 4)]

    def test_skip_geom_differs_from_predecessor(self):
        frames_spec = [
            ("I_FRAME", []),           # frame 0 — anchor
            ("CANDIDATE", self._STRONG_MVS),  # frame 1 — non-SKIP with real MVs
            ("SKIP", []),              # frame 2 — SKIP with empty MVs
        ]
        output_frames, raw_records, stats = _run_parse_video_mocked(
            frames_spec, full_decode=True, return_raw=True,
        )
        # Collect raw records by frame_idx
        by_idx = {r["frame_idx"]: r for r in raw_records}
        assert 1 in by_idx, "Expected raw record for non-SKIP frame at idx 1"
        assert 2 in by_idx, "Expected raw record for SKIP frame at idx 2"

        cand_rec = by_idx[1]
        skip_rec = by_idx[2]

        # Non-SKIP frame must have non-zero geometry (it has real MVs)
        cand_geom = {k: cand_rec[k] for k in _GEOM_KEYS}
        skip_geom = {k: skip_rec[k] for k in _GEOM_KEYS}

        # The CANDIDATE geometry must be non-zero (sanity check on our MVs)
        assert any(v != 0.0 for v in cand_geom.values()), (
            "CANDIDATE frame with non-zero MVs must produce non-zero geometry; ",
            f"got {cand_geom}"
        )

        # The SKIP geometry must be all-zero (empty MVs)
        for key, val in skip_geom.items():
            assert val == pytest.approx(0.0), (
                f"SKIP raw record {key}={val} is non-zero; stale carry-over detected!"
            )

        # Double-check: SKIP geometry != CANDIDATE geometry
        assert skip_geom != cand_geom, (
            "SKIP frame inherited geometry from preceding CANDIDATE — stale carry-over!"
        )

    def test_skip_geom_equals_zero_reference(self):
        """SKIP geometry must equal compute_motion_geometry([], w, h)."""
        frames_spec = [
            ("I_FRAME", []),
            ("CANDIDATE", self._STRONG_MVS),
            ("SKIP", []),
        ]
        output_frames, raw_records, stats = _run_parse_video_mocked(
            frames_spec, full_decode=True, return_raw=True,
        )
        by_idx = {r["frame_idx"]: r for r in raw_records}
        skip_rec = by_idx[2]
        for key, expected in GEOM_ZERO.items():
            assert skip_rec[key] == pytest.approx(expected), (
                f"SKIP raw record {key}={skip_rec[key]} != reference zero {expected}"
            )


# ---------------------------------------------------------------------------
# Test 3: Non-SKIP survivor behavior is byte-for-byte unchanged
# ---------------------------------------------------------------------------


class TestSurvivorGeometryUnchanged:
    """
    For non-SKIP frames that appear in output_frames, the geometry values
    in the output_frames dict must equal compute_motion_geometry applied
    to the same motion_vectors — i.e. the pre-existing non-SKIP code path
    is untouched by the fix.
    """

    _MVS = [(0, 0, 16, 16, 6, 0), (16, 0, 32, 16, 0, 6)]

    def test_survivor_output_frames_geometry_matches_reference(self):
        frames_spec = [
            ("I_FRAME", []),           # frame 0
            ("CANDIDATE", self._MVS),  # frame 1 — survivor
        ]
        output_frames, raw_records, stats = _run_parse_video_mocked(
            frames_spec, full_decode=True, return_raw=True,
        )
        # Find the CANDIDATE in output_frames
        survivors = [f for f in output_frames if f["frame_idx"] == 1]
        assert survivors, "CANDIDATE frame must appear in output_frames"
        survivor = survivors[0]

        # Compute the reference geometry using the real motion vectors
        # that charon_v would have extracted from the frame.
        # Our mock gives motion_x * 4 / 4 = motion_x for each MV.
        expected_mvs = [
            (mv[0], mv[1], mv[2], mv[3], mv[4], mv[5])
            for mv in self._MVS
        ]
        ref_geom = compute_motion_geometry(expected_mvs, 320, 240)

        for key, ref_val in ref_geom.items():
            assert key in survivor, f"Geometry key {key!r} missing from output_frames entry"
            assert survivor[key] == pytest.approx(ref_val, rel=1e-5), (
                f"output_frames geometry {key}={survivor[key]} != reference {ref_val}"
            )

    def test_survivor_in_output_frames_not_in_skip(self):
        """SKIP frames must NEVER appear in output_frames."""
        frames_spec = [
            ("I_FRAME", []),
            ("CANDIDATE", self._MVS),
            ("SKIP", []),
        ]
        output_frames, raw_records, stats = _run_parse_video_mocked(
            frames_spec, full_decode=True, return_raw=True,
        )
        tiers = {f["tier"] for f in output_frames}
        assert "SKIP" not in tiers, (
            "SKIP tier must never appear in output_frames (survivor list)"
        )

    def test_raw_record_survivor_geometry_matches_output_frames(self):
        """
        The geometry embedded in the raw_record for a non-SKIP frame must
        match the geometry stored in output_frames for the same frame.
        """
        frames_spec = [
            ("I_FRAME", []),
            ("CANDIDATE", self._MVS),
        ]
        output_frames, raw_records, stats = _run_parse_video_mocked(
            frames_spec, full_decode=True, return_raw=True,
        )
        of = {f["frame_idx"]: f for f in output_frames}
        rr = {r["frame_idx"]: r for r in raw_records}
        assert 1 in of and 1 in rr, "Both dicts must contain frame idx 1"
        for key in _GEOM_KEYS:
            assert of[1][key] == pytest.approx(rr[1][key], rel=1e-5), (
                f"Geometry mismatch for {key}: output_frames={of[1][key]} raw_record={rr[1][key]}"
            )


# ---------------------------------------------------------------------------
# Test 4: compute_motion_geometry contract for empty / non-empty vectors
# ---------------------------------------------------------------------------


class TestComputeMotionGeometryContract:
    """Unit-tests for compute_motion_geometry alone (no av dependency)."""

    def test_empty_mvs_returns_all_zero(self):
        result = compute_motion_geometry([], 320, 240)
        assert result == {
            "divergence": 0.0, "curl": 0.0, "jacobian_frobenius": 0.0,
            "hessian_max_eigenvalue": 0.0, "motion_entropy": 0.0,
        }

    def test_zero_dimension_returns_all_zero(self):
        mvs = [(0, 0, 16, 16, 1.0, 0.0)]
        assert compute_motion_geometry(mvs, 0, 240) == GEOM_ZERO
        assert compute_motion_geometry(mvs, 320, 0) == GEOM_ZERO

    def test_nonzero_mvs_produce_nonzero_geometry(self):
        # Two MVs with different directions — should produce non-trivial divergence
        mvs = [(0, 0, 16, 16, 8.0, 0.0), (16, 0, 32, 16, -8.0, 0.0)]
        result = compute_motion_geometry(mvs, 64, 64)
        assert any(v != 0.0 for v in result.values()), (
            f"Non-zero MVs produced all-zero geometry: {result}"
        )

    def test_result_keys_unchanged(self):
        result = compute_motion_geometry([], 320, 240)
        assert set(result.keys()) == {
            "divergence", "curl", "jacobian_frobenius",
            "hessian_max_eigenvalue", "motion_entropy",
        }


# ---------------------------------------------------------------------------
# Test 5: Selective decode stats instrumentation (P1-03)
# ---------------------------------------------------------------------------


class TestSelectiveDecodeStats:
    """
    Verifies that stats dictionaries contain correct total decoded,
    pixel-processed, and skip ratio numbers.
    """

    def test_ffmpeg_always_decodes_all_frames(self):
        """total_frames_decoded_by_ffmpeg must equal total_frames."""
        frames_spec = [
            ("I_FRAME", []),
            ("SKIP", []),
            ("CANDIDATE", []),
            ("SKIP", []),
        ]
        output_frames, raw_records, stats = _run_parse_video_mocked(
            frames_spec, full_decode=False, return_raw=True,
        )
        assert stats["total_frames_decoded_by_ffmpeg"] == len(frames_spec)
        assert stats["total_frames_decoded_by_ffmpeg"] == stats["total"]

    def test_selective_gating_reduces_pixel_processing(self):
        """
        Under full_decode=False, if there is a SKIP frame,
        frames_with_pixel_processing must be less than total frames decoded,
        and skip ratio must be > 0.
        """
        frames_spec = [
            ("I_FRAME", []),
            ("SKIP", []),      # SKIP frame -> no pixel processing under full_decode=False
            ("CANDIDATE", []),
        ]
        output_frames, raw_records, stats = _run_parse_video_mocked(
            frames_spec, full_decode=False, return_raw=True,
        )
        assert stats["frames_with_pixel_processing"] == 2  # I_FRAME and CANDIDATE
        assert stats["frames_with_pixel_processing"] < stats["total_frames_decoded_by_ffmpeg"]
        assert stats["pixel_processing_skip_ratio"] == pytest.approx(1.0 - 2/3)

    def test_full_decode_processes_all_frames(self):
        """Under full_decode=True, every frame receives pixel processing, skip ratio is 0.0."""
        frames_spec = [
            ("I_FRAME", []),
            ("SKIP", []),
            ("CANDIDATE", []),
        ]
        output_frames, raw_records, stats = _run_parse_video_mocked(
            frames_spec, full_decode=True, return_raw=True,
        )
        assert stats["frames_with_pixel_processing"] == 3
        assert stats["frames_with_pixel_processing"] == stats["total_frames_decoded_by_ffmpeg"]
        assert stats["pixel_processing_skip_ratio"] == pytest.approx(0.0)

