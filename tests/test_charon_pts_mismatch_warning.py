"""Item 8 regression test: the 1-to-1 packet/frame audit in
iris.charon_v.parse_video previously referenced has_none_pts/has_duplicate_pts
without ever defining them in parse_video's scope -- a latent NameError bug
that would have fired instead of the documented "silently skip" behavior, on
the exact malformed-PTS streams the audit exists to catch. Fixed by having
_demux_packet_curve optionally return the flags it already computes
internally (return_pts_flags=True), and by turning the previously-bare skip
into a structured RuntimeWarning.
"""
from __future__ import annotations

import os
import warnings

import pytest

from iris.charon_v import _demux_packet_curve, parse_video

REAL_VIDEO = os.path.join(
    os.path.dirname(__file__), "..", "eval", "data", "nextqa", "NExTVideo_flat", "6936757706.mp4"
)


def test_demux_packet_curve_default_signature_unchanged():
    """The two other call sites (phase6_scene_calibrate.py,
    tests/test_demux_curve.py) unpack exactly 4 values -- default behavior
    (return_pts_flags=False) must keep returning a 4-tuple."""
    if not os.path.exists(REAL_VIDEO):
        pytest.skip("test video not on disk")
    result = _demux_packet_curve(REAL_VIDEO)
    assert len(result) == 4


def test_demux_packet_curve_return_pts_flags():
    if not os.path.exists(REAL_VIDEO):
        pytest.skip("test video not on disk")
    result = _demux_packet_curve(REAL_VIDEO, return_pts_flags=True)
    assert len(result) == 5
    all_frame_energies, iframe_indices, energies, pts_to_packet, (has_none_pts, has_duplicate_pts) = result
    assert isinstance(has_none_pts, bool)
    assert isinstance(has_duplicate_pts, bool)


def test_parse_video_no_longer_raises_nameerror_on_pts_mismatch(monkeypatch):
    """The core bug: has_none_pts/has_duplicate_pts must be bound in
    parse_video's scope. Force the mismatch branch (by monkeypatching
    _demux_packet_curve to report a pts_to_packet dict shorter than the real
    decoded frame count, with has_none_pts=True) and confirm parse_video
    raises neither NameError nor ValueError."""
    if not os.path.exists(REAL_VIDEO):
        pytest.skip("test video not on disk")

    real_demux = _demux_packet_curve

    def fake_demux(video_path, return_pts_flags=False):
        all_frame_energies, iframe_indices, energies, pts_to_packet = real_demux(video_path)
        # Force a mismatch: drop one entry from pts_to_packet so
        # total_frames != len(pts_to_packet) inside parse_video.
        if pts_to_packet:
            dropped_key = next(iter(pts_to_packet))
            pts_to_packet = dict(pts_to_packet)
            del pts_to_packet[dropped_key]
        if return_pts_flags:
            return all_frame_energies, iframe_indices, energies, pts_to_packet, (True, False)
        return all_frame_energies, iframe_indices, energies, pts_to_packet

    monkeypatch.setattr("iris.charon_v._demux_packet_curve", fake_demux)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output_frames, stats = parse_video(REAL_VIDEO, return_stats=True)

    # No exception was raised (this is the regression the bug would have
    # caused: NameError on has_none_pts/has_duplicate_pts).
    assert isinstance(output_frames, list)
    assert stats["total"] > 0

    pts_warnings = [w for w in caught if "CHARON-PTS-001" in str(w.message)]
    assert len(pts_warnings) == 1, f"expected exactly one CHARON-PTS-001 warning, got {caught}"
    assert issubclass(pts_warnings[0].category, RuntimeWarning)
    msg = str(pts_warnings[0].message)
    assert "anomaly=none_pts" in msg
    assert "video_path=" in msg
    assert "expected_packets=" in msg
    assert "actual_decoded_frames=" in msg


def test_parse_video_still_raises_on_genuine_mismatch_without_pts_anomaly(monkeypatch):
    """A frame/packet count mismatch on a WELL-FORMED PTS stream (no
    none/duplicate PTS) is a real bug signal, not an expected malformed-stream
    artifact -- must still raise ValueError, not be downgraded to a warning."""
    if not os.path.exists(REAL_VIDEO):
        pytest.skip("test video not on disk")

    real_demux = _demux_packet_curve

    def fake_demux(video_path, return_pts_flags=False):
        all_frame_energies, iframe_indices, energies, pts_to_packet = real_demux(video_path)
        if pts_to_packet:
            dropped_key = next(iter(pts_to_packet))
            pts_to_packet = dict(pts_to_packet)
            del pts_to_packet[dropped_key]
        if return_pts_flags:
            return all_frame_energies, iframe_indices, energies, pts_to_packet, (False, False)
        return all_frame_energies, iframe_indices, energies, pts_to_packet

    monkeypatch.setattr("iris.charon_v._demux_packet_curve", fake_demux)

    with pytest.raises(ValueError, match="1-to-1 audited mapping failed"):
        parse_video(REAL_VIDEO, return_stats=True)
