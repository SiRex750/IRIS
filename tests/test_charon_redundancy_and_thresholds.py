"""Regression tests for:
- P1-04: repeated packet demux and container opens
- P1-05: adaptive threshold overriding and reporting
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from iris.charon_v import parse_video, compute_motion_geometry
import iris.ingest as ingest_mod
from iris.iris_config import IRISConfig

# Reference zero geometry
GEOM_ZERO = compute_motion_geometry([], 320, 240)

# PyAV mock helpers matching test_charon_full_decode_geometry.py
class _FakeSideDataEntry:
    def __init__(self, src_x, src_y, dst_x, dst_y, motion_x, motion_y):
        self.src_x, self.src_y = src_x, src_y
        self.dst_x, self.dst_y = dst_x, dst_y
        self.motion_x = motion_x * 4
        self.motion_y = motion_y * 4

class _FakeSideData:
    def __init__(self, mvs):
        self._mvs = [_FakeSideDataEntry(*mv) for mv in mvs]
        self.type = types.SimpleNamespace(name="MOTION_VECTORS")
    def __iter__(self):
        return iter(self._mvs)

def _make_frame(frame_idx, tier, mvs, width=320, height=240):
    frame = MagicMock()
    frame.pts = frame_idx
    frame.time = float(frame_idx)
    frame.width = width
    frame.height = height
    frame.key_frame = (tier == "I_FRAME")
    frame.to_ndarray.return_value = np.full((height, width), 128, dtype=np.uint8)
    frame.to_image.return_value = MagicMock()
    frame.pict_type = types.SimpleNamespace(name="I" if tier == "I_FRAME" else "P")
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

# ---------------------------------------------------------------------------
# Test P1-04: Repeated Packet Demux & Container Open
# ---------------------------------------------------------------------------

@patch("iris.ingest.get_clip_embedding_from_pil", lambda pil, device: np.zeros(512, dtype=np.float32))
@patch("iris.ingest.get_semantic_and_clip_caption", lambda pil, frame, emb, device: {"semantic_caption": "stub"})
@patch("iris.codec_validator.validate_video")
def test_ingest_no_redundant_demux_or_fps_calls(mock_validate):
    """
    Verifies that calling ingest() invokes _demux_packet_curve exactly once
    and does not call get_stream_fps at all (as fps is obtained from stats).
    """
    mock_validate.return_value = MagicMock(status="pass", reasons=[])
    frames_spec = [
        ("I_FRAME", []),
        ("CANDIDATE", []),
        ("SKIP", []),
    ]
    mock_frames = [_make_frame(i, tier, mvs) for i, (tier, mvs) in enumerate(frames_spec)]
    pts_to_packet = {0: (10000, True), 1: (1, False), 2: (-1, False)}
    
    all_frame_energies = [(i, abs(pts_to_packet[i][0])) for i in range(3)]
    iframe_indices = [0]
    energies = [abs(pts_to_packet[i][0]) for i in range(3)]

    mock_stream = _make_stream()
    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]
    mock_container.decode.return_value = iter(mock_frames)

    # Wrap the functions we want to spy on with MagicMock canned returns
    # _demux_packet_curve is always called with return_pts_flags=True from
    # parse_video now (item 8: has_none_pts/has_duplicate_pts must be bound
    # in parse_video's scope) -- the 5th element is a synthetic (False, False)
    # since these PTS values are well-formed by construction.
    canned_return = (all_frame_energies, iframe_indices, energies, pts_to_packet, (False, False))
    spy_demux = MagicMock(return_value=canned_return)
    spy_fps = MagicMock(return_value=25.0)

    config = IRISConfig()
    config.candidate_thresh = 0.0
    config.salient_thresh = 0.0
    config.adaptive = False

    import iris.charon_v as charon_mod
    with (
        patch.object(charon_mod, "_demux_packet_curve", spy_demux),
        patch.object(charon_mod, "get_stream_fps", spy_fps),
        patch("iris.charon_v.av.open", return_value=mock_container),
    ):
        idx = ingest_mod.ingest("fake.mp4", config, nms_window=0)

    # 1. Assert _demux_packet_curve is called exactly once (inside parse_video)
    assert spy_demux.call_count == 1, (
        f"_demux_packet_curve called {spy_demux.call_count} times; expected exactly 1 (P1-04)."
    )

    # 2. Assert get_stream_fps is called exactly zero times
    assert spy_fps.call_count == 0, (
        f"get_stream_fps called {spy_fps.call_count} times; expected exactly 0 (P1-04)."
    )

    # 3. Assert index has the expected stats values
    assert idx.video_path == "fake.mp4"
    assert idx.frames_processed == 2  # I_FRAME and CANDIDATE (SKIP is filtered out)

# ---------------------------------------------------------------------------
# Test P1-05: Adaptive Thresholds Clashing
# ---------------------------------------------------------------------------

def _run_parse_video_mocked_for_thresholds(
    frames_spec,
    salient_thresh,
    candidate_thresh,
    adaptive=True,
):
    import iris.charon_v as charon_mod
    mock_frames = [_make_frame(i, tier, []) for i, (tier, _) in enumerate(frames_spec)]
    pts_to_packet = {}
    for i, (tier, _) in enumerate(frames_spec):
        is_kf = (tier == "I_FRAME")
        if tier == "SKIP":
            ps = -1
        elif tier == "I_FRAME":
            ps = 10000
        else:
            ps = 1
        pts_to_packet[i] = (ps, is_kf)

    all_frame_energies = [(i, abs(pts_to_packet[i][0])) for i in range(len(frames_spec))]
    iframe_indices = [i for i, (t, _) in enumerate(frames_spec) if t == "I_FRAME"]
    if not iframe_indices:
        iframe_indices = [0]
    energies = [abs(pts_to_packet[i][0]) for i in range(len(frames_spec))]

    mock_stream = _make_stream()
    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]
    mock_container.decode.return_value = iter(mock_frames)

    with (
        patch.object(charon_mod, "_demux_packet_curve",
                     return_value=(all_frame_energies, iframe_indices, energies, pts_to_packet, (False, False))),
        patch("iris.charon_v.av.open", return_value=mock_container),
    ):
        _, stats = charon_mod.parse_video(
            "fake.mp4",
            return_stats=True,
            adaptive=adaptive,
            salient_thresh=salient_thresh,
            candidate_thresh=candidate_thresh,
        )
    return stats

def test_adaptive_thresholds_non_destructive():
    """
    Verifies that under adaptive=True, the configured thresholds survive
    in the stats dictionary while effective thresholds are correctly applied.
    """
    frames_spec = [
        ("I_FRAME", []),
        ("CANDIDATE", []),
        ("SKIP", []),
    ]
    stats = _run_parse_video_mocked_for_thresholds(
        frames_spec, salient_thresh=0.9, candidate_thresh=0.8, adaptive=True
    )

    # 1. Configured thresholds survive clobbering
    assert stats["configured_salient_thresh"] == 0.9
    assert stats["configured_candidate_thresh"] == 0.8

    # 2. Effective thresholds differ from configured (override took place)
    assert stats["effective_salient_thresh_used"] != 0.9
    assert stats["effective_candidate_thresh_used"] != 0.8

    # 3. Deprecated aliases pointing to effective thresholds exist and match
    assert stats["salient_thresh_used"] == stats["effective_salient_thresh_used"]
    assert stats["candidate_thresh_used"] == stats["effective_candidate_thresh_used"]

def test_non_adaptive_thresholds_match():
    """
    Verifies that under adaptive=False, configured and effective thresholds match.
    """
    frames_spec = [
        ("I_FRAME", []),
        ("CANDIDATE", []),
    ]
    stats = _run_parse_video_mocked_for_thresholds(
        frames_spec, salient_thresh=0.9, candidate_thresh=0.8, adaptive=False
    )

    assert stats["configured_salient_thresh"] == 0.9
    assert stats["configured_candidate_thresh"] == 0.8
    assert stats["effective_salient_thresh_used"] == 0.9
    assert stats["effective_candidate_thresh_used"] == 0.8
    assert stats["salient_thresh_used"] == 0.9
    assert stats["candidate_thresh_used"] == 0.8
