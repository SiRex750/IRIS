"""
Tests for iris/codec_validator.py.

Covers: valid H.264 clip, garbage-file rejection, missing-path rejection,
and assert_valid raising on garbage input.
"""
from __future__ import annotations

import os
import socket
import urllib.request

import pytest

from iris.codec_validator import assert_valid, validate_video


# ── video resolution (mirrors test_charon_v.py pattern) ───────────────────

def _resolve_video() -> str | None:
    local = "mov_bbb.mp4"
    if os.path.exists(local):
        return local
    url = "https://www.w3schools.com/html/mov_bbb.mp4"
    socket.setdefaulttimeout(5.0)
    try:
        urllib.request.urlretrieve(url, local)
        return local
    except Exception:
        return None


@pytest.fixture(scope="module")
def garbage_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("garbage") / "bad.mp4"
    p.write_bytes(b"not a video")
    return str(p)


# ── tests ──────────────────────────────────────────────────────────────────

def test_valid_h264_ok():
    video = _resolve_video()
    if video is None:
        pytest.skip("mov_bbb.mp4 unavailable and download failed")
    result = validate_video(video)
    assert result.status in {"ok", "warn"}, (
        f"Expected ok/warn for a valid H.264 clip, got '{result.status}': {result.reasons}"
    )
    assert result.codec == "h264", f"Expected codec 'h264', got '{result.codec}'"
    assert result.pts_complete is True, "pts_complete must be True for mov_bbb.mp4"
    assert result.keyframe_found is True, "keyframe_found must be True for mov_bbb.mp4"
    assert result.mv_available is True, "mv_available must be True for mov_bbb.mp4"


def test_garbage_file_rejected(garbage_path):
    result = validate_video(garbage_path)
    assert result.status == "reject", (
        f"Expected 'reject' for garbage input, got '{result.status}'"
    )
    assert len(result.reasons) > 0, "reasons must be non-empty for a rejected file"


def test_missing_path_rejected():
    result = validate_video("/no/such/file/does_not_exist.mp4")
    assert result.status == "reject"
    assert len(result.reasons) > 0


def test_assert_valid_raises_on_garbage(garbage_path):
    with pytest.raises(ValueError):
        assert_valid(garbage_path)
