"""
Tests for the Phase-4 curve sanity tool.

Verifies that:
  - pts-sorted packet keyframe flags fully agree with decoded frame.key_frame flags
  - Spearman rho between non-I packet sizes and non-I luma-diffs exceeds 0.40
  - The 95th-percentile survivor fraction is sparse but non-zero: (0.0, 0.5)
"""
from __future__ import annotations

import os
import socket
import sys
import urllib.request
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so `scripts` is importable as a namespace package.
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.phase4_curve_sanity import build_curves  # noqa: E402


# ── video resolution (same pattern as test_charon_v.py) ───────────────────

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
def curves():
    video = _resolve_video()
    if video is None:
        pytest.skip("mov_bbb.mp4 unavailable and download failed")
    return build_curves(video)


# ── assertions ─────────────────────────────────────────────────────────────

def test_keyframe_alignment_full(curves):
    """pts-sorted packet order must perfectly reproduce decoded display order."""
    assert curves["alignment_agree"] == curves["alignment_total"], (
        f"Keyframe flag mismatch: only {curves['alignment_agree']} of "
        f"{curves['alignment_total']} positions agree"
    )


def test_spearman_rho_non_i(curves):
    """Non-I Spearman rho must be > 0.40 (packet size and luma-diff are correlated)."""
    rho = curves["spearman_rho"]
    assert rho > 0.40, f"Spearman rho (non-I) = {rho:.4f}, expected > 0.40"


def test_p95_survivor_fraction(curves):
    """95th-percentile survivor fraction must be sparse but non-zero."""
    frac = curves["frac_above_p95"]
    assert 0.0 < frac < 0.5, (
        f"95th-pct survivor fraction = {frac:.4f}, expected in (0.0, 0.5)"
    )
