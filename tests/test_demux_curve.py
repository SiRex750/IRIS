"""
Tests for _demux_packet_curve added to iris/charon_v.py.

Verifies structural invariants and exact agreement with the Phase-4 sanity
reference curve produced by scripts/phase4_curve_sanity.build_curves.
"""
from __future__ import annotations

import math
import os
import socket
import sys
import urllib.request
from pathlib import Path

import pytest

from iris.charon_v import _demux_packet_curve

# Allow importing scripts as a namespace package.
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.phase4_curve_sanity import build_curves  # noqa: E402


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
def clip() -> str:
    video = _resolve_video()
    if video is None:
        pytest.skip("mov_bbb.mp4 unavailable and download failed")
    return video


@pytest.fixture(scope="module")
def curve(clip):
    return _demux_packet_curve(clip)


@pytest.fixture(scope="module")
def reference(clip):
    return build_curves(clip)


# ── tests ──────────────────────────────────────────────────────────────────

def test_lengths_and_indices(curve):
    all_frame_energies, iframe_indices, energies = curve
    N = len(all_frame_energies)

    assert N > 0, "all_frame_energies must be non-empty"

    # display indices must be exactly 0..N-1 in order
    assert [e[0] for e in all_frame_energies] == list(range(N)), (
        "display_idx values must be exactly 0..N-1 in order"
    )

    # iframe_indices must be a non-empty subset that includes 0
    assert len(iframe_indices) > 0, "iframe_indices must be non-empty"
    assert set(iframe_indices).issubset(range(N)), (
        "iframe_indices must be a subset of valid display indices"
    )
    assert 0 in iframe_indices, "iframe_indices must include display index 0"

    # energies covers exactly the non-keyframe frames
    assert len(energies) == N - len(iframe_indices), (
        f"energies length {len(energies)} != N - len(iframe_indices) "
        f"({N} - {len(iframe_indices)} = {N - len(iframe_indices)})"
    )


def test_matches_step0_reference(curve, reference):
    all_frame_energies, iframe_indices, _ = curve

    # packet_curve from build_curves is a list of (pts, size, is_keyframe)
    ref_packet_curve = reference["packet_curve"]

    # Size sequence must match exactly
    helper_sizes = [e[1] for e in all_frame_energies]
    ref_sizes    = [float(p[1]) for p in ref_packet_curve]
    assert helper_sizes == ref_sizes, (
        f"Packet-size sequence mismatch: first divergence at index "
        f"{next(i for i,(a,b) in enumerate(zip(helper_sizes,ref_sizes)) if a!=b)}"
        if helper_sizes != ref_sizes else ""
    )

    # iframe_indices must match keyframe positions in the reference curve
    ref_iframe_indices = [i for i, p in enumerate(ref_packet_curve) if p[2]]
    assert iframe_indices == ref_iframe_indices, (
        f"iframe_indices mismatch: helper={iframe_indices} ref={ref_iframe_indices}"
    )


def test_nonempty_percentile_pool(curve):
    _, _, energies = curve
    assert len(energies) > 0, "energies (non-keyframe pool) must be non-empty"
    assert all(math.isfinite(v) for v in energies), (
        "all energies values must be finite floats"
    )
