"""
Proves parse_video's SALIENT/CANDIDATE gate has switched to packet-size signal.

Checks that:
  - adaptive thresholds are byte-scale (> 1.0), not luma [0, 1]
  - every output frame carries a "packet_size" key
  - SALIENT survivors have packet_size strictly above the salient threshold
  - CANDIDATE survivors have packet_size >= the candidate threshold
  - the overall survivor fraction is sane (non-zero, below 50 %)
"""
from __future__ import annotations

import os
import socket
import urllib.request

import pytest

from iris.charon_v import parse_video, PEAK_WINDOW_SECONDS


def _get_scene_thresh(stats: dict, frame_idx: int) -> tuple[float, float]:
    """Return (salient, candidate) threshold for the scene whose [start,end) contains frame_idx."""
    for (start, end), salient in stats["salient_thresh_per_scene"].items():
        if start <= frame_idx < end:
            return salient, stats["candidate_thresh_per_scene"][(start, end)]
    raise KeyError(f"frame_idx {frame_idx} not found in any scene range")


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
def parsed():
    video = _resolve_video()
    if video is None:
        pytest.skip("mov_bbb.mp4 unavailable and download failed")
    output, stats = parse_video(video, return_stats=True)
    return output, stats


def test_thresholds_are_byte_scale(parsed):
    _, stats = parsed
    assert stats["salient_thresh_used"] > 1.0, (
        f"salient_thresh_used={stats['salient_thresh_used']:.4f} is not byte-scale"
    )
    assert stats["candidate_thresh_used"] > 1.0, (
        f"candidate_thresh_used={stats['candidate_thresh_used']:.4f} is not byte-scale"
    )
    assert stats["salient_thresh_used"] > stats["candidate_thresh_used"], (
        "salient threshold must exceed candidate threshold"
    )


def test_all_frames_have_packet_size(parsed):
    output, _ = parsed
    missing = [f["frame_idx"] for f in output if "packet_size" not in f]
    assert not missing, f"frames missing 'packet_size' key: {missing[:5]}"


def test_salient_survivors_above_threshold(parsed):
    output, stats = parsed
    violations = []
    for f in output:
        if f["tier"] == "SALIENT":
            scene_salient, _ = _get_scene_thresh(stats, f["frame_idx"])
            if f["packet_size"] <= scene_salient:
                violations.append((f["frame_idx"], f["packet_size"], scene_salient))
    assert not violations, (
        f"{len(violations)} SALIENT frames have packet_size <= their scene salient_thresh: "
        f"first=(frame={violations[0][0]}, ps={violations[0][1]:.1f}, thresh={violations[0][2]:.1f})"
        if violations else ""
    )


def test_candidate_survivors_at_or_above_threshold(parsed):
    output, stats = parsed
    violations = []
    for f in output:
        if f["tier"] == "CANDIDATE":
            scene_salient, scene_candidate = _get_scene_thresh(stats, f["frame_idx"])
            if f["packet_size"] < scene_candidate:
                violations.append(("below_candidate", f["frame_idx"], f["packet_size"], scene_candidate))
            elif f["packet_size"] > scene_salient:
                violations.append(("above_salient", f["frame_idx"], f["packet_size"], scene_salient))
    assert not violations, (
        f"{len(violations)} CANDIDATE frames outside their scene [candidate, salient] band: "
        f"first={violations[0]}"
        if violations else ""
    )


def test_survivor_fraction_is_sane(parsed):
    output, stats = parsed
    total = stats["total"]
    frac = len(output) / total
    assert 0.0 < frac < 0.5, (
        f"Survivor fraction {frac:.4f} is outside (0.0, 0.5) — "
        f"{len(output)} output frames out of {total} total"
    )


def test_peak_order_derived(parsed):
    _, stats = parsed
    order = stats["peak_order_used"]
    assert order >= 3, f"peak_order_used must be >= 3, got {order}"
    # For mov_bbb.mp4 (≈24fps), max(3, round(PEAK_WINDOW_SECONDS * 24)) = 12, not 15.
    # 15 would only be correct for an exact 30fps clip; verify we're not hardcoding it.
    assert order != 15, (
        f"peak_order_used={order} matches the old hardcoded literal; "
        f"expected max(3, round({PEAK_WINDOW_SECONDS}*fps)) ≈ 12 for a 24fps clip"
    )


def test_per_scene_stats_present(parsed):
    _, stats = parsed
    assert "num_scenes" in stats and stats["num_scenes"] >= 1, (
        f"num_scenes missing or zero: {stats.get('num_scenes')}"
    )
    assert "salient_thresh_per_scene" in stats, "salient_thresh_per_scene missing from stats"
    assert "candidate_thresh_per_scene" in stats, "candidate_thresh_per_scene missing from stats"
    assert len(stats["salient_thresh_per_scene"]) == stats["num_scenes"], (
        f"salient_thresh_per_scene length {len(stats['salient_thresh_per_scene'])} "
        f"!= num_scenes {stats['num_scenes']}"
    )
    assert len(stats["candidate_thresh_per_scene"]) == stats["num_scenes"], (
        f"candidate_thresh_per_scene length {len(stats['candidate_thresh_per_scene'])} "
        f"!= num_scenes {stats['num_scenes']}"
    )


def test_global_thresh_is_median_not_min(parsed):
    import numpy as np
    _, stats = parsed
    per_scene_vals = list(stats["salient_thresh_per_scene"].values())
    assert stats["salient_thresh_used"] >= min(per_scene_vals), (
        "salient_thresh_used must be >= min of per-scene values (median >= min)"
    )
    if stats["num_scenes"] >= 3:
        assert stats["salient_thresh_used"] == pytest.approx(float(np.median(per_scene_vals))), (
            f"salient_thresh_used ({stats['salient_thresh_used']}) should equal "
            f"median of per-scene values ({float(np.median(per_scene_vals))}), "
            f"not min ({min(per_scene_vals)})"
        )
