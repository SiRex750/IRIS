"""
Pins the direction of the Phase-4 residual-channel swap.

Test A: packet_size spike → action_score spike + is_peak.
Test B: luma_diff_energy spike alone → no peak (field is diagnostic only).
"""
from __future__ import annotations

from iris.action_score import ActionScoreModule


_N = 60
_SPIKE_IDX = _N // 2  # frame 30
_BASELINE = 0.1
_SPIKE = 1.0


def _make_frames(*, packet_size_spike: bool, luma_diff_spike: bool) -> list[dict]:
    frames = []
    for i in range(_N):
        at_spike = i == _SPIKE_IDX
        frames.append({
            "frame_idx": i,
            "packet_size": _SPIKE if (at_spike and packet_size_spike) else _BASELINE,
            "luma_diff_energy": _SPIKE if (at_spike and luma_diff_spike) else _BASELINE,
            "motion_magnitude": _BASELINE,
            "luma_entropy": _BASELINE,
        })
    return frames


def test_packet_size_spike_is_peak():
    """Residual responds to packet_size: spike frame must be peak with max action_score."""
    frames = _make_frames(packet_size_spike=True, luma_diff_spike=False)
    records = ActionScoreModule().score_all(frames)

    peaks = [r for r in records if r["is_peak"]]
    assert peaks, "Expected at least one peak when packet_size spikes"

    max_score_idx = max(range(_N), key=lambda i: records[i]["action_score"])
    assert max_score_idx == _SPIKE_IDX, (
        f"Max action_score at frame {max_score_idx}, expected spike frame {_SPIKE_IDX}"
    )
    assert records[_SPIKE_IDX]["is_peak"], (
        f"Spike frame {_SPIKE_IDX} must be flagged is_peak"
    )


def test_luma_diff_spike_ignored():
    """luma_diff_energy is now diagnostic only: a spike in it must NOT produce a peak."""
    frames = _make_frames(packet_size_spike=False, luma_diff_spike=True)
    records = ActionScoreModule().score_all(frames)

    peaks = [r for r in records if r["is_peak"]]
    assert not peaks, (
        "No peak expected when only luma_diff_energy spikes (diagnostic field, not scored); "
        f"got peaks at frames {[r['frame_idx'] for r in peaks]}"
    )
