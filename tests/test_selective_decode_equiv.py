"""
Oracle A/B equivalence test: selective decode must produce identical survivor
sets and per-frame features compared to full (legacy) decode.

full_decode=True  → expensive pixel work on every frame (legacy behaviour)
full_decode=False → expensive pixel work only on non-SKIP survivors (default)
"""
from __future__ import annotations

import os
import socket
import urllib.request

import pytest

from iris.charon_v import parse_video

_GEOM_KEYS = (
    "divergence",
    "curl",
    "jacobian_frobenius",
    "hessian_max_eigenvalue",
    "motion_entropy",
)


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
def both():
    video = _resolve_video()
    if video is None:
        pytest.skip("mov_bbb.mp4 unavailable and download failed")
    out_full, stats_full, raw_full = parse_video(
        video, return_stats=True, return_raw=True, full_decode=True
    )
    out_sel, stats_sel, raw_sel = parse_video(
        video, return_stats=True, return_raw=True, full_decode=False
    )
    return out_full, stats_full, raw_full, out_sel, stats_sel, raw_sel


def test_survivor_sets_equal(both):
    out_full, _, _, out_sel, _, _ = both
    ids_full = [(f["frame_idx"], f["tier"]) for f in out_full]
    ids_sel  = [(f["frame_idx"], f["tier"]) for f in out_sel]
    assert ids_sel == ids_full, (
        f"Survivor (frame_idx, tier) lists differ — "
        f"first 5 full: {ids_full[:5]}, sel: {ids_sel[:5]}"
    )


def test_survivor_fields_equal(both):
    out_full, _, raw_full, out_sel, _, raw_sel = both
    for f_full, f_sel in zip(out_full, out_sel):
        idx = f_full["frame_idx"]
        r_full = raw_full[idx]
        r_sel  = raw_sel[idx]

        assert f_sel["packet_size"] == f_full["packet_size"], (
            f"frame {idx}: packet_size mismatch "
            f"(full={f_full['packet_size']}, sel={f_sel['packet_size']})"
        )
        assert r_sel["motion_magnitude"] == r_full["motion_magnitude"], (
            f"frame {idx}: motion_magnitude mismatch"
        )
        assert r_sel["luma_entropy"] == r_full["luma_entropy"], (
            f"frame {idx}: luma_entropy mismatch"
        )
        for k in _GEOM_KEYS:
            assert f_sel.get(k) == f_full.get(k), (
                f"frame {idx}: geometry field '{k}' mismatch "
                f"(full={f_full.get(k)}, sel={f_sel.get(k)})"
            )
        # luma_diff_energy is allowed to differ (gap-diff vs adjacent-diff)
        # but must be present in both
        assert "luma_diff_energy" in f_full, \
            f"frame {idx}: luma_diff_energy missing from full output"
        assert "luma_diff_energy" in f_sel, \
            f"frame {idx}: luma_diff_energy missing from sel output"


def test_raw_record_density(both):
    _, stats_full, raw_full, _, stats_sel, raw_sel = both
    assert len(raw_full) == stats_full["total"], (
        f"raw_full length {len(raw_full)} != total {stats_full['total']}"
    )
    assert len(raw_sel) == stats_sel["total"], (
        f"raw_sel length {len(raw_sel)} != total {stats_sel['total']}"
    )
    assert len(raw_full) == len(raw_sel), (
        f"raw_full ({len(raw_full)}) and raw_sel ({len(raw_sel)}) length mismatch"
    )


def test_raw_packet_size_matches(both):
    _, _, raw_full, _, _, raw_sel = both
    for i, (r_full, r_sel) in enumerate(zip(raw_full, raw_sel)):
        assert r_sel["packet_size"] == r_full["packet_size"], (
            f"raw_record[{i}]: packet_size mismatch "
            f"(full={r_full['packet_size']}, sel={r_sel['packet_size']})"
        )


def test_savings(both):
    _, stats_full, _, out_sel, stats_sel, _ = both
    assert stats_sel["frames_expensive_processed"] == len(out_sel), (
        f"selective: frames_expensive_processed ({stats_sel['frames_expensive_processed']}) "
        f"should equal survivor count ({len(out_sel)})"
    )
    assert stats_sel["frames_expensive_processed"] < stats_sel["total"], (
        f"selective: expected some SKIP frames but "
        f"frames_expensive_processed == total == {stats_sel['total']}"
    )
    assert stats_full["frames_expensive_processed"] == stats_full["total"], (
        f"full_decode: frames_expensive_processed ({stats_full['frames_expensive_processed']}) "
        f"should equal total ({stats_full['total']})"
    )
