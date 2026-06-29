"""
Phase-4 codec validator.

Validates a video's preconditions (pts completeness, keyframe anchor, MV
availability) before the demux-first gate operates on it. Bounded — never
decodes the entire file. Pure validation; no side effects.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import av


@dataclass
class ValidationResult:
    status: str           # "ok" | "warn" | "reject"
    codec: str | None
    reasons: list[str]    # human-readable findings (warn + reject mixed)
    mv_available: bool
    pts_complete: bool
    keyframe_found: bool


_SUPPORTED_CODECS = {"h264", "hevc"}
_MV_PROBE_CAP = 60


def validate_video(video_path: str, max_probe_packets: int = 240) -> ValidationResult:
    """
    Probe *video_path* and return a ValidationResult.

    The function is bounded: the demux probe stops at *max_probe_packets*
    non-flush packets and the decode probe stops at the first non-keyframe
    frame (or _MV_PROBE_CAP frames, whichever comes first).
    """
    reasons: list[str] = []
    has_reject = False
    has_warn = False

    def _reject(msg: str) -> None:
        nonlocal has_reject
        has_reject = True
        reasons.append(msg)

    def _warn(msg: str) -> None:
        nonlocal has_warn
        has_warn = True
        reasons.append(msg)

    codec: str | None = None
    mv_available = False
    pts_complete = False
    keyframe_found = False

    # ── Step 1: open container ─────────────────────────────────────────────
    if not os.path.exists(video_path):
        _reject(f"cannot open container: path does not exist: {video_path}")
        return ValidationResult(
            status="reject", codec=None, reasons=reasons,
            mv_available=False, pts_complete=False, keyframe_found=False,
        )

    try:
        container = av.open(video_path)
    except Exception as exc:
        _reject(f"cannot open container: {exc}")
        return ValidationResult(
            status="reject", codec=None, reasons=reasons,
            mv_available=False, pts_complete=False, keyframe_found=False,
        )

    try:
        # ── Step 2: check video stream ────────────────────────────────────
        if not container.streams.video:
            _reject("no video stream")
            return ValidationResult(
                status="reject", codec=None, reasons=reasons,
                mv_available=False, pts_complete=False, keyframe_found=False,
            )

        stream = container.streams.video[0]

        # ── Step 3: codec check ───────────────────────────────────────────
        codec = getattr(stream.codec_context, "name", None)
        if codec not in _SUPPORTED_CODECS:
            _warn(
                f"codec '{codec}' is not h264/hevc; "
                "motion-vector export may be unavailable"
            )

        # ── Step 4: PTS + keyframe probe (zero decode) ────────────────────
        pts_complete = True
        packets_examined = 0
        for pkt in container.demux(stream):
            if pkt.size == 0:   # flush packet
                continue
            if pkt.pts is None:
                pts_complete = False
            if pkt.is_keyframe:
                keyframe_found = True
            packets_examined += 1
            if packets_examined >= max_probe_packets:
                break

        if not pts_complete:
            _reject("packet pts missing; display-order re-sort impossible")
        if not keyframe_found:
            _reject(
                f"no keyframe anchor in first {max_probe_packets} packets"
            )

    finally:
        container.close()

    # ── Step 5: MV-availability probe (bounded decode) ────────────────────
    try:
        container2 = av.open(video_path)
        try:
            stream2 = container2.streams.video[0]
            stream2.codec_context.options = {"flags2": "+export_mvs"}

            frames_decoded = 0
            mv_probe_done = False
            for frame in container2.decode(video=0):
                frames_decoded += 1
                if not frame.key_frame:
                    # First P/B frame — check for motion-vector side data.
                    mv_available = False
                    try:
                        for sd in frame.side_data:
                            if getattr(sd.type, "name", None) == "MOTION_VECTORS":
                                mv_available = True
                                break
                    except (AttributeError, TypeError):
                        mv_available = False
                    mv_probe_done = True
                    break
                if frames_decoded >= _MV_PROBE_CAP:
                    break

            if not mv_probe_done:
                _warn(
                    f"no non-keyframe found in first {_MV_PROBE_CAP} frames; "
                    "MV availability unknown"
                )
            elif not mv_available:
                _warn(
                    "motion vectors unavailable; motion geometry will be zero"
                )
        finally:
            container2.close()
    except Exception as exc:
        mv_available = False
        _warn(f"MV probe failed: {exc}; motion geometry will be zero")

    # ── Step 6: final status ───────────────────────────────────────────────
    if has_reject:
        status = "reject"
    elif has_warn:
        status = "warn"
    else:
        status = "ok"

    return ValidationResult(
        status=status,
        codec=codec,
        reasons=reasons,
        mv_available=mv_available,
        pts_complete=pts_complete,
        keyframe_found=keyframe_found,
    )


def assert_valid(video_path: str) -> ValidationResult:
    """Raise ValueError if *video_path* fails validation; otherwise return the result."""
    result = validate_video(video_path)
    if result.status == "reject":
        raise ValueError(result.reasons)
    return result
