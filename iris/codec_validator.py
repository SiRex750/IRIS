"""
Phase-4 codec validator.

Validates a video's preconditions (pts completeness, keyframe anchor, MV
availability) before the demux-first gate operates on it. Supporting both
fast prefix checks and strict complete-file decode validations.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
import av


@dataclass
class ValidationResult:
    status: str                         # "ok" | "warn" | "reject"
    severity: str                       # "none" | "low" | "high"
    reasons: list[str]                  # Reject reasons
    warnings: list[str]                 # Warning reasons
    codec: str | None
    container: str | None
    inspected_packet_count: int
    inspected_frame_count: int
    validation_level: str               # "fast" | "strict"
    complete_stream_checked: bool
    mv_available: bool
    pts_complete: bool
    keyframe_found: bool


_SUPPORTED_CODECS = {"h264", "hevc"}
_MV_PROBE_CAP = 60


def validate_video(
    video_path: str,
    level: str = "fast",
    max_probe_packets: int = 240,
) -> ValidationResult:
    """
    Probe *video_path* and return a ValidationResult.

    - "fast": performs header/prefix checks (bounds packet checks to max_probe_packets).
    - "strict": demuxes and decodes the complete stream to ensure structural integrity.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    has_reject = False
    has_warn = False

    def _reject(msg: str) -> None:
        nonlocal has_reject
        has_reject = True
        reasons.append(msg)

    def _warn(msg: str) -> None:
        nonlocal has_warn
        has_warn = True
        warnings.append(msg)

    codec: str | None = None
    container_format: str | None = None
    mv_available = False
    pts_complete = True
    keyframe_found = False
    inspected_packet_count = 0
    inspected_frame_count = 0

    # ── Step 1: open container ─────────────────────────────────────────────
    if not os.path.exists(video_path):
        _reject(f"cannot open container: path does not exist: {video_path}")
        return ValidationResult(
            status="reject", severity="high", reasons=reasons, warnings=warnings,
            codec=None, container=None, inspected_packet_count=0, inspected_frame_count=0,
            validation_level=level, complete_stream_checked=False,
            mv_available=False, pts_complete=False, keyframe_found=False,
        )

    try:
        container = av.open(video_path)
        container_format = container.format.name
    except Exception as exc:
        _reject(f"cannot open container: {exc}")
        return ValidationResult(
            status="reject", severity="high", reasons=reasons, warnings=warnings,
            codec=None, container=None, inspected_packet_count=0, inspected_frame_count=0,
            validation_level=level, complete_stream_checked=False,
            mv_available=False, pts_complete=False, keyframe_found=False,
        )

    try:
        # ── Step 2: check video stream ────────────────────────────────────
        if not container.streams.video:
            _reject("no video stream")
            return ValidationResult(
                status="reject", severity="high", reasons=reasons, warnings=warnings,
                codec=None, container=container_format, inspected_packet_count=0, inspected_frame_count=0,
                validation_level=level, complete_stream_checked=False,
                mv_available=False, pts_complete=False, keyframe_found=False,
            )

        stream = container.streams.video[0]
        codec = getattr(stream.codec_context, "name", None)

        if codec not in _SUPPORTED_CODECS:
            _warn(f"codec '{codec}' is not h264/hevc; motion-vector export may be unavailable")

        # Check duration and FPS metadata (non-finite or <= 0 check)
        duration = getattr(stream, "duration", None)
        fps = getattr(stream, "average_rate", None)
        if duration is not None and duration <= 0:
            _warn("Video duration metadata is zero or negative")
        if fps is not None and float(fps) <= 0.0:
            _reject("Video average framerate is zero or negative")

        # ── Step 3: Fast Prefix or Strict Full Validation ────────────────
        pts_seen: set[int] = set()

        if level == "fast":
            # Bounded prefix validation
            for pkt in container.demux(stream):
                if pkt.size == 0:
                    continue
                if pkt.pts is None:
                    pts_complete = False
                else:
                    if pkt.pts in pts_seen:
                        _reject(f"duplicate PTS {pkt.pts} found in prefix")
                    pts_seen.add(pkt.pts)
                if pkt.is_keyframe:
                    keyframe_found = True
                inspected_packet_count += 1
                if inspected_packet_count >= max_probe_packets:
                    break

            if not pts_complete:
                _reject("packet pts missing in prefix")
            if not keyframe_found:
                _reject(f"no keyframe anchor in first {max_probe_packets} packets")

        else:
            # "strict": complete stream validation
            # Track decodability
            stream.codec_context.options = {"flags2": "+export_mvs"}
            try:
                for pkt in container.demux(stream):
                    if pkt.size == 0:
                        continue
                    if pkt.pts is None:
                        pts_complete = False
                    else:
                        if pkt.pts in pts_seen:
                            _reject(f"duplicate PTS {pkt.pts} found in stream")
                        pts_seen.add(pkt.pts)
                    if pkt.is_keyframe:
                        keyframe_found = True
                    inspected_packet_count += 1

                    # Decode packets sequentially to verify stream structural integrity
                    for frame in pkt.decode():
                        inspected_frame_count += 1
                        if not frame.key_frame:
                            # Verify if MVs are present in any P/B frame
                            try:
                                for sd in frame.side_data:
                                    if getattr(sd.type, "name", None) == "MOTION_VECTORS":
                                        mv_available = True
                                        break
                            except Exception:
                                pass
            except Exception as exc:
                _reject(f"strict validation stream decode/read error: {exc}")

            if inspected_packet_count == 0:
                _reject("empty or truncated video stream (zero packets)")
            if not pts_complete:
                _reject("packet pts missing in stream")
            if not keyframe_found:
                _reject("no keyframe found in entire video stream")

    finally:
        container.close()

    # If fast mode, run a bounded decoder to check motion vector availability
    if level == "fast" and not has_reject:
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
                        mv_available = False
                        try:
                            for sd in frame.side_data:
                                if getattr(sd.type, "name", None) == "MOTION_VECTORS":
                                    mv_available = True
                                    break
                        except Exception:
                            mv_available = False
                        mv_probe_done = True
                        break
                    if frames_decoded >= _MV_PROBE_CAP:
                        break
                inspected_frame_count = frames_decoded

                if not mv_probe_done:
                    _warn(f"no non-keyframe found in prefix; MV availability unknown")
                elif not mv_available:
                    _warn("motion vectors unavailable; motion geometry will be zero")
            finally:
                container2.close()
        except Exception as exc:
            mv_available = False
            _warn(f"MV probe failed: {exc}")

    # Determine final status and severity
    if has_reject:
        status = "reject"
        severity = "high"
    elif has_warn:
        status = "warn"
        severity = "low"
    else:
        status = "ok"
        severity = "none"

    return ValidationResult(
        status=status,
        severity=severity,
        reasons=reasons,
        warnings=warnings,
        codec=codec,
        container=container_format,
        inspected_packet_count=inspected_packet_count,
        inspected_frame_count=inspected_frame_count,
        validation_level=level,
        complete_stream_checked=(level == "strict"),
        mv_available=mv_available,
        pts_complete=pts_complete,
        keyframe_found=keyframe_found,
    )


def assert_valid(video_path: str, level: str = "fast") -> ValidationResult:
    """Raise ValueError if *video_path* fails validation; otherwise return the result."""
    result = validate_video(video_path, level=level)
    if result.status == "reject":
        raise ValueError(", ".join(result.reasons))
    return result
