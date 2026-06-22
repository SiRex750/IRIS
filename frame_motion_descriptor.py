"""
FrameMotionDescriptor — carries per-frame motion geometry from Charon-V into L1.

These fields are computed from optical flow / motion vector analysis in charon_v.py.
L1 Elysium uses hessian_max_eigenvalue and motion_entropy in the keep_score eviction formula.

Owner: Track A
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FrameMotionDescriptor:
    """
    Motion geometry snapshot for a single decoded frame.

    All values are raw floats — normalisation happens inside L1ElysiumCache
    when computing keep_score, not here.
    """

    frame_idx:              int
    timestamp_sec:          float

    # Core codec signal (also produced by action_score.py, kept here for L1 access)
    residual_energy:        float = 0.0

    # Motion field geometry — computed from dense optical flow or motion vectors
    divergence:             float = 0.0   # positive = expanding, negative = converging
    curl:                   float = 0.0   # rotation magnitude
    jacobian_frobenius:     float = 0.0   # total motion field intensity
    hessian_max_eigenvalue: float = 0.0   # sharpness of motion boundary (→ hessian_boundary)
    motion_entropy:         float = 0.0   # chaos / unpredictability of motion