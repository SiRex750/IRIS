"""
CachedFrame — one entry in L1 Elysium.

Holds everything L1 needs to:
  1. Decide whether to keep or evict this frame (via keep_score)
  2. Return the frame's data to the pipeline at query time

Owner: Track A
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frame_motion_descriptor import FrameMotionDescriptor


@dataclass
class CachedFrame:
    """
    A single frame held inside L1 Elysium.

    Fields set once at admission:
        frame_idx, timestamp_sec, action_score, persistence_value,
        is_peak, motion, admitted_at

    Fields updated after admission:
        query_similarity  <- updated when a user query arrives
        pagerank          <- updated when L2 graph sends scores
        embedding         <- filled when the VLM encoder runs
    """

    # --- Identity ---
    frame_idx:         int
    timestamp_sec:     float

    # --- From action_score.py ---
    action_score:      float   # continuous 0.0 -> 1.0 importance
    persistence_value: float   # how prominent this peak is (0.0 -> 1.0)
    is_peak:           bool    # True if persistence_value >= threshold

    # --- From charon_v.py via FrameMotionDescriptor ---
    motion: FrameMotionDescriptor

    # --- Set by L1 at admission time ---
    # Monotonically increasing counter — used to compute recency.
    # Frame admitted 10th gets admitted_at=10. Frame admitted 50th gets 50.
    # Newer frames have higher numbers = higher recency.
    admitted_at: int = 0

    # --- Updated after admission ---
    query_similarity: float = 0.0  # cosine sim vs query embedding (0.0 -> 1.0)
    pagerank:         float = 0.0  # score from L2 graph (0.0 -> 1.0)

    # --- Visual embedding ---
    # None until the VLM encoder processes this frame.
    # When filled: numpy array of shape (D,), dtype=bfloat16
    embedding: np.ndarray | None = None

    def keep_score(
        self,
        total_admitted: int,
        w_action:   float = 0.30,
        w_query:    float = 0.20,
        w_persist:  float = 0.15,
        w_pagerank: float = 0.10,
        w_entropy:  float = 0.10,
        w_hessian:  float = 0.10,
        w_recency:  float = 0.05,
    ) -> float:
        """
        How much L1 wants to keep this frame.

        Higher score = keep it.
        Lower score = evict it first.

        Args:
            total_admitted: how many frames L1 has admitted in total so far.
                            Used to compute how recent this frame is.
        """
        # Recency: 1.0 if just admitted, decays toward 0.0 as newer frames arrive
        recency = 1.0 - (total_admitted - self.admitted_at) / max(total_admitted, 1)
        recency = max(0.0, recency)

        return (
            w_action   * self.action_score
            + w_query  * self.query_similarity
            + w_persist * self.persistence_value
            + w_pagerank * self.pagerank
            + w_entropy  * self.motion.motion_entropy
            + w_hessian  * self.motion.hessian_max_eigenvalue
            + w_recency  * recency
        )