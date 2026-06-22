"""
L1 Elysium — codec-pressure-aware active context cache for IRIS.

Replaces the text-based HADES cache with a video-native implementation.
Eviction is driven by a composite keep_score that combines:
    action_score, query_similarity, persistence, pagerank,
    motion_entropy, hessian_boundary, and recency.

Capacity and eviction weights come from IRISConfig so that
GEPA can tune them between pipeline runs without code changes.

Owner: Track A
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from cached_frame import CachedFrame
from iris_config import IRISConfig


class L1ElysiumCache:
    """
    Active context cache for IRIS.

    Holds the most relevant CachedFrame entries for the current
    pipeline state. When full, the frame with the lowest keep_score
    is evicted to make room for the incoming frame.

    Lifecycle of one frame inside L1:
        1. admit()            frame enters L1 with action_score + motion data
        2. update_pagerank()  L2 pushes PageRank scores back into L1 frames
        3. query()            query embedding arrives, similarities computed,
                              top-k frames returned to Aria
        4. _evict_one()       called automatically by admit() when over capacity

    Basic usage:
        cache = L1ElysiumCache()
        cache.admit(cached_frame)
        results = cache.query(query_embedding, top_k=5)
    """

    def __init__(self, config: IRISConfig | None = None) -> None:
        self._config: IRISConfig = config or IRISConfig()
        self._frames: dict[int, CachedFrame] = {}
        self._admission_counter: int = 0

    # ── Public interface ───────────────────────────────────────────────────

    def admit(self, frame: CachedFrame) -> None:
        if frame.frame_idx in self._frames:
            self._frames[frame.frame_idx] = frame
            return

        if self.is_full:
            self._evict_one()

        frame.admitted_at = self._admission_counter
        self._frames[frame.frame_idx] = frame
        self._admission_counter += 1

    def query(
    self,
    query_embedding: np.ndarray,
    top_k: int = 5,
    ) -> list[CachedFrame]:
        if not self._frames:
            return []

        # Compute query norm once — reused for every frame's similarity.
        # Cast to float32 because bfloat16 lacks full numpy support.
        q = query_embedding.astype(np.float32)
        q_norm = float(np.linalg.norm(q))

        for frame in self._frames.values():

            # Frame not yet encoded — no embedding to compare against.
            if frame.embedding is None:
                frame.query_similarity = 0.0
                continue

            # Cast frame embedding to float32 for computation.
            f = frame.embedding.astype(np.float32)
            f_norm = float(np.linalg.norm(f))

            # Guard against zero vectors — dot / 0 is undefined.
            if q_norm < 1e-8 or f_norm < 1e-8:
                frame.query_similarity = 0.0
                continue

            # Cosine similarity, clipped to [0, 1].
            sim = float(np.dot(q, f) / (q_norm * f_norm))
            frame.query_similarity = max(0.0, min(1.0, sim))

        # Re-sort by keep_score now that query_similarity is updated.
        # reverse=True because higher score = more relevant = comes first.
        sorted_frames = sorted(
            self._frames.values(),
            key=self._keep_score,
            reverse=True,
        )

        return sorted_frames[:top_k]
    
    def update_pagerank(self, scores: dict[int, float]) -> None:
        for frame_idx, score in scores.items():
            if frame_idx in self._frames:
                self._frames[frame_idx].pagerank = float(score)
        # frame_idx not in L1 — silently ignore, L2 may score frames
        # that were already evicted

    # ── Private helpers ────────────────────────────────────────────────────

    def _evict_one(self) -> None:
        if not self._frames:
            return

        victim_idx = min(
            self._frames,
            key=lambda idx: self._keep_score(self._frames[idx])
        )
        del self._frames[victim_idx]

    def _keep_score(self, frame: CachedFrame) -> float:
        return frame.keep_score(
            total_admitted=self._admission_counter,
            w_action=self._config.l1_w_action,
            w_query=self._config.l1_w_query,
            w_persist=self._config.l1_w_persist,
            w_pagerank=self._config.l1_w_pagerank,
            w_entropy=self._config.l1_w_entropy,
            w_hessian=self._config.l1_w_hessian,
            w_recency=self._config.l1_w_recency,
        )

    # ── Utility ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._frames)

    def __contains__(self, frame_idx: int) -> bool:
        return frame_idx in self._frames

    def frames(self) -> Iterator[CachedFrame]:
        return iter(self._frames.values())

    @property
    def is_full(self) -> bool:
        return len(self._frames) >= self._config.l1_capacity

    # ── Shared fact-text helper ────────────────────────────────────────────

    @staticmethod
    def _frame_to_fact_text(frame: "CachedFrame") -> str:
        """Single source of truth for how a CachedFrame is expressed as text.

        Used by both as_context_text() and set_facts so the format is never
        duplicated and can never drift out of sync between them.
        """
        base = f"Frame {frame.frame_idx} at {frame.timestamp_sec:.2f}s"
        if getattr(frame, "caption", None):
            return (
                f"{base}: {frame.caption}. "
                f"(action score {frame.action_score:.4f}, persistence {frame.persistence_value:.4f})"
            )
        return (
            f"{base} depicts action score {frame.action_score:.4f}, "
            f"persistence {frame.persistence_value:.4f}."
        )

    def as_context_text(self) -> str:
        lines = ["Fact Cache:"]
        for frame in self._frames.values():
            lines.append(self._frame_to_fact_text(frame))
        return "\n".join(lines)

    @property
    def set_facts(self) -> dict:
        class MockFactEntry:
            def __init__(self, text: str):
                self.text = text

        facts = {}
        for frame in self._frames.values():
            text = self._frame_to_fact_text(frame)
            facts[text] = MockFactEntry(text)
        return facts