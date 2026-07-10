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

from iris.cached_frame import CachedFrame
from iris.iris_config import IRISConfig


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
        self.hits: int = 0
        self.misses: int = 0

    # ── Public interface ───────────────────────────────────────────────────

    def admit(self, frame: CachedFrame) -> None:
        if frame.frame_idx in self._frames:
            self.hits += 1
            self._frames[frame.frame_idx] = frame
            return

        self.misses += 1
        if self.is_full:
            self._evict_one()

        frame.admitted_at = self._admission_counter
        self._frames[frame.frame_idx] = frame
        self._admission_counter += 1

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        query_motion_embedding: np.ndarray | None = None,
    ) -> list[CachedFrame]:
        """Rank cached frames by similarity to the query and return top-k.

        Supports dual-space retrieval (Contribution 3):
            total_sim = visual_weight × visual_sim + motion_weight × motion_sim

        When *query_motion_embedding* is ``None``, pure visual retrieval is
        used (backwards compatible with all existing call-sites).

        Args:
            query_embedding: Visual query vector (any dtype, cast to float32).
            top_k: Maximum number of frames to return.
            query_motion_embedding: Optional 6-D motion query vector.  When
                provided, dual-space retrieval is activated.
        """
        if not self._frames:
            return []

        # --- visual query norm (computed once) ---
        q = query_embedding.astype(np.float32)
        q_norm = float(np.linalg.norm(q))

        # --- motion query norm (computed once, if provided) ---
        qm: np.ndarray | None = None
        qm_norm: float = 0.0
        use_dual = query_motion_embedding is not None
        if use_dual:
            qm = query_motion_embedding.astype(np.float32)
            qm_norm = float(np.linalg.norm(qm))

        w_visual = self._config.l1_visual_query_weight
        w_motion = self._config.l1_motion_query_weight

        for frame in self._frames.values():
            # --- visual similarity ---
            visual_sim = 0.0
            if frame.embedding is not None:
                f = frame.embedding.astype(np.float32)
                f_norm = float(np.linalg.norm(f))
                if q_norm >= 1e-8 and f_norm >= 1e-8:
                    visual_sim = max(0.0, min(1.0, float(np.dot(q, f) / (q_norm * f_norm))))

            # --- motion similarity (Contribution 3) ---
            motion_sim = 0.0
            if use_dual and frame.motion_embedding is not None and qm is not None:
                fm = frame.motion_embedding.astype(np.float32)
                fm_norm = float(np.linalg.norm(fm))
                if qm_norm >= 1e-8 and fm_norm >= 1e-8:
                    motion_sim = max(0.0, min(1.0, float(np.dot(qm, fm) / (qm_norm * fm_norm))))

            # --- combined similarity ---
            if use_dual:
                frame.query_similarity = w_visual * visual_sim + w_motion * motion_sim
            else:
                frame.query_similarity = visual_sim

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

    # ── Fact-text helpers ──────────────────────────────────────────────────
    #
    # Two separate outputs serve two different consumers:
    #
    #   _frame_to_display_text()  Full human-readable string including numeric
    #                             pipeline metrics (action_score, persistence).
    #                             Used by as_context_text() → ARIA's prompt.
    #                             ARIA can legitimately reference these numbers
    #                             in prose ("this frame had a high action score").
    #
    #   _frame_to_nli_fact()      Semantic content ONLY — no numbers, no metric
    #                             names.  Used by set_facts → Cerberus-V's fact
    #                             pool.  An NLI model can only meaningfully reason
    #                             about natural-language scene descriptions; numeric
    #                             metadata phrased as sentences invites false
    #                             contradiction verdicts (Fix 10/11 root cause).
    #                             Returns None for frames without a caption — a
    #                             bare numeric string is not a real fact and should
    #                             not enter the NLI pool at all.

    @staticmethod
    def _frame_to_display_text(frame: "CachedFrame") -> str:
        """Full display string for a frame, including numeric pipeline metrics.

        Used by as_context_text() so ARIA's prompt context is rich and complete.
        NOT used for Cerberus-V NLI verification.
        """
        caption_val = getattr(frame, "caption", None)
        if isinstance(caption_val, dict):
            semantic_caption = caption_val.get("semantic_caption") or "[CAPTION_FAILED]"
        else:
            semantic_caption = caption_val or "[CAPTION_FAILED]"
            
        parts = [
            f"Frame {frame.frame_idx}",
            f"Timestamp: {frame.timestamp_sec:.1f}s",
            "",
            f"Caption:\n{semantic_caption}",
            "",
            f"Action Score:\n{frame.action_score:.2f}",
            "",
            f"Persistence:\n{frame.persistence_value:.2f}"
        ]
        
        if frame.is_peak:
            parts.extend([
                "",
                f"Selection Reason:\nTop peak after NMS."
            ])
            
        return "\n".join(parts)

    @staticmethod
    def _frame_to_nli_fact(frame: "CachedFrame") -> str | None:
        """Semantic-only fact string for Cerberus-V NLI verification.

        Returns ONLY the natural-language scene description — no action_score,
        no persistence_value, no bare numbers.  An NLI model cannot meaningfully
        reason about internal pipeline metrics, and numeric metadata phrased as
        sentences produces false contradiction verdicts.

        Returns None if the frame has no caption.  A frame without a semantic
        caption has nothing meaningful to contribute to the NLI fact pool.
        """
        caption_val = getattr(frame, "caption", None)
        if not caption_val:
            return None
            
        if isinstance(caption_val, dict):
            semantic_caption = caption_val.get("semantic_caption")
        else:
            semantic_caption = caption_val
            
        # Reject silent fallback, failed captions, or legacy fake content
        if not semantic_caption or semantic_caption == "[CAPTION_FAILED]" or "rabbit" in semantic_caption.lower() or "meadow" in semantic_caption.lower():
            return None
            
        return f"Frame {frame.frame_idx} at {frame.timestamp_sec:.2f}s: {semantic_caption}."

    def as_context_text(self) -> str:
        """Full context string for ARIA's prompt.

        Includes numeric pipeline metrics so ARIA can reference them in prose.
        Uses _frame_to_display_text(), not the NLI-only variant.
        """
        frame_blocks = []
        for frame in self._frames.values():
            frame_blocks.append(self._frame_to_display_text(frame))
        return "\n\n---\n\n".join(frame_blocks)

    @property
    def set_facts(self) -> dict:
        """Semantic-only fact pool for Cerberus-V NLI verification.

        Uses _frame_to_nli_fact() — numeric pipeline metrics are excluded.
        Frames without a caption are skipped entirely (None return value)
        rather than contributing a meaningless numeric string to the pool.
        """
        class MockFactEntry:
            def __init__(self, text: str):
                self.text = text

        facts = {}
        for frame in self._frames.values():
            text = self._frame_to_nli_fact(frame)
            if text is None:
                continue  # no caption → no NLI-checkable fact for this frame
            facts[text] = MockFactEntry(text)
        return facts
