"""
L2TieredIndex — codec-tier-aware FAISS indexing for L2 Asphodel.

Routes each frame embedding to a different FAISS index based on the
codec-derived importance tier:

    PEAK frames      → IndexFlatIP    (exact search, microseconds)
    SALIENT frames   → IndexHNSWFlat  (approximate, <1ms)
    CANDIDATE frames → IndexPQ        (compressed, ~255× smaller)
    SKIP frames      → Not indexed    (stored raw in L3 Tartarus only)

At query time all three indexes are searched and results are merged by
inner-product score before returning top-k to the caller.

Owner: Track A
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import faiss
import numpy as np

from iris_config import IRISConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Tier enum
# ---------------------------------------------------------------------------

class FrameTier(str, Enum):
    """Codec-derived importance tier assigned by Charon-V / action_score."""
    PEAK = "PEAK"
    SALIENT = "SALIENT"
    CANDIDATE = "CANDIDATE"
    SKIP = "SKIP"


# ---------------------------------------------------------------------------
#  Per-tier metadata
# ---------------------------------------------------------------------------

@dataclass
class _TierRecord:
    """Internal bookkeeping for one FAISS index tier."""
    index: faiss.Index | None = None
    # Maps FAISS internal row position → original frame_idx.
    row_to_frame: list[int] = field(default_factory=list)
    trained: bool = False


# ---------------------------------------------------------------------------
#  Main class
# ---------------------------------------------------------------------------

class L2TieredIndex:
    """Codec-tier-aware FAISS index for L2 Asphodel.

    Contribution 4 — each frame is routed to a different index backend
    based on its codec-derived importance tier.

    Usage::

        idx = L2TieredIndex(config)
        idx.add(frame_idx=10, embedding=emb, action_score=0.9, is_peak=True)
        results = idx.search(query_embedding, top_k=5)
    """

    def __init__(self, config: IRISConfig | None = None) -> None:
        self._config: IRISConfig = config or IRISConfig()
        self._dim: int = self._config.l2_embed_dim

        # --- PEAK tier: exact brute-force inner product ---
        self._peak = _TierRecord(
            index=faiss.IndexFlatIP(self._dim),
            trained=True,  # FlatIP needs no training
        )

        # --- SALIENT tier: HNSW approximate search ---
        hnsw = faiss.IndexHNSWFlat(self._dim, self._config.l2_hnsw_m)
        hnsw.hnsw.efSearch = self._config.l2_hnsw_ef_search
        self._salient = _TierRecord(
            index=hnsw,
            trained=True,  # HNSW trains on-the-fly
        )

        # --- CANDIDATE tier: product quantization ---
        # PQ requires explicit training. We defer creation until we have
        # enough vectors to train (>= l2_pq_min_train).  Until then, a
        # temporary FlatIP collects candidate embeddings.
        self._candidate_buffer: list[tuple[int, np.ndarray]] = []
        self._candidate = _TierRecord(index=None, trained=False)

        # Total frame count across tiers
        self._total_frames: int = 0

    # ── Tier routing ──────────────────────────────────────────────────────

    def _classify_tier(
        self,
        action_score: float,
        is_peak: bool,
    ) -> FrameTier:
        """Determine which FAISS backend a frame should enter."""
        if is_peak:
            return FrameTier.PEAK
        if action_score >= self._config.l2_salient_action_thresh:
            return FrameTier.SALIENT
        return FrameTier.CANDIDATE

    # ── Add ───────────────────────────────────────────────────────────────

    def add(
        self,
        frame_idx: int,
        embedding: np.ndarray,
        action_score: float,
        is_peak: bool = False,
        persistence_value: float = 0.0,
    ) -> FrameTier:
        """Insert one frame embedding into the appropriate tiered index.

        Args:
            frame_idx: Unique frame identifier.
            embedding: Float32 vector of shape ``(dim,)``.
            action_score: Continuous importance score in [0, 1].
            is_peak: Whether this frame was detected as a peak.
            persistence_value: Peak prominence (unused for routing but
                kept for API symmetry with other components).

        Returns:
            The ``FrameTier`` the frame was routed to.
        """
        vec = np.ascontiguousarray(
            embedding.reshape(1, -1).astype(np.float32)
        )
        assert vec.shape[1] == self._dim, (
            f"Embedding dim {vec.shape[1]} != configured dim {self._dim}"
        )

        tier = self._classify_tier(action_score, is_peak)

        if tier == FrameTier.PEAK:
            self._peak.index.add(vec)
            self._peak.row_to_frame.append(frame_idx)

        elif tier == FrameTier.SALIENT:
            self._salient.index.add(vec)
            self._salient.row_to_frame.append(frame_idx)

        elif tier == FrameTier.CANDIDATE:
            self._candidate_buffer.append((frame_idx, vec.copy()))
            self._maybe_train_pq()

        self._total_frames += 1
        return tier

    def add_batch(
        self,
        frame_indices: list[int],
        embeddings: np.ndarray,
        action_scores: list[float],
        is_peaks: list[bool],
    ) -> list[FrameTier]:
        """Batch insert multiple frames.

        Args:
            frame_indices: List of frame identifiers.
            embeddings: Float32 array of shape ``(N, dim)``.
            action_scores: Parallel list of importance scores.
            is_peaks: Parallel list of peak flags.

        Returns:
            List of ``FrameTier`` assignments (one per frame).
        """
        tiers: list[FrameTier] = []
        for i, fidx in enumerate(frame_indices):
            t = self.add(
                frame_idx=fidx,
                embedding=embeddings[i],
                action_score=action_scores[i],
                is_peak=is_peaks[i],
            )
            tiers.append(t)
        return tiers

    # ── PQ training ───────────────────────────────────────────────────────

    def _maybe_train_pq(self) -> None:
        """Train the PQ index once enough candidate vectors are buffered.

        PQ requires at least ``2^nbits`` vectors for clustering. We also
        respect ``l2_pq_min_train`` as an additional floor.
        """
        pq_cluster_min = 2 ** self._config.l2_pq_nbits  # e.g. 256 for 8 bits
        effective_min = max(self._config.l2_pq_min_train, pq_cluster_min)

        if self._candidate.trained:
            return
        if len(self._candidate_buffer) < effective_min:
            return

        # Stack buffered vectors for training
        vecs = np.vstack([v for _, v in self._candidate_buffer])

        pq = faiss.IndexPQ(
            self._dim,
            self._config.l2_pq_m,
            self._config.l2_pq_nbits,
        )
        pq.train(vecs)
        pq.add(vecs)

        self._candidate.index = pq
        self._candidate.row_to_frame = [fidx for fidx, _ in self._candidate_buffer]
        self._candidate.trained = True
        self._candidate_buffer.clear()

        logger.info(
            "PQ index trained with %d candidate vectors (dim=%d, m=%d, nbits=%d)",
            vecs.shape[0], self._dim,
            self._config.l2_pq_m, self._config.l2_pq_nbits,
        )

        logger.info(
            "PQ index trained with %d candidate vectors (dim=%d, m=%d, nbits=%d)",
            vecs.shape[0], self._dim,
            self._config.l2_pq_m, self._config.l2_pq_nbits,
        )

    def force_train_pq(self) -> None:
        """Force PQ training even if buffer is below threshold.

        Falls back to FlatIP if buffer has fewer vectors than PQ requires.
        """
        if self._candidate.trained:
            return

        if not self._candidate_buffer:
            return

        vecs = np.vstack([v for _, v in self._candidate_buffer])
        n = vecs.shape[0]

        # PQ requires at least 2^nbits vectors per subquantizer to train.
        # If we don't have enough, fall back to FlatIP.
        pq_min = 2 ** self._config.l2_pq_nbits
        if n < pq_min:
            logger.warning(
                "Only %d candidate vectors (need %d for PQ) — falling back to FlatIP",
                n, pq_min,
            )
            flat = faiss.IndexFlatIP(self._dim)
            flat.add(vecs)
            self._candidate.index = flat
        else:
            pq = faiss.IndexPQ(
                self._dim,
                self._config.l2_pq_m,
                self._config.l2_pq_nbits,
            )
            pq.train(vecs)
            pq.add(vecs)
            self._candidate.index = pq

        self._candidate.row_to_frame = [fidx for fidx, _ in self._candidate_buffer]
        self._candidate.trained = True
        self._candidate_buffer.clear()

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search all three tier indexes and merge results by score.

        Args:
            query_embedding: Float32 query vector of shape ``(dim,)``.
            top_k: Number of results to return.

        Returns:
            List of dicts ``{"frame_idx": int, "score": float, "tier": str}``
            sorted descending by score.
        """
        # Ensure any remaining candidate buffer is indexed
        if not self._candidate.trained and self._candidate_buffer:
            self.force_train_pq()

        q = np.ascontiguousarray(
            query_embedding.reshape(1, -1).astype(np.float32)
        )
        assert q.shape[1] == self._dim

        all_results: list[tuple[float, int, str]] = []

        # Search each tier
        for tier_name, tier_rec in [
            ("PEAK", self._peak),
            ("SALIENT", self._salient),
            ("CANDIDATE", self._candidate),
        ]:
            if tier_rec.index is None or tier_rec.index.ntotal == 0:
                continue

            k = min(top_k, tier_rec.index.ntotal)
            scores, ids = tier_rec.index.search(q, k)

            for score, idx in zip(scores[0], ids[0]):
                if idx < 0:  # FAISS returns -1 for unfilled slots
                    continue
                frame_idx = tier_rec.row_to_frame[idx]
                all_results.append((float(score), frame_idx, tier_name))

        # Sort descending by score
        all_results.sort(key=lambda x: x[0], reverse=True)

        return [
            {"frame_idx": fidx, "score": score, "tier": tier}
            for score, fidx, tier in all_results[:top_k]
        ]

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return per-tier frame counts and memory estimates."""
        peak_n = self._peak.index.ntotal if self._peak.index else 0
        salient_n = self._salient.index.ntotal if self._salient.index else 0
        candidate_n = (
            self._candidate.index.ntotal
            if self._candidate.index
            else len(self._candidate_buffer)
        )

        # Memory estimates (bytes)
        bytes_per_float32 = 4
        peak_mem = peak_n * self._dim * bytes_per_float32
        salient_mem = salient_n * self._dim * bytes_per_float32  # HNSW stores full vectors
        if (
            self._candidate.index is not None
            and self._candidate.trained
            and isinstance(self._candidate.index, faiss.IndexPQ)
        ):
            # PQ: each vector is compressed to pq_m bytes
            candidate_mem = candidate_n * self._config.l2_pq_m
        else:
            candidate_mem = candidate_n * self._dim * bytes_per_float32

        # What a single flat index would cost for all frames
        flat_equivalent_mem = self._total_frames * self._dim * bytes_per_float32

        return {
            "total_frames": self._total_frames,
            "peak": {"count": peak_n, "memory_bytes": peak_mem},
            "salient": {"count": salient_n, "memory_bytes": salient_mem},
            "candidate": {
                "count": candidate_n,
                "memory_bytes": candidate_mem,
                "pq_trained": self._candidate.trained,
                "buffer_pending": len(self._candidate_buffer),
            },
            "flat_equivalent_memory_bytes": flat_equivalent_mem,
            "memory_reduction_ratio": (
                flat_equivalent_mem / max(peak_mem + salient_mem + candidate_mem, 1)
            ),
        }

    @property
    def total_indexed(self) -> int:
        """Total number of frames across all tiers (including buffer)."""
        return self._total_frames
