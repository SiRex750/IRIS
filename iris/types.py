"""Shared dataclasses defining the contract between ingest() and query().

Phase 3 spine. No logic lives here — these are the serialization and
hand-off types only. The live L2Asphodel graph is intentionally NOT a
serialized field: it is rebuilt from `frames` on load (projection + rebuild),
so Phase 6's AsphodelNode rewrite cannot invalidate an on-disk index.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FrameRecord:
    """One selected, fully-enriched frame — the lightweight projection that
    survives to disk. Deliberately DROPS pil_image and raw motion_vectors
    (needed only during ingest for captioning/embedding) to keep the index
    small and serializable.
    """
    frame_idx:         int
    timestamp:         float
    # Charon-V / action-score features (deterministic, per-video)
    luma_diff_energy:  float
    luma_entropy:      float
    motion_magnitude:  float
    action_score:      float
    persistence_value: float
    is_peak:           bool
    # Motion geometry — carried for exact L1 keep-score / eviction parity.
    # Only motion_entropy + hessian_max_eigenvalue enter keep_score today, but
    # all five populate FrameMotionDescriptor in wrapper_populate_cache.
    divergence:             float = 0.0
    curl:                   float = 0.0
    jacobian_frobenius:     float = 0.0
    hessian_max_eigenvalue: float = 0.0
    motion_entropy:         float = 0.0
    # ARIA enrichment (computed once in ingest)
    caption:           str | dict | None = None
    clip_embedding:    np.ndarray | None = None   # float32 [512]
    # Codec carriage — true packet size from Charon-V demux; 0.0 if unavailable.
    # pict_type: 'I', 'P', 'B', or '?' (unknown / skipped frame).
    # Defaults are safe for old serialized indices and the legacy parity path.
    packet_size:   float = 0.0
    pict_type:     str   = "?"
    # Zero-decode packet-curve valley-scene assignment (Phase 6 scene-sparse
    # groundwork). -1 = unassigned (old serialized indices; flat graph never sets it).
    scene_id:      int   = -1
    # L2 structural score, filled after graph build
    pagerank_score:    float = 0.0
    # Pre-computed codec confidence signal (query-independent, stored at ingest).
    # Per-pict-type-normalized by default; 0.5 = neutral/missing.
    codec_conf:        float = 0.5


@dataclass
class IRISIndex:
    """Everything query() needs with NO video re-read and NO per-query rebuild.

    Holds the flat projection (`frames`). The live L2Asphodel graph is attached
    in-memory on `_graph` — by ingest() directly, or by load_index() via the
    public batch_* rebuild API. `_graph` is excluded from equality and repr and
    is never serialized.
    """
    video_path:               str
    frames:                   list[FrameRecord]
    # video-level scalars (precomputed; were misleadingly named "query_*" in pipeline.py)
    index_action_score:       float
    # reporting parity with old run_pipeline result dict
    stats:                    dict[str, Any]
    frames_processed:         int
    peak_count:               int
    skipped_frames_ratio:     float
    storage_reduction_factor: float
    # provenance: the config that built this index (query reuses these weights)
    config_snapshot:          dict[str, Any]
    schema_version:           int = 1
    # Real scene boundaries, {scene_id: (start_time_s, end_time_s)}. Additive
    # (Part 3c): previously charon_v.compute_valley_scene_boundaries's
    # (start_idx, end_idx) spans were computed at ingest time to assign each
    # frame's integer scene_id, then discarded. This persists the boundary
    # times themselves (start_idx/fps, end_idx/fps -- average-fps
    # approximation, since exact per-boundary-frame PTS would require
    # decoding a frame the zero-decode packet-curve path never touches).
    # Empty dict for old serialized indices (schema-compatible default) and
    # for the hermetic/synthetic-records path (no real packet curve).
    scene_spans:              dict[int, tuple[float, float]] = field(default_factory=dict)
    # in-memory only; excluded from serialization, equality, and repr
    _graph:                   Any = field(default=None, repr=False, compare=False)
    # Edgeless per-scene centroid index (Phase 6 scene-sparse groundwork): NOT a
    # rep graph, no rep-rep edges. {scene_id: mean CLIP embedding}. Deterministic
    # function of frames+scene_id, so it is rebuilt (not serialized) on load,
    # same lifecycle as _graph.
    _scene_centroids:         Any = field(default=None, repr=False, compare=False)
    _l1_cache:                Any = field(default=None, repr=False, compare=False)
