"""
IRISConfig and ConfigManager for IRIS pipeline configuration.

IRISConfig holds all tunable thresholds and weights across the pipeline.
ConfigManager loads GEPA-output JSON and injects config into components.

Owner: Track A (L1 fields)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Default config path relative to the package root (configs/default_iris_config.json)
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default_iris_config.json"


@dataclass
class IRISConfig:
    # ── Charon-V ──────────────────────────────────────────────────────────
    salient_thresh:   float = 0.35   # luma-diff floor for SALIENT tier
    candidate_thresh: float = 0.08   # luma-diff floor for CANDIDATE tier
    peak_order:       int   = 3      # argrelextrema window for PEAK detection
    adaptive:         bool  = True   # Whether Charon-V uses adaptive thresholding
    threshold_mode:   str   = "adaptive"  # "fixed" | "adaptive"


    # ── L2 Asphodel retrieval ─────────────────────────────────────────────
    alpha: float = 0.4   # semantic weight in L2 retrieval blend
    beta:  float = 0.3   # motion weight in L2 retrieval blend
    gamma: float = 0.2   # persistence weight in L2 retrieval blend
    delta: float = 0.1   # PageRank/graph-structure weight in legacy retrieval blend
    retrieval_strategy: str = "hybrid"  # strategy: "peak_only", "top_k_action", "peak_neighbors", "hybrid"
    # CFG-004: Changed default from "legacy" to "ppr" — PPR is the correct default for Track A
    ranking_mode: str = "ppr"  # "legacy" = α·sem+β·motion+γ·persist+δ·pagerank; "ppr" = query-conditioned Personalized PageRank
    codec_conf_source: str = "packet_size"  # "packet_size" = true demux size; "action_score" = proxy (Phase-6 diag fallback)
    codec_conf_pictype_norm: bool  = True   # per-pict-type normalization; False = global C_raw baseline (ablation)
    ppr_lambda:             float = 0.5    # rank-space blend weight: λ·sem_rank + (1-λ)·codec_rank
    ppr_damping:            float = 0.5    # Damping factor d passed to nx.pagerank (teleport probability is 1 - d)
    
    # ── L2 Asphodel Graph representation mode ──
    graph_mode:             str   = "scene_sparse"  # "flat" | "scene_sparse"
    scene_shortlist_width:  int   = 0      # scene_sparse coarse-prune width; 0 = auto max(4, ceil(sqrt(num_scenes)))
    scene_shortcut_margin:  float = 0.015  # margin > tau -> short-circuit (no PPR descent)
    scene_neighbor_window:  int   = 30     # anchor +/- N frames pulled into descent pool
    scene_diag:             bool  = False  # measurement-only divergence instrumentation
    scene_crossscene_mode:  str   = "rep_only"  # "all" | "threshold" | "rep_only"
    scene_crossscene_threshold_pctile: float = 75.0  # only used when scene_crossscene_mode="threshold"

    # ── Graph edge weights configuration ──
    graph_edge_mode: str = "hierarchical_sparse"  # "hierarchical_sparse" or "fully_connected"
    graph_temporal_window: int = 1  # connect each indexed frame to this many temporal neighbors
    graph_semantic_top_k: int = 4  # salient/salient semantic cross-root neighbors
    graph_motion_top_k: int = 2  # nearest action/motion neighbors per node
    graph_semantic_threshold: float = 0.5  # minimum cosine for salient semantic cross edges
    graph_debug_retrieval: bool = False  # print per-node retrieval score contributions only when enabled
    graph_export_max_edges: int = 5000  # cap debug/UI graph edge export

    # ── Cerberus-V ────────────────────────────────────────────────────────
    cerberus_high_thresh: float = 0.70  # action_score >= this → full NLI
    cerberus_low_thresh:  float = 0.35  # action_score >= this → filtered NLI
    disable_nli:          bool  = False # completely bypass DeBERTa NLI, use ner_only
    cerberus_mode:        str   = "v2" # "legacy" or "v2"

    # ── Answerer Backend (Prompt 1) ────────────────────────────────────────
    answerer_backend:       str   = "llama_server"
    answerer_endpoint:      str   = "http://127.0.0.1:8091/v1"
    answerer_model:         str   = "granite4:micro"
    answerer_schema_format: bool  = True
    answerer_max_tokens:    int   = 1024
    answerer_timeout:       float = 600.0
    # Fixed sampling seed for the answerer LLM call. temperature=0.0 alone is
    # NOT sufficient for determinism on Ollama/llama-server -- sampler state
    # and batching can still introduce token-level variance (confirmed by a
    # real smoke test: 1/12 repeated identical questions produced a different
    # final answer with temperature=0.0 and no seed). Every backend forwards
    # this as a top-level "seed" field.
    answerer_seed:          int   = 42
    # Ollama-specific: forces the model to be evicted from Ollama's in-memory
    # "warm" cache immediately after each request (0 seconds). Empirically
    # required for determinism on top of answerer_seed/temperature=0.0 -- a
    # real smoke test found that even with seed pinned, a request served
    # against an already-"warm" (previously-loaded) Ollama model handle
    # produced different tokens than one served against a freshly (re)loaded
    # handle, while repeated freshly-loaded calls were bit-identical to each
    # other (confirmed with num_thread pinned to 1 too, which did NOT fix
    # it -- ruling out floating-point reduction-order nondeterminism as the
    # cause). Only meaningful for answerer_backend="llama" (LlamaBackend);
    # LlamaServerBackend already disables its own analogous prompt cache via
    # cache_prompt=False and has no "keep_alive" concept.
    answerer_keep_alive:    int   = 0

    # ── Action Score Module ────────────────────────────────────────────────
    packet_size_weight:     float = 0.5
    luma_diff_weight:       float | None = None  # deprecated
    motion_weight:          float = 0.3
    luma_entropy_weight:         float = 0.2
    peak_distance:          int   = 5
    peak_prominence:        float = 0.05
    persistence_threshold:  float = 0.4
    max_prominence:         float = 0.5   # calibrated global max prominence (Fix 3 Option B)

    # ── L2 retrieve top-k ─────────────────────────────────────────────────
    l2_retrieve_top_k:      int   = 5

    # ── Captioner Backend ──────────────────────────────────────────────────
    captioner_backend:      str   = "minicpm"  # "minicpm", "blip" or "moondream" -- seated/verified production captioner is minicpm-v4.6 (see MiniCPMCaptioner)

    # ── Visual Debug Mode ──────────────────────────────────────────────────
    visual_debug_mode:      bool  = False

    # ── QA debug trace mode ─────────────────────────────────────────────────
    # When True, iris.query.query() collects a full per-query diagnostic trace
    # (retrieval telemetry, retrieved frame images, captions, the exact Granite
    # prompt/output, Cerberus verification, timings) and writes it under
    # debug_traces/<query_id>/. Every trace-collection call site is behind
    # `if config.debug_trace:` so the cost is one boolean check per stage when
    # disabled (default) -- no behavior or output change either way (see
    # iris/debug_trace.py, tests/test_debug_trace.py::test_disabled_is_zero_overhead
    # and test_answers_identical_with_trace_enabled).
    debug_trace:             bool  = False
    debug_trace_dir:         str   = "debug_traces"

    # ── ARIA / LLM model (CFG-005) ─────────────────────────────────────────
    # Override the Ollama/OpenAI model used by ARIA. Empty string = use backend default.
    aria_model:             str   = ""

    # ── CLIP revision (CFG-005) ────────────────────────────────────────────
    # CLIP model variant to load. Must match what was used to build the index.
    clip_revision:          str   = "ViT-B/32"

    # ── L1 Elysium — capacity ─────────────────────────────────────────────
    # Maximum number of CachedFrame entries L1 holds at once.
    # When exceeded, the frame with the lowest keep_score is evicted.
    use_l1: bool = False
    l1_capacity: int = 64
    l1_hessian_saturation_scale: float = 10.0
    codec_validation_level: str = "strict"  # "fast" | "strict"

    # ── L1 Elysium — eviction weights ────────────────────────────────────
    # These seven weights are used in CachedFrame.keep_score().
    # They must sum to 1.0. GEPA will tune them between runs.
    l1_w_action:   float = 0.30
    l1_w_query:    float = 0.20
    l1_w_persist:  float = 0.15
    l1_w_pagerank: float = 0.10
    l1_w_entropy:  float = 0.10
    l1_w_hessian:  float = 0.10
    l1_w_recency:  float = 0.05

    # ── L1 Elysium — dual-vector query weights (Contribution 3) ────────
    # Controls the blend between visual-embedding similarity and motion-
    # embedding similarity when ranking frames during query().
    # Must sum to 1.0. GEPA can shift toward motion for CCTV anomaly
    # queries or toward visual for sports highlights.
    l1_visual_query_weight: float = 0.70
    l1_motion_query_weight: float = 0.30

    # ── L2 Tiered Index (Contribution 4) ─────────────────────────────
    # Embedding dimensionality (must match VLM encoder output).
    l2_embed_dim:       int   = 512
    # HNSW parameters for SALIENT tier
    l2_hnsw_m:          int   = 32    # graph connectivity
    l2_hnsw_ef_search:  int   = 64    # search depth
    # Product Quantization parameters for CANDIDATE tier
    l2_pq_m:            int   = 8     # number of subquantizers
    l2_pq_nbits:        int   = 8     # bits per subquantizer
    # Minimum candidate count before PQ training; below this, fall back to FlatIP
    l2_pq_min_train:    int   = 100
    # Tier routing threshold for SALIENT (frames with action_score >= this
    # that are not peaks go to HNSW)
    l2_salient_action_thresh: float = 0.35

    def __post_init__(self) -> None:
        if self.luma_diff_weight is not None:
            import warnings
            warnings.warn(
                "luma_diff_weight is deprecated; use packet_size_weight instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.packet_size_weight != 0.5 and self.packet_size_weight != self.luma_diff_weight:
                raise ValueError("Conflicting values provided for deprecated 'luma_diff_weight' and new 'packet_size_weight'.")
            self.packet_size_weight = self.luma_diff_weight

    def validate(self) -> None:
        """CFG-002: Sanity-check all config values using ValueError (not assert,
        which is stripped under python -O). Called by ConfigManager after loading.
        Raises ValueError with a clear message if anything is wrong.
        """
        def _check(condition: bool, msg: str) -> None:
            if not condition:
                raise ValueError(f"IRISConfig validation failed: {msg}")

        _check(self.threshold_mode in {"fixed", "adaptive"}, f"Invalid threshold_mode '{self.threshold_mode}'")
        _check(self.salient_thresh > self.candidate_thresh, f"salient_thresh ({self.salient_thresh}) must be greater than candidate_thresh ({self.candidate_thresh})")
        _check(self.codec_validation_level in {"fast", "strict"}, f"Invalid codec_validation_level '{self.codec_validation_level}'")
        if not self.disable_nli:
            _check(0 < self.cerberus_low_thresh < self.cerberus_high_thresh < 1.0,
                   "Cerberus thresholds must satisfy: 0 < low < high < 1")
        _check(self.l1_capacity > 0, "l1_capacity must be a positive integer")
        _check(self.l1_hessian_saturation_scale > 0.0, "l1_hessian_saturation_scale must be positive")
        _check(self.packet_size_weight >= 0 and self.motion_weight >= 0 and self.luma_entropy_weight >= 0,
               "Action score weights must be non-negative")
        _check((self.packet_size_weight + self.motion_weight + self.luma_entropy_weight) > 0,
               "Action score weights must sum to a positive value")
        _check(self.peak_distance > 0, "peak_distance must be positive")
        _check(self.peak_prominence >= 0, "peak_prominence must be non-negative")
        _check(0.0 <= self.persistence_threshold <= 1.0,
               "persistence_threshold must be between 0 and 1")
        _check(self.max_prominence > 0, "max_prominence must be positive")
        _check(self.l2_retrieve_top_k > 0, "l2_retrieve_top_k must be positive")
        _check(self.captioner_backend in {"blip", "moondream", "minicpm"},
               f"Invalid captioner_backend '{self.captioner_backend}'")
        _check(self.answerer_backend in {"llama_server", "llama", "openai", "mock"},
               f"Invalid answerer_backend '{self.answerer_backend}'")
        _check(self.answerer_max_tokens > 0, "answerer_max_tokens must be positive")
        _check(isinstance(self.answerer_seed, int), "answerer_seed must be an int")
        _check(isinstance(self.answerer_keep_alive, int), "answerer_keep_alive must be an int")
        _check(self.answerer_timeout > 0, "answerer_timeout must be positive")
        _check(self.alpha >= 0.0, "alpha must be non-negative")
        _check(self.beta >= 0.0, "beta must be non-negative")
        _check(self.gamma >= 0.0, "gamma must be non-negative")
        _check(self.delta >= 0.0, "delta must be non-negative")
        _check(abs(self.alpha + self.beta + self.gamma + self.delta - 1.0) < 1e-5, "alpha, beta, gamma, delta must sum to 1.0")
        _check(self.retrieval_strategy in {"peak_only", "top_k_action", "peak_neighbors", "hybrid"},
               f"Invalid retrieval_strategy '{self.retrieval_strategy}'")
        _check(self.ranking_mode in {"legacy", "ppr"},
               f"Invalid ranking_mode '{self.ranking_mode}'")
        _check(self.codec_conf_source in {"packet_size", "action_score"},
               f"Invalid codec_conf_source '{self.codec_conf_source}'")
        _check(0.0 <= self.ppr_lambda <= 1.0,
               f"ppr_lambda must be in [0.0, 1.0], got {self.ppr_lambda}")
        _check(0.0 < self.ppr_damping < 1.0,
               f"ppr_damping must be in (0.0, 1.0), got {self.ppr_damping}")
        _check(self.cerberus_mode in {"legacy", "v2"},
               f"Invalid cerberus_mode '{self.cerberus_mode}'")
        _check(self.graph_mode in {"flat", "scene_sparse"},
               f"Invalid graph_mode '{self.graph_mode}'")
        _check(self.scene_shortlist_width >= 0,
               "scene_shortlist_width must be non-negative (0 = auto)")
        _check(self.scene_shortcut_margin >= 0.0,
               "scene_shortcut_margin must be non-negative")
        _check(self.scene_neighbor_window >= 0,
               "scene_neighbor_window must be non-negative")
        _check(self.scene_crossscene_mode in {"all", "threshold", "rep_only"},
               f"Invalid scene_crossscene_mode '{self.scene_crossscene_mode}'")
        _check(0.0 <= self.scene_crossscene_threshold_pctile <= 100.0,
               "scene_crossscene_threshold_pctile must be in [0, 100]")
        _check(self.graph_edge_mode in {"hierarchical_sparse", "fully_connected"},
               f"Invalid graph_edge_mode '{self.graph_edge_mode}'")
        _check(self.graph_temporal_window >= 0,
               "graph_temporal_window must be non-negative")
        _check(self.graph_semantic_top_k >= 0,
               "graph_semantic_top_k must be non-negative")
        _check(self.graph_motion_top_k >= 0,
               "graph_motion_top_k must be non-negative")
        _check(0.0 <= self.graph_semantic_threshold <= 1.0,
               "graph_semantic_threshold must be in [0, 1]")
        _check(self.graph_export_max_edges > 0,
               "graph_export_max_edges must be positive")

        l1_weight_sum = round(
            self.l1_w_action + self.l1_w_query + self.l1_w_persist
            + self.l1_w_pagerank + self.l1_w_entropy
            + self.l1_w_hessian + self.l1_w_recency,
            6,
        )
        _check(abs(l1_weight_sum - 1.0) < 1e-4,
               f"L1 eviction weights must sum to 1.0, got {l1_weight_sum}")

        dv_sum = round(self.l1_visual_query_weight + self.l1_motion_query_weight, 6)
        _check(abs(dv_sum - 1.0) < 1e-4,
               f"Dual-vector query weights must sum to 1.0, got {dv_sum}")
        _check(self.l1_visual_query_weight >= 0.0,
               "l1_visual_query_weight must be non-negative")
        _check(self.l1_motion_query_weight >= 0.0,
               "l1_motion_query_weight must be non-negative")

        _check(self.l2_embed_dim > 0, "l2_embed_dim must be positive")
        _check(self.l2_hnsw_m > 0, "l2_hnsw_m must be positive")
        _check(self.l2_hnsw_ef_search > 0, "l2_hnsw_ef_search must be positive")
        _check(self.l2_pq_m > 0, "l2_pq_m must be positive")
        _check(self.l2_pq_nbits > 0, "l2_pq_nbits must be positive")
        _check(self.l2_embed_dim % self.l2_pq_m == 0,
               f"l2_embed_dim ({self.l2_embed_dim}) must be divisible by l2_pq_m ({self.l2_pq_m})")
        _check(self.l2_pq_min_train > 0, "l2_pq_min_train must be positive")
        _check(0.0 <= self.l2_salient_action_thresh <= 1.0,
               "l2_salient_action_thresh must be in [0, 1]")


class ConfigManager:
    """
    Loads IRISConfig from a JSON file (written by GEPA).

    CFG-001: When config_path is None, automatically probes
    configs/default_iris_config.json relative to the package root before
    falling back to hardcoded IRISConfig dataclass defaults.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is not None:
            self._path: Path | None = Path(config_path)
        elif _DEFAULT_CONFIG_PATH.exists():
            # CFG-001: auto-load shipped default config
            self._path = _DEFAULT_CONFIG_PATH
        else:
            self._path = None
        self._config: IRISConfig = self._load()

    def get_config(self) -> IRISConfig:
        """Return the currently loaded config."""
        return self._config

    def reload(self) -> None:
        """
        Re-read config from file.
        Call this after GEPA writes a new JSON to pick up updated weights
        without restarting the pipeline.
        """
        self._config = self._load()

    def _load(self) -> IRISConfig:
        """
        Internal: parse JSON into IRISConfig, then validate.
        If no path given (or file missing), return hardcoded defaults.
        """
        if self._path is None or not self._path.exists():
            cfg = IRISConfig()
            cfg.validate()
            return cfg

        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)

        valid_keys = set(IRISConfig.__dataclass_fields__.keys())
        # CFG-003: Warn on unknown keys so misconfigured JSON is visible
        unknown = [k for k in data if k not in valid_keys and not k.startswith("_")]
        if unknown:
            print(
                f"[IRISConfig WARNING] Unknown config keys ignored: {unknown}. "
                f"Check {self._path} for typos.",
                file=sys.stderr,
            )
        filtered = {k: v for k, v in data.items() if k in valid_keys}

        cfg = IRISConfig(**filtered)
        cfg.validate()
        return cfg
