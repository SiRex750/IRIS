"""
IRISConfig and ConfigManager for IRIS pipeline configuration.

IRISConfig holds all tunable thresholds and weights across the pipeline.
ConfigManager loads GEPA-output JSON and injects config into components.

Owner: Track A (L1 fields)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class IRISConfig:
    # ── Charon-V ──────────────────────────────────────────────────────────
    salient_thresh:   float = 0.35   # residual floor for SALIENT tier
    candidate_thresh: float = 0.08   # residual floor for CANDIDATE tier
    peak_order:       int   = 3      # argrelextrema window for PEAK detection
    adaptive:         bool  = True   # Whether Charon-V uses adaptive thresholding


    # ── L2 Asphodel retrieval ─────────────────────────────────────────────
    alpha: float = 0.4   # semantic weight in L2 retrieval blend
    beta:  float = 0.6   # motion weight in L2 retrieval blend

    # ── Cerberus-V ────────────────────────────────────────────────────────
    cerberus_high_thresh: float = 0.70  # action_score >= this → full NLI
    cerberus_low_thresh:  float = 0.35  # action_score >= this → filtered NLI
    disable_nli:          bool  = False # completely bypass DeBERTa NLI, use ner_only

    # ── Action Score Module ────────────────────────────────────────────────
    residual_weight:        float = 0.5
    motion_weight:          float = 0.3
    entropy_weight:         float = 0.2
    peak_distance:          int   = 5
    peak_prominence:        float = 0.05
    persistence_threshold:  float = 0.4
    max_prominence:         float = 0.5   # calibrated global max prominence (Fix 3 Option B)

    # ── L2 retrieve top-k ─────────────────────────────────────────────────
    l2_retrieve_top_k:      int   = 5

    # ── L1 Elysium — capacity ─────────────────────────────────────────────
    # Maximum number of CachedFrame entries L1 holds at once.
    # When exceeded, the frame with the lowest keep_score is evicted.
    l1_capacity: int = 64

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

    def validate(self) -> None:
        """
        Sanity-check all config values.
        Called by ConfigManager after loading.
        Raises AssertionError with a clear message if anything is wrong.
        """
        if not self.disable_nli:
            assert 0 < self.cerberus_low_thresh < self.cerberus_high_thresh < 1.0, (
                "Cerberus thresholds must satisfy: 0 < low < high < 1"
            )
        assert self.l1_capacity > 0, (
            "l1_capacity must be a positive integer"
        )

        assert self.residual_weight >= 0 and self.motion_weight >= 0 and self.entropy_weight >= 0, (
            "Action score weights must be non-negative"
        )
        assert (self.residual_weight + self.motion_weight + self.entropy_weight) > 0, (
            "Action score weights must sum to a positive value"
        )
        assert self.peak_distance > 0, "peak_distance must be positive"
        assert self.peak_prominence >= 0, "peak_prominence must be non-negative"
        assert 0.0 <= self.persistence_threshold <= 1.0, "persistence_threshold must be between 0 and 1"
        assert self.max_prominence > 0, "max_prominence must be positive"
        assert self.l2_retrieve_top_k > 0, "l2_retrieve_top_k must be positive"

        l1_weight_sum = round(
            self.l1_w_action + self.l1_w_query + self.l1_w_persist
            + self.l1_w_pagerank + self.l1_w_entropy
            + self.l1_w_hessian + self.l1_w_recency,
            6,
        )
        assert abs(l1_weight_sum - 1.0) < 1e-4, (
            f"L1 eviction weights must sum to 1.0, got {l1_weight_sum}"
        )


class ConfigManager:
    """
    Loads IRISConfig from a JSON file (written by GEPA).
    Falls back to IRISConfig defaults if no file is given.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._path = Path(config_path) if config_path else None
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
        If no path given, return defaults.
        """
        if self._path is None or not self._path.exists():
            cfg = IRISConfig()
            cfg.validate()
            return cfg

        with open(self._path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build IRISConfig — only accept keys that are actual fields.
        # Unknown keys in the JSON are silently ignored so old JSON
        # files don't crash a newer version of IRISConfig.
        valid_keys = IRISConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}

        cfg = IRISConfig(**filtered)
        cfg.validate()
        return cfg