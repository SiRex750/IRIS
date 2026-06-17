"""
IRISConfig and ConfigManager for IRIS pipeline configuration.

IRISConfig holds all tunable thresholds and retrieval weights.
ConfigManager loads GEPA-output JSON and injects config into
pipeline components.

Fields:
    salient_thresh: float      # default 0.35 — PEAK/SALIENT floor
    candidate_thresh: float    # default 0.08 — CANDIDATE floor
    alpha: float               # default 0.4  — semantic weight in L2 retrieval
    beta: float                # default 0.6  — motion weight in L2 retrieval
    peak_order: int            # default 3    — argrelextrema window for PEAK detection

Owner: Track A
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class IRISConfig:
    salient_thresh: float = 0.35
    candidate_thresh: float = 0.08
    alpha: float = 0.4
    beta: float = 0.6
    peak_order: int = 3

    def validate(self) -> None:
        # TODO: implement range checks
        pass


class ConfigManager:
    def __init__(self, config_path: str | Path | None = None) -> None:
        # TODO: implement — load from JSON if path given, else use defaults
        pass

    def get_config(self) -> IRISConfig:
        # TODO: implement
        pass

    def reload(self) -> None:
        # TODO: implement — hot-reload for GEPA tuning loop
        pass
