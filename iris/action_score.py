from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks


@dataclass
class ActionScoreConfig:
    """
    Phase-1 config for continuous Action Score.

    Later, these values can move into iris_config.py and be tuned by GEPA.
    """
    packet_size_weight: float = 0.5
    luma_diff_weight: float | None = None  # deprecated
    motion_weight: float = 0.3
    luma_entropy_weight: float = 0.2

    peak_distance: int = 5
    peak_prominence: float = 0.05
    persistence_threshold: float = 0.4
    max_prominence: float = 0.5

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


class ActionScoreModule:
    """
    Converts raw Charon-V codec features into continuous Action Score.

    Input per frame:
        frame_idx
        packet_size        (codec coded packet size — residual channel)
        motion_magnitude
        luma_entropy
        luma_diff_energy   (diagnostic only; not consumed by scorer)

    Output per frame:
        frame_idx
        action_score
        is_peak
        persistence_value

    Important:
        This does not replace the PEAK / SALIENT / CANDIDATE / SKIP tier decisions,
        which are already determined by Charon-V's codec-tier admission gate. It
        computes continuous action scores for the admitted frames.
    """

    def __init__(self, config: ActionScoreConfig | None = None) -> None:
        self.config = config or ActionScoreConfig()

    def score_all(self, frame_features: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not frame_features:
            return []

        residual = np.array(
            [float(f.get("packet_size", 0.0)) for f in frame_features],
            dtype=np.float32,
        )

        motion = np.array(
            [float(f.get("motion_magnitude", 0.0)) for f in frame_features],
            dtype=np.float32,
        )

        luma_entropy = np.array(
            [float(f.get("luma_entropy", 0.0)) for f in frame_features],
            dtype=np.float32,
        )

        residual_n = self._normalize(residual)
        motion_n = self._normalize(motion)
        entropy_n = self._normalize(luma_entropy)

        if self.config.packet_size_weight < 0 or self.config.motion_weight < 0 or self.config.luma_entropy_weight < 0:
            raise ValueError("Action Score weights must be non-negative.")
        weight_sum = (
            self.config.packet_size_weight
            + self.config.motion_weight
            + self.config.luma_entropy_weight
        )

        if weight_sum <= 0:
            raise ValueError("Action Score weights must sum to a positive value.")

        action_score = (
            self.config.packet_size_weight * residual_n
            + self.config.motion_weight * motion_n
            + self.config.luma_entropy_weight * entropy_n
        ) / weight_sum

        action_score = np.clip(action_score, 0.0, 1.0)

        peak_indices, peak_properties = find_peaks(
            action_score,
            distance=self.config.peak_distance,
            prominence=self.config.peak_prominence,
        )

        prominences = peak_properties.get("prominences", np.array([], dtype=np.float32))

        persistence_by_index: dict[int, float] = {}

        if len(peak_indices) > 0:
            # P1-06: Divide by the globally configured max_prominence so that
            # persistence values are comparable across different videos.
            # Dividing by the local video maximum (np.max(prominences)) made
            # persistence scores incomparable between videos processed in the
            # same pipeline run.
            configured_max = float(self.config.max_prominence)
            if configured_max < 1e-8:
                configured_max = 1.0

            for idx, prominence in zip(peak_indices, prominences):
                persistence_value = max(0.0, min(float(prominence / configured_max), 1.0))
                persistence_by_index[int(idx)] = persistence_value

        records: list[dict[str, Any]] = []

        for i, feature in enumerate(frame_features):
            persistence_value = persistence_by_index.get(i, 0.0)

            # Find prominence if it is a peak in peak_indices
            prom = 0.0
            if i in peak_indices:
                idx_pos = np.where(peak_indices == i)[0]
                if len(idx_pos) > 0:
                    prom = float(prominences[idx_pos[0]])

            records.append(
                {
                    "frame_idx": int(feature["frame_idx"]),
                    "action_score": float(action_score[i]),
                    "is_peak": bool(
                        persistence_value >= self.config.persistence_threshold
                    ),
                    "persistence_value": float(persistence_value),
                    "packet_size_contrib": float(residual_n[i]),
                    "motion_contrib": float(motion_n[i]),
                    "luma_entropy_contrib": float(entropy_n[i]),
                    "prominence_contrib": float(prom),
                }
            )

        return records

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        
        # Fall back to standard min-max scaling for small arrays (e.g. synthetic unit tests)
        if len(values) < 50:
            min_value = float(np.min(values))
            max_value = float(np.max(values))
        else:
            # Use robust percentiles for real videos to mitigate outlier saturation
            min_value = float(np.percentile(values, 2))
            max_value = float(np.percentile(values, 98))

        if abs(max_value - min_value) < 1e-8:
            # Percentile range collapsed — fall back to global min/max
            min_value = float(np.min(values))
            max_value = float(np.max(values))
            if abs(max_value - min_value) < 1e-8:
                # Truly constant signal — return 0.0 (no variation, not neutral midpoint).
                # P1-06: Returning 0.5 for a constant signal falsely implies moderate
                # activity; 0.0 correctly conveys that there is nothing to discriminate.
                return np.zeros_like(values, dtype=np.float32)
            normalized = (values - min_value) / (max_value - min_value)
            return np.clip(normalized, 0.0, 1.0)

        normalized = (values - min_value) / (max_value - min_value)
        return np.clip(normalized, 0.0, 1.0)