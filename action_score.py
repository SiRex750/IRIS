from __future__ import annotations
from collections import deque
import numpy as np
import scipy.signal


class FrameFeatureBuffer:
    """
    Maintains a rolling window of Charon-V per-frame records.
    """
    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self.window: deque[dict] = deque(maxlen=window_size)

    def push(self, record: dict) -> None:
        """
        Pushes a new per-frame record into the rolling window.
        Expected record format:
        {
            "frame_idx":        int,
            "frame_type":       str,    # "I", "P", "B"
            "residual_energy":  float,
            "motion_magnitude": float,
            "entropy":          float
        }
        """
        self.window.append(record)

    def clear(self) -> None:
        self.window.clear()


class ActionScoreModule:
    """
    Computes a composite action score and determines peak frames using
    topological persistence.
    """
    def __init__(
        self,
        w1: float = 0.5,
        w2: float = 0.3,
        w3: float = 0.2,
        persistence_thresh: float = 0.4
    ) -> None:
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.persistence_thresh = persistence_thresh

    def compute_action_score(self, energies: list[float]) -> tuple[float, bool]:
        """
        Runs single-parameter persistence on the scalar residual energy time-series
        using scipy.signal.find_peaks with prominence as the persistence proxy.
        
        Phase 2 TODO: Implement multi-parameter persistence across residual energy + motion + entropy jointly.
        """
        if len(energies) < 2:
            return 0.0, False

        # Normalize energies to [0, 1] to make the default threshold of 0.4 meaningful
        min_e = min(energies)
        max_e = max(energies)
        if max_e > min_e:
            norm_energies = [(e - min_e) / (max_e - min_e) for e in energies]
        else:
            norm_energies = energies

        # Pad at the right to evaluate the latest frame as a peak if it is a local maximum
        pad_value = min(norm_energies) - 1.0
        padded_energies = np.append(norm_energies, [pad_value])

        # Find peaks with prominence calculation
        peaks, properties = scipy.signal.find_peaks(padded_energies, prominence=0.0)

        latest_idx_in_window = len(norm_energies) - 1
        persistence_value = 0.0
        is_peak = False

        if latest_idx_in_window in peaks:
            peak_loc = np.where(peaks == latest_idx_in_window)[0][0]
            persistence_value = float(properties["prominences"][peak_loc])
            is_peak = persistence_value >= self.persistence_thresh

        return persistence_value, is_peak

    def score(self, buf: FrameFeatureBuffer) -> dict:
        """
        Processes the current rolling window and returns the action score and
        peak decision for the latest frame.
        """
        if not buf.window:
            return {
                "frame_idx": -1,
                "action_score": 0.0,
                "is_peak": False,
                "persistence_value": 0.0
            }

        latest_record = buf.window[-1]
        frame_idx = latest_record["frame_idx"]

        # Collect features in the window for min-max normalization
        energies = [r["residual_energy"] for r in buf.window]
        motions = [r["motion_magnitude"] for r in buf.window]

        def normalize(val: float, val_list: list[float]) -> float:
            if not val_list:
                return 0.0
            min_v = min(val_list)
            max_v = max(val_list)
            if max_v > min_v:
                return (val - min_v) / (max_v - min_v)
            else:
                # If all values are identical, return the raw value itself to handle flat signals gracefully
                return val

        norm_energy = normalize(latest_record["residual_energy"], energies)
        norm_motion = normalize(latest_record["motion_magnitude"], motions)
        entropy = latest_record["entropy"]

        # Compute composite score
        action_score = (self.w1 * norm_energy) + (self.w2 * norm_motion) + (self.w3 * entropy)

        # Run persistence-filtered peak detection on residual energy
        persistence_value, is_peak = self.compute_action_score(energies)

        return {
            "frame_idx": frame_idx,
            "action_score": float(action_score),
            "is_peak": is_peak,
            "persistence_value": persistence_value
        }
