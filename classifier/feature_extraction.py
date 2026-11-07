"""Feature extraction and user rolling pause history tracking."""
from collections import deque
from typing import Dict, List, Optional
import numpy as np
from config.sensitivity import SensitivityPreset, get_preset_config


class UserPauseTracker:
    """Maintains sliding-window history of observed pauses per speaker."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._history: Dict[str, deque] = {}

    def add_pause(self, speaker_id: str, pause_duration_sec: float):
        if speaker_id not in self._history:
            self._history[speaker_id] = deque(maxlen=self.window_size)
        self._history[speaker_id].append(pause_duration_sec)

    def get_stats(self, speaker_id: str) -> tuple[float, float]:
        """Returns (mean_pause_sec, std_dev_pause_sec). Defaults to typical 0.8s if no history."""
        if speaker_id not in self._history or len(self._history[speaker_id]) == 0:
            return 0.8, 0.2
        pauses = list(self._history[speaker_id])
        mean_p = float(np.mean(pauses))
        std_p = float(np.std(pauses)) if len(pauses) > 1 else 0.1
        return mean_p, std_p

    def reset_speaker(self, speaker_id: str):
        if speaker_id in self._history:
            self._history[speaker_id].clear()


class PauseFeatureExtractor:
    """Extracts standardized feature vectors for the adaptive pause classifier."""

    def __init__(self, tracker: Optional[UserPauseTracker] = None):
        self.tracker = tracker or UserPauseTracker()

    def extract_features(
        self,
        current_pause_sec: float,
        speaker_id: str = "default_user",
        preset: str | SensitivityPreset = SensitivityPreset.NORMAL,
    ) -> np.ndarray:
        """
        Construct feature vector:
        [current_pause_sec, recent_mean_pause_sec, pause_std_dev, preset_weight]
        """
        recent_mean, std_dev = self.tracker.get_stats(speaker_id)
        preset_cfg = get_preset_config(preset)
        
        feature_vector = np.array(
            [
                current_pause_sec,
                recent_mean,
                std_dev,
                preset_cfg.weight,
            ],
            dtype=np.float32,
        )
        return feature_vector

