"""Fixed-threshold baseline policies."""
from typing import List, Tuple


class FixedThresholdBaseline:
    """Standard fixed silence timeout policy (e.g. 500ms, 800ms, 1200ms)."""

    def __init__(self, threshold_sec: float = 0.8):
        self.threshold_sec = threshold_sec

    def predict(self, pause_duration_sec: float) -> str:
        """Returns 'END' if silence exceeds threshold, else 'CONTINUE'."""
        if pause_duration_sec >= self.threshold_sec:
            return "END"
        return "CONTINUE"

    def evaluate_batch(self, pauses: List[float]) -> List[str]:
        return [self.predict(p) for p in pauses]

