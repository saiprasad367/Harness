"""Voice Activity Detection using Silero VAD."""
from .silero_detector import SileroVADDetector, VADState, SpeechSegment, PauseEvent

__all__ = [
    "SileroVADDetector",
    "VADState",
    "SpeechSegment",
    "PauseEvent",
]

