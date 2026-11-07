"""Global configuration and hyperparameters for the Voice Harness."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AudioConfig:
    """Audio sampling, chunking, and streaming configuration."""
    sample_rate: int = 16000          # Required 16 kHz for Silero VAD & Moshi
    channels: int = 1                 # Mono audio
    chunk_size_ms: int = 200          # 160ms - 320ms streaming chunk per project spec
    dtype: str = "float32"            # Audio representation

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * (self.chunk_size_ms / 1000.0))


@dataclass
class VADConfig:
    """Voice Activity Detection settings using Silero VAD."""
    threshold: float = 0.5            # Speech probability threshold (report experimented 0.3 - 0.5)
    min_speech_duration_ms: int = 100 # Minimum duration to register valid speech
    min_silence_duration_ms: int = 200 # Minimum silence duration to treat as candidate pause
    speech_pad_ms: int = 30           # Padding before/after detected speech boundaries


@dataclass
class ClassifierConfig:
    """Adaptive pause classifier configuration."""
    rolling_history_window: int = 5   # Number of previous pauses to average
    feature_names: List[str] = field(default_factory=lambda: [
        "current_pause_sec",
        "recent_mean_pause_sec",
        "pause_std_dev",
        "preset_weight",
    ])
    model_type: str = "logistic_regression"
    model_save_path: Path = OUTPUT_DIR / "adaptive_pause_classifier.pkl"


@dataclass
class ControllerConfig:
    """Turn-taking controller and barge-in settings."""
    barge_in_enabled: bool = True
    barge_in_sensitivity_threshold: float = 0.6
    backchannel_enabled: bool = False
    backchannel_min_pause_sec: float = 1.8
    default_preset: str = "normal"

