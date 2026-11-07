"""Sensitivity presets for accessibility awareness: Patient, Normal, Quick."""
from enum import Enum
from dataclasses import dataclass


class SensitivityPreset(str, Enum):
    PATIENT = "patient"
    NORMAL = "normal"
    QUICK = "quick"


@dataclass
class PresetConfig:
    name: str
    weight: float              # Numeric feature fed into classifier (-1.0 to 1.0)
    baseline_threshold_sec: float # Static baseline comparison cutoff
    min_wait_boost_sec: float  # Additional tolerance before deciding turn completion
    description: str


PRESET_CONFIGS = {
    SensitivityPreset.PATIENT: PresetConfig(
        name="patient",
        weight=1.0,
        baseline_threshold_sec=1.5,
        min_wait_boost_sec=0.8,
        description="Allows extended pauses before responding; designed for speech hesitations and dysarthric speakers.",
    ),
    SensitivityPreset.NORMAL: PresetConfig(
        name="normal",
        weight=0.0,
        baseline_threshold_sec=0.8,
        min_wait_boost_sec=0.0,
        description="Balanced turn-taking timing for standard conversational pacing.",
    ),
    SensitivityPreset.QUICK: PresetConfig(
        name="quick",
        weight=-1.0,
        baseline_threshold_sec=0.4,
        min_wait_boost_sec=-0.3,
        description="Fast conversational responses with minimal pause tolerance.",
    ),
}


def get_preset_config(preset: str | SensitivityPreset) -> PresetConfig:
    if isinstance(preset, str):
        preset = SensitivityPreset(preset.lower())
    return PRESET_CONFIGS[preset]

