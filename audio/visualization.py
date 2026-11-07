"""
Audio visualization tools: waveform with speech/pause overlays, and spectrograms.
Supports both Matplotlib (for static report export and Streamlit display) with clean light theme styling.
"""
from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

from vad.silero_detector import SpeechSegment, PauseEvent


def plot_waveform_with_vad(
    audio_tensor: torch.Tensor,
    sample_rate: int = 16000,
    speech_segments: Optional[List[SpeechSegment]] = None,
    pause_events: Optional[List[PauseEvent]] = None,
    title: str = "Audio Waveform & VAD Boundaries",
    save_path: Optional[Path] = None,
    theme: str = "light",
) -> plt.Figure:
    """
    Renders waveform with speech and pause interval overlays.
    Supports clean white light theme with purple and emerald accents.
    """
    audio_np = audio_tensor.detach().cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()

    duration_sec = len(audio_np) / sample_rate
    time_axis = np.linspace(0, duration_sec, len(audio_np))

    # Theme colors
    if theme == "light":
        bg_fig = "#FFFFFF"
        bg_ax = "#F8FAFC"
        wave_color = "#64748B"
        speech_color = "#10B981"
        pause_color = "#8B5CF6"  # Purple for hesitation pause
        trailing_color = "#EC4899"  # Pink/Rose for trailing pause
        text_color = "#0F172A"
        label_color = "#475569"
        grid_color = "#E2E8F0"
        legend_bg = "#FFFFFF"
        legend_edge = "#CBD5E1"
    else:
        bg_fig = "#0B0B0B"
        bg_ax = "#141414"
        wave_color = "#94A3B8"
        speech_color = "#10B981"
        pause_color = "#F59E0B"
        trailing_color = "#EF4444"
        text_color = "#FFFFFF"
        label_color = "#A1A1AA"
        grid_color = "#262626"
        legend_bg = "#1F1F1F"
        legend_edge = "#333333"

    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor=bg_fig)
    ax.set_facecolor(bg_ax)

    # Waveform
    ax.plot(time_axis, audio_np, color=wave_color, alpha=0.75, linewidth=0.7, label="Waveform")

    # Speech intervals overlay
    if speech_segments:
        for idx, seg in enumerate(speech_segments):
            ax.axvspan(
                seg.start_sec,
                seg.end_sec,
                color=speech_color,
                alpha=0.25,
                label="Speech" if idx == 0 else "",
            )

    # Pause intervals overlay
    if pause_events:
        for idx, pause in enumerate(pause_events):
            color = pause_color if not pause.is_trailing else trailing_color
            ax.axvspan(
                pause.start_sec,
                pause.end_sec,
                color=color,
                alpha=0.28,
                linestyle="--",
                label="Pause" if idx == 0 else "",
            )

    ax.set_title(title, color=text_color, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Time (seconds)", color=label_color, fontsize=10)
    ax.set_ylabel("Amplitude", color=label_color, fontsize=10)
    ax.tick_params(colors=label_color)
    ax.grid(True, color=grid_color, linestyle=":", alpha=0.8)
    ax.set_xlim(0, max(0.1, duration_sec))
    ax.set_ylim(-1.05, 1.05)

    # Legend
    legend = ax.legend(loc="upper right", facecolor=legend_bg, edgecolor=legend_edge)
    for text in legend.get_texts():
        text.set_color(text_color)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_spectrogram(
    audio_tensor: torch.Tensor,
    sample_rate: int = 16000,
    title: str = "Audio Spectrogram (16 kHz)",
    save_path: Optional[Path] = None,
    theme: str = "light",
) -> plt.Figure:
    """Renders audio spectrogram with crisp light theme styling."""
    audio_np = audio_tensor.detach().cpu().numpy()
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()

    bg_fig = "#FFFFFF" if theme == "light" else "#0B0B0B"
    bg_ax = "#F8FAFC" if theme == "light" else "#141414"
    text_color = "#0F172A" if theme == "light" else "#FFFFFF"
    label_color = "#475569" if theme == "light" else "#A1A1AA"

    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor=bg_fig)
    ax.set_facecolor(bg_ax)

    Pxx, freqs, bins, im = ax.specgram(
        audio_np,
        NFFT=512,
        Fs=sample_rate,
        noverlap=256,
        cmap="viridis" if theme == "light" else "magma",
    )

    ax.set_title(title, color=text_color, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Time (seconds)", color=label_color, fontsize=10)
    ax.set_ylabel("Frequency (Hz)", color=label_color, fontsize=10)
    ax.tick_params(colors=label_color)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    return fig
