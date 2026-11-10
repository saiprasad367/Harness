"""
Silero VAD detector — speech boundary and candidate pause extraction.

IMPORTANT — Silero frame size constraint:
  The Silero TorchScript model requires EXACTLY 512 samples per inference call at 16 kHz.
  Passing any other size raises a TorchScript ValueError.

  process_chunk() handles arbitrary incoming chunk sizes by internally splitting them
  into 512-sample frames, running inference on each frame, and returning the
  aggregated (mean) speech probability for the whole chunk.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple
import numpy as np

import torch
from silero_vad import load_silero_vad, get_speech_timestamps

from config.settings import VADConfig

log = logging.getLogger(__name__)

# Required frame size for Silero VAD at 16 kHz (hard constraint from TorchScript model)
SILERO_FRAME_SIZE = 512
SILERO_SAMPLE_RATE = 16000


class VADState(str, Enum):
    SPEECH = "speech"
    SILENCE = "silence"


@dataclass
class SpeechSegment:
    """A contiguous region of detected speech."""
    start_sec: float
    end_sec: float
    duration_sec: float


@dataclass
class PauseEvent:
    """A silence interval between speech regions."""
    start_sec: float
    end_sec: float
    duration_sec: float
    is_pre_speech: bool = False   # Leading silence before first speech
    is_trailing: bool = False     # Trailing silence after last speech


@dataclass
class StreamingVADUpdate:
    """State snapshot returned after processing one streaming chunk."""
    speech_prob: float            # Mean frame-level speech probability for this chunk
    state: VADState               # SPEECH or SILENCE
    current_pause_sec: float      # Accumulated silence duration since last speech
    is_new_pause_event: bool      # True if speech → silence transition just occurred
    new_pause_duration_sec: float = 0.0


class SileroVADDetector:
    """
    Wrapper around Silero VAD supporting both offline timestamp extraction
    and online streaming with correct 512-sample frame handling.
    """

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self.sample_rate = SILERO_SAMPLE_RATE
        log.info("Loading Silero VAD model...")
        self.model = load_silero_vad()
        self.reset_stream()
        log.info("Silero VAD model loaded.")

    def reset_stream(self):
        """Reset all internal streaming state for a new session."""
        self.model.reset_states()
        self._streaming_state = VADState.SILENCE
        self._current_pause_frames: int = 0
        self._current_speech_frames: int = 0
        self._has_speech_started: bool = False
        self._frame_buffer: List[float] = []  # partial leftover samples

    # ------------------------------------------------------------------
    # Offline API: full-file processing
    # ------------------------------------------------------------------

    def extract_speech_and_pauses(
        self,
        audio_tensor: torch.Tensor,
        sample_rate: int = SILERO_SAMPLE_RATE,
    ) -> Tuple[List[SpeechSegment], List[PauseEvent]]:
        """
        Offline extraction of speech segments and candidate pause events.
        Uses Silero's get_speech_timestamps which handles frame sizes internally.
        """
        if audio_tensor.ndim != 1:
            raise ValueError(f"Expected 1D audio tensor, got shape {audio_tensor.shape}")
        if sample_rate != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, got {sample_rate} Hz")

        timestamps = get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=sample_rate,
            threshold=self.config.threshold,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
            return_seconds=True,
        )

        speech_segments: List[SpeechSegment] = []
        for ts in timestamps:
            speech_segments.append(
                SpeechSegment(
                    start_sec=round(ts["start"], 3),
                    end_sec=round(ts["end"], 3),
                    duration_sec=round(ts["end"] - ts["start"], 3),
                )
            )

        total_duration_sec = round(audio_tensor.shape[0] / sample_rate, 3)
        pause_events = self._compute_pauses(speech_segments, total_duration_sec)
        return speech_segments, pause_events

    def _compute_pauses(
        self,
        speech_segments: List[SpeechSegment],
        total_duration_sec: float,
    ) -> List[PauseEvent]:
        """Derive pause events from the gaps between speech segments."""
        min_pause_sec = self.config.min_silence_duration_ms / 1000.0
        pause_events: List[PauseEvent] = []

        if not speech_segments:
            # Entire file is silent
            pause_events.append(
                PauseEvent(
                    start_sec=0.0,
                    end_sec=total_duration_sec,
                    duration_sec=total_duration_sec,
                    is_pre_speech=True,
                    is_trailing=True,
                )
            )
            return pause_events

        # 1. Leading silence before first speech
        if speech_segments[0].start_sec > 0.05:
            pause_events.append(
                PauseEvent(
                    start_sec=0.0,
                    end_sec=speech_segments[0].start_sec,
                    duration_sec=round(speech_segments[0].start_sec, 3),
                    is_pre_speech=True,
                    is_trailing=False,
                )
            )

        # 2. Inter-speech pauses
        for i in range(len(speech_segments) - 1):
            gap_start = speech_segments[i].end_sec
            gap_end = speech_segments[i + 1].start_sec
            gap_dur = round(gap_end - gap_start, 3)
            if gap_dur >= min_pause_sec:
                pause_events.append(
                    PauseEvent(
                        start_sec=gap_start,
                        end_sec=gap_end,
                        duration_sec=gap_dur,
                        is_pre_speech=False,
                        is_trailing=False,
                    )
                )

        # 3. Trailing silence after last speech
        trailing_start = speech_segments[-1].end_sec
        trailing_dur = round(total_duration_sec - trailing_start, 3)
        if trailing_dur >= min_pause_sec:
            pause_events.append(
                PauseEvent(
                    start_sec=trailing_start,
                    end_sec=total_duration_sec,
                    duration_sec=trailing_dur,
                    is_pre_speech=False,
                    is_trailing=True,
                )
            )

        return pause_events

    # ------------------------------------------------------------------
    # Streaming API: per-chunk processing
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: Any) -> StreamingVADUpdate:
        """
        Process an arbitrary-size streaming audio chunk.

        Silero requires exactly SILERO_FRAME_SIZE (512) samples per call.
        Accepts torch.Tensor or numpy.ndarray.
        """
        if isinstance(chunk, np.ndarray):
            chunk = torch.from_numpy(chunk).float()
        elif not isinstance(chunk, torch.Tensor):
            chunk = torch.as_tensor(chunk, dtype=torch.float32)

        if chunk.ndim > 1:
            chunk = chunk.squeeze()

        # Convert to Python list for buffering, then operate in numpy-land
        samples = chunk.detach().cpu().float().tolist()

        # Extend leftover buffer with new samples
        self._frame_buffer.extend(samples)

        frame_probs: List[float] = []

        # Process all complete 512-sample frames
        while len(self._frame_buffer) >= SILERO_FRAME_SIZE:
            frame_samples = self._frame_buffer[:SILERO_FRAME_SIZE]
            self._frame_buffer = self._frame_buffer[SILERO_FRAME_SIZE:]

            frame_tensor = torch.tensor(frame_samples, dtype=torch.float32)
            with torch.no_grad():
                prob = self.model(frame_tensor, self.sample_rate).item()
            frame_probs.append(prob)

        # If no complete frames yet, use last known probability
        if not frame_probs:
            mean_prob = 0.0
        else:
            mean_prob = sum(frame_probs) / len(frame_probs)

        # Update streaming state based on aggregated probability
        is_speech = mean_prob >= 0.40
        prev_state = self._streaming_state
        is_new_pause_event = False

        chunk_len = len(samples)

        if is_speech:
            self._has_speech_started = True
            self._streaming_state = VADState.SPEECH
            self._current_pause_frames = 0
            self._current_speech_frames += chunk_len
        else:
            self._streaming_state = VADState.SILENCE
            self._current_pause_frames += chunk_len
            self._current_speech_frames = 0
            if prev_state == VADState.SPEECH:
                is_new_pause_event = True

        current_pause_sec = round(self._current_pause_frames / self.sample_rate, 3)

        return StreamingVADUpdate(
            speech_prob=round(mean_prob, 4),
            state=self._streaming_state,
            current_pause_sec=current_pause_sec,
            is_new_pause_event=is_new_pause_event,
        )
