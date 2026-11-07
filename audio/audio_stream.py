"""
Audio streaming module: thread-safe queue buffering, real-time live microphone capture,
real hardware speaker playback, and file-based TORGO acoustic streaming.
"""
import asyncio
import queue
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, Generator, List, Optional
import numpy as np
import torch
from .preprocessing import load_audio, frame_generator


class AudioStreamBuffer:
    """Thread-safe queue for streaming real-time audio frames with overflow drop."""

    def __init__(self, maxsize: int = 200):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.dropped_frames: int = 0

    def put(self, frame: torch.Tensor):
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # Drop oldest frame to maintain low latency and prevent lag accumulation
            try:
                self._queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
            self._queue.put_nowait(frame)

    def get(self, timeout: Optional[float] = None) -> Optional[torch.Tensor]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    def clear(self):
        with self._queue.mutex:
            self._queue.queue.clear()


class MicrophoneStreamer:
    """
    Real-time microphone capture using sounddevice.
    Uses thread-safe buffering with automatic 16 kHz mono conversion.
    No synthetic or fake samples.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size_ms: int = 200,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_size_ms = chunk_size_ms
        self.chunk_samples = int(sample_rate * (chunk_size_ms / 1000.0))
        self.device_index = device_index
        self._stream = None
        self._is_running = False
        self._buffer = AudioStreamBuffer(maxsize=150)
        self.total_chunks_captured = 0
        self.last_capture_time = 0.0
        self.last_rms_db: float = -96.0

    @property
    def is_running(self) -> bool:
        return self._is_running

    @staticmethod
    def list_input_devices() -> List[Dict[str, any]]:
        """Enumerates genuine physical audio input devices from sounddevice."""
        devices = []
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            for idx, d in enumerate(devs):
                if d.get("max_input_channels", 0) > 0:
                    devices.append({
                        "index": idx,
                        "name": d.get("name", f"Input Device {idx}"),
                        "channels": d.get("max_input_channels"),
                        "default_samplerate": d.get("default_samplerate", 16000),
                        "hostapi": d.get("hostapi"),
                    })
        except Exception as e:
            print(f"Warning: Could not query audio input devices: {e}")
        return devices

    def _audio_callback(self, indata, frames, time_info, status):
        """PortAudio C-callback executed in audio hardware thread."""
        if status:
            pass  # Input overflow/underflow flag
        data = indata[:, 0].copy().astype(np.float32)

        # Compute real RMS level (dBFS) for audio meter
        rms = float(np.sqrt(np.mean(data ** 2)))
        self.last_rms_db = round(20.0 * np.log10(max(rms, 1e-5)), 1)

        tensor = torch.from_numpy(data)
        self._buffer.put(tensor)
        self.total_chunks_captured += 1
        self.last_capture_time = time.time()

    def start(self) -> bool:
        """Starts real-time microphone stream from selected hardware device."""
        if self._is_running:
            return True

        import sounddevice as sd
        try:
            self._buffer.clear()
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_samples,
                device=self.device_index,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_running = True
            self.total_chunks_captured = 0
            return True
        except Exception as e:
            self._is_running = False
            self._stream = None
            raise RuntimeError(f"Microphone initialization failed on device {self.device_index}: {e}")

    def get_chunk(self, timeout: float = 0.5) -> Optional[torch.Tensor]:
        """Fetches next audio chunk from thread-safe buffer."""
        return self._buffer.get(timeout=timeout)

    def stop(self):
        """Stops and cleans up audio stream."""
        self._is_running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None
        self._buffer.clear()


class AudioPlayer:
    """
    Real-time speaker audio playback using sounddevice.OutputStream.
    Supports low-latency playback buffering and instantaneous barge-in cancellation.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device_index: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.device_index = device_index
        self._stream = None
        self._is_active = False
        self._playback_queue: queue.Queue = queue.Queue(maxsize=100)
        self.total_chunks_played = 0

    @staticmethod
    def list_output_devices() -> List[Dict[str, any]]:
        """Enumerates genuine physical audio output devices (speakers/headphones)."""
        devices = []
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            for idx, d in enumerate(devs):
                if d.get("max_output_channels", 0) > 0:
                    devices.append({
                        "index": idx,
                        "name": d.get("name", f"Output Device {idx}"),
                        "channels": d.get("max_output_channels"),
                        "default_samplerate": d.get("default_samplerate", 16000),
                    })
        except Exception as e:
            print(f"Warning: Could not query audio output devices: {e}")
        return devices

    def _audio_callback(self, outdata, frames, time_info, status):
        """Hardware callback delivering audio to speakers."""
        try:
            chunk = self._playback_queue.get_nowait()
            chunk_len = len(chunk)
            if chunk_len < frames:
                outdata[:chunk_len, 0] = chunk
                outdata[chunk_len:, 0] = 0
            else:
                outdata[:, 0] = chunk[:frames]
            self.total_chunks_played += 1
        except queue.Empty:
            outdata.fill(0)

    def start(self) -> bool:
        """Initializes hardware output stream."""
        if self._is_active:
            return True

        import sounddevice as sd
        try:
            self.flush()
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device_index,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_active = True
            return True
        except Exception as e:
            self._is_active = False
            self._stream = None
            raise RuntimeError(f"Speaker initialization failed on device {self.device_index}: {e}")

    def play_chunk(self, audio_tensor: torch.Tensor):
        """Queues audio chunk for playback through speakers."""
        data = audio_tensor.detach().cpu().numpy().astype(np.float32)
        try:
            self._playback_queue.put_nowait(data)
        except queue.Full:
            pass

    def flush(self):
        """Clears pending playback buffer instantly (used during barge-in)."""
        with self._playback_queue.mutex:
            self._playback_queue.queue.clear()

    def stop(self):
        """Stops audio player and releases hardware stream."""
        self._is_active = False
        self.flush()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None


class FileAudioStreamer:
    """Simulates real-time audio streaming from pre-recorded TORGO audio files."""

    def __init__(
        self,
        file_path: str | Path,
        sample_rate: int = 16000,
        chunk_size_ms: int = 200,
        real_time_pace: bool = False,
    ):
        self.file_path = Path(file_path)
        self.sample_rate = sample_rate
        self.chunk_size_ms = chunk_size_ms
        self.chunk_samples = int(sample_rate * (chunk_size_ms / 1000.0))
        self.real_time_pace = real_time_pace
        self.audio, _ = load_audio(self.file_path, target_sr=self.sample_rate)

    def stream_sync(self) -> Generator[torch.Tensor, None, None]:
        """Synchronously yield sequential audio chunks."""
        chunk_duration = self.chunk_size_ms / 1000.0
        for chunk in frame_generator(self.audio, self.chunk_samples):
            if self.real_time_pace:
                time.sleep(chunk_duration)
            yield chunk

    async def stream_async(self) -> AsyncGenerator[torch.Tensor, None]:
        """Asynchronously yield sequential audio chunks with simulated delay."""
        chunk_duration = self.chunk_size_ms / 1000.0
        for chunk in frame_generator(self.audio, self.chunk_samples):
            if self.real_time_pace:
                await asyncio.sleep(chunk_duration)
            yield chunk
