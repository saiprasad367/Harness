"""Audio loading, resampling, normalization, and window framing."""
from pathlib import Path
from typing import Generator, Tuple
import math
import numpy as np
import scipy.signal
import soundfile as sf
import torch


def load_audio(
    file_path: str | Path,
    target_sr: int = 16000,
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file, convert to mono, resample to target sample rate, and normalize.
    Returns:
        audio_tensor: 1D torch.Tensor of shape (samples,), float32 in [-1.0, 1.0]
        sample_rate: target_sr
    """
    path_str = str(file_path)
    
    # Read audio using soundfile (reliable across formats)
    data, orig_sr = sf.read(path_str, dtype="float32")
    
    # Handle multi-channel to mono conversion
    if data.ndim > 1:
        data = np.mean(data, axis=1)
        
    # Resample using scipy if needed
    if orig_sr != target_sr:
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        data = scipy.signal.resample_poly(data, up, down).astype(np.float32)
        
    # Peak normalization
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
        
    tensor = torch.from_numpy(data.astype(np.float32))
    return tensor, target_sr


def resample_audio(
    audio: torch.Tensor,
    orig_sr: int,
    target_sr: int = 16000,
) -> torch.Tensor:
    """Resample an in-memory 1D audio tensor."""
    if orig_sr == target_sr:
        return audio
    data = audio.detach().cpu().numpy()
    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = scipy.signal.resample_poly(data, up, down).astype(np.float32)
    return torch.from_numpy(resampled)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize numpy audio array to range [-1.0, 1.0]."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def frame_generator(
    audio: torch.Tensor,
    frame_size_samples: int,
    step_samples: int | None = None,
) -> Generator[torch.Tensor, None, None]:
    """Yield sequential non-overlapping or overlapping frames from an audio tensor."""
    if step_samples is None:
        step_samples = frame_size_samples
        
    total_samples = audio.shape[0]
    for start in range(0, total_samples, step_samples):
        end = start + frame_size_samples
        if end <= total_samples:
            yield audio[start:end]
        else:
            pad_amount = end - total_samples
            frame = audio[start:]
            yield torch.nn.functional.pad(frame, (0, pad_amount))
