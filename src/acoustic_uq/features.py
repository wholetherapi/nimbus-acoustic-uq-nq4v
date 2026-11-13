"""Log-mel feature extraction for field recordings."""
from __future__ import annotations

import numpy as np


def log_mel_features(waveform: np.ndarray, sample_rate: int = 22050, n_mels: int = 64) -> np.ndarray:
    """Compute a normalised log-mel spectrogram for a mono waveform."""
    if waveform.ndim != 1:
        raise ValueError("expected a mono waveform")
    frame = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)
    if len(waveform) < frame:
        waveform = np.pad(waveform, (0, frame - len(waveform)))
    frames = []
    for start in range(0, len(waveform) - frame + 1, hop):
        window = waveform[start : start + frame] * np.hanning(frame)
        spectrum = np.abs(np.fft.rfft(window)) ** 2
        bands = np.array_split(spectrum, n_mels)
        frames.append([float(np.mean(band)) for band in bands])
    mel = np.asarray(frames, dtype=np.float32)
    mel = np.log(mel + 1e-8)
    return (mel - mel.mean()) / (mel.std() + 1e-8)
