"""Feature extraction: turn raw fan audio into an acoustic-thermal feature stream.

Two feature families are extracted per frame:

* a coarse **spectral** representation (log-magnitude STFT bands), and
* an **RPM/blade-pass envelope** — the slowly varying fundamental that tracks
  fan speed, which is the component most plausibly coupled to GPU thermal load.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal


@dataclass
class FeatureConfig:
    frame_s: float = 0.25       # analysis frame length
    hop_s: float = 0.125        # frame hop
    n_bands: int = 32           # log-spaced spectral bands
    rpm_lowpass_hz: float = 2.0  # envelope smoothing for the RPM track


@dataclass
class FeatureStream:
    times: np.ndarray            # (n_frames,) frame center times, seconds
    spectral: np.ndarray         # (n_frames, n_bands) log-magnitude bands
    rpm_envelope: np.ndarray     # (n_frames,) blade-pass fundamental track (Hz)
    sample_rate: int

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            times=self.times,
            spectral=self.spectral,
            rpm_envelope=self.rpm_envelope,
            sample_rate=self.sample_rate,
        )

    @staticmethod
    def load(path: str) -> "FeatureStream":
        d = np.load(path)
        return FeatureStream(
            times=d["times"],
            spectral=d["spectral"],
            rpm_envelope=d["rpm_envelope"],
            sample_rate=int(d["sample_rate"]),
        )


def _log_bands(mag: np.ndarray, freqs: np.ndarray, n_bands: int) -> np.ndarray:
    """Aggregate an STFT magnitude column into log-spaced frequency bands."""
    lo, hi = 20.0, freqs[-1]
    edges = np.geomspace(lo, hi, n_bands + 1)
    out = np.empty(n_bands, dtype=np.float32)
    for i in range(n_bands):
        sel = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i] = mag[sel].mean() if sel.any() else 0.0
    return np.log1p(out)


def extract(samples: np.ndarray, sample_rate: int, cfg: FeatureConfig | None = None) -> FeatureStream:
    cfg = cfg or FeatureConfig()
    nperseg = int(cfg.frame_s * sample_rate)
    hop = int(cfg.hop_s * sample_rate)
    noverlap = max(0, nperseg - hop)

    freqs, times, zxx = sp_signal.stft(
        samples, fs=sample_rate, nperseg=nperseg, noverlap=noverlap
    )
    mag = np.abs(zxx)  # (n_freqs, n_frames)

    spectral = np.stack(
        [_log_bands(mag[:, j], freqs, cfg.n_bands) for j in range(mag.shape[1])]
    )

    # RPM/blade-pass track: peak frequency per frame, then low-pass smoothed.
    peak_hz = freqs[np.argmax(mag, axis=0)]
    if len(peak_hz) > 3:
        b, a = sp_signal.butter(2, cfg.rpm_lowpass_hz, fs=1.0 / cfg.hop_s)
        rpm_envelope = sp_signal.filtfilt(b, a, peak_hz)
    else:
        rpm_envelope = peak_hz

    return FeatureStream(
        times=times.astype(np.float32),
        spectral=spectral.astype(np.float32),
        rpm_envelope=rpm_envelope.astype(np.float32),
        sample_rate=sample_rate,
    )
