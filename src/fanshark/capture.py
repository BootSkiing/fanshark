"""Audio capture of cooling-fan acoustics.

The real capture path reads from a microphone via a backend such as
``sounddevice``/PortAudio. To keep the scaffold dependency-light and runnable
offline, the default backend synthesizes a plausible fan-hum signal (a set of
blade-pass harmonics with slow RPM drift plus broadband noise) so the rest of
the pipeline can be exercised end to end.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass

import numpy as np


@dataclass
class CaptureConfig:
    duration_s: float = 60.0
    sample_rate: int = 48_000
    base_rpm: float = 1_800.0  # fan speed; blade-pass freq = rpm/60 * n_blades
    n_blades: int = 9


def synth_fan_hum(cfg: CaptureConfig, seed: int = 0) -> np.ndarray:
    """Generate a synthetic fan-hum waveform for offline testing.

    Models blade-pass tone + harmonics with slow RPM drift, over broadband
    airflow noise. This is a stand-in for a real microphone capture, not a
    physical simulation of thermal-to-acoustic coupling.
    """
    rng = np.random.default_rng(seed)
    n = int(cfg.duration_s * cfg.sample_rate)
    t = np.arange(n) / cfg.sample_rate

    # Slow RPM drift (thermal throttling would nudge this over seconds).
    drift = 1.0 + 0.03 * np.sin(2 * np.pi * 0.05 * t)
    blade_pass = (cfg.base_rpm / 60.0) * cfg.n_blades * drift

    phase = 2 * np.pi * np.cumsum(blade_pass) / cfg.sample_rate
    signal = np.zeros(n, dtype=np.float64)
    for k, amp in enumerate((1.0, 0.5, 0.25, 0.12), start=1):
        signal += amp * np.sin(k * phase)

    signal += 0.4 * rng.standard_normal(n)  # broadband airflow noise
    signal /= np.max(np.abs(signal)) + 1e-9
    return signal.astype(np.float32)


def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    """Write float samples in [-1, 1] to a 16-bit PCM WAV file."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file into float samples in [-1, 1]."""
    with wave.open(path, "rb") as w:
        sample_rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    pcm16 = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    return pcm16 / 32767.0, sample_rate


def capture(path: str, cfg: CaptureConfig | None = None, backend: str = "synth") -> int:
    """Capture ``cfg.duration_s`` of audio to ``path``. Returns sample count.

    ``backend="synth"`` generates a test signal. A ``"mic"`` backend would hook
    a real recording library here; it is intentionally left unimplemented so the
    scaffold never records without an explicit dependency being wired in.
    """
    cfg = cfg or CaptureConfig()
    if backend == "synth":
        samples = synth_fan_hum(cfg)
    else:
        raise NotImplementedError(
            f"backend {backend!r} not wired in; install a mic backend and implement it"
        )
    write_wav(path, samples, cfg.sample_rate)
    return len(samples)
