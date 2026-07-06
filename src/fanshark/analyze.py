"""Leakage analysis: correlate the acoustic feature stream against a known workload.

This is the honest core of the tool. Given a feature stream and a *workload
trace* (when the target was busy vs. idle, or its per-interval load level), we
estimate how much of the workload is recoverable from the acoustics:

* Pearson correlation between the RPM envelope and the load trace,
* a simple mutual-information estimate (binned) as a leakage proxy, and
* an upper-bound channel-rate estimate in bits/second.

It does **not** attempt weight recovery — that is out of scope and, per the
README, not physically plausible through this channel.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from .features import FeatureStream


@dataclass
class WorkloadTrace:
    """Ground-truth workload activity, sampled over time.

    ``times`` and ``load`` are equal-length arrays; ``load`` is a scalar
    activity level per timestamp (e.g., 0=idle .. 1=saturated).
    """

    times: np.ndarray
    load: np.ndarray

    @staticmethod
    def from_json(path: str) -> "WorkloadTrace":
        with open(path) as f:
            d = json.load(f)
        return WorkloadTrace(
            times=np.asarray(d["times"], dtype=np.float64),
            load=np.asarray(d["load"], dtype=np.float64),
        )


@dataclass
class LeakageReport:
    correlation: float
    mutual_info_bits: float
    channel_rate_bits_per_s: float
    n_frames: int
    duration_s: float

    def to_markdown(self) -> str:
        verdict = (
            "SIGNAL DETECTED" if abs(self.correlation) >= 0.3 else "no meaningful signal"
        )
        return (
            "# fanshark leakage report\n\n"
            f"- **Verdict:** {verdict}\n"
            f"- Frames analyzed: {self.n_frames}\n"
            f"- Duration: {self.duration_s:.1f} s\n"
            f"- RPM↔load correlation: {self.correlation:+.3f}\n"
            f"- Mutual information: {self.mutual_info_bits:.3f} bits/frame\n"
            f"- Est. channel rate: {self.channel_rate_bits_per_s:.3f} bits/s\n\n"
            "> Reminder: this measures workload leakage, not weight recovery. "
            "A high correlation means the host's acoustics track *what it is "
            "doing*, not *what it knows*.\n"
        )


def _resample_to(times_src: np.ndarray, values: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    return np.interp(times_dst, times_src, values)


def _mutual_info_binned(x: np.ndarray, y: np.ndarray, bins: int = 8) -> float:
    """Plug-in mutual-information estimate over ``bins`` equal-frequency bins (bits)."""
    if len(x) < bins * 2:
        return 0.0
    xb = np.digitize(x, np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    yb = np.digitize(y, np.quantile(y, np.linspace(0, 1, bins + 1)[1:-1]))
    joint = np.histogram2d(xb, yb, bins=bins)[0]
    joint = joint / joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log2(joint[nz] / (px @ py)[nz])))


def analyze(stream: FeatureStream, workload: WorkloadTrace) -> LeakageReport:
    load_on_frames = _resample_to(workload.times, workload.load, stream.times)

    rpm = stream.rpm_envelope
    if np.std(rpm) < 1e-9 or np.std(load_on_frames) < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(rpm, load_on_frames)[0, 1])

    mi = _mutual_info_binned(rpm, load_on_frames)

    duration = float(stream.times[-1] - stream.times[0]) if len(stream.times) > 1 else 0.0
    frame_rate = (len(stream.times) - 1) / duration if duration > 0 else 0.0
    rate = mi * frame_rate

    return LeakageReport(
        correlation=corr,
        mutual_info_bits=mi,
        channel_rate_bits_per_s=rate,
        n_frames=len(stream.times),
        duration_s=duration,
    )


def report_to_json(report: LeakageReport) -> str:
    return json.dumps(asdict(report), indent=2)
