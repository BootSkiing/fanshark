"""End-to-end scaffold tests: capture -> features -> analyze."""

import numpy as np

from fanshark.analyze import WorkloadTrace, analyze
from fanshark.capture import CaptureConfig, synth_fan_hum
from fanshark.features import extract


def test_synth_capture_shape():
    cfg = CaptureConfig(duration_s=2.0, sample_rate=16_000)
    sig = synth_fan_hum(cfg)
    assert sig.shape == (32_000,)
    assert np.abs(sig).max() <= 1.0 + 1e-6


def test_feature_extraction_shapes():
    cfg = CaptureConfig(duration_s=3.0, sample_rate=16_000)
    sig = synth_fan_hum(cfg)
    stream = extract(sig, cfg.sample_rate)
    assert stream.spectral.ndim == 2
    assert stream.spectral.shape[0] == len(stream.times)
    assert len(stream.rpm_envelope) == len(stream.times)


def test_analyze_returns_report():
    cfg = CaptureConfig(duration_s=3.0, sample_rate=16_000)
    sig = synth_fan_hum(cfg)
    stream = extract(sig, cfg.sample_rate)
    # Flat "idle" workload -> no correlation expected.
    workload = WorkloadTrace(
        times=np.array([0.0, 3.0]),
        load=np.array([0.0, 0.0]),
    )
    report = analyze(stream, workload)
    assert report.n_frames == len(stream.times)
    assert report.correlation == 0.0  # flat load has zero variance
