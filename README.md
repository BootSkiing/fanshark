# fanshark

> **Research proof-of-concept.** `fanshark` is a scaffold for exploring the
> **GLARE** (Gradient-Latent Acoustic Resonance Exfiltration) side-channel
> described in [`abstract.txt`](./abstract.txt). It is a defensive-research and
> educational tool for measuring whether an inference host leaks *any*
> recoverable signal through its cooling-fan acoustics. It does **not** recover
> model weights, and the headline claims in the abstract (91.7% weight recovery)
> are not reproduced or endorsed here — see [Reality check](#reality-check).

## What it actually does

`fanshark` implements the *measurable* half of the pipeline: capture fan/airflow
audio, turn it into a spectral–thermal feature stream, and correlate that stream
against a known inference workload to quantify **leakage** (mutual information /
correlation), not to reconstruct parameters.

```
 microphone ──▶ capture ──▶ features ──▶ correlate ──▶ leakage report
 (fan hum)      (WAV/PCM)   (mel + RPM   (vs. workload  (bits/sec, MI,
                            envelope)     trace)         confidence)
```

## Install

```bash
python -m pip install -e .[dev]
```

## Usage

```bash
# 1. Record fan acoustics while a target workload runs
fanshark capture --duration 60 --out session.wav

# 2. Extract acoustic-thermal features
fanshark features session.wav --out session.npz

# 3. Correlate against a workload trace to estimate leakage
fanshark analyze session.npz --workload trace.json --report report.md
```

Run `fanshark --help` for the full CLI.

## Reality check

The abstract is a (deliberately over-claimed) conference-style piece. Before
building on it, note:

- **Channel capacity.** Cooling fans are a low-bandwidth, heavily low-pass
  thermal actuator. Full weight recovery from fan RPM is not physically
  plausible at the accuracy claimed.
- **Fabricated references.** `RFC 9981`, `CVE-2025-88213`, and
  `CVE-2026-01472` do not correspond to real published advisories/RFCs.
- **Legit core.** Acoustic/EM/power side-channels *are* real research areas.
  What is realistic is detecting coarse workload signatures (idle vs. busy,
  batch size class, model-family fingerprints) — which is what this tool
  measures.

## Ethics & scope

Use only against systems you own or are explicitly authorized to test.
Intended for defensive assessment ("does my inference host leak a signal?")
and education. See [`SECURITY.md`](./SECURITY.md).

## License

MIT — see [`LICENSE`](./LICENSE).
