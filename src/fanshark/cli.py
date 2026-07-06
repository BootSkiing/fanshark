"""fanshark command-line interface."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .analyze import WorkloadTrace, analyze
from .capture import CaptureConfig, capture, read_wav
from .features import FeatureConfig, FeatureStream, extract


def _cmd_capture(args: argparse.Namespace) -> int:
    cfg = CaptureConfig(duration_s=args.duration, sample_rate=args.rate)
    n = capture(args.out, cfg, backend=args.backend)
    print(f"captured {n} samples ({args.duration:.1f}s) -> {args.out}")
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    samples, sr = read_wav(args.wav)
    stream = extract(samples, sr, FeatureConfig())
    stream.save(args.out)
    print(f"extracted {len(stream.times)} frames -> {args.out}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    stream = FeatureStream.load(args.features)
    workload = WorkloadTrace.from_json(args.workload)
    report = analyze(stream, workload)
    md = report.to_markdown()
    if args.report:
        with open(args.report, "w") as f:
            f.write(md)
        print(f"wrote report -> {args.report}")
    print(md)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fanshark",
        description="Measure acoustic side-channel leakage from GPU cooling fans (research PoC).",
    )
    p.add_argument("--version", action="version", version=f"fanshark {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("capture", help="record fan acoustics to a WAV file")
    c.add_argument("--duration", type=float, default=60.0, help="seconds to capture")
    c.add_argument("--rate", type=int, default=48_000, help="sample rate (Hz)")
    c.add_argument("--backend", default="synth", choices=["synth", "mic"])
    c.add_argument("--out", required=True, help="output WAV path")
    c.set_defaults(func=_cmd_capture)

    f = sub.add_parser("features", help="extract acoustic-thermal features from a WAV")
    f.add_argument("wav", help="input WAV file")
    f.add_argument("--out", required=True, help="output .npz feature file")
    f.set_defaults(func=_cmd_features)

    a = sub.add_parser("analyze", help="estimate leakage vs. a workload trace")
    a.add_argument("features", help="input .npz feature file")
    a.add_argument("--workload", required=True, help="workload trace JSON (times, load)")
    a.add_argument("--report", help="write a markdown report to this path")
    a.set_defaults(func=_cmd_analyze)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
