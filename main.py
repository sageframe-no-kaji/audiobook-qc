#!/usr/bin/env python3
"""
main.py

Audiobook quality-control scanner entry point.

Usage:
    python main.py /path/to/wav_folder [options]

Run  ``python main.py --help``  for the full option list.
"""

from __future__ import annotations

import argparse
import os
import sys

import audiobook_qc.config as _defaults
from audiobook_qc.config import Config
from audiobook_qc.ranking import rank_events
from audiobook_qc.report_csv import write_csv
from audiobook_qc.report_html import write_html
from audiobook_qc.scanner import scan_directory

ALL_DETECTORS = ["clipping", "click", "plosive", "noise_floor_jump"]


def _build_parser() -> argparse.ArgumentParser:
    d = _defaults  # shorthand for default values in help strings

    parser = argparse.ArgumentParser(
        prog="audiobook_qc",
        description=(
            "Detect and rank audio quality issues in a folder of WAV files.\n"
            "Produces qc_report.csv and review.html in the scanned folder."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py ./chapters\n"
            "  python main.py ./chapters --skip noise_floor_jump\n"
            "  python main.py ./chapters --max-primary 50 --max-secondary 100\n"
            "  python main.py ./chapters --click-threshold 8.0 --plosive-threshold 6.0\n"
            "  python main.py ./chapters --noise-db-threshold 10.0\n"
        ),
    )

    # ── Positional ─────────────────────────────────────────────────────────
    parser.add_argument(
        "folder",
        help="Path to the directory containing .wav files to scan.",
    )

    # ── Skip ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=ALL_DETECTORS,
        default=[],
        metavar="DETECTOR",
        help=(
            "One or more detector names to disable.  "
            f"Choices: {', '.join(ALL_DETECTORS)}."
        ),
    )

    # ── Report caps ────────────────────────────────────────────────────────
    caps = parser.add_argument_group("report caps")
    caps.add_argument(
        "--max-primary",
        type=int,
        default=d.MAX_PRIMARY,
        metavar="N",
        help=f"Maximum number of primary-tier events to export (default: {d.MAX_PRIMARY}).",
    )
    caps.add_argument(
        "--max-secondary",
        type=int,
        default=d.MAX_SECONDARY,
        metavar="N",
        help=f"Maximum number of secondary-tier events to export (default: {d.MAX_SECONDARY}).",
    )

    # ── Clipping ───────────────────────────────────────────────────────────
    clipping = parser.add_argument_group("clipping detection")
    clipping.add_argument(
        "--clipping-threshold",
        type=float,
        default=d.CLIPPING_THRESHOLD,
        metavar="F",
        help=(
            f"Absolute sample-value ceiling above which a sample is clipped "
            f"(default: {d.CLIPPING_THRESHOLD})."
        ),
    )

    # ── Click ──────────────────────────────────────────────────────────────
    click = parser.add_argument_group("click detection")
    click.add_argument(
        "--click-band-low",
        type=float,
        default=float(d.CLICK_BAND[0]),
        metavar="HZ",
        help=(
            "Low-frequency edge of the click detection band in Hz "
            f"(default: {d.CLICK_BAND[0]})."
        ),
    )
    click.add_argument(
        "--click-band-high",
        type=float,
        default=float(d.CLICK_BAND[1]),
        metavar="HZ",
        help=(
            "High-frequency edge of the click detection band in Hz "
            f"(default: {d.CLICK_BAND[1]})."
        ),
    )
    click.add_argument(
        "--click-threshold",
        type=float,
        default=d.CLICK_THRESHOLD_MULTIPLIER,
        metavar="F",
        help=(
            "RMS-to-median-baseline ratio required to flag a click frame "
            f"(default: {d.CLICK_THRESHOLD_MULTIPLIER})."
        ),
    )
    click.add_argument(
        "--click-max-duration-ms",
        type=float,
        default=float(d.CLICK_MAX_DURATION_MS),
        metavar="MS",
        help=(
            "Runs longer than this value in milliseconds are not treated as clicks "
            f"(default: {d.CLICK_MAX_DURATION_MS})."
        ),
    )
    click.add_argument(
        "--click-merge-ms",
        type=float,
        default=float(d.CLICK_MERGE_MS),
        metavar="MS",
        help=(
            "Clicks within this gap in milliseconds are merged into one event "
            f"(default: {d.CLICK_MERGE_MS})."
        ),
    )

    # ── Plosive ────────────────────────────────────────────────────────────
    plosive = parser.add_argument_group("plosive detection")
    plosive.add_argument(
        "--plosive-band-low",
        type=float,
        default=float(d.PLOSIVE_BAND[0]),
        metavar="HZ",
        help=(
            "Low-frequency edge of the plosive detection band in Hz "
            f"(default: {d.PLOSIVE_BAND[0]})."
        ),
    )
    plosive.add_argument(
        "--plosive-band-high",
        type=float,
        default=float(d.PLOSIVE_BAND[1]),
        metavar="HZ",
        help=(
            "High-frequency edge of the plosive detection band in Hz "
            f"(default: {d.PLOSIVE_BAND[1]})."
        ),
    )
    plosive.add_argument(
        "--plosive-threshold",
        type=float,
        default=d.PLOSIVE_THRESHOLD_MULTIPLIER,
        metavar="F",
        help=(
            "RMS-to-median-baseline ratio required to flag a plosive frame "
            f"(default: {d.PLOSIVE_THRESHOLD_MULTIPLIER})."
        ),
    )
    plosive.add_argument(
        "--plosive-max-duration-ms",
        type=float,
        default=float(d.PLOSIVE_MAX_DURATION_MS),
        metavar="MS",
        help=(
            "Runs longer than this value in milliseconds are not classed as plosives "
            f"(default: {d.PLOSIVE_MAX_DURATION_MS})."
        ),
    )
    plosive.add_argument(
        "--plosive-merge-ms",
        type=float,
        default=float(d.PLOSIVE_MERGE_MS),
        metavar="MS",
        help=(
            "Plosives within this gap in milliseconds are merged into one event "
            f"(default: {d.PLOSIVE_MERGE_MS})."
        ),
    )

    # ── Noise floor ────────────────────────────────────────────────────────
    noise = parser.add_argument_group("noise-floor jump detection")
    noise.add_argument(
        "--noise-window-seconds",
        type=float,
        default=d.NOISE_WINDOW_SECONDS,
        metavar="S",
        help=(
            "Length of each RMS measurement window in seconds "
            f"(default: {d.NOISE_WINDOW_SECONDS})."
        ),
    )
    noise.add_argument(
        "--noise-db-threshold",
        type=float,
        default=d.NOISE_DB_DELTA_THRESHOLD,
        metavar="DB",
        help=(
            "Minimum dB difference between adjacent windows to flag a jump "
            f"(default: {d.NOISE_DB_DELTA_THRESHOLD})."
        ),
    )
    noise.add_argument(
        "--noise-min-separation",
        type=float,
        default=d.NOISE_MIN_SEPARATION_SECONDS,
        metavar="S",
        help=(
            "Minimum time in seconds between reported noise-floor events "
            f"(default: {d.NOISE_MIN_SEPARATION_SECONDS})."
        ),
    )

    # ── Framing / DSP ──────────────────────────────────────────────────────
    dsp = parser.add_argument_group("DSP / framing")
    dsp.add_argument(
        "--frame-size",
        type=int,
        default=d.FRAME_SIZE,
        metavar="N",
        help=f"FFT frame size in samples (default: {d.FRAME_SIZE}).",
    )
    dsp.add_argument(
        "--hop-length",
        type=int,
        default=d.HOP_LENGTH,
        metavar="N",
        help=f"Hop length between frames in samples (default: {d.HOP_LENGTH}).",
    )
    dsp.add_argument(
        "--rolling-median-seconds",
        type=float,
        default=d.ROLLING_MEDIAN_SECONDS,
        metavar="S",
        help=(
            "Width of the rolling-median baseline estimator in seconds "
            f"(default: {d.ROLLING_MEDIAN_SECONDS})."
        ),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    folder: str = args.folder
    skip: set[str] = set(args.skip)

    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    cfg = Config(
        frame_size=args.frame_size,
        hop_length=args.hop_length,
        max_primary=args.max_primary,
        max_secondary=args.max_secondary,
        click_band_low=args.click_band_low,
        click_band_high=args.click_band_high,
        click_threshold=args.click_threshold,
        click_max_duration_ms=args.click_max_duration_ms,
        click_merge_ms=args.click_merge_ms,
        plosive_band_low=args.plosive_band_low,
        plosive_band_high=args.plosive_band_high,
        plosive_threshold=args.plosive_threshold,
        plosive_max_duration_ms=args.plosive_max_duration_ms,
        plosive_merge_ms=args.plosive_merge_ms,
        noise_window_seconds=args.noise_window_seconds,
        noise_db_threshold=args.noise_db_threshold,
        noise_min_separation=args.noise_min_separation,
        clipping_threshold=args.clipping_threshold,
        rolling_median_seconds=args.rolling_median_seconds,
    )

    if skip:
        print(f"Skipping detectors: {', '.join(sorted(skip))}")

    print(f"Scanning WAV files in: {folder}\n")

    wav_files, all_events = scan_directory(folder, skip=skip, cfg=cfg)
    total_candidates = len(all_events)

    if not wav_files:
        print(f"No .wav files found in '{folder}'.", file=sys.stderr)
        write_csv([], folder)
        write_html([], folder)
        print(f"Output written to {os.path.join(folder, 'qc_report.csv')}")
        print(f"Review interface written to {os.path.join(folder, 'review.html')}")
        sys.exit(0)

    print(f"\nScanned {len(wav_files)} file(s). Ranking events …\n")

    ranked = rank_events(all_events, cfg=cfg)

    csv_path = write_csv(ranked, folder)
    html_path = write_html(ranked, folder)

    primary_count = sum(
        1 for ev in ranked if ev.get("tier") == "primary"  # type: ignore[typeddict-item]
    )
    secondary_count = sum(
        1 for ev in ranked if ev.get("tier") == "secondary"  # type: ignore[typeddict-item]
    )

    print(f"Total files scanned:             {len(wav_files)}")
    print(f"Total candidate events detected: {total_candidates}")
    print(f"Primary exported:                {primary_count}")
    print(f"Secondary exported:              {secondary_count}")
    print(f"Output written to {csv_path}")
    print(f"Review interface written to {html_path}")


if __name__ == "__main__":
    main()
