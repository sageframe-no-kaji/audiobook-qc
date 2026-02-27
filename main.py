#!/usr/bin/env python3
"""
main.py

Audiobook quality-control scanner entry point.

Usage:
    python main.py /path/to/wav_folder
"""

from __future__ import annotations

import argparse
import sys

from audiobook_qc.ranking import rank_events
from audiobook_qc.report_csv import write_csv
from audiobook_qc.report_html import write_html
from audiobook_qc.scanner import scan_directory

ALL_DETECTORS = ["clipping", "click", "plosive", "noise_floor_jump"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audiobook QC scanner",
    )
    parser.add_argument("folder", help="Path to folder containing .wav files")
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=ALL_DETECTORS,
        default=[],
        metavar="DETECTOR",
        help=(
            "Detector(s) to skip. "
            f"Choices: {', '.join(ALL_DETECTORS)}"
        ),
    )
    args = parser.parse_args()

    import os

    folder: str = args.folder
    skip: set[str] = set(args.skip)

    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    if skip:
        print(f"Skipping detectors: {', '.join(sorted(skip))}")

    print(f"Scanning WAV files in: {folder}\n")

    wav_files, all_events = scan_directory(folder, skip=skip)
    total_candidates = len(all_events)

    if not wav_files:
        print(f"No .wav files found in '{folder}'.", file=sys.stderr)
        # Write empty outputs so callers always have files to inspect
        write_csv([], folder)
        write_html([], folder)
        print(f"Output written to {os.path.join(folder, 'qc_report.csv')}")
        print(f"Review interface written to {os.path.join(folder, 'review.html')}")
        sys.exit(0)

    print(f"\nScanned {len(wav_files)} file(s). Ranking events …\n")

    ranked = rank_events(all_events)

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
