"""
audiobook_qc/scanner.py

Directory walker and per-file loader.
Converts audio to mono, normalises peak, and delegates to detectors.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import soundfile as sf

from audiobook_qc.config import Config
from audiobook_qc.detectors import Event, run_all_detectors, to_mono


def scan_file(
    filepath: str,
    skip: set[str] | None = None,
    cfg: Config | None = None,
) -> list[Event]:
    """
    Load one WAV file, convert to mono, run detectors.
    Returns a flat list of event dicts.
    Skips the file gracefully on any read error.
    """
    try:
        audio_raw, sr = sf.read(filepath, dtype="float32", always_2d=True)
    except Exception as exc:
        print(f"  [SKIP] Cannot read '{filepath}': {exc}", file=sys.stderr)
        return []

    audio = to_mono(audio_raw)

    if len(audio) == 0:
        return []

    # Clamp to [-1, 1]; soundfile decodes float PCM in-range, but some
    # non-standard files produced by DAWs may exceed unity.
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak

    basename = os.path.basename(filepath)
    return run_all_detectors(audio, sr, basename, skip=skip, cfg=cfg)


def scan_directory(
    folder: str,
    skip: set[str] | None = None,
    cfg: Config | None = None,
) -> tuple[list[str], list[Event]]:
    """
    Recursively scan *folder* for WAV files, run all detectors on each.

    Parameters
    ----------
    skip : set[str] | None
        Detector names to exclude.

    Returns
    -------
    wav_files : list[str]
        Absolute paths of every .wav file discovered (sorted).
    all_events : list[Event]
        Flat list of every raw event across all files.
    """
    wav_files = sorted(
        glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True)
    )

    all_events: list[Event] = []
    for fp in wav_files:
        print(f"  Processing: {os.path.basename(fp)}")
        all_events.extend(scan_file(fp, skip=skip, cfg=cfg))

    return wav_files, all_events
