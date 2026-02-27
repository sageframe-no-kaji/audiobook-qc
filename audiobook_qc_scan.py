#!/usr/bin/env python3
"""
audiobook_qc_scan.py

Audiobook quality control scanner — detection and ranked logging only.
Detects clipping, clicks, plosives, and noise floor jumps in WAV files.
Produces a severity-ranked CSV report with primary and secondary tiers.

Usage:
    python audiobook_qc_scan.py /path/to/wav_folder
"""

from __future__ import annotations

from typing import TypedDict, cast

import glob
import os
import sys

import librosa
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal
import soundfile as sf

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (edit these to tune detection)
# ─────────────────────────────────────────────────────────────────────────────

FRAME_SIZE = 2048
HOP_LENGTH = 512

MAX_PRIMARY = 75
MAX_SECONDARY = 200

CLICK_BAND = (2500, 10000)  # Hz
PLOSIVE_BAND = (20, 180)  # Hz

CLICK_THRESHOLD_MULTIPLIER = 5.0
PLOSIVE_THRESHOLD_MULTIPLIER = 4.0

CLICK_MAX_DURATION_MS = 15
PLOSIVE_MAX_DURATION_MS = 80

CLICK_MERGE_MS = 150
PLOSIVE_MERGE_MS = 200

NOISE_WINDOW_SECONDS = 1.0
NOISE_DB_DELTA_THRESHOLD = 14.0

CLIPPING_THRESHOLD = 0.998

# Minimum gap enforced between noise-floor-jump events
NOISE_MIN_SEPARATION_SECONDS = 10.0

# Width of the rolling baseline estimator — approximately 1 second of frames
ROLLING_MEDIAN_SECONDS = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# EVENT TYPE
# ─────────────────────────────────────────────────────────────────────────────


class Event(TypedDict):
    filename: str
    timestamp: float
    issue_type: str
    severity: float


class _WorkEvent(Event, total=False):
    """Internal merge-only type: inherits all required Event fields plus the
    optional end_time tracking field used during gap-merging."""

    end_time: float


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Return a 1-D float32 mono array from any channel count."""
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


def bandpass_filter(
    audio: np.ndarray, lowcut: float, highcut: float, sr: int, order: int = 4
) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter via second-order sections (stable).
    Gracefully degrades when cutoffs exceed the valid Nyquist range.
    """
    nyq = sr / 2.0
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 1.0 - 1e-6)
    if low >= high:
        return audio.copy()
    sos = scipy.signal.butter(order, [low, high], btype="band", output="sos")
    filtered: np.ndarray = np.asarray(scipy.signal.sosfilt(sos, audio))
    return filtered.astype(np.float32)


def compute_frame_rms(
    audio: np.ndarray, frame_size: int, hop_length: int
) -> np.ndarray:
    """
    Compute per-frame RMS energy using librosa.
    Returns 1-D array of shape (n_frames,).
    """
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_size, hop_length=hop_length, center=True
    )
    return rms[0].astype(np.float32)


def rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Element-wise rolling median with a symmetric window.
    scipy.ndimage.median_filter is O(N * W) but avoids O(N²) accumulation.
    """
    w = max(window, 3)
    return scipy.ndimage.median_filter(
        arr.astype(np.float64), size=w, mode="reflect"
    ).astype(np.float32)


def find_runs(boolean_array: np.ndarray):
    """
    Return (starts, ends) — 1-D integer arrays of inclusive run boundaries
    where boolean_array is True.
    """
    arr = np.asarray(boolean_array, dtype=np.bool_)
    if arr.size == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    padded = np.concatenate(([False], arr, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return starts, ends


def merge_events_by_gap(
    events: list[_WorkEvent], merge_gap_s: float
) -> list[_WorkEvent]:
    """
    Merge events of the same type that are separated by <= merge_gap_s seconds.
    The merged event inherits the earliest timestamp and the maximum severity.
    Input events must all carry 'timestamp', 'end_time', and 'severity' keys.
    """
    if not events:
        return []
    events = sorted(events, key=lambda e: e["timestamp"])
    merged: list[_WorkEvent] = [cast(_WorkEvent, {**events[0]})]
    for ev in events[1:]:
        prev = merged[-1]
        gap = ev["timestamp"] - prev.get("end_time", prev["timestamp"])
        if gap <= merge_gap_s:
            prev["end_time"] = max(
                prev.get("end_time", prev["timestamp"]),
                ev.get("end_time", ev["timestamp"]),
            )
            prev["severity"] = max(prev["severity"], ev["severity"])
        else:
            merged.append(cast(_WorkEvent, {**ev}))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def detect_clipping(audio: np.ndarray, sr: int, filename: str) -> list[Event]:
    """
    Detect continuous clipped regions (|sample| >= CLIPPING_THRESHOLD).
    One event per contiguous region.
    Severity = number of clipped samples in the region.
    Timestamp = first clipped sample in seconds.
    """
    clipped = np.abs(audio) >= CLIPPING_THRESHOLD
    starts, ends = find_runs(clipped)
    events: list[Event] = []
    for s, e in zip(starts, ends):
        count = int(e - s + 1)
        timestamp = float(s) / sr
        events.append(
            cast(
                Event,
                {
                    "filename": filename,
                    "timestamp": timestamp,
                    "issue_type": "clipping",
                    "severity": float(count),
                },
            )
        )
    return events


def detect_transients(
    audio: np.ndarray,
    sr: int,
    filename: str,
    band: tuple[float, float],
    threshold_mult: float,
    max_duration_ms: float,
    merge_ms: float,
    issue_type: str,
) -> list[Event]:
    """
    Generic short-transient detector used for both clicks and plosives.

    Pipeline:
      1. Band-pass filter to isolate frequency band.
      2. Compute short-time RMS.
      3. Compute rolling-median baseline.
      4. Flag frames where RMS > threshold_mult × baseline.
      5. Discard flagged runs whose duration exceeds max_duration_ms.
      6. Merge nearby events within merge_ms.

    Severity = peak (frame_rms / baseline_rms) across the event window.
    """
    if len(audio) < FRAME_SIZE:
        return []

    filtered = bandpass_filter(audio, float(band[0]), float(band[1]), sr)

    rms = compute_frame_rms(filtered, FRAME_SIZE, HOP_LENGTH)
    if rms.size == 0:
        return []

    median_win = max(int(ROLLING_MEDIAN_SECONDS * sr / HOP_LENGTH), 3)
    med_rms = rolling_median(rms, median_win)

    # Avoid division by near-zero baseline (silent or DC-only content)
    safe_med = np.where(med_rms > 1e-9, med_rms, np.inf)
    ratio = rms / safe_med

    flagged = ratio > threshold_mult
    starts, ends = find_runs(flagged)
    if starts.size == 0:
        return []

    # Duration in frames corresponding to the maximum allowed ms
    frame_dur_ms = HOP_LENGTH / sr * 1000.0
    max_frames = max_duration_ms / frame_dur_ms

    hop_over_sr = HOP_LENGTH / sr
    candidates: list[_WorkEvent] = []
    for s, e in zip(starts, ends):
        span = int(e - s + 1)
        if span > max_frames:
            continue  # too long to be click/plosive
        peak_ratio = float(np.max(ratio[s : e + 1]))
        t_start = float(s) * hop_over_sr
        t_end = float(e) * hop_over_sr
        candidates.append(
            cast(
                _WorkEvent,
                {
                    "filename": filename,
                    "timestamp": t_start,
                    "end_time": t_end,
                    "issue_type": issue_type,
                    "severity": peak_ratio,
                },
            )
        )

    if not candidates:
        return []

    merge_gap_s = merge_ms / 1000.0
    merged = merge_events_by_gap(candidates, merge_gap_s)

    # Strip the internal end_time tracking key before returning
    result: list[Event] = []
    for ev in merged:
        ev.pop("end_time", None)
        result.append(
            cast(
                Event,
                {
                    "filename": ev["filename"],
                    "timestamp": ev["timestamp"],
                    "issue_type": ev["issue_type"],
                    "severity": ev["severity"],
                },
            )
        )
    return result


def detect_noise_floor_jumps(audio: np.ndarray, sr: int, filename: str) -> list[Event]:
    """
    Detect sudden jumps in the noise floor between adjacent RMS windows.

    Pipeline:
      1. Tile audio into non-overlapping windows of NOISE_WINDOW_SECONDS.
      2. Compute RMS per window, convert to dBFS.
      3. Flag transitions where |ΔdB| >= NOISE_DB_DELTA_THRESHOLD.
      4. Enforce NOISE_MIN_SEPARATION_SECONDS between kept events.

    Severity = |ΔdB| at the transition.
    Timestamp = start of the earlier of the two windows.
    """
    window_samples = max(int(NOISE_WINDOW_SECONDS * sr), 1)
    n_windows = len(audio) // window_samples
    if n_windows < 2:
        return []

    segments = audio[: n_windows * window_samples].reshape(n_windows, window_samples)
    rms_vals = np.sqrt(np.mean(segments**2, axis=1))
    db_vals = 20.0 * np.log10(rms_vals + 1e-12)
    delta = np.diff(db_vals)  # shape (n_windows - 1,)

    flagged_idx = np.where(np.abs(delta) >= NOISE_DB_DELTA_THRESHOLD)[0]
    if flagged_idx.size == 0:
        return []

    # Build candidate list (one per flagged transition)
    candidates: list[Event] = []
    for idx in flagged_idx:
        candidates.append(
            cast(
                Event,
                {
                    "filename": filename,
                    "timestamp": float(idx) * NOISE_WINDOW_SECONDS,
                    "issue_type": "noise_floor_jump",
                    "severity": float(np.abs(delta[idx])),
                },
            )
        )

    # Enforce minimum separation — greedy chronological pass
    events: list[Event] = [candidates[0]]
    for cand in candidates[1:]:
        if cand["timestamp"] - events[-1]["timestamp"] >= NOISE_MIN_SEPARATION_SECONDS:
            events.append(cand)

    return events


# ─────────────────────────────────────────────────────────────────────────────
# FILE SCANNER
# ─────────────────────────────────────────────────────────────────────────────


def scan_file(filepath: str) -> list[Event]:
    """
    Load one WAV file, convert to mono, run all four detectors.
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

    events: list[Event] = []
    events.extend(detect_clipping(audio, sr, basename))
    events.extend(
        detect_transients(
            audio,
            sr,
            basename,
            band=CLICK_BAND,
            threshold_mult=CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=CLICK_MAX_DURATION_MS,
            merge_ms=CLICK_MERGE_MS,
            issue_type="click",
        )
    )
    events.extend(
        detect_transients(
            audio,
            sr,
            basename,
            band=PLOSIVE_BAND,
            threshold_mult=PLOSIVE_THRESHOLD_MULTIPLIER,
            max_duration_ms=PLOSIVE_MAX_DURATION_MS,
            merge_ms=PLOSIVE_MERGE_MS,
            issue_type="plosive",
        )
    )
    events.extend(detect_noise_floor_jumps(audio, sr, basename))

    return events


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python audiobook_qc_scan.py /path/to/wav_folder", file=sys.stderr)
        sys.exit(1)

    folder = sys.argv[1]

    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    wav_files = sorted(glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True))

    output_path = "qc_report.csv"
    header_cols = [
        "rank_global",
        "tier",
        "filename",
        "timestamp_seconds",
        "timestamp_minsec",
        "issue_type",
        "severity_score",
    ]

    if not wav_files:
        print(f"No .wav files found in '{folder}'.", file=sys.stderr)
        pd.DataFrame(columns=header_cols).to_csv(output_path, index=False)
        print(f"Output written to {output_path}")
        sys.exit(0)

    print(f"Scanning {len(wav_files)} file(s) in: {folder}\n")

    all_events: list[Event] = []
    for fp in wav_files:
        print(f"  Processing: {os.path.basename(fp)}")
        all_events.extend(scan_file(fp))

    total_candidates = len(all_events)

    # ── Global ranking by severity (descending) ───────────────────────────────
    all_events.sort(key=lambda e: e["severity"], reverse=True)

    keep = MAX_PRIMARY + MAX_SECONDARY
    all_events = all_events[:keep]

    # ── Assign tiers and build output rows ────────────────────────────────────
    rows = []
    for rank, ev in enumerate(all_events, start=1):
        tier = "primary" if rank <= MAX_PRIMARY else "secondary"
        ts = round(ev["timestamp"], 3)
        mins, secs = divmod(ts, 60)
        minsec = f"{int(mins)}:{secs:06.3f}"
        rows.append(
            {
                "rank_global": rank,
                "tier": tier,
                "filename": ev["filename"],
                "timestamp_seconds": ts,
                "timestamp_minsec": minsec,
                "issue_type": ev["issue_type"],
                "severity_score": round(ev["severity"], 4),
            }
        )

    df = pd.DataFrame(rows, columns=header_cols)
    df.to_csv(output_path, index=False)

    primary_count = sum(1 for r in rows if r["tier"] == "primary")
    secondary_count = sum(1 for r in rows if r["tier"] == "secondary")

    print(f"\nTotal files scanned:             {len(wav_files)}")
    print(f"Total candidate events detected: {total_candidates}")
    print(f"Primary exported:                {primary_count}")
    print(f"Secondary exported:              {secondary_count}")
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
