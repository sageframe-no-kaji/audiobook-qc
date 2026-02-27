"""
audiobook_qc/detectors.py

All low-level signal-processing utilities and detection algorithms.
Detection math is identical to the original audiobook_qc_scan.py.
"""

from __future__ import annotations

from typing import TypedDict, cast

import numpy as np
import scipy.ndimage
import scipy.signal
import librosa

from audiobook_qc.config import (
    CLIPPING_THRESHOLD,
    Config,
    FRAME_SIZE,
    HOP_LENGTH,
    NOISE_DB_DELTA_THRESHOLD,
    NOISE_MIN_SEPARATION_SECONDS,
    NOISE_WINDOW_SECONDS,
    ROLLING_MEDIAN_SECONDS,
)


# ─────────────────────────────────────────────────────────────────────────────
# EVENT TYPES
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


def detect_clipping(
    audio: np.ndarray,
    sr: int,
    filename: str,
    clipping_threshold: float = CLIPPING_THRESHOLD,
) -> list[Event]:
    """
    Detect continuous clipped regions (|sample| >= clipping_threshold).
    One event per contiguous region.
    Severity = number of clipped samples in the region.
    Timestamp = first clipped sample in seconds.
    """
    clipped = np.abs(audio) >= clipping_threshold
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
    frame_size: int = FRAME_SIZE,
    hop_length: int = HOP_LENGTH,
    rolling_median_seconds: float = ROLLING_MEDIAN_SECONDS,
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
    if len(audio) < frame_size:
        return []

    filtered = bandpass_filter(audio, float(band[0]), float(band[1]), sr)

    rms = compute_frame_rms(filtered, frame_size, hop_length)
    if rms.size == 0:
        return []

    median_win = max(int(rolling_median_seconds * sr / hop_length), 3)
    med_rms = rolling_median(rms, median_win)

    # Avoid division by near-zero baseline (silent or DC-only content)
    safe_med = np.where(med_rms > 1e-9, med_rms, np.inf)
    ratio = rms / safe_med

    flagged = ratio > threshold_mult
    starts, ends = find_runs(flagged)
    if starts.size == 0:
        return []

    # Duration in frames corresponding to the maximum allowed ms
    frame_dur_ms = hop_length / sr * 1000.0
    max_frames = max_duration_ms / frame_dur_ms

    hop_over_sr = hop_length / sr
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


def detect_noise_floor_jumps(
    audio: np.ndarray,
    sr: int,
    filename: str,
    noise_window_seconds: float = NOISE_WINDOW_SECONDS,
    noise_db_threshold: float = NOISE_DB_DELTA_THRESHOLD,
    noise_min_separation: float = NOISE_MIN_SEPARATION_SECONDS,
) -> list[Event]:
    """
    Detect sudden jumps in the noise floor between adjacent RMS windows.

    Pipeline:
      1. Tile audio into non-overlapping windows of noise_window_seconds.
      2. Compute RMS per window, convert to dBFS.
      3. Flag transitions where |ΔdB| >= noise_db_threshold.
      4. Enforce noise_min_separation seconds between kept events.

    Severity = |ΔdB| at the transition.
    Timestamp = start of the earlier of the two windows.
    """
    window_samples = max(int(noise_window_seconds * sr), 1)
    n_windows = len(audio) // window_samples
    if n_windows < 2:
        return []

    segments = audio[: n_windows * window_samples].reshape(n_windows, window_samples)
    rms_vals = np.sqrt(np.mean(segments**2, axis=1))
    db_vals = 20.0 * np.log10(rms_vals + 1e-12)
    delta = np.diff(db_vals)  # shape (n_windows - 1,)

    flagged_idx = np.where(np.abs(delta) >= noise_db_threshold)[0]
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
                    "timestamp": float(idx) * noise_window_seconds,
                    "issue_type": "noise_floor_jump",
                    "severity": float(np.abs(delta[idx])),
                },
            )
        )

    # Enforce minimum separation — greedy chronological pass
    events: list[Event] = [candidates[0]]
    for cand in candidates[1:]:
        if cand["timestamp"] - events[-1]["timestamp"] >= noise_min_separation:
            events.append(cand)

    return events


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPERS — run all detectors on a single audio array
# ─────────────────────────────────────────────────────────────────────────────


def run_all_detectors(
    audio: np.ndarray,
    sr: int,
    filename: str,
    skip: set[str] | None = None,
    cfg: Config | None = None,
) -> list[Event]:
    """Run every detector and return the combined flat event list.

    Parameters
    ----------
    skip : set[str] | None
        Detector names to exclude (e.g. {"noise_floor_jump"}).
    cfg : Config | None
        Runtime parameter overrides.  ``None`` uses compiled-in defaults.
    """
    _skip = skip or set()
    c = cfg or Config()
    events: list[Event] = []
    if "clipping" not in _skip:
        events.extend(detect_clipping(audio, sr, filename, c.clipping_threshold))
    if "click" not in _skip:
        events.extend(
            detect_transients(
                audio,
                sr,
                filename,
                band=(c.click_band_low, c.click_band_high),
                threshold_mult=c.click_threshold,
                max_duration_ms=c.click_max_duration_ms,
                merge_ms=c.click_merge_ms,
                issue_type="click",
                frame_size=c.frame_size,
                hop_length=c.hop_length,
                rolling_median_seconds=c.rolling_median_seconds,
            )
        )
    if "plosive" not in _skip:
        events.extend(
            detect_transients(
                audio,
                sr,
                filename,
                band=(c.plosive_band_low, c.plosive_band_high),
                threshold_mult=c.plosive_threshold,
                max_duration_ms=c.plosive_max_duration_ms,
                merge_ms=c.plosive_merge_ms,
                issue_type="plosive",
                frame_size=c.frame_size,
                hop_length=c.hop_length,
                rolling_median_seconds=c.rolling_median_seconds,
            )
        )
    if "noise_floor_jump" not in _skip:
        events.extend(
            detect_noise_floor_jumps(
                audio,
                sr,
                filename,
                noise_window_seconds=c.noise_window_seconds,
                noise_db_threshold=c.noise_db_threshold,
                noise_min_separation=c.noise_min_separation,
            )
        )
    return events
