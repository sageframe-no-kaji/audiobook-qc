"""
tests/test_audiobook_qc.py

Full test suite for the refactored audiobook_qc package.

Run with:
    .venv/bin/pytest tests/test_audiobook_qc.py -v \
        --cov=audiobook_qc --cov=main --cov-report=term-missing
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

# Ensure project root is on path regardless of invocation directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

import audiobook_qc.config as cfg  # noqa: E402
from audiobook_qc.detectors import (  # noqa: E402
    Event,
    _WorkEvent,
    bandpass_filter,
    compute_frame_rms,
    detect_clipping,
    detect_noise_floor_jumps,
    detect_transients,
    find_runs,
    merge_events_by_gap,
    rolling_median,
    run_all_detectors,
    to_mono,
)
from audiobook_qc.ranking import rank_events  # noqa: E402
from audiobook_qc.report_csv import HEADER_COLS, write_csv  # noqa: E402
from audiobook_qc.report_html import write_html  # noqa: E402
from audiobook_qc.scanner import scan_directory, scan_file  # noqa: E402

TEST_FILES_DIR = Path(__file__).parent.parent / "test_files"

SR = 44100
RNG = np.random.default_rng(0)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES / HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _quiet_noise(duration_s: float, sr: int = SR) -> np.ndarray:
    n = int(duration_s * sr)
    return (RNG.standard_normal(n) * 1e-4).astype(np.float32)


def _sine_burst(freq_hz: float, duration_s: float, amp: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _inject(base: np.ndarray, burst: np.ndarray, t_seconds: float, sr: int = SR) -> np.ndarray:
    audio = base.copy()
    start = int(t_seconds * sr)
    end = min(start + len(burst), len(audio))
    audio[start:end] += burst[: end - start]
    return audio


def _write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    sf.write(str(path), audio, sr, subtype="PCM_16")


def _make_event(
    filename: str = "t.wav",
    timestamp: float = 1.0,
    issue_type: str = "clipping",
    severity: float = 10.0,
) -> Event:
    return cast(
        Event,
        {
            "filename": filename,
            "timestamp": timestamp,
            "issue_type": issue_type,
            "severity": severity,
        },
    )


def _make_work_event(ts: float, end: float, severity: float = 1.0) -> _WorkEvent:
    return cast(
        _WorkEvent,
        {
            "filename": "test.wav",
            "timestamp": ts,
            "end_time": end,
            "issue_type": "click",
            "severity": severity,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────


class TestConfig:
    def test_frame_size(self):
        assert cfg.FRAME_SIZE == 2048

    def test_hop_length(self):
        assert cfg.HOP_LENGTH == 512

    def test_max_primary(self):
        assert cfg.MAX_PRIMARY == 75

    def test_max_secondary(self):
        assert cfg.MAX_SECONDARY == 200

    def test_click_band(self):
        assert cfg.CLICK_BAND == (2500, 10000)

    def test_plosive_band(self):
        assert cfg.PLOSIVE_BAND == (20, 180)

    def test_click_threshold_multiplier(self):
        assert cfg.CLICK_THRESHOLD_MULTIPLIER == pytest.approx(5.0)

    def test_plosive_threshold_multiplier(self):
        assert cfg.PLOSIVE_THRESHOLD_MULTIPLIER == pytest.approx(4.0)

    def test_click_max_duration_ms(self):
        assert cfg.CLICK_MAX_DURATION_MS == 15

    def test_plosive_max_duration_ms(self):
        assert cfg.PLOSIVE_MAX_DURATION_MS == 80

    def test_click_merge_ms(self):
        assert cfg.CLICK_MERGE_MS == 150

    def test_plosive_merge_ms(self):
        assert cfg.PLOSIVE_MERGE_MS == 200

    def test_noise_window_seconds(self):
        assert cfg.NOISE_WINDOW_SECONDS == pytest.approx(1.0)

    def test_noise_db_delta_threshold(self):
        assert cfg.NOISE_DB_DELTA_THRESHOLD == pytest.approx(14.0)

    def test_noise_min_separation_seconds(self):
        assert cfg.NOISE_MIN_SEPARATION_SECONDS == pytest.approx(10.0)

    def test_clipping_threshold(self):
        assert cfg.CLIPPING_THRESHOLD == pytest.approx(0.998)

    def test_rolling_median_seconds(self):
        assert cfg.ROLLING_MEDIAN_SECONDS == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# DETECTORS — to_mono
# ─────────────────────────────────────────────────────────────────────────────


class TestToMono:
    def test_already_mono_returns_float32(self):
        a = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        out = to_mono(a)
        assert out.ndim == 1
        assert out.dtype == np.float32

    def test_stereo_averages_channels(self):
        left = np.array([0.4, 0.8], dtype=np.float32)
        right = np.array([0.0, 0.4], dtype=np.float32)
        stereo = np.stack([left, right], axis=1)
        out = to_mono(stereo)
        expected = np.array([0.2, 0.6], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-7)

    def test_stereo_output_dtype_is_float32(self):
        stereo = RNG.standard_normal((100, 2)).astype(np.float64)
        out = to_mono(stereo)
        assert out.dtype == np.float32

    def test_multi_channel(self):
        arr = np.array([[0.3, 0.6, 0.9]], dtype=np.float32)
        out = to_mono(arr)
        np.testing.assert_allclose(out[0], 0.6, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# DETECTORS — bandpass_filter
# ─────────────────────────────────────────────────────────────────────────────


class TestBandpassFilter:
    def test_output_shape_matches_input(self):
        audio = _quiet_noise(1.0)
        out = bandpass_filter(audio, 2500.0, 10000.0, SR)
        assert out.shape == audio.shape

    def test_output_dtype_is_float32(self):
        audio = _quiet_noise(0.5)
        out = bandpass_filter(audio, 2500.0, 10000.0, SR)
        assert out.dtype == np.float32

    def test_degenerate_returns_copy(self):
        audio = _quiet_noise(0.5)
        out = bandpass_filter(audio, 25000.0, 100.0, SR)
        np.testing.assert_array_equal(out, audio)

    def test_dc_attenuated(self):
        dc = np.ones(SR, dtype=np.float32) * 0.5
        out = bandpass_filter(dc, 2500.0, 10000.0, SR)
        assert np.abs(out).mean() < 0.01

    def test_passband_preserved(self):
        tone = _sine_burst(5000.0, 0.5, amp=1.0)
        out = bandpass_filter(tone, 2500.0, 10000.0, SR)
        rms_in = float(np.sqrt(np.mean(tone[SR // 4:] ** 2)))
        rms_out = float(np.sqrt(np.mean(out[SR // 4:] ** 2)))
        assert rms_out > 0.5 * rms_in


# ─────────────────────────────────────────────────────────────────────────────
# DETECTORS — compute_frame_rms
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeFrameRms:
    def test_silent_audio_near_zero(self):
        audio = np.zeros(SR, dtype=np.float32)
        rms = compute_frame_rms(audio, cfg.FRAME_SIZE, cfg.HOP_LENGTH)
        assert rms.max() < 1e-7

    def test_dc_signal_nonzero_rms(self):
        audio = np.ones(SR, dtype=np.float32) * 0.5
        rms = compute_frame_rms(audio, cfg.FRAME_SIZE, cfg.HOP_LENGTH)
        assert rms.mean() > 0.4

    def test_output_is_1d_float32(self):
        audio = _quiet_noise(1.0)
        rms = compute_frame_rms(audio, cfg.FRAME_SIZE, cfg.HOP_LENGTH)
        assert rms.ndim == 1
        assert rms.dtype == np.float32

    def test_output_length(self):
        audio = _quiet_noise(1.0)
        rms = compute_frame_rms(audio, cfg.FRAME_SIZE, cfg.HOP_LENGTH)
        expected = 1 + len(audio) // cfg.HOP_LENGTH
        assert rms.shape[0] == expected


# ─────────────────────────────────────────────────────────────────────────────
# DETECTORS — rolling_median
# ─────────────────────────────────────────────────────────────────────────────


class TestRollingMedian:
    def test_constant_unchanged(self):
        arr = np.full(100, 0.5, dtype=np.float32)
        out = rolling_median(arr, 11)
        np.testing.assert_allclose(out, arr, atol=1e-6)

    def test_output_dtype_float32(self):
        arr = np.arange(50, dtype=np.float64)
        out = rolling_median(arr, 5)
        assert out.dtype == np.float32

    def test_shape_preserved(self):
        arr = RNG.standard_normal(200).astype(np.float32)
        out = rolling_median(arr, 15)
        assert out.shape == arr.shape

    def test_spike_smoothed(self):
        arr = np.ones(101, dtype=np.float32) * 0.1
        arr[50] = 1000.0
        out = rolling_median(arr, 11)
        assert out[50] < 10.0

    def test_window_clamped_minimum_three(self):
        arr = np.ones(10, dtype=np.float32)
        out = rolling_median(arr, 1)
        assert out.shape == arr.shape


# ─────────────────────────────────────────────────────────────────────────────
# DETECTORS — find_runs
# ─────────────────────────────────────────────────────────────────────────────


class TestFindRuns:
    def test_empty_array(self):
        starts, ends = find_runs(np.array([], dtype=bool))
        assert len(starts) == 0
        assert len(ends) == 0

    def test_all_false(self):
        starts, ends = find_runs(np.zeros(10, dtype=bool))
        assert len(starts) == 0

    def test_all_true(self):
        starts, ends = find_runs(np.ones(5, dtype=bool))
        assert list(starts) == [0]
        assert list(ends) == [4]

    def test_single_true_at_start(self):
        starts, ends = find_runs(np.array([True, False, False, False]))
        assert list(starts) == [0]
        assert list(ends) == [0]

    def test_single_true_at_end(self):
        starts, ends = find_runs(np.array([False, False, True]))
        assert list(starts) == [2]
        assert list(ends) == [2]

    def test_multiple_runs(self):
        a = np.array([False, True, True, False, True, False, True, True, True])
        starts, ends = find_runs(a)
        assert list(starts) == [1, 4, 6]
        assert list(ends) == [2, 4, 8]

    def test_single_element_true(self):
        starts, ends = find_runs(np.array([True]))
        assert list(starts) == [0]
        assert list(ends) == [0]


# ─────────────────────────────────────────────────────────────────────────────
# DETECTORS — merge_events_by_gap
# ─────────────────────────────────────────────────────────────────────────────


class TestMergeEventsByGap:
    def test_empty_returns_empty(self):
        assert merge_events_by_gap([], 0.15) == []

    def test_single_event_passes_through(self):
        ev = _make_work_event(1.0, 1.01)
        merged = merge_events_by_gap([ev], 0.15)
        assert len(merged) == 1
        assert merged[0]["timestamp"] == pytest.approx(1.0)

    def test_within_gap_merged(self):
        e1 = _make_work_event(1.0, 1.01, severity=3.0)
        e2 = _make_work_event(1.06, 1.07, severity=5.0)
        merged = merge_events_by_gap([e1, e2], 0.15)
        assert len(merged) == 1
        assert merged[0]["severity"] == pytest.approx(5.0)
        assert merged[0]["timestamp"] == pytest.approx(1.0)

    def test_outside_gap_not_merged(self):
        e1 = _make_work_event(1.0, 1.01)
        e2 = _make_work_event(2.0, 2.01)
        merged = merge_events_by_gap([e1, e2], 0.15)
        assert len(merged) == 2

    def test_max_severity_kept(self):
        e1 = _make_work_event(1.0, 1.01, severity=2.0)
        e2 = _make_work_event(1.05, 1.06, severity=7.0)
        e3 = _make_work_event(1.10, 1.11, severity=4.0)
        merged = merge_events_by_gap([e1, e2, e3], 0.15)
        assert merged[0]["severity"] == pytest.approx(7.0)

    def test_unsorted_input_sorted_first(self):
        e1 = _make_work_event(2.0, 2.01)
        e2 = _make_work_event(1.0, 1.01)
        merged = merge_events_by_gap([e1, e2], 0.15)
        assert len(merged) == 2
        assert merged[0]["timestamp"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# DETECTORS — detect_clipping
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectClipping:
    def test_no_clipping(self):
        audio = np.full(SR, 0.5, dtype=np.float32)
        events = detect_clipping(audio, SR, "t.wav")
        assert events == []

    def test_single_clipped_region(self):
        audio = np.zeros(SR, dtype=np.float32)
        audio[100:110] = 1.0
        events = detect_clipping(audio, SR, "t.wav")
        assert len(events) == 1
        assert events[0]["issue_type"] == "clipping"
        assert events[0]["severity"] == pytest.approx(10.0)
        assert events[0]["timestamp"] == pytest.approx(100 / SR, abs=1e-6)

    def test_two_separate_regions(self):
        audio = np.zeros(SR, dtype=np.float32)
        audio[100:105] = 1.0
        audio[500:510] = 1.0
        events = detect_clipping(audio, SR, "t.wav")
        assert len(events) == 2

    def test_negative_clipping(self):
        audio = np.zeros(SR, dtype=np.float32)
        audio[200:210] = -1.0
        events = detect_clipping(audio, SR, "t.wav")
        assert len(events) == 1

    def test_severity_equals_sample_count(self):
        audio = np.zeros(SR, dtype=np.float32)
        audio[1000:1025] = 0.999
        events = detect_clipping(audio, SR, "t.wav")
        assert len(events) == 1
        assert events[0]["severity"] == pytest.approx(25.0)

    def test_just_below_threshold_not_detected(self):
        audio = np.full(SR, 0.997, dtype=np.float32)
        events = detect_clipping(audio, SR, "t.wav")
        assert events == []

    def test_silent_no_clipping(self):
        audio = np.zeros(SR, dtype=np.float32)
        events = detect_clipping(audio, SR, "t.wav")
        assert events == []

    def test_filename_in_event(self):
        audio = np.zeros(SR, dtype=np.float32)
        audio[50:60] = 1.0
        events = detect_clipping(audio, SR, "chapter01.wav")
        assert events[0]["filename"] == "chapter01.wav"


# ─────────────────────────────────────────────────────────────────────────────
# DETECTORS — detect_transients
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectTransients:
    def test_too_short_returns_empty(self):
        audio = np.zeros(10, dtype=np.float32)
        events = detect_transients(
            audio, SR, "t.wav",
            band=cfg.CLICK_BAND,
            threshold_mult=cfg.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=cfg.CLICK_MAX_DURATION_MS,
            merge_ms=cfg.CLICK_MERGE_MS,
            issue_type="click",
        )
        assert events == []

    def test_silent_returns_empty(self):
        audio = np.zeros(SR * 3, dtype=np.float32)
        events = detect_transients(
            audio, SR, "t.wav",
            band=cfg.CLICK_BAND,
            threshold_mult=cfg.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=cfg.CLICK_MAX_DURATION_MS,
            merge_ms=cfg.CLICK_MERGE_MS,
            issue_type="click",
        )
        assert events == []

    def test_constant_tone_below_threshold(self):
        audio = _sine_burst(5000.0, 3.0, amp=0.1)
        events = detect_transients(
            audio, SR, "t.wav",
            band=cfg.CLICK_BAND,
            threshold_mult=cfg.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=cfg.CLICK_MAX_DURATION_MS,
            merge_ms=cfg.CLICK_MERGE_MS,
            issue_type="click",
        )
        assert events == []

    def test_long_burst_filtered_by_duration(self):
        audio = _quiet_noise(3.0)
        burst = _sine_burst(5000.0, 0.05, amp=0.5)
        audio = _inject(audio, burst, 1.5)
        events = detect_transients(
            audio, SR, "t.wav",
            band=cfg.CLICK_BAND,
            threshold_mult=cfg.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=1.0,
            merge_ms=cfg.CLICK_MERGE_MS,
            issue_type="click",
        )
        assert events == []

    def test_click_burst_detected_permissive(self):
        audio = _quiet_noise(4.0)
        burst = _sine_burst(5000.0, 0.03, amp=0.5)
        audio = _inject(audio, burst, 2.0)
        events = detect_transients(
            audio, SR, "t.wav",
            band=cfg.CLICK_BAND,
            threshold_mult=cfg.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=500.0,
            merge_ms=cfg.CLICK_MERGE_MS,
            issue_type="click",
        )
        assert len(events) >= 1
        assert all(e["issue_type"] == "click" for e in events)
        assert all(e["severity"] > 1.0 for e in events)
        assert any(1.5 <= e["timestamp"] <= 2.5 for e in events)

    def test_plosive_burst_detected(self):
        audio = _quiet_noise(4.0)
        burst = _sine_burst(80.0, 0.05, amp=0.5)
        audio = _inject(audio, burst, 2.0)
        events = detect_transients(
            audio, SR, "t.wav",
            band=cfg.PLOSIVE_BAND,
            threshold_mult=cfg.PLOSIVE_THRESHOLD_MULTIPLIER,
            max_duration_ms=500.0,
            merge_ms=cfg.PLOSIVE_MERGE_MS,
            issue_type="plosive",
        )
        assert len(events) >= 1
        assert all(e["issue_type"] == "plosive" for e in events)

    def test_two_close_bursts_merged(self):
        audio = _quiet_noise(6.0)
        burst = _sine_burst(5000.0, 0.02, amp=0.5)
        audio = _inject(audio, burst, 2.0)
        audio = _inject(audio, burst, 2.05)
        events = detect_transients(
            audio, SR, "t.wav",
            band=cfg.CLICK_BAND,
            threshold_mult=cfg.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=500.0,
            merge_ms=200.0,
            issue_type="click",
        )
        assert len(events) == 1

    def test_two_far_bursts_not_merged(self):
        audio = _quiet_noise(8.0)
        burst = _sine_burst(5000.0, 0.03, amp=0.5)
        audio = _inject(audio, burst, 1.0)
        audio = _inject(audio, burst, 6.0)
        events = detect_transients(
            audio, SR, "t.wav",
            band=cfg.CLICK_BAND,
            threshold_mult=cfg.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=500.0,
            merge_ms=cfg.CLICK_MERGE_MS,
            issue_type="click",
        )
        assert len(events) == 2

    def test_no_end_time_in_returned_events(self):
        audio = _quiet_noise(4.0)
        burst = _sine_burst(5000.0, 0.03, amp=0.5)
        audio = _inject(audio, burst, 2.0)
        events = detect_transients(
            audio, SR, "t.wav",
            band=cfg.CLICK_BAND,
            threshold_mult=cfg.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=500.0,
            merge_ms=cfg.CLICK_MERGE_MS,
            issue_type="click",
        )
        for ev in events:
            assert "end_time" not in ev

    def test_empty_rms_early_exit(self, monkeypatch):
        import audiobook_qc.detectors as det
        monkeypatch.setattr(
            det, "compute_frame_rms", lambda *a, **kw: np.array([], dtype=np.float32)
        )
        audio = _quiet_noise(3.0)
        events = detect_transients(
            audio, SR, "t.wav",
            band=cfg.CLICK_BAND,
            threshold_mult=cfg.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=cfg.CLICK_MAX_DURATION_MS,
            merge_ms=cfg.CLICK_MERGE_MS,
            issue_type="click",
        )
        assert events == []


# ─────────────────────────────────────────────────────────────────────────────
# DETECTORS — detect_noise_floor_jumps
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectNoiseFloorJumps:
    def _audio_with_jump(self, low_amp: float = 0.001, high_amp: float = 0.1) -> np.ndarray:
        quiet = (RNG.standard_normal(SR * 2) * low_amp).astype(np.float32)
        loud = (RNG.standard_normal(SR * 2) * high_amp).astype(np.float32)
        return np.concatenate([quiet, loud])

    def test_short_audio_returns_empty(self):
        audio = _quiet_noise(0.3)
        events = detect_noise_floor_jumps(audio, SR, "t.wav")
        assert events == []

    def test_constant_amplitude_returns_empty(self):
        audio = np.full(SR * 5, 0.05, dtype=np.float32)
        events = detect_noise_floor_jumps(audio, SR, "t.wav")
        assert events == []

    def test_silent_returns_empty(self):
        audio = np.zeros(SR * 3, dtype=np.float32)
        events = detect_noise_floor_jumps(audio, SR, "t.wav")
        assert events == []

    def test_jump_detected(self):
        audio = self._audio_with_jump()
        events = detect_noise_floor_jumps(audio, SR, "t.wav")
        assert len(events) >= 1
        assert all(e["issue_type"] == "noise_floor_jump" for e in events)
        assert all(e["severity"] >= cfg.NOISE_DB_DELTA_THRESHOLD for e in events)

    def test_timestamp_near_jump(self):
        audio = self._audio_with_jump()
        events = detect_noise_floor_jumps(audio, SR, "t.wav")
        assert any(1.0 <= e["timestamp"] <= 2.5 for e in events)

    def test_min_separation_enforced(self):
        seg = int(0.5 * SR)
        pieces = []
        for i in range(20):
            amp = 0.1 if i % 2 == 0 else 0.001
            pieces.append((RNG.standard_normal(seg) * amp).astype(np.float32))
        audio = np.concatenate(pieces)
        events = detect_noise_floor_jumps(audio, SR, "t.wav")
        if len(events) > 1:
            timestamps = sorted(e["timestamp"] for e in events)
            for a, b in zip(timestamps, timestamps[1:]):
                assert b - a >= cfg.NOISE_MIN_SEPARATION_SECONDS - 1e-9

    def test_severity_is_delta_db(self):
        audio = self._audio_with_jump(low_amp=0.001, high_amp=0.1)
        events = detect_noise_floor_jumps(audio, SR, "t.wav")
        assert max(e["severity"] for e in events) > 20.0

    def test_event_fields_present(self):
        audio = self._audio_with_jump()
        events = detect_noise_floor_jumps(audio, SR, "t.wav")
        for ev in events:
            for key in ("filename", "timestamp", "issue_type", "severity"):
                assert key in ev
            assert ev["filename"] == "t.wav"


# ─────────────────────────────────────────────────────────────────────────────
# DETECTORS — run_all_detectors
# ─────────────────────────────────────────────────────────────────────────────


class TestRunAllDetectors:
    def test_returns_list(self):
        audio = _quiet_noise(1.0)
        events = run_all_detectors(audio, SR, "t.wav")
        assert isinstance(events, list)

    def test_clipping_included(self):
        audio = np.zeros(SR * 2, dtype=np.float32)
        audio[500:510] = 1.0
        events = run_all_detectors(audio, SR, "t.wav")
        types = [e["issue_type"] for e in events]
        assert "clipping" in types

    def test_all_event_fields_present(self):
        audio = _quiet_noise(1.0)
        events = run_all_detectors(audio, SR, "t.wav")
        for ev in events:
            for key in ("filename", "timestamp", "issue_type", "severity"):
                assert key in ev


# ─────────────────────────────────────────────────────────────────────────────
# SCANNER — scan_file
# ─────────────────────────────────────────────────────────────────────────────


class TestScanFile:
    def test_unreadable_returns_empty(self, tmp_path):
        result = scan_file(str(tmp_path / "nonexistent.wav"))
        assert result == []

    def test_corrupt_returns_empty(self, tmp_path):
        p = tmp_path / "bad.wav"
        p.write_bytes(b"NOTAWAV")
        assert scan_file(str(p)) == []

    def test_returns_list(self, tmp_path):
        p = tmp_path / "ok.wav"
        _write_wav(p, _quiet_noise(1.0))
        assert isinstance(scan_file(str(p)), list)

    def test_stereo_no_crash(self, tmp_path):
        stereo = RNG.standard_normal((SR, 2)).astype(np.float32) * 0.01
        p = tmp_path / "stereo.wav"
        sf.write(str(p), stereo, SR, subtype="PCM_16")
        assert isinstance(scan_file(str(p)), list)

    def test_clipped_file_returns_clipping_event(self, tmp_path):
        audio = np.zeros(SR * 2, dtype=np.float32)
        audio[1000:1020] = 1.0
        p = tmp_path / "clipped.wav"
        sf.write(str(p), audio, SR, subtype="FLOAT")
        result = scan_file(str(p))
        assert any(e["issue_type"] == "clipping" for e in result)

    def test_peak_above_unity_normalised_no_crash(self, tmp_path):
        audio = np.full(SR, 2.0, dtype=np.float32)
        p = tmp_path / "hot.wav"
        sf.write(str(p), audio, SR, subtype="FLOAT")
        assert isinstance(scan_file(str(p)), list)

    def test_empty_audio_returns_empty(self, tmp_path):
        p = tmp_path / "empty.wav"
        sf.write(str(p), np.zeros((0,), dtype=np.float32), SR, subtype="FLOAT")
        assert scan_file(str(p)) == []

    def test_event_fields_present(self, tmp_path):
        audio = np.zeros(SR * 2, dtype=np.float32)
        audio[500:510] = 1.0
        p = tmp_path / "events.wav"
        sf.write(str(p), audio, SR, subtype="FLOAT")
        events = scan_file(str(p))
        for ev in events:
            for key in ("filename", "timestamp", "issue_type", "severity"):
                assert key in ev

    def test_filename_is_basename_only(self, tmp_path):
        audio = np.zeros(SR * 2, dtype=np.float32)
        audio[500:510] = 1.0
        p = tmp_path / "chapter01.wav"
        sf.write(str(p), audio, SR, subtype="FLOAT")
        events = scan_file(str(p))
        for ev in events:
            assert "/" not in ev["filename"]
            assert ev["filename"] == "chapter01.wav"


# ─────────────────────────────────────────────────────────────────────────────
# SCANNER — scan_directory
# ─────────────────────────────────────────────────────────────────────────────


class TestScanDirectory:
    def test_empty_folder_returns_empty_lists(self, tmp_path):
        wavs, events = scan_directory(str(tmp_path))
        assert wavs == []
        assert events == []

    def test_finds_wav_files(self, tmp_path):
        _write_wav(tmp_path / "a.wav", _quiet_noise(0.5))
        _write_wav(tmp_path / "b.wav", _quiet_noise(0.5))
        wavs, _ = scan_directory(str(tmp_path))
        assert len(wavs) == 2

    def test_recursive_discovery(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_wav(sub / "deep.wav", _quiet_noise(0.5))
        wavs, _ = scan_directory(str(tmp_path))
        assert len(wavs) == 1

    def test_non_wav_ignored(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello")
        wavs, _ = scan_directory(str(tmp_path))
        assert wavs == []

    def test_returns_sorted_paths(self, tmp_path):
        for name in ("c.wav", "a.wav", "b.wav"):
            _write_wav(tmp_path / name, _quiet_noise(0.2))
        wavs, _ = scan_directory(str(tmp_path))
        assert wavs == sorted(wavs)

    def test_events_list_type(self, tmp_path):
        _write_wav(tmp_path / "x.wav", _quiet_noise(0.5))
        _, events = scan_directory(str(tmp_path))
        assert isinstance(events, list)


# ─────────────────────────────────────────────────────────────────────────────
# RANKING — rank_events
# ─────────────────────────────────────────────────────────────────────────────


class TestRankEvents:
    def test_empty_input_returns_empty(self):
        assert rank_events([]) == []

    def test_rank_global_assigned_from_one(self):
        events = [
            _make_event(severity=5.0),
            _make_event(severity=10.0),
            _make_event(severity=3.0),
        ]
        ranked = rank_events(events)
        assert ranked[0]["rank_global"] == 1
        assert ranked[1]["rank_global"] == 2
        assert ranked[2]["rank_global"] == 3

    def test_sorted_by_severity_descending(self):
        events = [
            _make_event(severity=1.0),
            _make_event(severity=100.0),
            _make_event(severity=50.0),
        ]
        ranked = rank_events(events)
        scores = [e["severity"] for e in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_tier_primary_up_to_max_primary(self):
        events = [_make_event(severity=float(i)) for i in range(cfg.MAX_PRIMARY + 5)]
        ranked = rank_events(events)
        primaries = [e for e in ranked if e["tier"] == "primary"]
        assert len(primaries) == cfg.MAX_PRIMARY

    def test_tier_secondary_beyond_max_primary(self):
        n = cfg.MAX_PRIMARY + 10
        events = [_make_event(severity=float(i)) for i in range(n)]
        ranked = rank_events(events)
        for ev in ranked:
            rank = ev["rank_global"]
            if rank <= cfg.MAX_PRIMARY:
                assert ev["tier"] == "primary"
            else:
                assert ev["tier"] == "secondary"

    def test_capped_at_max_primary_plus_secondary(self):
        n = cfg.MAX_PRIMARY + cfg.MAX_SECONDARY + 50
        events = [_make_event(severity=float(i)) for i in range(n)]
        ranked = rank_events(events)
        assert len(ranked) == cfg.MAX_PRIMARY + cfg.MAX_SECONDARY

    def test_deterministic_sort_on_ties(self):
        """Tie-breaking uses filename, timestamp, issue_type — result is stable."""
        events = [
            _make_event(filename="b.wav", timestamp=1.0, severity=5.0),
            _make_event(filename="a.wav", timestamp=2.0, severity=5.0),
            _make_event(filename="a.wav", timestamp=1.0, severity=5.0),
        ]
        ranked = rank_events(events)
        assert ranked[0]["filename"] == "a.wav"
        assert ranked[0]["timestamp"] == pytest.approx(1.0)
        assert ranked[1]["filename"] == "a.wav"
        assert ranked[1]["timestamp"] == pytest.approx(2.0)
        assert ranked[2]["filename"] == "b.wav"

    def test_severity_cast_to_float(self):
        events = [_make_event(severity=7)]  # int input
        ranked = rank_events(events)
        assert isinstance(ranked[0]["severity"], float)


# ─────────────────────────────────────────────────────────────────────────────
# REPORT_CSV — write_csv
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteCsv:
    def test_empty_events_writes_header_only(self, tmp_path):
        path = write_csv([], str(tmp_path))
        df = pd.read_csv(path)
        assert list(df.columns) == HEADER_COLS
        assert len(df) == 0

    def test_output_in_folder(self, tmp_path):
        path = write_csv([], str(tmp_path))
        assert Path(path).parent == tmp_path

    def test_columns_match_spec(self, tmp_path):
        ev = _make_event()
        ev["rank_global"] = 1  # type: ignore[typeddict-unknown-key]
        ev["tier"] = "primary"  # type: ignore[typeddict-unknown-key]
        path = write_csv([ev], str(tmp_path))
        df = pd.read_csv(path)
        assert list(df.columns) == HEADER_COLS

    def test_timestamp_rounded_to_3_decimals(self, tmp_path):
        ev = _make_event(timestamp=1.123456789)
        ev["rank_global"] = 1  # type: ignore[typeddict-unknown-key]
        ev["tier"] = "primary"  # type: ignore[typeddict-unknown-key]
        path = write_csv([ev], str(tmp_path))
        df = pd.read_csv(path)
        assert df["timestamp_seconds"].iloc[0] == pytest.approx(1.123, abs=1e-3)

    def test_severity_rounded_to_4_decimals(self, tmp_path):
        ev = _make_event(severity=3.123456789)
        ev["rank_global"] = 1  # type: ignore[typeddict-unknown-key]
        ev["tier"] = "primary"  # type: ignore[typeddict-unknown-key]
        path = write_csv([ev], str(tmp_path))
        df = pd.read_csv(path)
        assert df["severity_score"].iloc[0] == pytest.approx(3.1235, abs=1e-4)

    def test_minsec_format(self, tmp_path):
        ev = _make_event(timestamp=90.0)  # 1:30.000
        ev["rank_global"] = 1  # type: ignore[typeddict-unknown-key]
        ev["tier"] = "primary"  # type: ignore[typeddict-unknown-key]
        path = write_csv([ev], str(tmp_path))
        df = pd.read_csv(path)
        assert df["timestamp_minsec"].iloc[0] == "1:30.000"

    def test_multiple_rows_written(self, tmp_path):
        evs = []
        for i in range(5):
            ev = _make_event(timestamp=float(i), severity=float(10 - i))
            ev["rank_global"] = i + 1  # type: ignore[typeddict-unknown-key]
            ev["tier"] = "primary"  # type: ignore[typeddict-unknown-key]
            evs.append(ev)
        path = write_csv(evs, str(tmp_path))
        df = pd.read_csv(path)
        assert len(df) == 5


# ─────────────────────────────────────────────────────────────────────────────
# REPORT_HTML — write_html
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteHtml:
    def test_creates_file(self, tmp_path):
        path = write_html([], str(tmp_path))
        assert Path(path).exists()

    def test_filename_is_review_html(self, tmp_path):
        path = write_html([], str(tmp_path))
        assert Path(path).name == "review.html"

    def test_output_in_folder(self, tmp_path):
        path = write_html([], str(tmp_path))
        assert Path(path).parent == tmp_path

    def test_empty_events_still_valid_html(self, tmp_path):
        path = write_html([], str(tmp_path))
        content = Path(path).read_text()
        assert "<!DOCTYPE html>" in content
        assert "<table>" in content

    def test_row_present_for_each_event(self, tmp_path):
        events = []
        for i in range(3):
            ev = _make_event(filename=f"ch{i:02d}.wav", timestamp=float(i * 10))
            ev["rank_global"] = i + 1  # type: ignore[typeddict-unknown-key]
            ev["tier"] = "primary"  # type: ignore[typeddict-unknown-key]
            events.append(ev)
        path = write_html(events, str(tmp_path))
        content = Path(path).read_text()
        for i in range(3):
            assert f"ch{i:02d}.wav" in content

    def test_primary_row_class(self, tmp_path):
        ev = _make_event()
        ev["rank_global"] = 1  # type: ignore[typeddict-unknown-key]
        ev["tier"] = "primary"  # type: ignore[typeddict-unknown-key]
        path = write_html([ev], str(tmp_path))
        content = Path(path).read_text()
        assert "primary-row" in content

    def test_secondary_row_class(self, tmp_path):
        ev = _make_event()
        ev["rank_global"] = cfg.MAX_PRIMARY + 1  # type: ignore[typeddict-unknown-key]
        ev["tier"] = "secondary"  # type: ignore[typeddict-unknown-key]
        path = write_html([ev], str(tmp_path))
        content = Path(path).read_text()
        assert "secondary-row" in content

    def test_audio_player_source_uses_basename(self, tmp_path):
        ev = _make_event(filename="chapter01.wav")
        ev["rank_global"] = 1  # type: ignore[typeddict-unknown-key]
        ev["tier"] = "primary"  # type: ignore[typeddict-unknown-key]
        path = write_html([ev], str(tmp_path))
        content = Path(path).read_text()
        assert 'src="chapter01.wav"' in content

    def test_jump_button_has_timestamp(self, tmp_path):
        ev = _make_event(timestamp=42.5)
        ev["rank_global"] = 1  # type: ignore[typeddict-unknown-key]
        ev["tier"] = "primary"  # type: ignore[typeddict-unknown-key]
        path = write_html([ev], str(tmp_path))
        content = Path(path).read_text()
        assert "42.5" in content

    def test_checkbox_per_row(self, tmp_path):
        evs = []
        for i in range(3):
            ev = _make_event()
            ev["rank_global"] = i + 1  # type: ignore[typeddict-unknown-key]
            ev["tier"] = "primary"  # type: ignore[typeddict-unknown-key]
            evs.append(ev)
        path = write_html(evs, str(tmp_path))
        content = Path(path).read_text()
        assert content.count('type="checkbox"') == 3

    def test_export_selected_js_present(self, tmp_path):
        path = write_html([], str(tmp_path))
        content = Path(path).read_text()
        assert "exportSelected" in content
        assert "playIssue" in content
        assert "seekRel" in content
        assert "updateTime" in content

    def test_xss_filename_escaped(self, tmp_path):
        ev = _make_event(filename='<script>alert(1)</script>.wav')
        ev["rank_global"] = 1  # type: ignore[typeddict-unknown-key]
        ev["tier"] = "primary"  # type: ignore[typeddict-unknown-key]
        path = write_html([ev], str(tmp_path))
        content = Path(path).read_text()
        assert "<script>alert(1)</script>" not in content


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — integration
# ─────────────────────────────────────────────────────────────────────────────


class TestMain:
    """End-to-end tests for main.py CLI."""

    def _run_main(self, folder: str) -> None:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import importlib
        import main as main_mod
        importlib.reload(main_mod)
        with patch("sys.argv", ["main.py", folder]):
            main_mod.main()

    def test_no_args_exits_1(self):
        import main as main_mod
        with patch("sys.argv", ["main.py"]):
            with pytest.raises(SystemExit) as exc:
                main_mod.main()
        assert exc.value.code == 1

    def test_bad_path_exits_1(self, tmp_path):
        import main as main_mod
        bogus = str(tmp_path / "does_not_exist")
        with patch("sys.argv", ["main.py", bogus]):
            with pytest.raises(SystemExit) as exc:
                main_mod.main()
        assert exc.value.code == 1

    def test_empty_folder_exits_0_and_writes_output(self, tmp_path):
        import main as main_mod
        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()
        with patch("sys.argv", ["main.py", str(wav_dir)]):
            with pytest.raises(SystemExit) as exc:
                main_mod.main()
        assert exc.value.code == 0
        assert (wav_dir / "qc_report.csv").exists()
        assert (wav_dir / "review.html").exists()

    def test_empty_csv_has_correct_columns(self, tmp_path):
        import main as main_mod
        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()
        with patch("sys.argv", ["main.py", str(wav_dir)]):
            with pytest.raises(SystemExit):
                main_mod.main()
        df = pd.read_csv(str(wav_dir / "qc_report.csv"))
        assert list(df.columns) == HEADER_COLS
        assert len(df) == 0

    def test_wav_folder_produces_csv_and_html(self, tmp_path):
        import main as main_mod
        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()
        audio = np.zeros(SR * 3, dtype=np.float32)
        audio[1000:1020] = 1.0
        sf.write(str(wav_dir / "ch01.wav"), audio, SR, subtype="FLOAT")
        with patch("sys.argv", ["main.py", str(wav_dir)]):
            main_mod.main()
        assert (wav_dir / "qc_report.csv").exists()
        assert (wav_dir / "review.html").exists()

    def test_csv_has_clipping_event(self, tmp_path):
        import main as main_mod
        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()
        audio = np.zeros(SR * 3, dtype=np.float32)
        audio[1000:1020] = 1.0
        sf.write(str(wav_dir / "ch01.wav"), audio, SR, subtype="FLOAT")
        with patch("sys.argv", ["main.py", str(wav_dir)]):
            main_mod.main()
        df = pd.read_csv(str(wav_dir / "qc_report.csv"))
        assert any(df["issue_type"] == "clipping")

    def test_csv_severity_descending(self, tmp_path):
        import main as main_mod
        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()
        audio = np.zeros(SR * 5, dtype=np.float32)
        audio[100:110] = 1.0
        audio[5000:5030] = 1.0
        audio[10000:10015] = 1.0
        sf.write(str(wav_dir / "ch01.wav"), audio, SR, subtype="FLOAT")
        with patch("sys.argv", ["main.py", str(wav_dir)]):
            main_mod.main()
        df = pd.read_csv(str(wav_dir / "qc_report.csv"))
        scores = df["severity_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_rank_starts_at_one(self, tmp_path):
        import main as main_mod
        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()
        audio = np.zeros(SR * 2, dtype=np.float32)
        audio[500:510] = 1.0
        sf.write(str(wav_dir / "ch01.wav"), audio, SR, subtype="FLOAT")
        with patch("sys.argv", ["main.py", str(wav_dir)]):
            main_mod.main()
        df = pd.read_csv(str(wav_dir / "qc_report.csv"))
        assert df["rank_global"].iloc[0] == 1
        assert list(df["rank_global"]) == list(range(1, len(df) + 1))

    def test_tier_values_valid(self, tmp_path):
        import main as main_mod
        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()
        audio = np.zeros(SR * 2, dtype=np.float32)
        audio[500:510] = 1.0
        sf.write(str(wav_dir / "ch01.wav"), audio, SR, subtype="FLOAT")
        with patch("sys.argv", ["main.py", str(wav_dir)]):
            main_mod.main()
        df = pd.read_csv(str(wav_dir / "qc_report.csv"))
        assert set(df["tier"].unique()).issubset({"primary", "secondary"})

    def test_html_contains_filename(self, tmp_path):
        import main as main_mod
        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()
        audio = np.zeros(SR * 2, dtype=np.float32)
        audio[500:510] = 1.0
        sf.write(str(wav_dir / "myfile.wav"), audio, SR, subtype="FLOAT")
        with patch("sys.argv", ["main.py", str(wav_dir)]):
            main_mod.main()
        content = (wav_dir / "review.html").read_text()
        assert "myfile.wav" in content

    @pytest.mark.skipif(
        not TEST_FILES_DIR.exists() or not list(TEST_FILES_DIR.glob("*.wav")),
        reason="No real WAV files in test_files/",
    )
    def test_integration_real_files(self, tmp_path):
        import main as main_mod
        with patch("sys.argv", ["main.py", str(TEST_FILES_DIR)]):
            main_mod.main()
        df = pd.read_csv(str(TEST_FILES_DIR / "qc_report.csv"))
        assert list(df.columns) == HEADER_COLS
        if len(df) > 0:
            assert df["rank_global"].iloc[0] == 1
            scores = df["severity_score"].tolist()
            assert scores == sorted(scores, reverse=True)
