"""
tests/test_audiobook_qc_scan.py

Full test suite for audiobook_qc_scan.py.
Run with:
    .venv/bin/pytest tests/ -v --cov=audiobook_qc_scan --cov-report=term-missing
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

# Make sure the project root is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).parent.parent))

import audiobook_qc_scan as qc  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

SR = 44100  # sample rate used for all synthetic audio
RNG = np.random.default_rng(42)

TEST_FILES_DIR = Path(__file__).parent.parent / "test_files"


def _quiet_noise(duration_s: float, sr: int = SR) -> np.ndarray:
    """White noise at very low amplitude — provides a stable RMS baseline."""
    n = int(duration_s * sr)
    return (RNG.standard_normal(n) * 1e-4).astype(np.float32)


def _sine_burst(
    freq_hz: float, duration_s: float, amp: float, sr: int = SR
) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _inject(
    base: np.ndarray, burst: np.ndarray, t_seconds: float, sr: int = SR
) -> np.ndarray:
    audio = base.copy()
    start = int(t_seconds * sr)
    end = min(start + len(burst), len(audio))
    audio[start:end] += burst[: end - start]
    return audio


def _write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    sf.write(str(path), audio, sr, subtype="PCM_16")


# ─────────────────────────────────────────────────────────────────────────────
# to_mono
# ─────────────────────────────────────────────────────────────────────────────


class TestToMono:
    def test_already_mono_returns_float32(self):
        a = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        out = qc.to_mono(a)
        assert out.ndim == 1
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, a.astype(np.float32))

    def test_stereo_averages_channels(self):
        # shape (N, 2)
        left = np.array([0.4, 0.8], dtype=np.float32)
        right = np.array([0.0, 0.4], dtype=np.float32)
        stereo = np.stack([left, right], axis=1)
        out = qc.to_mono(stereo)
        expected = np.array([0.2, 0.6], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-7)

    def test_stereo_output_dtype_is_float32(self):
        stereo = RNG.standard_normal((100, 2)).astype(np.float64)
        out = qc.to_mono(stereo)
        assert out.dtype == np.float32

    def test_multi_channel(self):
        """Three channels → mean of all three."""
        arr = np.array([[0.3, 0.6, 0.9]], dtype=np.float32)
        out = qc.to_mono(arr)
        np.testing.assert_allclose(out[0], 0.6, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# bandpass_filter
# ─────────────────────────────────────────────────────────────────────────────


class TestBandpassFilter:
    def test_output_shape_matches_input(self):
        audio = _quiet_noise(1.0)
        out = qc.bandpass_filter(audio, 2500.0, 10000.0, SR)
        assert out.shape == audio.shape

    def test_output_dtype_is_float32(self):
        audio = _quiet_noise(0.5)
        out = qc.bandpass_filter(audio, 2500.0, 10000.0, SR)
        assert out.dtype == np.float32

    def test_degenerate_low_ge_high_returns_copy(self):
        """When low >= high after normalisation, fall back to identity copy."""
        audio = _quiet_noise(0.5)
        # Pass extreme values that force low >= high after nyquist normalisation
        out = qc.bandpass_filter(audio, 25000.0, 100.0, SR)
        np.testing.assert_array_equal(out, audio)

    def test_bandpass_attenuates_out_of_band(self):
        """Energy at DC (0 Hz) is attenuated after a 2500-10000 Hz bandpass."""
        # Pure DC offset should be nearly eliminated by the highpass portion
        dc = np.ones(SR, dtype=np.float32) * 0.5
        out = qc.bandpass_filter(dc, 2500.0, 10000.0, SR)
        assert np.abs(out).mean() < 0.01  # DC suppressed

    def test_passband_signal_preserved(self):
        """A pure 5 kHz sine should pass through the 2500-10000 Hz filter."""
        tone = _sine_burst(5000.0, 0.5, amp=1.0)
        out = qc.bandpass_filter(tone, 2500.0, 10000.0, SR)
        # After settling, most of the RMS should be preserved
        rms_in = float(np.sqrt(np.mean(tone[SR // 4 :] ** 2)))
        rms_out = float(np.sqrt(np.mean(out[SR // 4 :] ** 2)))
        assert rms_out > 0.5 * rms_in  # at least half energy preserved


# ─────────────────────────────────────────────────────────────────────────────
# compute_frame_rms
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeFrameRms:
    def test_silent_audio_returns_near_zero(self):
        audio = np.zeros(SR, dtype=np.float32)
        rms = qc.compute_frame_rms(audio, qc.FRAME_SIZE, qc.HOP_LENGTH)
        assert rms.max() < 1e-7

    def test_dc_signal_has_nonzero_rms(self):
        audio = np.ones(SR, dtype=np.float32) * 0.5
        rms = qc.compute_frame_rms(audio, qc.FRAME_SIZE, qc.HOP_LENGTH)
        assert rms.mean() > 0.4

    def test_output_is_1d_float32(self):
        audio = _quiet_noise(1.0)
        rms = qc.compute_frame_rms(audio, qc.FRAME_SIZE, qc.HOP_LENGTH)
        assert rms.ndim == 1
        assert rms.dtype == np.float32

    def test_output_length_matches_librosa_expectation(self):
        audio = _quiet_noise(1.0)
        rms = qc.compute_frame_rms(audio, qc.FRAME_SIZE, qc.HOP_LENGTH)
        # With center=True the number of frames = 1 + len(audio)//hop_length
        expected_frames = 1 + len(audio) // qc.HOP_LENGTH
        assert rms.shape[0] == expected_frames


# ─────────────────────────────────────────────────────────────────────────────
# rolling_median
# ─────────────────────────────────────────────────────────────────────────────


class TestRollingMedian:
    def test_constant_array_unchanged(self):
        arr = np.full(100, 0.5, dtype=np.float32)
        out = qc.rolling_median(arr, 11)
        np.testing.assert_allclose(out, arr, atol=1e-6)

    def test_output_dtype_is_float32(self):
        arr = np.arange(50, dtype=np.float64)
        out = qc.rolling_median(arr, 5)
        assert out.dtype == np.float32

    def test_output_shape_preserved(self):
        arr = RNG.standard_normal(200).astype(np.float32)
        out = qc.rolling_median(arr, 15)
        assert out.shape == arr.shape

    def test_spike_smoothed(self):
        """A single large spike should be smoothed away by the median."""
        arr = np.ones(101, dtype=np.float32) * 0.1
        arr[50] = 1000.0
        out = qc.rolling_median(arr, 11)
        # The spike value should be drastically reduced in the output
        assert out[50] < 10.0

    def test_window_minimum_three(self):
        """Window < 3 should be clamped to 3 without error."""
        arr = np.ones(10, dtype=np.float32)
        out = qc.rolling_median(arr, 1)
        assert out.shape == arr.shape


# ─────────────────────────────────────────────────────────────────────────────
# find_runs
# ─────────────────────────────────────────────────────────────────────────────


class TestFindRuns:
    def test_empty_array(self):
        starts, ends = qc.find_runs(np.array([], dtype=bool))
        assert len(starts) == 0
        assert len(ends) == 0

    def test_all_false_no_runs(self):
        a = np.zeros(10, dtype=bool)
        starts, ends = qc.find_runs(a)
        assert len(starts) == 0

    def test_all_true_single_run(self):
        a = np.ones(5, dtype=bool)
        starts, ends = qc.find_runs(a)
        assert list(starts) == [0]
        assert list(ends) == [4]

    def test_single_true_at_start(self):
        a = np.array([True, False, False, False])
        starts, ends = qc.find_runs(a)
        assert list(starts) == [0]
        assert list(ends) == [0]

    def test_single_true_at_end(self):
        a = np.array([False, False, True])
        starts, ends = qc.find_runs(a)
        assert list(starts) == [2]
        assert list(ends) == [2]

    def test_multiple_runs(self):
        a = np.array([False, True, True, False, True, False, True, True, True])
        starts, ends = qc.find_runs(a)
        assert list(starts) == [1, 4, 6]
        assert list(ends) == [2, 4, 8]

    def test_single_element_true(self):
        starts, ends = qc.find_runs(np.array([True]))
        assert list(starts) == [0]
        assert list(ends) == [0]


# ─────────────────────────────────────────────────────────────────────────────
# merge_events_by_gap
# ─────────────────────────────────────────────────────────────────────────────


def _work_event(ts: float, end: float, severity: float = 1.0) -> qc._WorkEvent:
    from typing import cast

    return cast(
        qc._WorkEvent,
        {
            "filename": "test.wav",
            "timestamp": ts,
            "end_time": end,
            "issue_type": "click",
            "severity": severity,
        },
    )


class TestMergeEventsByGap:
    def test_empty_input_returns_empty(self):
        assert qc.merge_events_by_gap([], 0.15) == []

    def test_single_event_passes_through(self):
        ev = _work_event(1.0, 1.01)
        merged = qc.merge_events_by_gap([ev], 0.15)
        assert len(merged) == 1
        assert merged[0]["timestamp"] == pytest.approx(1.0)

    def test_two_events_within_gap_merged(self):
        """Events 0.05 s apart with a 0.15 s merge window → one event."""
        e1 = _work_event(1.0, 1.01, severity=3.0)
        e2 = _work_event(1.06, 1.07, severity=5.0)
        merged = qc.merge_events_by_gap([e1, e2], 0.15)
        assert len(merged) == 1
        assert merged[0]["severity"] == pytest.approx(5.0)  # max severity kept
        assert merged[0]["timestamp"] == pytest.approx(1.0)  # earliest timestamp

    def test_two_events_outside_gap_not_merged(self):
        e1 = _work_event(1.0, 1.01)
        e2 = _work_event(2.0, 2.01)
        merged = qc.merge_events_by_gap([e1, e2], 0.15)
        assert len(merged) == 2

    def test_merge_uses_max_severity(self):
        e1 = _work_event(1.0, 1.01, severity=2.0)
        e2 = _work_event(1.05, 1.06, severity=7.0)
        e3 = _work_event(1.10, 1.11, severity=4.0)
        merged = qc.merge_events_by_gap([e1, e2, e3], 0.15)
        assert len(merged) == 1
        assert merged[0]["severity"] == pytest.approx(7.0)

    def test_events_sorted_before_merging(self):
        """Unsorted input should still produce correct result."""
        e1 = _work_event(2.0, 2.01)
        e2 = _work_event(1.0, 1.01)
        merged = qc.merge_events_by_gap([e1, e2], 0.15)
        # 2.0 - 1.01 = 0.99 > 0.15 → not merged
        assert len(merged) == 2
        assert merged[0]["timestamp"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# detect_clipping
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectClipping:
    def test_no_clipping_returns_empty(self):
        audio = np.zeros(SR, dtype=np.float32) + 0.5
        events = qc.detect_clipping(audio, SR, "test.wav")
        assert events == []

    def test_single_clipped_region(self):
        audio = np.zeros(SR, dtype=np.float32)
        audio[100:110] = 1.0  # 10 clipped samples
        events = qc.detect_clipping(audio, SR, "test.wav")
        assert len(events) == 1
        assert events[0]["issue_type"] == "clipping"
        assert events[0]["severity"] == pytest.approx(10.0)
        assert events[0]["timestamp"] == pytest.approx(100 / SR, abs=1e-6)
        assert events[0]["filename"] == "test.wav"

    def test_two_separate_clipped_regions(self):
        audio = np.zeros(SR, dtype=np.float32)
        audio[100:105] = 1.0
        audio[500:510] = 1.0
        events = qc.detect_clipping(audio, SR, "test.wav")
        assert len(events) == 2

    def test_negative_clipping_detected(self):
        audio = np.zeros(SR, dtype=np.float32)
        audio[200:210] = -1.0  # abs() >= threshold
        events = qc.detect_clipping(audio, SR, "test.wav")
        assert len(events) == 1

    def test_severity_equals_clipped_sample_count(self):
        audio = np.zeros(SR, dtype=np.float32)
        audio[1000:1025] = 0.999  # 25 samples at exactly CLIPPING_THRESHOLD
        events = qc.detect_clipping(audio, SR, "test.wav")
        assert len(events) == 1
        assert events[0]["severity"] == pytest.approx(25.0)

    def test_border_value_just_below_threshold_not_detected(self):
        audio = np.full(SR, 0.997, dtype=np.float32)  # below 0.998
        events = qc.detect_clipping(audio, SR, "test.wav")
        assert events == []

    def test_silent_audio_no_clipping(self):
        audio = np.zeros(SR, dtype=np.float32)
        events = qc.detect_clipping(audio, SR, "test.wav")
        assert events == []


# ─────────────────────────────────────────────────────────────────────────────
# detect_transients  (shared engine for click & plosive)
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectTransients:
    # -- boundary / early-exit paths --

    def test_audio_too_short_returns_empty(self):
        audio = np.zeros(10, dtype=np.float32)  # << FRAME_SIZE
        events = qc.detect_transients(
            audio,
            SR,
            "t.wav",
            band=qc.CLICK_BAND,
            threshold_mult=qc.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=qc.CLICK_MAX_DURATION_MS,
            merge_ms=qc.CLICK_MERGE_MS,
            issue_type="click",
        )
        assert events == []

    def test_silent_audio_returns_empty(self):
        audio = np.zeros(SR * 3, dtype=np.float32)
        events = qc.detect_transients(
            audio,
            SR,
            "t.wav",
            band=qc.CLICK_BAND,
            threshold_mult=qc.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=qc.CLICK_MAX_DURATION_MS,
            merge_ms=qc.CLICK_MERGE_MS,
            issue_type="click",
        )
        assert events == []

    def test_constant_tone_below_threshold_returns_empty(self):
        """Constant-amplitude signal → ratio ≈ 1.0 → no frames exceed threshold."""
        audio = _sine_burst(5000.0, 3.0, amp=0.1)
        events = qc.detect_transients(
            audio,
            SR,
            "t.wav",
            band=qc.CLICK_BAND,
            threshold_mult=qc.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=qc.CLICK_MAX_DURATION_MS,
            merge_ms=qc.CLICK_MERGE_MS,
            issue_type="click",
        )
        assert events == []

    # -- "too long → filtered → no candidates" path --

    def test_long_burst_filtered_out_by_duration(self):
        """A loud burst whose flagged run exceeds max_duration_ms is discarded."""
        audio = _quiet_noise(3.0)
        burst = _sine_burst(5000.0, 0.05, amp=0.5)  # 50 ms burst
        audio = _inject(audio, burst, 1.5)
        # max_duration_ms = 1ms → max_frames = 1/(512/SR*1000) << span → filtered
        events = qc.detect_transients(
            audio,
            SR,
            "t.wav",
            band=qc.CLICK_BAND,
            threshold_mult=qc.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=1.0,  # absurdly small — guaranteed to reject
            merge_ms=qc.CLICK_MERGE_MS,
            issue_type="click",
        )
        assert events == []

    # -- detection path --

    def test_click_burst_detected_with_permissive_duration(self):
        """A clear 5 kHz burst is detected when max_duration_ms is generous."""
        audio = _quiet_noise(4.0)
        burst = _sine_burst(5000.0, 0.03, amp=0.5)  # 30 ms, well within band
        audio = _inject(audio, burst, 2.0)
        events = qc.detect_transients(
            audio,
            SR,
            "t.wav",
            band=qc.CLICK_BAND,
            threshold_mult=qc.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=500.0,  # permissive
            merge_ms=qc.CLICK_MERGE_MS,
            issue_type="click",
        )
        assert len(events) >= 1
        assert all(e["issue_type"] == "click" for e in events)
        assert all(e["severity"] > 1.0 for e in events)
        # Timestamp should be roughly around 2 s
        timestamps = [e["timestamp"] for e in events]
        assert any(1.5 <= t <= 2.5 for t in timestamps)

    def test_plosive_burst_detected(self):
        """A clear low-frequency burst is detected in the plosive band."""
        audio = _quiet_noise(4.0)
        burst = _sine_burst(80.0, 0.05, amp=0.5)  # 80 Hz, 50 ms
        audio = _inject(audio, burst, 2.0)
        events = qc.detect_transients(
            audio,
            SR,
            "t.wav",
            band=qc.PLOSIVE_BAND,
            threshold_mult=qc.PLOSIVE_THRESHOLD_MULTIPLIER,
            max_duration_ms=500.0,
            merge_ms=qc.PLOSIVE_MERGE_MS,
            issue_type="plosive",
        )
        assert len(events) >= 1
        assert all(e["issue_type"] == "plosive" for e in events)

    def test_two_close_bursts_merged(self):
        """Two clicks within CLICK_MERGE_MS → merged into one event."""
        audio = _quiet_noise(6.0)
        burst = _sine_burst(5000.0, 0.02, amp=0.5)
        audio = _inject(audio, burst, 2.0)
        audio = _inject(audio, burst, 2.05)  # 50 ms apart → within 150 ms merge window
        events = qc.detect_transients(
            audio,
            SR,
            "t.wav",
            band=qc.CLICK_BAND,
            threshold_mult=qc.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=500.0,
            merge_ms=200.0,  # explicit 200 ms merge window
            issue_type="click",
        )
        assert len(events) == 1

    def test_two_far_bursts_not_merged(self):
        """Two clicks 3 s apart → two separate events."""
        audio = _quiet_noise(8.0)
        burst = _sine_burst(5000.0, 0.03, amp=0.5)
        audio = _inject(audio, burst, 1.0)
        audio = _inject(audio, burst, 6.0)
        events = qc.detect_transients(
            audio,
            SR,
            "t.wav",
            band=qc.CLICK_BAND,
            threshold_mult=qc.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=500.0,
            merge_ms=qc.CLICK_MERGE_MS,
            issue_type="click",
        )
        assert len(events) == 2

    def test_empty_rms_array_returns_empty(self, monkeypatch):
        """If compute_frame_rms somehow returns empty → early exit (line 237)."""
        monkeypatch.setattr(
            qc, "compute_frame_rms", lambda *a, **kw: np.array([], dtype=np.float32)
        )
        audio = _quiet_noise(3.0)
        events = qc.detect_transients(
            audio,
            SR,
            "t.wav",
            band=qc.CLICK_BAND,
            threshold_mult=qc.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=qc.CLICK_MAX_DURATION_MS,
            merge_ms=qc.CLICK_MERGE_MS,
            issue_type="click",
        )
        assert events == []

    def test_returned_events_have_no_end_time_key(self):
        """end_time is an internal key and must not appear in returned events."""
        audio = _quiet_noise(4.0)
        burst = _sine_burst(5000.0, 0.03, amp=0.5)
        audio = _inject(audio, burst, 2.0)
        events = qc.detect_transients(
            audio,
            SR,
            "t.wav",
            band=qc.CLICK_BAND,
            threshold_mult=qc.CLICK_THRESHOLD_MULTIPLIER,
            max_duration_ms=500.0,
            merge_ms=qc.CLICK_MERGE_MS,
            issue_type="click",
        )
        for ev in events:
            assert "end_time" not in ev


# ─────────────────────────────────────────────────────────────────────────────
# detect_noise_floor_jumps
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectNoiseFloorJumps:
    def _audio_with_jump(
        self, low_amp: float = 0.001, high_amp: float = 0.1
    ) -> np.ndarray:
        """4 seconds: 2 s quiet then 2 s loud."""
        quiet = (RNG.standard_normal(SR * 2) * low_amp).astype(np.float32)
        loud = (RNG.standard_normal(SR * 2) * high_amp).astype(np.float32)
        return np.concatenate([quiet, loud])

    def test_short_audio_returns_empty(self):
        """Audio shorter than 2 noise windows → no events."""
        audio = _quiet_noise(0.3)  # less than 1 second
        events = qc.detect_noise_floor_jumps(audio, SR, "t.wav")
        assert events == []

    def test_constant_amplitude_returns_empty(self):
        audio = np.full(SR * 5, 0.05, dtype=np.float32)
        events = qc.detect_noise_floor_jumps(audio, SR, "t.wav")
        assert events == []

    def test_silent_audio_returns_empty(self):
        audio = np.zeros(SR * 3, dtype=np.float32)
        events = qc.detect_noise_floor_jumps(audio, SR, "t.wav")
        assert events == []

    def test_amplitude_jump_detected(self):
        audio = self._audio_with_jump()
        events = qc.detect_noise_floor_jumps(audio, SR, "t.wav")
        assert len(events) >= 1
        assert all(e["issue_type"] == "noise_floor_jump" for e in events)
        assert all(e["severity"] >= qc.NOISE_DB_DELTA_THRESHOLD for e in events)

    def test_event_timestamp_near_jump_point(self):
        audio = self._audio_with_jump()
        events = qc.detect_noise_floor_jumps(audio, SR, "t.wav")
        assert len(events) >= 1
        # The jump happens at t=2s; timestamp should be in the 1.5–2.5 s range
        assert any(1.0 <= e["timestamp"] <= 2.5 for e in events)

    def test_minimum_separation_enforced(self):
        """
        Multiple jumps spaced 0.5 s apart — only events separated by ≥
        NOISE_MIN_SEPARATION_SECONDS=2.0 s should survive.
        """
        # alternating loud/quiet every 0.5 s for 10 s
        seg = int(0.5 * SR)
        pieces = []
        for i in range(20):
            amp = 0.1 if i % 2 == 0 else 0.001
            pieces.append((RNG.standard_normal(seg) * amp).astype(np.float32))
        audio = np.concatenate(pieces)

        events = qc.detect_noise_floor_jumps(audio, SR, "t.wav")
        if len(events) > 1:
            timestamps = sorted(e["timestamp"] for e in events)
            for a, b in zip(timestamps, timestamps[1:]):
                assert b - a >= qc.NOISE_MIN_SEPARATION_SECONDS - 1e-9

    def test_severity_is_delta_db(self):
        audio = self._audio_with_jump(low_amp=0.001, high_amp=0.1)
        events = qc.detect_noise_floor_jumps(audio, SR, "t.wav")
        assert len(events) >= 1
        # ΔdB = 20*log10(0.1/0.001) = 40 dB → severity should be close to that
        max_sev = max(e["severity"] for e in events)
        assert max_sev > 20.0

    def test_event_has_correct_fields(self):
        audio = self._audio_with_jump()
        events = qc.detect_noise_floor_jumps(audio, SR, "t.wav")
        assert len(events) >= 1
        for ev in events:
            assert "filename" in ev
            assert "timestamp" in ev
            assert "issue_type" in ev
            assert "severity" in ev
            assert ev["filename"] == "t.wav"


# ─────────────────────────────────────────────────────────────────────────────
# scan_file
# ─────────────────────────────────────────────────────────────────────────────


class TestScanFile:
    def test_unreadable_file_returns_empty_no_crash(self, tmp_path):
        fake = tmp_path / "nonexistent.wav"
        result = qc.scan_file(str(fake))
        assert result == []

    def test_corrupt_file_returns_empty_no_crash(self, tmp_path):
        p = tmp_path / "corrupt.wav"
        p.write_bytes(b"NOTAWAV!!!")
        result = qc.scan_file(str(p))
        assert result == []

    def test_returns_list(self, tmp_path):
        audio = _quiet_noise(1.0)
        p = tmp_path / "clean.wav"
        _write_wav(p, audio)
        result = qc.scan_file(str(p))
        assert isinstance(result, list)

    def test_stereo_file_processed_without_crash(self, tmp_path):
        stereo = RNG.standard_normal((SR, 2)).astype(np.float32) * 0.01
        p = tmp_path / "stereo.wav"
        sf.write(str(p), stereo, SR, subtype="PCM_16")
        result = qc.scan_file(str(p))
        assert isinstance(result, list)

    def test_clipped_file_returns_clipping_event(self, tmp_path):
        audio = np.zeros(SR * 2, dtype=np.float32)
        audio[1000:1020] = 1.0  # 20 clipped samples
        p = tmp_path / "clipped.wav"
        # Write as float — preserves exact sample values
        sf.write(str(p), audio, SR, subtype="FLOAT")
        result = qc.scan_file(str(p))
        clipping = [e for e in result if e["issue_type"] == "clipping"]
        assert len(clipping) >= 1

    def test_peak_above_unity_normalised(self, tmp_path):
        """Files with peak > 1.0 are normalised before analysis (no crash)."""
        audio = np.full(SR, 2.0, dtype=np.float32)  # all samples at 2.0
        p = tmp_path / "hot.wav"
        sf.write(str(p), audio, SR, subtype="FLOAT")
        result = qc.scan_file(str(p))
        assert isinstance(result, list)

    def test_empty_audio_array_returns_empty(self, tmp_path):
        """If sf.read returns a zero-length signal, scan_file returns [] (line 378)."""
        p = tmp_path / "empty_signal.wav"
        # Write a genuinely empty WAV (0 samples) — soundfile accepts this
        sf.write(str(p), np.zeros((0,), dtype=np.float32), SR, subtype="FLOAT")
        result = qc.scan_file(str(p))
        assert result == []

    def test_event_fields_present(self, tmp_path):
        audio = np.zeros(SR * 2, dtype=np.float32)
        audio[500:510] = 1.0
        p = tmp_path / "ev.wav"
        sf.write(str(p), audio, SR, subtype="FLOAT")
        result = qc.scan_file(str(p))
        for ev in result:
            assert "filename" in ev
            assert "timestamp" in ev
            assert "issue_type" in ev
            assert "severity" in ev

    @pytest.mark.skipif(
        not TEST_FILES_DIR.exists() or not list(TEST_FILES_DIR.glob("*.wav")),
        reason="No real WAV files in test_files/",
    )
    def test_real_wav_files_produce_events_list(self):
        """Integration smoke test: real audiobook WAVs produce a list."""
        for wav in sorted(TEST_FILES_DIR.glob("*.wav")):
            result = qc.scan_file(str(wav))
            assert isinstance(result, list), f"scan_file failed for {wav.name}"
            for ev in result:
                assert ev["issue_type"] in (
                    "clipping",
                    "click",
                    "plosive",
                    "noise_floor_jump",
                )
                assert ev["timestamp"] >= 0.0
                assert ev["severity"] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# main() — CLI behaviour
# ─────────────────────────────────────────────────────────────────────────────


class TestMain:
    def test_no_args_exits_with_error(self):
        with patch("sys.argv", ["audiobook_qc_scan.py"]):
            with pytest.raises(SystemExit) as exc_info:
                qc.main()
        assert exc_info.value.code == 1

    def test_bad_path_exits_with_error(self, tmp_path):
        bogus = str(tmp_path / "does_not_exist")
        with patch("sys.argv", ["audiobook_qc_scan.py", bogus]):
            with pytest.raises(SystemExit) as exc_info:
                qc.main()
        assert exc_info.value.code == 1

    def test_empty_folder_writes_empty_csv_and_exits_zero(self, tmp_path):
        empty_dir = tmp_path / "wavs"
        empty_dir.mkdir()
        out_csv = tmp_path / "qc_report.csv"

        with patch("sys.argv", ["audiobook_qc_scan.py", str(empty_dir)]):
            with patch("audiobook_qc_scan.output_path", str(out_csv), create=True):
                # main() writes to cwd "qc_report.csv"; redirect via chdir
                orig_dir = os.getcwd()
                os.chdir(tmp_path)
                try:
                    with pytest.raises(SystemExit) as exc_info:
                        qc.main()
                finally:
                    os.chdir(orig_dir)
        assert exc_info.value.code == 0
        assert out_csv.exists()
        import pandas as pd

        df = pd.read_csv(str(out_csv))
        assert list(df.columns) == [
            "rank_global",
            "tier",
            "filename",
            "timestamp_seconds",
            "issue_type",
            "severity_score",
        ]
        assert len(df) == 0

    def test_folder_with_wav_produces_csv(self, tmp_path):
        """A folder with one WAV should produce a qc_report.csv."""
        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()
        audio = np.zeros(SR * 3, dtype=np.float32)
        audio[1000:1010] = 1.0  # ensure at least one clipping event
        sf.write(str(wav_dir / "chapter01.wav"), audio, SR, subtype="FLOAT")

        out_csv = tmp_path / "qc_report.csv"
        orig_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch("sys.argv", ["audiobook_qc_scan.py", str(wav_dir)]):
                qc.main()
        finally:
            os.chdir(orig_dir)

        assert out_csv.exists()
        import pandas as pd

        df = pd.read_csv(str(out_csv))
        assert list(df.columns) == [
            "rank_global",
            "tier",
            "filename",
            "timestamp_seconds",
            "issue_type",
            "severity_score",
        ]
        # At minimum the clipping event should be there
        assert any(df["issue_type"] == "clipping")

    def test_ranking_and_tiers_correct(self, tmp_path):
        """rank_global starts at 1, severity descending, tiers assigned correctly."""
        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()
        audio = np.zeros(SR * 5, dtype=np.float32)
        # Inject multiple clipped regions with varying lengths
        audio[100:110] = 1.0  # 10 samples
        audio[5000:5030] = 1.0  # 30 samples (highest severity)
        audio[10000:10015] = 1.0  # 15 samples
        sf.write(str(wav_dir / "ch01.wav"), audio, SR, subtype="FLOAT")

        orig_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch("sys.argv", ["audiobook_qc_scan.py", str(wav_dir)]):
                qc.main()
        finally:
            os.chdir(orig_dir)

        import pandas as pd

        df = pd.read_csv(str(tmp_path / "qc_report.csv"))
        assert df["rank_global"].iloc[0] == 1
        assert list(df["rank_global"]) == list(range(1, len(df) + 1))
        # Severity must be non-increasing
        scores = df["severity_score"].tolist()
        assert scores == sorted(scores, reverse=True)
        # All tiers are valid
        assert set(df["tier"].unique()).issubset({"primary", "secondary"})

    def test_primary_secondary_cap(self, tmp_path):
        """First MAX_PRIMARY rows are primary; rest secondary."""
        import pandas as pd
        import audiobook_qc_scan as qc_mod

        wav_dir = tmp_path / "wavs"
        wav_dir.mkdir()

        # Create audio with many clipping events (spaced apart)
        audio = np.zeros(SR * 60, dtype=np.float32)
        for i in range(0, qc_mod.MAX_PRIMARY + 10):
            start = i * SR // 2  # one clip per 0.5 s
            audio[start : start + 5] = 1.0

        sf.write(str(wav_dir / "many.wav"), audio, SR, subtype="FLOAT")

        orig_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch("sys.argv", ["audiobook_qc_scan.py", str(wav_dir)]):
                qc_mod.main()
        finally:
            os.chdir(orig_dir)

        df = pd.read_csv(str(tmp_path / "qc_report.csv"))
        prim = df[df["tier"] == "primary"]
        sec = df[df["tier"] == "secondary"]
        assert len(prim) <= qc_mod.MAX_PRIMARY
        # All primary ranks come before secondary ranks
        if len(sec) > 0:
            assert prim["rank_global"].max() < sec["rank_global"].min()

    @pytest.mark.skipif(
        not TEST_FILES_DIR.exists() or not list(TEST_FILES_DIR.glob("*.wav")),
        reason="No real WAV files in test_files/",
    )
    def test_integration_real_files(self, tmp_path):
        """End-to-end: scan real audiobook chapters, CSV produced with valid schema."""
        import pandas as pd

        orig_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            with patch("sys.argv", ["audiobook_qc_scan.py", str(TEST_FILES_DIR)]):
                qc.main()
        finally:
            os.chdir(orig_dir)

        out_csv = tmp_path / "qc_report.csv"
        assert out_csv.exists()
        df = pd.read_csv(str(out_csv))

        required_cols = [
            "rank_global",
            "tier",
            "filename",
            "timestamp_seconds",
            "issue_type",
            "severity_score",
        ]
        assert list(df.columns) == required_cols

        if len(df) > 0:
            assert df["rank_global"].iloc[0] == 1
            scores = df["severity_score"].tolist()
            assert scores == sorted(scores, reverse=True)
            assert set(df["tier"].unique()).issubset({"primary", "secondary"})
            assert set(df["issue_type"].unique()).issubset(
                {"clipping", "click", "plosive", "noise_floor_jump"}
            )
            assert (df["timestamp_seconds"] >= 0).all()
            assert (df["severity_score"] >= 0).all()
