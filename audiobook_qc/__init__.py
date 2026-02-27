"""audiobook_qc — modular audiobook quality-control scanner."""

from audiobook_qc.config import (
    CLICK_BAND,
    CLICK_MAX_DURATION_MS,
    CLICK_MERGE_MS,
    CLICK_THRESHOLD_MULTIPLIER,
    CLIPPING_THRESHOLD,
    FRAME_SIZE,
    HOP_LENGTH,
    MAX_PRIMARY,
    MAX_SECONDARY,
    NOISE_DB_DELTA_THRESHOLD,
    NOISE_MIN_SEPARATION_SECONDS,
    NOISE_WINDOW_SECONDS,
    PLOSIVE_BAND,
    PLOSIVE_MAX_DURATION_MS,
    PLOSIVE_MERGE_MS,
    PLOSIVE_THRESHOLD_MULTIPLIER,
    ROLLING_MEDIAN_SECONDS,
)
from audiobook_qc.detectors import (
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
    to_mono,
)
from audiobook_qc.ranking import rank_events
from audiobook_qc.report_csv import write_csv
from audiobook_qc.report_html import write_html
from audiobook_qc.scanner import scan_directory, scan_file

__all__ = [
    # config
    "CLICK_BAND",
    "CLICK_MAX_DURATION_MS",
    "CLICK_MERGE_MS",
    "CLICK_THRESHOLD_MULTIPLIER",
    "CLIPPING_THRESHOLD",
    "FRAME_SIZE",
    "HOP_LENGTH",
    "MAX_PRIMARY",
    "MAX_SECONDARY",
    "NOISE_DB_DELTA_THRESHOLD",
    "NOISE_MIN_SEPARATION_SECONDS",
    "NOISE_WINDOW_SECONDS",
    "PLOSIVE_BAND",
    "PLOSIVE_MAX_DURATION_MS",
    "PLOSIVE_MERGE_MS",
    "PLOSIVE_THRESHOLD_MULTIPLIER",
    "ROLLING_MEDIAN_SECONDS",
    # detectors
    "Event",
    "_WorkEvent",
    "bandpass_filter",
    "compute_frame_rms",
    "detect_clipping",
    "detect_noise_floor_jumps",
    "detect_transients",
    "find_runs",
    "merge_events_by_gap",
    "rolling_median",
    "to_mono",
    # scanner
    "scan_directory",
    "scan_file",
    # ranking
    "rank_events",
    # reports
    "write_csv",
    "write_html",
]
