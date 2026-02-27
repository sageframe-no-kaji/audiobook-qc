"""
audiobook_qc/config.py

All tunable constants for the audiobook QC scanner, plus the ``Config``
dataclass used to carry runtime overrides through the processing pipeline.

Edit the module-level constants to change the compiled-in defaults.
Pass a ``Config`` instance at runtime to override any value per-run.
"""

from __future__ import annotations

import dataclasses

# ─────────────────────────────────────────────────────────────────────────────
# FRAMING
# ─────────────────────────────────────────────────────────────────────────────

FRAME_SIZE = 2048
HOP_LENGTH = 512

# ─────────────────────────────────────────────────────────────────────────────
# REPORT CAPS
# ─────────────────────────────────────────────────────────────────────────────

MAX_PRIMARY = 75
MAX_SECONDARY = 200

# ─────────────────────────────────────────────────────────────────────────────
# TRANSIENT BANDS  (Hz)
# ─────────────────────────────────────────────────────────────────────────────

CLICK_BAND = (2500, 10000)  # Hz
PLOSIVE_BAND = (20, 180)  # Hz

# ─────────────────────────────────────────────────────────────────────────────
# TRANSIENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

CLICK_THRESHOLD_MULTIPLIER = 5.0
PLOSIVE_THRESHOLD_MULTIPLIER = 4.0

CLICK_MAX_DURATION_MS = 15
PLOSIVE_MAX_DURATION_MS = 80

CLICK_MERGE_MS = 150
PLOSIVE_MERGE_MS = 200

# ─────────────────────────────────────────────────────────────────────────────
# NOISE FLOOR
# ─────────────────────────────────────────────────────────────────────────────

NOISE_WINDOW_SECONDS = 1.0
NOISE_DB_DELTA_THRESHOLD = 14.0

# Minimum gap enforced between noise-floor-jump events
NOISE_MIN_SEPARATION_SECONDS = 10.0

# ─────────────────────────────────────────────────────────────────────────────
# CLIPPING
# ─────────────────────────────────────────────────────────────────────────────

CLIPPING_THRESHOLD = 0.998

# ─────────────────────────────────────────────────────────────────────────────
# ROLLING BASELINE
# ─────────────────────────────────────────────────────────────────────────────

# Width of the rolling baseline estimator — approximately 1 second of frames
ROLLING_MEDIAN_SECONDS = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME CONFIG — carries per-run overrides through the pipeline
# ─────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Config:
    """
    All tunable scanner parameters as a single portable object.

    Every field defaults to the corresponding module-level constant so that
    ``Config()`` is identical to a hard-coded defaults run.
    """

    # ── Framing ───────────────────────────────────────────────────────────────
    frame_size: int = FRAME_SIZE
    hop_length: int = HOP_LENGTH

    # ── Report caps ───────────────────────────────────────────────────────────
    max_primary: int = MAX_PRIMARY
    max_secondary: int = MAX_SECONDARY

    # ── Click detection ───────────────────────────────────────────────────────
    click_band_low: float = float(CLICK_BAND[0])
    click_band_high: float = float(CLICK_BAND[1])
    click_threshold: float = CLICK_THRESHOLD_MULTIPLIER
    click_max_duration_ms: float = float(CLICK_MAX_DURATION_MS)
    click_merge_ms: float = float(CLICK_MERGE_MS)

    # ── Plosive detection ─────────────────────────────────────────────────────
    plosive_band_low: float = float(PLOSIVE_BAND[0])
    plosive_band_high: float = float(PLOSIVE_BAND[1])
    plosive_threshold: float = PLOSIVE_THRESHOLD_MULTIPLIER
    plosive_max_duration_ms: float = float(PLOSIVE_MAX_DURATION_MS)
    plosive_merge_ms: float = float(PLOSIVE_MERGE_MS)

    # ── Noise-floor detection ─────────────────────────────────────────────────
    noise_window_seconds: float = NOISE_WINDOW_SECONDS
    noise_db_threshold: float = NOISE_DB_DELTA_THRESHOLD
    noise_min_separation: float = NOISE_MIN_SEPARATION_SECONDS

    # ── Clipping detection ────────────────────────────────────────────────────
    clipping_threshold: float = CLIPPING_THRESHOLD

    # ── Rolling baseline ──────────────────────────────────────────────────────
    rolling_median_seconds: float = ROLLING_MEDIAN_SECONDS
