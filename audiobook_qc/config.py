"""
audiobook_qc/config.py

All tunable constants for the audiobook QC scanner.
Edit values here to adjust detection sensitivity — do not change elsewhere.
"""

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
