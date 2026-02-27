# audiobook_qc

**Automated quality-control scanner for audiobook WAV files.**

audiobook_qc walks a folder of `.wav` files, runs four broadband signal-processing detectors, globally ranks every detected event by severity, and writes a machine-readable CSV plus a browser-based interactive review interface — all from a single command.

---

## Features

- **Four detectors** — clipping, clicks, plosives, noise-floor jumps
- **Global severity ranking** across all files in a session
- **Interactive HTML review interface** with inline audio playback, per-event notes, and one-click CSV export of selected events
- **Fully configurable** — every detector parameter is exposed as a CLI flag and as a field on the `Config` dataclass for programmatic use
- **Fast** — per-file workload is parallelised; a 10-hour audiobook project typically finishes in under two minutes
- **Zero server required** — the HTML report is self-contained and opens directly in any modern browser
- **228 tests**, lint-clean (flake8, max-line-length 100)

---

## Requirements

- Python 3.11 or later
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [librosa](https://librosa.org/)
- [soundfile](https://pysoundfile.readthedocs.io/)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/audio-sensor.git
cd audio-sensor

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install numpy scipy librosa soundfile

# 4. (Optional) install in editable mode so the package is importable from anywhere
pip install -e .
```

---

## Quick Start

```bash
# Scan an entire audiobook project folder
python main.py ./chapters

# Skip the noise-floor detector
python main.py ./chapters --skip noise_floor_jump

# Be more permissive — only flag very loud clicks and plosives
python main.py ./chapters --click-threshold 8.0 --plosive-threshold 7.0

# Raise the report caps so more events are exported
python main.py ./chapters --max-primary 150 --max-secondary 400
```

After the scan completes you will find two files inside `./chapters/`:

| File | Description |
|------|-------------|
| `qc_report.csv` | All ranked events in CSV format |
| `review.html` | Interactive browser review interface |

Open `review.html` in your browser, listen to each flagged timestamp, add notes, check the boxes you want to act on, and click **Export Selected** to download a filtered CSV.

---

## CLI Reference

```
python main.py FOLDER [options]
```

### Positional

| Argument | Description |
|----------|-------------|
| `FOLDER` | Path to the directory containing `.wav` files |

### Detector control

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--skip` | `clipping` `click` `plosive` `noise_floor_jump` | *(none)* | Disable one or more detectors (space-separated) |

### Report caps

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-primary` | int | `75` | Maximum number of primary-tier events exported |
| `--max-secondary` | int | `200` | Maximum number of secondary-tier events exported |

### Clipping detection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--clipping-threshold` | float | `0.998` | Sample-value ceiling (0–1) above which a sample is clipped |

### Click detection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--click-band-low` | Hz | `2500` | Lower bound of click bandpass filter |
| `--click-band-high` | Hz | `10000` | Upper bound of click bandpass filter |
| `--click-threshold` | float | `5.0` | RMS/baseline ratio required to flag a click frame |
| `--click-max-duration-ms` | ms | `15` | Runs longer than this are discarded (not clicks) |
| `--click-merge-ms` | ms | `150` | Clicks within this gap are merged into one event |

### Plosive detection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--plosive-band-low` | Hz | `20` | Lower bound of plosive bandpass filter |
| `--plosive-band-high` | Hz | `180` | Upper bound of plosive bandpass filter |
| `--plosive-threshold` | float | `4.0` | RMS/baseline ratio required to flag a plosive frame |
| `--plosive-max-duration-ms` | ms | `80` | Runs longer than this are discarded (not plosives) |
| `--plosive-merge-ms` | ms | `200` | Plosives within this gap are merged into one event |

### Noise-floor jump detection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--noise-window-seconds` | sec | `1.0` | Length of each RMS measurement window |
| `--noise-db-threshold` | dB | `14.0` | Minimum dB difference between windows to flag a jump |
| `--noise-min-separation` | sec | `10.0` | Minimum gap between consecutive noise-floor events |

### DSP / framing

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--frame-size` | samples | `2048` | FFT frame size (power of two recommended) |
| `--hop-length` | samples | `512` | Hop stride between frames |
| `--rolling-median-seconds` | sec | `1.0` | Width of the rolling-median baseline estimator |

---

## Detectors

### Clipping

Scans every sample against `--clipping-threshold`.  Consecutive clipped samples are grouped into runs; runs shorter than 3 samples are ignored as single-sample noise.  Severity score is the run length — longer saturated regions score higher.

### Click

1. Bandpass-filter to `[click-band-low, click-band-high]` Hz
2. Compute short-time RMS with frame size `frame-size` and hop `hop-length`
3. Compute a rolling-median baseline over `rolling-median-seconds`
4. Flag frames where `RMS > click-threshold × baseline`
5. Discard flagged runs longer than `click-max-duration-ms`
6. Merge events within `click-merge-ms` of each other

Severity score is the peak `frame_rms / baseline_rms` ratio within the event window.

### Plosive

Identical pipeline to the click detector, but operating in the low-frequency `[plosive-band-low, plosive-band-high]` Hz band with `plosive-threshold` and `plosive-max-duration-ms`.  Captures mouth pops, breath blasts, and microphone thumps.

### Noise-floor jump

Divides the file into `noise-window-seconds`-long windows, computes the RMS in dB for each, then flags pairs of adjacent windows whose dB levels differ by at least `noise-db-threshold`.  A minimum separation of `noise-min-separation` seconds is enforced between events so that a single environmental change only creates one flag.

Severity score is the absolute dB difference between the two adjacent windows.

---

## Ranking

After all files are processed, every event from every file is pooled and sorted by severity score (descending).  Events are then split into two tiers:

| Tier | Contents | Cap |
|------|----------|-----|
| **primary** | Highest-severity events across the project | `--max-primary` (75) |
| **secondary** | Next-highest events | `--max-secondary` (200) |

Only ranked events are written to the output files.  Events below both caps are silently discarded.

---

## Output Files

### `qc_report.csv`

| Column | Description |
|--------|-------------|
| `rank_global` | 1-based severity rank across all files |
| `tier` | `primary` or `secondary` |
| `filename` | Base filename of the source WAV |
| `timestamp_seconds` | Event position in fractional seconds |
| `timestamp_minsec` | Event position as `MM:SS` |
| `issue_type` | `clipping`, `click`, `plosive`, or `noise_floor_jump` |
| `severity_score` | Numeric severity (higher = worse) |

### `review.html`

A fully self-contained HTML page with:

- **Inline audio players** — seek each file to the flagged timestamp and play with a single click
- **Playback controls** — jump ±5s around the event
- **Notes column** — free-text textarea saved in browser session storage
- **Row checkboxes** — select individual events for export
- **Export Selected** — downloads a CSV of the checked rows with notes appended
- **Filter / sort** — click any column header to sort; a search box filters visible rows

---

## Python API

All functionality is importable as a library.  The `Config` dataclass is the central object that carries runtime overrides.

```python
from audiobook_qc import Config, scan_directory, rank_events, write_csv, write_html

# Build a custom config — only override what you need
cfg = Config(
    clipping_threshold=0.99,
    click_threshold=6.0,
    max_primary=100,
    max_secondary=300,
)

# Scan a folder
wav_files, events = scan_directory("./chapters", skip={"noise_floor_jump"}, cfg=cfg)

# Rank
ranked = rank_events(events, cfg=cfg)

# Write reports
write_csv(ranked, "./chapters")
write_html(ranked, "./chapters")
```

### `Config` fields

| Field | Type | Default |
|-------|------|---------|
| `frame_size` | `int` | `2048` |
| `hop_length` | `int` | `512` |
| `max_primary` | `int` | `75` |
| `max_secondary` | `int` | `200` |
| `click_band_low` | `float` | `2500.0` |
| `click_band_high` | `float` | `10000.0` |
| `click_threshold` | `float` | `5.0` |
| `click_max_duration_ms` | `float` | `15.0` |
| `click_merge_ms` | `float` | `150.0` |
| `plosive_band_low` | `float` | `20.0` |
| `plosive_band_high` | `float` | `180.0` |
| `plosive_threshold` | `float` | `4.0` |
| `plosive_max_duration_ms` | `float` | `80.0` |
| `plosive_merge_ms` | `float` | `200.0` |
| `noise_window_seconds` | `float` | `1.0` |
| `noise_db_threshold` | `float` | `14.0` |
| `noise_min_separation` | `float` | `10.0` |
| `clipping_threshold` | `float` | `0.998` |
| `rolling_median_seconds` | `float` | `1.0` |

---

## Project Structure

```
audio-sensor/
├── main.py                     # CLI entry point
├── audiobook_qc/
│   ├── __init__.py             # Public API re-exports
│   ├── config.py               # All constants + Config dataclass
│   ├── detectors.py            # Signal-processing algorithms
│   ├── scanner.py              # Directory walker and file loader
│   ├── ranking.py              # Global severity ranker
│   ├── report_csv.py           # CSV report writer
│   └── report_html.py          # Interactive HTML report writer
├── tests/
│   └── test_audiobook_qc.py    # 228 pytest tests
└── docs/
    ├── audiobook_qc.1          # Man page (groff/troff)
    └── issue_types.md          # Detailed issue-type reference
```

---

## Running Tests

```bash
# From the repository root with the virtual environment active
python -m pytest tests/ -v
```

228 tests cover the `Config` dataclass, all four detectors (default and custom params), ranking logic, scanner integration, CLI flag parsing, and the `_build_parser()` function.

---

## Linting

```bash
python -m flake8 audiobook_qc/ main.py tests/ --max-line-length=100
```

The codebase is lint-clean (zero warnings).

---

## Man Page

The man page is located at `docs/audiobook_qc.1`.  To view it:

```bash
man ./docs/audiobook_qc.1
```

Or render it to plain text:

```bash
nroff -man docs/audiobook_qc.1 | less
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Add or update tests in `tests/test_audiobook_qc.py` for any changed behaviour.
3. Ensure all tests pass (`pytest tests/`) and the code is lint-clean (`flake8`).
4. Open a pull request with a clear description of the change.

All four detectors have corresponding test classes; new detector parameters should be covered by both a default-value test and a custom-value test.

---

## License

MIT — see [LICENSE](LICENSE) for details.
