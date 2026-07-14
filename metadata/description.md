# audiobook-qc — description

A signal-processing quality-control scanner that finds audio defects in WAV audiobooks and ranks them by severity for human review.

Between recording an audiobook and assembling it lies the tedious question of which files have problems worth fixing. audiobook-qc answers it: it walks a folder of WAV files, applies four broadband detectors — clipping, clicks, plosives, and noise-floor jumps — scores each detected event by severity, pools every candidate across all files, and ranks them globally into a primary and a secondary tier. It writes two outputs to the scanned folder: a machine-readable qc_report.csv with per-event rank, tier, timestamp, issue type, and severity, and a self-contained review.html interface with per-file playback, sortable and filterable rows, per-event notes, and one-click export of selected events. It is a companion in the audiobook workflow, kindred to m4Bookmaker, that bridges capture and distribution by telling you exactly which files need remediation before final assembly.

Python 3.11+ with numpy, scipy, librosa, and soundfile for the DSP; 228 tests, flake8-clean; MIT-licensed. Feature-complete.
