"""
audiobook_qc/report_csv.py

Builds the CSV quality-control report from ranked events.
"""

from __future__ import annotations

import os

import pandas as pd

from audiobook_qc.detectors import Event

HEADER_COLS = [
    "rank_global",
    "tier",
    "filename",
    "timestamp_seconds",
    "timestamp_minsec",
    "issue_type",
    "severity_score",
]


def _minsec(ts: float) -> str:
    mins, secs = divmod(ts, 60)
    return f"{int(mins)}:{secs:06.3f}"


def write_csv(ranked_events: list[Event], folder: str) -> str:
    """
    Build a DataFrame from *ranked_events* and write ``qc_report.csv``
    inside *folder*.

    Returns the absolute path of the written file.
    """
    output_path = os.path.join(folder, "qc_report.csv")

    if not ranked_events:
        pd.DataFrame(columns=HEADER_COLS).to_csv(output_path, index=False)
        return output_path

    rows = []
    for ev in ranked_events:
        ts = round(float(ev["timestamp"]), 3)
        rows.append(
            {
                "rank_global": ev["rank_global"],  # type: ignore[typeddict-item]
                "tier": ev["tier"],  # type: ignore[typeddict-item]
                "filename": ev["filename"],
                "timestamp_seconds": ts,
                "timestamp_minsec": _minsec(ts),
                "issue_type": ev["issue_type"],
                "severity_score": round(float(ev["severity"]), 4),
            }
        )

    df = pd.DataFrame(rows, columns=HEADER_COLS)
    df.to_csv(output_path, index=False)
    return output_path
