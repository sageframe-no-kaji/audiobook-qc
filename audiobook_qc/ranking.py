"""
audiobook_qc/ranking.py

Global severity ranking, tier assignment, and capping.
"""

from __future__ import annotations

from audiobook_qc.config import MAX_PRIMARY, MAX_SECONDARY
from audiobook_qc.detectors import Event


def rank_events(all_events: list[Event]) -> list[Event]:
    """
    Sort events by severity (descending), assign rank_global and tier,
    then return only the top MAX_PRIMARY + MAX_SECONDARY events.

    Sort key is deterministic: (-severity, filename, timestamp, issue_type).
    Each returned dict gains two extra keys: ``rank_global`` (int) and
    ``tier`` (``"primary"`` | ``"secondary"``).
    """
    for ev in all_events:
        ev["severity"] = float(ev["severity"])

    all_events.sort(
        key=lambda e: (
            -e["severity"],
            e["filename"],
            e["timestamp"],
            e["issue_type"],
        )
    )

    for idx, ev in enumerate(all_events):
        ev["rank_global"] = idx + 1  # type: ignore[typeddict-unknown-key]

    keep = MAX_PRIMARY + MAX_SECONDARY
    ranked = all_events[:keep]

    for ev in ranked:
        rank = ev["rank_global"]  # type: ignore[typeddict-item]
        if rank <= MAX_PRIMARY:
            ev["tier"] = "primary"  # type: ignore[typeddict-unknown-key]
        else:
            ev["tier"] = "secondary"  # type: ignore[typeddict-unknown-key]

    return ranked
