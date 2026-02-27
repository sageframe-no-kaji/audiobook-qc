"""
audiobook_qc/ranking.py

Global severity ranking, tier assignment, and capping.
"""

from __future__ import annotations

from audiobook_qc.config import Config
from audiobook_qc.detectors import Event


def rank_events(
    all_events: list[Event],
    cfg: Config | None = None,
) -> list[Event]:
    """
    Sort events by severity (descending), assign rank_global and tier,
    then return only the top MAX_PRIMARY + MAX_SECONDARY events.

    Sort key is deterministic: (-severity, filename, timestamp, issue_type).
    Each returned dict gains two extra keys: ``rank_global`` (int) and
    ``tier`` (``"primary"`` | ``"secondary"``).

    Raises RuntimeError on missing or non-numeric severity fields.
    """
    c = cfg or Config()
    max_primary = c.max_primary
    max_secondary = c.max_secondary
    # ── 1. Validate and coerce severity before any sort ──────────────────────
    for ev in all_events:
        if "severity" not in ev:
            raise RuntimeError("Missing severity field.")
        try:
            ev["severity"] = float(ev["severity"])
        except Exception:
            raise RuntimeError("Non-numeric severity detected.")

    # ── 2. Deterministic global sort ─────────────────────────────────────────
    all_events.sort(
        key=lambda e: (
            -e["severity"],
            e["filename"],
            e["timestamp"],
            e["issue_type"],
        )
    )

    # ── 3. Assign rank_global across the FULL sorted list ────────────────────
    for idx, ev in enumerate(all_events):
        ev["rank_global"] = idx + 1  # type: ignore[typeddict-unknown-key]

    # ── 4. Slice after rank assignment ───────────────────────────────────────
    limit = max_primary + max_secondary
    all_events = all_events[:limit]

    # ── 5. Assign tier using rank_global; drop anything that falls outside ───
    for ev in all_events:
        rank = ev["rank_global"]  # type: ignore[typeddict-item]
        if rank <= max_primary:
            ev["tier"] = "primary"  # type: ignore[typeddict-unknown-key]
        elif rank <= max_primary + max_secondary:
            ev["tier"] = "secondary"  # type: ignore[typeddict-unknown-key]
        else:
            ev["tier"] = None  # type: ignore[typeddict-unknown-key]

    all_events = [  # type: ignore[assignment]
        ev for ev in all_events if ev.get("tier") is not None  # type: ignore[typeddict-item]
    ]

    return all_events
