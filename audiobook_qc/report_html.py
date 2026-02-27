"""
audiobook_qc/report_html.py

Generates a static review.html file for manual QC sign-off.
No external CSS or JS frameworks; everything is self-contained.
"""

from __future__ import annotations

import html
import os

from audiobook_qc.detectors import Event


def write_html(ranked_events: list[Event], folder: str) -> str:
    """
    Generate ``review.html`` inside *folder* from *ranked_events*.

    Returns the absolute path of the written file.
    """
    output_path = os.path.join(folder, "review.html")

    rows_html = _build_rows(ranked_events)
    page = _PAGE_TEMPLATE.replace("{{ROWS}}", rows_html)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(page)

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _build_rows(events: list[Event]) -> str:
    parts: list[str] = []
    for ev in events:
        rank = int(ev["rank_global"])  # type: ignore[typeddict-item]
        tier = str(ev["tier"])  # type: ignore[typeddict-item]
        filename = html.escape(str(ev["filename"]))
        ts = round(float(ev["timestamp"]), 3)
        mins, secs = divmod(ts, 60)
        minsec = html.escape(f"{int(mins)}:{secs:06.3f}")
        issue = html.escape(str(ev["issue_type"]))
        sev = round(float(ev["severity"]), 4)

        row_class = "primary-row" if tier == "primary" else "secondary-row"

        parts.append(
            f'    <tr class="{row_class}">\n'
            f'      <td>{rank}</td>\n'
            f'      <td>{html.escape(tier)}</td>\n'
            f'      <td>{filename}</td>\n'
            f'      <td data-seconds="{ts}">{minsec}</td>\n'
            f'      <td>{issue}</td>\n'
            f'      <td>{sev}</td>\n'
            f'      <td class="player-cell">\n'
            f'        <audio controls preload="metadata"'
            f' ontimeupdate="updateTime(this)">\n'
            f'          <source src="{filename}" type="audio/wav">\n'
            f'        </audio>'
            f'<span class="cur-time">0:00</span>\n'
            f'        <div class="player-btns">\n'
            f'          <button onclick="playIssue(this, {ts})">'
            f"\u25b6 Play issue</button>\n"
            f'          <button onclick="seekRel(this, -5)">'
            f"\u23ea \u22125s</button>\n"
            f'          <button onclick="seekRel(this, 5)">'
            f"\u23e9 +5s</button>\n"
            f"        </div>\n"
            f"      </td>\n"
            f'      <td><input type="checkbox"></td>\n'
            f'      <td><textarea class="comment" rows="2" cols="28"'
            f' placeholder="Notes\u2026"></textarea></td>\n'
            f"    </tr>\n"
        )

    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Audiobook QC Review</title>
  <style>
    body {
      font-family: sans-serif;
      padding: 1rem 2rem;
    }
    h1 {
      margin-bottom: 0.5rem;
    }
    .controls {
      margin-bottom: 1rem;
    }
    table {
      border-collapse: collapse;
      width: 100%;
    }
    th, td {
      border: 1px solid #ccc;
      padding: 0.4rem 0.6rem;
      text-align: left;
      vertical-align: middle;
    }
    th {
      background: #f0f0f0;
    }
    tr.primary-row {
      background-color: #ffe0e0;
    }
    tr.secondary-row {
      background-color: #fffde0;
    }
    .player-cell {
      white-space: nowrap;
    }
    audio {
      height: 32px;
      vertical-align: middle;
    }
    .cur-time {
      display: inline-block;
      font-family: monospace;
      font-size: 0.85rem;
      margin: 0 6px;
      min-width: 4ch;
      vertical-align: middle;
    }
    .player-btns {
      margin-top: 4px;
    }
    .player-btns button {
      margin-right: 4px;
      cursor: pointer;
    }
    textarea.comment {
      resize: vertical;
      font-family: sans-serif;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <h1>Audiobook QC Review</h1>
  <div class="controls">
    <button onclick="exportSelected()">Export Selected as CSV</button>
  </div>
  <table>
    <thead>
      <tr>
        <th>rank</th>
        <th>tier</th>
        <th>filename</th>
        <th>timestamp</th>
        <th>issue_type</th>
        <th>severity</th>
        <th>player</th>
        <th>select</th>
        <th>notes</th>
      </tr>
    </thead>
    <tbody>
{{ROWS}}    </tbody>
  </table>
  <script>
    function updateTime(audio) {
      var span = audio.nextElementSibling;
      if (span && span.classList.contains("cur-time")) {
        var t = audio.currentTime;
        var m = Math.floor(t / 60);
        var s = t % 60;
        span.textContent = m + ":" + (s < 10 ? "0" : "") + s.toFixed(1);
      }
    }

    function playIssue(button, timestamp) {
      var row = button.closest("tr");
      var audio = row.querySelector("audio");
      audio.currentTime = Math.max(0, timestamp - 5);
      audio.play();
    }

    function seekRel(button, delta) {
      var row = button.closest("tr");
      var audio = row.querySelector("audio");
      audio.currentTime = Math.max(0, audio.currentTime + delta);
      audio.play();
    }

    function exportSelected() {
      var rows = document.querySelectorAll("tbody tr");
      var csv = "rank_global,tier,filename,timestamp_seconds,issue_type,severity_score,notes\\n";

      rows.forEach(function(row) {
        var checkbox = row.querySelector("input[type='checkbox']");
        if (checkbox && checkbox.checked) {
          var cols = row.querySelectorAll("td");
          var note = (row.querySelector("textarea.comment") || {value: ""}).value;
          note = note.replace(/\r?\n/g, " ").replace(/,/g, ";");
          csv += [
            cols[0].innerText,
            cols[1].innerText,
            cols[2].innerText,
            cols[3].dataset.seconds,
            cols[4].innerText,
            cols[5].innerText,
            "\\"" + note + "\\""
          ].join(",") + "\\n";
        }
      });

      var blob = new Blob([csv], { type: "text/csv" });
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = "selected_issues.csv";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  </script>
</body>
</html>
"""
