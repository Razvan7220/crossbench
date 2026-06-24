"""Export results in various formats."""

from __future__ import annotations

import csv
import io
import json
from typing import Any


def to_json(data: Any, indent: int = 2) -> str:
    return json.dumps(data, indent=indent, default=str)


def to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def to_markdown_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if not rows:
        return "*No results*"

    cols = columns or list(rows[0].keys())
    lines = []

    # Header
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")

    # Rows
    for row in rows:
        values = []
        for c in cols:
            v = row.get(c, "")
            if isinstance(v, float):
                values.append(f"{v:.4f}")
            else:
                values.append(str(v) if v is not None else "")
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)
