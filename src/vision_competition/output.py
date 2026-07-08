from __future__ import annotations

from pathlib import Path

from .config import CompetitionConfig
from .types import OutputRow


def result_filename(config: CompetitionConfig) -> str:
    return f"{config.team_prefix}-{config.team_name}-R{config.round_id}.txt"


def write_result_file(config: CompetitionConfig, rows: list[OutputRow]) -> Path:
    out_dir = Path(config.result_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / result_filename(config)

    lines = ["START"]
    for row in rows:
        lines.append(f"{row.object_id};{row.num};{row.table_id}")
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def serialize_rows(rows: list[OutputRow]) -> str:
    lines = ["START"]
    lines.extend(f"{row.object_id};{row.num};{row.table_id}" for row in rows)
    lines.append("END")
    return "\n".join(lines)

