from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from precise_attribution.core.models import RunRecord


class JsonlRecorder:
    """仅追加写入的原始记录器。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_many(self, records: Iterable[RunRecord]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    def read_all(self) -> list[RunRecord]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        records.append(RunRecord.from_dict(json.loads(line)))
                    except Exception as exc:
                        raise ValueError(f"JSONL 第 {line_number} 行无法解析") from exc
        return records
