"""JsonFileSink — NDJSON 输出到 logs/app.ndjson, 给 vector/loki/journalctl 消费.

Phase 1 新增; 与 ConsoleSink / EsSink 完全解耦。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..log_record import LogRecord


class JsonFileSink:
    name = "jsonfile"

    def __init__(self, path: str | None = None):
        env = os.getenv("LOG_JSONFILE_PATH")
        self._path = Path(path or env or "logs/app.ndjson")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = None

    def _ensure(self):
        if self._f is None or self._f.closed:
            self._f = open(self._path, "a", encoding="utf-8")
        return self._f

    def write(self, record: LogRecord) -> None:
        try:
            doc = {
                "@timestamp": record.timestamp,
                "trace_id": record.trace_id,
                "span_id": record.span_id,
                "parent_span_id": record.parent_span_id,
                "domain": record.domain,
                "level": record.level,
                "message": record.message,
                "fields": dict(record.fields),
                "duration_ms": record.duration_ms,
            }
            f = self._ensure()
            f.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")
            f.flush()
        except Exception:
            # 写入失败不阻塞业务, 上层 metrics 记录
            pass

    def close(self) -> None:
        if self._f and not self._f.closed:
            self._f.close()
