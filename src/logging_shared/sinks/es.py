"""EsSink — 把 LogRecord 经由 UnifiedEsWriter 写入 ES (logs-* 索引).

Phase 1 不拆分新索引，沿用现有 logs-* mapping。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..log_record import LogRecord


class EsSink:
    name = "es"

    def __init__(self, index_type: str = "logs"):
        self._index_type = index_type

    async def write_async(self, record: LogRecord) -> None:
        """异步写入 ES. 调用方负责 await, 让 Dispatcher 不阻塞业务."""
        from ..unified_writer import UnifiedEsWriter
        doc = self._to_doc(record)
        try:
            await UnifiedEsWriter.get_instance().write(self._index_type, doc)
        except Exception:
            # sink 层不重试 — 调用方按 sink error 入 metrics
            raise

    def write(self, record: LogRecord) -> None:
        # 同步路径: 把异步任务推到事件循环.
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(self.write_async(record))
        except RuntimeError:
            # 没在事件循环 — 跳过（设计原则 P6: 不静默, 上层 metrics 记录）
            pass

    def _to_doc(self, record: LogRecord) -> dict[str, Any]:
        return {
            "@timestamp": datetime.fromtimestamp(record.timestamp, tz=timezone.utc).isoformat(),
            "trace_id": record.trace_id,
            "span_id": record.span_id,
            "parent_span_id": record.parent_span_id,
            "domain": record.domain,
            "level": record.level,
            "message": record.message,
            "fields": dict(record.fields),
            "duration_ms": record.duration_ms,
            "resource": dict(record.resource),
            "service": record.resource.get("service.name", "unknown"),
        }
