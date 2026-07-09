"""LogRecord — 阶段 1 引入的核心数据结构。

对齐 OpenTelemetry Logs Data Model + 1.30+ GenAI Attributes。
唯一信源：所有日志走 Dispatcher.emit(record)。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR"]
LogDomain = Literal["http", "agent", "llm", "db", "system", "span"]


def _new_trace_id() -> str:
    return uuid.uuid4().hex[:32]


def _new_span_id() -> str:
    return uuid.uuid4().hex[:16]


def _default_resource() -> dict[str, str]:
    return {"service.name": "unknown"}


@dataclass(frozen=True)
class LogRecord:
    """对齐 OpenTelemetry Logs Data Model 的结构化日志记录。

    fields 字典遵循 OTel 命名: http.method, http.status_code, http.target,
    gen_ai.model, gen_ai.usage.input_tokens 等（详见 spec 文档 §4.1 表）。
    """

    level: LogLevel
    domain: LogDomain
    message: str
    fields: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    trace_id: str = field(default_factory=_new_trace_id)
    span_id: str = field(default_factory=_new_span_id)
    parent_span_id: str | None = None
    duration_ms: float | None = None
    error: BaseException | None = None
    resource: Mapping[str, str] = field(default_factory=_default_resource)
