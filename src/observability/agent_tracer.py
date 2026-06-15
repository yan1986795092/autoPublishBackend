"""轻量级 Agent 追踪 —— 符合 OTel 接口 → UnifiedEsWriter

替代 OTel tracer 的 no-op span，将真实的 trace 写入 ES。
提供 AgentTracer / AgentSpan，镜像 OTel 的 start_as_current_span
API，但直接写入 UnifiedEsWriter，避免 OTel exporter 的开销。

用法（在 nodes.py / orchestrator.py 中替换）：
    from src.observability.agent_tracer import get_agent_tracer
    tracer = get_agent_tracer()
    with tracer.start_as_current_span("agent.classify", ...) as span:
        span.set_attribute("key", "value")
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from src.logging_shared.unified_writer import UnifiedEsWriter


@dataclass
class AgentSpan:
    """轻量级 span，退出时写入 UnifiedEsWriter。

    镜像 OTel Span 接口：add_event, set_attribute, set_status, record_exception。
    """

    name: str
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    kind: str = "INTERNAL"
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    status: str = "UNSET"
    start_time: float = 0.0

    def __post_init__(self):
        if not self.span_id:
            self.span_id = uuid.uuid4().hex[:8]
        self.start_time = time.time()

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        self.events.append({
            "name": name,
            "attributes": attributes or {},
            "timestamp": time.time(),
        })

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, status: Any) -> None:
        """接受 OTel Status 对象或普通字符串。"""
        if hasattr(status, 'status_code'):
            self.status = status.status_code.name if hasattr(status.status_code, 'name') else str(status.status_code)
        else:
            self.status = str(status)

    def record_exception(self, exc: Exception) -> None:
        self.attributes["error.type"] = type(exc).__name__
        self.attributes["error.message"] = str(exc)
        self.add_event("exception", {"exception.message": str(exc)})

    def end(self) -> None:
        duration_us = int((time.time() - self.start_time) * 1_000_000)
        writer = UnifiedEsWriter.get_instance()

        tags = []
        for k, v in self.attributes.items():
            if isinstance(v, bool):
                tags.append({"key": k, "vType": "BOOL", "vBool": v})
            elif isinstance(v, (int, float)):
                tags.append({"key": k, "vType": "INT64" if isinstance(v, int) else "DOUBLE", "vInt" if isinstance(v, int) else "vDouble": v})
            else:
                tags.append({"key": k, "vType": "STRING", "vStr": str(v)})

        span_doc = {
            "traceID": self.trace_id or uuid.uuid4().hex[:16],
            "spanID": self.span_id,
            "operationName": self.name,
            "references": [],
            "tags": tags,
            "logs": [
                {
                    "timestamp": int(e["timestamp"] * 1_000_000),
                    "fields": [{"key": k, "vType": "STRING", "vStr": str(v)} for k, v in e.get("attributes", {}).items()],
                }
                for e in self.events
            ],
            "process": {"serviceName": "autopublish-api", "tags": []},
            "startTime": int(self.start_time * 1_000_000),
            "startTimeMillis": int(self.start_time * 1000),
            "durationMicros": duration_us,
            "status": self.status,
        }
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(writer.write("traces", span_doc))
        except RuntimeError:
            pass  # 无事件循环，跳过 ES 写入


class AgentTracer:
    """轻量级 tracer，生成 AgentSpan → UnifiedEsWriter。

    接口兼容 OTel Tracer.start_as_current_span()。
    """
    def __init__(self, trace_id: str = ""):
        self._trace_id = trace_id

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        attributes: dict | None = None,
        kind: Any = None,
        **kwargs,
    ) -> Generator[AgentSpan, None, None]:
        span = AgentSpan(
            name=name,
            trace_id=self._trace_id or kwargs.get("trace_id", ""),
            attributes=dict(attributes or {}),
            kind=str(kind or "INTERNAL"),
        )
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status("ERROR")
            raise
        finally:
            span.end()


_tracer_cache: dict[str, AgentTracer] = {}


def get_agent_tracer(trace_id: str = "") -> AgentTracer:
    """获取或创建一个 AgentTracer。"""
    key = trace_id or "_default"
    if key not in _tracer_cache:
        _tracer_cache[key] = AgentTracer(trace_id=trace_id)
    return _tracer_cache[key]
