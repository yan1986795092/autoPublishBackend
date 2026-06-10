"""Lightweight agent tracing — OTel-compatible interface → UnifiedEsWriter

Replaces the OTel tracer's no-op spans with real trace writes to ES.
Provides AgentTracer / AgentSpan that mirror OTel's start_as_current_span
API but write directly to UnifiedEsWriter, avoiding the OTel exporter overhead.

Usage (swap in nodes.py / orchestrator.py):
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
    """Lightweight span that writes to UnifiedEsWriter on exit.

    Mirrors OTel Span interface: add_event, set_attribute, set_status, record_exception.
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
        """Accepts OTel Status objects or plain strings."""
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
        writer.write("traces", span_doc)


class AgentTracer:
    """Lightweight tracer producing AgentSpan → UnifiedEsWriter.

    Interface-compatible with OTel Tracer.start_as_current_span().
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
    """Get or create an AgentTracer."""
    key = trace_id or "_default"
    if key not in _tracer_cache:
        _tracer_cache[key] = AgentTracer(trace_id=trace_id)
    return _tracer_cache[key]
