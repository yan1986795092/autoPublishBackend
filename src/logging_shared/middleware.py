"""FastAPI 中间件: per-request access-log + trace span.

每个 HTTP 请求:
1. 生成/提取 trace_id，绑定到 structlog context
2. 执行请求
3. 输出一行 access-log（类似 uvicorn 风格）
4. 构造 Jaeger 格式 span，通过 UnifiedEsWriter 写入 traces-*
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .fields import TRACE_ID
from .config import get_logger
from .unified_writer import UnifiedEsWriter

TRACE_ID_HEADER = "X-Trace-Id"


def _get_or_create_trace_id(request: Request) -> str:
    trace_id = request.headers.get(TRACE_ID_HEADER)
    if trace_id:
        return trace_id
    return uuid.uuid4().hex[:16]


def _build_trace_span(
    method: str,
    path: str,
    url: str,
    trace_id: str,
    span_id: str,
    status_code: int,
    elapsed_us: int,
    service_name: str,
    error_info: str = "",
) -> dict:
    """Construct a trace span dict (Jaeger-compatible format) for UnifiedEsWriter."""
    span: dict = {
        "traceID": trace_id,
        "spanID": span_id,
        "operationName": f"{method} {path}",
        "references": [],
        "tags": [
            {"key": "http.method", "vType": "STRING", "vStr": method},
            {"key": "http.url", "vType": "STRING", "vStr": url},
            {"key": "http.status_code", "vType": "INT64", "vInt": status_code},
            {"key": "service.name", "vType": "STRING", "vStr": service_name},
        ],
        "logs": [],
        "process": {"serviceName": service_name, "tags": []},
        "startTime": int((time.time() * 1_000_000) - elapsed_us),
        "startTimeMillis": int((time.time() * 1000) - (elapsed_us // 1000)),
        "durationMicros": elapsed_us,
        "fields": {"traceID": trace_id, "spanID": span_id},
    }
    if error_info:
        span["tags"].append({"key": "error", "vType": "BOOL", "vBool": True})
        span["tags"].append({"key": "error.kind", "vType": "STRING", "vStr": error_info})
    return span


class LoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI 中间件: 单行 access-log + trace span 写入 UnifiedEsWriter."""

    def __init__(self, app: ASGIApp, service_name: str):
        super().__init__(app)
        self.service_name = service_name
        self._http_logger = None

    @property
    def logger(self):
        if self._http_logger is None:
            self._http_logger = get_logger("http")
        return self._http_logger

    @staticmethod
    def _fmt_ms(ms: float) -> str:
        if ms >= 1000:
            return f"{ms / 1000:.1f}s"
        if ms >= 1:
            return f"{ms:.0f}ms"
        if ms > 0:
            return f"{ms:.0f}µs"
        return "0ms"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        trace_id = _get_or_create_trace_id(request)
        span_id = uuid.uuid4().hex[:8]
        start_ns = time.perf_counter_ns()
        method = request.method
        path = request.url.path

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            **{TRACE_ID: trace_id, "service": self.service_name}
        )

        request.state.trace_id = trace_id

        try:
            response = await call_next(request)
            elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000, 2)
            elapsed_us = int((time.perf_counter_ns() - start_ns) / 1000)
            status_code = response.status_code

            # Single access-log line (replaces http_request_start + http_request_end)
            self.logger.info(
                f"{method} {path} {status_code} {self._fmt_ms(elapsed_ms)}",
                _type="http",
                method=method,
                path=path,
                status=status_code,
                duration_ms=elapsed_ms,
                trace_id=trace_id,
            )

            # Write Jaeger span via UnifiedEsWriter
            span = _build_trace_span(
                method=method, path=path, url=str(request.url),
                trace_id=trace_id, span_id=span_id,
                status_code=status_code, elapsed_us=elapsed_us,
                service_name=self.service_name,
            )
            await UnifiedEsWriter.get_instance().write("traces", span)

            response.headers[TRACE_ID_HEADER] = trace_id
            return response

        except Exception as exc:
            elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000, 2)
            elapsed_us = int((time.perf_counter_ns() - start_ns) / 1000)
            error_info = f"{type(exc).__name__}: {exc}"

            self.logger.error(
                f"{method} {path} 500 {self._fmt_ms(elapsed_ms)}",
                _type="http",
                method=method,
                path=path,
                status=500,
                duration_ms=elapsed_ms,
                trace_id=trace_id,
                error=error_info[:200],
            )

            span = _build_trace_span(
                method=method, path=path, url=str(request.url),
                trace_id=trace_id, span_id=span_id,
                status_code=500, elapsed_us=elapsed_us,
                service_name=self.service_name, error_info=error_info,
            )
            await UnifiedEsWriter.get_instance().write("traces", span)

            raise
