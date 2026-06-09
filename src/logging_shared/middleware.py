"""FastAPI 请求/响应结构化日志中间件 + 链路追踪

每个 HTTP 请求:
1. 生成/提取 trace_id，绑定到 structlog context
2. 记录 http_request_start/end 日志 → logs-*
3. 构造 Jaeger 格式 span，直写 traces-*（绕过 OTel Collector）
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .fields import (
    TRACE_ID, DURATION_MS, LEVEL, EVENT, HTTP_REQUEST_START,
    HTTP_REQUEST_END, HTTP_REQUEST_ERROR, REQUEST_METHOD, REQUEST_PATH,
    STATUS_CODE, SOURCE_APP,
)
from .config import get_logger

TRACE_ID_HEADER = "X-Trace-Id"

# ── ES Span Writer ─────────────────────────────────────────────────────────────

_es_span_client: "AsyncElasticsearch | None" = None
_es_span_queue: "asyncio.Queue[dict] | None" = None
_es_span_worker: "asyncio.Task | None" = None
_es_span_running = False


def _get_es_url() -> str:
    import os
    return os.getenv(
        "ELASTICSEARCH_URL",
        os.getenv("OTEL_ELASTICSEARCH_URL", "http://localhost:9200"),
    )


def _get_trace_index() -> str:
    return f"traces-{datetime.now(timezone.utc).strftime('%Y.%m.%d')}"


async def _get_span_es_client() -> "AsyncElasticsearch | None":
    global _es_span_client
    if _es_span_client is None:
        from elasticsearch import AsyncElasticsearch
        url = _get_es_url()
        try:
            _es_span_client = AsyncElasticsearch(
                hosts=[url],
                request_timeout=30,
                retry_on_timeout=True,
                max_retries=3,
            )
            info = await _es_span_client.info()
            get_logger("tracing").info(
                "TracingMiddleware ES client connected to %s", info["version"]["number"]
            )
        except Exception as e:
            get_logger("tracing").warning(
                "TracingMiddleware: cannot connect to ES at %s: %s", url, e
            )
            _es_span_client = None
    return _es_span_client


async def _span_worker(batch_size: int = 50, flush_interval: float = 2.0) -> None:
    global _es_span_running
    _es_span_running = True
    buffer: list[dict] = []

    while _es_span_running or buffer:
        try:
            item = await asyncio.wait_for(
                _es_span_queue.get(), timeout=flush_interval
            ) if _es_span_running else await _es_span_queue.get()
            buffer.append(item)
            _es_span_queue.task_done()
        except asyncio.TimeoutError:
            pass

        if buffer and len(buffer) >= batch_size:
            await _flush_spans(buffer)
            buffer.clear()

    if buffer:
        await _flush_spans(buffer)


async def _flush_spans(buffer: list[dict]) -> None:
    if not buffer:
        return
    es = await _get_span_es_client()
    if not es:
        return
    operations = []
    for doc in buffer:
        operations.append({"index": {"_index": _get_trace_index()}})
        operations.append(doc)
    try:
        resp = await es.bulk(operations=operations, refresh=False)
        if resp.get("errors"):
            get_logger("tracing").warning("ES span bulk write had errors")
    except Exception as e:
        get_logger("tracing").error("ES span bulk write failed: %s", e)


async def start_span_writer() -> None:
    global _es_span_queue, _es_span_worker
    if _es_span_worker is not None:
        return
    _es_span_queue = asyncio.Queue(maxsize=10000)
    _es_span_worker = asyncio.create_task(_span_worker())


async def stop_span_writer() -> None:
    global _es_span_running, _es_span_worker, _es_span_client
    _es_span_running = False
    if _es_span_worker:
        _es_span_worker.cancel()
        try:
            await _es_span_worker
        except asyncio.CancelledError:
            pass
        _es_span_worker = None
    if _es_span_client:
        await _es_span_client.close()
        _es_span_client = None


# ── Middleware ─────────────────────────────────────────────────────────────────

def _get_or_create_trace_id(request: Request) -> str:
    """优先从 X-Trace-Id header 读取，否则生成新的。"""
    trace_id = request.headers.get(TRACE_ID_HEADER)
    if trace_id:
        return trace_id
    return uuid.uuid4().hex[:16]


class LoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI 中间件：为每个请求注入 trace_id，记录日志 + 链路追踪。

    数据流向：
    - logs-* ：http_request_start / http_request_end（structlog 输出）
    - traces-*：Jaeger 格式 span（直写 ES，绕过 OTel Collector）
    """

    def __init__(self, app: ASGIApp, service_name: str):
        super().__init__(app)
        self.service_name = service_name
        self.logger = get_logger("http")

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

        source_app = request.headers.get("X-Source-App", "")

        log = self.logger.bind(
            **{TRACE_ID: trace_id, REQUEST_METHOD: method, REQUEST_PATH: path}
        )

        log.info(
            **{
                EVENT: HTTP_REQUEST_START,
                TRACE_ID: trace_id,
                REQUEST_METHOD: method,
                REQUEST_PATH: path,
                SOURCE_APP: source_app or None,
            },
        )

        # Jaeger 格式 span（构造，不走 OTel SDK）
        span: dict = {
            "operationName": f"{method} {path}",
            "references": [],
            "tags": [
                {"key": "http.method", "vType": "STRING", "vStr": method},
                {"key": "http.url", "vType": "STRING", "vStr": str(request.url)},
                {"key": "service.name", "vType": "STRING", "vStr": self.service_name},
            ],
            "logs": [],
            "process": {"serviceName": self.service_name, "tags": []},
        }

        try:
            response = await call_next(request)
            elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000, 2)
            elapsed_us = int((time.perf_counter_ns() - start_ns) / 1000)

            status_code = response.status_code
            span["tags"].append(
                {"key": "http.status_code", "vType": "INT64", "vInt": status_code}
            )
            if status_code >= 400:
                span["tags"].append({"key": "error", "vType": "BOOL", "vBool": True})

            log.info(
                **{
                    EVENT: HTTP_REQUEST_END,
                    STATUS_CODE: status_code,
                    DURATION_MS: elapsed_ms,
                },
            )
            response.headers[TRACE_ID_HEADER] = trace_id
            response.headers["X-Span-Id"] = span_id

            # 直写 traces-*
            span["traceID"] = trace_id
            span["spanID"] = span_id
            span["startTime"] = int((time.time() * 1_000_000) - elapsed_us)
            span["startTimeMillis"] = span["startTime"] // 1000
            span["durationMicros"] = elapsed_us
            span["fields"] = {"traceID": trace_id, "spanID": span_id}
            _emit_span(span)

            return response

        except Exception as exc:
            elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000, 2)
            elapsed_us = int((time.perf_counter_ns() - start_ns) / 1000)

            span["traceID"] = trace_id
            span["spanID"] = span_id
            span["startTime"] = int((time.time() * 1_000_000) - elapsed_us)
            span["startTimeMillis"] = span["startTime"] // 1000
            span["durationMicros"] = elapsed_us
            span["fields"] = {"traceID": trace_id, "spanID": span_id}
            span["tags"].extend([
                {"key": "error", "vType": "BOOL", "vBool": True},
                {"key": "error.kind", "vType": "STRING", "vStr": type(exc).__name__},
                {"key": "error.message", "vType": "STRING", "vStr": str(exc)},
            ])
            _emit_span(span)

            log.error(
                **{
                    EVENT: HTTP_REQUEST_ERROR,
                    STATUS_CODE: 500,
                    DURATION_MS: elapsed_ms,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            raise


def _emit_span(span: dict) -> None:
    """非阻塞地将 span 加入写入队列。"""
    if _es_span_queue is None:
        return
    try:
        _es_span_queue.put_nowait(span)
    except asyncio.QueueFull:
        pass
