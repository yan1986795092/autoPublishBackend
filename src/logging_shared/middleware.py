"""FastAPI 请求/响应结构化日志中间件"""

import time
import uuid
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


def _get_or_create_trace_id(request: Request) -> str:
    """优先从 X-Trace-Id header 读取，否则生成新的。"""
    trace_id = request.headers.get(TRACE_ID_HEADER)
    if trace_id:
        return trace_id
    return uuid.uuid4().hex[:16]


class LoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI 中间件：为每个请求注入 trace_id，记录请求/响应日志。"""

    def __init__(self, app: ASGIApp, service_name: str):
        super().__init__(app)
        self.service_name = service_name
        self.logger = get_logger("http")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        trace_id = _get_or_create_trace_id(request)
        started_at = time.perf_counter()
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

        try:
            response = await call_next(request)
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
            log.info(
                **{
                    EVENT: HTTP_REQUEST_END,
                    STATUS_CODE: response.status_code,
                    DURATION_MS: elapsed_ms,
                },
            )
            response.headers[TRACE_ID_HEADER] = trace_id
            return response
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
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
