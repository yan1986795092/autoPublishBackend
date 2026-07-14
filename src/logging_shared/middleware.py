"""FastAPI 中间件: 降噪版 access-log + trace span。

相较于原版，核心改进：
1. /metrics、/health、/docs 等路径静默（不输出日志）
2. OPTIONS 请求静默
3. 业务路径使用 TraceContext 输出 ⚡ REQUEST / ⚡ RESPONSE 摘要
4. 4xx warning 级别，5xx error 级别，2xx/3xx info 级别
5. 内部错误仍然保持 trace span 写入 ES
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
from .trace_context import TraceContext
from .unified_writer import UnifiedEsWriter

TRACE_ID_HEADER = "X-Trace-Id"

# 静默路径 — 不输出 access-log
SILENT_PATHS = frozenset({
    "/metrics", "/health", "/favicon.ico",
    "/docs", "/redoc", "/openapi.json",
})

# 静默方法
SILENT_METHODS = frozenset({"OPTIONS"})


def _is_silent(method: str, path: str) -> bool:
    """判断是否应当静默（不输出任何日志）"""
    if method in SILENT_METHODS:
        return True
    if path in SILENT_PATHS:
        return True
    # swagger UI 的静态资源
    if path.startswith(("/docs/", "/redoc/")):
        return True
    return False


def _get_or_create_trace_id(request: Request) -> str:
    trace_id = request.headers.get(TRACE_ID_HEADER)
    if trace_id:
        return trace_id
    return uuid.uuid4().hex[:16]


def _build_trace_span(
    method: str, path: str, url: str,
    trace_id: str, span_id: str,
    status_code: int, elapsed_us: int,
    service_name: str,
    error_info: str = "",
) -> dict:
    """构造一个 trace span 字典（Jaeger 兼容格式），用于 UnifiedEsWriter。"""
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
            # P1.2: OTel 对齐标签（与 Jaeger 命名共存）
            {"key": "otel.trace_id", "vType": "STRING", "vStr": trace_id},
            {"key": "otel.span_id", "vType": "STRING", "vStr": span_id},
            {"key": "otel.http.method", "vType": "STRING", "vStr": method},
            {"key": "otel.http.target", "vType": "STRING", "vStr": path},
            {"key": "otel.http.status_code", "vType": "INT64", "vInt": status_code},
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
    """FastAPI 中间件: 降噪版 access-log + trace span。"""

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

        # ── P0.5：捕获请求上下文字段 ──
        user_id = getattr(request.state, "user_id", "") or request.headers.get("X-User-Id", "")
        session_id = getattr(request.state, "session_id", "") or request.headers.get("X-Session-Id", "")
        tenant_id = request.headers.get("X-Tenant-Id", "")
        # client_ip: 优先 X-Forwarded-For，其次 X-Real-IP，最后直连 IP
        forwarded = request.headers.get("X-Forwarded-For", "")
        client_ip = forwarded.split(",")[0].strip() if forwarded else ""
        if not client_ip:
            client_ip = request.headers.get("X-Real-IP", "")
        if not client_ip and request.client:
            client_ip = request.client.host or ""
        # P1.2: 补充 user_agent / referer / query_string
        user_agent = request.headers.get("User-Agent", "")
        referer = request.headers.get("Referer", "")
        query_string = str(request.query_params) if request.query_params else ""

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            **{
                TRACE_ID: trace_id,
                "service": self.service_name,
                "user_id": user_id,
                "session_id": session_id,
                "client_ip": client_ip,
                "tenant_id": tenant_id,
                "user_agent": user_agent[:128] if user_agent else "",
            }
        )

        request.state.trace_id = trace_id
        request.state.user_id = user_id
        request.state.session_id = session_id
        request.state.client_ip = client_ip
        request.state.tenant_id = tenant_id

        # ── 静默路径：不输出日志，只设 trace_id ──
        if _is_silent(method, path):
            try:
                response = await call_next(request)
            except Exception:
                raise
            response.headers[TRACE_ID_HEADER] = trace_id
            return response

        # ── 业务路径：创建 TraceContext 并输出日志 ──
        trace_ctx = TraceContext(
            trace_id=trace_id, method=method, path=path,
            user_id=user_id, session_id=session_id,
            client_ip=client_ip, tenant_id=tenant_id,
        )
        request.state.trace_ctx = trace_ctx

        # 尝试读取请求体摘要（仅 POST/PUT/PATCH）
        if method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
                text = body.decode("utf-8", errors="replace")[:200]
                trace_ctx.request_summary = text.replace("\n", " ").strip()
                trace_ctx.request_size = len(body)
            except Exception:
                pass

        # Phase 2 关键改动：HTTP 行改 _type=http, 字段对齐 OTel
        try:
            self.logger.info(
                f"REQUEST  {method} {path}",
                _type="http",
                event_name="http_request",
                message=trace_ctx.request_summary or "",
                http_method=method,
                http_target=path,
                trace_id=trace_id,
                user_id=user_id or None,
                session_id=session_id or None,
                client_ip=client_ip or None,
                request_size=trace_ctx.request_size or None,
                user_agent=user_agent[:128] or None,
                referer=referer[:256] or None,
                query_string=query_string[:256] or None,
            )
        except UnicodeEncodeError as _exc:
            import sys as _sys
            _sys.stderr.write(f"[log-encoding-error] REQUEST trace={trace_id[:12]} method={method} path={path} err={_exc}\n")

        try:
            response = await call_next(request)
            elapsed_ms = round((time.perf_counter_ns() - start_ns) / 1_000_000, 2)
            elapsed_us = int((time.perf_counter_ns() - start_ns) / 1000)
            status_code = response.status_code

            trace_ctx.status_code = status_code
            trace_ctx.response_summary = _summarize_response(response)

            # P0.5：捕获响应大小
            content_length = response.headers.get("content-length")
            if content_length:
                trace_ctx.response_size = int(content_length)

            # P0.5：从响应头或响应体提取 business_code
            business_code = response.headers.get("X-Business-Code", "")
            if not business_code and response.headers.get("content-type", "").startswith("application/json"):
                try:
                    body = response.body
                    if body:
                        import json as _json
                        _parsed = _json.loads(body)
                        bc = _parsed.get("code") or _parsed.get("business_code") or _parsed.get("status")
                        if bc and isinstance(bc, str):
                            business_code = str(bc)
                        trace_ctx.response_size = len(body)
                except Exception:
                    pass

            # 4xx/5xx 用更高的日志级别
            log_method = self.logger.info
            if status_code >= 500:
                log_method = self.logger.error
            elif status_code >= 400:
                log_method = self.logger.warning

            try:
                log_method(
                    f"RESPONSE  {trace_ctx.to_response_line()}",
                    _type="http",
                    event_name="http_response",
                    message=trace_ctx.response_summary or "",
                    http_method=method,
                    http_target=path,
                    http_status_code=status_code,
                    duration_ms=elapsed_ms,
                    trace_id=trace_id,
                    user_id=user_id or None,
                    session_id=session_id or None,
                    client_ip=client_ip or None,
                    request_size=trace_ctx.request_size or None,
                    response_size=trace_ctx.response_size or None,
                    business_code=business_code or None,
                )
            except UnicodeEncodeError as _exc:
                import sys as _sys
                _sys.stderr.write(f"[log-encoding-error] RESPONSE trace={trace_id[:12]} status={status_code} err={_exc}\n")

            # ES trace span（始终写入，不受静默影响）
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

            trace_ctx.status_code = 500
            trace_ctx.error = error_info[:200]

            try:
                self.logger.error(
                    f"RESPONSE  {trace_ctx.to_response_line()}",
                    _type="http",
                    event_name="http_response_error",
                    message=trace_ctx.error or "",
                    http_method=method,
                    http_target=path,
                    http_status_code=500,
                    duration_ms=elapsed_ms,
                    trace_id=trace_id,
                    error=error_info[:200],
                    user_id=user_id or None,
                    session_id=session_id or None,
                    client_ip=client_ip or None,
                )
            except UnicodeEncodeError as _exc:
                import sys as _sys
                _sys.stderr.write(f"[log-encoding-error] RESPONSE_ERROR trace={trace_id[:12]} err={_exc}\n")

            span = _build_trace_span(
                method=method, path=path, url=str(request.url),
                trace_id=trace_id, span_id=span_id,
                status_code=500, elapsed_us=elapsed_us,
                service_name=self.service_name, error_info=error_info,
            )
            await UnifiedEsWriter.get_instance().write("traces", span)

            raise


def _summarize_response(response: Response) -> str:
    """从响应中提取摘要文本"""
    ct = response.headers.get("content-type", "")
    cl = response.headers.get("content-length", "")
    parts = []
    if "text/event-stream" in ct:
        parts.append("SSE stream")
    elif "application/json" in ct:
        parts.append("json")
    else:
        parts.append(ct.split(";")[0] if ct else "unknown")
    if cl:
        size = int(cl)
        if size < 1024:
            parts.append(f"{size}B")
        elif size < 1024 * 1024:
            parts.append(f"{size/1024:.0f}KB")
        else:
            parts.append(f"{size/1024/1024:.1f}MB")
    return " ".join(parts)
