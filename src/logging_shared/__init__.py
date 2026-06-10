"""统一日志模块 — 两服务共用

导出 fields（字段常量）、config（structlog 配置）、middleware（FastAPI 中间件），
以及统一的 get_logger() 函数。
"""

from .agent_emitter import AgentEventEmitter, get_agent_event_emitter
from .agent_events import AgentEvent, EventType, TokenUsage, calculate_cost_usd
from .fields import (
    TRACE_ID, SPAN_ID, SERVICE_NAME, LOG_TIME, LEVEL, EVENT, MESSAGE,
    DURATION_MS, STATUS_CODE, ERROR_TYPE, ERROR_MESSAGE,
    LLM_MODEL, LLM_PROVIDER, LLM_PROMPT_TOKENS, LLM_COMPLETION_TOKENS,
    LLM_TOTAL_TOKENS, LLM_LATENCY_MS, LLM_REQUEST_ID, LLM_STREAMING,
    REQUEST_METHOD, REQUEST_PATH, REQUEST_BODY_SUMMARY, RESPONSE_BODY_SUMMARY,
    SOURCE_APP, USER_ID, CONTENT_TYPE,
    LLM_REQUEST_START, LLM_REQUEST_END, HTTP_REQUEST_START, HTTP_REQUEST_END,
    HTTP_REQUEST_ERROR,
)
from .config import setup, get_logger
from .middleware import LoggingMiddleware, start_span_writer, stop_span_writer
from .unified_writer import UnifiedEsWriter

__all__ = [
    "setup", "get_logger", "LoggingMiddleware", "UnifiedEsWriter",
    "start_span_writer", "stop_span_writer",
    "AgentEvent", "EventType", "TokenUsage", "calculate_cost_usd",
    "AgentEventEmitter", "get_agent_event_emitter",
    "TRACE_ID", "SPAN_ID", "SERVICE_NAME", "LOG_TIME", "LEVEL", "EVENT",
    "MESSAGE", "DURATION_MS", "STATUS_CODE", "ERROR_TYPE", "ERROR_MESSAGE",
    "LLM_MODEL", "LLM_PROVIDER", "LLM_PROMPT_TOKENS", "LLM_COMPLETION_TOKENS",
    "LLM_TOTAL_TOKENS", "LLM_LATENCY_MS", "LLM_REQUEST_ID", "LLM_STREAMING",
    "REQUEST_METHOD", "REQUEST_PATH", "REQUEST_BODY_SUMMARY", "RESPONSE_BODY_SUMMARY",
    "SOURCE_APP", "USER_ID", "CONTENT_TYPE",
    "LLM_REQUEST_START", "LLM_REQUEST_END", "HTTP_REQUEST_END",
    "HTTP_REQUEST_START", "HTTP_REQUEST_ERROR",
]
