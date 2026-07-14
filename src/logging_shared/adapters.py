"""适配器：将旧版 AgentEvent 转换为新 Dispatcher 的 LogRecord。

用法::

    from .adapters import agent_event_to_logrecord
    from .dispatcher import get_dispatcher

    record = agent_event_to_logrecord(event)
    get_dispatcher().emit(record)
"""

from __future__ import annotations

from typing import Any

from .agent_events import AgentEvent
from .log_record import LogLevel, LogRecord  # noqa: F401  # BaseException used in type hint


def agent_event_to_logrecord(event: AgentEvent) -> LogRecord:
    """将 AgentEvent 转换为 LogRecord 供 Dispatcher 分派。

    所有 AgentEvent 的字段（含 P0/P2 新增的请求上下文和三层字段）
    都被映射到 LogRecord.fields 字典中，保留 OTel 风格命名。
    """
    # 规范化日志级别
    level_map = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARN",
        "WARN": "WARN",
        "ERROR": "ERROR",
        "CRITICAL": "ERROR",
    }
    log_level: LogLevel = level_map.get(event.level.upper(), "INFO")  # type: ignore[assignment]

    # 构造结构化 fields 字典
    fields: dict[str, Any] = {
        "event_type": event.event_type,
        "agent.name": event.agent_name,
        "run_id": event.run_id,
        "step_index": event.step_index,
        "session_id": event.session_id,
        # 请求上下文
        "user_id": event.user_id or None,
        "client_ip": event.client_ip or None,
        "http_method": event.http_method or None,
        "http_path": event.http_path or None,
        "business_code": event.business_code or None,
        # 三层字段
        "total_cost_usd": event.total_cost_usd,
        "final_status": event.final_status,
        "task_duration_ms": event.task_duration_ms,
        "decision_reason": event.decision_reason,
        "standardized_error_code": event.standardized_error_code,
    }

    # 工具调用字段（可选）
    if event.tool_name:
        fields["tool.name"] = event.tool_name
    if event.tool_input:
        fields["tool.params"] = event.tool_input
    if event.tool_output:
        fields["tool.output"] = event.tool_output
    if event.thought:
        fields["agent.thought"] = event.thought

    # LLM 字段
    if event.llm_model:
        fields["gen_ai.model"] = event.llm_model
    if event.llm_provider:
        fields["gen_ai.provider"] = event.llm_provider
    if event.token_usage:
        fields["gen_ai.usage.prompt_tokens"] = event.token_usage.prompt_tokens
        fields["gen_ai.usage.completion_tokens"] = event.token_usage.completion_tokens
        fields["gen_ai.usage.total_tokens"] = event.token_usage.total_tokens
        fields["gen_ai.usage.cost_usd"] = event.token_usage.cost_usd

    # 错误字段
    error_type = event.error_type or event.standardized_error_code
    if error_type:
        fields["error_type"] = error_type
    if event.error_message:
        fields["error_message"] = event.error_message
    if event.error_stack:
        fields["error_stack"] = event.error_stack

    # 构造 error 对象（如果有）
    error_obj: BaseException | None = None
    if event.error_type and event.error_message:
        error_obj = Exception(f"[{event.error_type}] {event.error_message}")

    return LogRecord(
        level=log_level,
        domain="agent",
        message=event.message or event.event_type,
        fields=fields,
        timestamp=event.timestamp,
        trace_id=event.trace_id,
        span_id=event.parent_span_id or "",
        parent_span_id=event.parent_span_id,
        duration_ms=event.duration_ms,
        error=error_obj,
        resource={"service.name": event.service or "autopublish-api"},
    )
