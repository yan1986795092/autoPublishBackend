from __future__ import annotations

import os
from typing import Any, Callable

from .agent_events import AgentEvent, TokenUsage, calculate_cost_usd
from .config import get_logger
from .fields import (
    CLIENT_IP, USER_ID, BUSINESS_CODE, ERROR_CODE,
    REQUEST_SIZE, RESPONSE_SIZE,
)


class AgentEventEmitter:
    def __init__(self, env: str = "development", es_writer: Callable[[dict[str, Any]], None] | None = None):
        self._env = env
        self._logger = get_logger("agent.events")
        self._step_counters: dict[str, int] = {}
        self._es_writer = es_writer or self._default_es_writer

    def _default_es_writer(self, event_dict: dict[str, Any]) -> None:
        """默认 ES 写入器：使用 UnifiedEsWriter。"""
        try:
            from .unified_writer import UnifiedEsWriter
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(UnifiedEsWriter.get_instance().write("events", event_dict))
                else:
                    pass  # 没有正在运行的事件循环，跳过
            except RuntimeError:
                pass
        except Exception:
            pass

    def reset_counter(self, run_id: str) -> None:
        self._step_counters[run_id] = 0

    def _get_step_index(self, run_id: str) -> int:
        current = self._step_counters.get(run_id, 0)
        self._step_counters[run_id] = current + 1
        return current

    def _truncate(self, text: str | None, max_length: int) -> str | None:
        """Phase 2 关键改动：production 不再静默截断 thought.

        上下游依赖 thought 全部内容做 agent 决策，截断会误导 agent。
        控制台/ES 端由 Sampler 显式控制采样率。
        """
        return text

    def _truncate_dict(self, payload: dict[str, Any] | None, max_length: int) -> dict[str, Any] | None:
        if payload is None:
            return None
        return payload

    def _prepare_event(self, event: AgentEvent) -> dict[str, Any]:
        if event.run_id and event.step_index == 0:
            event.step_index = self._get_step_index(event.run_id)
        event.thought = self._truncate(event.thought, 2000)
        event.tool_output = self._truncate(event.tool_output, 500)
        event.message = self._truncate(event.message, 500) or ""
        event.tool_input = self._truncate_dict(event.tool_input, 500)
        if event.token_usage and event.llm_model:
            event.token_usage.cost_usd = calculate_cost_usd(event.token_usage, event.llm_model)
        return event.to_dict()

    async def emit(
        self,
        event: AgentEvent,
        es_writer: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        event_dict = self._prepare_event(event)
        self._emit_console(event_dict)
        (es_writer or self._es_writer)(event_dict)
        # P3.3: 双写到新 Dispatcher
        try:
            from .adapters import agent_event_to_logrecord
            from .dispatcher import get_dispatcher
            get_dispatcher().emit(agent_event_to_logrecord(event))
        except Exception:
            pass  # Dispatcher 异常不阻塞业务
        return event_dict

    def emit_sync(
        self,
        event: AgentEvent,
        es_writer: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        event_dict = self._prepare_event(event)
        self._emit_console(event_dict)
        (es_writer or self._es_writer)(event_dict)
        # P3.3: 双写到新 Dispatcher
        try:
            from .adapters import agent_event_to_logrecord
            from .dispatcher import get_dispatcher
            get_dispatcher().emit(agent_event_to_logrecord(event))
        except Exception:
            pass  # Dispatcher 异常不阻塞业务
        return event_dict

    def _emit_console(self, event_dict: dict[str, Any]) -> None:
        """输出到控制台 — 统一结构化路径。

        所有环境走同一套结构化输出（_type="agent" + 全字段）。
        level 过滤由 structlog filter_by_level 处理，emitter 不重复判断。
        """
        level = str(event_dict.get("level", "INFO")).upper()
        event_type = event_dict.get("event_type", "")

        # Phase 2 关键改动：字段对齐 OTel（向后兼容保留原字段名）
        console_event: dict[str, Any] = {
            "_type": "agent",
            "event_type": event_type,
            "agent.name": event_dict.get("agent_name", ""),
            "run_id": event_dict.get("run_id", ""),
            "duration_ms": event_dict.get("duration_ms"),
            "token_usage": event_dict.get("token_usage"),
            "tool.name": event_dict.get("tool_name", ""),
            "gen_ai.model": event_dict.get("llm_model", ""),
            "error_type": event_dict.get("error_type", ""),
            "error_message": event_dict.get("error_message", ""),
            "trace_id": event_dict.get("trace_id", ""),
            "level": level,
            "agent_name": event_dict.get("agent_name", ""),  # 别名兼容
            # P0 新增：请求上下文字段（使用 fields.py 常量）
            USER_ID: event_dict.get("user_id", ""),
            CLIENT_IP: event_dict.get("client_ip", ""),
            "http_method": event_dict.get("http_method", ""),
            "http_path": event_dict.get("http_path", ""),
            BUSINESS_CODE: event_dict.get("business_code", ""),
        }

        # 直接输出完整内容，不截断
        if event_type == "thought":
            thought = event_dict.get("thought", "")
            if thought:
                console_event["agent.thought_summary"] = thought

        elif event_type == "tool_call":
            tool_name = event_dict.get("tool_name", "")
            tool_input = event_dict.get("tool_input")
            if tool_input is not None:
                console_event["tool.params"] = tool_input
                console_event["message"] = str(tool_input)

        elif event_type == "tool_result":
            output = event_dict.get("tool_output", "")
            if output:
                console_event["tool.output_summary"] = output
                console_event["message"] = output

        elif event_type in ("agent_start", "plan", "agent_finish"):
            msg = event_dict.get("message", "") or event_type
            if msg:
                console_event["message"] = msg

        try:
            self._logger.info(event_type, **console_event)
        except UnicodeEncodeError as _exc:
            import sys as _sys
            _sys.stderr.write(f"[log-encoding-error] agent_emit event={event_type} trace={console_event.get('trace_id','')[:12]} err={_exc}\n")


_emitter_singleton: AgentEventEmitter | None = None


def get_agent_event_emitter(env: str | None = None) -> AgentEventEmitter:
    global _emitter_singleton
    if _emitter_singleton is None:
        _emitter_singleton = AgentEventEmitter(env=env or os.getenv("LOG_ENV", "development"))
    return _emitter_singleton
