from __future__ import annotations

import os
from typing import Any, Callable

from .agent_events import AgentEvent, TokenUsage, calculate_cost_usd
from .config import get_logger
from .agent_events import AgentEvent, TokenUsage, calculate_cost_usd
from .config import get_logger


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
                    UnifiedEsWriter.get_instance().write("events", event_dict)
                else:
                    pass  # No running loop, skip
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
        if text is None:
            return None
        if self._env == "production" and len(text) > max_length:
            return text[:max_length] + "..."
        return text

    def _truncate_dict(self, payload: dict[str, Any] | None, max_length: int) -> dict[str, Any] | None:
        if payload is None:
            return None
        rendered = str(payload)
        if self._env == "production" and len(rendered) > max_length:
            return {"preview": rendered[:max_length] + "..."}
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
        return event_dict

    def emit_sync(
        self,
        event: AgentEvent,
        es_writer: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        event_dict = self._prepare_event(event)
        self._emit_console(event_dict)
        (es_writer or self._es_writer)(event_dict)
        return event_dict

    def _emit_console(self, event_dict: dict[str, Any]) -> None:
        """输出到控制台 — 降噪版 (Rich 结构化日志)

        规则:
        - thought / tool_input / tool_output 永不打印到控制台
        - 生产环境 (env=production): 只打印 WARNING/ERROR
        - 开发环境: 通过 _type="agent" 路由到 RichConsoleRenderer 渲染
        """
        level = str(event_dict.get("level", "INFO")).upper()
        event_type = event_dict.get("event_type", "")

        # 生产环境：只输出 WARNING+
        if self._env == "production" and level not in ("WARNING", "ERROR"):
            return

        if self._env == "production":
            # 生产：简约错误行
            msg = event_dict.get("message", "") or event_type
            self._logger.warning("agent %s: %s", event_type, msg)
            return

        # 开发：通过 structlog + _type="agent" 路由到 RichConsoleRenderer
        self._logger.info(
            event_type,
            _type="agent",
            event_type=event_type,
            agent_name=event_dict.get("agent_name", ""),
            run_id=event_dict.get("run_id", ""),
            duration_ms=event_dict.get("duration_ms"),
            token_usage=event_dict.get("token_usage"),
            tool_name=event_dict.get("tool_name", ""),
            llm_model=event_dict.get("llm_model", ""),
            error_type=event_dict.get("error_type", ""),
            error_message=event_dict.get("error_message", ""),
            level=level,
        )


_emitter_singleton: AgentEventEmitter | None = None


def get_agent_event_emitter(env: str | None = None) -> AgentEventEmitter:
    global _emitter_singleton
    if _emitter_singleton is None:
        _emitter_singleton = AgentEventEmitter(env=env or os.getenv("LOG_ENV", "development"))
    return _emitter_singleton
