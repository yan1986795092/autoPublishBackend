"""tool_event 适配器 — 把现有 AgentEvent 转 OTel 命名后落 ES。

设计上不直接复制 13 个 handler，每个 handler 都显式 emit 工具事件；
只在执行 plan / Agent 编排层做集中记录。事故复盘时这些数据足以回答：
- 某 Agent 调用了哪些工具
- 每次调用的入参/出参/耗时
- 哪个 step 失败了

**索引策略**：复用现有 `agent-events-*` 索引（UnifiedEsWriter 的 "events" 通道）。
按 `event_type`（= tool_call / tool_result / tool_error / handoff / ...）查询区分阶段。
理由：现有 backend / llm-gateway 已写 agent-events-*；新建 4 套独立索引意味着双倍
运维（4 套 ILM + 4 套 template + 4 套 alias）但量级未达。
"""

from __future__ import annotations

from datetime import datetime, timezone
from datetime import datetime, timezone
import time
import uuid
from typing import Any

from .trace_context_var import get_current_trace


def _ctx() -> tuple[str, str, str | None]:
    """从 contextvars 拿 trace/span/agent_name."""
    cur = get_current_trace()
    if cur is not None:
        return cur.trace_id, cur.span_id, cur.agent_name
    return "", "", None


async def dispatch_tool_event(
    *,
    event_type: str,           # "tool_call" | "tool_result" | "tool_error"
    tool_name: str,
    params: dict[str, Any] | None = None,
    output_summary: str | None = None,
    duration_ms: float | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    task_id: str | None = None,
) -> None:
    """把工具调用前后事件落 ES (复用 agent-events-* 索引).

    业务侧可显式 await；也可 fire-and-forget:
        from src.logging_shared.tool_event_adapter import emit_tool_event
        emit_tool_event(event_type='tool_call', tool_name='web_search', params={...})
    """
    import os
    if os.getenv("LOG_TOOL_EVENTS_ENABLED", "true").lower() in ("false", "0", "no"):
        return

    trace_id, span_id, agent_name = _ctx()

    doc: dict[str, Any] = {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "tool.name": tool_name,
        "agent.name": agent_name,
        "trace_id": trace_id,
        "span_id": span_id,
        "task_id": task_id,
        "duration_ms": duration_ms,
        "service": "agent",
    }

    # OTel 字段命名（与现有 agent-events-* schema 对齐）
    if params is not None:
        doc["tool.params"] = params
    if output_summary is not None:
        doc["tool.output_summary"] = output_summary
    if error_type is not None:
        doc["error_type"] = error_type
    if error_message is not None:
        doc["error_message"] = error_message[:1000]

    try:
        from .unified_writer import UnifiedEsWriter
        # 复用 "events" 通道 → 写入 agent-events-{date} 索引
        # 不再使用 __target_index__ 自定义路由
        await UnifiedEsWriter.get_instance().write("events", doc)
    except RuntimeError:
        pass  # 没有 running event loop
    except Exception:
        pass  # 不阻塞业务


def emit_tool_event(
    *,
    event_type: str,
    tool_name: str,
    params: dict[str, Any] | None = None,
    output_summary: str | None = None,
    duration_ms: float | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    task_id: str | None = None,
) -> None:
    """fire-and-forget 调用; 主用于非异步路径或已经拿到 result 的位置."""
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        loop.create_task(dispatch_tool_event(
            event_type=event_type,
            tool_name=tool_name,
            params=params,
            output_summary=output_summary,
            duration_ms=duration_ms,
            error_type=error_type,
            error_message=error_message,
            task_id=task_id,
        ))
    except RuntimeError:
        # 没在事件循环 — 创建一次性后台线程跑
        import threading
        threading.Thread(
            target=lambda: __import__("asyncio").run(dispatch_tool_event(
                event_type=event_type,
                tool_name=tool_name,
                params=params,
                output_summary=output_summary,
                duration_ms=duration_ms,
                error_type=error_type,
                error_message=error_message,
                task_id=task_id,
            )),
            daemon=True,
            name="tool-event-dispatch",
        ).start()


def tool_event(
    *,
    name: str | None = None,
    include_params: bool = True,
):
    """装饰器: 同步包装异步 handler, 自动记录 tool_call + tool_result.

    用法:
        @tool_event(name="web_search")
        async def my_search(query: str) -> dict:
            ...

        装饰后会自动:
        - 在 await 前 dispatch tool_call
        - 在 await 后 dispatch tool_result (含耗时/输出摘要)
        - 异常时 dispatch tool_error
    """
    import functools
    import inspect

    def decorator(fn):
        tool_name = name or fn.__name__

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                params: dict[str, Any] | None = None
                if include_params and args:
                    try:
                        # 仅取前几个简单参数, dict 类全量
                        if isinstance(args[0], dict):
                            params = {k: v for k, v in args[0].items() if isinstance(v, (str, int, float, bool))}
                        else:
                            params = {"args": str(args[:3])[:200]}
                    except Exception:
                        params = None
                emit_tool_event(
                    event_type="tool_call",
                    tool_name=tool_name,
                    params=params,
                )
                t0 = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                    output_summary = None
                    if isinstance(result, dict):
                        s = result.get("content") or result.get("output") or ""
                        if isinstance(s, str):
                            output_summary = s[:500]
                        elif s:
                            output_summary = str(s)[:500]
                    emit_tool_event(
                        event_type="tool_result",
                        tool_name=tool_name,
                        output_summary=output_summary,
                        duration_ms=duration_ms,
                    )
                    return result
                except Exception as exc:
                    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                    emit_tool_event(
                        event_type="tool_error",
                        tool_name=tool_name,
                        duration_ms=duration_ms,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    raise
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            emit_tool_event(event_type="tool_call", tool_name=tool_name)
            try:
                result = fn(*args, **kwargs)
                duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                emit_tool_event(event_type="tool_result", tool_name=tool_name, duration_ms=duration_ms)
                return result
            except Exception as exc:
                duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                emit_tool_event(
                    event_type="tool_error",
                    tool_name=tool_name,
                    duration_ms=duration_ms,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                raise
        return sync_wrapper

    return decorator
