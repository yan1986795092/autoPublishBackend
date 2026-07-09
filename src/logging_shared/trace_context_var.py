"""trace 上下文 — Phase 0 引入，使用 Python contextvars。

问题：12 个 handler 内部都直接调用 await call_llm(...).
每个 handler 单独加 trace_id 参数会改 12 处。
解决方案：在 middleware / orchestrator / nodes 层把 trace_id 推入当前
contextvars；call_llm 默认从 contextvars 读取。业务侧仍可显式覆盖。

Phase 1 (Dispatcher 引入) 后这个模块会被 LogisticsRecorder 替代，接口保持兼容。
"""

from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TraceContextVar:
    """一次 LLM 调用的 span 元数据 — 由调用方在进入 handler 前 push。"""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    agent_name: str
    source: str = "auto"  # "auto" | "explicit"
    raw: bool = False


# contextvars: asyncio-safe
_current_trace: contextvars.ContextVar[TraceContextVar | None] = contextvars.ContextVar(
    "log_current_trace", default=None
)


def new_trace_id() -> str:
    """16 字节 (32 hex 字符)，与 OTel W3C 规范一致。"""
    return uuid.uuid4().hex[:32]


def new_span_id() -> str:
    return uuid.uuid4().hex[:16]


def push_trace(
    *,
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    agent_name: str,
    raw: bool = False,
) -> TraceContextVar:
    """在 handler 执行前 push trace 上下文；返回 token 用于 pop。"""
    ctx = TraceContextVar(
        trace_id=trace_id or new_trace_id(),
        span_id=span_id or new_span_id(),
        parent_span_id=parent_span_id,
        agent_name=agent_name,
        raw=raw,
    )
    _current_trace.set(ctx)
    return ctx


def pop_trace(token: TraceContextVar) -> None:
    _current_trace.set(None)


def get_current_trace() -> TraceContextVar | None:
    """call_llm 在 dispatch_prompt 前读取此值。

    返回 None 表示没有显式 push；call_llm 会自动生成 trace/span（向后兼容路径）。
    """
    return _current_trace.get()
