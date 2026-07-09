"""GenAI 子系统数据类 — 事故复盘的最后一公里。

3 个 dataclass:
- PromptRecord: 一次 LLM 调用的完整输入 (含 system prompt 与 messages 列表)
- CompletionRecord: 一次 LLM 调用的完整输出
- HandoffRecord: Agent 之间的交接事件 (含 state snapshot)

这些数据写入现有 `agent-events-*` 索引(与 tool_event 共用同一通道),
按 `event_type` 字段区分: `llm_prompt` / `llm_completion` / `handoff`。
原计划分配独立索引 (gen-ai-prompts / gen-ai-completions / agent-handoffs) 已被取消 ——
复用现有 agent-events-* 避免双倍运维 (ILM / template / alias 三套)。

字段命名对齐 OpenTelemetry GenAI Semantic Conventions 1.30+。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def _new_span_id() -> str:
    """8 字节 (16 hex 字符) span_id，与 OTel 规范一致。"""
    return uuid.uuid4().hex[:16]


def _new_trace_id() -> str:
    """16 字节 (32 hex 字符) trace_id，与 OTel 规范一致。"""
    return uuid.uuid4().hex[:32]


@dataclass(frozen=True)
class PromptRecord:
    """一次 LLM 调用的完整输入 — 写入 agent-events-* (event_type=llm_prompt)。

    默认 PII 脱敏（由 Dispatcher 层 redactor 处理），业务侧需 opt-in 明文时传 raw=True。
    字段说明：
        trace_id, span_id, parent_span_id: W3C Trace Context (32/16/16 hex)
        agent_name: 发起本次 LLM 调用的 Agent 名（用于在写作/审核/修复链路中定位）
        model: 模型标识
        messages: 序列化后的 LangChain messages → dict 列表
        tool_definitions: LLM 当时可用的 function schema 列表（关键 — 复盘时知道 LLM 看到了什么工具）
        raw: True = 明文落盘，False（默认）= 走 PII 脱敏
    """

    agent_name: str
    model: str
    messages: list[dict[str, Any]]
    trace_id: str = field(default_factory=_new_trace_id)
    span_id: str = field(default_factory=_new_span_id)
    parent_span_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    tool_definitions: list[dict] | None = None
    prompt_tokens: int | None = None
    temperature: float | None = None
    raw: bool = False  # True = 跳过脱敏
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionRecord:
    """一次 LLM 调用的完整输出 — 写入 agent-events-* (event_type=llm_completion)."""

    agent_name: str
    model: str
    content: str
    finish_reason: str | None = None
    trace_id: str = field(default_factory=_new_trace_id)
    span_id: str = field(default_factory=_new_span_id)
    parent_span_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    tool_calls: list[dict[str, Any]] | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    raw: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HandoffRecord:
    """Agent 间交接事件 — 写入 agent-events-* (event_type=handoff)。

    事故复盘时这是关键证据：
        - "为什么 ReviewAgent 决定不通过"
        - "FixAgent 收到了什么上下文"
        - "本次交接时整个 plan 的 cost 已经累计多少"
    """

    from_agent: str
    to_agent: str
    handoff_reason: str
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=_new_trace_id)
    span_id: str = field(default_factory=_new_span_id)
    parent_span_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    task_id: str | None = None
    cost_so_far_usd: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# 索引策略说明: 全部事件统一写入 agent-events-* 索引
# - llm_prompt / llm_completion / handoff 由 genai_dispatcher.py 经 "events" 通道写入
# - tool_call / tool_result / tool_error 由 tool_event_adapter.py 经 "events" 通道写入
# 查询时按 event_type 字段过滤区分阶段。
