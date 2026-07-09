"""顶层 dispatch 入口 — 统一写入 agent-events-* 索引。

设计原则: 不再使用独立索引(gen-ai-prompts / gen-ai-completions / agent-handoffs),
而是写入现有 `agent-events-{date}`, 通过 `event_type` 字段区分:

    llm_prompt      →  LLM 输入(redacted)
    llm_completion  →  LLM 输出(redacted)
    handoff         →  Agent 交接

理由 (见 docs/superpowers/specs/2026-06-08-observability-redesign.md):
1. backend / llm-gateway 已经按 agent-events-* 写; 重复新建索引意味着双倍 ILM/template/alias 维护.
2. 同 trace_id 下, 用 event_type 过滤即可分阶段; 查询性能等价.
3. 字段冲突风险低: 现有 agent-events-* schema 是 dynamic:false, 新增字段显式声明.

调用方只通过这里发往 GenAI 三个 sink; 接口兼容旧的 emit_*/dispatch_* 调用.
"""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime, timezone
from typing import Any

from .genai_records import (
    PromptRecord,
    CompletionRecord,
    HandoffRecord,
)
from .redactor import DefaultRedactor, Redactor, NoopRedactor


# 全局单例 redactor — 业务侧一般不需要换
_global_redactor: Redactor = DefaultRedactor()
_env_flag_lock = threading.Lock()


def get_redactor() -> Redactor:
    return _global_redactor


def set_redactor(redactor: Redactor) -> None:
    """注入自定义 redactor。"""
    global _global_redactor
    _global_redactor = redactor


def is_genai_sinks_enabled() -> bool:
    """回滚开关 — LOG_GENAI_SINK_ENABLED=false 时停写 GenAI 索引。"""
    return os.getenv("LOG_GENAI_SINK_ENABLED", "true").lower() not in ("false", "0", "no")


def is_handoff_sinks_enabled() -> bool:
    return os.getenv("LOG_HANDOFF_SINK_ENABLED", "true").lower() not in ("false", "0", "no")


async def _safe_write(event_type: str, doc: dict[str, Any]) -> None:
    """写入 agent-events-* 索引 (与 tool_event_adapter 共用同一通道).

    不再使用 __target_index__ 旁路到独立索引. 通过 event_type 区分阶段:
      llm_prompt / llm_completion / handoff / tool_call / tool_result / tool_error
    """
    try:
        from .unified_writer import UnifiedEsWriter
        writer = UnifiedEsWriter.get_instance()
        # 走 "events" 通道 → agent-events-{date}
        enriched = dict(doc)
        enriched["event_type"] = event_type
        await writer.write("events", enriched)
    except RuntimeError:
        # 没有运行中的事件循环（schedule_and_execute 内部 thread-pool 调用等），跳过
        pass
    except Exception:
        # sink 抛异常不应当阻塞业务（规范 P6: 不静默 → 后续可接 metrics）
        pass


async def dispatch_prompt(record: PromptRecord) -> None:
    """把 PromptRecord 写到 agent-events-* (event_type=llm_prompt).

    默认 PII 脱敏；若 record.raw=True 则跳过。
    若 LOG_GENAI_SINK_ENABLED=false，直接返回（不报错）。
    """
    if not is_genai_sinks_enabled():
        return
    redacted = get_redactor().redact_prompt(record)
    doc = _prompt_to_doc(redacted)
    await _safe_write("llm_prompt", doc)


async def dispatch_completion(record: CompletionRecord) -> None:
    """把 CompletionRecord 写到 agent-events-* (event_type=llm_completion)."""
    if not is_genai_sinks_enabled():
        return
    redacted = get_redactor().redact_completion(record)
    doc = _completion_to_doc(redacted)
    await _safe_write("llm_completion", doc)


async def dispatch_handoff(record: HandoffRecord) -> None:
    """把 HandoffRecord 写到 agent-events-* (event_type=handoff)."""
    if not is_handoff_sinks_enabled():
        return
    redacted = get_redactor().redact_handoff(record)
    doc = _handoff_to_doc(redacted)
    await _safe_write("handoff", doc)


# ── fire-and-forget 同步封装 ─────────────────────────────────────────
# 业务侧 handler 在同步路径里也能调，自动调度到事件循环


def emit_prompt(record: PromptRecord) -> None:
    """同步 fire-and-forget — 用于非异步路径"""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(dispatch_prompt(record))
    except RuntimeError:
        # 没在事件循环中 → 用后台线程跑（最差情况：延迟落库，但不影响业务返回）
        threading.Thread(
            target=lambda: asyncio.run(dispatch_prompt(record)),
            daemon=True,
            name="genai-prompt-dispatch",
        ).start()


def emit_completion(record: CompletionRecord) -> None:
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(dispatch_completion(record))
    except RuntimeError:
        threading.Thread(
            target=lambda: asyncio.run(dispatch_completion(record)),
            daemon=True,
            name="genai-completion-dispatch",
        ).start()


def emit_handoff(record: HandoffRecord) -> None:
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(dispatch_handoff(record))
    except RuntimeError:
        threading.Thread(
            target=lambda: asyncio.run(dispatch_handoff(record)),
            daemon=True,
            name="handoff-dispatch",
        ).start()


# ── 文档化辅助函数 ─────────────────────────────────────────────────


def _msg_to_dict(msg: Any) -> dict[str, Any]:
    """LangChain BaseMessage → dict (兼容 OpenAI / Anthropic 格式)."""
    if isinstance(msg, dict):
        return msg
    d: dict[str, Any] = {}
    if hasattr(msg, "type"):
        d["role"] = msg.type
    elif hasattr(msg, "role"):
        d["role"] = msg.role
    content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        d["content"] = content
    elif isinstance(content, list):
        d["content"] = content
    else:
        d["content"] = str(content)
    # 工具调用块
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = list(tool_calls)
    # tool result 的 tool_call_id
    tcid = getattr(msg, "tool_call_id", None)
    if tcid:
        d["tool_call_id"] = tcid
    name = getattr(msg, "name", None)
    if name:
        d["name"] = name
    return d


def _ts(ts: float) -> str:
    """float timestamp → ES 期望的 ISO 字符串（兜底所有 record.timestamp 传 float 的情况）。"""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _prompt_to_doc(record: PromptRecord) -> dict[str, Any]:
    return {
        "@timestamp": _ts(record.timestamp),
        "trace_id": record.trace_id,
        "span_id": record.span_id,
        "parent_span_id": record.parent_span_id,
        "agent_name": record.agent_name,
        "gen_ai.model": record.model,
        "messages": [
            {
                "role": m.get("role", m.get("type", "?")),
                "content": m.get("content", ""),
                **({"tool_call_id": m["tool_call_id"]} if "tool_call_id" in m else {}),
                **({"tool_calls": m["tool_calls"]} if "tool_calls" in m else {}),
            }
            for m in record.messages
        ],
        "tool_definitions": record.tool_definitions,
        "gen_ai.usage.input_tokens": record.prompt_tokens,
        "temperature": record.temperature,
        "redacted": (not record.raw),
        "extra": record.extra,
        "service": "auto-publish",
    }


def _completion_to_doc(record: CompletionRecord) -> dict[str, Any]:
    return {
        "@timestamp": _ts(record.timestamp),
        "trace_id": record.trace_id,
        "span_id": record.span_id,
        "parent_span_id": record.parent_span_id,
        "agent_name": record.agent_name,
        "gen_ai.model": record.model,
        "gen_ai.response.finish_reason": record.finish_reason,
        "content": record.content,
        "tool_calls": record.tool_calls,
        "gen_ai.usage.output_tokens": record.completion_tokens,
        "gen_ai.usage.cost_usd": record.cost_usd,
        "redacted": (not record.raw),
        "extra": record.extra,
        "service": "auto-publish",
    }


def _handoff_to_doc(record: HandoffRecord) -> dict[str, Any]:
    return {
        "@timestamp": _ts(record.timestamp),
        "trace_id": record.trace_id,
        "span_id": record.span_id,
        "parent_span_id": record.parent_span_id,
        "from_agent": record.from_agent,
        "to_agent": record.to_agent,
        "handoff_reason": record.handoff_reason,
        "task_id": record.task_id,
        "state_snapshot": record.state_snapshot,
        "cost_so_far_usd": record.cost_so_far_usd,
        "redacted": (not record.extra.get("_raw", False)),
        "extra": record.extra,
        "service": "auto-publish",
    }


# 暴露给 tool_utils 的标准化函数
serialize_messages = _msg_to_dict
def _ts_to_iso(ts: float) -> str:
    """将 float 时间戳转为 ISO 8601 字符串（ES @timestamp 标准格式）"""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()



