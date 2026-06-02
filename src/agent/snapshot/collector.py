"""SnapshotCollector — Agent 执行时的快照采集器 + ContextVar 上下文管理"""
from __future__ import annotations
import contextvars
import time
from datetime import datetime, timezone
from typing import Any, Optional

from .models import (
    ExecutionSnapshot, PromptMessageSnapshot, ContextSnapshot,
    LlmRequestSnapshot, LlmResponseSnapshot,
)

_collector_var: contextvars.ContextVar[Optional["SnapshotCollector"]] = \
    contextvars.ContextVar("snapshot_collector", default=None)


def set_collector(collector: Optional["SnapshotCollector"]) -> None:
    """设置当前 context 的 collector。"""
    _collector_var.set(collector)


def get_collector() -> Optional["SnapshotCollector"]:
    """获取当前 context 的 collector。"""
    return _collector_var.get()


def generate_snapshot_id() -> str:
    """生成唯一快照 ID。"""
    import hashlib, time as t
    raw = f"snap_{t.time_ns()}"
    return f"snap_{hashlib.md5(raw.encode()).hexdigest()[:12]}"


class SnapshotCollector:
    """Agent 执行快照采集器

    在 BaseAgent.execute() 中创建，通过 contextvars 注入当前执行链。
    PromptBuilder 和 call_llm 通过 get_collector() 自动注册数据。
    """

    def __init__(self, agent_name: str, profile: dict,
                 task_id: str | None = None,
                 parent_snapshot_id: str | None = None):
        self._snapshot_id = generate_snapshot_id()
        self._agent_name = agent_name
        self._profile = profile
        self._task_id = task_id
        self._parent_snapshot_id = parent_snapshot_id
        self._start = time.perf_counter()
        self.tags: list[str] = []

        self._prompt_messages: list[PromptMessageSnapshot] = []
        self._tool_calls: list[dict] = []
        self._memory_ids: list[str] = []
        self._llm_request: LlmRequestSnapshot | None = None
        self._llm_response: LlmResponseSnapshot | None = None

    def record_prompt_message(self, msg: PromptMessageSnapshot) -> None:
        """记录一条 prompt message。"""
        self._prompt_messages.append(msg)

    def record_tool_call(self, tool: str, input_summary: str,
                         output_summary: str) -> None:
        """记录一次工具调用。"""
        self._tool_calls.append({
            "tool": tool,
            "input_summary": input_summary,
            "output_summary": output_summary,
        })

    def record_memory_ids(self, ids: list[str]) -> None:
        """记录使用的记忆条目 ID。"""
        self._memory_ids.extend(ids)

    def record_llm_call(self, *, model: str, params: dict = None,
                        messages: list = None,
                        response_content: str = "",
                        finish_reason: str = "",
                        prompt_tokens: int = 0,
                        completion_tokens: int = 0,
                        total_tokens: int = 0,
                        latency_ms: int = 0) -> None:
        """记录 LLM 调用。"""
        self._llm_request = LlmRequestSnapshot(
            model=model,
            params=params or {},
            messages_count=len(messages) if messages else 0,
            total_prompt_chars=sum(len(str(m.get("content", ""))) for m in (messages or [])),
        )
        self._llm_response = LlmResponseSnapshot(
            content=response_content,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )

    def finalize(self, result: dict) -> ExecutionSnapshot:
        """执行完毕后生成最终快照。"""
        duration = int((time.perf_counter() - self._start) * 1000)
        snapshot = ExecutionSnapshot(
            snapshot_id=self._snapshot_id,
            agent_name=self._agent_name,
            timestamp=datetime.utcnow(),
            duration_ms=duration,
            task_id=self._task_id,
            parent_snapshot_id=self._parent_snapshot_id,
            prompt_messages=list(self._prompt_messages),
            context=ContextSnapshot(
                profile_snapshot=dict(self._profile) if self._profile else None,
                memory_ids=list(self._memory_ids),
                tool_calls=list(self._tool_calls),
            ),
            llm_request=self._llm_request,
            llm_response=self._llm_response,
            result_summary=str(result.get("status", "")),
            result=result,
            tags=list(self.tags),
        )
        # Compute hash via to_dict() which generates captured_hash
        snapshot.captured_hash = snapshot.to_dict().get("captured_hash", "")
        return snapshot
