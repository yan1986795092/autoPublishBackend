"""工作记忆 — 当前会话的对话上下文和 token 追踪

包装已有的 ChatHistory，增加会话级别的上下文管理和 token 用量追踪。
"""

from __future__ import annotations
from typing import Optional

from .chat_history import ChatHistory


class WorkingMemory:
    """工作记忆：当前会话的短期上下文

    职责：
    - 跟踪当前对话轮次
    - 管理 token 预算（防超限）
    - 为 Agent 提供最近对话上下文
    """

    def __init__(self, max_messages: int = 20):
        self.chat_history = ChatHistory(max_messages=max_messages)
        self.current_task_id: Optional[str] = None
        self.token_usage: dict[str, int] = {}

    # ── 对话管理 ──

    def add_user_message(self, content: str) -> None:
        self.chat_history.add_user(content)

    def add_assistant_message(self, content: str) -> None:
        self.chat_history.add_assistant(content)

    def get_recent_context(self, max_chars: int = 2000) -> str:
        """获取最近对话上下文，用于注入 prompt"""
        return self.chat_history.to_prompt_context(max_chars=max_chars)

    def clear_session(self) -> None:
        self.chat_history.clear()
        self.token_usage.clear()

    # ── Token 追踪 ──

    def track_token(self, agent_type: str, tokens: int) -> None:
        self.token_usage[agent_type] = self.token_usage.get(agent_type, 0) + tokens

    def total_tokens(self) -> int:
        return sum(self.token_usage.values())

    def token_summary(self) -> str:
        parts = [f"{k}: {v}" for k, v in sorted(self.token_usage.items())]
        return f"会话 Token 用量：{' | '.join(parts)}" if parts else ""

    # ── 任务绑定 ──

    def bind_task(self, task_id: str) -> None:
        self.current_task_id = task_id

    @property
    def is_active(self) -> bool:
        return len(self.chat_history) > 0
