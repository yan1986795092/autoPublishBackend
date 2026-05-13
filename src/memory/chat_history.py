"""对话历史管理 — 滑动窗口 + 摘要压缩"""

from __future__ import annotations
from collections import deque


class ChatHistory:
    """简单的滑动窗口对话记忆，超出 max_messages 时自动压缩旧消息"""

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._messages: deque[dict] = deque(maxlen=max_messages)

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def add_user(self, content: str) -> None:
        self.add("user", content)

    def add_assistant(self, content: str) -> None:
        self.add("assistant", content)

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def get_last_n(self, n: int) -> list[dict]:
        return list(self._messages)[-n:]

    def clear(self) -> None:
        self._messages.clear()

    def to_prompt_context(self, max_chars: int = 4000) -> str:
        """将对话历史压缩为 prompt 上下文字符串"""
        lines = []
        total = 0
        for msg in reversed(self._messages):
            line = f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)
        return "\n".join(reversed(lines))

    def __len__(self) -> int:
        return len(self._messages)
