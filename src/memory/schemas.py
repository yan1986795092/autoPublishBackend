"""记忆模块数据模型"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryItem:
    """一条记忆条目"""
    text: str
    namespace: str = "general"
    score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_prompt(self, max_chars: int = 500) -> str:
        content = self.text[:max_chars]
        source = self.metadata.get("agent_type", self.namespace)
        return f"[{source}] {content}"


@dataclass
class Episode:
    """一条历史经验记录"""
    agent_type: str
    input_summary: str
    output_summary: str
    success: bool = True
    tokens_used: int = 0
    total_steps: int = 0
    created_at: str = ""

    def to_prompt(self) -> str:
        status = "成功" if self.success else "失败"
        return f"[{self.agent_type}] {self.input_summary[:80]} → {status} ({self.output_summary[:100]})"
