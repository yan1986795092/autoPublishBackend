"""对话引擎数据模型"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConversationContext:
    """一次对话请求的上下文"""
    message: str
    conversation_id: str
    user_id: str
    history: list[dict] = field(default_factory=list)
    user_profile: str = ""


@dataclass
class GenerateTrigger:
    """LLM 判断可以生成时输出的触发指令"""
    action: str  # "create" | "revise" | "expand"
    topic: str = ""
    platform: str = ""
    style: str = ""
    length: str = ""
    tone: str = ""
    audience: str = ""
    keywords: list[str] = field(default_factory=list)
    revision_focus: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> GenerateTrigger:
        return cls(
            action=data.get("action", "create"),
            topic=data.get("topic", ""),
            platform=data.get("platform", ""),
            style=data.get("style", ""),
            length=data.get("length", ""),
            tone=data.get("tone", ""),
            audience=data.get("audience", ""),
            keywords=data.get("keywords", []),
            revision_focus=data.get("revision_focus", ""),
        )

    def to_prompt(self) -> str:
        """转换为传给 Agent Pipeline 的 prompt"""
        parts = [self.topic] if self.topic else []
        meta_parts = []
        if self.platform:
            meta_parts.append(f"[平台: {self.platform}]")
        if self.style:
            meta_parts.append(f"[风格: {self.style}]")
        if self.length:
            meta_parts.append(f"[篇幅: {self.length}]")
        if self.tone:
            meta_parts.append(f"[语调: {self.tone}]")
        if self.audience:
            meta_parts.append(f"[受众: {self.audience}]")
        if self.keywords:
            meta_parts.append(f"[关键词: {', '.join(self.keywords)}]")
        if self.revision_focus:
            meta_parts.append(f"[修改重点: {self.revision_focus}]")
        if meta_parts:
            parts.append("\n" + " ".join(meta_parts))
        return " ".join(parts) if parts else self.topic


@dataclass
class ConversationResponse:
    """对话引擎的一次回复"""
    text: str
    trigger: Optional[GenerateTrigger] = None
