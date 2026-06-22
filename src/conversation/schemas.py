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
        return PipelineRequest.from_trigger(self).to_prompt_text()


@dataclass
class PipelineRequest:
    """对话 → Agent Pipeline 的显式契约（替代裸 _prompt 字符串）

    字段直接对应 Agent 任务 input，避免数据在传输过程中被丢字段。
    """
    action: str
    topic: str
    platform: str = ""
    style: str = ""
    length: str = ""
    tone: str = ""
    audience: str = ""
    keywords: list[str] = field(default_factory=list)
    revision_focus: str = ""

    @classmethod
    def from_trigger(cls, trigger: "GenerateTrigger") -> "PipelineRequest":
        return cls(
            action=trigger.action,
            topic=trigger.topic,
            platform=trigger.platform,
            style=trigger.style,
            length=trigger.length,
            tone=trigger.tone,
            audience=trigger.audience,
            keywords=list(trigger.keywords),
            revision_focus=trigger.revision_focus,
        )

    def to_prompt_text(self) -> str:
        """生成给 Agent 的 prompt 文本（topic 在前，元信息在后）"""
        parts = []
        if self.topic:
            parts.append(self.topic.strip())
        meta = []
        if self.platform:    meta.append(f"[平台: {self.platform}]")
        if self.style:       meta.append(f"[风格: {self.style}]")
        if self.length:      meta.append(f"[篇幅: {self.length}]")
        if self.tone:        meta.append(f"[语调: {self.tone}]")
        if self.audience:    meta.append(f"[受众: {self.audience}]")
        if self.keywords:    meta.append(f"[关键词: {', '.join(self.keywords)}]")
        if self.revision_focus:
            meta.append(f"[修改重点: {self.revision_focus}]")
        if meta:
            parts.append(" ".join(meta))
        return "\n".join(parts)


@dataclass
class ConversationResponse:
    """对话引擎的一次回复"""
    text: str
    trigger: Optional[GenerateTrigger] = None
