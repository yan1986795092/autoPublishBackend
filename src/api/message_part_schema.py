"""MessagePart Schema — 前后端共用的权威定义

参考 docs/superpowers/plans/2026-07-13-message-parts-schema-driven-rendering.md Task 1。

前端对应定义：frontend/src/types/message.ts（已存在，本文件与之一一对应）。
新增/修改 Part 类型时，必须同时更新两端。
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Literal, Union
import uuid


class PartState(str, Enum):
    STREAMING = "streaming"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class PartType(str, Enum):
    REASONING = "reasoning"
    TEXT = "text"
    TOOL_CALL = "tool_call"
    PLAN = "plan"
    ERROR = "error"
    ARTIFACT = "artifact"


class ArtifactType(str, Enum):
    BLOG = "blog"
    SOCIAL_POST = "social_post"
    COPYWRITING = "copywriting"
    CODE = "code"
    IMAGE = "image"
    TUTORIAL = "tutorial"


def new_part_id() -> str:
    """生成 Part ID（前端无 UUID，用短 hex 即可）。"""
    return f"part_{uuid.uuid4().hex[:12]}"


@dataclass
class ReasoningPart:
    id: str = field(default_factory=new_part_id)
    type: Literal["reasoning"] = "reasoning"
    state: str = PartState.STREAMING.value
    content: str = ""
    agent: str = ""
    round: int = 0
    collapsed_by_default: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TextPart:
    id: str = field(default_factory=new_part_id)
    type: Literal["text"] = "text"
    state: str = PartState.STREAMING.value
    content: str = ""
    agent: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolCallPart:
    id: str = field(default_factory=new_part_id)
    type: Literal["tool_call"] = "tool_call"
    state: str = PartState.STREAMING.value
    tool_name: str = ""
    display_name: str = ""
    icon: str = "⚙️"
    input: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    duration_ms: int = 0
    round: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlanTask:
    """计划中的一个子任务（spec §3.1 PlanPart.plan_data.tasks[]）。"""

    id: str = field(default_factory=new_part_id)
    agent: str = ""
    label: str = ""
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlanPart:
    id: str = field(default_factory=new_part_id)
    type: Literal["plan"] = "plan"
    state: str = PartState.DONE.value
    plan_data: dict = field(default_factory=dict)
    # 兼容旧前端 message.ts 的字段（前端 PlanPart 用 goal/reasoning/tasks）
    goal: str = ""
    reasoning: str = ""
    tasks: list[PlanTask] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ErrorPart:
    id: str = field(default_factory=new_part_id)
    type: Literal["error"] = "error"
    state: str = PartState.ERROR.value
    message: str = ""
    code: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArtifactPart:
    id: str = field(default_factory=new_part_id)
    type: Literal["artifact"] = "artifact"
    state: str = PartState.DONE.value
    artifact_type: str = ArtifactType.BLOG.value
    body: str = ""
    title: str = ""
    platform: str = ""
    structured: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# Discriminated union
MessagePart = Union[
    ReasoningPart, TextPart, ToolCallPart, PlanPart, ErrorPart, ArtifactPart
]


def from_dict(data: dict) -> MessagePart:
    """从 dict 构造正确的 Part 实例（type discriminator）。"""
    t = data.get("type")
    if t == "reasoning":
        return ReasoningPart(**{k: v for k, v in data.items() if k in ReasoningPart.__dataclass_fields__})
    if t == "text":
        return TextPart(**{k: v for k, v in data.items() if k in TextPart.__dataclass_fields__})
    if t == "tool_call":
        return ToolCallPart(**{k: v for k, v in data.items() if k in ToolCallPart.__dataclass_fields__})
    if t == "plan":
        # tasks 字段是 list[PlanTask]，需要嵌套转换
        kwargs = {k: v for k, v in data.items() if k in PlanPart.__dataclass_fields__ and k != "tasks"}
        if "tasks" in data and isinstance(data["tasks"], list):
            kwargs["tasks"] = [PlanTask(**t) if isinstance(t, dict) else t for t in data["tasks"]]
        return PlanPart(**kwargs)
    if t == "error":
        return ErrorPart(**{k: v for k, v in data.items() if k in ErrorPart.__dataclass_fields__})
    if t == "artifact":
        return ArtifactPart(**{k: v for k, v in data.items() if k in ArtifactPart.__dataclass_fields__})
    raise ValueError(f"Unknown part type: {t}")
