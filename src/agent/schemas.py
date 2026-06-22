from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"


@dataclass
class AgentTask:
    id: str
    agent_type: str
    input: dict[str, Any]
    context: dict[str, Any] = field(default_factory=dict)
    parent_task_id: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 2
    can_parallel_with: list[str] = field(default_factory=list)
    retry_on_low_quality: bool = False
    priority: str = "medium"


@dataclass
class UsageInfo:
    tokens_used: int = 0
    estimated_cost: float = 0.0
    time_elapsed: float = 0.0
    model: str = ""


@dataclass
class AgentResult:
    task_id: str
    status: TaskStatus
    agent_type: str = ""
    output: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    usage: UsageInfo = field(default_factory=UsageInfo)
    agent_events: Optional[dict] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, TaskStatus) else self.status
        return d


# ═══════════════════════════════════════════
# 输入/输出 Schema 合约
# ═══════════════════════════════════════════


class AgentSchema:
    """Agent 类型契约"""

    def __init__(self, name: str, input_fields: dict[str, tuple], output_fields: dict[str, tuple]):
        self.name = name
        # {field_name: (type, required, description)}
        self.input_fields = input_fields
        self.output_fields = output_fields

    def validate_input(self, data: dict) -> list[str]:
        missing = []
        for field, (_, required, _) in self.input_fields.items():
            if required and (field not in data or data.get(field) in (None, "", "null")):
                missing.append(field)
        return missing

    def validate_output(self, data: dict) -> list[str]:
        missing = []
        for field, (_, required, _) in self.output_fields.items():
            if required and (field not in data or data.get(field) in (None, "", "null")):
                missing.append(field)
        return missing


WRITING_SCHEMA = AgentSchema("WritingAgent", {
    "topic":        (str, True,  "写作主题"),
    "content_type": (str, False, "内容类型（blog/script/xhs_post 等）"),
    "platform":     (str, False, "目标平台代码"),
    "style":        (str, False, "写作风格描述"),
    "tone":         (str, False, "语调"),
    "audience":     (str, False, "目标读者"),
    "keywords":     (list, False, "关键词列表"),
    "length":       (str, False, "篇幅"),
}, {
    "content":      (str, True,  "正文内容"),
    "type":         (str, True,  "内容类型"),
})

REVIEW_SCHEMA = AgentSchema("ReviewAgent", {
    "content":      (str, True,  "待审核内容"),
    "content_type": (str, False, "内容类型"),
    "platform":     (str, False, "目标平台"),
}, {
    "content":      (str, True,  "审核反馈"),
    "passed":       (bool, True, "是否通过"),
})

FIX_SCHEMA = AgentSchema("FixAgent", {
    "content":      (str, True,  "待修改内容"),
    "feedback":     (str, False, "修改意见"),
}, {
    "content":      (str, True,  "修改后内容"),
    "type":         (str, True,  "内容类型"),
})

AGENT_SCHEMAS: dict[str, AgentSchema] = {
    "WritingAgent": WRITING_SCHEMA,
    "ReviewAgent": REVIEW_SCHEMA,
    "FixAgent": FIX_SCHEMA,
}


def get_schema(agent_type: str) -> AgentSchema | None:
    return AGENT_SCHEMAS.get(agent_type)


@dataclass
class ExecutionPlan:
    """Manager Agent 制定的执行计划"""
    goal: str
    tasks: list[AgentTask]
    reasoning: str = ""
    estimated_steps: int = 0
    parallel_groups: list[list[str]] = field(default_factory=list)
    plan_method: str = "llm"
