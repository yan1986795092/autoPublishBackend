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
# 执行计划
# ═══════════════════════════════════════════


@dataclass
class ExecutionPlan:
    """Manager Agent 制定的执行计划"""
    goal: str
    tasks: list[AgentTask]
    reasoning: str = ""
    estimated_steps: int = 0
    parallel_groups: list[list[str]] = field(default_factory=list)
    plan_method: str = "llm"
