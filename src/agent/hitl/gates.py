"""HITL 门定义 — 4 种人机交互模式"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class DecisionGate:
    """决策门：人类在多个选项中做选择"""
    gate_id: str
    task_id: str
    question: str
    options: list[dict]              # [{"label": "是", "value": "yes"}, ...]
    context: dict = field(default_factory=dict)
    timeout: float = 300.0
    default_value: Optional[str] = None
    state: str = "pending"           # pending | resolved | timed_out
    result: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DecisionGate:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ApprovalGate:
    """审批门：人类审批后继续"""
    gate_id: str
    task_id: str
    target_agent: str
    content_summary: str
    full_content: str = ""
    instructions: str = ""           # 审核要点
    state: str = "pending"           # pending | approved | rejected
    feedback: str = ""               # 人类反馈

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gate_type"] = "approval"
        return d


@dataclass
class CollaborativeEditGate:
    """协作编辑门：人类直接修改 Agent 产出"""
    gate_id: str
    task_id: str
    target_agent: str
    draft_content: str
    edited_content: Optional[str] = None
    edit_instructions: str = ""
    iteration_count: int = 0
    max_iterations: int = 3
    state: str = "pending"           # pending | edited | skipped

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gate_type"] = "collaborative_edit"
        return d


@dataclass
class FeedbackGate:
    """反馈门：人类提供自由反馈，Agent 据此修改"""
    gate_id: str
    task_id: str
    target_agent: str
    content: str
    feedback: Optional[str] = None
    state: str = "pending"           # pending | received | acknowledged

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gate_type"] = "feedback"
        return d
