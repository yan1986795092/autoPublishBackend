"""反思模型定义"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReflectionResult:
    """反思结果"""
    agent_type: str                    # 被反思的 Agent
    target_task_id: str                # 被反思的任务 ID
    reflection_type: str               # "self" / "peer" / "meta"
    score: float                       # 0.0 - 1.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    passed: bool = True
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent_type": self.agent_type,
            "target_task_id": self.target_task_id,
            "reflection_type": self.reflection_type,
            "score": self.score,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "passed": self.passed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReflectionResult:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ReflectionConfig:
    """反思配置"""
    max_self_reflection_rounds: int = 2   # 自省最大轮次
    enable_peer_review: bool = False       # 启用同行评审
    enable_meta_cognition: bool = False    # 启用元认知
    peer_reviewers: list[str] = field(
        default_factory=lambda: ["ReviewAgent"],
    )
    quality_threshold: float = 0.7         # 质量阈值，低于此值需要修改
    self_reflection_agents: list[str] = field(
        default_factory=lambda: [
            "WritingAgent", "CodeAgent", "TranslationAgent",
            "SEOAgent", "DiagramAgent",
        ],
    )
