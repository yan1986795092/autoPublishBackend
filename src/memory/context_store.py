"""Context Store —— 分层多智能体上下文管理的数据模型。"""
from __future__ import annotations

import time as _time_module

from dataclasses import dataclass, field as _dc_field
from dataclasses import field  # 重新导出，避免冲突使用
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# ──────────────────────────────────────────
# 项目阶段
# ──────────────────────────────────────────


class ProjectStage(str, Enum):
    CONCEPTUALIZATION = "conceptualization"
    OUTLINING = "outlining"
    DRAFTING = "drafting"
    REVISION = "revision"
    POLISHING = "polishing"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


VALID_TRANSITIONS: Dict[ProjectStage, Set[ProjectStage]] = {
    ProjectStage.CONCEPTUALIZATION: {ProjectStage.OUTLINING, ProjectStage.ABANDONED},
    ProjectStage.OUTLINING: {ProjectStage.DRAFTING, ProjectStage.CONCEPTUALIZATION, ProjectStage.ABANDONED},
    ProjectStage.DRAFTING: {ProjectStage.REVISION, ProjectStage.ABANDONED},
    ProjectStage.REVISION: {ProjectStage.POLISHING, ProjectStage.DRAFTING, ProjectStage.ABANDONED},
    ProjectStage.POLISHING: {ProjectStage.COMPLETED, ProjectStage.REVISION, ProjectStage.ABANDONED},
    ProjectStage.COMPLETED: set(),
    ProjectStage.ABANDONED: set(),
}


# ──────────────────────────────────────────
# 创意锚点
# ──────────────────────────────────────────


@dataclass
class CreativeAnchors:
    """永不压缩或丢弃的创意元素。"""
    story_outline: str = ""
    character_profiles: Dict[str, str] = field(default_factory=dict)
    world_setting: str = ""
    style_guide: str = ""
    plot_milestones: List[str] = field(default_factory=list)


# ──────────────────────────────────────────
# 创意决策
# ──────────────────────────────────────────


@dataclass
class CreativeDecision:
    """记录的创意决策，保存在 creative_log 中（永不压缩）。"""
    decision_text: str
    tags: List[str] = field(default_factory=list)
    turn_number: int = 0
    agent: str = ""
    timestamp: str = ""

    @classmethod
    def create(cls, decision_text: str, tags: List[str], agent: str, turn_number: int) -> "CreativeDecision":
        import datetime
        return cls(
            decision_text=decision_text,
            tags=tags,
            turn_number=turn_number,
            agent=agent,
            timestamp=datetime.datetime.now().isoformat(),
        )


# ──────────────────────────────────────────
# 思考步骤（审计链）
# ──────────────────────────────────────────


@dataclass
class ThinkingStep:
    """思考追踪中的单个步骤（审计链）。"""
    agent: str
    action: str  # "read" | "write" | "route" | "compress" | "log_creative_decision"
    step_field: str = ""  # 重命名以避免与 dataclasses.field 冲突
    detail: str = ""
    timestamp: float = _dc_field(default_factory=_time_module.time)


# ──────────────────────────────────────────
# 结构化记忆条目
# ──────────────────────────────────────────


@dataclass
class StructuredMemoryEntry:
    """结构化全局记忆图中的节点。"""
    entry_type: str  # "entity" | "plot_event" | "project" | "style_template" | "creative_summary"
    entity_id: Optional[str] = None
    project_id: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    vector_summary: str = ""
    linked_content_refs: List[str] = field(default_factory=list)


# ──────────────────────────────────────────
# 锚点偏移
# ──────────────────────────────────────────


@dataclass
class AnchorDrift:
    """检测到的创意锚点与实际内容之间的不一致。"""
    anchor_field: str
    drift_type: str  # "contradiction" | "implication_conflict" | "tone_mismatch"
    anchor_value: str
    content_evidence: str
    severity: str = "medium"  # "low" | "medium" | "high"


# ──────────────────────────────────────────
# 项目清单
# ──────────────────────────────────────────


@dataclass
class ProjectManifest:
    project_id: str
    title: str = ""
    project_type: str = "novel"  # "novel" | "short_story" | "article" | "script"
    stage: ProjectStage = ProjectStage.CONCEPTUALIZATION
    created_at: str = ""
    turn_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "title": self.title,
            "project_type": self.project_type,
            "stage": self.stage.value,
            "created_at": self.created_at,
            "turn_count": self.turn_count,
        }


# ──────────────────────────────────────────
# Agent 私有状态
# ──────────────────────────────────────────


@dataclass
class AgentPrivateState:
    """仅对所属 Agent 可见的私有状态。"""
    internal_reasoning: str = ""
    confidence_score: float = 0.0
    local_draft_cache: str = ""


# ──────────────────────────────────────────
# 内容 / 审阅 / SEO 块
# ──────────────────────────────────────────


@dataclass
class ContentBlock:
    text: str = ""
    turn_number: int = 0
    agent: str = ""


@dataclass
class ReviewBlock:
    text: str = ""
    approved: bool = False
    turn_number: int = 0


@dataclass
class SEOBlock:
    text: str = ""
    keywords: List[str] = field(default_factory=list)


# ──────────────────────────────────────────
# 共享上下文存储
# ──────────────────────────────────────────


@dataclass
class SharedContextStore:
    # 系统层（所有 Agent 可读；仅 Consolidator 写入）
    user_request: str = ""
    project_manifest: Optional[ProjectManifest] = None
    global_memory: List[StructuredMemoryEntry] = field(default_factory=list)
    thinking_trace: List[ThinkingStep] = field(default_factory=list)
    creative_log: List[CreativeDecision] = field(default_factory=list)

    # 任务层（每个 Agent 有权限控制的读写）—— 始终为列表
    writing_content: List[ContentBlock] = field(default_factory=list)
    review_feedback: List[ReviewBlock] = field(default_factory=list)
    seo_suggestions: List[SEOBlock] = field(default_factory=list)
    creative_anchors: CreativeAnchors = field(default_factory=CreativeAnchors)

    # Agent 层（每个 Agent 私有）
    agent_private: Dict[str, AgentPrivateState] = field(default_factory=dict)

    # 路由元数据
    session_id: str = ""
    conversation_id: str = ""
    similarity_score: float = 0.0
    is_new_session: bool = False

    def get_or_create_agent_private(self, agent_name: str) -> AgentPrivateState:
        if agent_name not in self.agent_private:
            self.agent_private[agent_name] = AgentPrivateState()
        return self.agent_private[agent_name]

    def add_thinking_step(self, agent: str, action: str, step_field: str = "", detail: str = "") -> None:
        self.thinking_trace.append(ThinkingStep(
            agent=agent, action=action, step_field=step_field, detail=detail,
        ))

    # 便捷访问器 —— 每个列表的最后一项（保持向后兼容）
    @property
    def last_writing_content(self) -> Optional[ContentBlock]:
        return self.writing_content[-1] if self.writing_content else None

    @property
    def last_review_feedback(self) -> Optional[ReviewBlock]:
        return self.review_feedback[-1] if self.review_feedback else None

    @property
    def last_seo_suggestions(self) -> Optional[SEOBlock]:
        return self.seo_suggestions[-1] if self.seo_suggestions else None


# ──────────────────────────────────────────
# Agent 权限
# ──────────────────────────────────────────


@dataclass
class AgentPermission:
    readable_fields: Set[str]
    writable_fields: Set[str]


AGENT_PERMISSIONS: Dict[str, AgentPermission] = {
    "WritingAgent": AgentPermission(
        readable_fields={
            "system", "task.writing_content", "task.review_feedback",
            "task.seo_suggestions", "task.creative_anchors", "system.creative_log",
        },
        writable_fields={
            "task.writing_content", "self.internal_reasoning", "system.creative_log",
        },
    ),
    "ReviewAgent": AgentPermission(
        readable_fields={
            "system", "task.writing_content", "task.creative_anchors",
            "system.creative_log", "task.review_feedback",
        },
        writable_fields={"task.review_feedback", "self.internal_reasoning"},
    ),
    "SEOAgent": AgentPermission(
        readable_fields={
            "system", "task.writing_content", "task.creative_anchors",
        },
        writable_fields={"task.seo_suggestions", "self.internal_reasoning"},
    ),
}


# ──────────────────────────────────────────
# 写入结果
# ──────────────────────────────────────────


class WriteResult(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
