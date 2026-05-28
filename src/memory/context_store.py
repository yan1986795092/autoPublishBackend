"""Context Store — data models for layered multi-agent context management."""
from __future__ import annotations

import time as _time_module

from dataclasses import dataclass, field as _dc_field
from dataclasses import field  # re-export for non-conflicting use
from enum import Enum
from typing import Any, Dict, List, Optional, Set

# ──────────────────────────────────────────
# Project Stage
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
# Creative Anchors
# ──────────────────────────────────────────


@dataclass
class CreativeAnchors:
    """Creative elements that are NEVER compressed or discarded."""
    story_outline: str = ""
    character_profiles: Dict[str, str] = field(default_factory=dict)
    world_setting: str = ""
    style_guide: str = ""
    plot_milestones: List[str] = field(default_factory=list)


# ──────────────────────────────────────────
# Creative Decisions
# ──────────────────────────────────────────


@dataclass
class CreativeDecision:
    """A logged creative decision, preserved in creative_log (never compressed)."""
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
# Thinking Step (audit chain)
# ──────────────────────────────────────────


@dataclass
class ThinkingStep:
    """A single step in the thinking trace (audit chain)."""
    agent: str
    action: str  # "read" | "write" | "route" | "compress" | "log_creative_decision"
    step_field: str = ""  # renamed to avoid shadowing dataclasses.field
    detail: str = ""
    timestamp: float = _dc_field(default_factory=_time_module.time)


# ──────────────────────────────────────────
# Structured Memory Entry
# ──────────────────────────────────────────


@dataclass
class StructuredMemoryEntry:
    """A node in the structured global memory graph."""
    entry_type: str  # "entity" | "plot_event" | "project" | "style_template" | "creative_summary"
    entity_id: Optional[str] = None
    project_id: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    vector_summary: str = ""
    linked_content_refs: List[str] = field(default_factory=list)


# ──────────────────────────────────────────
# Anchor Drift
# ──────────────────────────────────────────


@dataclass
class AnchorDrift:
    """A detected inconsistency between creative anchors and actual content."""
    anchor_field: str
    drift_type: str  # "contradiction" | "implication_conflict" | "tone_mismatch"
    anchor_value: str
    content_evidence: str
    severity: str = "medium"  # "low" | "medium" | "high"


# ──────────────────────────────────────────
# Project Manifest
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
# Agent Private State
# ──────────────────────────────────────────


@dataclass
class AgentPrivateState:
    """Private state visible only to the owning agent."""
    internal_reasoning: str = ""
    confidence_score: float = 0.0
    local_draft_cache: str = ""


# ──────────────────────────────────────────
# Content / Review / SEO Blocks
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
# Shared Context Store
# ──────────────────────────────────────────


@dataclass
class SharedContextStore:
    # SystemLayer (all agents can read; only Consolidator writes)
    user_request: str = ""
    project_manifest: Optional[ProjectManifest] = None
    global_memory: List[StructuredMemoryEntry] = field(default_factory=list)
    thinking_trace: List[ThinkingStep] = field(default_factory=list)
    creative_log: List[CreativeDecision] = field(default_factory=list)

    # TaskLayer (permissioned read/write per agent) — always lists
    writing_content: List[ContentBlock] = field(default_factory=list)
    review_feedback: List[ReviewBlock] = field(default_factory=list)
    seo_suggestions: List[SEOBlock] = field(default_factory=list)
    creative_anchors: CreativeAnchors = field(default_factory=CreativeAnchors)

    # AgentLayer (private to each agent)
    agent_private: Dict[str, AgentPrivateState] = field(default_factory=dict)

    # Routing metadata
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

    # Convenience accessors — last item in each list (preserves backward compatibility)
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
# Agent Permissions
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
# Write Result
# ──────────────────────────────────────────


class WriteResult(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
