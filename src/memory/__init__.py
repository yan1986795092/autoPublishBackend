"""记忆模块 — Multi-Agent 系统的三层记忆

提供统一的记忆管理入口，包含：
- WorkingMemory:   工作记忆（当前会话上下文）
- SemanticMemory:  语义记忆（向量化长期知识）
- EpisodicMemory:  情景记忆（历史任务经验）
- MemoryManager:   统一入口，集成到 Agent 编排流程

使用方式：
    from src.memory import memory_manager

    # Agent 执行前注入记忆
    ctx = await memory_manager.build_agent_context(agent_type, task_input, user_id)

    # Agent 执行后沉淀记忆
    await memory_manager.remember(agent_type, task_input, result, user_id)
"""

from .manager import MemoryManager, memory_manager
from .working_memory import WorkingMemory
from .semantic_memory import SemanticMemory
from .episodic_memory import EpisodicMemory
from .schemas import MemoryItem, Episode

# Context management (multi-agent layered system)
from .context_store import (
    SharedContextStore,
    CreativeAnchors,
    CreativeDecision,
    StructuredMemoryEntry,
    ProjectManifest,
    ProjectStage,
    AnchorDrift,
    ThinkingStep,
    ContentBlock,
    ReviewBlock,
    SEOBlock,
    AgentPrivateState,
    AGENT_PERMISSIONS,
    VALID_TRANSITIONS,
    WriteResult,
)
from .context_manager import ContextManager
from .agent_scoped_view import (
    build_agent_context,
    write_to_context,
    log_creative_decision,
    get_context_for_writing_agent,
    filter_task_fields,
    log_review_anchor_drift,
    extract_anchors_from_content,
    update_creative_anchors,
    check_anchor_drift_from_review,
)
from .consolidator import (
    ConsolidatorNode,
    AnchorConsistencyChecker,
    ConsolidationResult,
)
from .structured_memory import (
    EntityMemory,
    PlotMemory,
    ProjectIndex,
    CreativeSummaryMemory,
)

__all__ = [
    # Existing
    "MemoryManager",
    "memory_manager",
    "WorkingMemory",
    "SemanticMemory",
    "EpisodicMemory",
    "MemoryItem",
    "Episode",
    # Context store
    "SharedContextStore",
    "CreativeAnchors",
    "CreativeDecision",
    "StructuredMemoryEntry",
    "ProjectManifest",
    "ProjectStage",
    "AnchorDrift",
    "ThinkingStep",
    "ContentBlock",
    "ReviewBlock",
    "SEOBlock",
    "AgentPrivateState",
    "AGENT_PERMISSIONS",
    "VALID_TRANSITIONS",
    "WriteResult",
    # Context management
    "ContextManager",
    "build_agent_context",
    "write_to_context",
    "log_creative_decision",
    "get_context_for_writing_agent",
    "filter_task_fields",
    "log_review_anchor_drift",
    "extract_anchors_from_content",
    "update_creative_anchors",
    "check_anchor_drift_from_review",
    # Consolidator
    "ConsolidatorNode",
    "AnchorConsistencyChecker",
    "ConsolidationResult",
    # Structured memory
    "EntityMemory",
    "PlotMemory",
    "ProjectIndex",
    "CreativeSummaryMemory",
]
