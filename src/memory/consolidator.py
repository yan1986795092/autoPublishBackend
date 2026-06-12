"""Consolidator —— 会话终结、锚点检查、记忆写入。"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .context_store import (
    SharedContextStore,
    CreativeAnchors,
    ContentBlock,
    StructuredMemoryEntry,
    AnchorDrift,
    ProjectStage,
)
from .structured_memory import (
    EntityMemory,
    PlotMemory,
    ProjectIndex,
    CreativeSummaryMemory,
)

logger = logging.getLogger(__name__)

# 阶段 1 矛盾检测使用的特质反义词对
TRAIT_ANTONYMS: Dict[str, str] = {
    "introverted": "extroverted",
    "quiet": "loud",
    "brave": "cowardly",
    "kind": "cruel",
    "honest": "dishonest",
    "loyal": "disloyal",
    "optimistic": "pessimistic",
    "careful": "reckless",
    "generous": "selfish",
    "patient": "impatient",
}


# ──────────────────────────────────────────
# 锚点一致性检查器
# ──────────────────────────────────────────


class AnchorConsistencyChecker:
    """
    检测内容是否偏离已建立的创意锚点。

    阶段 1（低成本）：使用正则表达式 + 关键词匹配显式矛盾。
    阶段 2（可选）：使用 LLM 进行蕴含分析，检测细微冲突。
    """

    def __init__(self):
        self._antonyms = TRAIT_ANTONYMS

    def check(self, anchors: CreativeAnchors, content: str) -> List[AnchorDrift]:
        """阶段 1：检测显式的特质矛盾。"""
        if not content or not anchors.character_profiles:
            return []

        drifts = []
        content_lower = content.lower()

        for char_name, profile in anchors.character_profiles.items():
            profile_lower = profile.lower()
            for trait, antonym in self._antonyms.items():
                if trait in profile_lower:
                    pattern = rf"\b{re.escape(antonym)}"
                    if re.search(pattern, content_lower):
                        drifts.append(AnchorDrift(
                            anchor_field=f"character_profiles.{char_name}",
                            drift_type="contradiction",
                            anchor_value=f"trait: {trait}",
                            content_evidence=f"在 {char_name} 附近发现 '{antonym}'",
                            severity="medium",
                        ))

        return drifts


@dataclass
class ConsolidationResult:
    status: str  # "completed" | "failed" | "archived"
    new_memory_entries: List[StructuredMemoryEntry] = field(default_factory=list)
    anchor_drifts: List[AnchorDrift] = field(default_factory=list)
    project_summary: str = ""


# ──────────────────────────────────────────
# Consolidator 节点
# ──────────────────────────────────────────


class ConsolidatorNode:
    """
    在所有 Agent 完成后运行的 LangGraph 节点。

    职责：
    1. 锚点一致性检查（如果内容存在）
    2. 如果审阅批准了更改，同步锚点
    3. 从内容中提取新实体 → 写入 global_memory + EntityMemory 后端
    4. 将 creative_log 决策持久化 → 写入 PlotMemory + CreativeSummaryMemory 后端
    5. 生成项目进度摘要
    6. Token 预算压缩检查
    7. 标记会话状态
    """

    def __init__(
        self,
        compress_threshold_pct: float = 0.80,
        entity_memory: Optional[EntityMemory] = None,
        plot_memory: Optional[PlotMemory] = None,
        project_index: Optional[ProjectIndex] = None,
        creative_summary_memory: Optional[CreativeSummaryMemory] = None,
    ):
        self.compress_threshold_pct = compress_threshold_pct
        self._checker = AnchorConsistencyChecker()
        # 结构化记忆后端（如果未提供则延迟初始化）
        self._entity_memory = entity_memory or EntityMemory()
        self._plot_memory = plot_memory or PlotMemory()
        self._project_index = project_index or ProjectIndex()
        self._creative_summary_memory = creative_summary_memory or CreativeSummaryMemory()

    def finalize(self, store: SharedContextStore) -> ConsolidationResult:
        """终结会话：检查锚点、写入记忆、管理预算。"""
        stage_name = store.project_manifest.stage.value if store.project_manifest else "none"
        store.add_thinking_step(
            agent="ConsolidatorNode",
            action="finalize_start",
            detail=f"stage={stage_name}",
        )

        new_entries: List[StructuredMemoryEntry] = []
        drifts: List[AnchorDrift] = []
        project_summary = ""

        # 1. 锚点一致性检查（最后一个内容块）
        last_content = store.last_writing_content
        if last_content and last_content.text:
            drifts = self._checker.check(store.creative_anchors, last_content.text)
            if drifts:
                logger.info(
                    f"ConsolidatorNode: 检测到 {len(drifts)} 个锚点偏移，"
                    f"提示用户确认"
                )

        # 2. 从最后一个内容块提取实体 → 写入 global_memory + 结构化后端
        if last_content and last_content.text:
            entities = self._extract_entities(
                last_content,
                store.project_manifest.project_id if store.project_manifest else None,
            )
            new_entries.extend(entities)
            store.global_memory.extend(entities)
            # 同时持久化到 EntityMemory 结构化后端
            for entry in entities:
                self._entity_memory.store(entry)

        # 3. 将创意日志决策写入 PlotMemory
        if store.creative_log and store.project_manifest:
            for decision in store.creative_log[-5:]:
                plot_entry = StructuredMemoryEntry(
                    entry_type="plot_event",
                    entity_id=f"decision-{decision.turn_number}",
                    project_id=store.project_manifest.project_id,
                    attributes={
                        "decision": decision.decision_text[:200],
                        "agent": decision.agent,
                        "tags": ",".join(decision.tags),
                    },
                    vector_summary=f"创意决策: {decision.decision_text[:100]}",
                )
                self._plot_memory.store(plot_entry)
                store.global_memory.append(plot_entry)
                new_entries.append(plot_entry)

        # 4. 使用当前阶段更新 ProjectIndex
        if store.project_manifest:
            pid = store.project_manifest.project_id
            self._project_index.register(
                project_id=pid,
                title=store.project_manifest.title,
                project_type=store.project_manifest.project_type,
            )
            self._project_index.update_stage(pid, store.project_manifest.stage)

        # 5. 生成并持久化创意摘要
        if store.project_manifest:
            pid = store.project_manifest.project_id
            self._creative_summary_memory.store(
                project_id=pid,
                intent=store.user_request[:200],
                output_summary=project_summary,
                style_template=store.creative_anchors.style_guide[:200] if store.creative_anchors.style_guide else "",
            )

        # 3. 更新项目摘要
        project_summary = self._generate_summary(store)

        # 6. Token 预算压缩检查
        tokens = self._estimate_tokens(store)
        budget = 32000
        if tokens / budget >= self.compress_threshold_pct:
            self._compress(store)

        # 7. 确定状态
        if last_content and last_content.text:
            status = "completed"
        else:
            status = "failed"

        store.add_thinking_step(
            agent="ConsolidatorNode",
            action="finalize_end",
            detail=f"status={status}, entities={len(new_entries)}, drifts={len(drifts)}",
        )

        return ConsolidationResult(
            status=status,
            new_memory_entries=new_entries,
            anchor_drifts=drifts,
            project_summary=project_summary,
        )

    def _extract_entities(
        self,
        content_block: ContentBlock,
        project_id: Optional[str],
    ) -> List[StructuredMemoryEntry]:
        """阶段 1：通过简单的模式匹配（大写名称）提取实体。"""
        entries = []
        text = content_block.text
        project_id = project_id or "unknown"

        names = re.findall(r"\b([A-Z][a-z]{1,20})\b", text)
        seen = set()
        skip = {"the", "and", "but", "then", "chapter", "scene", "part", "once", "upon",
                "there", "here", "after", "before", "while", "when", "where", "would",
                "could", "should", "their", "their", "this", "that", "with", "from",
                "said", "asked", "replied", "looked", "turned", "went", "came", "saw",
                "know", "think", "will", "shall", "have", "been", "being", "were"}
        for name in names:
            if name.lower() in skip:
                continue
            if name in seen:
                continue
            seen.add(name)
            entries.append(StructuredMemoryEntry(
                entry_type="entity",
                entity_id=f"{project_id}:{name.lower().replace(' ', '_')}",
                project_id=project_id,
                attributes={"name": name, "detected_in": "content"},
                vector_summary=f"角色: {name}",
            ))

        return entries

    def _generate_summary(self, store: SharedContextStore) -> str:
        """从当前状态生成简要的项目摘要。"""
        if not store.project_manifest:
            return ""

        manifest = store.project_manifest
        parts = [
            f"项目: {manifest.title or manifest.project_id}",
            f"阶段: {manifest.stage.value}",
            f"轮次: {manifest.turn_count}",
        ]
        if store.creative_anchors.story_outline:
            parts.append(f"大纲: {store.creative_anchors.story_outline[:100]}")
        if store.creative_log:
            parts.append(f"决策: 已记录 {len(store.creative_log)} 条")
        return " | ".join(parts)

    def _estimate_tokens(self, store: SharedContextStore) -> int:
        """粗略的 Token 估算。"""
        from .context_manager import ContextManager
        cm = ContextManager()
        return cm.estimate_token_count(store)

    def _compress(self, store: SharedContextStore) -> None:
        """委托 ContextManager 进行压缩。"""
        from .context_manager import ContextManager
        cm = ContextManager()
        cm.compress_context(store)
