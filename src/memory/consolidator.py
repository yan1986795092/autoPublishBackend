"""Consolidator — session finalization, anchor checking, memory writes."""
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

# Trait antonym pairs for Phase 1 contradiction detection
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
# Anchor Consistency Checker
# ──────────────────────────────────────────


class AnchorConsistencyChecker:
    """
    Detects when content diverges from established creative anchors.

    Phase 1 (low-cost): regex + keyword match for explicit contradictions.
    Phase 2 (optional): LLM-based implication analysis for subtle conflicts.
    """

    def __init__(self):
        self._antonyms = TRAIT_ANTONYMS

    def check(self, anchors: CreativeAnchors, content: str) -> List[AnchorDrift]:
        """Phase 1: detect explicit trait contradictions."""
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
                            content_evidence=f"'{antonym}' found near {char_name}",
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
# Consolidator Node
# ──────────────────────────────────────────


class ConsolidatorNode:
    """
    LangGraph node that runs after all agents complete.

    Responsibilities:
    1. Anchor consistency check (if content exists)
    2. Sync anchors if review approved changes
    3. Extract new entities from content → write to global_memory + EntityMemory backend
    4. Persist creative_log decisions → PlotMemory + CreativeSummaryMemory backends
    5. Generate project progress summary
    6. Token budget compression check
    7. Mark session status
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
        # Structured memory backends (lazy init if not provided)
        self._entity_memory = entity_memory or EntityMemory()
        self._plot_memory = plot_memory or PlotMemory()
        self._project_index = project_index or ProjectIndex()
        self._creative_summary_memory = creative_summary_memory or CreativeSummaryMemory()

    def finalize(self, store: SharedContextStore) -> ConsolidationResult:
        """Finalize a session: check anchors, write memory, manage budget."""
        stage_name = store.project_manifest.stage.value if store.project_manifest else "none"
        store.add_thinking_step(
            agent="ConsolidatorNode",
            action="finalize_start",
            detail=f"stage={stage_name}",
        )

        new_entries: List[StructuredMemoryEntry] = []
        drifts: List[AnchorDrift] = []
        project_summary = ""

        # 1. Anchor consistency check (last content block)
        last_content = store.last_writing_content
        if last_content and last_content.text:
            drifts = self._checker.check(store.creative_anchors, last_content.text)
            if drifts:
                logger.info(
                    f"ConsolidatorNode: detected {len(drifts)} anchor drifts, "
                    f"prompting user confirmation"
                )

        # 2. Extract entities from last content block → both global_memory + structured backend
        if last_content and last_content.text:
            entities = self._extract_entities(
                last_content,
                store.project_manifest.project_id if store.project_manifest else None,
            )
            new_entries.extend(entities)
            store.global_memory.extend(entities)
            # Also persist to EntityMemory structured backend
            for entry in entities:
                self._entity_memory.store(entry)

        # 3. Write creative log decisions to PlotMemory
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
                    vector_summary=f"Creative decision: {decision.decision_text[:100]}",
                )
                self._plot_memory.store(plot_entry)
                store.global_memory.append(plot_entry)
                new_entries.append(plot_entry)

        # 4. Update ProjectIndex with current stage
        if store.project_manifest:
            pid = store.project_manifest.project_id
            self._project_index.register(
                project_id=pid,
                title=store.project_manifest.title,
                project_type=store.project_manifest.project_type,
            )
            self._project_index.update_stage(pid, store.project_manifest.stage)

        # 5. Generate and persist creative summary
        if store.project_manifest:
            pid = store.project_manifest.project_id
            self._creative_summary_memory.store(
                project_id=pid,
                intent=store.user_request[:200],
                output_summary=project_summary,
                style_template=store.creative_anchors.style_guide[:200] if store.creative_anchors.style_guide else "",
            )

        # 3. Update project summary
        project_summary = self._generate_summary(store)

        # 6. Token budget compression check
        tokens = self._estimate_tokens(store)
        budget = 32000
        if tokens / budget >= self.compress_threshold_pct:
            self._compress(store)

        # 7. Determine status
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
        """Phase 1: extract entities via simple pattern matching (capitalized names)."""
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
                vector_summary=f"Character: {name}",
            ))

        return entries

    def _generate_summary(self, store: SharedContextStore) -> str:
        """Generate a brief project summary from current state."""
        if not store.project_manifest:
            return ""

        manifest = store.project_manifest
        parts = [
            f"Project: {manifest.title or manifest.project_id}",
            f"Stage: {manifest.stage.value}",
            f"Turns: {manifest.turn_count}",
        ]
        if store.creative_anchors.story_outline:
            parts.append(f"Outline: {store.creative_anchors.story_outline[:100]}")
        if store.creative_log:
            parts.append(f"Decisions: {len(store.creative_log)} logged")
        return " | ".join(parts)

    def _estimate_tokens(self, store: SharedContextStore) -> int:
        """Rough token estimate."""
        from .context_manager import ContextManager
        cm = ContextManager()
        return cm.estimate_token_count(store)

    def _compress(self, store: SharedContextStore) -> None:
        """Delegate compression to ContextManager."""
        from .context_manager import ContextManager
        cm = ContextManager()
        cm.compress_context(store)
