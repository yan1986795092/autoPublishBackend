"""结构化记忆后端 — 实体、情节、项目索引、创作摘要。"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .context_store import StructuredMemoryEntry, ProjectStage

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────
# EntityMemory（实体记忆）
# ──────────────────────────────────────────


class EntityMemory:
    """
    存储角色、场景和组织实体。
    支持通过 entity_id 和 project_id 精确查找。
    """

    def __init__(self):
        self._by_entity_id: Dict[str, StructuredMemoryEntry] = {}
        self._by_project_id: Dict[str, List[str]] = {}

    def store(self, entry: StructuredMemoryEntry) -> None:
        if entry.entry_type != "entity" or not entry.entity_id:
            return
        self._by_entity_id[entry.entity_id] = entry
        self._by_project_id.setdefault(entry.project_id or "", []).append(entry.entity_id)
        logger.debug(f"EntityMemory: 已存储 {entry.entity_id}")

    def retrieve_by_entity_id(self, entity_id: str) -> List[StructuredMemoryEntry]:
        entry = self._by_entity_id.get(entity_id)
        return [entry] if entry else []

    def retrieve_by_project(self, project_id: str) -> List[StructuredMemoryEntry]:
        entity_ids = self._by_project_id.get(project_id, [])
        return [self._by_entity_id[eid] for eid in entity_ids if eid in self._by_entity_id]

    def retrieve_by_name(self, name: str, project_id: Optional[str] = None) -> List[StructuredMemoryEntry]:
        results = []
        for entry in self._by_entity_id.values():
            if entry.attributes.get("name", "").lower() == name.lower():
                if project_id is None or entry.project_id == project_id:
                    results.append(entry)
        return results

    def update(self, entity_id: str, new_attributes: Dict[str, str]) -> bool:
        if entity_id not in self._by_entity_id:
            return False
        entry = self._by_entity_id[entity_id]
        entry.attributes.update(new_attributes)
        return True

    def to_structured_entries(self) -> List[StructuredMemoryEntry]:
        return list(self._by_entity_id.values())


# ──────────────────────────────────────────
# PlotMemory（情节记忆）
# ──────────────────────────────────────────


class PlotMemory:
    """
    按时间顺序存储情节事件和转折点。
    """

    def __init__(self):
        self._by_project_id: Dict[str, List[StructuredMemoryEntry]] = {}

    def store(self, entry: StructuredMemoryEntry) -> None:
        if entry.entry_type != "plot_event":
            return
        project_id = entry.project_id or ""
        self._by_project_id.setdefault(project_id, []).append(entry)
        logger.debug(f"PlotMemory: 已为项目 {project_id} 存储事件")

    def retrieve_by_project(self, project_id: str) -> List[StructuredMemoryEntry]:
        return self._by_project_id.get(project_id, [])

    def retrieve_timeline(self, project_id: str) -> List[Dict[str, Any]]:
        """将情节事件以时间线字典形式返回，用于上下文注入。"""
        events = self.retrieve_by_project(project_id)
        return [
            {
                "turn": e.attributes.get("turn", "?"),
                "event": e.attributes.get("event", ""),
                "summary": e.vector_summary,
            }
            for e in sorted(events, key=lambda x: int(x.attributes.get("turn", 0)))
        ]


# ──────────────────────────────────────────
# ProjectIndex（项目索引）
# ──────────────────────────────────────────


class ProjectIndex:
    """
    所有项目的注册表，包含其元数据。
    """

    def __init__(self):
        self._projects: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        project_id: str,
        title: str,
        project_type: str,
    ) -> None:
        self._projects[project_id] = {
            "project_id": project_id,
            "title": title,
            "type": project_type,
            "stage": ProjectStage.CONCEPTUALIZATION.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        logger.info(f"ProjectIndex: 已注册 {project_id} ({title})")

    def get(self, project_id: str) -> Optional[Dict[str, Any]]:
        return self._projects.get(project_id)

    def update_stage(self, project_id: str, stage: ProjectStage) -> bool:
        if project_id not in self._projects:
            return False
        self._projects[project_id]["stage"] = stage.value
        self._projects[project_id]["updated_at"] = datetime.now().isoformat()
        return True

    def update_title(self, project_id: str, title: str) -> bool:
        if project_id not in self._projects:
            return False
        self._projects[project_id]["title"] = title
        self._projects[project_id]["updated_at"] = datetime.now().isoformat()
        return True

    def list_all(self) -> List[Dict[str, Any]]:
        return list(self._projects.values())

    def list_by_type(self, project_type: str) -> List[Dict[str, Any]]:
        return [p for p in self._projects.values() if p["type"] == project_type]


# ──────────────────────────────────────────
# CreativeSummaryMemory（创作摘要记忆）
# ──────────────────────────────────────────


class CreativeSummaryMemory:
    """
    存储每次会话的创作摘要（意图 + 输出 + 风格模板）。
    """

    def __init__(self):
        self._by_project: Dict[str, List[Dict[str, Any]]] = {}

    def store(
        self,
        project_id: str,
        intent: str,
        output_summary: str,
        style_template: str = "",
    ) -> None:
        entry = {
            "project_id": project_id,
            "intent": intent,
            "output_summary": output_summary,
            "style_template": style_template,
            "created_at": datetime.now().isoformat(),
        }
        self._by_project.setdefault(project_id, []).append(entry)
        logger.debug(f"CreativeSummaryMemory: 已为 {project_id} 存储摘要")

    def get_latest(self, project_id: str) -> Optional[Dict[str, Any]]:
        entries = self._by_project.get(project_id, [])
        return entries[-1] if entries else None

    def get_all(self, project_id: str) -> List[Dict[str, Any]]:
        return self._by_project.get(project_id, [])

    def to_structured_entries(self) -> List[StructuredMemoryEntry]:
        """将所有摘要转换为 StructuredMemoryEntry 格式。"""
        entries = []
        for project_id, summaries in self._by_project.items():
            for s in summaries:
                entries.append(StructuredMemoryEntry(
                    entry_type="creative_summary",
                    project_id=project_id,
                    attributes={
                        "intent": s["intent"],
                        "output_summary": s["output_summary"],
                        "style_template": s.get("style_template", ""),
                    },
                    vector_summary=f"{s['intent']}: {s['output_summary'][:100]}",
                ))
        return entries
