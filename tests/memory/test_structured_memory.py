"""Tests for structured memory backends."""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.memory.structured_memory import (
    EntityMemory, PlotMemory, ProjectIndex, CreativeSummaryMemory,
)
from src.memory.context_store import StructuredMemoryEntry, ProjectStage


class TestEntityMemory:
    def test_store_and_retrieve_by_entity_id(self):
        em = EntityMemory()
        entry = StructuredMemoryEntry(
            entry_type="entity",
            entity_id="novel-1:hero",
            project_id="novel-1",
            attributes={"name": "Hero", "trait": "brave"},
            vector_summary="Character named Hero, brave.",
        )
        em.store(entry)

        results = em.retrieve_by_entity_id("novel-1:hero")
        assert len(results) == 1
        assert results[0].attributes["name"] == "Hero"

    def test_retrieve_by_project(self):
        em = EntityMemory()
        em.store(StructuredMemoryEntry(
            entry_type="entity", entity_id="p1:hero",
            project_id="p1", attributes={"name": "Hero"},
            vector_summary="Hero.",
        ))
        em.store(StructuredMemoryEntry(
            entry_type="entity", entity_id="p2:villain",
            project_id="p2", attributes={"name": "Villain"},
            vector_summary="Villain.",
        ))

        p1_entities = em.retrieve_by_project("p1")
        assert len(p1_entities) == 1
        assert p1_entities[0].entity_id == "p1:hero"

    def test_update_entity_attributes(self):
        em = EntityMemory()
        em.store(StructuredMemoryEntry(
            entry_type="entity", entity_id="p1:hero",
            project_id="p1", attributes={"name": "Hero"},
            vector_summary="Hero.",
        ))
        ok = em.update("p1:hero", {"trait": "brave", "role": "protagonist"})

        assert ok is True
        updated = em.retrieve_by_entity_id("p1:hero")[0]
        assert updated.attributes["trait"] == "brave"
        assert updated.attributes["role"] == "protagonist"

    def test_update_nonexistent_returns_false(self):
        em = EntityMemory()
        ok = em.update("nonexistent:id", {"trait": "brave"})
        assert ok is False

    def test_retrieve_by_name(self):
        em = EntityMemory()
        em.store(StructuredMemoryEntry(
            entry_type="entity", entity_id="p1:hero",
            project_id="p1", attributes={"name": "Hero"},
            vector_summary="Hero.",
        ))

        results = em.retrieve_by_name("Hero")
        assert len(results) == 1


class TestPlotMemory:
    def test_store_and_retrieve_by_project(self):
        pm = PlotMemory()
        entry = StructuredMemoryEntry(
            entry_type="plot_event",
            entity_id="evt-1",
            project_id="novel-1",
            attributes={"event": "Hero leaves home", "turn": "5"},
            vector_summary="Hero leaves home.",
        )
        pm.store(entry)

        events = pm.retrieve_by_project("novel-1")
        assert len(events) == 1
        assert "leaves home" in events[0].attributes["event"]

    def test_non_plot_event_ignored(self):
        pm = PlotMemory()
        pm.store(StructuredMemoryEntry(
            entry_type="entity", entity_id="e1",
            project_id="p1", attributes={},
            vector_summary="",
        ))
        events = pm.retrieve_by_project("p1")
        assert len(events) == 0

    def test_retrieve_timeline(self):
        pm = PlotMemory()
        pm.store(StructuredMemoryEntry(
            entry_type="plot_event", entity_id="e1",
            project_id="p1", attributes={"event": "Hero leaves", "turn": "1"},
            vector_summary="Hero leaves.",
        ))
        pm.store(StructuredMemoryEntry(
            entry_type="plot_event", entity_id="e2",
            project_id="p1", attributes={"event": "Hero meets mentor", "turn": "3"},
            vector_summary="Mentor.",
        ))

        timeline = pm.retrieve_timeline("p1")
        assert len(timeline) == 2
        assert timeline[0]["turn"] == "1"


class TestProjectIndex:
    def test_register_and_get_project(self):
        idx = ProjectIndex()
        idx.register("novel-1", "The Hero's Journey", "novel")

        project = idx.get("novel-1")
        assert project is not None
        assert project["title"] == "The Hero's Journey"
        assert project["type"] == "novel"
        assert project["stage"] == ProjectStage.CONCEPTUALIZATION.value

    def test_update_stage(self):
        idx = ProjectIndex()
        idx.register("novel-1", "Test", "novel")

        ok = idx.update_stage("novel-1", ProjectStage.DRAFTING)
        assert ok is True
        project = idx.get("novel-1")
        assert project["stage"] == ProjectStage.DRAFTING.value

    def test_list_by_type(self):
        idx = ProjectIndex()
        idx.register("n1", "Novel 1", "novel")
        idx.register("n2", "Novel 2", "novel")
        idx.register("s1", "Story 1", "short_story")

        novels = idx.list_by_type("novel")
        assert len(novels) == 2

    def test_update_nonexistent_returns_false(self):
        idx = ProjectIndex()
        ok = idx.update_stage("nonexistent", ProjectStage.DRAFTING)
        assert ok is False


class TestCreativeSummaryMemory:
    def test_store_and_retrieve_latest(self):
        csm = CreativeSummaryMemory()
        csm.store(
            project_id="novel-1",
            intent="Hero's journey of self-discovery",
            output_summary="Chapter 1: Hero leaves home, meets mentor.",
            style_template="Terse, evocative prose.",
        )

        summary = csm.get_latest("novel-1")
        assert summary is not None
        assert "Hero's journey" in summary["intent"]
        assert "Chapter 1" in summary["output_summary"]

    def test_get_latest_nonexistent_returns_none(self):
        csm = CreativeSummaryMemory()
        result = csm.get_latest("nonexistent")
        assert result is None

    def test_get_all(self):
        csm = CreativeSummaryMemory()
        csm.store("p1", "Intent 1", "Summary 1")
        csm.store("p1", "Intent 2", "Summary 2")

        all_summaries = csm.get_all("p1")
        assert len(all_summaries) == 2

    def test_to_structured_entries(self):
        csm = CreativeSummaryMemory()
        csm.store("p1", "Intent", "Summary")

        entries = csm.to_structured_entries()
        assert len(entries) == 1
        assert entries[0].entry_type == "creative_summary"
